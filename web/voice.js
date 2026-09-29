// Browser voice client (SPEC §1/§3/§8 scaffold): one WebSocket per tab,
// getUserMedia with echoCancellation -> worklet -> PCM chunks; heartbeat
// every ~2 s carrying mic RMS; toggle arms capture (arming is explicit,
// never automatic). TTS playback queue + barge-in land with real TTS.
const $ = (id) => document.getElementById(id);
const ws = new WebSocket(
  `ws://${location.host}/ws/voice?client=${Date.now()}-${Math.random()
    .toString(36).slice(2, 8)}`);
let armed = false, audioCtx = null, workletNode = null, micStream = null;
let rms = 0, speaking = false;

function log(who, text) {
  const li = document.createElement('li');
  li.innerHTML = `<b>${who}</b> ${text}`;
  $('transcript').appendChild(li);
}

function renderDiff(msg) {
  const li = document.createElement('li');
  li.innerHTML = `<b>diff</b> <code>${msg.section}</code><br>
    <del>${msg.find}</del> → <ins>${msg.replace}</ins><br>
    <button data-a="approve">Apply</button>
    <button data-a="reject">Discard</button>`;
  li.querySelectorAll('button').forEach((b) => (b.onclick = () => {
    ws.send(JSON.stringify({ type: b.dataset.a,
                             diff_id: msg.diff_id, section: msg.section }));
    b.parentElement.querySelectorAll('button').forEach((x) => (x.disabled = true));
  }));
  $('transcript').appendChild(li);
}

ws.onmessage = (ev) => {
  const msg = JSON.parse(ev.data);
  if (msg.type === 'assistant_text') { log('agent', msg.text); speaking = false; }
  else if (msg.type === 'turn_started') { speaking = true; }
  else if (msg.type === 'turn_interrupted') { log('agent', '[interrupted]'); speaking = false; }
  else if (msg.type === 'armed') { log('sys', msg.ok ? 'armed (endpoint ours)' : `armed elsewhere: ${msg.holder}`); }
  else if (msg.type === 'diff') { renderDiff(msg); }
  else if (msg.type === 'diff_resolved') {
    log('sys', msg.applied ? 'applied ✓' : `not applied: ${msg.reason}`);
  }
  else if (msg.type === 'error') { log('sys', `error[${msg.where}]: ${msg.message}`); }
};

async function setArmed(on) {
  if (on && !audioCtx) {
    audioCtx = new AudioContext();
    await audioCtx.audioWorklet.addModule('/static/capture.js');
    micStream = await navigator.mediaDevices.getUserMedia({
      audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: false },
    });
    const src = audioCtx.createMediaStreamSource(micStream);
    workletNode = new AudioWorkletNode(audioCtx, 'capture');
    workletNode.port.onmessage = (ev) => {
      if (ws.readyState === 1 && armed) ws.send(ev.data);
      const view = new Int16Array(ev.data);
      let sum = 0;
      for (let i = 0; i < view.length; i += 8) sum += view[i] * view[i];
      rms = Math.sqrt(sum / (view.length / 8)) / 0x7fff; // persistent meter feed
    };
    src.connect(workletNode);
  }
  armed = on;
  if (audioCtx) (on ? audioCtx.resume() : audioCtx.suspend());
  ws.send(JSON.stringify({ type: on ? 'arm' : 'disarm' }));
  $('toggle').textContent = on ? 'Voice: ON' : 'Voice: OFF';
}

// Capture = toggle AND tab-focused (§3): tabbing away pauses capture,
// the toggle state is preserved.
document.addEventListener('visibilitychange', () => {
  if (audioCtx) (document.hidden ? audioCtx.suspend() : (armed && audioCtx.resume()));
});

$('toggle').onclick = () => setArmed(!armed);
$('send').onclick = () => {
  const text = $('input').value.trim();
  if (text) { log('me', text); ws.send(JSON.stringify({ type: 'typed', text })); $('input').value = ''; }
};
$('input').addEventListener('keydown', (e) => { if (e.key === 'Enter') $('send').click(); });

// One heartbeat every ~2 s carrying mic RMS (§8: liveness + meter + watchdog).
setInterval(() => {
  if (ws.readyState === 1) ws.send(JSON.stringify({ type: 'heartbeat', rms }));
  $('meter').style.width = `${Math.min(100, rms * 400)}%`;
}, 2000);
