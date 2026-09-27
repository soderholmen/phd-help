# Mobile browser voice pipeline support

Research for [issue #15](https://github.com/soderholmen/phd-help/issues/15). Can a mobile browser be a
full voice endpoint (continuous capture + barge-in) for conversation mode, or must the phone be
view-only / push-to-talk?

- **Date:** 2026-09-27
- **Method:** primary sources — WebKit Bugzilla + WebKit source/commits, Chromium source, MDN +
  MDN browser-compat-data, Chrome developer docs, and first-party engineering threads in voice-agent
  project trackers (LiveKit, Pipecat). Version numbers given for every capability claim.
- **Platform anchors (current as of writing):** iOS 26.5.x / Mobile Safari 26.5.2 (WebKit 605.1.15);
  Chrome for Android 149/150. Both observed in production telemetry reported to WebKit in
  July 2026 ([WebKit bug 319706](https://bugs.webkit.org/show_bug.cgi?id=319706)).

## TL;DR verdict

| Capability | Android Chrome | iOS Safari |
|---|---|---|
| Foreground: gUM + AudioWorklet + WS + streamed TTS | **Works** | **Works** |
| Barge-in through the phone's own speaker | **Mostly works** (AEC good, device variance) | **Degraded** (self-echo leaks into VAD; convergence lag on first utterance) |
| Mic capture survives screen lock / background | **Likely works** (capture exempts tab from freezing; mic foreground service keeps process alive) — verify on device | **Fails** — audio track is muted on page hide, by design |
| WebSocket survives background | Works while tab unfrozen | **Fails** — socket closes on background; reconnect to IP-based URLs can hang |
| Hands-free loop across screen lock | Plausible | **Impossible** — must pause and recover on foreground |

**Bottom line:** Android Chrome can be a full conversation-mode endpoint. iOS Safari can be a full
conversation-mode endpoint **only while the screen is on and the tab is visible**; screen lock or
app-switch mutes the mic and kills the socket, so iOS needs a foreground-only mode with an explicit
recover-on-unlock path. Push-to-talk does not rescue iOS background use — the mic is muted in the
background regardless of how capture was armed.

---

## 1. Continuous capture (getUserMedia + AudioWorklet) and backgrounding

### Capability floor (both platforms, foreground)

- AudioWorklet: Chrome 66, Safari 14.1 / iOS 14.5 (MDN
  [browser-compat-data](https://github.com/mdn/browser-compat-data), `api/AudioWorklet.json`).
  `AudioContext({ sampleRate })` honored since Chrome 74 / Safari 14.1
  (`api/AudioContext.json`, `options_sampleRate_parameter`; WebKit
  [bug 216425](https://bugs.webkit.org/show_bug.cgi?id=216425) landed the iOS support in 2020).
  The prototype's 48 kHz → 16 kHz worklet downsample is fine on both.
- getUserMedia requires a **secure context** on both platforms (MDN
  [getUserMedia](https://developer.mozilla.org/en-US/docs/Web/API/MediaDevices/getUserMedia) —
  "only available in secure contexts"; Safari 26.4 additionally tightened
  `MediaDeviceInfo` to secure contexts,
  [WebKit features in Safari 26.4](https://webkit.org/blog/17862/webkit-features-for-safari-26-4/)).
  **Deployment note:** the home server over ZeroTier must be served over HTTPS with a certificate
  the phone trusts (real cert via DNS-01, or a private CA installed on the devices). Plain
  `http://10.x.x.x` cannot open a mic at all.

### iOS Safari: background = muted capture, by design

This is not a bug; it is WebKit's tested, intentional policy:

- WebKit added "mute audio capture automatically when page is not visible" in 2019
  ([bug 198307](https://bugs.webkit.org/show_bug.cgi?id=198307), commit
  `5cf7b9825935`): "In case document gets in the background, interrupt the audio track…
  CoreAudioCaptureSourceIOS requires the audio source be interrupted if the app has not the right
  background mode."
- The behavior is still tested today:
  [`LayoutTests/platform/ios/mediastream/audio-muted-in-background-tab.html`](https://github.com/WebKit/WebKit/blob/main/LayoutTests/platform/ios/mediastream/audio-muted-in-background-tab.html)
  asserts `audioTrack.muted === true` (via `onmute`) when page visibility goes false.
- What Apple *did* improve: keeping the process alive so capture recovers quickly on return.
  Safari 18-era fix "Take a background assertion for processes having muted capture"
  ([bug 278560](https://bugs.webkit.org/show_bug.cgi?id=278560), commit `760a765e8b6f`, Aug 2024,
  backported to the Safari 18 branch): "a WebRTC connection continues sending black frames in a
  backgrounded tab on iOS" — i.e. tracks stay alive but deliver muted (silent/black) data, not real
  audio. Open bug
  [289794](https://bugs.webkit.org/show_bug.cgi?id=289794) (2025) — "Unmuting microphone by pinching
  AirPods after the page is backgrounded does not work" — confirms backgrounded pages sit in a
  muted-mic state.
- Screen lock is the same event as backgrounding. It has its own history of regressions:
  [bug 241400](https://bugs.webkit.org/show_bug.cgi?id=241400) (iOS 15.5 turned the mic off on
  auto-lock; not reproducible in 15.6 beta).
- Home-screen web apps (Add-to-Home-Screen / standalone) are **worse** than Safari tabs: mic stops
  in background there even where Safari recovers (comment history in
  [bug 226620](https://bugs.webkit.org/show_bug.cgi?id=226620), Oct 2024). WKWebView-based apps have
  background audio capture disabled outright
  ([bug 217948](https://bugs.webkit.org/show_bug.cgi?id=217948)).
- Reliability caveat on current iOS: production telehealth telemetry (July 2026) showed ~20% of
  iPhone sessions hitting a permanent mic-acquisition failure after an `audiomxd` media-services
  reset, vs 0 occurrences on Android Chrome the same day
  ([bug 319706](https://bugs.webkit.org/show_bug.cgi?id=319706), since fixed). Treat iOS capture as
  needing a re-acquire-and-retry path regardless of backgrounding.

### iOS Safari: background also kills the socket and (historically) WebAudio

- Backgrounding closes the live WebSocket (code 1006), and — open as of Feb 2026 — a *new*
  `WebSocket()` to an IP-based URL (the ZeroTier case) can hang in CONNECTING permanently after
  resume; hostname-based URLs recover
  ([bug 308073](https://bugs.webkit.org/show_bug.cgi?id=308073)). **Use a hostname, not a bare
  ZeroTier IP, for the wss:// endpoint.**
- WebAudio-in-background has a long regression saga: AudioContext suspended on background was
  fixed for iOS 16 ([bug 237878](https://bugs.webkit.org/show_bug.cgi?id=237878), r291390),
  re-broken in iOS 17.0 and fixed again March 2024 → Safari 17.4
  ([bug 261554](https://bugs.webkit.org/show_bug.cgi?id=261554), commit `275558@main`), and still
  has open failure modes: `AudioContext.resume()` never resolving after background-suspend
  ([bug 281566](https://bugs.webkit.org/show_bug.cgi?id=281566), open) and silent AudioContext
  after PWA background return ([bug 291892](https://bugs.webkit.org/show_bug.cgi?id=291892), open).
- A page that is *audible* gets a foreground activity assertion and keeps running
  (`WebPageProxy::ProcessActivityState::takeAudibleActivity`, same commit as 278560) — so TTS can
  keep playing in the background, but with the mic muted the conversation loop is dead anyway.

### Android Chrome: background capture is protected

- Chrome freezes hidden tabs (including screen-off) — "Freezing is already enabled on Mobile"
  (Chromium doc
  [`freezing_opt_out_opt_in.md`](https://chromium.googlesource.com/chromium/src/+/HEAD/chrome/browser/performance_manager/docs/freezing_opt_out_opt_in.md);
  a frozen page "cannot run any tasks… DOM timers, XHR requests… will not run"; community-observed
  ~5-minute trigger on Android since at least Chrome 94,
  [SO 69454468](https://stackoverflow.com/questions/69454468)).
- But the shared freezing policy exempts exactly our page: `CannotFreezeReason` includes
  `kCapturingAudio`, `kAudible`, `kRecentlyAudible` (5-min audio protection window), `kWebRTC`,
  `kNotificationPermission`
  ([`components/performance_manager/freezing/freezing_policy.cc`](https://github.com/chromium/chromium/blob/main/components/performance_manager/freezing/freezing_policy.cc),
  `cannot_freeze_reason.cc`; `kFreezingAudioProtectionTime`/`kFreezingVisibleProtectionTime` =
  5 min in `components/performance_manager/features.cc`). The documented desktop criteria list
  includes "the page is currently capturing user media (webcam, microphone)".
- Android process survival: Chrome runs active web capture inside a **foreground service typed
  `camera|microphone`** (`MediaCaptureNotificationService`,
  [`chrome/android/java/AndroidManifest.xml`](https://github.com/chromium/chromium/blob/main/chrome/android/java/AndroidManifest.xml)),
  and media playback gets a `mediaPlayback` FGS. That keeps the renderer's process alive with the
  screen off.
- Residual risks: Android Doze defers background network for long-idle devices
  ([developer.android.com/doze](https://developer.android.com/training/monitoring-device-state/doze-standby))
  — an active FGS is exempt, but aggressive OEM battery managers are not standardized; and the
  mobile freezing path historically ignored some opt-out machinery, so the exemption behavior
  deserves an on-device test (see checklist).

## 2. echoCancellation and full-duplex barge-in on the phone's own speaker

### What the engines do

- **Chrome (Android):** mic capture goes through the WebRTC AudioProcessing pipeline by default —
  the constraint resolver scores `kBrowserDecides` echo cancellation and `kApmProcessed`
  ("WebRTC/AudioService AEC, AGC, NS, and ML Voice Isolation") as the preferred candidate
  ([`media_stream_constraints_util_audio.cc`](https://github.com/chromium/chromium/blob/main/third_party/blink/renderer/modules/mediastream/media_stream_constraints_util_audio.cc)).
  Extended modes `"all"` / `"remote-only"` (MDN
  [echoCancellation](https://developer.mozilla.org/en-US/docs/Web/API/MediaTrackConstraints/echoCancellation))
  are still behind a runtime flag in Chrome.
- **Safari (iOS):** the `echoCancellation` constraint works (fixed circa Safari 13,
  [bug 179411](https://bugs.webkit.org/show_bug.cgi?id=179411)) and WebKit made it **default true**
  when unspecified in 2023 → Safari 17
  ([bug 257495](https://bugs.webkit.org/show_bug.cgi?id=257495), commit `264721@main`).

### What actually happens with speakerphone barge-in (2025 field reports)

- **iOS is the problem child.** LiveKit voice-agent field report (Oct–Nov 2025): with
  `echoCancellation: true` (+ all `goog*` constraints) on iPhone built-in mics, the agent's own TTS
  is picked up and detected as user speech, creating interruption loops — "almost the same in all
  browsers" on iOS, worse during the first spoken message; obstructing the mic or reverberation
  re-triggers it. LiveKit's engineer: AEC needs to cache speaker audio as a reference, so the
  **first utterance always leaks until the AEC converges** (mitigation: play silence/tone before
  the first real speech); switching output to the earpiece would fix it but "is not possible on the
  browser level" ([livekit/agents#3758](https://github.com/livekit/agents/issues/3758)).
- Pipecat saw the same class of failure on Safari (macOS) with bot audio played through an `<audio>`
  element — bot speech transcribed as user speech; fine on Firefox and fine on Safari with
  headphones ([pipecat-ai/pipecat#3062](https://github.com/pipecat-ai/pipecat/issues/3062),
  Nov–Dec 2025). Maintainer: "Echo cancellation… require[s] browser support… There's not much we
  can do to change how the browsers operate."
- **iOS routing quirk that bites barge-in:** starting the mic forces audio output to the loud
  speaker even with headphones connected (Safari and Chrome-on-iOS, Feb 2025 report,
  [SO 79401143](https://stackoverflow.com/questions/79401143), referencing WebKit
  [bug 196539](https://bugs.webkit.org/show_bug.cgi?id=196539)). Workaround: set
  `navigator.audioSession.type = 'play-and-record'` *after* getUserMedia to kick iOS into using the
  headset. `navigator.audioSession` is **Safari-only** (Safari 16.4+; not in Chrome — MDN BCD
  `api/AudioSession.json`).
- Android Chrome: no equivalent systemic complaint in these sources; AEC quality varies by device
  (hardware AEC availability is device-dependent), so expect "mostly works, occasionally trips the
  VAD at high TTS volume" rather than the iOS-grade loop.

### Practical mitigations (both platforms)

1. Request `{ echoCancellation: true, noiseSuppression: true, autoGainControl: true }` explicitly
   (don't rely on defaults on Safari < 17).
2. Warm up the AEC: emit a short silent/low-level preroll before the first real TTS utterance.
3. Barge-in policy: require ≥1–2 words (or a VAD score margin) to interrupt while TTS is playing,
   and/or duck TTS volume during the vulnerable window.
4. On iOS, set `audioSession.type = 'play-and-record'` post-gUM so headset users get proper
   routing; document that speakerphone barge-in on iPhone is the weakest link.
5. Server-side: keep the silero-VAD hangover, and consider a TTS-echo subtraction / similarity
   check against the just-played utterance before honoring a barge-in (the "secret sauce"
   hypothesis in the LiveKit thread is exactly this kind of server-side gating).

## 3. Sustained WebSocket PCM + Web Audio TTS scheduling queue

- All primitives exist on both platforms at current versions (AudioWorklet, WebSocket,
  AudioBufferSourceNode scheduling). The desktop prototype's architecture (worklet → 16 kHz PCM
  over WS → server VAD/STT → streamed TTS scheduled in a Web Audio queue) is portable unchanged;
  stop/resume mid-stream for barge-in is just `stop()` + re-scheduling on the destination, which
  works on both.
- The platform difference is **background socket lifetime**, not throughput:
  - Android: while the tab is exempt from freezing (capturing/audible), the socket and worklet keep
    running with the screen off.
  - iOS: the socket closes on background (bug 308073) and capture is muted, so the loop must be
    torn down and rebuilt on foreground: reconnect WS (via hostname, not IP), resume/recreate the
    AudioContext (resume can hang — bug 281566; be ready to construct a fresh one inside the
    unlock gesture… which iOS won't give you automatically; realistically require a visible "resume"
    tap after long backgrounds), and re-acquire the mic track.
- Future option on iOS: Safari 26.4 shipped **WebTransport** (HTTP/3, no head-of-line blocking) —
  a better fit for live audio than TCP WebSocket
  ([WebKit features in Safari 26.4](https://webkit.org/blog/17862/webkit-features-for-safari-26-4/));
  Chrome has had it since 97. Not required for v1.

## 4. Platform policy that degrades the hands-free loop

- **User-gesture / autoplay:** both engines gate audible playback on user activation. Chrome:
  Web Audio autoplay policy since Chrome 71 — an AudioContext created before a gesture starts
  `suspended` and must be `resume()`d in a gesture handler
  ([developer.chrome.com/blog/autoplay](https://developer.chrome.com/blog/autoplay/)). Safari has
  the same requirement (MDN
  [Autoplay guide](https://developer.mozilla.org/en-US/docs/Web/Media/Guides/Autoplay): playback
  blocked "in a tab which has not yet had any user interaction"). Consequence: the conversation-mode
  toggle tap arms audio + mic for the session; no further gestures needed while foregrounded.
- **Permission persistence:** Safari persists the per-site mic grant (grant-persistency is a tested
  WebKit behavior,
  [`getUserMedia-grant-persistency.html`](https://github.com/WebKit/WebKit/blob/main/LayoutTests/fast/mediastream/getUserMedia-grant-persistency.html));
  users manage it in Safari's per-site settings. Chrome persists grants per origin, but **Safety
  Check auto-revokes permissions for sites you rarely visit** (Chromium commit `6fcd885033`,
  Dec 2022) — a personal tool used daily is safe in practice, but a re-tap after a vacation week is
  possible. Both platforms show a visible mic-in-use indicator (orange dot / mic icon) — fine for a
  personal tool, but the user will see it all session.
- **Throttling:** Chrome — 5-min hidden-tab freeze unless exempt (capturing/audible exempt; see §1).
  iOS — backgrounded pages are throttled/suspended except when audible or capturing
  (`WebPageProxy` activity assertions), and timers are clamped; the practical effect is the same
  "foreground-only" constraint as the muted-capture rule.

## Verdict for the app

- **Android Chrome: full conversation mode — yes.** Continuous capture + barge-in works; background
  (screen-off) survival is protected by the capture/audible freezing exemptions and the
  microphone foreground service, but validate on the actual device (OEM battery managers are the
  wildcard). Expect occasional VAD false-triggers from self-echo at high volume; mitigate with
  barge-in word-count gating.
- **iOS Safari: conversation mode only while the phone is in hand and unlocked.** Foreground works
  end-to-end, but screen lock / app switch mutes the mic (by design, tested behavior) and closes
  the WebSocket; there is no push-to-talk or background-audio trick that restores capture. Design
  the iOS endpoint as: arms on tap, auto-pauses on `visibilitychange`/`pagehide` (stop VAD, close
  cleanly), and on return shows a "tap to resume" state rather than pretending to be hands-free.
  Headphones materially improve the barge-in experience on iPhone.
- **Voice-endpoint handoff (CONTEXT.md):** the "only the endpoint client captures" rule holds;
  iOS just means the handoff happens implicitly whenever the phone locks — the server should treat
  an iOS endpoint as *leased while foregrounded*, and the UI should make losing the lease obvious.
- **Deployment:** HTTPS with a phone-trusted cert on the ZeroTier host, and a **hostname** (mDNS or
  a real domain) rather than a bare IP for the wss:// URL (WebKit bug 308073).

## On-device verification checklist (what primary sources can't settle)

1. Android: screen off for 10+ min with mic hot + silent (no TTS) — does the WS keep streaming?
   (tests the `kCapturingAudio` freeze exemption + Doze on the actual OEM build.)
2. Android: same with TTS playing periodically (audible exemption + mediaPlayback FGS).
3. iOS: confirm `track.onmute` fires on lock and that a foreground-return recovery path
   (re-gUM + WS reconnect via hostname + fresh AudioContext) reliably restores the loop on
   Safari 26.x.
4. iOS: barge-in false-trigger rate on speakerphone at 3 volume levels, with/without
   `audioSession.type='play-and-record'`, with/without the AEC warmup preroll.
5. Both: 60-minute continuous session (worklet + WS) for drift/leak; iOS 26 audiomxd-recovery retry
   path (bug 319706 class).

## Source index

| Claim | Source |
|---|---|
| AudioWorklet / sampleRate support versions | MDN BCD `api/AudioWorklet.json`, `api/AudioContext.json`; WebKit bugs 216425 |
| iOS mutes capture when page hidden (by design, still tested) | WebKit bug 198307 + commit `5cf7b9825935`; `LayoutTests/platform/ios/mediastream/audio-muted-in-background-tab.html` |
| iOS keeps muted-capture process alive (Safari 18) | WebKit bug 278560, commit `760a765e8b6f` (Aug 2024) |
| iOS backgrounded mic mute state (2025) | WebKit bug 289794 (open) |
| iOS lock-screen mic regressions | WebKit bug 241400 (iOS 15.5) |
| Home-screen apps lose background mic | WebKit bug 226620 comments (Oct 2024) |
| WKWebView background capture disabled | WebKit bug 217948 |
| iOS WS closes on background; IP reconnect hangs | WebKit bug 308073 (Feb 2026, open) |
| iOS AudioContext background saga | WebKit bugs 237878 (iOS 16), 261554 (→ Safari 17.4), 281566 (open), 291892 (open) |
| iOS 26 capture-acquisition fragility vs Android | WebKit bug 319706 (Jul 2026, production telemetry) |
| Chrome freezing on mobile + exemptions | Chromium `freezing_opt_out_opt_in.md`; `components/performance_manager/freezing/{freezing_policy.cc,cannot_freeze_reason.cc}`; `features.cc` (5-min protection times); SO 69454468 |
| Chrome mic foreground service | `chrome/android/java/AndroidManifest.xml` (`MediaCaptureNotificationService`, `camera\|microphone`) |
| Chrome APM AEC default | `third_party/blink/renderer/modules/mediastream/media_stream_constraints_util_audio.cc` |
| Safari echoCancellation works / defaults true | WebKit bugs 179411, 257495 (→ Safari 17) |
| iPhone speakerphone barge-in loops, AEC convergence | livekit/agents#3758 (Oct–Nov 2025) |
| Safari self-echo through VAD | pipecat-ai/pipecat#3062 (Nov–Dec 2025) |
| iOS forces speaker when mic active; AudioSession fix | SO 79401143 (Feb 2025); WebKit bug 196539; BCD `api/AudioSession.json` (Safari 16.4, no Chrome) |
| Autoplay/gesture requirements | developer.chrome.com/blog/autoplay (Chrome 71); MDN Autoplay guide |
| Permission persistence / Safety Check revocation | WebKit `getUserMedia-grant-persistency.html`; Chromium commit `6fcd885033` |
| Secure-context requirement | MDN getUserMedia; WebKit Safari 26.4 release notes |
| WebTransport on Safari 26.4 | webkit.org blog 17862 |
