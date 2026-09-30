// The corpus surface (SPEC §6 door 1): upload, per-PDF status, retry,
// pin. The registry doors already existed server-side (/corpus/*); this
// is the front door. Status polls because ingest is background work —
// §6 wants it visible, not rot. No search UI: the agent answers corpus
// questions by voice (§6), the UI only feeds and watches the corpus.
const $ = (id) => document.getElementById(id);

const esc = (s) => String(s).replace(/[&<>"]/g, (c) => (
  { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));

const STATUS = { queued: '⏳ queued', extracting: '⚙ extracting',
                 indexed: '✓ indexed', failed: '✗ failed' };

async function refreshDocs() {
  const docs = await (await fetch('/corpus/docs')).json();
  const ul = $('corpus-docs');
  ul.innerHTML = docs.map((d) => `
    <li class="doc ${d.status}">
      <div class="row">
        <span class="st ${d.status}">${STATUS[d.status] || esc(d.status)}</span>
        <span class="ttl">${esc(d.title || '(untitled)')}
          <span class="id">${esc(d.doc_id.slice(0, 8))}</span></span>
        ${d.pinned_in.length
          ? `<span class="pin" title="pinned in: ${esc(d.pinned_in.join(', '))}">📌</span>`
          : ''}
        <span class="meta">${d.chunk_count ? d.chunk_count + ' chunks · ' : ''}${d.arxiv ? 'arXiv:' + esc(d.arxiv) + ' · ' : ''}${esc(d.source)}</span>
        <span class="acts">
          ${d.status === 'failed'
            ? `<button data-a="retry" data-id="${esc(d.doc_id)}">Retry</button>` : ''}
          <button data-a="${d.pinned_here ? 'unpin' : 'pin'}" data-id="${esc(d.doc_id)}">
            ${d.pinned_here ? 'Unpin' : 'Pin'}</button>
        </span>
      </div>
      ${d.error ? `<div class="err">${esc(d.error)}</div>` : ''}
    </li>`).join('')
    || '<li class="doc empty">corpus is empty — add a PDF above</li>';
  ul.querySelectorAll('button').forEach((b) => (b.onclick = async () => {
    b.disabled = true;
    await fetch(`/corpus/${b.dataset.id}/${b.dataset.a}`, { method: 'POST' });
    refreshDocs();
  }));
}

$('corpus-upload').onclick = async () => {
  const file = $('corpus-file').files[0];
  if (!file) return;
  const q = new URLSearchParams();
  const title = $('corpus-title').value.trim();
  const arxiv = $('corpus-arxiv').value.trim();
  if (title) q.set('title', title);
  if (arxiv) q.set('arxiv', arxiv);
  const btn = $('corpus-upload');
  btn.disabled = true;
  try {
    await fetch('/corpus/upload?' + q.toString(), { method: 'POST', body: file });
    $('corpus-file').value = ''; $('corpus-title').value = '';
    $('corpus-arxiv').value = '';
  } finally {
    btn.disabled = false;
    refreshDocs();  // the doc appears queued immediately; ingest follows
  }
};

refreshDocs();
// Ingest moves status in the background; poll while the tab is visible,
// catch up the moment it is focused again (§3's visibility discipline).
setInterval(() => { if (!document.hidden) refreshDocs(); }, 3000);
document.addEventListener('visibilitychange', () => {
  if (!document.hidden) refreshDocs();
});
