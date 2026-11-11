const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];

(() => {
  const root = document.documentElement;
  if (!root) return;
  const key = 'willspeak-theme';
  const valid = value => ['system', 'light', 'dark'].includes(value);
  let preference = 'system';
  try {
    const saved = localStorage.getItem(key);
    if (valid(saved)) preference = saved;
  } catch (_) {  }

  const media = typeof matchMedia === 'function' ? matchMedia('(prefers-color-scheme: dark)') : null;
  function apply() {
    if (preference === 'system') root.removeAttribute('data-theme');
    else root.setAttribute('data-theme', preference);
    root.style.colorScheme = preference === 'system' ? (media ? (media.matches ? 'dark' : 'light') : 'light dark') : preference;
    for (const select of $$('[data-theme-select]')) select.value = preference;
  }
  apply();

  const onSystemChange = () => { if (preference === 'system') apply(); };
  if (media?.addEventListener) media.addEventListener('change', onSystemChange);
  else if (media?.addListener) media.addListener(onSystemChange);

  function connect() {
    apply();
    for (const select of $$('[data-theme-select]')) {
      select.addEventListener('change', () => {
        preference = valid(select.value) ? select.value : 'system';
        try { localStorage.setItem(key, preference); } catch (_) {  }
        apply();
      });
    }
  }
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', connect);
  else connect();

  globalThis.addEventListener?.('storage', event => {
    if (event.key !== key && event.key !== null) return;
    preference = valid(event.newValue) ? event.newValue : 'system';
    apply();
  });
})();

function escapeHtml(value) {
  return String(value ?? '').replace(/[&<>"']/g, ch => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
  })[ch]);
}

function fmtDur(s) {
  s = Math.round(s || 0);
  const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60), x = s % 60;
  return h ? `${h}h ${m}m` : m ? `${m}m ${x}s` : `${x}s`;
}

async function post(url, obj) {
  const fd = new FormData();
  for (const [k, v] of Object.entries(obj || {}))
    if (v !== null && v !== undefined) fd.append(k, v);
  const r = await fetch(url, { method: 'POST', body: fd });
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}
async function get(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}

function encodeWav(chunks, sampleRate) {
  let n = 0;
  for (const c of chunks) n += c.length;
  const buf = new ArrayBuffer(44 + n * 4);
  const dv = new DataView(buf);
  const str = (o, s) => { for (let i = 0; i < s.length; i++) dv.setUint8(o + i, s.charCodeAt(i)); };
  str(0, 'RIFF'); dv.setUint32(4, 36 + n * 4, true); str(8, 'WAVE');
  str(12, 'fmt '); dv.setUint32(16, 16, true);
  dv.setUint16(20, 3, true);
  dv.setUint16(22, 1, true);
  dv.setUint32(24, sampleRate, true);
  dv.setUint32(28, sampleRate * 4, true);
  dv.setUint16(32, 4, true); dv.setUint16(34, 32, true);
  str(36, 'data'); dv.setUint32(40, n * 4, true);
  let o = 44;
  for (const c of chunks) for (let i = 0; i < c.length; i++, o += 4) dv.setFloat32(o, c[i], true);
  return new Blob([buf], { type: 'audio/wav' });
}

function navBadge(totals) {
  const a = $$('.app-nav a, header nav a').find(x => x.getAttribute('href') === '/review');
  if (!a) return;
  const count = Number(totals?.pending);
  const pending = Number.isFinite(count) && count > 0 ? Math.floor(count) : 0;
  const badge = a.querySelector('[data-nav-badge]');
  if (badge) {
    badge.textContent = pending ? String(pending) : '';
    badge.hidden = !pending;
  } else {
    const label = a.querySelector('.nav-label') || a;
    label.textContent = pending ? `Review (${pending})` : 'Review';
  }
  a.setAttribute('aria-label', pending ? `Review, ${pending} ${pending === 1 ? 'take' : 'takes'} waiting` : 'Review');
}

document.addEventListener('DOMContentLoaded', () => {
  for (const label of document.querySelectorAll('label')) {
    if (label.htmlFor || label.querySelector('input, select, textarea')) continue;
    const next = label.nextElementSibling;
    const control = next?.matches('input, select, textarea') ? next : next?.querySelector('input, select, textarea');
    if (control?.id) label.htmlFor = control.id;
  }
});
