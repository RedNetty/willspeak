(function () {
  const param = new URLSearchParams(location.search).get('mic');
  const active = param === 'pi' || (param !== 'local' && !globalThis.isSecureContext);
  const BLOCK = 128, METER_EVERY = 16;
  let unlock, badge;
  const gesture = active ? new Promise(resolve => { unlock = resolve; }) : Promise.resolve();

  function setBadge(text, bad) {
    if (!badge) return;
    badge.textContent = text;
    badge.style.background = bad ? '#8a1c1c' : '#1d3b2a';
  }

  async function describe() {
    try {
      const s = await (await fetch('/api/pimic/status', { cache: 'no-store' })).json();
      if (s.error) setBadge(`Box microphone: ${s.error}`, true);
      else if (!s.microphones.length && s.source !== 'custom command') setBadge('Box microphone: no USB microphone is plugged into the box', true);
      else setBadge(`Box microphone${s.microphones.length ? ': ' + s.microphones[0] : ''}${s.running ? ' · listening' : ''}`, false);
    } catch (_) { setBadge('Box microphone: the box is not answering', true); }
  }

  function startOverlay() {
    badge = document.createElement('div');
    badge.id = 'pimic-badge';
    badge.style.cssText = 'position:fixed;left:12px;bottom:12px;z-index:9998;padding:6px 10px;border-radius:6px;color:#fff;font:13px/1.3 system-ui,sans-serif;max-width:min(90vw,520px);opacity:.92;pointer-events:none';
    document.body.append(badge);
    setBadge('Box microphone', false);
    const cover = document.createElement('button');
    cover.type = 'button';
    cover.id = 'pimic-start';
    cover.style.cssText = 'position:fixed;inset:0;z-index:9999;border:0;width:100%;background:rgba(20,20,20,.88);color:#fff;font:600 28px/1.3 system-ui,sans-serif;cursor:pointer';
    cover.style.display = 'flex'; cover.style.flexDirection = 'column'; cover.style.alignItems = 'center'; cover.style.justifyContent = 'center'; cover.style.gap = '12px';
    cover.innerHTML = '<span style="display:block;font:600 34px/1.2 system-ui,sans-serif">Click to start</span>'
      + '<span style="display:block;font:400 18px/1.4 system-ui,sans-serif;opacity:.8;max-width:30em;text-align:center">This page listens with the box’s microphone and speaks on this computer.</span>';
    cover.onclick = () => {
      try { const u = new SpeechSynthesisUtterance(' '); u.volume = 0; speechSynthesis.speak(u); } catch (_) {}
      try { const C = globalThis.AudioContext || globalThis.webkitAudioContext; const c = new C(); c.resume().then(() => c.close()); } catch (_) {}
      cover.remove();
      unlock();
    };
    document.body.append(cover);
    describe();
    setInterval(describe, 5000);
  }

  async function open() {
    await gesture;
    const abort = new AbortController();
    let on = false, ended = false;
    const port = { onmessage: null, postMessage(m) { if (m === 'start') on = true; else if (m === 'stop') on = false; } };
    const node = { port, onprocessorerror: null, connect() {}, disconnect() { on = false; } };
    const track = {
      kind: 'audio', enabled: true, readyState: 'live', onended: null,
      stop() { if (track.readyState === 'ended') return; track.readyState = 'ended'; ended = true; abort.abort(); },
    };
    const stream = { active: true, getTracks: () => [track], getAudioTracks: () => [track] };
    const resp = await fetch('/api/pimic/stream', { signal: abort.signal, cache: 'no-store' });
    if (!resp.ok || !resp.body) {
      let msg = `HTTP ${resp.status}`;
      try { msg = (await resp.json()).detail || msg; } catch (_) {}
      setBadge(`Box microphone: ${msg}`, true);
      throw new Error(msg);
    }
    const rate = +resp.headers.get('X-PCM-Rate') || 48000;
    const ctx = {
      sampleRate: rate, state: 'running',
      resume: async () => {}, close: async () => { ctx.state = 'closed'; track.stop(); },
    };
    const send = data => { try { port.onmessage?.({ data }); } catch (e) { console.error(e); } };
    (async () => {
      const reader = resp.body.getReader();
      let carry = null, frame = new Float32Array(BLOCK), fill = 0;
      let blocks = 0, peak = 0, sumsq = 0, n = 0;
      try {
        for (;;) {
          const { value, done } = await reader.read();
          if (done) break;
          let bytes = value;
          if (carry !== null) { const b = new Uint8Array(bytes.length + 1); b[0] = carry; b.set(bytes, 1); bytes = b; carry = null; }
          if (bytes.length & 1) { carry = bytes[bytes.length - 1]; bytes = bytes.subarray(0, bytes.length - 1); }
          const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
          for (let i = 0; i < bytes.length; i += 2) {
            const x = view.getInt16(i, true) / 32768;
            frame[fill++] = x;
            const a = Math.abs(x); if (a > peak) peak = a; sumsq += x * x; n++;
            if (fill === BLOCK) {
              if (on) send({ pcm: frame });
              frame = new Float32Array(BLOCK); fill = 0;
              if (++blocks >= METER_EVERY) {
                send({ meter: { peak, rms: Math.sqrt(sumsq / n) } });
                blocks = 0; peak = 0; sumsq = 0; n = 0;
              }
            }
          }
        }
      } catch (_) {}
      if (!ended) {
        track.readyState = 'ended'; ended = true;
        setBadge('Box microphone: the stream stopped', true);
        try { track.onended?.(); } catch (_) {}
      }
    })();
    return { stream, ctx, node };
  }

  globalThis.PiMic = { active, open };
  if (active) {
    if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', startOverlay);
    else startOverlay();
  }
})();
