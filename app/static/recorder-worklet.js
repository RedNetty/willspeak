class Capture extends AudioWorkletProcessor {
  constructor() {
    super();
    this.on = false;
    this.blocks = 0; this.peak = 0; this.sumsq = 0; this.n = 0;
    this.port.onmessage = (e) => {
      if (e.data === 'start') this.on = true;
      else if (e.data === 'stop') this.on = false;
    };
  }
  process(inputs) {
    const ch = inputs[0] && inputs[0][0];
    if (!ch) return true;
    for (let i = 0; i < ch.length; i++) {
      const a = Math.abs(ch[i]);
      if (a > this.peak) this.peak = a;
      this.sumsq += ch[i] * ch[i];
    }
    this.n += ch.length;
    if (this.on) this.port.postMessage({ pcm: new Float32Array(ch) });
    if (++this.blocks >= 16) {
      this.port.postMessage({ meter: { peak: this.peak, rms: Math.sqrt(this.sumsq / this.n) } });
      this.blocks = 0; this.peak = 0; this.sumsq = 0; this.n = 0;
    }
    return true;
  }
}
registerProcessor('capture', Capture);
