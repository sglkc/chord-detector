// mic-processor.js
//
// AudioWorkletProcessor that forwards mono Float32 chunks from the
// browser's input to the main thread via transferable ArrayBuffers.
//
// Registered from app.js via a Blob URL so the worklet does not need
// to be served as a separate file (one less 404 surface area).
//
// Why an AudioWorklet (not a ScriptProcessor)?
//   * Lower latency (runs on a dedicated audio render thread).
//   * Audio params are guaranteed by the spec to be 128-sample
//     blocks; we accumulate to roughly 4096-sample chunks before
//     posting to avoid spamming the main thread.

class MicProcessor extends AudioWorkletProcessor {
  constructor(options) {
    super();
    // ``chunkSize`` is configurable via worklet options so tests can
    // crank it down. Default 4096 samples ~= 85 ms @ 48 kHz.
    const opt = (options && options.processorOptions) || {};
    this.chunkSize = opt.chunkSize || 4096;
    this.buffer = new Float32Array(this.chunkSize);
    this.bufferFill = 0;

    this.port.onmessage = (event) => {
      if (event.data === "stop") {
        this.stopped = true;
      }
    };
    this.stopped = false;
  }

  process(inputs) {
    if (this.stopped) return false;

    const input = inputs[0];
    if (!input || input.length === 0) return true;

    // Take the first channel (mono). Browsers may deliver N>1
    // channels depending on the source; we just want the first.
    const channel = input[0];
    if (!channel || channel.length === 0) return true;

    let read = 0;
    while (read < channel.length) {
      const space = this.chunkSize - this.bufferFill;
      const toCopy = Math.min(space, channel.length - read);
      this.buffer.set(channel.subarray(read, read + toCopy), this.bufferFill);
      this.bufferFill += toCopy;
      read += toCopy;

      if (this.bufferFill === this.chunkSize) {
        // Transfer the underlying buffer to avoid copying on postMessage.
        const out = this.buffer;
        this.port.postMessage(out, [out.buffer]);
        this.buffer = new Float32Array(this.chunkSize);
        this.bufferFill = 0;
      }
    }
    return true;
  }
}

registerProcessor("mic-processor", MicProcessor);
