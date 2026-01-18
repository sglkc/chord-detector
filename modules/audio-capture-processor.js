/**
 * Audio Worklet Processor for Real-time Chord Detection
 * 
 * Runs in a separate audio thread for low-latency audio capture.
 * Collects audio samples and sends them to the main thread at specified intervals.
 */

class AudioCaptureProcessor extends AudioWorkletProcessor {
  constructor(options) {
    super();

    // Configuration
    this.bufferSize = options.processorOptions?.bufferSize || 4096;
    this.sampleRate = options.processorOptions?.sampleRate || 48000;

    // Buffer for collecting samples
    this.buffer = new Float32Array(this.bufferSize);
    this.bufferIndex = 0;

    // Handle messages from main thread
    this.port.onmessage = (event) => {
      if (event.data.type === 'config') {
        this.bufferSize = event.data.bufferSize || this.bufferSize;
        this.buffer = new Float32Array(this.bufferSize);
        this.bufferIndex = 0;
      }
    };
  }

  process(inputs, outputs, parameters) {
    const input = inputs[0];

    if (input && input.length > 0) {
      const samples = input[0]; // First channel (mono)

      if (samples) {
        // Add samples to buffer
        for (let i = 0; i < samples.length; i++) {
          this.buffer[this.bufferIndex++] = samples[i];

          // When buffer is full, send to main thread
          if (this.bufferIndex >= this.bufferSize) {
            // Send a copy of the buffer
            this.port.postMessage({
              type: 'audio',
              samples: this.buffer.slice()
            });

            this.bufferIndex = 0;
          }
        }
      }
    }

    // Keep processor alive
    return true;
  }
}

registerProcessor('audio-capture-processor', AudioCaptureProcessor);
