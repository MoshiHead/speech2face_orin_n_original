// @ts-nocheck
function asMs(samples) {
  return (samples * 1000 / sampleRate).toFixed(1);
}

function asSamples(mili) {
  return Math.round(mili * sampleRate / 1000);
}

class MoshiProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    console.log("Moshi processor lives", currentFrame, sampleRate);
    console.log(currentTime);

    // Buffer length definitions
    this.initialBufferSamples = asSamples(160);
    this.initialPartialBufferSamples = asSamples(20);
    this.partialBufferIncrement = asSamples(10);
    this.maxPartialWithIncrements = asSamples(160);
    this.hardMaxBufferSamples = asSamples(1500);
    this.emergencyTargetSamples = asSamples(500);

    // State and metrics
    this.initState();

    this.port.onmessage = (event) => {
      if (event.data.type == "reset") {
        console.log("Reset audio processor state.");
        this.initState();
        return;
      }
      let frame = event.data.frame;
      this.lastMicDuration = Number(event.data.micDuration || 0);
      this.frames.push(frame);
      if (this.currentSamples() >= this.initialBufferSamples && !this.started) {
        this.start();
      }
      if (this.pidx < 20) {
        console.log(this.timestamp(), "Got packet", this.pidx++, asMs(this.currentSamples()), asMs(frame.length))

      }
      if (this.currentSamples() >= this.hardMaxBufferSamples) {
        this.overflowCount += 1;
        let droppedNow = 0;
        console.log(
          this.timestamp(),
          "Emergency audio overflow",
          asMs(this.currentSamples()),
          asMs(this.hardMaxBufferSamples),
        );
        while (this.currentSamples() > this.emergencyTargetSamples) {
          let first = this.frames[0];
          let to_remove = this.currentSamples() - this.emergencyTargetSamples;
          to_remove = Math.min(first.length - this.offsetInFirstBuffer, to_remove);
          this.offsetInFirstBuffer += to_remove;
          droppedNow += to_remove;
          this.timeInStream += to_remove / sampleRate;
          if (this.offsetInFirstBuffer == first.length) {
            this.frames.shift();
            this.offsetInFirstBuffer = 0;
          }
        }
        this.droppedSamples += droppedNow;
        console.log(
          this.timestamp(),
          "Emergency samples dropped",
          asMs(droppedNow),
        );
      }
      this.reportStats();
    };
  }

  initState() {
    this.frames = new Array();
    this.offsetInFirstBuffer = 0;
    this.firstOut = false;
    this.remainingPartialBufferSamples = 0;
    this.timeInStream = 0.;
    this.lastMicDuration = 0.;
    this.reportCounter = 0;
    this.partialBufferSamples = this.initialPartialBufferSamples;
    this.resetStart();

    // Metrics
    this.totalAudioPlayed = 0.;
    this.actualAudioPlayed = 0.;
    this.maxDelay = 0.;
    this.minDelay = 2000.;
    this.underrunCount = 0;
    this.overflowCount = 0;
    this.droppedSamples = 0;
    // Debug
    this.pidx = 0;

  }

  timestamp() {
    return Date.now() % 1000;
  }

  reportStats() {
    this.port.postMessage({
      totalAudioPlayed: this.totalAudioPlayed,
      actualAudioPlayed: this.actualAudioPlayed,
      streamTime: this.timeInStream,
      queuedSamples: this.currentSamples(),
      started: this.started,
      underrunCount: this.underrunCount,
      overflowCount: this.overflowCount,
      droppedSamples: this.droppedSamples,
      delay: this.lastMicDuration - this.timeInStream,
      minDelay: this.minDelay,
      maxDelay: this.maxDelay,
    });
  }

  currentSamples() {
    let samples = 0;
    for (let k = 0; k < this.frames.length; k++) {
      samples += this.frames[k].length
    }
    samples -= this.offsetInFirstBuffer;
    return samples;
  }

  resetStart() {
    this.started = false;
  }

  start() {
    this.started = true;
    this.remainingPartialBufferSamples = this.partialBufferSamples;
    this.firstOut = true;
  }

  canPlay() {
    return this.started && this.frames.length > 0 && this.remainingPartialBufferSamples <= 0;
  }

  process(inputs, outputs, parameters) {
    let delay = this.currentSamples() / sampleRate;
    if (this.canPlay()) {
      this.maxDelay = Math.max(this.maxDelay, delay);
      this.minDelay = Math.min(this.minDelay, delay);
    }
    const output = outputs[0][0];
    if (!this.canPlay()) {
      if (this.actualAudioPlayed > 0) {
        this.totalAudioPlayed += output.length / sampleRate;
      }
      this.remainingPartialBufferSamples -= output.length;
      if (++this.reportCounter >= 8) {
        this.reportCounter = 0;
        this.reportStats();
      }
      return true;
    }
    if (this.firstOut) {
      console.log(this.timestamp(), "Audio resumed", asMs(this.currentSamples()), this.remainingPartialBufferSamples);
    }
    let first = this.frames[0];
    let out_idx = 0;
    while (out_idx < output.length && this.frames.length) {
      let first = this.frames[0];
      let to_copy = Math.min(first.length - this.offsetInFirstBuffer, output.length - out_idx);
      output.set(first.subarray(this.offsetInFirstBuffer, this.offsetInFirstBuffer + to_copy), out_idx);
      this.offsetInFirstBuffer += to_copy;
      out_idx += to_copy;
      if (this.offsetInFirstBuffer == first.length) {
        this.offsetInFirstBuffer = 0;
        this.frames.shift();
      }
    }
    if (this.firstOut) {
      this.firstOut = false;
      for (let i = 0; i < out_idx; i++) {
        output[i] *= i / out_idx;
      }
    }
    if (out_idx < output.length) {
      this.underrunCount += 1;
      console.log(this.timestamp(), "Missed some audio", output.length - out_idx);
      this.partialBufferSamples += this.partialBufferIncrement;
      this.partialBufferSamples = Math.min(this.partialBufferSamples, this.maxPartialWithIncrements);
      console.log("Increased partial buffer to", asMs(this.partialBufferSamples));
      // We ran out of a buffer, let's revert to the started state to replenish it.
      this.resetStart();
      for (let i = 0; i < out_idx; i++) {
        output[i] *= (out_idx - i) / out_idx;
      }
    }
    this.totalAudioPlayed += output.length / sampleRate;
    this.actualAudioPlayed += out_idx / sampleRate;
    this.timeInStream += out_idx / sampleRate;
    if (++this.reportCounter >= 8) {
      this.reportCounter = 0;
      this.reportStats();
    }
    return true;
  }
}
registerProcessor("moshi-processor", MoshiProcessor);
