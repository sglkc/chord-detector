/**
 * CQT WASM Module Test
 * Tests the AssemblyScript CQT implementation
 */
import { CQTExtractor } from '../index.js';

// Generate test audio signal (sine wave with harmonics)
function generateTestSignal(sampleRate, duration, frequency) {
  const numSamples = Math.floor(sampleRate * duration);
  const signal = new Float32Array(numSamples);

  // Generate fundamental + harmonics (simulating a musical tone)
  for (let i = 0; i < numSamples; i++) {
    const t = i / sampleRate;
    signal[i] = 0.5 * Math.sin(2 * Math.PI * frequency * t);       // Fundamental
    signal[i] += 0.3 * Math.sin(2 * Math.PI * frequency * 2 * t);  // 2nd harmonic
    signal[i] += 0.15 * Math.sin(2 * Math.PI * frequency * 3 * t); // 3rd harmonic
    signal[i] += 0.1 * Math.sin(2 * Math.PI * frequency * 4 * t);  // 4th harmonic
  }

  return signal;
}

async function runTests() {
  console.log('=== CQT WASM Module Tests ===\n');

  const cqt = new CQTExtractor();

  // Test configuration matching the model
  const config = {
    audio: {
      sampleRate: 48000,
      minFrequency: 130.81,  // C3
      hopSize: 512
    },
    classification: {
      cqtBins: 36,
      cqtTimeFrames: 87
    }
  };

  // Generate test signal (C3 = 130.81 Hz)
  const testFrequency = 130.81;
  const duration = 1.0;  // 1 second
  console.log(`Generating test signal: ${testFrequency} Hz (C3), ${duration}s`);

  const audioData = generateTestSignal(
    config.audio.sampleRate,
    duration,
    testFrequency
  );

  console.log(`Audio samples: ${audioData.length}`);
  console.log('');

  // Test 1: Feature extraction
  console.log('Test 1: Feature extraction for classification');
  try {
    const startTime = performance.now();
    const features = await cqt.extractFeatures(audioData, config);
    const endTime = performance.now();

    console.log(`✓ Feature extraction completed in ${(endTime - startTime).toFixed(2)}ms`);
    console.log(`  Output shape: [${config.classification.cqtBins}, ${config.classification.cqtTimeFrames}]`);
    console.log(`  Total values: ${features.length}`);

    // Verify shape
    const expectedSize = config.classification.cqtBins * config.classification.cqtTimeFrames;
    if (features.length === expectedSize) {
      console.log('✓ Output shape is correct');
    } else {
      console.log(`✗ Expected ${expectedSize} values, got ${features.length}`);
    }

    // Check for NaN/Infinity
    let hasInvalidValues = false;
    for (let i = 0; i < features.length; i++) {
      if (!isFinite(features[i])) {
        hasInvalidValues = true;
        break;
      }
    }
    if (!hasInvalidValues) {
      console.log('✓ No invalid values (NaN/Infinity)');
    } else {
      console.log('✗ Found invalid values');
    }

  } catch (error) {
    console.log(`✗ Feature extraction failed: ${error.message}`);
    throw error;
  }

  // Test 2: Full CQT extraction
  console.log('\nTest 2: Full CQT spectrogram extraction');
  try {
    const startTime = performance.now();
    const cqtResult = cqt.extractFullCQT(audioData);
    const endTime = performance.now();

    console.log(`✓ Full CQT completed in ${(endTime - startTime).toFixed(2)}ms`);
    console.log(`  Frames: ${cqtResult.numFrames}`);
    console.log(`  Bins: ${cqtResult.numBins}`);
    console.log(`  FFT length: ${cqtResult.fftLength}`);

  } catch (error) {
    console.log(`✗ Full CQT failed: ${error.message}`);
    throw error;
  }

  // Test 3: Performance benchmark
  console.log('\nTest 3: Performance benchmark (10 iterations)');
  const iterations = 10;
  const times = [];

  for (let i = 0; i < iterations; i++) {
    const startTime = performance.now();
    await cqt.extractFeatures(audioData, config);
    times.push(performance.now() - startTime);
  }

  const avgTime = times.reduce((a, b) => a + b, 0) / times.length;
  const minTime = Math.min(...times);
  const maxTime = Math.max(...times);

  console.log(`  Average: ${avgTime.toFixed(2)}ms`);
  console.log(`  Min: ${minTime.toFixed(2)}ms`);
  console.log(`  Max: ${maxTime.toFixed(2)}ms`);

  console.log('\n=== All tests completed ===');
}

// Run tests
runTests().catch(console.error);
