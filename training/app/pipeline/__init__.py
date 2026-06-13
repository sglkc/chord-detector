"""Streaming DSP pipeline for the chord-detection demo.

Sub-modules
-----------
audio_buffer    : bounded ring buffer for raw audio samples.
cqt_stream      : incremental Constant-Q Transform of the audio buffer.
onset_detector  : Superflux onset envelope + peak picking.
segment_buffer  : state machine that builds classification windows.
classifier      : CNN inference (TensorFlow / Keras).
runner          : top-level orchestrator wired to a WebSocket.
"""
