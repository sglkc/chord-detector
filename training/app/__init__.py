"""Real-time chord detection demo (browser microphone -> librosa + CNN).

Modules
-------
config        : shared constants (CQT, onset, sample-rate, model labels).
server        : FastAPI HTTP + WebSocket entrypoint.
pipeline      : streaming audio -> CQT -> onset -> segment -> classifier.
static        : vanilla HTML/CSS/JS served at /static/.
templates     : Jinja-free HTML page served at /.
"""

__all__ = ["config", "pipeline", "server"]
