import whisper
from pathlib import Path


_model_cache: dict = {}


def load_model(model_name: str = "medium") -> whisper.Whisper:
    """Load and cache Whisper model to avoid reloading on every call."""
    if model_name not in _model_cache:
        print(f"[transcriber] Loading Whisper model: {model_name} ...")
        _model_cache[model_name] = whisper.load_model(model_name)
        print(f"[transcriber] Model '{model_name}' loaded.")
    return _model_cache[model_name]


def transcribe(audio_path: str, model_name: str = "medium", language: str = None) -> dict:
    """
    Transcribe audio using Whisper.

    Args:
        audio_path: Path to audio file (WAV preferred, 16kHz mono)
        model_name: Whisper model size (tiny/base/small/medium/large)
        language: Force language code e.g. 'en', 'hi'. None = auto-detect.

    Returns:
        {
            "text": full transcript string,
            "segments": list of {start, end, text} dicts,
            "language": detected or forced language code
        }
    """
    model = load_model(model_name)

    print(f"[transcriber] Transcribing: {audio_path}")

    options = {
        "task": "transcribe",
        "verbose": False,
    }
    if language:
        options["language"] = language

    result = model.transcribe(audio_path, **options)

    segments = [
        {
            "start": round(seg["start"], 2),
            "end": round(seg["end"], 2),
            "text": seg["text"].strip(),
        }
        for seg in result["segments"]
    ]

    print(f"[transcriber] Done. Detected language: {result['language']} | Segments: {len(segments)}")

    return {
        "text": result["text"].strip(),
        "segments": segments,
        "language": result["language"],
    }