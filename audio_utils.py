import os
import yt_dlp
from pydub import AudioSegment
from pathlib import Path

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)


def download_youtube_audio(url: str) -> str:
    """Download audio from a YouTube URL using yt-dlp. Returns path to mp3 file."""

    output_path = str(UPLOAD_DIR / "%(title)s.%(ext)s")

    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": output_path,
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
                "preferredquality": "192",
            }
        ],
        "quiet": True,
        "no_warnings": True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        title = info.get("title", "audio")
        # yt-dlp renames to .mp3 after postprocessing
        filename = ydl.prepare_filename(info).rsplit(".", 1)[0] + ".mp3"

    print(f"[audio_utils] Downloaded: {filename}")
    return filename


def preprocess_audio(file_path: str) -> str:
    """
    Use pydub to:
    - Convert any format to mono WAV 16kHz (optimal for Whisper)
    - Normalize volume
    Returns path to processed WAV file.
    """
    audio = AudioSegment.from_file(file_path)

    
    audio = audio.set_channels(1)

   
    audio = audio.set_frame_rate(16000)

   
    target_dBFS = -20.0
    change_in_dBFS = target_dBFS - audio.dBFS
    audio = audio.apply_gain(change_in_dBFS)

    out_path = str(UPLOAD_DIR / (Path(file_path).stem + "_processed.wav"))
    audio.export(out_path, format="wav")

    duration_sec = len(audio) / 1000
    print(f"[audio_utils] Preprocessed → {out_path} | Duration: {duration_sec:.1f}s")
    return out_path


def get_audio_info(file_path: str) -> dict:
    """Return basic metadata about the audio file."""
    audio = AudioSegment.from_file(file_path)
    return {
        "duration_seconds": round(len(audio) / 1000, 2),
        "channels": audio.channels,
        "frame_rate": audio.frame_rate,
        "file_size_mb": round(os.path.getsize(file_path) / (1024 * 1024), 2),
    }