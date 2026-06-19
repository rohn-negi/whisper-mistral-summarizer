import os
from pathlib import Path
from audio_utils import download_youtube_audio, preprocess_audio, get_audio_info
from transcriber import transcribe
from summarizer import summarize


LANGUAGE_MAP = {
    "English": "en",
    "Hindi": "hi",
    "Spanish": "es",
    "French": "fr",
    "German": "de",
}


def run_pipeline(
    source: str,                   
    whisper_model: str = "medium",
    output_language: str = "English",
    whisper_language: str = None,   
) -> dict:
    """
    Full pipeline:
      1. Download (if YouTube URL) or use uploaded file
      2. Preprocess audio with pydub
      3. Transcribe with Whisper
      4. Summarize with Mistral via LangChain

    Returns a state dict with all results.
    """
    state = {}

   
    is_youtube = source.startswith("http://") or source.startswith("https://")

    if is_youtube:
        print("\n[pipeline] Step 1 — Downloading YouTube audio...")
        raw_path = download_youtube_audio(source)
    else:
        print("\n[pipeline] Step 1 — Using uploaded file...")
        raw_path = source

    state["raw_path"] = raw_path
    state["audio_info"] = get_audio_info(raw_path)
    print(f"[pipeline] Audio info: {state['audio_info']}")

    print("\n[pipeline] Step 2 — Preprocessing audio with pydub...")
    processed_path = preprocess_audio(raw_path)
    state["processed_path"] = processed_path


    print(f"\n[pipeline] Step 3 — Transcribing with Whisper ({whisper_model})...")
    transcription = transcribe(
        audio_path=processed_path,
        model_name=whisper_model,
        language=whisper_language,
    )
    state["transcript"] = transcription["text"]
    state["segments"] = transcription["segments"]
    state["detected_language"] = transcription["language"]

    print(f"\n[pipeline] Transcript preview:\n{state['transcript'][:300]}...\n")

    
    print(f"\n[pipeline] Step 4 — Summarizing with Mistral (output: {output_language})...")
    summary_result = summarize(
        transcript=state["transcript"],
        output_language=output_language,
    )
    state["summary"] = summary_result["summary"]
    state["topics"] = summary_result["topics"]
    state["entities"] = summary_result["entities"]

    print("\n[pipeline] ✓ Pipeline complete.")
    return state


if __name__ == "__main__":
    import sys

    source = input("Enter file path or YouTube URL: ").strip()
    model = input("Whisper model (base/small/medium/large) [medium]: ").strip() or "medium"
    lang = input("Summary output language [English]: ").strip() or "English"

    result = run_pipeline(source=source, whisper_model=model, output_language=lang)

    print("\n" + "="*60)
    print("TRANSCRIPT")
    print("="*60)
    print(result["transcript"])

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(result["summary"])

    print("\n" + "="*60)
    print("TOPICS")
    print("="*60)
    for t in result["topics"]:
        print(f"  • {t}")

    print("\nENTITIES:", ", ".join(result["entities"]))