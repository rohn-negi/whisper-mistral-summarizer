import gradio as gr
from pipeline import run_pipeline

WHISPER_MODELS = ["tiny", "base", "small", "medium", "large"]
LANGUAGES = ["English", "Hindi", "Spanish", "French", "German"]


def process(audio_file, youtube_url, whisper_model, output_language):
    """Gradio handler — called when user clicks Transcribe."""

    if not audio_file and not youtube_url.strip():
        return "⚠️ Provide an audio file or YouTube URL.", "", "", ""

    source = youtube_url.strip() if youtube_url.strip() else audio_file

    try:
        result = run_pipeline(
            source=source,
            whisper_model=whisper_model,
            output_language=output_language,
        )
    except Exception as e:
        return f"❌ Error: {e}", "", "", ""

    transcript = result["transcript"]
    summary = result["summary"]
    topics = "\n".join(f"• {t}" for t in result["topics"]) or "—"
    entities = ", ".join(result["entities"]) or "—"

    return transcript, summary, topics, entities


with gr.Blocks(title="Whisper + Mistral Summarizer") as demo:
    gr.Markdown("# 🎙️ Whisper + Mistral — Transcribe & Summarize")
    gr.Markdown("Upload an audio file or paste a YouTube link. Whisper transcribes, Mistral summarizes.")

    with gr.Row():
        with gr.Column(scale=1):
            audio_input = gr.Audio(
                label="Upload audio file",
                type="filepath",
                sources=["upload"],
            )
            yt_input = gr.Textbox(
                label="Or paste YouTube URL",
                placeholder="https://www.youtube.com/watch?v=...",
            )
            whisper_model = gr.Dropdown(
                choices=WHISPER_MODELS,
                value="medium",
                label="Whisper model",
            )
            output_lang = gr.Dropdown(
                choices=LANGUAGES,
                value="English",
                label="Summary language",
            )
            submit_btn = gr.Button("Transcribe & Summarize", variant="primary")

        with gr.Column(scale=2):
            transcript_out = gr.Textbox(label="Transcript", lines=10)
            summary_out = gr.Textbox(label="Summary", lines=8)
            with gr.Row():
                topics_out = gr.Textbox(label="Key Topics", lines=5)
                entities_out = gr.Textbox(label="Named Entities", lines=5)

    submit_btn.click(
        fn=process,
        inputs=[audio_input, yt_input, whisper_model, output_lang],
        outputs=[transcript_out, summary_out, topics_out, entities_out],
    )


if __name__ == "__main__":
    demo.launch()