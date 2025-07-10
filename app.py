import gradio as gr
import openai
import os
import tempfile
import scipy.io.wavfile

openai.api_key = os.getenv("OPENAI_API_KEY")  # Chave segura via variável

def transcrever(audio_file):
    if audio_file is None:
        return "Nenhum áudio enviado."

    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(audio_file.read())
            tmp.flush()
            with open(tmp.name, "rb") as f:
                result = openai.Audio.transcribe("whisper-1", f, language="pt")
                return result["text"]
    except Exception as e:
        return f"Erro ao transcrever: {e}"

gr.Interface(
    fn=transcrever,
    inputs=gr.Audio(source="upload", type="filepath"),
    outputs="text",
    title="Transcritor Whisper OpenAI",
    description="Envie um áudio para transcrição em português usando Whisper da OpenAI"
).launch()