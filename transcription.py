# transcription.py
import whisper
import tempfile

# Carga el modelo una sola vez al importar el módulo
_model = whisper.load_model("small")

def transcribe_audio_bytes(audio_bytes: bytes, ext: str = "opus") -> str:
    """
    Recibe el contenido de un archivo de audio en bytes y su extensión,
    escribe un archivo temporal, lo transcribe y devuelve el texto en español.
    """
    # Crea un archivo temporal con la extensión adecuada
    with tempfile.NamedTemporaryFile(suffix=f".{ext}", delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    # Invoca Whisper
    res = _model.transcribe(tmp_path, language="es")
    return res["text"].strip()
