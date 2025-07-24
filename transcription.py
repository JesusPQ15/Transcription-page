import os
import whisper
import tempfile
import ffmpeg
import glob

# Lee el modelo desde la variable de entorno (por defecto "small")
model_name = os.getenv("WHISPER_MODEL", "tiny")
_model = whisper.load_model(model_name)

def split_audio(path: str, chunk_s: int = 30) -> list[str]:
    """
    Divide el archivo en disco en trozos de `chunk_s` segundos.
    Devuelve la lista de rutas a esos fragmentos.
    """
    # Carpeta temporal para los segmentos
    temp_dir = tempfile.mkdtemp(prefix="chunks_")
    i = 0
    segments = []
    while True:
        out_path = os.path.join(temp_dir, f"chunk_{i}.wav")
        try:
            (
                ffmpeg
                .input(path, ss=i * chunk_s, t=chunk_s)
                .output(out_path, ac=1, ar=16000, format="wav", loglevel="error")
                .run(overwrite_output=True)
            )
        except ffmpeg.Error:
            break  # no quedan más datos
        if not os.path.exists(out_path) or os.path.getsize(out_path) == 0:
            break
        segments.append(out_path)
        i += 1
    return segments


def transcribe_audio_bytes(audio_bytes: bytes, ext: str = "opus") -> str:
    """
    Recibe bytes de audio, los guarda temporalmente, los divide en trozos,
    transcribe cada uno y devuelve el texto concatenado.
    """
    # 1) Guarda bytes de entrada en un archivo temporal
    with tempfile.NamedTemporaryFile(suffix=f".{ext}", delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    try:
        # 2) Divide en trozos de 30 s
        chunks = split_audio(tmp_path, chunk_s=30)

        # 3) Transcribe cada trozo y concatena
        textos = []
        for seg_path in chunks:
            result = _model.transcribe(seg_path, language="es")
            textos.append(result["text"].strip())

        # Si no hubo segmentos (audio < 30 s), cae al caso de 1 solo chunk
        if not textos:
            res = _model.transcribe(tmp_path, language="es")
            return res["text"].strip()

        return "\n\n".join(textos)

    finally:
        # 4) Limpia todos los temporales
        try:
            os.remove(tmp_path)
        except OSError:
            pass
        for seg in chunks:
            try:
                os.remove(seg)
            except OSError:
                pass
        # y elimina el directorio si está vacío
        try:
            os.rmdir(os.path.dirname(chunks[0]))
        except Exception:
            pass