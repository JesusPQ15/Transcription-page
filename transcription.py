import os
import gc
import tempfile
import ffmpeg
import torch
from faster_whisper import WhisperModel

# Limitar PyTorch a un hilo para reducir uso de CPU/RAM
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

# Carga el modelo quantizado (tiny/base/…) desde la variable de entorno
model_name = os.getenv("WHISPER_MODEL", "tiny")
_model = WhisperModel(
    model_size_or_path=model_name,
    device="cpu",
    compute_type="int8"    # quantización 8-bit para menor consumo
)

def preprocess_audio(audio_bytes: bytes) -> bytes:
    """
    Convierte bytes de entrada a WAV mono 8kHz PCM-16.
    """
    process = (
        ffmpeg
        .input("pipe:0")
        .output(
            "pipe:1", format="wav", ac=1, ar=8000,
            sample_fmt="s16", loglevel="error"
        )
        .run_async(pipe_stdin=True, pipe_stdout=True, pipe_stderr=True)
    )
    out, err = process.communicate(audio_bytes)
    if process.returncode != 0:
        raise RuntimeError(f"FFmpeg error: {err.decode().strip()}")
    return out

def split_audio(path: str, chunk_s: int = 30) -> list[str]:
    """
    Divide el archivo WAV en disco en trozos de `chunk_s` segundos a 8kHz.
    Devuelve la lista de rutas a esos fragmentos.
    """
    temp_dir = tempfile.mkdtemp(prefix="chunks_")
    segments = []
    i = 0
    while True:
        out_path = os.path.join(temp_dir, f"chunk_{i}.wav")
        (
            ffmpeg
            .input(path, ss=i * chunk_s, t=chunk_s)
            .output(
                out_path, ac=1, ar=8000, format="wav", loglevel="error"
            )
            .run(quiet=True, overwrite_output=True)
        )
        if not os.path.exists(out_path) or os.path.getsize(out_path) == 0:
            break
        segments.append(out_path)
        i += 1
    return segments

def transcribe_audio_bytes(audio_bytes: bytes) -> str:
    """
    1) Preprocesa el audio en memoria (8kHz).
    2) Guarda como WAV temporal.
    3) Si dura <=30s, transcribe directamente.
       Si dura más, segmenta en trozos de 30s y transcribe cada uno.
    4) Concatena y devuelve el texto.
    5) Limpia recursos y fuerza GC.
    """
    # 1) Preprocesamiento in-memory
    wav_bytes = preprocess_audio(audio_bytes)

    # 2) Guarda el WAV a disco
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(wav_bytes)
        input_path = f.name

    try:
        # Obtener duración real
        info = ffmpeg.probe(input_path, show_entries="format=duration", select_streams="a")
        dur = float(info["format"]["duration"])

        texts: list[str] = []
        if dur <= 30:
            # Transcribe todo de una vez
            segments, _ = _model.transcribe(input_path, language="es")
            texts = [seg.text.strip() for seg in segments]
        else:
            # Divide y transcribe por fragmentos
            chunks = split_audio(input_path, chunk_s=30)
            for chunk in chunks:
                segs, _ = _model.transcribe(chunk, language="es")
                texts.extend([s.text.strip() for s in segs])

        return "\n\n".join(texts)

    finally:
        # Limpieza del archivo temporal
        try:
            os.remove(input_path)
        except OSError:
            pass
        # Fuerza recolección de basura
        gc.collect()
