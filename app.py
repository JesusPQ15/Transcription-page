from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import traceback
from transcription import transcribe_audio_bytes

app = FastAPI(title="Transcriptor de Audio", version="1.0")

@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    try:
        # Verifica extensión
        ext = file.filename.rsplit(".", 1)[-1].lower()
        if ext not in ("opus", "mp3", "wav", "m4a"):
            raise HTTPException(status_code=400, detail="Formato no soportado")

        # Lee bytes del archivo
        data = await file.read()

        # Llama a la función de transcripción
        texto = transcribe_audio_bytes(data, ext=ext)

        return {"filename": file.filename, "text": texto}

    except Exception as e:
        # Imprime el stack trace en los Service Logs
        traceback.print_exc()
        # Devuelve un JSON con el error al cliente
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )
