from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import traceback

from transcription import transcribe_audio_bytes

app = FastAPI(title="Transcriptor de Audio", version="1.0")

# Si quieres tu frontend en "/"
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    try:
        # Verifica extensión soportada
        ext = file.filename.rsplit(".", 1)[-1].lower()
        if ext not in ("opus", "mp3", "wav", "m4a"):
            raise HTTPException(status_code=400, detail="Formato no soportado")

        # Lee bytes del archivo
        data = await file.read()

        # Llama a la función de transcripción (solo pasamos los bytes)
        texto = transcribe_audio_bytes(data)

        return {"filename": file.filename, "text": texto}

    except Exception as e:
        # Stack completo en tus Service Logs
        traceback.print_exc()
        # Error amigable al cliente
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )
