from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi import FastAPI, Request, File, UploadFile, HTTPException
from transcription import transcribe_audio_bytes

app = FastAPI(title="Transcriptor de Audio", version="1.0")

# Monta estáticos y templates
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    # Renderiza templates/index.html
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):

    # Verifica extensión
    ext = file.filename.rsplit(".", 1)[-1].lower()
    if ext not in ("opus", "mp3", "wav", "m4a"):
        raise HTTPException(status_code=400, detail="Formato no soportado")
    # Lee bytes del archivo
    data = await file.read()
    # Llama a la función de transcripción
    texto = transcribe_audio_bytes(data, ext=ext)
    return {"filename": file.filename, "text": texto}
