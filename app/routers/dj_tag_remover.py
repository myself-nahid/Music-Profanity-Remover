import os
import tempfile
import shutil
from typing import List

from fastapi import (
    APIRouter, Request, File, UploadFile, Form, Depends, HTTPException
)
from fastapi.responses import FileResponse
from fastapi.templating import Jinja2Templates
from pydub import AudioSegment
from faster_whisper import WhisperModel

from ..dependencies import get_transcription_model

router = APIRouter()
templates = Jinja2Templates(directory="app/templates")
# Path to the DJ tag audio file
DJ_TAG_PATH = 'C:\\Users\\nahid\\Desktop\\JVai-up\\Music-Profanity-Remover\\dj\\dj dj gudda.mp3'

# Load DJ tag into memory
try:
    DJ_TAG = AudioSegment.from_file(DJ_TAG_PATH).normalize()
except FileNotFoundError:
    DJ_TAG = None # Handle missing file gracefully

@router.get("/dj-tag-remover", include_in_schema=False)
async def get_upload_form(request: Request):
    """Serves the upload page for the DJ tag remover tool."""
    if DJ_TAG is None:
        raise HTTPException(status_code=500, detail=f"DJ Tag audio file not found at: {DJ_TAG_PATH}")
    return templates.TemplateResponse("upload.html", {
        "request": request,
        "endpoint": "/dj-tag-remover/transcribe"
    })

@router.post("/dj-tag-remover/transcribe", include_in_schema=False)
async def transcribe_audio(
    request: Request,
    audio: UploadFile = File(...),
    model: WhisperModel = Depends(get_transcription_model)
):
    """Transcribes audio and prepares it for editing."""
    suffix = os.path.splitext(audio.filename)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        shutil.copyfileobj(audio.file, temp_file)
        temp_path = temp_file.name

    segments, _ = model.transcribe(temp_path, beam_size=3, word_timestamps=True)
    words = []
    idx = 0
    for seg in segments:
        for w in seg.words:
            t = w.word.strip()
            if not t:
                continue
            words.append({'idx': idx, 'text': t, 'start': w.start, 'end': w.end})
            idx += 1

    return templates.TemplateResponse("editor.html", {
        "request": request,
        "words": words,
        "orig_path": temp_path,
        "endpoint": "/dj-tag-remover/process"
    })

@router.post("/dj-tag-remover/process")
async def process_audio_removal(
    orig_path: str = Form(...),
    remove: List[str] = Form([]),
    idx: List[int] = Form(...),
    start: List[float] = Form(...),
    end: List[float] = Form(...),
):
    """Removes selected words and replaces them with a DJ tag."""
    if DJ_TAG is None:
        raise HTTPException(status_code=500, detail="DJ Tag audio file is not loaded.")

    audio = AudioSegment.from_file(orig_path)
    words = sorted(list(zip(idx, [s * 1000 for s in start], [e * 1000 for e in end])))
    remove_indices = set(map(int, remove))

    out = AudioSegment.empty()
    cursor = 0
    for idx_val, s_ms, e_ms in words:
        if idx_val in remove_indices:
            out += audio[cursor:int(s_ms)]
            out += DJ_TAG
            cursor = int(e_ms)

    out += audio[cursor:]
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as out_fp:
        out.export(out_fp.name, format='mp3')
        return FileResponse(out_fp.name, media_type='audio/mpeg', filename='cleaned_dj_tag.mp3')