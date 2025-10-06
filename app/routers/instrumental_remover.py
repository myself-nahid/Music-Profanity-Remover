import os
import tempfile
import shutil
from typing import List

from fastapi import (
    APIRouter, Request, File, UploadFile, Form, Depends
)
from fastapi.responses import FileResponse
from fastapi.templating import Jinja2Templates
from pydub import AudioSegment
from pydub.effects import low_pass_filter, high_pass_filter
import librosa
import numpy as np
import scipy.io.wavfile
from faster_whisper import WhisperModel

from ..dependencies import get_transcription_model, get_instrumental_cache

router = APIRouter()
templates = Jinja2Templates(directory="app/templates")

@router.get("/instrumental-remover", include_in_schema=False)
async def get_upload_form(request: Request):
    """Serves the upload page for the instrumental remover tool."""
    return templates.TemplateResponse("upload.html", {
        "request": request,
        "endpoint": "/instrumental-remover/transcribe"
    })

@router.post("/instrumental-remover/transcribe", include_in_schema=False)
async def transcribe_audio(
    request: Request,
    audio: UploadFile = File(...),
    model: WhisperModel = Depends(get_transcription_model),
    cache: dict = Depends(get_instrumental_cache)
):
    """Transcribes audio and prepares it for editing."""
    # Save the uploaded file to a temporary path
    suffix = os.path.splitext(audio.filename)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        shutil.copyfileobj(audio.file, temp_file)
        temp_path = temp_file.name

    # Transcribe
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

    # Preload instrumental (audio minus vocals) into cache
    y, sr = librosa.load(temp_path, sr=None)
    y_instr = librosa.effects.harmonic(y)
    cache[temp_path] = (y_instr, sr)

    return templates.TemplateResponse("editor.html", {
        "request": request,
        "words": words,
        "orig_path": temp_path,
        "endpoint": "/instrumental-remover/process"
    })

@router.post("/instrumental-remover/process")
async def process_audio_removal(
    orig_path: str = Form(...),
    remove: List[str] = Form([]),
    idx: List[int] = Form(...),
    start: List[float] = Form(...),
    end: List[float] = Form(...),
    cache: dict = Depends(get_instrumental_cache)
):
    """Removes selected words and replaces them with instrumental snippets."""
    audio = AudioSegment.from_file(orig_path)
    y_instr, sr = cache[orig_path]

    words = sorted(list(zip(idx, [s * 1000 for s in start], [e * 1000 for e in end])))
    remove_indices = set(map(int, remove))

    out = AudioSegment.empty()
    cursor = 0
    for idx_val, s_ms, e_ms in words:
        if idx_val in remove_indices:
            gap_ms = int(e_ms - s_ms)
            start_sample = int(s_ms / 1000 * sr)
            snippet = y_instr[start_sample:start_sample + int(gap_ms / 1000 * sr)]

            if len(snippet) == 0:
                fill_seg = AudioSegment.silent(duration=gap_ms)
            else:
                needed = int(np.ceil(gap_ms / 1000 * sr / len(snippet)))
                tiled = np.tile(snippet, needed)[:int(gap_ms / 1000 * sr)]
                with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_fp:
                    scipy.io.wavfile.write(tmp_fp.name, sr, tiled.astype(np.int16))
                    fill_seg = AudioSegment.from_file(tmp_fp.name)

            # Post-processing
            fill_seg = fill_seg.fade_in(100).fade_out(100)
            fill_seg = low_pass_filter(fill_seg, cutoff=2500)
            fill_seg = high_pass_filter(fill_seg, cutoff=100)
            fill_seg = fill_seg.normalize()

            out += audio[cursor:int(s_ms)] + fill_seg
            cursor = int(e_ms)

    out += audio[cursor:]
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as out_fp:
        out.export(out_fp.name, format='mp3')
        return FileResponse(out_fp.name, media_type='audio/mpeg', filename='cleaned_instrumental.mp3')