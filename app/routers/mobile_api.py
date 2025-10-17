import os
import tempfile
import shutil
from typing import List

from fastapi import (
    APIRouter, File, UploadFile, Form, Depends, HTTPException
)
from pydub import AudioSegment
from pydub.effects import low_pass_filter, high_pass_filter
from fastapi.responses import FileResponse
import librosa
import numpy as np
import scipy.io.wavfile
from faster_whisper import WhisperModel

from ..dependencies import get_transcription_model, get_instrumental_cache

# --- Pre-load DJ Tag for the API ---
DJ_TAG_PATH = 'C:\\Users\\nahid\\Desktop\\JVai-up\\Music-Profanity-Remover\\dj\\dj dj gudda.mp3' # IMPORTANT: Change if needed
try:
    DJ_TAG = AudioSegment.from_file(DJ_TAG_PATH).normalize()
except FileNotFoundError:
    DJ_TAG = None

# Create a new router for our API
router = APIRouter(
    prefix="/api/v1",  # All endpoints in this file will start with /api/v1
    tags=["Mobile API"]
)

@router.post("/transcribe")
async def api_transcribe_audio(
    audio: UploadFile = File(...),
    model: WhisperModel = Depends(get_transcription_model),
):
    """
    API endpoint to upload audio and get a JSON response with the transcript.
    This is used by the mobile app.
    """
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
            
    # Return a JSON response
    return {"orig_path": temp_path, "words": words}


@router.post("/process/instrumental")
async def api_process_instrumental(
    orig_path: str = Form(...),
    remove: List[int] = Form(...),
    idx: List[int] = Form(...),
    start: List[float] = Form(...),
    end: List[float] = Form(...),
    cache: dict = Depends(get_instrumental_cache)
):
    """Processes the audio by replacing words with instrumental."""
    # This logic is copied from the web router, but now serves the API
    
    # First, ensure the instrumental is cached if it's not already
    if orig_path not in cache:
        y, sr = librosa.load(orig_path, sr=None)
        y_instr = librosa.effects.harmonic(y)
        cache[orig_path] = (y_instr, sr)
        
    audio = AudioSegment.from_file(orig_path)
    y_instr, sr = cache[orig_path]

    words = sorted(list(zip(idx, [s * 1000 for s in start], [e * 1000 for e in end])))

    out = AudioSegment.empty()
    cursor = 0
    for idx_val, s_ms, e_ms in words:
        if idx_val in remove:
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

            fill_seg = fill_seg.fade_in(100).fade_out(100).normalize()
            fill_seg = low_pass_filter(fill_seg, cutoff=2500)
            fill_seg = high_pass_filter(fill_seg, cutoff=100)
            
            out += audio[cursor:int(s_ms)] + fill_seg
            cursor = int(e_ms)

    out += audio[cursor:]
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as out_fp:
        out.export(out_fp.name, format='mp3')
        return FileResponse(out_fp.name, media_type='audio/mpeg', filename='cleaned_instrumental.mp3')

@router.post("/process/dj-tag")
async def api_process_dj_tag(
    orig_path: str = Form(...),
    remove: List[int] = Form(...),
    idx: List[int] = Form(...),
    start: List[float] = Form(...),
    end: List[float] = Form(...),
):
    """Processes the audio by replacing words with a DJ tag."""
    if DJ_TAG is None:
        raise HTTPException(status_code=500, detail="DJ Tag audio file is not loaded on the server.")

    audio = AudioSegment.from_file(orig_path)
    words = sorted(list(zip(idx, [s * 1000 for s in start], [e * 1000 for e in end])))

    out = AudioSegment.empty()
    cursor = 0
    for idx_val, s_ms, e_ms in words:
        if idx_val in remove:
            out += audio[cursor:int(s_ms)]
            out += DJ_TAG
            cursor = int(e_ms)

    out += audio[cursor:]
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as out_fp:
        out.export(out_fp.name, format='mp3')
        return FileResponse(out_fp.name, media_type='audio/mpeg', filename='cleaned_dj_tag.mp3')