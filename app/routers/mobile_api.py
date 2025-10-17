import os
import tempfile
import shutil
from typing import List

from fastapi import (
    APIRouter, File, UploadFile, Depends, HTTPException
)
from pydantic import BaseModel
from pydub import AudioSegment
from pydub.effects import low_pass_filter, high_pass_filter
from fastapi.responses import FileResponse
import librosa
import numpy as np
import scipy.io.wavfile
from faster_whisper import WhisperModel

# Import the new cache dependency
from ..dependencies import get_transcription_model, get_instrumental_cache, get_transcript_cache

# --- Pre-load DJ Tag ---
DJ_TAG_PATH = 'C:\\Users\\nahid\\Desktop\\JVai-up\\Music-Profanity-Remover\\dj\\dj dj gudda.mp3' 
try:
    DJ_TAG = AudioSegment.from_file(DJ_TAG_PATH).normalize()
except FileNotFoundError:
    DJ_TAG = None

# new versioned router
router = APIRouter(
    prefix="/api/v2",
    tags=["Mobile API v2 (Simplified)"]
)

# --- Pydantic Model for the new simplified request ---
class ProcessRequest(BaseModel):
    orig_path: str
    remove_indices: List[int]

@router.post("/transcribe")
async def api_transcribe_audio(
    audio: UploadFile = File(...),
    model: WhisperModel = Depends(get_transcription_model),
    transcript_cache: dict = Depends(get_transcript_cache),
):
    """
    Step 1: Upload audio. Server transcribes and caches the word list.
    """
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
            if not t: continue
            words.append({'idx': idx, 'text': t, 'start': w.start, 'end': w.end})
            idx += 1
    
    # NEW: Save the full word list to our cache
    transcript_cache[temp_path] = words
            
    return {"orig_path": temp_path, "words": words}


@router.post("/process/instrumental")
async def api_process_instrumental_v2(
    request: ProcessRequest, # Use the Pydantic model for a clean JSON request
    cache: dict = Depends(get_instrumental_cache),
    transcript_cache: dict = Depends(get_transcript_cache),
):
    """Step 2: Process audio with a simple JSON payload."""
    orig_path = request.orig_path
    
    # NEW: Retrieve the full word list from the cache
    if orig_path not in transcript_cache:
        raise HTTPException(status_code=404, detail="Transcription data not found or expired. Please re-transcribe.")
    
    all_words = transcript_cache[orig_path]
    
    # The rest of the logic remains similar...
    if orig_path not in cache:
        y, sr = librosa.load(orig_path, sr=None)
        y_instr = librosa.effects.harmonic(y)
        cache[orig_path] = (y_instr, sr)
        
    audio = AudioSegment.from_file(orig_path)
    y_instr, sr = cache[orig_path]

    out = AudioSegment.empty()
    cursor = 0
    for word_data in all_words:
        s_ms = word_data['start'] * 1000
        e_ms = word_data['end'] * 1000
        
        if word_data['idx'] in request.remove_indices:
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

            fill_seg = fill_seg.fade_in(50).fade_out(50).normalize()
            
            out += audio[cursor:int(s_ms)] + fill_seg
            cursor = int(e_ms)

    out += audio[cursor:]
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as out_fp:
        out.export(out_fp.name, format='mp3')
        return FileResponse(out_fp.name, media_type='audio/mpeg', filename='cleaned_instrumental.mp3')


@router.post("/process/dj-tag")
async def api_process_dj_tag_v2(
    request: ProcessRequest, # Use the Pydantic model
    transcript_cache: dict = Depends(get_transcript_cache),
):
    """Step 2: Process audio with a simple JSON payload."""
    if DJ_TAG is None:
        raise HTTPException(status_code=500, detail="DJ Tag audio file is not loaded on the server.")

    orig_path = request.orig_path
    
    if orig_path not in transcript_cache:
        raise HTTPException(status_code=404, detail="Transcription data not found or expired. Please re-transcribe.")
    
    all_words = transcript_cache[orig_path]
    audio = AudioSegment.from_file(orig_path)

    out = AudioSegment.empty()
    cursor = 0
    for word_data in all_words:
        s_ms = word_data['start'] * 1000
        e_ms = word_data['end'] * 1000

        if word_data['idx'] in request.remove_indices:
            out += audio[cursor:int(s_ms)]
            out += DJ_TAG
            cursor = int(e_ms)

    out += audio[cursor:]
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as out_fp:
        out.export(out_fp.name, format='mp3')
        return FileResponse(out_fp.name, media_type='audio/mpeg', filename='cleaned_dj_tag.mp3')