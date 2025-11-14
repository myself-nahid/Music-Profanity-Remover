import os
import tempfile
import shutil
from typing import List, Optional
from enum import Enum

from fastapi import (
    APIRouter, File, UploadFile, Depends, HTTPException
)
from pydantic import BaseModel, Field
from pydub import AudioSegment
from fastapi.responses import FileResponse
import librosa
import numpy as np
import scipy.io.wavfile
from faster_whisper import WhisperModel

from ..dependencies import get_transcription_model, get_instrumental_cache, get_transcript_cache, get_percussive_cache

# --- Pre-load DJ Tag ---
DJ_TAG_PATH = os.getenv(
    'DJ_TAG_PATH', 
    os.path.join(os.path.dirname(__file__), '..', '..', 'dj', 'dj dj gudda.mp3')
)

DJ_TAG = None
DJ_TAG_ERROR = None

try:
    if os.path.exists(DJ_TAG_PATH):
        DJ_TAG = AudioSegment.from_file(DJ_TAG_PATH).normalize()
        print(f"✓ DJ Tag loaded successfully from: {DJ_TAG_PATH}")
    else:
        DJ_TAG_ERROR = f"DJ Tag file not found at: {DJ_TAG_PATH}"
        print(f"✗ {DJ_TAG_ERROR}")
except Exception as e:
    DJ_TAG_ERROR = f"Failed to load DJ Tag: {str(e)}"
    print(f"✗ {DJ_TAG_ERROR}")

router = APIRouter(
    prefix="/api/v2",
    tags=["Mobile API v2 (Enhanced)"]
)

# --- Enhanced Pydantic Models ---
class MixMode(str, Enum):
    """How to blend DJ tag with audio"""
    REPLACE = "replace"  # Original: Replace audio completely
    OVERLAY = "overlay"  # New: Mix DJ tag over instrumental/beat
    DUCKING = "ducking"  # New: Lower original volume, overlay DJ tag

class ProcessRequest(BaseModel):
    orig_path: str
    remove_indices: List[int] = Field(
        description="List of word indices to remove/replace"
    )
    mix_mode: MixMode = Field(
        default=MixMode.OVERLAY,
        description="How to blend DJ tag: replace, overlay, or ducking"
    )
    tag_volume: float = Field(
        default=0.0,
        ge=-20.0,
        le=10.0,
        description="DJ tag volume adjustment in dB (0 = original, -10 = quieter, +5 = louder)"
    )
    background_volume: float = Field(
        default=-8.0,
        ge=-40.0,
        le=0.0,
        description="Background music volume when DJ tag plays (only for ducking mode)"
    )
    use_percussive: bool = Field(
        default=True,
        description="Use percussive (beat) component instead of harmonic for instrumental fill"
    )
    fade_duration: int = Field(
        default=30,
        ge=0,
        le=200,
        description="Crossfade duration in milliseconds"
    )

@router.post("/transcribe")
async def api_transcribe_audio(
    audio: UploadFile = File(...),
    model: WhisperModel = Depends(get_transcription_model),
    transcript_cache: dict = Depends(get_transcript_cache),
):
    """
    Step 1: Upload audio. Server transcribes and caches the word list.
    Returns all words with indices so user can choose which to remove.
    """
    suffix = os.path.splitext(audio.filename)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        shutil.copyfileobj(audio.file, temp_file)
        temp_path = temp_file.name

    segments, _ = model.transcribe(
        temp_path, 
        beam_size=1,
        word_timestamps=True,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=500)
    )
    
    words = []
    idx = 0
    for seg in segments:
        for w in seg.words:
            t = w.word.strip()
            if not t: continue
            words.append({
                'idx': idx, 
                'text': t, 
                'start': w.start, 
                'end': w.end,
                'duration': w.end - w.start
            })
            idx += 1
    
    transcript_cache[temp_path] = words
            
    return {
        "orig_path": temp_path, 
        "words": words,
        "total_words": len(words),
        "message": "Select word indices to remove and choose processing options"
    }


@router.post("/process/instrumental")
async def api_process_instrumental_v2(
    request: ProcessRequest,
    instrumental_cache: dict = Depends(get_instrumental_cache),
    percussive_cache: dict = Depends(get_percussive_cache),
    transcript_cache: dict = Depends(get_transcript_cache),
):
    """
    Step 2: Process audio with high-quality instrumental/percussive fill.
    NEW: Can use percussive (beat) component for cleaner, rhythm-focused fill.
    """
    orig_path = request.orig_path
    
    if orig_path not in transcript_cache:
        raise HTTPException(
            status_code=404, 
            detail="Transcription data not found or expired. Please re-transcribe."
        )
    
    all_words = transcript_cache[orig_path]
    
    # Choose between harmonic (instrumental) or percussive (beat) separation
    if request.use_percussive:
        # Use percussive component (drums, beats, rhythm)
        if orig_path not in percussive_cache:
            y, sr = librosa.load(orig_path, sr=22050)
            # Separate into harmonic and percussive
            y_perc = librosa.effects.percussive(y, margin=3.0)
            percussive_cache[orig_path] = (y_perc, sr)
        y_fill, sr = percussive_cache[orig_path]
        fill_type = "percussive (beat)"
    else:
        # Use harmonic component (melody, instruments)
        if orig_path not in instrumental_cache:
            y, sr = librosa.load(orig_path, sr=22050)
            y_harm = librosa.effects.harmonic(y, margin=2.0)
            instrumental_cache[orig_path] = (y_harm, sr)
        y_fill, sr = instrumental_cache[orig_path]
        fill_type = "harmonic (instrumental)"
    
    audio = AudioSegment.from_file(orig_path)

    out = AudioSegment.empty()
    cursor = 0
    temp_wav = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
    temp_wav.close()
    
    for word_data in all_words:
        s_ms = word_data['start'] * 1000
        e_ms = word_data['end'] * 1000
        
        if word_data['idx'] in request.remove_indices:
            gap_ms = int(e_ms - s_ms)
            start_sample = int(s_ms / 1000 * sr)
            
            # Extract fill segment from the same timestamp (keeps rhythm aligned)
            snippet = y_fill[start_sample:start_sample + int(gap_ms / 1000 * sr)]

            if len(snippet) == 0:
                fill_seg = AudioSegment.silent(duration=gap_ms)
            else:
                # Ensure exact length match
                target_samples = int(gap_ms / 1000 * sr)
                if len(snippet) < target_samples:
                    # Tile if needed
                    needed = int(np.ceil(target_samples / len(snippet)))
                    snippet = np.tile(snippet, needed)[:target_samples]
                else:
                    snippet = snippet[:target_samples]
                
                scipy.io.wavfile.write(temp_wav.name, sr, snippet.astype(np.int16))
                fill_seg = AudioSegment.from_file(temp_wav.name)

            # Apply crossfade for seamless transitions
            fill_seg = fill_seg.fade_in(request.fade_duration).fade_out(request.fade_duration).normalize()
            
            out += audio[cursor:int(s_ms)] + fill_seg
            cursor = int(e_ms)

    out += audio[cursor:]
    
    try:
        os.unlink(temp_wav.name)
    except:
        pass
    
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as out_fp:
        out.export(out_fp.name, format='mp3', bitrate='256k')
        return FileResponse(
            out_fp.name, 
            media_type='audio/mpeg', 
            filename=f'cleaned_{fill_type.replace(" ", "_")}.mp3',
            headers={
                "X-Fill-Type": fill_type,
                "X-Words-Removed": str(len(request.remove_indices))
            }
        )


@router.post("/process/dj-tag")
async def api_process_dj_tag_v2(
    request: ProcessRequest,
    transcript_cache: dict = Depends(get_transcript_cache),
    instrumental_cache: dict = Depends(get_instrumental_cache),
    percussive_cache: dict = Depends(get_percussive_cache),
):
    """
    Step 2: Process audio with DJ tag.
    NEW FEATURES:
    - Overlay mode: Mix DJ tag over beat (keeps rhythm)
    - Ducking mode: Lower music volume, overlay DJ tag
    - Volume control for DJ tag and background
    - User selects which words to tag (full control)
    """
    if DJ_TAG is None:
        error_msg = DJ_TAG_ERROR or "DJ Tag audio file is not loaded on the server."
        raise HTTPException(
            status_code=503,
            detail={
                "error": error_msg,
                "dj_tag_path": DJ_TAG_PATH,
                "solution": "Please set DJ_TAG_PATH environment variable or ensure file exists"
            }
        )

    orig_path = request.orig_path
    
    if orig_path not in transcript_cache:
        raise HTTPException(
            status_code=404, 
            detail="Transcription data not found or expired. Please re-transcribe."
        )
    
    all_words = transcript_cache[orig_path]
    audio = AudioSegment.from_file(orig_path)
    
    # Adjust DJ tag volume
    dj_tag_adjusted = DJ_TAG + request.tag_volume

    if request.mix_mode == MixMode.REPLACE:
        # Original behavior: Replace audio with DJ tag
        segments = []
        cursor = 0
        
        for word_data in all_words:
            s_ms = int(word_data['start'] * 1000)
            e_ms = int(word_data['end'] * 1000)

            if word_data['idx'] in request.remove_indices:
                segments.append(audio[cursor:s_ms])
                segments.append(dj_tag_adjusted)
                cursor = e_ms

        segments.append(audio[cursor:])
        out = sum(segments, AudioSegment.empty())
        
    elif request.mix_mode == MixMode.OVERLAY:
        # NEW: Overlay DJ tag over instrumental/beat (keeps rhythm seamless)
        
        # Get percussive component for background beat
        if request.use_percussive:
            if orig_path not in percussive_cache:
                y, sr = librosa.load(orig_path, sr=22050)
                y_perc = librosa.effects.percussive(y, margin=3.0)
                percussive_cache[orig_path] = (y_perc, sr)
            y_bg, sr = percussive_cache[orig_path]
        else:
            if orig_path not in instrumental_cache:
                y, sr = librosa.load(orig_path, sr=22050)
                y_harm = librosa.effects.harmonic(y, margin=2.0)
                instrumental_cache[orig_path] = (y_harm, sr)
            y_bg, sr = instrumental_cache[orig_path]
        
        temp_wav = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
        temp_wav.close()
        
        out = audio
        
        for word_data in all_words:
            s_ms = int(word_data['start'] * 1000)
            e_ms = int(word_data['end'] * 1000)
            gap_ms = e_ms - s_ms

            if word_data['idx'] in request.remove_indices:
                # Extract beat/instrumental for this segment
                start_sample = int(s_ms / 1000 * sr)
                target_samples = int(gap_ms / 1000 * sr)
                snippet = y_bg[start_sample:start_sample + target_samples]
                
                if len(snippet) > 0:
                    if len(snippet) < target_samples:
                        needed = int(np.ceil(target_samples / len(snippet)))
                        snippet = np.tile(snippet, needed)[:target_samples]
                    
                    scipy.io.wavfile.write(temp_wav.name, sr, snippet.astype(np.int16))
                    bg_seg = AudioSegment.from_file(temp_wav.name).normalize()
                else:
                    bg_seg = AudioSegment.silent(duration=gap_ms)
                
                # Adjust DJ tag to match segment length
                if len(dj_tag_adjusted) > gap_ms:
                    tag_seg = dj_tag_adjusted[:gap_ms]
                else:
                    # Loop DJ tag if needed
                    repeats = int(np.ceil(gap_ms / len(dj_tag_adjusted)))
                    tag_seg = (dj_tag_adjusted * repeats)[:gap_ms]
                
                # Overlay: Mix DJ tag with background beat
                mixed = bg_seg.overlay(tag_seg)
                
                # Replace segment in output
                out = out[:s_ms] + mixed + out[e_ms:]
        
        try:
            os.unlink(temp_wav.name)
        except:
            pass
            
    else:  # MixMode.DUCKING
        # NEW: Lower original audio volume, overlay DJ tag (radio-style ducking)
        out = audio
        
        for word_data in all_words:
            s_ms = int(word_data['start'] * 1000)
            e_ms = int(word_data['end'] * 1000)
            gap_ms = e_ms - s_ms

            if word_data['idx'] in request.remove_indices:
                # Extract and duck the background
                bg_seg = audio[s_ms:e_ms] + request.background_volume  # Lower volume
                
                # Adjust DJ tag length
                if len(dj_tag_adjusted) > gap_ms:
                    tag_seg = dj_tag_adjusted[:gap_ms]
                else:
                    repeats = int(np.ceil(gap_ms / len(dj_tag_adjusted)))
                    tag_seg = (dj_tag_adjusted * repeats)[:gap_ms]
                
                # Overlay DJ tag on ducked background
                mixed = bg_seg.overlay(tag_seg)
                
                # Apply crossfade for smooth transitions
                if request.fade_duration > 0:
                    mixed = mixed.fade_in(request.fade_duration).fade_out(request.fade_duration)
                
                out = out[:s_ms] + mixed + out[e_ms:]
    
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as out_fp:
        out.export(out_fp.name, format='mp3', bitrate='256k')
        return FileResponse(
            out_fp.name, 
            media_type='audio/mpeg', 
            filename=f'cleaned_dj_tag_{request.mix_mode.value}.mp3',
            headers={
                "X-Mix-Mode": request.mix_mode.value,
                "X-Words-Tagged": str(len(request.remove_indices)),
                "X-Tag-Volume": str(request.tag_volume)
            }
        )


@router.get("/dj-tag/status")
async def dj_tag_status():
    """Check if DJ tag is loaded and available."""
    return {
        "loaded": DJ_TAG is not None,
        "path": DJ_TAG_PATH,
        "exists": os.path.exists(DJ_TAG_PATH),
        "error": DJ_TAG_ERROR,
        "duration_ms": len(DJ_TAG) if DJ_TAG else 0
    }


@router.get("/processing-options")
async def get_processing_options():
    """Get available processing options and parameters."""
    return {
        "mix_modes": {
            "replace": "Replace audio completely with DJ tag",
            "overlay": "Mix DJ tag over instrumental/beat (keeps rhythm)",
            "ducking": "Lower music volume and overlay DJ tag (radio-style)"
        },
        "parameters": {
            "tag_volume": {
                "description": "DJ tag volume adjustment in dB",
                "range": [-20.0, 10.0],
                "default": 0.0
            },
            "background_volume": {
                "description": "Background music volume when DJ tag plays (ducking mode only)",
                "range": [-40.0, 0.0],
                "default": -8.0
            },
            "use_percussive": {
                "description": "Use percussive (beat) instead of harmonic (melody) for fill",
                "default": True
            },
            "fade_duration": {
                "description": "Crossfade duration in milliseconds",
                "range": [0, 200],
                "default": 30
            }
        }
    }