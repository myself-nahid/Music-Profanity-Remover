import os
import tempfile
import shutil
import io
import re
import subprocess
from typing import List, Optional
from enum import Enum

from fastapi import APIRouter, File, UploadFile, Depends, HTTPException, BackgroundTasks
from pydantic import BaseModel
from pydub import AudioSegment, effects
from fastapi.responses import FileResponse
from faster_whisper import WhisperModel

from ..dependencies import (
    get_transcription_model, 
    get_transcript_cache, 
)

# --- CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DJ_TAG_PATH = os.path.join(BASE_DIR, '..', 'dj', 'dj dj gudda.mp3')
STEMS_OUTPUT_DIR = os.path.join(BASE_DIR, 'temp_stems')

# Ensure temp directory exists
os.makedirs(STEMS_OUTPUT_DIR, exist_ok=True)

DJ_TAG = None

# --- CLIENT KILL LIST ---
BAD_WORDS = {
    "nigga", "niggaz", "nigger", "niggers", 
    "fuck", "fucking", "fucked", "fucker", "motherfucker",
    "bitch", "bitches", 
    "damn", "ass", "hoe", "hoes", "dick", "cock",
    "shit", "shitting", "pussy", "slut",
    "nicka", "nickas", "nika", "nicker", "fck", "sht", "bih"
}

# --- HELPERS ---

def load_assets():
    global DJ_TAG
    if DJ_TAG is None and os.path.exists(DJ_TAG_PATH):
        raw = AudioSegment.from_file(DJ_TAG_PATH).normalize()
        # Compress DJ Tag
        DJ_TAG = effects.compress_dynamic_range(raw, threshold=-10.0, ratio=3.0)
        DJ_TAG = DJ_TAG.fade_in(10).fade_out(100)
        print("✓ DJ Tag Loaded & Mastered")

def normalize_text(text: str) -> str:
    return re.sub(r'[^a-zA-Z\s]', '', text).lower().strip()

def clean_hallucinations(words: List[dict]) -> List[dict]:
    cleaned = []
    repetition_count = 0
    last_text = ""
    for w in words:
        duration = w['end'] - w['start']
        text = normalize_text(w['text'])
        if duration < 0.05: continue 
        if text == last_text and len(text) > 0:
            repetition_count += 1
        else:
            repetition_count = 0
            last_text = text
        if repetition_count > 4: continue
        cleaned.append(w)
    return cleaned

def separate_stems(input_path: str) -> dict:
    """
    Separates audio using Demucs.
    Returns: {'vocals': path, 'instrumental': path}
    """
    print(f"🚀 Starting AI Separation for: {input_path}")
    
    # -n htdemucs: High Quality
    # --two-stems=vocals: Splits into 'vocals.wav' and 'no_vocals.wav' (Instrumental)
    cmd = [
        "demucs",
        "-n", "htdemucs", 
        "--two-stems", "vocals",
        str(input_path),
        "-o", STEMS_OUTPUT_DIR
    ]
    
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        print(f"Demucs Error: {e}")
        raise HTTPException(500, "AI Stem Separation Failed.")

    filename_no_ext = os.path.splitext(os.path.basename(input_path))[0]
    result_dir = os.path.join(STEMS_OUTPUT_DIR, "htdemucs", filename_no_ext)
    
    return {
        "vocals": os.path.join(result_dir, "vocals.wav"),
        "instrumental": os.path.join(result_dir, "no_vocals.wav")
    }

# --- ROUTER ---
router = APIRouter(prefix="/api/v2", tags=["Mobile API v2"])

class EffectType(str, Enum):
    VOCAL_SCRAMBLE = "vocal_scramble"
    DJ_SCRATCH = "dj_scratch"
    CLEAN_MUTE = "clean_mute"
    DJ_TAG = "dj_tag" 

class ProcessRequest(BaseModel):
    orig_path: str
    remove_indices: List[int] 
    effect_type: EffectType = EffectType.DJ_TAG
    max_effects: Optional[int] = None
    effect_volume: float = 0.0
    fade_duration: int = 15 # 15ms fade avoids clicking
    tag_start_time: float = 0.0 

@router.on_event("startup")
async def startup_event():
    load_assets()

@router.post("/transcribe")
async def api_transcribe_audio(
    audio: UploadFile = File(...),
    model=Depends(get_transcription_model),
    transcript_cache: dict = Depends(get_transcript_cache),
):
    suffix = os.path.splitext(audio.filename)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        shutil.copyfileobj(audio.file, temp_file)
        temp_path = temp_file.name

    # Use VAD filter to ignore beats during transcription
    segments, _ = model.transcribe(
        temp_path, 
        beam_size=5, 
        word_timestamps=True,
        vad_filter=True,
        condition_on_previous_text=False
    )
    
    raw_words = []
    for seg in segments:
        for w in seg.words:
            t = w.word.strip()
            if not t: continue
            raw_words.append({'text': t, 'start': w.start, 'end': w.end})
    
    cleaned_words_data = clean_hallucinations(raw_words)
    
    final_words = []
    idx = 0
    for w in cleaned_words_data:
        clean_text = normalize_text(w['text'])
        is_profanity = clean_text in BAD_WORDS
        final_words.append({
            'idx': idx,
            'text': w['text'],
            'start': w['start'],
            'end': w['end'],
            'is_profanity': is_profanity
        })
        idx += 1
    
    transcript_cache[temp_path] = final_words
    return {"orig_path": temp_path, "words": final_words}

@router.post("/process")
async def api_process_precise(
    request: ProcessRequest,
    background_tasks: BackgroundTasks,
    transcript_cache: dict = Depends(get_transcript_cache),
):
    if request.orig_path not in transcript_cache:
        raise HTTPException(404, "Transcription not found.")
    
    all_words = transcript_cache[request.orig_path]
    target_indices = set(request.remove_indices)
    
    if request.max_effects:
        target_indices = set(sorted(list(target_indices))[:request.max_effects])
    
    words_to_process = [w for w in all_words if w['idx'] in target_indices]
    
    # ---------------------------------------------------------
    # STEP 1: AI STEM SEPARATION
    # ---------------------------------------------------------
    try:
        stems = separate_stems(request.orig_path)
        vocal_track = AudioSegment.from_file(stems["vocals"])
        instrumental_track = AudioSegment.from_file(stems["instrumental"])
    except Exception as e:
        print(e)
        raise HTTPException(500, "Failed to separate stems.")

    # ---------------------------------------------------------
    # STEP 2: EDIT THE VOCAL TRACK (INSTRUMENTAL DROP TECHNIQUE)
    # ---------------------------------------------------------
    
    # Sort reverse so index slicing doesn't break
    words_to_process.sort(key=lambda x: x['start'], reverse=True)

    # Padding: 80ms to catch the edges of words
    PAD = 80

    for word in words_to_process:
        s_ms = int(word['start'] * 1000)
        e_ms = int(word['end'] * 1000)
        
        # Apply safety padding
        s_ms = max(0, s_ms - PAD)
        e_ms = min(len(vocal_track), e_ms + PAD)
        duration_ms = e_ms - s_ms
        
        # --- THE FIX ---
        # Instead of Reversing, we replace the vocal word with SILENCE.
        # This creates a "hole" in the vocal track.
        # When overlayed on the instrumental, the instrumental will shine through.
        clean_slice = AudioSegment.silent(duration=duration_ms)
        
        # We apply a fade OUT to the audio before the cut, and a fade IN after.
        # This prevents the "click/pop" sound.
        fade_len = request.fade_duration # 15ms is usually perfect
        
        # Cut and fade the remaining vocal parts
        vocal_before = vocal_track[:s_ms].fade_out(fade_len)
        vocal_after = vocal_track[e_ms:].fade_in(fade_len)
        
        # Reassemble Vocal Track
        vocal_track = vocal_before + clean_slice + vocal_after

    # ---------------------------------------------------------
    # STEP 3: MERGE (Seamless Playback)
    # ---------------------------------------------------------
    # We overlay the "Holey" Vocals onto the "Continuous" Instrumental.
    # The beat never stops because instrumental_track was never touched.
    final_mix = instrumental_track.overlay(vocal_track)

    # ---------------------------------------------------------
    # STEP 4: DJ TAG (Once at Start/Drop)
    # ---------------------------------------------------------
    if request.effect_type == EffectType.DJ_TAG and DJ_TAG:
        tag_time_ms = int(request.tag_start_time * 1000)
        if tag_time_ms < len(final_mix) - 1000:
            tag_sound = DJ_TAG + request.effect_volume + 2.0
            final_mix = final_mix.overlay(tag_sound, position=tag_time_ms)

    # Export
    if final_mix.dBFS > -0.5:
        final_mix = final_mix.normalize(headroom=0.5)

    out_filename = f'final_serato_{len(target_indices)}.mp3'
    
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as out_fp:
        final_mix.export(out_fp.name, format='mp3', bitrate='320k', parameters=["-q:a", "0"])
        
        # Cleanup
        try:
            shutil.rmtree(STEMS_OUTPUT_DIR)
            os.makedirs(STEMS_OUTPUT_DIR, exist_ok=True)
        except: pass
        
        return FileResponse(
            out_fp.name, 
            media_type='audio/mpeg', 
            filename=out_filename,
            headers={"X-Effects-Applied": str(len(words_to_process))}
        )