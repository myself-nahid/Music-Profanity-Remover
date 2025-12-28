import os
import tempfile
import shutil
import io
import re
import subprocess
from typing import List, Optional, Tuple
from enum import Enum

from fastapi import APIRouter, File, UploadFile, Depends, HTTPException, BackgroundTasks
from pydantic import BaseModel
from pydub import AudioSegment, effects
from fastapi.responses import FileResponse
from faster_whisper import WhisperModel

# Ensure these are imported from your dependencies file
from ..dependencies import (
    get_transcription_model, 
    get_transcript_cache, 
)

# --- CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DJ_TAG_PATH = os.path.join(BASE_DIR, '..', 'dj', 'dj dj gudda.mp3')
STEMS_OUTPUT_DIR = os.path.join(BASE_DIR, 'temp_stems')

os.makedirs(STEMS_OUTPUT_DIR, exist_ok=True)

DJ_TAG = None

# --- CLIENT KILL LIST ---
BAD_WORDS = {
    "nigga", "niggaz", "nigger", "niggers", 
    "fuck", "fucking", "fucked", "fucker", "motherfucker",
    "bitch", "bitches", 
    "damn", "ass", "hoe", "hoes", "dick", "cock",
    "shit", "shitting", "pussy", "slut",
    "nicka", "nickas", "nika", "nicker", "fck", "sht", "bih",
    "hell", "whore", "ass,", "ass."
}

# --- HELPERS ---

def load_assets():
    global DJ_TAG
    if DJ_TAG is None and os.path.exists(DJ_TAG_PATH):
        raw = AudioSegment.from_file(DJ_TAG_PATH).normalize()
        DJ_TAG = effects.compress_dynamic_range(raw, threshold=-10.0, ratio=3.0)
        DJ_TAG = DJ_TAG.fade_in(10).fade_out(100)
        print("✓ DJ Tag Loaded & Mastered")

def normalize_text(text: str) -> str:
    # Aggressive cleaning: remove everything that isn't a letter
    return re.sub(r'[^a-z]', '', text.lower())

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
    print(f"🚀 Starting AI Separation for: {input_path}")
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

def generate_reverb_tail(source_segment: AudioSegment, target_duration_ms: int) -> AudioSegment:
    """Generates a smooth ghost tail from the previous word."""
    tail_seed = source_segment[-50:] 
    loops_needed = (target_duration_ms // 50) + 1
    ghost_fill = tail_seed * loops_needed
    ghost_fill = ghost_fill[:target_duration_ms]
    ghost_fill = ghost_fill.fade_out(target_duration_ms)
    ghost_fill = ghost_fill - 6.0
    return ghost_fill

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
    fade_duration: int = 15
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
        full_song_path = temp_file.name

    # Step 1: Separate Vocals
    try:
        print("🚀 Isolating Vocals...")
        stems = separate_stems(full_song_path)
        vocal_path_for_ai = stems["vocals"]
    except Exception:
        vocal_path_for_ai = full_song_path 

    # Step 2: Transcribe
    print("🎤 Transcribing...")
    segments, _ = model.transcribe(
        vocal_path_for_ai, 
        beam_size=5, 
        word_timestamps=True,
        vad_filter=True,
        condition_on_previous_text=False
    )
    
    raw_words = []
    for seg in segments:
        for w in seg.words:
            # Clean punctuation logic moved to normalize_text
            t = w.word.strip()
            if not t: continue
            raw_words.append({'text': t, 'start': w.start, 'end': w.end})
    
    cleaned_words_data = clean_hallucinations(raw_words)
    
    final_words = []
    idx = 0
    for w in cleaned_words_data:
        clean_text = normalize_text(w['text'])
        
        # IMPROVED DETECTION
        is_profanity = False
        if clean_text in BAD_WORDS:
            is_profanity = True
        # Catch compound words like "dumbass" or "jackass"
        elif "ass" in clean_text and len(clean_text) > 3: 
             if any(x in clean_text for x in ["dumb", "jack", "kick", "kiss", "bad"]):
                 is_profanity = True

        final_words.append({
            'idx': idx,
            'text': w['text'],
            'start': w['start'],
            'end': w['end'],
            'is_profanity': is_profanity
        })
        idx += 1
    
    transcript_cache[full_song_path] = {
        "words": final_words,
        "stems": stems
    }
    
    return {"orig_path": full_song_path, "words": final_words}

@router.post("/process")
async def api_process_precise(
    request: ProcessRequest,
    background_tasks: BackgroundTasks,
    transcript_cache: dict = Depends(get_transcript_cache),
):
    if request.orig_path not in transcript_cache:
        raise HTTPException(404, "Transcription not found.")
    
    cached_data = transcript_cache[request.orig_path]
    all_words = cached_data["words"]
    stems_paths = cached_data.get("stems")

    target_indices = set(request.remove_indices)
    if request.max_effects:
        target_indices = set(sorted(list(target_indices))[:request.max_effects])
    
    words_to_process = [w for w in all_words if w['idx'] in target_indices]
    
    # 1. Load Stems
    try:
        if stems_paths and os.path.exists(stems_paths["vocals"]):
            vocal_track = AudioSegment.from_file(stems_paths["vocals"])
            instrumental_track = AudioSegment.from_file(stems_paths["instrumental"])
        else:
            stems = separate_stems(request.orig_path)
            vocal_track = AudioSegment.from_file(stems["vocals"])
            instrumental_track = AudioSegment.from_file(stems["instrumental"])
    except Exception as e:
        raise HTTPException(500, f"Processing Error: {e}")

    # 2. Edit Vocals
    words_to_process.sort(key=lambda x: x['start'], reverse=True)
    
    # WIDER PADDING for short words like "Ass"
    PAD_START = 50
    PAD_END = 150

    for word in words_to_process:
        s_ms = int(word['start'] * 1000)
        e_ms = int(word['end'] * 1000)
        
        s_ms = max(0, s_ms - PAD_START)
        e_ms = min(len(vocal_track), e_ms + PAD_END)
        duration_ms = e_ms - s_ms
        
        # Get context for ghost fill
        context_start = max(0, s_ms - 500)
        previous_context = vocal_track[context_start:s_ms]
        
        # Generate Ghost Fill (Spectral Repair)
        ghost_fill = AudioSegment.silent(duration=duration_ms)
        if len(previous_context) > 50:
            ghost_fill = generate_reverb_tail(previous_context, duration_ms)
            
        fade_len = request.fade_duration
        vocal_before = vocal_track[:s_ms].fade_out(fade_len)
        vocal_after = vocal_track[e_ms:].fade_in(fade_len)
        
        vocal_track = vocal_before + ghost_fill + vocal_after

    # 3. Merge
    final_mix = instrumental_track.overlay(vocal_track)
    
    # Ensure full length (Fixes Outro Bug)
    if len(final_mix) < len(instrumental_track):
         final_mix += AudioSegment.silent(duration=len(instrumental_track) - len(final_mix))

    # 4. Tag
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
        return FileResponse(out_fp.name, media_type='audio/mpeg', filename=out_filename)