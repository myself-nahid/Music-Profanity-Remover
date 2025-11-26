import os
import tempfile
import shutil
import io
from typing import List, Optional
from enum import Enum
import re

from fastapi import APIRouter, File, UploadFile, Depends, HTTPException, BackgroundTasks
from pydantic import BaseModel
from pydub import AudioSegment, silence, effects
from fastapi.responses import FileResponse
import librosa
import numpy as np
import scipy.io.wavfile

from ..dependencies import (
    get_transcription_model, 
    get_transcript_cache, 
)

# --- CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DJ_TAG_PATH = os.path.join(BASE_DIR, '..', 'dj', 'dj dj gudda.mp3')
SCRATCH_EFFECT_PATH = os.path.join(BASE_DIR, '..', 'effects', 'scratch.mp3')

DJ_TAG = None
SCRATCH_EFFECT = None

# --- UPDATED KILL LIST (Includes Whisper Hallucinations) ---
BAD_WORDS = {
    # The Client's List
    "nigga", "niggaz", "nigger", "niggers", 
    "fuck", "fucking", "fucked", "fucker",
    "bitch", "bitches", 
    "damn", 
    "ass", 
    "hoe", "hoes",
    "dick", 
    "shit", "shitting",
    "pussy", 
    "slut",
    # Whisper Common Misinterpretations/Hallucinations
    "nicka", "nickas", "nika", "nicker", "fck", "sht", "bih"
}

# --- HELPERS ---

def load_assets():
    global DJ_TAG, SCRATCH_EFFECT
    if DJ_TAG is None and os.path.exists(DJ_TAG_PATH):
        raw = AudioSegment.from_file(DJ_TAG_PATH).normalize()
        raw = trim_silence(raw, silence_thresh=-40)
        DJ_TAG = effects.compress_dynamic_range(raw, threshold=-10.0, ratio=3.0)
        DJ_TAG = DJ_TAG.fade_in(10).fade_out(100)
        print("✓ DJ Tag Loaded & Mastered")
    
    if SCRATCH_EFFECT is None and os.path.exists(SCRATCH_EFFECT_PATH):
        SCRATCH_EFFECT = AudioSegment.from_file(SCRATCH_EFFECT_PATH).normalize()
        print("✓ Scratch Effect Loaded")

def trim_silence(audio_segment, silence_thresh=-50.0):
    if len(audio_segment) == 0: return audio_segment
    chunks = silence.split_on_silence(
        audio_segment, min_silence_len=50, silence_thresh=silence_thresh, keep_silence=0
    )
    return chunks[0] if chunks else audio_segment

def segment_to_numpy(segment: AudioSegment):
    seg_mono = segment.set_channels(1)
    samples = np.array(seg_mono.get_array_of_samples())
    if seg_mono.sample_width == 2: 
        samples = samples.astype(np.float32) / 32768.0
    elif seg_mono.sample_width == 4:
        samples = samples.astype(np.float32) / 2147483648.0
    return samples

def numpy_to_audio_segment(audio_array: np.ndarray, sample_rate: int) -> AudioSegment:
    audio_array = np.clip(audio_array, -1.0, 1.0)
    audio_int16 = (audio_array * 32767).astype(np.int16)
    byte_io = io.BytesIO()
    scipy.io.wavfile.write(byte_io, sample_rate, audio_int16)
    byte_io.seek(0)
    return AudioSegment.from_wav(byte_io)

def generate_safety_bed(segment: AudioSegment) -> AudioSegment:
    """SAFETY MODE: Percussive Separation + Low Pass + Ducking"""
    if len(segment) < 20: return segment.silent(duration=len(segment))

    try:
        y = segment_to_numpy(segment)
        sr = segment.frame_rate
        # Aggressive Margin 5.0
        y_harm, y_perc = librosa.effects.hpss(y, margin=5.0) 
        
        percussive_bed = numpy_to_audio_segment(y_perc, sr)
        
        # Low Pass at 600Hz to kill vocal frequencies
        safe_bed = percussive_bed.low_pass_filter(600)
        
        # Duck volume
        safe_bed = safe_bed - 3.0
        
        return safe_bed.set_channels(2)

    except Exception as e:
        print(f"Separation Error: {e}")
        return AudioSegment.silent(duration=len(segment))

# --- NEW: HALLUCINATION CLEANER ---
def clean_hallucinations(words: List[dict]) -> List[dict]:
    """
    Removes Whisper loop glitches (e.g., 'Nicka' repeating 50 times).
    Logic: 
    1. If a word is extremely short (<0.02s) it's likely noise.
    2. If the exact same text repeats > 3 times with < 0.1s gaps, delete it.
    """
    cleaned = []
    repetition_count = 0
    last_text = ""
    
    for w in words:
        duration = w['end'] - w['start']
        text = w['text'].lower()
        
        # Filter 1: Ghost Words (Zero or Near-Zero Duration)
        if duration < 0.05: 
            continue 

        # Filter 2: Infinite Repetition Loop (The "Nicka" Bug)
        if text == last_text:
            repetition_count += 1
        else:
            repetition_count = 0
            last_text = text
            
        # If the same word appears more than 4 times in a row, stop adding it
        if repetition_count > 4:
            continue
            
        cleaned.append(w)
        
    return cleaned

def normalize_text(text: str) -> str:
    # Remove punctuation and lowercase
    return re.sub(r'[^\w\s]', '', text).lower()

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

    segments, _ = model.transcribe(temp_path, beam_size=5, word_timestamps=True)
    
    raw_words = []
    for seg in segments:
        for w in seg.words:
            t = w.word.strip()
            if not t: continue
            raw_words.append({
                'text': t, 
                'start': w.start, 
                'end': w.end
            })
    
    # STEP 1: Clean Hallucinations (Fixes the json output glitches)
    cleaned_words_data = clean_hallucinations(raw_words)
    
    # STEP 2: Index and Flag
    final_words = []
    idx = 0
    for w in cleaned_words_data:
        clean_text = normalize_text(w['text'])
        # Check against BAD_WORDS set
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
    
    if not os.path.exists(request.orig_path):
        raise HTTPException(404, "Audio file not found.")

    all_words = transcript_cache[request.orig_path]
    indices = set(request.remove_indices)
    
    if request.max_effects:
        indices = set(sorted(list(indices))[:request.max_effects])
    
    words_to_process = [w for w in all_words if w['idx'] in indices]
    
    try:
        base_track = AudioSegment.from_file(request.orig_path)
    except Exception:
        raise HTTPException(500, "Could not load audio.")

    librosa_y = None
    librosa_sr = base_track.frame_rate
    if request.effect_type == EffectType.VOCAL_SCRAMBLE:
        librosa_y, librosa_sr = librosa.load(request.orig_path, sr=librosa_sr)

    output_track = base_track

    # --- PASS 1: SAFETY CUT ---
    for word in words_to_process:
        s_ms = int(word['start'] * 1000)
        e_ms = int(word['end'] * 1000)
        
        # AGGRESSIVE PADDING 150ms
        pad = 150 
        s_ms = max(0, s_ms - pad)
        e_ms = min(len(base_track), e_ms + pad)
        duration_ms = e_ms - s_ms
        
        original_segment = base_track[s_ms:e_ms]
        
        # Generate Safety Bed (Filtered)
        replacement_chunk = generate_safety_bed(original_segment)
        
        # Vocal Scramble Overlay
        if request.effect_type == EffectType.VOCAL_SCRAMBLE and librosa_y is not None:
            start_sample = int((s_ms / 1000.0) * librosa_sr)
            end_sample = int((e_ms / 1000.0) * librosa_sr)
            y_slice = librosa_y[start_sample:end_sample]
            
            if len(y_slice) > 0:
                y_harm, y_perc = librosa.effects.hpss(y_slice, margin=3.0)
                vocal_seg = numpy_to_audio_segment(y_harm, librosa_sr)
                scrambled = vocal_seg.reverse() - 2.0
                replacement_chunk = replacement_chunk.overlay(scrambled)

        fade_len = 15
        if len(replacement_chunk) > 30:
            replacement_chunk = replacement_chunk.fade_in(fade_len).fade_out(fade_len)
        
        if len(replacement_chunk) != duration_ms:
            if len(replacement_chunk) > duration_ms:
                replacement_chunk = replacement_chunk[:duration_ms]
            else:
                replacement_chunk += AudioSegment.silent(duration=duration_ms - len(replacement_chunk))

        output_track = output_track[:s_ms] + replacement_chunk + output_track[e_ms:]

    # --- PASS 2: DJ TAG ---
    if request.effect_type in [EffectType.DJ_TAG, EffectType.DJ_SCRATCH]:
        effect_sound = None
        if request.effect_type == EffectType.DJ_TAG and DJ_TAG:
            effect_sound = DJ_TAG + request.effect_volume + 2.0
        elif request.effect_type == EffectType.DJ_SCRATCH and SCRATCH_EFFECT:
            effect_sound = SCRATCH_EFFECT + 4.0

        if effect_sound:
            for word in words_to_process:
                # Start tag early to cover transition
                s_ms = max(0, int(word['start'] * 1000) - 150)
                output_track = output_track.overlay(effect_sound, position=s_ms)

    # --- EXPORT ---
    if output_track.dBFS > -0.5:
        output_track = output_track.normalize(headroom=0.5)

    out_filename = f'final_{request.effect_type.value}_{len(indices)}.mp3'
    
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as out_fp:
        output_track.export(out_fp.name, format='mp3', bitrate='320k', parameters=["-q:a", "0"])
        return FileResponse(
            out_fp.name, 
            media_type='audio/mpeg', 
            filename=out_filename,
            headers={"X-Effects-Applied": str(len(words_to_process))}
        )