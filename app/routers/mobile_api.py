import os
import tempfile
import shutil
import io
import re
from typing import List, Optional, Tuple
from enum import Enum

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

# --- CLIENT KILL LIST ---
BAD_WORDS = {
    "nigga", "niggaz", "nigger", "niggers", 
    "fuck", "fucking", "fucked", "fucker", "motherfucker",
    "bitch", "bitches", 
    "damn", 
    "ass", 
    "hoe", "hoes",
    "dick", "cock",
    "shit", "shitting",
    "pussy", 
    "slut",
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

def normalize_text(text: str) -> str:
    """Strict English only to fix 'pussy' bug"""
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
        # Aggressive Margin
        y_harm, y_perc = librosa.effects.hpss(y, margin=5.0) 
        
        percussive_bed = numpy_to_audio_segment(y_perc, sr)
        
        # Low Pass at 600Hz
        safe_bed = percussive_bed.low_pass_filter(600)
        
        # Duck volume
        safe_bed = safe_bed - 3.0
        
        return safe_bed.set_channels(2)

    except Exception as e:
        print(f"Separation Error: {e}")
        return AudioSegment.silent(duration=len(segment))

def merge_intervals(intervals: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """Merges overlapping timing intervals to prevent audio glitches"""
    if not intervals:
        return []
    
    # Sort by start time
    intervals.sort(key=lambda x: x[0])
    
    merged = [intervals[0]]
    for current_start, current_end in intervals[1:]:
        last_start, last_end = merged[-1]
        
        # If overlap or adjacent (within 10ms), merge them
        if current_start <= last_end + 10:
            merged[-1] = (last_start, max(last_end, current_end))
        else:
            merged.append((current_start, current_end))
            
    return merged

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

    # --- CHANGE 2: OPTIMIZED TRANSCRIPTION SETTINGS ---
    segments, _ = model.transcribe(
        temp_path, 
        beam_size=5, 
        word_timestamps=True,
        # VAD Filter: Stops it from transcribing the beat as words
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=500),
        # Prevent Loops: Stops "Nicka Nicka Nicka" repetition
        condition_on_previous_text=False,
        # Temperature: Lowers creativity (stick to exactly what is heard)
        temperature=0.0
    )
    
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
        raise HTTPException(404, "Transcription not found or expired.")
    if not os.path.exists(request.orig_path):
        raise HTTPException(404, "Audio file not found on server.")

    all_words = transcript_cache[request.orig_path]
    target_indices = set(request.remove_indices)
    
    if request.max_effects:
        target_indices = set(sorted(list(target_indices))[:request.max_effects])
    
    # Identify the specific words selected
    words_to_process = [w for w in all_words if w['idx'] in target_indices]
    
    try:
        base_track = AudioSegment.from_file(request.orig_path)
    except Exception:
        raise HTTPException(500, "Could not load audio file.")

    librosa_y = None
    librosa_sr = base_track.frame_rate
    if request.effect_type == EffectType.VOCAL_SCRAMBLE:
        librosa_y, librosa_sr = librosa.load(request.orig_path, sr=librosa_sr)

    output_track = base_track
    PAD = 150 # 150ms Padding

    # --- PHASE 1: CALCULATE & MERGE CUT INTERVALS ---
    # We calculate the start/end (with padding) for every word, then merge them.
    # This treats "Damn it" as one cut instead of two overlapping cuts.
    
    raw_intervals = []
    for word in words_to_process:
        s_ms = int(word['start'] * 1000)
        e_ms = int(word['end'] * 1000)
        
        s_ms = max(0, s_ms - PAD)
        e_ms = min(len(base_track), e_ms + PAD)
        raw_intervals.append((s_ms, e_ms))
    
    # Consolidate overlapping cuts
    merged_intervals = merge_intervals(raw_intervals)
    
    # REVERSE SORT: Process from End to Start to maintain index integrity
    merged_intervals.sort(key=lambda x: x[0], reverse=True)

    # --- PHASE 2: EXECUTE CUTS (The Clean Mute) ---
    for s_ms, e_ms in merged_intervals:
        duration_ms = e_ms - s_ms
        original_segment = base_track[s_ms:e_ms]
        
        # 1. Generate Bed
        replacement_chunk = generate_safety_bed(original_segment)
        
        # 2. Vocal Scramble (If needed, we just add it to the whole chunk)
        if request.effect_type == EffectType.VOCAL_SCRAMBLE and librosa_y is not None:
            # We approximate the scramble across the whole merged chunk
            start_sample = int((s_ms / 1000.0) * librosa_sr)
            end_sample = int((e_ms / 1000.0) * librosa_sr)
            y_slice = librosa_y[start_sample:end_sample]
            
            if len(y_slice) > 0:
                y_harm, y_perc = librosa.effects.hpss(y_slice, margin=3.0)
                vocal_seg = numpy_to_audio_segment(y_harm, librosa_sr)
                scrambled = vocal_seg.reverse() - 2.0
                replacement_chunk = replacement_chunk.overlay(scrambled)

        # 3. Micro-Fades
        fade_len = 15
        if len(replacement_chunk) > 30:
            replacement_chunk = replacement_chunk.fade_in(fade_len).fade_out(fade_len)
        
        # 4. Fit Length
        if len(replacement_chunk) != duration_ms:
            if len(replacement_chunk) > duration_ms:
                replacement_chunk = replacement_chunk[:duration_ms]
            else:
                replacement_chunk += AudioSegment.silent(duration=duration_ms - len(replacement_chunk))

        # 5. Apply
        output_track = output_track[:s_ms] + replacement_chunk + output_track[e_ms:]

    # --- PHASE 3: APPLY TAGS (On Top of the Cuts) ---
    # We use the ORIGINAL word locations for tags, not the merged intervals.
    # We sort normally here so tags are applied start-to-end (doesn't matter for overlay)
    
    if request.effect_type in [EffectType.DJ_TAG, EffectType.DJ_SCRATCH]:
        effect_sound = None
        if request.effect_type == EffectType.DJ_TAG and DJ_TAG:
            effect_sound = DJ_TAG + request.effect_volume + 2.0
        elif request.effect_type == EffectType.DJ_SCRATCH and SCRATCH_EFFECT:
            effect_sound = SCRATCH_EFFECT + 4.0

        if effect_sound:
            for word in words_to_process:
                # Start tag at the beginning of the padding for that specific word
                s_ms = max(0, int(word['start'] * 1000) - PAD)
                output_track = output_track.overlay(effect_sound, position=s_ms)

    # --- EXPORT ---
    if output_track.dBFS > -0.5:
        output_track = output_track.normalize(headroom=0.5)

    # FIX: Use len(target_indices) instead of len(indices)
    out_filename = f'final_{request.effect_type.value}_{len(target_indices)}.mp3'
    
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as out_fp:
        output_track.export(out_fp.name, format='mp3', bitrate='320k', parameters=["-q:a", "0"])
        return FileResponse(
            out_fp.name, 
            media_type='audio/mpeg', 
            filename=out_filename,
            headers={"X-Effects-Applied": str(len(words_to_process))}
        )