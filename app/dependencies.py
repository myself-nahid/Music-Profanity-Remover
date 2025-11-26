from faster_whisper import WhisperModel
import torch

# Cache settings
MAX_CACHE_ITEMS = 100

# Check if GPU is available
try:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float16" if device == "cuda" else "int8"
    print(f"🚀 Initializing Whisper model on {device.upper()}")
except ImportError:
    device = "cpu"
    compute_type = "int8"
    print(f"🚀 Initializing Whisper model on CPU")

model = WhisperModel("tiny", device=device, compute_type=compute_type, num_workers=4)

def get_transcription_model():
    return model

# Caches with basic size management
INSTRUMENTAL_CACHE = {}
PERCUSSIVE_CACHE = {}
VOCAL_CACHE = {}
TRANSCRIPT_CACHE = {}

def get_transcript_cache():
    # Simple eviction policy: if too big, clear half
    if len(TRANSCRIPT_CACHE) > MAX_CACHE_ITEMS:
        keys = list(TRANSCRIPT_CACHE.keys())[:MAX_CACHE_ITEMS//2]
        for k in keys: del TRANSCRIPT_CACHE[k]
    return TRANSCRIPT_CACHE

def get_instrumental_cache(): return INSTRUMENTAL_CACHE
def get_percussive_cache(): return PERCUSSIVE_CACHE
def get_vocal_cache(): return VOCAL_CACHE