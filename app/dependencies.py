from faster_whisper import WhisperModel
import torch

# Check if GPU is available
try:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float16" if device == "cuda" else "int8"
    print(f"🚀 Initializing Whisper model on {device.upper()}")
except ImportError:
    device = "cpu"
    compute_type = "int8"
    print(f"🚀 Initializing Whisper model on CPU")

# --- CHANGE 1: UPGRADE MODEL SIZE ---
# "tiny" -> fast but dumb.
# "small" -> good balance for lyrics.
# "medium" -> best accuracy but slower.
MODEL_SIZE = "small" 

model = WhisperModel(
    MODEL_SIZE, 
    device=device,
    compute_type=compute_type,
    num_workers=4
)

print(f"✓ Whisper model loaded: {MODEL_SIZE} on {device}")

def get_transcription_model():
    return model

# ... (Keep the cache dictionaries below) ...
INSTRUMENTAL_CACHE = {}
PERCUSSIVE_CACHE = {}
VOCAL_CACHE = {}
TRANSCRIPT_CACHE = {}

def get_transcript_cache():
    if len(TRANSCRIPT_CACHE) > 100:
        keys = list(TRANSCRIPT_CACHE.keys())[:50]
        for k in keys: del TRANSCRIPT_CACHE[k]
    return TRANSCRIPT_CACHE

def get_instrumental_cache(): return INSTRUMENTAL_CACHE
def get_percussive_cache(): return PERCUSSIVE_CACHE
def get_vocal_cache(): return VOCAL_CACHE