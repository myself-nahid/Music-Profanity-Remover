from faster_whisper import WhisperModel

import torch

# Detect if GPU is available
device = "cuda" if torch.cuda.is_available() else "cpu"
compute_type = "float16" if device == "cuda" else "int8"

print(f"🚀 Loading Whisper model on {device.upper()}")

model = WhisperModel(
    "tiny",  
    device=device,
    compute_type=compute_type,
    num_workers=4  # Parallel processing
)

print(f"✓ Whisper model loaded: tiny on {device}")

def get_transcription_model():
    return model

# Cache for instrumental tracks
INSTRUMENTAL_CACHE = {}

def get_instrumental_cache():
    return INSTRUMENTAL_CACHE

# Cache to store the word lists between API calls
TRANSCRIPT_CACHE = {}

def get_transcript_cache():
    return TRANSCRIPT_CACHE