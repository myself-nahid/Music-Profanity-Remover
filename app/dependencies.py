from faster_whisper import WhisperModel

try:
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float16" if device == "cuda" else "int8"
    print(f"🚀 Initializing Whisper model on {device.upper()}")
except ImportError:
    device = "cpu"
    compute_type = "int8"
    print(f"🚀 Initializing Whisper model on CPU (torch not installed)")

# Load the Whisper model with optimized settings
model = WhisperModel(
    "tiny",  
    device=device,
    compute_type=compute_type,
    num_workers=4  # Parallel processing threads
)

print(f"✓ Whisper model loaded successfully: tiny on {device}")

def get_transcription_model():
    """Returns the pre-loaded Whisper transcription model"""
    return model


# Cache for harmonic/instrumental tracks (melody and instruments)
INSTRUMENTAL_CACHE = {}

def get_instrumental_cache():
    """Returns cache for harmonic (instrumental) audio separation"""
    return INSTRUMENTAL_CACHE


# NEW: Cache for percussive tracks (drums and beats)
# This is used to extract clean beats for filling censored sections
PERCUSSIVE_CACHE = {}

def get_percussive_cache():
    """Returns cache for percussive (beat/drum) audio separation"""
    return PERCUSSIVE_CACHE


# Cache to store transcribed word lists between API calls
# Maps: orig_path -> list of word dictionaries with timestamps
TRANSCRIPT_CACHE = {}

def get_transcript_cache():
    """Returns cache for storing transcription results between API calls"""
    return TRANSCRIPT_CACHE