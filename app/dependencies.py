from faster_whisper import WhisperModel

# Load the model once
model = WhisperModel("small", device="cpu", compute_type="int8")

def get_transcription_model():
    return model

# Cache for instrumental tracks
INSTRUMENTAL_CACHE = {}

def get_instrumental_cache():
    return INSTRUMENTAL_CACHE

# --- NEW ---
# Cache to store the word lists between API calls
TRANSCRIPT_CACHE = {}

def get_transcript_cache():
    return TRANSCRIPT_CACHE