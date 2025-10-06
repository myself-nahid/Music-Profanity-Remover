from faster_whisper import WhisperModel
# Load the model once and reuse it
model = WhisperModel("small", device="cpu", compute_type="int8")

def get_transcription_model():
    return model

# Cache for instrumental tracks (used by instrumental_remover)
INSTRUMENTAL_CACHE = {}

def get_instrumental_cache():
    return INSTRUMENTAL_CACHE