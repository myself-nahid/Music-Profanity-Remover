import os
import subprocess
from pydub import AudioSegment
from faster_whisper import WhisperModel

# --- CONFIGURATION ---
INPUT_FILE = r"C:\Users\Nahid Hasan\Desktop\Nahid\Music-Profanity-Remover\Dee Mula - Blow My High.mp3"
BAD_WORDS = ["fuck", "shit", "bitch", "damn", "ass"] # Expand this list!
MODEL_SIZE = "small" # Use 'medium' or 'large-v2' for better accuracy
OUTPUT_DIR = "stems_output"

def separate_stems(file_path):
    """
    Uses Demucs to split audio into: vocals.wav and no_vocals.wav
    """
    print("✂️  Separating Stems (This takes a moment)...")
    # We use 'htdemucs' (High Quality) and --two-stems=vocals to save time
    # This keeps Drums/Bass/Other merged as one 'instrumental' track
    command = f"demucs -n htdemucs --two-stems=vocals \"{file_path}\" -o {OUTPUT_DIR}"
    subprocess.run(command, shell=True, check=True)
    
    # Construct paths (Demucs output structure)
    filename = os.path.splitext(os.path.basename(file_path))[0]
    vocal_path = os.path.join(OUTPUT_DIR, "htdemucs", filename, "vocals.wav")
    inst_path = os.path.join(OUTPUT_DIR, "htdemucs", filename, "no_vocals.wav")
    
    return vocal_path, inst_path

def get_word_timestamps(vocal_path):
    """
    Uses Whisper to find WHERE the bad words are.
    """
    print("🧠 Listening to Vocals...")
    model = WhisperModel(MODEL_SIZE, device="cpu", compute_type="int8")
    
    segments, _ = model.transcribe(vocal_path, word_timestamps=True)
    
    bad_word_ranges = []
    
    for segment in segments:
        for word in segment.words:
            # Clean punctuation from word (e.g., "fuck!" -> "fuck")
            clean_word = word.word.lower().strip(".,?!\"'")
            if clean_word in BAD_WORDS:
                # Convert seconds to milliseconds
                start_ms = int(word.start * 1000)
                end_ms = int(word.end * 1000)
                bad_word_ranges.append((start_ms, end_ms, clean_word))
                print(f"   Found '{clean_word}' at {word.start:.2f}s")
                
    return bad_word_ranges

def surgical_censor(vocal_path, timestamps):
    """
    The Magic: Edits ONLY the vocal stem using Reverse + Crossfade
    """
    print("💉 Performing Surgery...")
    vocals = AudioSegment.from_file(vocal_path)
    
    # We process in reverse order to not mess up timestamps when cutting
    # (Though here we are replacing, so length stays same, but good practice)
    for start, end, word in reversed(timestamps):
        
        # 1. Grab the dirty slice
        dirty_slice = vocals[start:end]
        
        # 2. THE SMOOTH TEXTURE TRICK: 
        # Instead of silence (which sounds broken), we REVERSE the word.
        # This keeps the flow, the volume, and the rhyme, but hides the meaning.
        clean_slice = dirty_slice.reverse()
        
        # 3. Apply Crossfades (5ms) to prevent "clicking" noises
        clean_slice = clean_slice.fade_in(5).fade_out(5)
        
        # 4. Paste it back in
        vocals = vocals[:start] + clean_slice + vocals[end:]
        
    return vocals

def mix_and_master(vocals, instrumental_path):
    """
    Stitches the clean vocals back onto the banging beat.
    """
    print("🎚️  Mixing Stems...")
    beat = AudioSegment.from_file(instrumental_path)
    
    # Overlay vocals on top of beat
    final_mix = beat.overlay(vocals)
    
    output_name = "Clean_Radio_Edit.mp3"
    final_mix.export(output_name, format="mp3", bitrate="320k")
    print(f"✨ Success! Created {output_name}")

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    try:
        # 1. Split
        voc_file, inst_file = separate_stems(INPUT_FILE)
        
        # 2. Analyze
        timestamps = get_word_timestamps(voc_file)
        
        if not timestamps:
            print("✅ Song is already clean!")
        else:
            # 3. Edit Vocals
            clean_vocals = surgical_censor(voc_file, timestamps)
            
            # 4. Recombine
            mix_and_master(clean_vocals, inst_file)
            
    except Exception as e:
        print(f"❌ Error: {e}")