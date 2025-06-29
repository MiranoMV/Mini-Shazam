import os
import librosa
import pickle
from pydub import AudioSegment
from fingerprinting import preprocess_audio, get_peaks, generate_fingerprints

AUDIO_EXTS = (".mp3", ".m4a", ".flac", ".ogg", ".aac", ".wav", ".wma", ".opus", ".alac")

def convert_to_wav(filepath, outpath):
    try:
        audio = AudioSegment.from_file(filepath)
        audio.export(outpath, format="wav")
        print(f"  ✅ Converted {filepath} to {outpath}")
        return True
    except Exception as e:
        print(f"  ❌ Conversion failed for {filepath}: {e}")
        return False

def build_database(song_folder="music_wavs", db_file="database.pkl"):
    db = {}

    # Convert all audio files to wav if not already wav
    for filename in os.listdir(song_folder):
        ext = os.path.splitext(filename)[1].lower()
        if ext in AUDIO_EXTS and ext != ".wav":
            src = os.path.join(song_folder, filename)
            dst = os.path.join(song_folder, os.path.splitext(filename)[0] + ".wav")
            if not os.path.exists(dst):
                print(f"Converting {filename} to WAV...")
                convert_to_wav(src, dst)

    # Now fingerprint all wavs
    for filename in os.listdir(song_folder):
        if not filename.lower().endswith(".wav"):
            continue
        print(f"Processing: {filename}")
        filepath = os.path.join(song_folder, filename)
        try:
            y, sr = librosa.load(filepath, sr=None, mono=True)
        except Exception as e:
            print(f"❌ Could not load {filename}: {e}")
            continue

        y, sr = preprocess_audio(y, sr)   # Correct usage

        print("  After preprocess: samples =", len(y))

        peaks = get_peaks(y, sr)
        print("  Peaks found:", len(peaks))

        fingerprints = generate_fingerprints(peaks)
        print("  Fingerprints generated:", len(fingerprints))

        for h, t in fingerprints:
            if h not in db:
                db[h] = []
            db[h].append((filename, t))
    with open(db_file, "wb") as f:
        pickle.dump(db, f)
    print(f"Database built and saved to {db_file}")

if __name__ == "__main__":
    build_database()
