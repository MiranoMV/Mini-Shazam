'''import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from scipy.ndimage import maximum_filter

# ---------- Paste your get_peaks function here ----------
def get_peaks(y, sr):
    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=256))
    S_db = librosa.amplitude_to_db(S, ref=np.max)

    # Lower dB threshold, bigger local_max window for more tolerance
    local_max = maximum_filter(S_db, size=(21, 15)) == S_db
    detected_peaks = np.argwhere(local_max & (S_db > -65))

    peaks_by_time = {}
    for f, t in detected_peaks:
        peaks_by_time.setdefault(t, []).append((f, t))
    # Allow up to 12 strongest peaks per time slice (was 7)
    strong_peaks = []
    for t, plist in peaks_by_time.items():
        plist.sort(key=lambda x: S_db[x[0], x[1]], reverse=True)
        strong_peaks.extend(plist[:12])
    return strong_peaks

# ---------------------------------------------------------

# --- Change this to your song file path ---
SONG_FILE = "Warriyo, Laura Brehm - Mortals (feat. Laura Brehm) [NCS Release].wav"  # e.g., "Alan Walker - Dreamer [NCS Release].wav"

print("Loading song...")
y, sr = librosa.load(SONG_FILE, sr=None, mono=True)

print("Finding peaks...")
peaks = get_peaks(y, sr)

print(f"Number of peaks: {len(peaks)}")

# ---- Make the spectrogram ----
S = np.abs(librosa.stft(y, n_fft=2048, hop_length=256))
S_db = librosa.amplitude_to_db(S, ref=np.max)

plt.figure(figsize=(14, 6))
librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='log', cmap='magma')
plt.title(f"Spectrogram and Peaks: {SONG_FILE}")
plt.colorbar(format='%+2.0f dB')
plt.xlabel("Time (s)")
plt.ylabel("Frequency (Hz)")

# Overlay the peaks: convert time/frame to x,y on the spectrogram
times = librosa.frames_to_time([t for f, t in peaks], sr=sr, hop_length=256)
freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
peak_freqs = [freqs[f] for f, t in peaks]
plt.scatter(times, peak_freqs, color="cyan", s=8, marker="x", alpha=0.7, label="Peaks")
plt.legend(loc="upper right")
plt.tight_layout()
plt.show()

'''
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt

# Load audio
y, sr = librosa.load('Warriyo, Laura Brehm - Mortals (feat. Laura Brehm) [NCS Release].wav')
print("Audio duration (s):", len(y)/sr)

# Make STFT (spectrogram)
hop_length = 512
S = librosa.stft(y, n_fft=2048, hop_length=hop_length)

# Correct time axis (in seconds)
times = librosa.frames_to_time(np.arange(S.shape[1]), sr=sr, hop_length=hop_length)
print("Spectrogram duration (s):", times[-1])

# Plot spectrogram with correct time axis
plt.figure(figsize=(12, 4))
librosa.display.specshow(librosa.amplitude_to_db(np.abs(S)),
                        sr=sr, hop_length=hop_length,
                        x_axis='time', y_axis='hz')
plt.title("Spectrogram with correct time axis")
plt.show()
