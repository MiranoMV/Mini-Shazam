import numpy as np
import librosa
import hashlib
from scipy.ndimage import maximum_filter

#This script includes three funtions that can be used in the process of fingerprinting

def preprocess_audio(y, sr, target_sr=44100, target_rms=0.1):
    # 1. Resample to target sample rate
    if sr != target_sr:
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
        sr = target_sr

    # 2. Trim leading/trailing silence (top_db can be relaxed a bit)
    y, _ = librosa.effects.trim(y, top_db=32)

    # 3. Normalize volume to target RMS (robust to loud/quiet differences)
    rms = np.sqrt(np.mean(y**2))
    if rms > 0:
        y = y * (target_rms / rms)

    # 4. Remove DC offset (center signal)
    y = y - np.mean(y)

    # 5. Limit amplitude to [-1, 1]
    max_amp = np.max(np.abs(y))
    if max_amp > 1:
        y = y / max_amp

    return y, sr

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

def generate_fingerprints(peaks, fan_value=20):
    fingerprints = []
    peaks = sorted(peaks, key=lambda x: x[1])  # sort by time
    for i in range(len(peaks)):
        for j in range(1, fan_value):
            if i + j < len(peaks):
                f1, t1 = peaks[i]
                f2, t2 = peaks[i + j]
                dt = t2 - t1
                # Only accept pairs within a reasonable time difference
                if 5 < dt <= 200:
                    # Quantize values to make fingerprinting more robust to noise
                    quant_f1 = int(f1 / 2)
                    quant_f2 = int(f2 / 2)
                    quant_dt = int(dt / 2)
                    h = hashlib.sha1(f"{quant_f1}|{quant_f2}|{quant_dt}".encode()).hexdigest()[:20]
                    fingerprints.append((h, t1))
    return fingerprints