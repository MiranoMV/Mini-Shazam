# =============== IMPORTS =============== #
import streamlit as st                          # Streamlit for the app UI
import os                                       # For filesystem operations
import pickle                                   # For saving/loading song fingerprints database
import librosa                                  # Audio processing (loading, spectrograms, etc)
import lyricsgenius                             # For fetching lyrics (from Genius API)
import numpy as np                              # Numerical ops, e.g., on audio data
import matplotlib.pyplot as plt                 # Plotting (spectrograms, peaks)
import librosa.display                          # Helper for spectrogram display
import sounddevice as sd                        # For recording audio from mic
import soundfile as sf                          # For saving audio to disk
import colorsys                                 # For possible color transforms (optional)
import time                                     # Timing, time-stamps
from collections import Counter                 # Counting matching fingerprints
from pydub import AudioSegment                  # Converting audio formats
import io                                       # In-memory files for plots

# ========== CUSTOM MODULES (YOUR CODE) ========== #
from fingerprinting import preprocess_audio, get_peaks, generate_fingerprints
from songs_db import load_songs, add_song
from songs_lyrics import parse_artist_title, clean_lyrics, fetch_lyrics_genius

# =============== APP CONFIGURATION =============== #
SONG_FOLDER = "music_wavs"                      # Folder where song wavs are stored
DB_FILE = "database.pkl"                        # Path to fingerprints database
os.makedirs(SONG_FOLDER, exist_ok=True)         # Ensure folder exists

# =============== SESSION STATE TRUE RESET =============== #
# Handles a full reset of the app state while keeping history and some widget values
if st.session_state.get("do_reset", False):
    if "history" not in st.session_state:
        st.session_state["history"] = []        # Keep history between resets
    for k in list(st.session_state.keys()):
        if k not in ("slider_record_sec", "query_upload", "history"):
            del st.session_state[k]
    st.session_state["app_stage"] = "choose"    # Go to start page
    st.session_state["recording"] = False
    st.session_state["record_start"] = 0
    st.session_state["record_duration"] = st.session_state.get("slider_record_sec", 6)
    st.session_state["audio_buffer"] = None
    st.session_state["query_path"] = None
    st.session_state["recog_result"] = None
    st.session_state["recog_path"] = None
    st.session_state["do_reset"] = False
    st.rerun()                                 # Instantly re-runs the app for a true reset

# =============== PAGE STYLING (CSS) =============== #
st.set_page_config(page_title="Mini Shazam 👅", page_icon="🎵", layout="centered")
st.markdown("""
    <style>
    html, body, .main {
        background: #181f2b !important;
        color: #F1F1F1 !important;
        font-family: 'Inter', 'Segoe UI', Arial, sans-serif !important;
    }
    .big-btn {
        font-size: 1.6em !important;
        font-weight: 800 !important;
        border-radius: 22px !important;
        box-shadow: 0 4px 22px #1958892a;
        background: linear-gradient(95deg, #289afc 80%, #1565c0 100%);
        color: #fff !important;
        padding: 1.25em 2.8em !important;
        margin: 1.4em 0.85em 1.3em 0.85em !important;
        border: none !important;
        transition: all 0.15s;
        outline: none !important;
    }
    .big-btn:hover {
        background: linear-gradient(95deg, #1565c0 70%, #10549e 100%);
        color: #fff !important;
        box-shadow: 0 6px 32px #19588944;
        transform: scale(1.045);
    }
    .song-card {
        background: #232b3c;
        border-radius: 20px;
        box-shadow: 0 2px 24px #1a2233;
        margin: 2em 0 1em 0;
        padding: 2em 2.5em 1.2em 2.5em;
        text-align: center;
    }
    .song-title {
        font-size: 2.2em;
        font-weight: 800;
        color: #35d2ea;
        margin-bottom: 0.15em;
        letter-spacing: 0.03em;
    }
    .song-match-count {
        font-size: 1.18em;
        color: #b6dbfc;
        margin-bottom: 1em;
    }
    .stCheckbox>label {
        font-size: 1.12em;
        font-weight: 500;
        color: #aee5fa !important;
    }
    .stButton > button {
        font-weight: 2000 !important;
        letter-spacing: 0.8px;
        font-size: 3em;
    }
    </style>
""", unsafe_allow_html=True)


# =============== FINGERPRINT DATABASE CORE =============== #
AUDIO_EXTS = (".mp3", ".m4a", ".flac", ".ogg", ".aac", ".wav", ".wma", ".opus", ".alac")

def build_database(song_folder=SONG_FOLDER, db_file=DB_FILE):
    """Processes all .wav files in the database folder, computes fingerprints, saves as pickle."""
    db = {}
    for filename in os.listdir(song_folder):
        if not filename.lower().endswith(".wav"):
            continue
        filepath = os.path.join(song_folder, filename)
        try:
            y, sr = librosa.load(filepath, sr=8000, mono=True, duration=60)
        except Exception:
            continue
        y, sr = preprocess_audio(y, sr)
        peaks = get_peaks(y, sr)
        fingerprints = generate_fingerprints(peaks)
        for h, t in fingerprints:
            if h not in db:
                db[h] = []
            db[h].append((filename, t))
    with open(db_file, "wb") as f:
        pickle.dump(db, f)

def recognize(query_path, db_file=DB_FILE, song_folder=SONG_FOLDER):
    """Loads the DB, fingerprints the query, finds the best matching song via offset matching."""
    with open(db_file, "rb") as f:
        db = pickle.load(f)
    y, sr = librosa.load(query_path, sr=8000, mono=True, duration=15)
    y, sr = preprocess_audio(y, sr)
    peaks = get_peaks(y, sr)
    fingerprints = generate_fingerprints(peaks)
    offset_counter = Counter()
    for h, t in fingerprints:
        if h in db:
            for (song, song_t) in db[h]:
                delta = song_t - t
                offset_counter[(song, delta)] += 1
    if not offset_counter:
        return None, None
    ((best_song, best_offset), count) = offset_counter.most_common(1)[0]
    return best_song, count

# =============== CACHED AUDIO/PEAKS LOADERS =============== #
@st.cache_data(show_spinner=False)
def get_full_song_audio(song_path):
    """Loads and preprocesses the *full* song audio for classic spectrogram plot."""
    y, sr = librosa.load(song_path, sr=None, mono=True)
    y, sr = preprocess_audio(y, sr)
    return y, sr

@st.cache_data(show_spinner=False)
def get_peaks_plot_data(song_path):
    """Loads, preprocesses, and extracts peaks for a short chunk (for peaks/constellation display)."""
    y, sr = librosa.load(song_path, sr=8000, mono=True)
    y, sr = preprocess_audio(y, sr)
    peaks = get_peaks(y, sr)
    return y, sr, peaks

# =============== SPECTROGRAM/PLOT UTILS =============== #
def plot_debug_spectrogram_img_fast(y, sr, title="Spectrogram of recognized song", progress_callback=None):
    """Plots classic spectrogram image for audio. Progress is optional for Streamlit progress bar."""
    n_fft = 1024
    hop_length = 512
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))
    if progress_callback: progress_callback(40)
    S_db = librosa.amplitude_to_db(S, ref=np.max)
    if progress_callback: progress_callback(80)
    fig, ax = plt.subplots(figsize=(8, 3))
    librosa.display.specshow(S_db, sr=sr, hop_length=hop_length,
                             x_axis='time', y_axis='log', cmap='magma', ax=ax)
    ax.set_title(title)
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", dpi=120)
    plt.close(fig)
    buf.seek(0)
    if progress_callback: progress_callback(100)
    return buf

def plot_spectrogram_peaks_connections_fast(
    y, sr, peaks, fan_value=10, top_n=360, title="Spectrogram + Peaks + Connections"):
    """Plots spectrogram with selected peaks and shows connections (constellations/fingerprint links)."""
    import matplotlib.cm as cm
    n_fft = 1024
    hop_length = 512
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))
    S_db = librosa.amplitude_to_db(S, ref=np.max)
    fig, ax = plt.subplots(figsize=(8, 3))
    librosa.display.specshow(
        S_db, sr=sr, hop_length=hop_length,
        x_axis='time', y_axis='log', cmap='magma', ax=ax
    )
    ax.set_title(title)
    ax.set_ylim(32, sr // 2)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    max_frame = S_db.shape[1]
    peaks_plot = [p for p in peaks if p[1] < max_frame and p[0] < len(freqs)]
    if len(peaks_plot) > 0:
        peak_strengths = [S_db[f, t] for f, t in peaks_plot]
        idx = np.argsort(peak_strengths)[-top_n:]
        peaks_display = [peaks_plot[i] for i in idx]
    else:
        peaks_display = []
    colors = cm.viridis(np.linspace(0, 1, len(peaks_display)))
    peaks_sorted = sorted(peaks_display, key=lambda x: x[1])
    for i, (f1, t1) in enumerate(peaks_sorted):
        color = colors[i]
        for j in range(1, fan_value):
            if i + j < len(peaks_sorted):
                f2, t2 = peaks_sorted[i + j]
                dt = t2 - t1
                if 5 < dt <= 120:
                    freq1 = freqs[f1]
                    time1 = librosa.frames_to_time([t1], sr=sr, hop_length=hop_length, n_fft=n_fft)[0]
                    freq2 = freqs[f2]
                    time2 = librosa.frames_to_time([t2], sr=sr, hop_length=hop_length, n_fft=n_fft)[0]
                    ax.plot([time1, time2], [freq1, freq2], color=color, alpha=0.25, linewidth=0.9, zorder=1)
    if peaks_display:
        times_idx = [p[1] for p in peaks_display]
        freq_bins = [p[0] for p in peaks_display]
        times = librosa.frames_to_time(times_idx, sr=sr, hop_length=hop_length, n_fft=n_fft)
        freqs_plot = freqs[freq_bins]
        ax.scatter(times, freqs_plot, color='cyan', s=26, zorder=2, edgecolors='black', linewidths=0.4)
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", dpi=120)
    plt.close(fig)
    buf.seek(0)
    return buf

# =============== APP STATE MACHINE: NAVIGATION =============== #
app_stage = st.session_state.get("app_stage", "choose")  # "choose", "record", "upload", "result"

st.markdown("<h1 style='font-weight: 700; text-align:center;'>🎵 Mini Shazam</h1>", unsafe_allow_html=True)
st.markdown("<div style='font-size:1.22em; text-align:center;'>Identify a song by recording or uploading!<br>Get lyrics, see its audio fingerprint, and more.</div>", unsafe_allow_html=True)

# =============== "CHOOSE" MAIN MENU (WITH VINYL GIF, BUTTONS, HISTORY) =============== #
if app_stage == "choose":

    # Animated vinyl image at the top
    vinyl_gif_url = "https://media0.giphy.com/media/v1.Y2lkPTc5MGI3NjExbzB6cnlucWQyc3ZiaXc2c3Bpdm96Y3o2d2xmd3gybDczamtpcmNhbSZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/h4HgOdLMIyomSiFh0I/giphy.gif"
    st.markdown(
        f"""<div style='display: flex; justify-content: center; align-items: center; margin: 1.7em 0 1.6em 0;'>
        <img src="{vinyl_gif_url}" alt="Spinning Vinyl" width="220" height="220" style="border-radius:50%; box-shadow:0 3px 32px #0005;">
        </div>
        """, unsafe_allow_html=True)

    #How would you like to recognize your song?
    st.markdown("### How would you like to recognize your song?")
    
    # Two big blue buttons: record or upload
    col_rec, col_up = st.columns(2)

    #Record button
    with col_rec:
        if st.button("🎤 Record with Microphone", key="choose_record", use_container_width=True,help="Record a song with your mic", type="primary"):
            st.session_state["app_stage"] = "record"
            st.rerun()
        st.markdown("""
            <style>
            div[data-testid="stButton"] > button {
                composes: big-btn;
            }
            </style>
        """, unsafe_allow_html=True)

    #Upload song button
    with col_up:
        if st.button("⬆️ Upload Audio File", key="choose_upload", use_container_width=True,help="Upload an audio file", type="primary"):
            st.session_state["app_stage"] = "upload"
            st.rerun()
        st.markdown("""
            <style>
            div[data-testid="stButton"] > button {
                composes: big-btn;
            }
            </style>
        """, unsafe_allow_html=True)

    # --- Add song to DB (expander) only on choose page ---
    with st.expander("➕ Add a song to your database"):
        st.info("Upload any song (MP3, FLAC, WAV, OGG, ...). For best results, use clean studio versions!")
        uploaded_song = st.file_uploader("Choose a song to add:", type=[e[1:] for e in AUDIO_EXTS])
        song_name = st.text_input("Display name for the song:", "")
        spotify_url = st.text_input("Spotify link (optional):", "")
        add_btn = st.button("Add song to database 🎶", use_container_width=True)
        if uploaded_song and song_name and add_btn:
            ext = os.path.splitext(uploaded_song.name)[1].lower()
            file_path = os.path.join(SONG_FOLDER, song_name + ".wav")
            if ext == ".wav":
                with open(file_path, "wb") as f:
                    f.write(uploaded_song.read())
            elif ext in AUDIO_EXTS:
                temp_path = os.path.join(SONG_FOLDER, "temp_input" + ext)
                with open(temp_path, "wb") as f:
                    f.write(uploaded_song.read())
                try:
                    audio = AudioSegment.from_file(temp_path)
                    audio.export(file_path, format="wav")
                    os.remove(temp_path)
                except Exception as e:
                    st.error(f"Failed to convert uploaded file: {e}")
                    os.remove(temp_path)
                    file_path = None
            else:
                st.error("Unsupported file format!")
                file_path = None
            if file_path:
                st.success(f"✅ Added {song_name}.wav to your database!")
                add_song(song_name + ".wav", song_name, spotify_url)
                build_database()
                st.info("Database updated!")


    # --- Last recognized songs (history, up to 5) ---
    if "history" in st.session_state and st.session_state["history"]:
        st.markdown("### Last Recognized Songs")
        for item in reversed(st.session_state["history"][-5:]):  # last 5
            display_name = item['display_name']
            spotify_url = item['spotify_url']
            timestamp = item['timestamp']
            if "open.spotify.com/track/" in spotify_url:
                spotify_id = spotify_url.split("/track/")[1].split("?")[0]
                st.markdown(
                    f"""
                    <div style="margin-bottom: 1.1em;">
                        <iframe src="https://open.spotify.com/embed/track/{spotify_id}"
                            width="100%" height="80" frameborder="0"
                            style="border-radius:16px; box-shadow:0 2px 10px #0004;"
                            allowtransparency="true" allow="encrypted-media"></iframe>
                        <div style="text-align:right; color:#b6dbfc; font-size:0.95em; margin-top:0.2em;">
                            <span style='color:#aaa;'>at {timestamp}</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.markdown(
                    f"""
                    <div style="margin-bottom:0.85em; background:#232b3c; border-radius:14px; padding:0.9em 1.2em;">
                        <b style="color:#35d2ea;">{display_name}</b>
                        <div style="text-align:right; color:#b6dbfc; font-size:1.01em; margin-top:0.15em; margin-bottom:-0.3em;">
                            <span style='color:#aaa;'>at {timestamp}</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

# =============== RECORDING UI =============== #
if app_stage == "record":
    st.markdown("#### Record a sample with your microphone")
    record_sec = st.slider("Seconds to record:", 3, 15, 6, key="slider_record_sec")
    if not st.session_state.get("recording", False):
        if st.button("Start Recording 🎙️", key="record_start_btn", use_container_width=True, help="Record a song with your mic", type="primary"):
            st.session_state["recording"] = True
            st.session_state["record_start"] = time.time()
            st.session_state["record_duration"] = record_sec
            st.rerun()
            st.markdown("""
            <style>
            div[data-testid="stButton"] > button {
                composes: big-btn;
            }
            </style>
        """, unsafe_allow_html=True)
    elif st.session_state["recording"]:
        duration = st.session_state["record_duration"]
        start = st.session_state["record_start"]
        now = time.time()
        elapsed = now - start
        remaining = int(duration - elapsed + 1)
        if st.session_state.get("audio_buffer") is None:
            st.session_state["audio_buffer"] = sd.rec(
                int(duration * 8000), samplerate=8000, channels=1, dtype='float32'
            )
            st.rerun()
        if elapsed < duration:
            bar_width = min(int((elapsed / duration) * 100), 100)
            st.progress(bar_width, text=f"🎤 Recording... {remaining}s left")
            time.sleep(0.09)
            st.rerun()
        else:
            sd.wait()
            temp_path = "query.wav"
            rec = st.session_state.get("audio_buffer")
            if rec is not None:
                sf.write(temp_path, rec.flatten(), 8000)
            st.success("Recorded! 🎵")
            st.session_state["query_path"] = temp_path
            st.session_state["recording"] = False
            st.session_state["audio_buffer"] = None
            st.session_state["app_stage"] = "result"
            st.rerun()
    # Go back button
    if st.button("⬅️ Go Back", key="back_record", use_container_width=True):
        st.session_state["app_stage"] = "choose"
        st.rerun()

# =============== UPLOAD UI =============== #
if app_stage == "upload":
    st.markdown("#### Upload a short WAV audio sample (ideally 6-10s, clear sound).")
    uploaded_query = st.file_uploader("Upload song sample (WAV)", type=["wav"], key="query_upload2")
    if uploaded_query:
        with open("query.wav", "wb") as f:
            f.write(uploaded_query.read())
        st.session_state["query_path"] = "query.wav"
        st.session_state["app_stage"] = "result"
        st.rerun()
    if st.button("⬅️ Go Back", key="back_upload", use_container_width=True):
        st.session_state["app_stage"] = "choose"
        st.rerun()

# =============== RECOGNITION RESULT UI =============== #
if app_stage == "result":
    query_path = st.session_state.get("query_path", None)
    if query_path and os.path.exists(query_path):
        # Run recognition, only if query changed
        if ("recog_result" not in st.session_state or
            st.session_state.get("recog_path") != query_path):
            with st.spinner("🎶 Analyzing and recognizing the song..."):
                best_song, match_count = recognize(query_path)
                songs_info = load_songs()
            st.session_state["recog_result"] = (best_song, match_count, songs_info)
            st.session_state["recog_path"] = query_path
        else:
            best_song, match_count, songs_info = st.session_state["recog_result"]
 
        # Always show title box
        if best_song:
            # Save result to history if not a duplicate
            history = st.session_state.get("history", [])
            info = songs_info.get(best_song, {})
            entry = {
                "song": best_song,
                "display_name": info.get("display_name", best_song),
                "match_count": match_count,
                "spotify_url": info.get("spotify_url", ""),
                "timestamp": time.strftime("%H:%M:%S")  # or use datetime for date too
                }
            if not history or history[-1]["song"] != best_song:
                history.append(entry)
                st.session_state["history"] = history

            display_name = info.get('display_name', best_song)
            spotify_url = info.get('spotify_url', '')

            st.markdown(f"""<div class="song-card">
                <div class="song-title">{display_name}</div>
                <div class="song-match-count">🔎 Matched with <b>{match_count}</b> fingerprints</div>
            </div>""", unsafe_allow_html=True)
            if "open.spotify.com/track/" in spotify_url:
                spotify_id = spotify_url.split("/track/")[1].split("?")[0]
                embed_html = f'''
                <iframe src="https://open.spotify.com/embed/track/{spotify_id}"
                        width="100%" height="360" frameborder="0"
                        style="border-radius:18px; margin:0.7em 0; background:#232b3c;"
                        allowtransparency="true" allow="encrypted-media"></iframe>
                '''
                st.markdown(embed_html, unsafe_allow_html=True)
            elif spotify_url:
                st.markdown(f"[🔗 Open in Spotify]({spotify_url})", unsafe_allow_html=True)
            st.success(f"**Recognized:** {display_name}")


            # --- Show waveform & constellation (optional) ---

            show_waveform = st.checkbox("Show Query Waveform & Peaks", key="show_waveform_peaks")

            if show_waveform:

                # Load and preprocess the query audio
                y, sr = librosa.load(st.session_state["query_path"], sr=8000, mono=True, duration=15)
                y, sr = preprocess_audio(y, sr)
                peaks = get_peaks(y, sr)
                duration = len(y) / sr  # Duration of actual audio


                # PLOT 1: WAVEFORM
                fig1, ax1 = plt.subplots(figsize=(8, 2.5))
                librosa.display.waveshow(y, sr=sr, ax=ax1, color='#3ad1e6')
                ax1.set_title("Raw Waveform (Query)")
                ax1.set_xlim(0, duration)
                st.pyplot(fig1)
                plt.close(fig1)

                st.audio(query_path, format="audio/wav")

                # PLOT 2: CONSTELLATION ON SPECTROGRAM
                n_fft = 1024
                hop_length = 512
                S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))
                S_db = librosa.amplitude_to_db(S, ref=np.max)
                fig2, ax2 = plt.subplots(figsize=(8, 3))
                librosa.display.specshow(S_db, sr=sr, hop_length=hop_length,
                                        x_axis='time', y_axis='log', cmap='magma', ax=ax2)
                ax2.set_title("Constellation Plot (Detected Peaks)")
                ax2.set_xlim(0, duration)  # <- Limit to actual length
                # Overlay peaks/Only plot peaks within the audio's duration
                if peaks:
                    frame_times = librosa.frames_to_time([t for (f, t) in peaks], sr=sr, hop_length=hop_length, n_fft=n_fft)
                    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
                    # Mask for peaks inside audio range
                    mask = frame_times <= duration
                    ax2.scatter(frame_times[mask], [freqs[f] for i, (f, t) in enumerate(peaks) if mask[i]],color='cyan', s=28, zorder=2, edgecolors='black', linewidths=0.4)
                st.pyplot(fig2)
                plt.close(fig2)

                

            # Lyrics section (fetch on demand)
            artist, title = parse_artist_title(display_name)
            lyrics_state_key = f"lyrics_{best_song}"
            if lyrics_state_key not in st.session_state:
                st.session_state[lyrics_state_key] = None
            show_lyrics = st.checkbox("Show Lyrics", key=f"lyrics_toggle_{best_song}")
            if show_lyrics and st.session_state[lyrics_state_key] is None:
                with st.spinner("Fetching lyrics from Genius..."):
                    lyrics = fetch_lyrics_genius(artist, title)
                st.session_state[lyrics_state_key] = lyrics
            if show_lyrics:
                lyrics = st.session_state.get(lyrics_state_key, "")
                if lyrics and lyrics.strip() and not lyrics.lower().startswith('lyrics not found'):
                    cleaned = clean_lyrics(lyrics, title)
                    st.subheader("🎼 Lyrics")
                    st.code(cleaned)
                else:
                    st.info("No lyrics found for this song.")

            st.markdown("---")


            # Spectrogram visualizations
            st.subheader("🔊 Audio Fingerprint (Spectrogram)")
            song_path = os.path.join(SONG_FOLDER, best_song)

            y_full, sr_full = get_full_song_audio(song_path)
            classic_img_key = f"classic_spectrogram_img_{best_song}"
            if classic_img_key not in st.session_state:
                progress = st.progress(0, text="Preparing full-song spectrogram...")
                def prog_cb(val): progress.progress(val, text="Preparing full-song spectrogram...")
                st.session_state[classic_img_key] = plot_debug_spectrogram_img_fast(
                    y_full, sr_full, "Spectrogram of recognized song", progress_callback=prog_cb
                )
                progress.empty()
            st.image(st.session_state[classic_img_key], use_container_width=True)

            y_short, sr_short, peaks_short = get_peaks_plot_data(song_path)
            peaks_img_key = f"peaks_spectrogram_img_{best_song}"
            if peaks_img_key not in st.session_state:
                st.session_state[peaks_img_key] = None

            show_peaks = st.checkbox("Show Peaks & Connections", key=f"showpeaksbtn_{best_song}")
            if show_peaks:
                if st.session_state[peaks_img_key] is None:
                    st.session_state[peaks_img_key] = plot_spectrogram_peaks_connections_fast(
                        y_short, sr_short, peaks_short, fan_value=5, top_n=60,
                        title="Spectrogram + Peaks + Connections"
                    )
                st.image(st.session_state[peaks_img_key], use_container_width=True)

            st.markdown("---")
            st.button("🔄 Start Over", key="reset_btn", use_container_width=True, on_click=lambda: st.session_state.update({"do_reset": True}))
        else:
            st.error("❌ No match found. Try a longer/clearer sample, or add more songs to your database.")

st.caption("Made with Streamlit · Local, fast and private · By Milan Dragacevac!")
