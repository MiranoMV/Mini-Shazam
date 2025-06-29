import csv
import os

SONGS_CSV = "songs_db.csv"

def load_songs():
    """
    Load songs from the CSV as a dict:
    {filename: {"display_name": ..., "spotify_url": ...}, ...}
    """
    songs = {}
    if not os.path.exists(SONGS_CSV):
        return songs
    with open(SONGS_CSV, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 3:
                continue
            filename, display_name, spotify_url = row
            filename = filename.strip()
            display_name = display_name.strip()
            spotify_url = spotify_url.strip()
            if not filename:
                continue
            songs[filename] = {"display_name": display_name, "spotify_url": spotify_url}
    return songs

def add_song(filename, display_name, spotify_url):
    """
    Add a song to the CSV database if it does not already exist.
    """
    filename = filename.strip()
    display_name = display_name.strip()
    spotify_url = spotify_url.strip()

    exists = False
    if os.path.exists(SONGS_CSV):
        with open(SONGS_CSV, newline="", encoding="utf-8") as f:
            for row in csv.reader(f):
                # Compare stripped filenames (robust)
                if row and row[0].strip() == filename:
                    exists = True
                    break

    if not exists:
        with open(SONGS_CSV, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f, quoting=csv.QUOTE_ALL)
            writer.writerow([filename, display_name, spotify_url])
