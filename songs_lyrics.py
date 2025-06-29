import csv
import os
import re
import lyricsgenius


GENIUS_TOKEN = "pLBE73HN4p__wc9hw8KaB7hJA2cZmTQ7ZPPnjte1WpTK5YvyCD3JQLySD0sOjv_4" # <-- Genius Access Token


def parse_artist_title(display_name):
    # Split on ' - ' (first one), get artist and the rest
    if ' - ' in display_name:
        artist, rest = display_name.split(' - ', 1)
    else:
        # fallback, treat all as title
        return '', display_name

    # Remove stuff like [NCS Release], [Remix], etc.
    title = re.sub(r'\[.*?\]', '', rest).strip()
    # Remove trailing or leading spaces and dashes
    title = title.strip(" -")
    # Remove multiple spaces
    title = re.sub(' +', ' ', title)
    return artist.strip(), title.strip()

def clean_lyrics(lyrics, title):
    lines = lyrics.strip().splitlines()
    # Remove "Song Title Lyrics" line
    if lines and lines[0].strip().lower() == f"{title.lower()} lyrics":
        lines = lines[1:]
    # Remove contributor lines
    lines = [line for line in lines if "Contributor" not in line]
    # Remove empty lines at the top
    while lines and not lines[0].strip():
        lines = lines[1:]
    return "\n".join(lines)

genius = lyricsgenius.Genius(GENIUS_TOKEN, skip_non_songs=True, excluded_terms=["(Remix)", "(Live)"], remove_section_headers=True, timeout=10)

def fetch_lyrics_genius(artist, title):
    try:
        song = genius.search_song(title, artist)
        if song and song.lyrics:
            return song.lyrics
        else:
            return "Lyrics not found on Genius."
    except Exception as e:
        return f"Error: {e}"