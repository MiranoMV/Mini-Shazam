import os
from pydub import AudioSegment

# Folders
input_folder = "music_mp3s"
output_folder = "music_wavs"

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Convert all .mp3 and .mpeg files if not already converted
for filename in os.listdir(input_folder):
    if filename.lower().endswith(('.mp3', '.mpeg')):
        input_path = os.path.join(input_folder, filename)
        output_name = os.path.splitext(filename)[0] + ".wav"
        output_path = os.path.join(output_folder, output_name)
        
        # Check if output file already exists
        if os.path.exists(output_path):
            print(f"Skipped {output_name} (already exists)")
            continue
        
        print(f"Converting {filename} to {output_name}...")
        audio = AudioSegment.from_file(input_path)
        audio.export(output_path, format="wav")
        print(f"Saved to {output_path}")

print("All conversions done!")
