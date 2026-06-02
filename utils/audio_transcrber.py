import os
import re
import sys
import torch
import whisper
from pathlib import Path
from tqdm import tqdm


MODEL_NAME = "small"

INPUT_DIR = Path("./data/raw_audio")
OUTPUT_DIR = Path("./data/audio_transcript")

def extract_clean_title(filename: str) -> tuple[str, str]:
    stem = Path(filename).stem
    pattern = re.compile(r"Marvin\s+Mins?[ky]{2}\s*[\-–—]\s*(.*?)\s*(?:\(\d+[\u29f8/]\d+\))?$", re.IGNORECASE)
    match = pattern.search(stem)
    if match:
        raw_title = match.group(1).strip()
    else:
        raw_title = stem.strip()
        
    h1_title = raw_title.strip("'\"“”極‘’＂")
    
    clean_name = h1_title.lower()
    clean_name = re.sub(r"['\"“”極‘’＂：()（）;；:,\.\?\!！‘’]", "", clean_name)
    clean_name = re.sub(r"[\s\-\_]+", "_", clean_name)
    clean_name = clean_name.strip("_")
    
    return h1_title, f"{clean_name}.md"

def main():
    if not INPUT_DIR.exists():
        print(f"Error: Input directory '{INPUT_DIR.resolve()}' does not exist.")
        print("Please place your raw audio files in that directory.")
        sys.exit(1)
        
    extensions = {".mp3", ".m4a", ".wav"}
    audio_files = sorted([f for f in INPUT_DIR.glob("*") if f.suffix.lower() in extensions])
    
    if not audio_files:
        print(f"Error: No audio files found in '{INPUT_DIR.resolve()}'.")
        sys.exit(1)
        
    if not OUTPUT_DIR.exists():
        print(f"Creating output directory: {OUTPUT_DIR}")
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        
    if torch.backends.mps.is_available():
        device = "mps"
        print("Apple Silicon GPU acceleration (MPS) detected! Routing Whisper to GPU.")
    else:
        device = "cpu"
        print("MPS not available. Routing Whisper to CPU.")
        
    print(f"Loading Whisper model '{MODEL_NAME}' on device '{device}'...")
    model = whisper.load_model(MODEL_NAME, device=device)
    print("Whisper model loaded successfully.")
    
    print(f"\nStarting transcription of {len(audio_files)} audio files...")
    
    for audio_path in tqdm(audio_files, desc="Transcribing clips", unit="file"):
        try:
            h1_title, md_filename = extract_clean_title(audio_path.name)
            output_md_path = OUTPUT_DIR / md_filename
            
            if output_md_path.exists():
                continue
                
            result = model.transcribe(str(audio_path), fp16=False)
            transcript_text = result["text"].strip()
            
            md_content = f"# {h1_title}\n\n{transcript_text}\n"
            
            with open(output_md_path, "w", encoding="utf-8") as f:
                f.write(md_content)
                
        except Exception as e:
            print(f"\nError transcribing '{audio_path.name}': {e}")
            print("Skipping and continuing to next file...")
            continue
            
    print(f"\nTranscription pipeline finished! Output saved to: {OUTPUT_DIR.resolve()}")

if __name__ == "__main__":
    main()
