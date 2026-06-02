import os
import re
import sys
import subprocess
from pathlib import Path

os.environ["TORCH_DEVICE"] = "mps"

BOOK_PAGES = {
    "society_of_mind.pdf": "3-455",
    "emotion_machine.pdf": "12-355",
    "perceptrons.pdf": "8-298",
    "framework_for_representing_knowledge.pdf": "0-104",
    "logical_vs_analogical_ai.pdf": "0-17",
    "will_robots_inherit_the_earth.pdf": "0-6"
}

INPUT_DIR = Path("./data/books")
OUTPUT_DIR = Path("./data/books_markdown")

def clean_visual_artifacts(text: str) -> str:
    spaced_repeats_pattern = re.compile(r"([a-zA-Z|_\-\*\.·•])(\s+\1){2,}")
    continuous_repeats_pattern = re.compile(r"([|_\-\*\=\#\.·•]){4,}")
    
    cleaned_text = spaced_repeats_pattern.sub(" ", text)
    cleaned_text = continuous_repeats_pattern.sub(" ", cleaned_text)
    return cleaned_text

def process_pdf(pdf_path: Path):
    base_name = pdf_path.stem
    expected_md_path = OUTPUT_DIR / base_name / f"{base_name}.md"
    
    if expected_md_path.exists():
        print(f"Skipping {pdf_path.name} as it has already been extracted to: {expected_md_path}")
        return

    print(f"\n{'='*60}\nProcessing: {pdf_path.name}\n{'='*60}")
    
    page_range = BOOK_PAGES.get(pdf_path.name)
    
    script_dir = Path(__file__).resolve().parent
    venv_marker = script_dir / ".venv" / "bin" / "marker_single"
    if not venv_marker.exists():
        venv_marker = script_dir.parent / ".venv" / "bin" / "marker_single"
    marker_bin = str(venv_marker) if venv_marker.exists() else "marker_single"
    
    cmd = [
        marker_bin,
        str(pdf_path),
        "--output_dir",
        str(OUTPUT_DIR),
        "--disable_image_extraction"
    ]
    
    if page_range:
        print(f"Configuring verified page range: {page_range}")
        cmd.extend(["--page_range", page_range])
    else:
        print("No specific page range config found. Processing entire PDF.")
        
    print(f"Running CLI command: {' '.join(cmd)}")
    
    try:
        subprocess.run(cmd, check=True)
        print("marker_single command execution completed successfully.")
        
        base_name = pdf_path.stem
        expected_md_path = OUTPUT_DIR / base_name / f"{base_name}.md"
        
        md_file_path = None
        if expected_md_path.exists():
            md_file_path = expected_md_path
        else:
            search_path = OUTPUT_DIR / base_name
            if search_path.exists():
                md_files = list(search_path.glob("*.md"))
                if md_files:
                    md_file_path = md_files[0]
        
        if md_file_path and md_file_path.exists():
            print(f"Polishing Markdown file: {md_file_path}")
            
            with open(md_file_path, "r", encoding="utf-8") as f:
                content = f.read()
                
            polished_content = clean_visual_artifacts(content)
            
            with open(md_file_path, "w", encoding="utf-8") as f:
                f.write(polished_content)
                
            print(f"Successfully cleaned and polished: {md_file_path.name}")
        else:
            print(f"Warning: Could not locate the generated markdown file for {pdf_path.name}")
            
    except subprocess.CalledProcessError as e:
        print(f"Error: marker_single failed with exit code {e.returncode} for file {pdf_path.name}")
    except Exception as e:
        print(f"Error occurred during post-processing for {pdf_path.name}: {e}")

def main():
    in_venv = sys.prefix != sys.base_prefix or "VIRTUAL_ENV" in os.environ
    if not in_venv:
        print("Warning: You are running this script using a system Python interpreter rather than the local virtual environment (.venv).")
        print("Recommendation: Run with './.venv/bin/python pdf_text_extractor.py' or activate the virtual environment first via 'source .venv/bin/activate'.\n")

    if not INPUT_DIR.exists():
        print(f"Creating input directory: {INPUT_DIR}")
        INPUT_DIR.mkdir(parents=True, exist_ok=True)
        print(f"Please place your Marvin Minsky PDF books in: {INPUT_DIR.resolve()}")
        sys.exit(0)
        
    if not OUTPUT_DIR.exists():
        print(f"Creating output directory: {OUTPUT_DIR}")
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        
    pdf_files = sorted(list(INPUT_DIR.glob("*.pdf")))
    
    primary_pdfs = [f for f in pdf_files if f.name in BOOK_PAGES]
    other_pdfs = [f for f in pdf_files if f.name not in BOOK_PAGES]
    all_to_process = primary_pdfs + other_pdfs
    
    if not all_to_process:
        print(f"No PDF files found in {INPUT_DIR.resolve()}")
        print("Please ensure your PDF files are placed in that directory and named correctly.")
        print("Expected primary names:")
        for name in BOOK_PAGES.keys():
            print(f" - {name}")
        sys.exit(1)
        
    print(f"Found {len(all_to_process)} book(s) to process.")
    for pdf in all_to_process:
        process_pdf(pdf)
        
    print("\nExtraction and post-processing run finished!")

if __name__ == "__main__":
    main()
