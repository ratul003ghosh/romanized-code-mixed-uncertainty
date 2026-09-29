import argparse
import json
import logging
import os
import sys

from huggingface_hub import HfFileSystem, hf_hub_download

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger("prepare_data")

def main():
    parser = argparse.ArgumentParser(description="Download and prepare teacher inputs from Hugging Face.")
    parser.add_argument("--dataset", required=True, help="HF dataset ID (e.g., mlpaper/teacher-inputs)")
    parser.add_argument("--output", required=True, help="Output file path for teacher inputs")
    parser.add_argument("--splits", nargs="+", default=["train"], help="Split keywords to include (e.g., train, dev, test)")
    args = parser.parse_args()

    # Authenticate and list files
    try:
        fs = HfFileSystem()
        all_files = fs.ls(f"datasets/{args.dataset}")
    except Exception as e:
        log.error(f"Failed to access dataset '{args.dataset}'. Did you run 'hf auth login'?")
        log.error(f"Error: {e}")
        sys.exit(1)

    # Filter JSONL files
    jsonl_files = [f["name"] for f in all_files if f["name"].endswith(".jsonl")]
    
    if not jsonl_files:
        log.error("No JSONL files found in the dataset.")
        sys.exit(1)

    # Filter based on requested splits
    selected_files = []
    for f in jsonl_files:
        filename = f.split("/")[-1]
        # Match splits in filename, e.g., "teacher_inputs_train.jsonl"
        # Since we want to include dialect_train, hinglish_train, etc. if 'train' is requested
        if any(split in filename for split in args.splits):
            selected_files.append(filename)

    if not selected_files:
        log.error(f"No files matching splits {args.splits} found.")
        sys.exit(1)

    log.info(f"Found {len(selected_files)} files matching splits {args.splits}:")
    for f in selected_files:
        log.info(f" - {f}")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    
    if os.path.exists(args.output):
        log.error(f"Output file {args.output} already exists. Please remove it first.")
        sys.exit(1)

    # Download and combine
    total_records = 0
    malformed = 0
    seen_ids = set()

    with open(args.output, "w", encoding="utf-8") as out_f:
        for filename in selected_files:
            log.info(f"Downloading {filename}...")
            try:
                local_path = hf_hub_download(
                    repo_id=args.dataset,
                    repo_type="dataset",
                    filename=filename
                )
            except Exception as e:
                log.error(f"Failed to download {filename}: {e}")
                sys.exit(1)

            # Validate and append
            with open(local_path, "r", encoding="utf-8") as in_f:
                for i, line in enumerate(in_f):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        malformed += 1
                        continue
                    
                    # Minimum schema validation
                    if "id" not in record or ("clean_input" not in record and "text" not in record and "input" not in record):
                        malformed += 1
                        continue
                    
                    rec_id = record["id"]
                    if rec_id in seen_ids:
                        log.warning(f"Duplicate ID found: {rec_id} in {filename}")
                        continue
                        
                    seen_ids.add(rec_id)
                    out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    total_records += 1

    log.info(f"Preparation complete!")
    log.info(f"Total valid records written: {total_records}")
    log.info(f"Malformed records skipped: {malformed}")
    log.info(f"Output saved to: {args.output}")

if __name__ == "__main__":
    main()
