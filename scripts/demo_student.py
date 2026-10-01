import argparse
import json
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from scripts.build_student_data import SYSTEM_PROMPT

def main():
    parser = argparse.ArgumentParser(description="Interactive demo of the Student Model")
    parser.add_argument("--checkpoint", required=True, help="Path to student checkpoint (adapter dir)")
    args = parser.parse_args()

    meta_path = os.path.join(args.checkpoint, "run_metadata.json")
    adapter_dir = os.path.join(args.checkpoint, "adapter")
    
    if not os.path.exists(meta_path) or not os.path.isdir(adapter_dir):
        print(f"Error: Checkpoint {args.checkpoint} missing run_metadata.json or adapter/")
        return

    with open(meta_path, "r", encoding="utf-8") as f:
        run_meta = json.load(f)

    print("Loading model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(adapter_dir)
    model = AutoModelForCausalLM.from_pretrained(run_meta["base_model"], dtype=dtype)
    model = PeftModel.from_pretrained(model, adapter_dir)
    model = model.merge_and_unload()
    model.to(device)
    model.eval()

    print("\n--- Student Model Demo (type 'quit' to exit) ---")
    while True:
        text = input("\nEnter Banglish/Romanized prompt: ").strip()
        if text.lower() in ["quit", "exit"]:
            break
        if not text:
            continue

        prompt = tokenizer.apply_chat_template([
            {"role": "system", "content": run_meta.get("system_prompt", SYSTEM_PROMPT)},
            {"role": "user", "content": text}
        ], add_generation_prompt=True, tokenize=False)
        
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        print("\nSanitizing...")
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=512, do_sample=False)
            
        generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
        response = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        
        try:
            parsed = json.loads(response)
            print("\n[Sanitized Prompt]:", parsed.get("sanitized_prompt"))
            print("[PII Detected]:", json.dumps(parsed.get("pii", []), indent=2))
            print("[Uncertainty Spans]:", json.dumps(parsed.get("uncertainties", []), indent=2))
            print("[Routing Decision]:", parsed.get("routing"))
        except json.JSONDecodeError:
            print("\n[Raw Output (Failed to parse JSON)]:\n", response)

if __name__ == "__main__":
    main()
