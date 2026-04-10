import json
import os
import re
import base64
import time
from pathlib import Path
from datetime import datetime
from openai import OpenAI

MODEL = "Qwen/Qwen3.5-35B-A3B"
BASE_URL = "http://localhost:8000/v1"

OUTPUT_DIR = Path("lora_evaluation")
PROMPTS_FILE = Path("prompts-multiLan.txt")
PROGRESS_FILE = OUTPUT_DIR / "verification_progress.json"
RESULTS_FILE = OUTPUT_DIR / "verification_results.json"
LOG_FILE = OUTPUT_DIR / "logs" / f"verification_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

OUTPUT_DIR.mkdir(exist_ok=True)
(OUTPUT_DIR / "logs").mkdir(exist_ok=True)

client = OpenAI(
    api_key="EMPTY",
    base_url=BASE_URL
)

def log_print(message: str):
    print(message)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(message + "\n")

def load_prompts():
    with open(PROMPTS_FILE, "r", encoding="utf-8") as f:
        prompts = [line.strip() for line in f if line.strip()]
    return prompts

def get_expected_text(prompt: str):
    parts = prompt.split(",", 1)
    if len(parts) < 2:
        return None
    content = parts[1]
    text_match = re.search(r'"([^"]+)"', content)
    if text_match:
        return text_match.group(1)
    return None

def image_to_base64_url(image_path: str):
    with open(image_path, "rb") as f:
        image_data = base64.b64encode(f.read()).decode("utf-8")
    return f"data:image/jpeg;base64,{image_data}"

def verify_image(image_path: str, expected_text: str):
    prompt = f"""请仔细观察这张图片，判断图片中的文字是否正确显示了以下内容：
"{expected_text}"

请按以下格式回答：
1. 是否正确显示：是/否
2. 图片中实际显示的文字（如果能识别）：xxx
3. 简要说明：xxx"""

    try:
        image_url = image_to_base64_url(image_path)
        
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": image_url}
                    },
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            }
        ]
        
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            max_tokens=1024,
            temperature=0.1,
            top_p=0.8,
            extra_body={
                "top_k": 20,
                "chat_template_kwargs": {"enable_thinking": False},
            },
        )
        
        answer = response.choices[0].message.content.strip()
        is_correct = "是否正确显示：是" in answer
        
        return is_correct, answer
        
    except Exception as e:
        return None, f"Error: {e}"

def load_progress():
    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {
        "verified_images": {},
        "current_stage": "lora"
    }

def save_progress(progress: dict):
    with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
        json.dump(progress, f, ensure_ascii=False, indent=2)

def load_results():
    if RESULTS_FILE.exists():
        with open(RESULTS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {
        "lora_results": {},
        "baseline_results": {},
        "summary": {}
    }

def save_results(results: dict):
    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

def main():
    log_print("=" * 60)
    log_print("Image Verification Script (vLLM Local)")
    log_print(f"Model: {MODEL}")
    log_print(f"Base URL: {BASE_URL}")
    log_print(f"Progress file: {PROGRESS_FILE}")
    log_print(f"Results file: {RESULTS_FILE}")
    log_print("=" * 60)
    
    prompts = load_prompts()
    log_print(f"Total prompts: {len(prompts)}")
    
    progress = load_progress()
    results = load_results()
    
    lora_models = [
        "zlan-Armenian",
        "zlan-Georgian2", 
        "zlan-Hebrew",
        "zlan-Myanmar",
        "zlan-Persian",
        "zlan-Russian",
        "zlan-Tibetan",
        "zlan-Urdu",
        "zlan-Vietnamese",
    ]
    
    log_print("\n" + "=" * 60)
    log_print("PHASE 1: Verifying LoRA Images")
    log_print("=" * 60)
    
    for lora_name in lora_models:
        lora_dir = OUTPUT_DIR / "lora_images" / lora_name
        
        if not lora_dir.exists():
            log_print(f"\n[{lora_name}] Directory not found, skipping...")
            continue
        
        log_print(f"\n[{lora_name}] Verifying images...")
        
        if lora_name not in results["lora_results"]:
            results["lora_results"][lora_name] = {
                "total": 0,
                "correct": 0,
                "incorrect": 0,
                "error": 0,
                "details": []
            }
        
        image_files = sorted(lora_dir.glob("*.jpg"))
        
        for image_file in image_files:
            image_key = f"{lora_name}/{image_file.name}"
            
            if image_key in progress["verified_images"]:
                continue
            
            filename = image_file.name
            match = re.match(r"prompt_(\d+)_img_(\d+)\.jpg", filename)
            if not match:
                continue
            
            prompt_idx = int(match.group(1))
            img_idx = int(match.group(2))
            
            if prompt_idx >= len(prompts):
                continue
            
            prompt = prompts[prompt_idx]
            expected_text = get_expected_text(prompt)
            
            if not expected_text:
                log_print(f"  {filename}: No expected text found")
                continue
            
            log_print(f"  Verifying {filename}...")
            
            is_correct, response = verify_image(str(image_file), expected_text)
            
            if is_correct is None:
                log_print(f"    ERROR: {response}")
                results["lora_results"][lora_name]["error"] += 1
                results["lora_results"][lora_name]["total"] += 1
            else:
                status = "CORRECT" if is_correct else "INCORRECT"
                log_print(f"    {status}")
                results["lora_results"][lora_name]["total"] += 1
                if is_correct:
                    results["lora_results"][lora_name]["correct"] += 1
                else:
                    results["lora_results"][lora_name]["incorrect"] += 1
                
                results["lora_results"][lora_name]["details"].append({
                    "image": filename,
                    "prompt_idx": prompt_idx,
                    "expected_text": expected_text,
                    "is_correct": is_correct,
                    "response": response[:500] if response else None
                })
            
            progress["verified_images"][image_key] = {
                "is_correct": is_correct,
                "timestamp": datetime.now().isoformat()
            }
            save_progress(progress)
            save_results(results)
    
    log_print("\n" + "=" * 60)
    log_print("PHASE 2: Verifying Baseline Images")
    log_print("=" * 60)
    
    baseline_dir = OUTPUT_DIR / "baseline_images"
    
    if baseline_dir.exists():
        image_files = sorted(baseline_dir.glob("*.jpg"))
        
        for image_file in image_files:
            image_key = f"baseline/{image_file.name}"
            
            if image_key in progress["verified_images"]:
                continue
            
            filename = image_file.name
            match = re.match(r"prompt_(\d+)_img_(\d+)\.jpg", filename)
            if not match:
                continue
            
            prompt_idx = int(match.group(1))
            img_idx = int(match.group(2))
            
            if prompt_idx >= len(prompts):
                continue
            
            prompt = prompts[prompt_idx]
            expected_text = get_expected_text(prompt)
            
            if not expected_text:
                log_print(f"  {filename}: No expected text found")
                continue
            
            log_print(f"  Verifying {filename}...")
            
            is_correct, response = verify_image(str(image_file), expected_text)
            
            if is_correct is None:
                log_print(f"    ERROR: {response}")
                results["baseline_results"]["error"] = results["baseline_results"].get("error", 0) + 1
                results["baseline_results"]["total"] = results["baseline_results"].get("total", 0) + 1
            else:
                status = "CORRECT" if is_correct else "INCORRECT"
                log_print(f"    {status}")
                results["baseline_results"]["total"] = results["baseline_results"].get("total", 0) + 1
                if is_correct:
                    results["baseline_results"]["correct"] = results["baseline_results"].get("correct", 0) + 1
                else:
                    results["baseline_results"]["incorrect"] = results["baseline_results"].get("incorrect", 0) + 1
                
                if "details" not in results["baseline_results"]:
                    results["baseline_results"]["details"] = []
                    
                results["baseline_results"]["details"].append({
                    "image": filename,
                    "prompt_idx": prompt_idx,
                    "expected_text": expected_text,
                    "is_correct": is_correct,
                    "response": response[:500] if response else None
                })
            
            progress["verified_images"][image_key] = {
                "is_correct": is_correct,
                "timestamp": datetime.now().isoformat()
            }
            save_progress(progress)
            save_results(results)
    
    log_print("\n" + "=" * 60)
    log_print("Verification Complete!")
    log_print("=" * 60)
    
    log_print("\n--- LoRA Results Summary ---")
    for lora_name, data in results["lora_results"].items():
        if data["total"] > 0:
            accuracy = data["correct"] / data["total"] * 100
            log_print(f"{lora_name}: {data['correct']}/{data['total']} correct ({accuracy:.1f}%)")
    
    log_print("\n--- Baseline Results Summary ---")
    if results["baseline_results"].get("total", 0) > 0:
        accuracy = results["baseline_results"]["correct"] / results["baseline_results"]["total"] * 100
        log_print(f"Baseline: {results['baseline_results']['correct']}/{results['baseline_results']['total']} correct ({accuracy:.1f}%)")
    
    results["summary"]["completed_at"] = datetime.now().isoformat()
    save_results(results)

if __name__ == "__main__":
    main()
