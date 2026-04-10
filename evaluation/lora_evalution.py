import requests
import time
import json
import os
import re
from PIL import Image
from io import BytesIO
from pathlib import Path
from datetime import datetime

base_url = 'https://api-inference.modelscope.cn/'

API_KEYS = [
    "",
    "",
    "",
]

current_api_key_index = 0

def get_current_api_key():
    return API_KEYS[current_api_key_index]

def switch_api_key():
    global current_api_key_index
    if current_api_key_index < len(API_KEYS) - 1:
        current_api_key_index += 1
        log_print(f"Switching to backup API key #{current_api_key_index + 1}")
        return True
    return False

def get_headers():
    return {
        "Authorization": f"Bearer {get_current_api_key()}",
        "Content-Type": "application/json",
    }

OUTPUT_DIR = Path("lora_evaluation")
OUTPUT_DIR.mkdir(exist_ok=True)

log_dir = OUTPUT_DIR / "logs"
log_dir.mkdir(exist_ok=True)
log_filename = log_dir / f"evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

progress_file = OUTPUT_DIR / "progress.json"

MODEL = "Tongyi-MAI/Z-Image"

LORA_MODELS = [
    {"name": "zlan-Armenian", "lora_id": "ccArtermices/zlan-Armenian"},
    {"name": "zlan-Georgian2", "lora_id": "ccArtermices/zlan-Georgian2"},
    {"name": "zlan-Hebrew", "lora_id": "ccArtermices/zlan-Hebrew"},
    {"name": "zlan-Myanmar", "lora_id": "ccArtermices/zlan-Myanmar"},
    {"name": "zlan-Persian", "lora_id": "ccArtermices/zlan-Persian"},
    {"name": "zlan-Russian", "lora_id": "ccArtermices/zlan-Russian"},
    {"name": "zlan-Tibetan", "lora_id": "ccArtermices/zlan-Tibetan"},
    {"name": "zlan-Urdu", "lora_id": "ccArtermices/zlan-Urdu"},
    {"name": "zlan-Vietnamese", "lora_id": "ccArtermices/zlan-Vietnamese"},
]

VERIFY_MODELS = [
    "Qwen/Qwen3.5-27B",
    "Qwen/Qwen3.5-35B-A3B",
    "Qwen/Qwen3.5-397B-A17B",
    "Qwen/Qwen3.5-122B-A10B",
]

IMAGES_PER_PROMPT = 3

def log_print(message: str):
    print(message)
    with open(log_filename, "a", encoding="utf-8") as f:
        f.write(message + "\n")

def load_progress():
    if progress_file.exists():
        with open(progress_file, "r", encoding="utf-8") as f:
            return json.load(f)
    return {
        "lora_images": {},
        "baseline_images": [],
        "verification": {},
        "stage": "generation"
    }

def save_progress(progress: dict):
    with open(progress_file, "w", encoding="utf-8") as f:
        json.dump(progress, f, ensure_ascii=False, indent=2)

def load_prompts():
    prompts_file = Path("prompts-multiLan.txt")
    if not prompts_file.exists():
        log_print(f"Error: {prompts_file} not found")
        return []
    
    with open(prompts_file, "r", encoding="utf-8") as f:
        prompts = [line.strip() for line in f if line.strip()]
    return prompts

def generate_image_with_lora(prompt: str, lora_id: str, output_path: str):
    global current_api_key_index
    
    for attempt in range(len(API_KEYS)):
        headers = get_headers()
        
        try:
            payload = {
                "model": MODEL,
                "prompt": prompt,
                "loras": lora_id
            }
            
            response = requests.post(
                f"{base_url}v1/images/generations",
                headers={**headers, "X-ModelScope-Async-Mode": "true"},
                data=json.dumps(payload, ensure_ascii=False).encode('utf-8')
            )
            
            if response.status_code == 429:
                if not switch_api_key():
                    return False, "429"
                continue
            
            if response.status_code != 200:
                return False, f"HTTP {response.status_code}: {response.text[:200]}"
            
            task_id = response.json().get("task_id")
            if not task_id:
                return False, f"No task_id"
            
            max_retries = 60
            retry_count = 0
            
            while retry_count < max_retries:
                result = requests.get(
                    f"{base_url}v1/tasks/{task_id}",
                    headers={**headers, "X-ModelScope-Task-Type": "image_generation"},
                )
                
                if result.status_code == 429:
                    if not switch_api_key():
                        return False, "429"
                    headers = get_headers()
                    continue
                
                result.raise_for_status()
                data = result.json()
                
                status = data.get("task_status", "UNKNOWN")
                
                if status == "SUCCEED":
                    image_url = data["output_images"][0]
                    image = Image.open(BytesIO(requests.get(image_url).content))
                    image.save(output_path)
                    return True, "Success"
                elif status == "FAILED":
                    error_msg = data.get("message", "Unknown error")
                    return False, f"Task failed: {error_msg}"
                
                time.sleep(5)
                retry_count += 1
            
            return False, "Timeout"
            
        except Exception as e:
            return False, f"Error: {e}"
    
    return False, "All API keys exhausted"

def generate_image_baseline(prompt: str, output_path: str):
    global current_api_key_index
    
    for attempt in range(len(API_KEYS)):
        headers = get_headers()
        
        try:
            payload = {
                "model": MODEL,
                "prompt": prompt
            }
            
            response = requests.post(
                f"{base_url}v1/images/generations",
                headers={**headers, "X-ModelScope-Async-Mode": "true"},
                data=json.dumps(payload, ensure_ascii=False).encode('utf-8')
            )
            
            if response.status_code == 429:
                if not switch_api_key():
                    return False, "429"
                continue
            
            if response.status_code != 200:
                return False, f"HTTP {response.status_code}: {response.text[:200]}"
            
            task_id = response.json().get("task_id")
            if not task_id:
                return False, f"No task_id"
            
            max_retries = 60
            retry_count = 0
            
            while retry_count < max_retries:
                result = requests.get(
                    f"{base_url}v1/tasks/{task_id}",
                    headers={**headers, "X-ModelScope-Task-Type": "image_generation"},
                )
                
                if result.status_code == 429:
                    if not switch_api_key():
                        return False, "429"
                    headers = get_headers()
                    continue
                
                result.raise_for_status()
                data = result.json()
                
                status = data.get("task_status", "UNKNOWN")
                
                if status == "SUCCEED":
                    image_url = data["output_images"][0]
                    image = Image.open(BytesIO(requests.get(image_url).content))
                    image.save(output_path)
                    return True, "Success"
                elif status == "FAILED":
                    error_msg = data.get("message", "Unknown error")
                    return False, f"Task failed: {error_msg}"
                
                time.sleep(5)
                retry_count += 1
            
            return False, "Timeout"
            
        except Exception as e:
            return False, f"Error: {e}"
    
    return False, "All API keys exhausted"

def verify_image(image_path: str, expected_text: str, model: str):
    global current_api_key_index
    
    for attempt in range(len(API_KEYS)):
        headers = get_headers()
        
        try:
            prompt = f"""请仔细观察这张图片，判断图片中的文字是否正确显示了以下内容：
"{expected_text}"

请只回答"正确"或"错误"，不要有其他内容。"""

            payload = {
                "model": model,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": f"file://{image_path}"}},
                            {"type": "text", "text": prompt}
                        ]
                    }
                ]
            }
            
            response = requests.post(
                f"{base_url}v1/chat-messages",
                headers=headers,
                data=json.dumps(payload, ensure_ascii=False).encode('utf-8')
            )
            
            if response.status_code == 429:
                if not switch_api_key():
                    return None, "429"
                continue
            
            if response.status_code != 200:
                return None, f"HTTP {response.status_code}"
            
            result = response.json()
            answer = result.get("output", {}).get("text", "").strip()
            
            if "正确" in answer and "错误" not in answer:
                return True, answer
            else:
                return False, answer
            
        except Exception as e:
            return None, f"Error: {e}"
    
    return None, "All API keys exhausted"

def get_expected_text(prompt: str):
    parts = prompt.split(",", 1)
    if len(parts) < 2:
        return None
    content = parts[1]
    text_match = re.search(r'"([^"]+)"', content)
    if text_match:
        return text_match.group(1)
    return None

def run_generation_phase(progress: dict, prompts: list):
    log_print("\n" + "=" * 60)
    log_print("PHASE: Image Generation (LoRA + Baseline)")
    log_print("=" * 60)
    
    all_lora_429 = True
    baseline_429 = False
    
    for lora_info in LORA_MODELS:
        lora_name = lora_info["name"]
        lora_id = lora_info["lora_id"]
        
        if lora_name not in progress["lora_images"]:
            progress["lora_images"][lora_name] = []
        
        lora_dir = OUTPUT_DIR / "lora_images" / lora_name
        lora_dir.mkdir(parents=True, exist_ok=True)
        
        completed = set(progress["lora_images"][lora_name])
        lora_had_success = False
        
        log_print(f"\n[{lora_name}] LoRA ID: {lora_id}")
        
        for prompt_idx, prompt in enumerate(prompts):
            if not prompt.startswith(lora_name + ","):
                continue
            
            for img_idx in range(1, IMAGES_PER_PROMPT + 1):
                task_key = f"{prompt_idx}_{img_idx}"
                
                if task_key in completed:
                    continue
                
                output_filename = f"prompt_{prompt_idx:03d}_img_{img_idx}.jpg"
                output_path = lora_dir / output_filename
                
                log_print(f"  [{prompt_idx}] Generating image {img_idx}/3...")
                
                success, msg = generate_image_with_lora(prompt, lora_id, str(output_path))
                
                if success:
                    log_print(f"    SUCCESS: {output_filename}")
                    completed.add(task_key)
                    progress["lora_images"][lora_name] = list(completed)
                    save_progress(progress)
                    lora_had_success = True
                elif msg == "429":
                    log_print(f"    429 Rate Limited - Skipping to next LoRA/model")
                    progress["lora_images"][lora_name] = list(completed)
                    save_progress(progress)
                    break
                else:
                    log_print(f"    FAILED: {msg}")
                
                time.sleep(3)
            
            else:
                continue
            break
        
        if lora_had_success:
            all_lora_429 = False
    
    log_print("\n" + "-" * 40)
    log_print("Attempting Baseline (Z-Image without LoRA)...")
    log_print("-" * 40)
    
    baseline_dir = OUTPUT_DIR / "baseline_images"
    baseline_dir.mkdir(parents=True, exist_ok=True)
    
    baseline_completed = set(progress["baseline_images"])
    
    for prompt_idx, prompt in enumerate(prompts):
        for img_idx in range(1, IMAGES_PER_PROMPT + 1):
            task_key = f"{prompt_idx}_{img_idx}"
            
            if task_key in baseline_completed:
                continue
            
            output_filename = f"prompt_{prompt_idx:03d}_img_{img_idx}.jpg"
            output_path = baseline_dir / output_filename
            
            log_print(f"  [{prompt_idx}] Generating baseline image {img_idx}/3...")
            
            success, msg = generate_image_baseline(prompt, str(output_path))
            
            if success:
                log_print(f"    SUCCESS: {output_filename}")
                baseline_completed.add(task_key)
                progress["baseline_images"] = list(baseline_completed)
                save_progress(progress)
            elif msg == "429":
                log_print(f"    429 Rate Limited - Stopping baseline generation")
                progress["baseline_images"] = list(baseline_completed)
                save_progress(progress)
                baseline_429 = True
                break
            else:
                log_print(f"    FAILED: {msg}")
            
            time.sleep(3)
        
        else:
            continue
        break
    
    if all_lora_429 and baseline_429:
        log_print("\nAll generation APIs rate limited. Moving to verification phase.")
    
    return True

def run_verification_phase(progress: dict, prompts: list):
    log_print("\n" + "=" * 60)
    log_print("PHASE: Image Verification")
    log_print("=" * 60)
    
    results_file = OUTPUT_DIR / "verification_results.json"
    if results_file.exists():
        with open(results_file, "r", encoding="utf-8") as f:
            results = json.load(f)
    else:
        results = {"lora": {}, "baseline": {}}
    
    if "verification" not in progress:
        progress["verification"] = {}
    
    lora_images_dir = OUTPUT_DIR / "lora_images"
    baseline_images_dir = OUTPUT_DIR / "baseline_images"
    
    all_models_429 = True
    
    for lora_info in LORA_MODELS:
        lora_name = lora_info["name"]
        lora_dir = lora_images_dir / lora_name
        
        if not lora_dir.exists():
            continue
        
        if lora_name not in results["lora"]:
            results["lora"][lora_name] = {}
        
        if lora_name not in progress["verification"]:
            progress["verification"][lora_name] = []
        
        verified = set(progress["verification"][lora_name])
        
        log_print(f"\nVerifying {lora_name}...")
        
        for img_file in sorted(lora_dir.glob("*.jpg")):
            if img_file.name in verified:
                continue
            
            prompt_idx = int(img_file.stem.split("_")[1])
            if prompt_idx >= len(prompts):
                continue
            
            prompt = prompts[prompt_idx]
            expected_text = get_expected_text(prompt)
            
            if not expected_text:
                continue
            
            verified_this_image = False
            
            for model in VERIFY_MODELS:
                log_print(f"  Checking {img_file.name} with {model}...")
                
                result, msg = verify_image(str(img_file.absolute()), expected_text, model)
                
                if result is None and msg == "429":
                    log_print(f"    429 on {model}, trying next model...")
                    time.sleep(3)
                    continue
                
                if result is not None:
                    verified_this_image = True
                    all_models_429 = False
                    
                    is_correct = result
                    results["lora"][lora_name][img_file.name] = {
                        "correct": is_correct,
                        "model": model,
                        "response": msg
                    }
                    
                    status = "CORRECT" if is_correct else "WRONG"
                    log_print(f"    {status}")
                    
                    verified.add(img_file.name)
                    progress["verification"][lora_name] = list(verified)
                    save_progress(progress)
                    
                    with open(results_file, "w", encoding="utf-8") as f:
                        json.dump(results, f, ensure_ascii=False, indent=2)
                    break
            
            if not verified_this_image:
                log_print(f"    All models rate limited for this image")
                return False
            
            time.sleep(3)
    
    if baseline_images_dir.exists():
        log_print(f"\nVerifying baseline...")
        
        if "baseline" not in progress["verification"]:
            progress["verification"]["baseline"] = []
        
        baseline_verified = set(progress["verification"]["baseline"])
        
        for img_file in sorted(baseline_images_dir.glob("*.jpg")):
            if img_file.name in baseline_verified:
                continue
            
            prompt_idx = int(img_file.stem.split("_")[1])
            if prompt_idx >= len(prompts):
                continue
            
            prompt = prompts[prompt_idx]
            expected_text = get_expected_text(prompt)
            
            if not expected_text:
                continue
            
            verified_this_image = False
            
            for model in VERIFY_MODELS:
                log_print(f"  Checking {img_file.name} with {model}...")
                
                result, msg = verify_image(str(img_file.absolute()), expected_text, model)
                
                if result is None and msg == "429":
                    log_print(f"    429 on {model}, trying next model...")
                    time.sleep(3)
                    continue
                
                if result is not None:
                    verified_this_image = True
                    all_models_429 = False
                    
                    is_correct = result
                    results["baseline"][img_file.name] = {
                        "correct": is_correct,
                        "model": model,
                        "response": msg
                    }
                    
                    status = "CORRECT" if is_correct else "WRONG"
                    log_print(f"    {status}")
                    
                    baseline_verified.add(img_file.name)
                    progress["verification"]["baseline"] = list(baseline_verified)
                    save_progress(progress)
                    
                    with open(results_file, "w", encoding="utf-8") as f:
                        json.dump(results, f, ensure_ascii=False, indent=2)
                    break
            
            if not verified_this_image:
                log_print(f"    All models rate limited for this image")
                return False
            
            time.sleep(3)
    
    return not all_models_429

def calculate_statistics():
    log_print("\n" + "=" * 60)
    log_print("PHASE: Statistics Calculation")
    log_print("=" * 60)
    
    results_file = OUTPUT_DIR / "verification_results.json"
    if not results_file.exists():
        log_print("No verification results found")
        return
    
    with open(results_file, "r", encoding="utf-8") as f:
        results = json.load(f)
    
    stats_file = OUTPUT_DIR / "accuracy_statistics.md"
    
    with open(stats_file, "w", encoding="utf-8") as f:
        f.write("# LoRA vs Baseline Accuracy Comparison\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## LoRA Models Accuracy\n\n")
        f.write("| LoRA Model | Correct | Wrong | Total | Accuracy |\n")
        f.write("|------------|---------|-------|-------|----------|\n")
        
        total_lora_correct = 0
        total_lora_wrong = 0
        
        for lora_info in LORA_MODELS:
            name = lora_info["name"]
            if name not in results.get("lora", {}):
                continue
            
            lora_results = results["lora"][name]
            correct = sum(1 for v in lora_results.values() if v.get("correct", False))
            wrong = len(lora_results) - correct
            total = len(lora_results)
            accuracy = correct / total * 100 if total > 0 else 0
            
            total_lora_correct += correct
            total_lora_wrong += wrong
            
            f.write(f"| {name} | {correct} | {wrong} | {total} | {accuracy:.1f}% |\n")
        
        total_lora = total_lora_correct + total_lora_wrong
        lora_accuracy = total_lora_correct / total_lora * 100 if total_lora > 0 else 0
        f.write(f"| **Total** | **{total_lora_correct}** | **{total_lora_wrong}** | **{total_lora}** | **{lora_accuracy:.1f}%** |\n")
        
        f.write("\n## Baseline (No LoRA) Accuracy\n\n")
        
        baseline_results = results.get("baseline", {})
        baseline_correct = sum(1 for v in baseline_results.values() if v.get("correct", False))
        baseline_wrong = len(baseline_results) - baseline_correct
        baseline_total = len(baseline_results)
        baseline_accuracy = baseline_correct / baseline_total * 100 if baseline_total > 0 else 0
        
        f.write(f"| Correct | Wrong | Total | Accuracy |\n")
        f.write(f"|---------|-------|-------|----------|\n")
        f.write(f"| {baseline_correct} | {baseline_wrong} | {baseline_total} | {baseline_accuracy:.1f}% |\n")
        
        f.write("\n## Comparison Summary\n\n")
        f.write(f"| Model Type | Total Images | Correct | Accuracy |\n")
        f.write(f"|------------|--------------|---------|----------|\n")
        f.write(f"| LoRA (avg) | {total_lora} | {total_lora_correct} | {lora_accuracy:.1f}% |\n")
        f.write(f"| Baseline | {baseline_total} | {baseline_correct} | {baseline_accuracy:.1f}% |\n")
        
        improvement = lora_accuracy - baseline_accuracy
        f.write(f"\n**Improvement: {improvement:+.1f}%**\n")
    
    log_print(f"Statistics saved to: {stats_file}")
    
    with open(stats_file, "r", encoding="utf-8") as f:
        print(f.read())

def main():
    log_print("=" * 60)
    log_print("LoRA Evaluation Pipeline")
    log_print(f"Output directory: {OUTPUT_DIR}")
    log_print(f"Log file: {log_filename}")
    log_print("=" * 60)
    
    progress = load_progress()
    prompts = load_prompts()
    
    log_print(f"Total prompts: {len(prompts)}")
    
    run_generation_phase(progress, prompts)
    
    verification_success = run_verification_phase(progress, prompts)
    
    if verification_success:
        calculate_statistics()
        
        log_print("\n" + "=" * 60)
        log_print("ALL TASKS COMPLETED!")
        log_print("=" * 60)
    else:
        log_print("\n" + "=" * 60)
        log_print("Verification stopped due to rate limits.")
        log_print("Run again to continue.")
        log_print("=" * 60)

if __name__ == "__main__":
    main()
