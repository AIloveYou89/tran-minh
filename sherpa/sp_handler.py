# RunPod Serverless: Vietnamese ASR (Sherpa-ONNX, Zipformer-RNNT)
import os, time, uuid, base64, logging, subprocess, json, re
import runpod
from huggingface_hub import snapshot_download

# ------------------------------
# CONFIG
# ------------------------------
MODEL_ID  = os.getenv("MODEL_ID", "hynt/Zipformer-30M-RNNT-6000h")
MODEL_DIR = os.getenv("MODEL_DIR", "/models/Zipformer-30M-RNNT-6000h")
OUT_DIR   = os.getenv("OUT_DIR", "/runpod-volume/jobs")
NUM_THREADS = os.getenv("NUM_THREADS", "1")
HF_TOKEN = os.getenv("HF_TOKEN")

os.makedirs(OUT_DIR, exist_ok=True)

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("SherpaASR")

# ------------------------------
# MODEL DOWNLOAD
# ------------------------------
def ensure_model():
    if not os.path.exists(MODEL_DIR) or not os.listdir(MODEL_DIR):
        log.info(f"[MODEL] Downloading {MODEL_ID} → {MODEL_DIR}")
        snapshot_download(
            repo_id=MODEL_ID,
            local_dir=MODEL_DIR,
            local_dir_use_symlinks=False,
            token=HF_TOKEN
        )
    else:
        log.info(f"[MODEL] Found model at {MODEL_DIR}")

def find_model_files(model_dir: str):
    """Tìm các file model cần thiết"""
    tokens_candidates = ["tokens.txt", "bpe.model", "config.json"]
    tokens_path = None
    for t in tokens_candidates:
        p = os.path.join(model_dir, t)
        if os.path.exists(p):
            tokens_path = p
            break
    
    if not tokens_path:
        raise FileNotFoundError("Cannot find tokens file in model dir")
    
    encoder = os.path.join(model_dir, "encoder-epoch-20-avg-10.int8.onnx")
    if not os.path.exists(encoder):
        encoder = os.path.join(model_dir, "encoder-epoch-20-avg-10.onnx")
    
    decoder = os.path.join(model_dir, "decoder-epoch-20-avg-10.int8.onnx")
    if not os.path.exists(decoder):
        decoder = os.path.join(model_dir, "decoder-epoch-20-avg-10.onnx")
    
    joiner = os.path.join(model_dir, "joiner-epoch-20-avg-10.int8.onnx")
    if not os.path.exists(joiner):
        joiner = os.path.join(model_dir, "joiner-epoch-20-avg-10.onnx")
    
    return tokens_path, encoder, decoder, joiner

ensure_model()

# ------------------------------
# PARSE OUTPUT
# ------------------------------
def extract_result_from_output(stdout: str, stderr: str) -> dict:
    """
    Parse full JSON result từ sherpa-onnx output.
    Returns: {"text": "...", "timestamps": [...], "tokens": [...]}
    """
    combined = stderr + "\n" + stdout
    
    # Method 1: Parse complete JSON object
    for line in combined.split('\n'):
        line = line.strip()
        if line.startswith('{') and '"text"' in line and '"timestamps"' in line:
            try:
                obj = json.loads(line)
                if 'text' in obj and obj['text'].strip():
                    return {
                        "text": obj['text'].strip(),
                        "timestamps": obj.get('timestamps', []),
                        "tokens": obj.get('tokens', []),
                        "words": obj.get('words', [])
                    }
            except json.JSONDecodeError:
                continue
    
    # Method 2: Regex fallback - tìm text only
    json_pattern = r'\{[^{}]*"text"\s*:\s*"([^"]*)"[^{}]*\}'
    matches = re.findall(json_pattern, combined)
    if matches:
        return {
            "text": matches[-1].strip(),
            "timestamps": [],
            "tokens": [],
            "words": []
        }
    
    return {"text": "", "timestamps": [], "tokens": [], "words": []}

def create_word_segments(text: str, timestamps: list, tokens: list) -> list:
    """
    Tạo word-level segments từ token-level timestamps.
    Returns: [{"word": "TÔI", "start": 0.00, "end": 0.12}, ...]
    """
    if not timestamps or not tokens or len(timestamps) != len(tokens):
        return []
    
    segments = []
    current_word = ""
    word_start = None
    
    for i, (token, ts) in enumerate(zip(tokens, timestamps)):
        token = token.strip()
        
        # Token bắt đầu bằng space = từ mới
        if token.startswith(' ') or i == 0:
            # Lưu từ cũ nếu có
            if current_word and word_start is not None:
                word_end = timestamps[i-1] if i > 0 else ts
                segments.append({
                    "word": current_word.strip(),
                    "start": round(word_start, 2),
                    "end": round(word_end, 2)
                })
            
            # Bắt đầu từ mới
            current_word = token
            word_start = ts
        else:
            # Tiếp tục từ hiện tại
            current_word += token
    
    # Lưu từ cuối cùng
    if current_word and word_start is not None:
        segments.append({
            "word": current_word.strip(),
            "start": round(word_start, 2),
            "end": round(timestamps[-1], 2)
        })
    
    return segments

# ------------------------------
# CORE HANDLER
# ------------------------------
def handler(job):
    """
    Input JSON:
    {
      "audio_path": "/runpod-volume/audio/test.wav",
      "return": "text" | "json" | "base64",
      "include_timestamps": true/false (default: true),
      "outfile": "optional_output.txt"
    }
    """
    inp = job.get("input", {})
    audio_path = inp.get("audio_path")
    
    if not audio_path or not os.path.exists(audio_path):
        return {"error": f"Audio not found: {audio_path}"}

    job_id = str(uuid.uuid4())
    out_name = inp.get("outfile") or f"{job_id}.json"
    out_path = os.path.join(OUT_DIR, out_name)

    # Chuẩn hóa audio
    fixed_wav = f"/tmp/{job_id}_16k.wav"
    log.info(f"[AUDIO] Converting {audio_path} to 16kHz mono")
    
    ffmpeg_result = subprocess.run(
        ["ffmpeg", "-y", "-i", audio_path, "-ac", "1", "-ar", "16000", "-sample_fmt", "s16", fixed_wav],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    
    if ffmpeg_result.returncode != 0:
        log.error(f"[AUDIO] FFmpeg error: {ffmpeg_result.stderr}")
        return {"error": "Audio conversion failed", "details": ffmpeg_result.stderr}

    # Tìm model files
    try:
        tokens_path, encoder, decoder, joiner = find_model_files(MODEL_DIR)
        log.info(f"[MODEL] Using tokens: {os.path.basename(tokens_path)}")
    except Exception as e:
        return {"error": f"Model files not found: {str(e)}"}

    # Build command
    cmd = [
        "sherpa-onnx-offline",
        f"--tokens={tokens_path}",
        f"--encoder={encoder}",
        f"--decoder={decoder}",
        f"--joiner={joiner}",
        f"--num-threads={NUM_THREADS}",
        fixed_wav
    ]

    log.info(f"[JOB] Decoding {audio_path} (job {job_id})")
    
    t0 = time.time()
    result = subprocess.run(cmd, text=True, capture_output=True)
    elapsed = time.time() - t0

    if result.returncode != 0:
        log.error(f"[ERROR] Sherpa-ONNX failed with returncode {result.returncode}")
        return {
            "error": "Sherpa-ONNX failed",
            "returncode": result.returncode,
            "stderr": result.stderr[-500:]
        }

    # Parse full result
    asr_result = extract_result_from_output(result.stdout, result.stderr)
    transcript = asr_result["text"]
    
    if not transcript:
        log.warning("[TRANSCRIPT] Empty result!")
        transcript = "(No transcript detected)"
        asr_result["text"] = transcript
    else:
        log.info(f"[TRANSCRIPT] {transcript[:100]}...")
        log.info(f"[TOKENS] {len(asr_result['tokens'])} tokens, {len(asr_result['timestamps'])} timestamps")

    # Tạo word-level segments
    include_timestamps = inp.get("include_timestamps", True)
    if include_timestamps and asr_result["timestamps"]:
        word_segments = create_word_segments(
            transcript, 
            asr_result["timestamps"], 
            asr_result["tokens"]
        )
        asr_result["word_segments"] = word_segments
        log.info(f"[SEGMENTS] Created {len(word_segments)} word segments")

    # Save to file
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(asr_result, f, ensure_ascii=False, indent=2)

    log.info(f"[DONE] {audio_path} → {out_path} ({elapsed:.2f}s)")

    # Clean up
    try:
        os.remove(fixed_wav)
    except:
        pass

    # Return based on format
    return_format = inp.get("return", "json")
    
    if return_format == "text":
        # Chỉ trả text thuần
        return {
            "job_id": job_id,
            "elapsed_sec": round(elapsed, 2),
            "text": transcript,
            "path": out_path
        }
    
    elif return_format == "base64":
        # Text as base64
        b64 = base64.b64encode(transcript.encode("utf-8")).decode("utf-8")
        return {
            "job_id": job_id,
            "elapsed_sec": round(elapsed, 2),
            "text_b64": b64,
            "path": out_path
        }
    
    else:  # default: json
        # Full result with timestamps
        return {
            "job_id": job_id,
            "elapsed_sec": round(elapsed, 2),
            "path": out_path,
            "text": transcript,
            "timestamps": asr_result.get("timestamps", []),
            "tokens": asr_result.get("tokens", []),
            "word_segments": asr_result.get("word_segments", []),
            "num_tokens": len(asr_result.get("tokens", [])),
            "num_words": len(asr_result.get("word_segments", []))
        }

# ------------------------------
# START WORKER
# ------------------------------
runpod.serverless.start({"handler": handler})
