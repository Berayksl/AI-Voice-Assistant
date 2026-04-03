import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # fix: multiple OpenMP runtimes on Windows
import time
from faster_whisper import WhisperModel

# ── Config ──────────────────────────────────────────────────────────────────
MODEL_SIZE = "medium"    
DEVICE     = "cuda"
COMPUTE_TYPE = "float16"  
AUDIO_FILE = "test audio/test01_20s.wav"
# ────────────────────────────────────────────────────────────────────────────


def load_model(model_size: str = MODEL_SIZE):
    print(f"[ASR] Loading Whisper {model_size} on {DEVICE} ({COMPUTE_TYPE})...")
    t0 = time.time()
    model = WhisperModel(model_size, device=DEVICE, compute_type=COMPUTE_TYPE)
    print(f"[ASR] Model loaded in {time.time() - t0:.2f}s")
    return model


def transcribe(model: WhisperModel, audio_path: str, language: str = None):
    """
    Transcribe audio file. Returns transcript text + timing stats.
    language: None = auto-detect, or e.g. "en", "zh", "fr"
    """
    print(f"\n[ASR] Transcribing: {audio_path}")
    t0 = time.time()

    segments, info = model.transcribe(
        audio_path,
        language=language,
        beam_size=5,
        vad_filter=True,            # voice activity detection — skip silence
        vad_parameters=dict(min_silence_duration_ms=500),
    )

    # Collect segments (generator — must iterate to actually run inference)
    results = []
    for seg in segments:
        results.append(seg.text.strip())
        print(f"  [{seg.start:.1f}s → {seg.end:.1f}s] {seg.text.strip()}")

    elapsed = time.time() - t0
    full_text = " ".join(results)

    # Real-Time Factor: elapsed / audio_duration (lower is better; <1 = faster than real-time)
    audio_duration = info.duration
    rtf = elapsed / audio_duration if audio_duration > 0 else 0

    print(f"\n[ASR] ── Results ──────────────────────")
    print(f"  Detected language : {info.language} (prob={info.language_probability:.2f})")
    print(f"  Audio duration    : {audio_duration:.2f}s")
    print(f"  Inference time    : {elapsed:.2f}s")
    print(f"  Real-Time Factor  : {rtf:.3f}  ({'✅ faster than real-time' if rtf < 1 else '⚠️ slower than real-time'})")
    print(f"  Transcript        : {full_text}")
    print(f"────────────────────────────────────────\n")

    return full_text, {"rtf": rtf, "inference_time": elapsed, "audio_duration": audio_duration}


def benchmark_models(audio_path: str):
    """Compare small vs medium on same audio."""
    results = {}
    for size in ["small", "medium"]:
        model = load_model(size)
        text, stats = transcribe(model, audio_path)
        results[size] = {"transcript": text, **stats}
        del model  # free VRAM between runs
        import torch; torch.cuda.empty_cache()
    
    print("\n[ASR] ── Benchmark Comparison ──────────")
    for size, r in results.items():
        print(f"  {size:8s} | RTF={r['rtf']:.3f} | time={r['inference_time']:.2f}s")
    print("────────────────────────────────────────")
    return results


if __name__ == "__main__":
    if not os.path.exists(AUDIO_FILE):
        # Generate a quick test WAV using TTS or record one manually.
        # For now, create a short silent placeholder so the script doesn't crash.
        print(f"[ASR] Warning: '{AUDIO_FILE}' not found.")
        print("  → Record a short audio clip and save it as test_audio.wav")
        print("  → Or run: python -c \"import sounddevice as sd; import soundfile as sf; "
              "sf.write('test_audio.wav', sd.rec(int(5*16000), samplerate=16000, channels=1, dtype='int16', blocking=True), 16000)\"")
    else:
        model = load_model()
        transcribe(model, AUDIO_FILE)

        benchmark_models(AUDIO_FILE) #run the benchmark to compare small vs medium models on the same audio file