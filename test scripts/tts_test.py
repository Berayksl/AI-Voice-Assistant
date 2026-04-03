import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import time
import torch
import soundfile as sf
import sounddevice as sd
import numpy as np
from chatterbox.tts_turbo import ChatterboxTurboTTS

DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_DIR = "./tts_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Reference audio for zero-shot cloning (3–15s of clean speech)
REF_AUDIO  = r"./clone audio/elon musk sample.wav"

TEST_TEXT = (
    "Deep learning has revolutionized the field of artificial intelligence. "
    "Modern neural networks can now recognize speech, translate languages, "
    "and generate natural-sounding audio with remarkable accuracy."
)



def load_model():
    print(f"[TTS] Loading Chatterbox Turbo on {DEVICE}...")
    t0 = time.time()
    model = ChatterboxTurboTTS.from_pretrained(device=DEVICE)
    print(f"[TTS] Model loaded in {time.time()-t0:.2f}s\n")
    return model


def play_audio(wav: torch.Tensor, sr: int):
    """Play a torch audio tensor through speakers."""
    audio_np = wav.squeeze().cpu().numpy()
    sd.play(audio_np, sr)
    sd.wait()


def save_and_stats(wav: torch.Tensor, sr: int, filename: str, label: str, elapsed: float):
    """Save audio using soundfile (avoids TorchCodec issues on Windows) and print stats."""
    out_path = os.path.join(OUTPUT_DIR, filename)
    audio_np = wav.squeeze().cpu().numpy()
    sf.write(out_path, audio_np, sr)

    audio_duration = len(audio_np) / sr
    rtf = elapsed / audio_duration if audio_duration > 0 else 0

    print(f"[TTS] ── {label} ─────────────────────────")
    print(f"  Audio duration : {audio_duration:.2f}s")
    print(f"  Inference time : {elapsed:.2f}s")
    print(f"  RTF            : {rtf:.3f}  ({'✅ real-time' if rtf < 1 else '⚠️ slower than real-time'})")
    print(f"  Saved to       : {out_path}")
    print(f"────────────────────────────────────────────\n")

    return {"rtf": rtf, "inference_time": elapsed, "audio_duration": audio_duration}


def synthesize_default(model: ChatterboxTurboTTS, text: str, output_filename: str = "default_voice.wav"):
    """
    Synthesize using Chatterbox's default built-in voice.
    No reference audio needed.
    exaggeration: 0.0 (flat) → 1.0 (very expressive). Default 0.5.
    cfg_weight  : controls pacing. Lower = slower/more deliberate. Default 0.5.
    """
    print(f"[TTS] Synthesizing with default voice...")
    print(f"[TTS] Text: {text[:80]}{'...' if len(text) > 80 else ''}")

    t0 = time.time()
    wav = model.generate(
        text,
        exaggeration=0.5,
        cfg_weight=0.5,
    )
    elapsed = time.time() - t0

    stats = save_and_stats(wav, model.sr, output_filename, "Default Voice", elapsed)
    return wav, model.sr, stats


def synthesize_clone(model: ChatterboxTurboTTS, text: str, ref_audio: str,
                     output_filename: str = "zero_shot_clone.wav"):
    """
    Zero-shot voice cloning: synthesize text in the voice of ref_audio.
    ref_audio: path to a 3–15s WAV of the target speaker (clean, no background noise).
    """
    if not os.path.exists(ref_audio):
        print(f"[TTS] ❌ Reference audio not found: {ref_audio}")
        return None, None, {}

    print(f"[TTS] Zero-shot cloning from: {ref_audio}")
    print(f"[TTS] Text: {text[:80]}{'...' if len(text) > 80 else ''}")

    t0 = time.time()
    wav = model.generate(
        text,
        audio_prompt_path=ref_audio,
        exaggeration=0.5,
        cfg_weight=0.5,
    )
    elapsed = time.time() - t0

    stats = save_and_stats(wav, model.sr, output_filename, "Zero-Shot Clone", elapsed)
    return wav, model.sr, stats


def benchmark(model: ChatterboxTurboTTS, ref_audio: str):
    """Benchmark default voice vs zero-shot clone across multiple prompts."""
    test_texts = [
        "The weather today is sunny and warm.",
        "Neural networks can recognize speech and translate languages.",
        "Deep learning has revolutionized artificial intelligence in recent years.",
    ]

    print("[TTS] ── Benchmark ──────────────────────────────────")
    default_stats, clone_stats = [], []

    for i, text in enumerate(test_texts):
        _, _, s = synthesize_default(model, text, f"bench_default_{i}.wav")
        default_stats.append(s)
        _, _, s = synthesize_clone(model, text, ref_audio, f"bench_clone_{i}.wav")
        clone_stats.append(s)

    print(f"\n  {'#':<3} {'Default RTF':>12} {'Clone RTF':>10}")
    print(f"  {'-'*3} {'-'*12} {'-'*10}")
    for i in range(len(test_texts)):
        print(f"  {i:<3} {default_stats[i]['rtf']:>12.3f} {clone_stats[i]['rtf']:>10.3f}")

    avg_d = sum(s['rtf'] for s in default_stats) / len(default_stats)
    avg_c = sum(s['rtf'] for s in clone_stats)   / len(clone_stats)
    print(f"\n  Avg RTF — Default: {avg_d:.3f} | Zero-Shot Clone: {avg_c:.3f}")
    print("─────────────────────────────────────────────────────\n")
    return default_stats, clone_stats


if __name__ == "__main__":
    model = load_model()

    # default voice
    print("=" * 55)
    print("TEST 1: Default voice synthesis")
    print("=" * 55)
    wav, sr, stats = synthesize_default(model, TEST_TEXT)
    #play_audio(wav, sr)

    #zero-shot voice clonning
    print("=" * 55)
    print("TEST 2: Zero-shot voice cloning")
    print("=" * 55)
    wav, sr, stats = synthesize_clone(model, TEST_TEXT, REF_AUDIO)
    # if wav is not None:
    #     play_audio(wav, sr)

    benchmark(model, REF_AUDIO)