"""
Voice Selector UI
Opens a window letting the user:
  1. Select an existing audio file for voice cloning
  2. Record their own voice live for voice cloning
  3. Use the default voice

Returns the selected/recorded file path, or None for default voice.

Usage:
    from voice_selector_ui import select_voice
    ref_audio = select_voice()  # returns path str or None
"""

import os
import time
import threading
import tempfile
import tkinter as tk
from tkinter import filedialog
import numpy as np
import sounddevice as sd
import soundfile as sf


# ── Recording config ──────────────────────────────────────────────────────────
SCRIPT_DIR      = os.path.dirname(os.path.abspath(__file__))
SAMPLE_RATE     = 16000
CHANNELS        = 1
RECORDING_DIR   = os.path.join(SCRIPT_DIR, "voice_recordings")
# ─────────────────────────────────────────────────────────────────────────────


def select_voice() -> str | None:
    """
    Opens a GUI window for voice selection.
    Returns selected/recorded file path, or None for default voice.
    """
    result = {"path": None}

    root = tk.Tk()
    root.title("Voice Cloning Setup")
    root.geometry("500x380")
    root.resizable(False, False)

    # ── Styles ────────────────────────────────────────────────────────────────
    BG      = "#1e1e2e"
    CARD    = "#2a2a3e"
    ACCENT  = "#7c6af7"
    GREEN   = "#a6e3a1"
    RED     = "#f38ba8"
    FG      = "#cdd6f4"
    FG_DIM  = "#6c7086"
    FONT    = ("Segoe UI", 10)
    FONT_LG = ("Segoe UI", 13, "bold")
    FONT_SM = ("Segoe UI", 9)

    root.configure(bg=BG)

    # ── Title ─────────────────────────────────────────────────────────────────
    tk.Label(root, text="🎙️  Voice Assistant Setup",
             font=FONT_LG, bg=BG, fg=FG).pack(pady=(20, 4))
    tk.Label(root, text="Choose a voice to clone, or use the default voice.",
             font=FONT_SM, bg=BG, fg=FG_DIM).pack(pady=(0, 14))

    # ── Status label ──────────────────────────────────────────────────────────
    status_var = tk.StringVar(value="No voice selected — default will be used")
    status_label = tk.Label(
        root, textvariable=status_var,
        font=FONT_SM, bg=CARD, fg=FG_DIM,
        anchor="w", wraplength=420, justify="left", padx=12, pady=8
    )
    status_label.pack(fill="x", padx=30, pady=(0, 14))

    # ── Option 1: Browse file ─────────────────────────────────────────────────
    section1 = tk.LabelFrame(root, text=" Option 1: Select existing audio file ",
                              font=FONT_SM, bg=BG, fg=FG_DIM, bd=1, relief="groove")
    section1.pack(fill="x", padx=30, pady=(0, 10))

    def browse():
        file_path = filedialog.askopenfilename(
            title="Select reference audio file",
            initialdir=SCRIPT_DIR,
            filetypes=[
                ("Audio files", "*.wav *.mp3 *.flac *.ogg *.m4a"),
                ("All files", "*.*"),
            ]
        )
        if file_path:
            result["path"] = file_path
            status_var.set(f"✅  {os.path.basename(file_path)}")
            status_label.config(fg=GREEN)
            confirm_btn.config(text="Use Selected Voice")

    tk.Button(
        section1, text="Browse Audio File",
        font=FONT, bg=CARD, fg=FG,
        activebackground=ACCENT, activeforeground="white",
        relief="flat", cursor="hand2", padx=14, pady=5,
        command=browse,
    ).pack(pady=8)

    # ── Option 2: Record own voice ────────────────────────────────────────────
    section2 = tk.LabelFrame(root, text=" Option 2: Record your own voice ",
                              font=FONT_SM, bg=BG, fg=FG_DIM, bd=1, relief="groove")
    section2.pack(fill="x", padx=30, pady=(0, 14))

    rec_frame    = tk.Frame(section2, bg=BG)
    rec_frame.pack(pady=8)

    record_btn   = tk.Button(rec_frame, text="⏺  Start Recording",
                             font=FONT, bg=CARD, fg=FG,
                             activebackground=RED, activeforeground="white",
                             relief="flat", cursor="hand2", padx=14, pady=5)
    record_btn.pack(side="left", padx=6)

    timer_label  = tk.Label(rec_frame, text="", font=FONT_SM, bg=BG, fg=FG_DIM)
    timer_label.pack(side="left", padx=4)

    # Recording state
    rec_state    = {"active": False, "chunks": [], "thread": None, "start": 0}

    def update_timer():
        if rec_state["active"]:
            elapsed = time.time() - rec_state["start"]
            timer_label.config(text=f"{elapsed:.1f}s")
            root.after(100, update_timer)

    def do_record():
        """Record audio in a background thread."""
        with sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS,
                            dtype="float32") as stream:
            while rec_state["active"]:
                chunk, _ = stream.read(int(SAMPLE_RATE * 0.1))
                rec_state["chunks"].append(chunk.copy())

    def toggle_recording():
        if not rec_state["active"]:
            # Start recording
            rec_state["active"] = True
            rec_state["chunks"] = []
            rec_state["start"]  = time.time()
            rec_state["thread"] = threading.Thread(target=do_record, daemon=True)
            rec_state["thread"].start()
            record_btn.config(text="⏹  Stop Recording", bg="#3a1a2e", fg=RED)
            update_timer()
        else:
            # Stop recording
            rec_state["active"] = False
            rec_state["thread"].join()

            audio = np.concatenate(rec_state["chunks"], axis=0)
            duration = len(audio) / SAMPLE_RATE

            if duration < 3.0:
                timer_label.config(text="⚠️ Too short! Record at least 3s.", fg=RED)
                record_btn.config(text="⏺  Start Recording", bg=CARD, fg=FG)
                return

            # Save to file — create recordings folder only when needed
            os.makedirs(RECORDING_DIR, exist_ok=True)
            out_path = os.path.join(RECORDING_DIR, "my_voice.wav")
            sf.write(out_path, audio, SAMPLE_RATE)

            result["path"] = out_path
            status_var.set(f"✅  Recorded voice ({duration:.1f}s) → my_voice.wav")
            status_label.config(fg=GREEN)
            timer_label.config(text=f"✅ {duration:.1f}s saved", fg=GREEN)
            record_btn.config(text="⏺  Re-record", bg=CARD, fg=FG)
            confirm_btn.config(text="Use Recorded Voice")

    record_btn.config(command=toggle_recording)

    # ── Confirm button ────────────────────────────────────────────────────────
    def confirm():
        # Stop recording if still active
        if rec_state["active"]:
            rec_state["active"] = False
            if rec_state["thread"]:
                rec_state["thread"].join()
        root.destroy()

    confirm_btn = tk.Button(
        root, text="Use Default Voice",
        font=FONT, bg=ACCENT, fg="white",
        activebackground="#6a59e0", activeforeground="white",
        relief="flat", cursor="hand2", padx=20, pady=7,
        command=confirm,
    )
    confirm_btn.pack(pady=(0, 16))

    # ── Center window ─────────────────────────────────────────────────────────
    root.update_idletasks()
    w, h = root.winfo_width(), root.winfo_height()
    x = (root.winfo_screenwidth()  // 2) - (w // 2)
    y = (root.winfo_screenheight() // 2) - (h // 2)
    root.geometry(f"{w}x{h}+{x}+{y}")

    root.mainloop()
    return result["path"]


if __name__ == "__main__":
    selected = select_voice()
    if selected:
        print(f"Selected voice: {selected}")
    else:
        print("Using default voice.")