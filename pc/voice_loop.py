"""Phase 1 voice loop: you speak -> Whisper turns it into text -> Claude replies -> PC speaks it.

Run from robot/pc with the venv active:
    python voice_loop.py

Controls:
    Enter            start recording, then Enter again to stop
    type text+Enter  send typed text instead of speaking
    quit             exit
"""

import queue

import anthropic
import numpy as np
import pyttsx3
import sounddevice as sd
from faster_whisper import WhisperModel

CLAUDE_MODEL = "claude-haiku-5-5"
WHISPER_MODEL = "base.en"   # try "small.en" if it mishears you (slower, more accurate)
SAMPLE_RATE = 16000         # Whisper expects 16 kHz audio
MIC_DEVICE = None           # None = Windows default mic. Or a number from `python -m sounddevice`
MAX_HISTORY = 20            # messages of conversation kept for context

SYSTEM_PROMPT = """You are the voice of a small home robot that Doug is building.
Right now you are only a voice running on his PC: you have no body, wheels, arm or camera yet,
so never claim to see or move anything. Be friendly, a little playful, and helpful.
Your replies are spoken aloud, so keep them short and conversational: one to three sentences,
no lists, no markdown, no emojis, and spell out symbols and units as words."""


def record_until_enter() -> np.ndarray:
    """Record from the mic until the user presses Enter. Returns mono float32 audio."""
    chunks: queue.Queue = queue.Queue()

    def callback(indata, frames, time_info, status):
        chunks.put(indata.copy())

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32",
                        device=MIC_DEVICE, callback=callback):
        input()  # blocks while audio streams in

    parts = []
    while not chunks.empty():
        parts.append(chunks.get())
    if not parts:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate(parts).flatten()


def transcribe(model: WhisperModel, audio: np.ndarray) -> str:
    segments, _ = model.transcribe(audio, language="en", beam_size=1, vad_filter=True)
    return " ".join(seg.text.strip() for seg in segments).strip()


def speak(text: str) -> None:
    # A fresh engine each time avoids a known pyttsx3 hang on Windows after the first reply.
    engine = pyttsx3.init()
    engine.setProperty("rate", 175)
    engine.say(text)
    engine.runAndWait()
    engine.stop()


def ask_claude(client: anthropic.Anthropic, history: list) -> str:
    response = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=300,
        system=SYSTEM_PROMPT,
        messages=history,
    )
    return "".join(block.text for block in response.content if block.type == "text").strip()


def main() -> None:
    print("Loading speech model (the first run downloads it, about 150 MB)...")
    whisper = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8")
    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from the environment
    history: list = []

    print("\nReady. Press Enter to talk (Enter again to stop), or type a message. Type 'quit' to exit.\n")

    while True:
        try:
            line = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if line.lower() in ("quit", "exit"):
            break

        if line:
            user_text = line
        else:
            print("  Listening... press Enter to stop.")
            audio = record_until_enter()
            if len(audio) < SAMPLE_RATE * 0.3:
                print("  That was too short, try again.")
                continue
            user_text = transcribe(whisper, audio)
            if not user_text:
                print("  Didn't catch that, try again.")
                continue
            print(f"  You: {user_text}")

        history.append({"role": "user", "content": user_text})
        try:
            reply = ask_claude(client, history)
        except anthropic.APIError as err:
            print(f"  Claude error: {err}")
            history.pop()
            continue

        history.append({"role": "assistant", "content": reply})
        history = history[-MAX_HISTORY:]
        if history and history[0]["role"] != "user":
            history = history[1:]

        print(f"  Robot: {reply}")
        speak(reply)

    print("Bye!")


if __name__ == "__main__":
    main()
