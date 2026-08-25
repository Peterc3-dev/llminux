#!/usr/bin/env python3
"""LLMINUX ear daemon — hear → parse → act → speak.

Full voice loop: mic → energy VAD → Whisper STT (NPU) → sentinel parse →
verb execution → Kokoro TTS response.

Requires: flm.service running with --asr 1, whisper-v3:turbo pulled,
kokoro-v1.0.onnx + voices-v1.0.bin in project dir.
"""

import io
import json
import os
import re
import subprocess
import sys
import tempfile
import threading
import time
import wave
from pathlib import Path

import numpy as np
import sounddevice as sd
from kokoro_onnx import Kokoro

SAMPLE_RATE = 16000
CHANNELS = 1
BLOCK_SIZE = 1600  # 100ms at 16kHz

SPEECH_THRESHOLD = 0.015  # RMS float32, tune with --calibrate
SPEECH_PAD_S = 0.8
MIN_SPEECH_S = 0.5
MAX_SPEECH_S = 15

FLM_URL = "http://localhost:52625"
TRANSCRIBE_URL = f"{FLM_URL}/v1/audio/transcriptions"
SENTINEL_URL = f"{FLM_URL}/api/chat"
SENTINEL_MODEL = "qwen3:1.7b"

PROMPT_FILE = Path(__file__).parent / "sentinel_prompt.txt"
MODEL_DIR = Path(__file__).parent
KOKORO_MODEL = MODEL_DIR / "kokoro-v1.0.onnx"
KOKORO_VOICES = MODEL_DIR / "voices-v1.0.bin"
KOKORO_VOICE = "af_heart"
KOKORO_SPEED = 1.0
SPEAK_COOLDOWN_S = 1.0

WHISPER_NOISE = {"", "you", "thank you", "thanks", "bye", "okay",
                 "thank you.", "thanks.", "bye.", "you."}


def is_hallucination(text):
    words = text.strip().split()
    if len(words) < 4:
        return False
    chunks = [" ".join(words[i:i+2]) for i in range(len(words) - 1)]
    if not chunks:
        return False
    most_common = max(set(chunks), key=chunks.count)
    return chunks.count(most_common) / len(chunks) > 0.5


def rms(block):
    return float(np.sqrt(np.mean(block.astype(np.float32) ** 2)))


def to_wav(audio, sr):
    int16 = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(int16.tobytes())
    return buf.getvalue()


def transcribe(wav_bytes):
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as f:
        f.write(wav_bytes)
        f.flush()
        r = subprocess.run(
            ["curl", "-s", TRANSCRIBE_URL,
             "-F", f"file=@{f.name}",
             "-F", "model=whisper-v3:turbo",
             "-F", "response_format=json"],
            capture_output=True, text=True, timeout=30,
        )
    if r.returncode != 0 or not r.stdout.strip():
        raise RuntimeError(f"transcribe failed: {r.stderr[:200]}")
    resp = json.loads(r.stdout)
    if "error" in resp:
        raise RuntimeError(resp["error"])
    return resp.get("text", "").strip()


def fix_json(raw):
    s = raw.strip()
    if "<think>" in s:
        s = s.split("</think>")[-1].strip()
    s = re.sub(r"^```json\s*", "", s)
    s = re.sub(r"\s*```$", "", s)
    s = s.strip()
    try:
        json.loads(s)
        return s
    except json.JSONDecodeError:
        pass
    while s.endswith("}") and s.count("{") < s.count("}"):
        s = s[:-1]
    return s


def ask_sentinel(text, system_prompt):
    payload = json.dumps({
        "model": SENTINEL_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text},
        ],
        "stream": False,
        "options": {"num_predict": 100, "temperature": 0},
    })
    r = subprocess.run(
        ["curl", "-s", SENTINEL_URL,
         "-H", "Content-Type: application/json",
         "-d", payload],
        capture_output=True, text=True, timeout=30,
    )
    if r.returncode != 0 or not r.stdout.strip():
        raise RuntimeError(f"sentinel failed: {r.stderr[:200]}")
    resp = json.loads(r.stdout)
    if "error" in resp:
        raise RuntimeError(resp["error"])
    content = fix_json(resp["message"]["content"])
    parsed = json.loads(content)
    latency = round(resp.get("total_duration", 0) / 1e6)
    return parsed, latency


def execute_verb(verb):
    name = verb.get("verb", "")
    args = verb.get("args", {})

    if name == "run_command":
        cmd = args.get("cmd", "")
        if not cmd:
            return "No command specified."
        try:
            r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=10)
            out = (r.stdout.strip() or r.stderr.strip() or "Done.")
            return out[:500]
        except subprocess.TimeoutExpired:
            return "Command timed out."

    if name == "show_text":
        return args.get("text", "")

    if name == "disk_usage":
        r = subprocess.run(["df", "-h", "--output=size,avail,pcent", "/"],
                           capture_output=True, text=True)
        lines = [l for l in r.stdout.strip().split("\n") if not l.strip().startswith("Size")]
        if lines:
            parts = lines[0].split()
            if len(parts) >= 3:
                return f"{parts[1]} free of {parts[0]}, {parts[2]} used."
        return r.stdout.strip()[:300]

    if name == "list_dir":
        path = os.path.expanduser(args.get("path", "."))
        try:
            entries = sorted(os.listdir(path))[:20]
            return ", ".join(entries) if entries else "Empty directory."
        except OSError as e:
            return str(e)

    if name == "open_file":
        path = os.path.expanduser(args.get("path", ""))
        try:
            text = Path(path).read_text()[:500]
            return text or "Empty file."
        except OSError as e:
            return str(e)

    if name == "escalate":
        return f"Escalating: {args.get('task', 'unknown task')}. GPU brain not wired yet."

    if name == "play_media":
        return f"Media playback not wired yet: {args.get('query', '')}"

    return f"Unknown verb: {name}"


class Ear:
    def __init__(self, threshold=SPEECH_THRESHOLD, device=None, voice=KOKORO_VOICE):
        self.threshold = threshold
        self.device = device
        self.system_prompt = PROMPT_FILE.read_text().strip()
        self.recording = False
        self.buffer = []
        self.silence_blocks = 0
        self.speech_blocks = 0
        self.pad_blocks = int(SPEECH_PAD_S * SAMPLE_RATE / BLOCK_SIZE)
        self.min_blocks = int(MIN_SPEECH_S * SAMPLE_RATE / BLOCK_SIZE)
        self.max_blocks = int(MAX_SPEECH_S * SAMPLE_RATE / BLOCK_SIZE)
        self.processing = False
        self.voice = voice
        self.kokoro = Kokoro(str(KOKORO_MODEL), str(KOKORO_VOICES))

    def callback(self, indata, frames, time_info, status):
        if status:
            print(f"  audio: {status}", file=sys.stderr)
        block = indata[:, 0].copy()
        energy = rms(block)
        is_speech = energy > self.threshold

        if is_speech:
            self.silence_blocks = 0
            if not self.recording and not self.processing:
                self.recording = True
                self.speech_blocks = 0
                self.buffer = []
                print("  \033[33m◉ speech\033[0m", file=sys.stderr, end="\r")
            if self.recording:
                self.buffer.append(block)
                self.speech_blocks += 1
        elif self.recording:
            self.silence_blocks += 1
            self.buffer.append(block)
            if self.silence_blocks >= self.pad_blocks or self.speech_blocks >= self.max_blocks:
                self.recording = False
                if self.speech_blocks >= self.min_blocks:
                    audio = np.concatenate(self.buffer)
                    self.buffer = []
                    self.processing = True
                    threading.Thread(target=self._process, args=(audio,), daemon=True).start()
                else:
                    self.buffer = []

    def speak(self, text):
        if not text or not text.strip():
            return
        utterance = text.strip()[:300]
        try:
            t0 = time.monotonic()
            samples, sr = self.kokoro.create(utterance, voice=self.voice, speed=KOKORO_SPEED)
            tts_ms = int((time.monotonic() - t0) * 1000)
            duration = len(samples) / sr
            print(f"\033[35m  speak:\033[0m \"{utterance[:60]}\" ({duration:.1f}s, {tts_ms}ms gen)")
            pad = np.zeros(int(sr * 0.05), dtype=samples.dtype)
            samples = np.concatenate([pad, samples])
            sd.play(samples, sr)
            sd.wait()
        except Exception as e:
            print(f"  \033[31m✗ tts: {e}, falling back to espeak\033[0m", file=sys.stderr)
            subprocess.run(["espeak-ng", "-v", "en-us", "-s", "140", "-p", "30", utterance],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(SPEAK_COOLDOWN_S)

    def _process(self, audio):
        try:
            duration = len(audio) / SAMPLE_RATE
            wav = to_wav(audio, SAMPLE_RATE)
            print(f"  \033[36m◆ {duration:.1f}s captured, transcribing...\033[0m", file=sys.stderr)

            t0 = time.monotonic()
            text = transcribe(wav)
            stt_ms = int((time.monotonic() - t0) * 1000)

            if text.strip().lower() in WHISPER_NOISE or is_hallucination(text):
                print(f"  \033[90m○ noise/hallucination: '{text[:40]}' ({stt_ms}ms)\033[0m", file=sys.stderr)
                return

            print(f"\033[33m  heard:\033[0m \"{text}\" ({stt_ms}ms)")

            verb, sentinel_ms = ask_sentinel(text, self.system_prompt)
            verb_str = verb.get("verb", "?")
            args_str = json.dumps(verb.get("args", {}))
            print(f"\033[32m  verb:\033[0m {verb_str} {args_str} ({sentinel_ms}ms)")

            result = execute_verb(verb)
            print(f"\033[36m  result:\033[0m {result[:80]}")
            self.speak(result)
        except Exception as e:
            print(f"  \033[31m✗ {e}\033[0m", file=sys.stderr)
        finally:
            self.processing = False

    def run(self):
        dev_id = self.device
        dev_info = sd.query_devices(dev_id, "input") if dev_id is not None else sd.query_devices(kind="input")
        dev_name = dev_info["name"]
        print("\033[32mLLMINUX ear daemon\033[0m")
        print(f"  mic: {dev_name} ({SAMPLE_RATE}Hz mono)")
        print(f"  vad: energy threshold {self.threshold}")
        print(f"  stt: whisper-v3:turbo (NPU)")
        print(f"  sentinel: {SENTINEL_MODEL} (NPU)")
        print(f"  tts: kokoro-82m ({self.voice})")
        print()
        self.speak("Ears online.")
        print("Listening...\n")

        with sd.InputStream(
            device=self.device,
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype="float32",
            blocksize=BLOCK_SIZE,
            callback=self.callback,
        ):
            try:
                while True:
                    time.sleep(0.1)
            except KeyboardInterrupt:
                print("\nStopped.")


def say(text, kokoro=None, voice=KOKORO_VOICE):
    if kokoro:
        try:
            samples, sr = kokoro.create(text, voice=voice, speed=KOKORO_SPEED)
            sd.play(samples, sr)
            sd.wait()
            return
        except Exception:
            pass
    subprocess.run(["espeak-ng", "-v", "en-us", "-s", "140", "-p", "30", text],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def calibrate(device=None):
    kokoro = Kokoro(str(KOKORO_MODEL), str(KOKORO_VOICES))
    samples = []
    def cb(indata, frames, time_info, status):
        samples.append(rms(indata[:, 0]))

    say("Stay quiet.", kokoro)
    print("Stay quiet for 3 seconds...")
    with sd.InputStream(device=device, samplerate=SAMPLE_RATE, channels=1,
                        dtype="float32", blocksize=BLOCK_SIZE, callback=cb):
        time.sleep(3)
    noise = np.mean(samples)
    noise_peak = np.max(samples)

    print(f"  noise floor: {noise:.4f} (peak {noise_peak:.4f})")
    say("Speak now.", kokoro)
    print("\nSpeak normally for 3 seconds...")
    samples.clear()
    with sd.InputStream(device=device, samplerate=SAMPLE_RATE, channels=1,
                        dtype="float32", blocksize=BLOCK_SIZE, callback=cb):
        time.sleep(3)
    speech = np.mean(samples)
    speech_peak = np.max(samples)

    say("Done.", kokoro)
    print(f"  speech level: {speech:.4f} (peak {speech_peak:.4f})")
    print(f"  ratio: {speech / max(noise, 1e-6):.1f}x")
    threshold = noise_peak * 2.5
    print(f"\n  Recommended: python3 ear.py --threshold {threshold:.4f}")


def test_once(device=None):
    print("Say something (recording 4 seconds)...")
    audio = sd.rec(int(4 * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1,
                   dtype="float32", device=device)
    sd.wait()
    audio = audio[:, 0]
    wav = to_wav(audio, SAMPLE_RATE)
    print(f"  recorded {len(audio)/SAMPLE_RATE:.1f}s, transcribing...")
    try:
        text = transcribe(wav)
        print(f"  text: \"{text}\"")
    except Exception as e:
        print(f"  error: {e}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="LLMINUX ear daemon")
    p.add_argument("--threshold", type=float, default=SPEECH_THRESHOLD)
    p.add_argument("--device", type=int, default=None)
    p.add_argument("--voice", default=KOKORO_VOICE, help="Kokoro voice (default: af_heart)")
    p.add_argument("--calibrate", action="store_true")
    p.add_argument("--test", action="store_true", help="Record 4s and transcribe once")
    p.add_argument("--say", type=str, default=None, help="Speak a phrase and exit")
    args = p.parse_args()

    if args.say:
        kokoro = Kokoro(str(KOKORO_MODEL), str(KOKORO_VOICES))
        say(args.say, kokoro, args.voice)
    elif args.calibrate:
        calibrate(args.device)
    elif args.test:
        test_once(args.device)
    else:
        Ear(threshold=args.threshold, device=args.device, voice=args.voice).run()
