#!/usr/bin/env python3
"""Stream WAVs through openWakeWord exactly as ear.py does (1280-sample frames)
and report the peak score per file. Usage: spot_test.py november.onnx a.wav b.wav ..."""

import sys
import wave

import numpy as np
from openwakeword.model import Model

FRAME = 1280


def read_int16(path):
    with wave.open(path, "rb") as w:
        assert w.getframerate() == 16000 and w.getnchannels() == 1, path
        return np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)


def main():
    model_path, files = sys.argv[1], sys.argv[2:]
    m = Model(wakeword_models=[model_path], inference_framework="onnx")
    for f in files:
        x = read_int16(f)
        # 1 s of silence before and after so the rolling window sees the whole word
        x = np.concatenate([np.zeros(16000, np.int16), x, np.zeros(16000, np.int16)])
        m.reset()
        peak, at = 0.0, 0
        for i in range(0, len(x) - FRAME + 1, FRAME):
            s = max(m.predict(x[i:i + FRAME]).values())
            if s > peak:
                peak, at = s, i
        print(f"  {peak:.3f} @ {at/16000:5.2f}s  {f.rsplit('/', 1)[-1]}")


if __name__ == "__main__":
    main()
