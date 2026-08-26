#!/usr/bin/env python3
"""Train a tiny openWakeWord head for "November" from synthetic Kokoro clips.

Pipeline: WAV clips -> openWakeWord shared backbone (mel + embedding, ONNX, CPU)
-> [16 x 96] embedding windows -> small MLP -> november.onnx.

Inputs (16 kHz mono int16 WAV):
  <data>/pos/*.wav       "November" in many voices/speeds
  <data>/neg/*.wav       near-miss words and ordinary phrases
  <data>/ambient/*.wav   real room noise from the Shokz mic

Each clip is placed so the word ENDS near the end of a fixed buffer, which is
what openWakeWord sees at inference (rolling window of the last 16 frames).
"""

import argparse
import glob
import os
import random
import wave

import numpy as np
import torch
import torch.nn as nn
from openwakeword.utils import AudioFeatures

SR = 16000
BUF_S = 2.56                    # buffer fed to the backbone (~23 embedding frames)
BUF = int(BUF_S * SR)
WINDOW = 20                     # embedding frames per sample = 1.6 s; openWakeWord reads this
                                # from the ONNX input shape. 16 (1.28 s) clipped slow "November"s.
EMB = 96


def read_wav(path):
    with wave.open(path, "rb") as w:
        assert w.getframerate() == SR and w.getnchannels() == 1, path
        x = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    return x.astype(np.float32) / 32768.0


def trim_silence(x, thresh=0.01):
    idx = np.where(np.abs(x) > thresh)[0]
    if len(idx) == 0:
        return x
    return x[max(0, idx[0] - 800): idx[-1] + 800]


def place(clip, ambient, rng, end_jitter_s=0.25, snr_db=(5, 45), gain_db=(-6, 6), p_clean=0.35):
    """Return a BUF-length float buffer: ambient bed + clip ending near the end.

    The bed is OMITTED for a share of samples (p_clean) and otherwise spans a wide
    SNR range. The Shokz mic's real floor is near-silent, so the head must have
    seen clean speech as a NEGATIVE too — a first model trained with a loud bed on
    every sample fired at 1.0 on any clean word.
    """
    clip = clip[-BUF:]              # keep the END of the word — that's what the window sees
    bed_start = rng.integers(0, max(1, len(ambient) - BUF))
    bed = ambient[bed_start: bed_start + BUF].copy()
    if len(bed) < BUF:
        bed = np.pad(bed, (0, BUF - len(bed)))
    if rng.random() < p_clean:
        bed[:] = 0.0
    else:
        # scale bed to target SNR relative to clip
        c_rms = np.sqrt(np.mean(clip ** 2)) + 1e-8
        b_rms = np.sqrt(np.mean(bed ** 2)) + 1e-8
        snr = rng.uniform(*snr_db)
        bed *= (c_rms / (10 ** (snr / 20))) / b_rms
    end = BUF - int(rng.uniform(0, end_jitter_s) * SR)
    start = max(0, end - len(clip))
    out = bed
    out[start:end] += clip[: end - start]
    out *= 10 ** (rng.uniform(*gain_db) / 20)
    return np.clip(out, -1, 1)


def to_int16(x):
    return (x * 32767).astype(np.int16)


def stream_windows(feats, x_int16, pick):
    """Run a clip through the STREAMING feature path (what ear.py uses) and return
    the [WINDOW, 96] embedding windows for the frames where pick(frame_end_sample)
    is True. Mirrors openwakeword.Model.predict frame by frame."""
    feats.reset()
    out = []
    nframes = 0
    for i in range(0, len(x_int16) - FRAME + 1, FRAME):
        feats(x_int16[i: i + FRAME])
        nframes += 1
        # reset() seeds the feature buffer and fills the mel buffer with 76 rows of
        # ones; 8 mel rows per frame -> 10 frames to flush, then WINDOW frames of
        # steady state before the last-WINDOW window is clean.
        if nframes < WINDOW + 10:
            continue
        if pick(i + FRAME):
            out.append(feats.get_features(WINDOW)[0].astype(np.float32))   # drop batch axis
    return out


FRAME = 1280
PAD = 16000
LEAD = (WINDOW + 11) * FRAME    # leading zeros so the warm-up skip consumes only pad


def mine_streaming(feats, pos_files, neg_files, ambient, rng):
    """Positive windows ending at the word's end; negative windows everywhere else."""
    X_pos, X_neg = [], []
    for f in pos_files:
        clip = trim_silence(read_wav(f))
        x = np.concatenate([np.zeros(LEAD, np.float32), clip, np.zeros(PAD, np.float32)])
        end = LEAD + len(clip)
        X_pos += stream_windows(feats, to_int16(x), lambda e: end - 0.1 * SR <= e <= end + 0.3 * SR)
        # before the word has finished ("Novem"): negative
        X_neg += stream_windows(feats, to_int16(x), lambda e: e <= end - 0.35 * SR and e % (2 * FRAME) == 0)
    for f in neg_files:
        clip = trim_silence(read_wav(f))
        x = np.concatenate([np.zeros(LEAD, np.float32), clip, np.zeros(PAD // 2, np.float32)])
        X_neg += stream_windows(feats, to_int16(x), lambda e: (e // FRAME) % 2 == 0)
    # real room noise, every window
    X_neg += stream_windows(feats, to_int16(ambient[: 60 * SR]), lambda e: (e // FRAME) % 2 == 0)
    return np.stack(X_pos), np.stack(X_neg)


class Head(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(WINDOW * EMB, 128), nn.LayerNorm(128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return torch.sigmoid(self.net(x))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="november.onnx")
    ap.add_argument("--variants", type=int, default=4, help="augmented copies per clip")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    pos_files = sorted(glob.glob(os.path.join(args.data, "pos", "*.wav")))
    neg_files = sorted(glob.glob(os.path.join(args.data, "neg", "*.wav")))
    amb_files = sorted(glob.glob(os.path.join(args.data, "ambient", "*.wav")))
    assert pos_files and neg_files and amb_files, "need pos/, neg/, ambient/ WAVs"
    ambient = np.concatenate([read_wav(f) for f in amb_files])
    print(f"pos {len(pos_files)}  neg {len(neg_files)}  ambient {len(ambient)/SR:.1f}s")

    # Streaming sweeps every alignment, so negatives must be seen at ANY position in
    # the window (end_jitter up to the whole buffer); positives end near the end
    # with wider jitter than the spotter's frame hop.
    NEG_JITTER = BUF_S - 0.3
    buffers, labels = [], []
    for f in pos_files:
        clip = trim_silence(read_wav(f))
        for _ in range(args.variants):
            # Small jitter only: the whole word must stay inside the WINDOW, or the
            # head learns that "...vember" is a positive (fires on remember/December).
            buffers.append(place(clip, ambient, rng, end_jitter_s=0.15)); labels.append(1)
        # hard negative: the first ~60% of the word ("Novem") must NOT fire
        part = clip[: int(len(clip) * 0.6)]
        buffers.append(place(part, ambient, rng, end_jitter_s=NEG_JITTER)); labels.append(0)
    for f in neg_files:
        clip = trim_silence(read_wav(f))
        for _ in range(args.variants):
            buffers.append(place(clip, ambient, rng, end_jitter_s=NEG_JITTER)); labels.append(0)
    # pure ambient windows
    for _ in range(len(pos_files)):
        s = rng.integers(0, max(1, len(ambient) - BUF))
        bed = ambient[s: s + BUF]
        if len(bed) < BUF:
            bed = np.pad(bed, (0, BUF - len(bed)))
        buffers.append(np.clip(bed * 10 ** (rng.uniform(-6, 12) / 20), -1, 1)); labels.append(0)

    X = np.stack([to_int16(b) for b in buffers])
    y = np.array(labels, dtype=np.float32)
    print(f"samples {len(y)}  positives {int(y.sum())}")

    feats = AudioFeatures(inference_framework="onnx")
    emb = feats.embed_clips(X, batch_size=64)          # [N, frames, 96]
    print("embedding shape", emb.shape)
    assert emb.shape[1] >= WINDOW, "buffer too short for the window"
    E = emb[:, -WINDOW:, :].astype(np.float32)          # last WINDOW frames = what oww sees

    # Streaming-mined windows: the exact feature path ear.py runs, at every alignment.
    S_pos, S_neg = mine_streaming(feats, pos_files, neg_files, ambient, rng)
    print(f"streaming windows: pos {len(S_pos)}  neg {len(S_neg)}")
    E = np.concatenate([E, S_pos, S_neg])
    y = np.concatenate([y, np.ones(len(S_pos), np.float32), np.zeros(len(S_neg), np.float32)])
    print(f"total windows {len(y)}  positives {int(y.sum())}")

    idx = rng.permutation(len(y))
    n_val = max(1, len(y) // 10)
    val, tr = idx[:n_val], idx[n_val:]
    Xt, yt = torch.tensor(E[tr]), torch.tensor(y[tr])
    Xv, yv = torch.tensor(E[val]), torch.tensor(y[val])

    model = Head()
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-3)
    pos_w = torch.tensor((len(yt) - yt.sum()) / max(1.0, yt.sum()))
    lossf = nn.BCELoss(reduction="none")
    for ep in range(args.epochs):
        model.train()
        perm = torch.randperm(len(yt))
        tot = 0.0
        for i in range(0, len(perm), 64):
            b = perm[i: i + 64]
            p = model(Xt[b]).squeeze(1)
            w = torch.where(yt[b] > 0.5, pos_w, torch.tensor(1.0))
            loss = (lossf(p, yt[b]) * w).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(b)
        model.eval()
        with torch.no_grad():
            pv = model(Xv).squeeze(1)
            acc = ((pv > 0.5).float() == yv).float().mean().item()
            fp = ((pv > 0.5) & (yv < 0.5)).sum().item()
            fn = ((pv <= 0.5) & (yv > 0.5)).sum().item()
        if ep % 5 == 0 or ep == args.epochs - 1:
            print(f"epoch {ep:3d}  loss {tot/len(yt):.4f}  val acc {acc:.3f}  fp {fp}  fn {fn}")

    model.eval()
    dummy = torch.zeros(1, WINDOW, EMB)
    torch.onnx.export(model, dummy, args.out, input_names=["input"], output_names=["output"],
                      dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}}, opset_version=17,
                      dynamo=False)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
