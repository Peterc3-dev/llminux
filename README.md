# LLMINUX — Hybridized Edition

Linux where the LLM is the shell. Three-tier hybrid inference — NPU sentinel, GPU shell, GPU brain.

No GUI, no window manager, no click-dragging. The entire interface is natural language — voice or text — routed through a tiny always-on NPU model that parses commands into structured verbs and escalates to heavier models only when it needs to think.

## Architecture

```
                          ┌─────────────────┐
                          │  Whisper v3     │
                          │  turbo (NPU)   │
                          └────────┬────────┘
                                   │ transcription
                          ┌────────▼────────┐
                          │   NPU Sentinel  │
                          │  Qwen3-0.6B    │
                          │  always-on PID1 │
                          │  XDNA 2 / FLM  │
                          └──┬─────────┬────┘
                 direct verb │         │ escalate
            ┌────────────────┘         └──────────────┐
            │                                         │
   ┌────────▼────────┐                       ┌────────▼────────┐
   │  Display Verbs  │                       │   GPU Shell     │
   │  (dumb render)  │                       │  4B dense Q5/Q6 │
   └────────┬────────┘                       │  Vulkan iGPU    │
            │                                └────────┬────────┘
   ┌────────▼────────┐                  tool chains │  │ escalate
   │  Piper / Kokoro │                              │  │
   │  (mouth, CPU)   │               ┌──────────────┘  │
   └─────────────────┘               │       ┌─────────▼───────┐
                                     │       │   GPU Brain     │
                              ┌──────▼──┐    │  30B MoE Q3_K_M │
                              │ Display │    │  Vulkan iGPU    │
                              │ Verbs   │    └─────────────────┘
                              └─────────┘
```

### The Three Tiers

| Tier | Model | Hardware | Runtime | Role | Latency |
|------|-------|----------|---------|------|---------|
| **Sentinel** | Qwen3-0.6B | XDNA 2 NPU | FLM :52625 | Parse input → verbs, direct commands, display, routing | <50ms |
| **Shell** | 4B/8B dense Q5/Q6 | Radeon 890M | llama.cpp Vulkan :8090 | Tool-call chains, file ops, multi-step tasks | ~200ms prefill |
| **Brain** | 30B-A3B MoE Q3_K_M | Radeon 890M | llama.cpp Vulkan :8090 | Reasoning, composition, ambiguity | ~41 tok/s gen |

### Why Three Tiers

The sentinel is the kernel. It never sleeps, it owns the mic, it owns the display verbs, it owns the conversation state. 90% of OS commands — "open that file," "disk usage," "play music," "what time is it" — are trivially parseable. A 0.6B model on the NPU classifies and dispatches those without ever waking the GPU.

The GPU models are workers the sentinel spawns on demand:
- **Shell** for anything that needs tool-call chains (read → process → write)
- **Brain** for anything that needs actual reasoning

This is a real OS process model. PID 1 is cheap and fast. It forks heavier processes when it needs to think.

### Display Verbs

The sentinel (or any tier returning to it) issues structured display commands, not free-form text:

```
show_text    — render text/markdown to the surface
show_table   — render structured data as a table
show_image   — display an image or chart
show_diff    — show a before/after diff
confirm      — yes/no prompt, blocks until answered
notify       — transient toast notification
```

20 tokens per display call. Not 800 tokens of generated HTML. The surface is a dumb deterministic renderer with a fixed contract.

### Tool Calls

GPU tiers emit tool calls through JSON schema constraints (GBNF grammar in llama.cpp). NPU sentinel uses FLM's Ollama-compatible API with `format: json` where supported. Grammar enforcement is free and non-negotiable on the GPU tiers.

## Hardware Target

AMD Ryzen AI 9 HX 370 (GPD Pocket 4) — all three processors active:

| Processor | Silicon | Role | Runtime |
|-----------|---------|------|---------|
| **NPU** | XDNA 2 (aie2p) | Sentinel + Whisper STT | FLM v0.9.43 |
| **GPU** | Radeon 890M (gfx1150) | Shell + Brain LLM inference | llama.cpp Vulkan |
| **CPU** | Zen 5, 12 cores | Piper/Kokoro TTS | native |
| **RAM** | 32GB DDR5-5600 | Shared across all three | — |

NPU stack verified operational: `amdxdna` v0.10 in-tree, XRT 2.21.75, FLM validated, `/dev/accel/accel0` present. No ONNX Runtime needed — FLM is the NPU inference runtime with an Ollama-compatible API.

Both shell (4B Q6 ~4GB) and brain (30B Q3 ~14GB) can potentially co-reside in GPU GTT (~17.3GB ceiling), loaded/unloaded by the sentinel based on demand.

## Status

NPU sentinel live. Qwen3-1.7B parses commands at 93% verb accuracy, ~1.4s round-trip.

- [x] Concept + architecture designed
- [x] NPU stack verified operational (amdxdna, XRT, FLM)
- [x] Sentinel model pulled + prompt engineered (93% accuracy, 100% valid JSON)
- [x] CLI parser: `python3 sentinel.py "your command"`
- [ ] Pull Whisper-v3-turbo on NPU for STT
- [ ] Benchmark GPU shell tier (4B dense Q6 prefill + tool-call accuracy)
- [ ] Define display verb JSON schema
- [ ] Build minimal renderer
- [ ] Voice loop integration (Whisper → sentinel → Piper)
- [ ] Wire sentinel → GPU escalation path

## Roadmap

### Speculative Escalation

When the mic VAD fires, start prefilling the 30B's KV cache on the GPU with session context — before the sentinel decides whether to escalate. If it doesn't escalate, discard. If it does, the KV cache is already warm. Hides most escalation latency behind STT time.

### Entropy-Based Routing

Replace hardcoded verb heuristics with logprob entropy from the sentinel's output distribution. Low confidence on the verb token → automatic escalation. High confidence → execute. The model tells you when it's unsure without needing explicit "escalate" classification. Calibrate the threshold from logged corrections over time.

### Overnight Self-Distillation

Every escalation where the 30B corrects the sentinel's parse is a training pair. Run nightly LoRA fine-tuning on the NPU model using accumulated correction data. The OS converges on its user's vocabulary — escalation rate decays week over week. No cloud OS can do this.

### KV Cache as Page Cache

Persist per-conversation KV caches to disk (mmap'd), LRU-evicted like a traditional page cache. "Reopening" a previous task is a page-in, not a full re-prefill. Based on Prompt Cache / RadixAttention research.

### Token-Boundary Preemption

GPU decode is interruptible at every token boundary — checkpoint KV state, service the sentinel's interrupt, resume. Real priority scheduling for inference. The sentinel can preempt a long brain-tier generation to handle a quick command, then resume.

### Rust FSM as PID 1

Actual PID 1 should be a ~200-line deterministic Rust state machine, not the LLM. The sentinel is a respawnable child process. Never give a stochastic process init's unkillability. The FSM handles process lifecycle, signal routing, and watchdog — the sentinel handles language.

### Intent Provenance

Every state change records the utterance that caused it in the OS journal. Semantic undo ("undo what I did to the config yesterday") and session replay fall out for free. The journal tracks *why*, not just *what*.

### Prior Art

- [MemGPT](https://memgpt.ai/) — OS-metaphor context paging
- [AIOS](https://github.com/agiresearch/AIOS) (Rutgers) — LLM-agent OS kernel
- [PowerInfer-2](https://arxiv.org/abs/2406.06282) — heterogeneous NPU/GPU inference on phones
- [SGLang RadixAttention](https://arxiv.org/abs/2312.07104) — KV cache reuse
- [Gifford's Semantic File Systems](https://dl.acm.org/doi/10.1145/121132.121138) (1991) — content-addressable storage
- Karpathy's LLM-OS sketch — the conceptual ancestor

## License

MIT
