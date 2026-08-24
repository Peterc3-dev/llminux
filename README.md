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

NPU stack verified. First sentinel bringup in progress.

- [x] Concept + architecture designed
- [x] NPU stack verified operational (amdxdna, XRT, FLM)
- [ ] Pull sentinel model + Whisper on NPU
- [ ] First sentinel loop (parse → verb → display)
- [ ] Benchmark GPU shell tier (4B dense Q6 prefill + tool-call accuracy)
- [ ] Define display verb JSON schema
- [ ] Build minimal renderer
- [ ] Voice loop integration (Whisper → sentinel → Piper)

## License

MIT
