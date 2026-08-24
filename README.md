# LLMINUX — Hybridized Edition

Linux where the LLM is the shell. Three-tier hybrid inference — NPU sentinel, GPU shell, GPU brain.

No GUI, no window manager, no click-dragging. The entire interface is natural language — voice or text — routed through a tiny always-on NPU model that parses commands into structured verbs and escalates to heavier models only when it needs to think.

## Architecture

```
                          ┌─────────────────┐
                          │    Whisper       │
                          │  (ear, CPU/NPU)  │
                          └────────┬────────┘
                                   │ transcription
                          ┌────────▼────────┐
                          │   NPU Sentinel  │
                          │  ~1-2B dense    │
                          │  always-on PID1 │
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

| Tier | Model | Hardware | Role | Latency |
|------|-------|----------|------|---------|
| **Sentinel** | ~1-2B dense (Qwen3-0.6B) | NPU (XDNA 2) / CPU fallback | Parse input → verbs, direct commands, display, routing | <50ms |
| **Shell** | 4B/8B dense Q5/Q6 | GPU (Vulkan, 890M) | Tool-call chains, file ops, multi-step tasks | ~200ms prefill |
| **Brain** | 30B-A3B MoE Q3_K_M | GPU (Vulkan, 890M) | Reasoning, composition, ambiguity | ~41 tok/s gen |

### Why Three Tiers

The sentinel is the kernel. It never sleeps, it owns the mic, it owns the display verbs, it owns the conversation state. 90% of OS commands — "open that file," "disk usage," "play music," "what time is it" — are trivially parseable. A 1-2B model classifies and dispatches those without ever waking the GPU.

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

All tiers emit tool calls through JSON schema constraints (GBNF grammar in llama.cpp). At low quant on small active-param MoE, unconstrained JSON generation has unacceptable malformation rates. Grammar enforcement is free and non-negotiable.

## Hardware Target

AMD Ryzen AI 9 HX 370 (GPD Pocket 4):
- **NPU**: XDNA 2 (aie2p) — sentinel tier. Driver mainlined 6.14+, inference stack (ONNX RT XDNA EP) maturing. CPU fallback until ready.
- **GPU**: Radeon 890M (gfx1150) — shell + brain tiers via Vulkan compute. ~17.3 GB GTT ceiling.
- **CPU**: Zen 5, 12 cores — Whisper STT + Piper/Kokoro TTS. No GPU contention.
- **RAM**: 32GB DDR5-5600 shared across all three processors.

Both shell (4B Q6 ~4GB) and brain (30B Q3 ~14GB) can potentially co-reside in GTT (~17.3GB ceiling), loaded/unloaded by the sentinel based on demand.

## Status

Concept stage. The Signal bridge (tool-calling LLM agent via Signal) already proves the core inference-loop-as-shell pattern works with 6 tools.

**Next steps:**
1. Benchmark Qwen3-4B dense Q6 on Vulkan — prefill tok/s at 4k/8k context
2. Run 50 structured tool calls, measure JSON malformation rate
3. Prototype sentinel with Qwen3-0.6B on CPU (NPU parked until ONNX RT EP lands)
4. Define the display verb schema and build a minimal renderer

## License

MIT
