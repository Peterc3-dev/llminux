# LLMINUX

Linux where the LLM is the shell.

No GUI, no window manager, no click-dragging. The entire interface is natural language — voice or text — routed to a local LLM that controls the machine through tool calls and renders output through a fixed set of display verbs.

## Architecture

```
┌──────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────┐
│  Whisper  │───▶│  Shell Tier  │───▶│  Display     │───▶│  Piper/  │
│  (ear)   │    │  4B dense Q6 │    │  Verbs       │    │  Kokoro  │
│  CPU/NPU │    │  Vulkan iGPU │    │  (renderer)  │    │  (mouth) │
└──────────┘    └──────┬───────┘    └──────────────┘    └──────────┘
                       │ escalate
                ┌──────▼───────┐
                │  Brain Tier  │
                │  30B MoE Q3  │
                │  Vulkan iGPU │
                └──────────────┘
```

- **Ear**: Whisper small — speech-to-text, push-to-talk
- **Shell tier**: Qwen3-4B/8B dense at Q5/Q6 — handles commands, tool calls, structured output. Fast prefill, high tool-call accuracy
- **Brain tier**: Qwen3-30B-A3B MoE at Q3_K_M — escalation for reasoning, composition, ambiguity. ~41 tok/s on Radeon 890M
- **Mouth**: Piper or Kokoro-82M — faster-than-real-time TTS on CPU
- **Face**: Fixed display verbs (`show_text`, `show_table`, `show_image`, `show_diff`, `confirm`) — no free-form HTML generation

## Design Principles

1. **The inference loop is init.** Not an app that talks to an OS — the conversation loop is PID 1.
2. **Two-tier routing.** Small model handles 90% of work. Big model handles reasoning. Same pattern as [Arbiter OS](https://en.wikipedia.org/wiki/Microkernel).
3. **Structured output only.** Tool calls use JSON schema constraints. Display uses fixed verbs. No free-form generation for system operations.
4. **Prefill speed is the constraint.** Multi-step tool chains re-ingest growing context. Generation speed is solved; prompt processing decides whether this feels like an OS or a ticket queue.

## Hardware Target

AMD Ryzen AI 9 HX 370 (GPD Pocket 4):
- GPU: Radeon 890M (gfx1150) — Vulkan compute for LLM inference
- CPU: Zen 5, 12 cores — Whisper + TTS
- NPU: XDNA 2 — parked until Linux inference stack matures (amdxdna driver mainlined, ONNX RT EP not ready)
- RAM: 32GB DDR5-5600 shared

## Status

Concept stage. Next step: benchmark Qwen3-4B dense Q6 prefill speed and tool-call accuracy on Vulkan.

## License

MIT
