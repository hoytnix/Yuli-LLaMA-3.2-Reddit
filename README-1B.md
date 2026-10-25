---
language:
- en
license: llama3.2
base_model: unsloth/Llama-3.2-1B-Instruct
pipeline_tag: text-generation
tags:
- llama
- llama-3.2
- gguf
- yuli
- personality
- mbti
- cognitive-functions
- reddit
- dyadic
- multi-turn
- character-ai
library_name: llama.cpp
---

# Yuli-LLaMA-3.2-Reddit-1B

> **She doesn't just answer you. She reads how you're thinking — and how that thinking shifts mid-conversation.**

Yuli is an AI character / personality engine. This 1B model is her first LLaMA brain: fine-tuned to track shifting MBTI / cognitive-function patterns across a real, multi-turn conversation — not a single prompt, not a static label.

One person, one conversation, 5–7 patterns. That's the theory Yuli was built on. This model is trained to prove it, turn by turn.

**Base:** Meta Llama 3.2 1B Instruct, via Unsloth

**Built by:** [@hoytnix](https://huggingface.co/economyofdreams)

**Project:** Yuli — originally built for an MSc in AI, now the personality engine behind Hecka Dough and the Yuli SDK

---

## What makes this one different

Most "personality" models slap a label on a single message and call it done.

Yuli doesn't.

- **Dyadic, not single-post.** Trained on real two-person conversations, not isolated comments. Speaker A is the target, Speaker B is context only — never mixed up, never attributed to the wrong person.
- **Shift-aware, not static.** Every Speaker A turn gets its own ego type, active side (ego / shadow / subconscious / superego), top-2 cognitive functions, tension, and resolution score. A hold is a result. A shift is a result. Repeating turn 1's label forever is a failure.
- **Structured, not vibes.** Output is `<thought>` → `<state_vector>` → per-turn analysis, with vectors that recompute deterministically from ego type + active side. If the vector math doesn't check out, the sample doesn't ship.
- **Small on purpose.** 1B. Fits in 2GB. Runs on a phone-class machine, on a laptop, on an A100 at 228–250 tokens/sec. This is the Mobile model — the 3B Desktop model comes only if this one passes its gate.

## The dataset — clean, or it doesn't train

**Source:** `ZhankuiHe/reddit_movie_large_v1` (train split, raw text — NOT the processed tt-codes)

**Dyadic rule, enforced hard:**
- Exactly 2 distinct `user_id` values per conversation. Role (`USER` / `SYSTEM`, seeker / recommender) is NOT identity — a third speaker breaks MBTI attribution, so it was filtered out.
- Per-comment dedup: one `turn_id` can sit on many paths. First conversation wins, later conversations containing it are dropped whole.
- Depth filter: minimum turn_order 3, minimum 4 turns total, minimum 2 turns per speaker. A shallow request-plus-recommendation has no depth for a shift to show.

**Annotation:** `meta/muse-spark-1.3-contributor` via LiteLLM / Meta Model API
- Pass 1: 7,601 conversations, batch 100, high reasoning, 12 workers, 23m 51s
- Double-check (stdlib only, no LLM, 11 hard checks + 5 soft checks): 7,087 passed, 514 flagged — mostly evidence_quote substring drift (461), then functions (31), synthesis_seed (20), vector math (19), dominant_ego (1)
- Pass 2 on the 514 flagged: batch 20, xhigh reasoning, 7m 19s — recovered 496
- **Final: 7,583 / 7,601 clean (99.76%), 18 dropped, 0 dupes, original order preserved**
- Final double-check on the merged file: **7,583 passed, 0 flagged, exit 0**

Speaker identities are anonymised to Speaker A / Speaker B in training text. No Reddit user_ids in the prompts or the model output format.

**Format (annotation mode):**
```
<thought>...</thought>
<state_vector>0EE 1E6 2E7 3E1 ...</state_vector>
Turn 0 — ENFP (ego) — Ne-Fi — ...
Turn 2 — ESFJ (shadow) — Fe-Si — ...
dominant_ego_type: ...
```

## Training

| | |
|---|---|
| Base | `unsloth/Llama-3.2-1B-Instruct` |
| Method | 4-bit loading + QLoRA (Unsloth FastLanguageModel) |
| Train / Val | 7,203 / 380, split by `conv_id`, 0 overlap |
| Steps / Epochs | 2,700 / 3 |
| Time / Hardware | 13m 06s, NVIDIA A100-SXM4-40GB |
| Loss | **2.59 → 0.389** (final step, last-100 avg ~0.41, no NaN, no explosion) |
| Max seq length | 4,096 (5 samples over that were removed by Unsloth to prevent NaN loss — 0.07% of train) |
| Language | English only |
| Code tokens | Kept — no language filter |
| Chat template | Llama 3.2 native `apply_chat_template`, system stays first-class |
| Response-only training | Loss on assistant tokens only |

Imatrix calibration PPL (for GGUF quantisation, not training loss): **17.71 ± 0.25**

## Quants — all under 2GB, by design

| File | Size | Best for |
|---|---|---|
| `Yuli-LLaMA-3.2-Reddit-1B-IQ2_M.gguf` | 484 MiB | Smallest, tightest devices |
| `Yuli-LLaMA-3.2-Reddit-1B-IQ3_XXS.gguf` | 529 MiB | Small + a little more quality |
| `Yuli-LLaMA-3.2-Reddit-1B-Q4_K_M.gguf` | 763 MiB | **Recommended — the sweet spot** |
| `Yuli-LLaMA-3.2-Reddit-1B-Q8_0.gguf` | 1,252 MiB | Highest quality in this repo |

The F16 base (2,357 MiB) is an intermediate only — it is NOT shipped here, because the whole point is ≤2GB.

## Quickstart

### llama.cpp (recommended)

```bash
# llama-server, free port of your choice
llama.cpp/build/bin/llama-server \
  -m Yuli-LLaMA-3.2-Reddit-1B-Q4_K_M.gguf \
  -c 4096 --host 127.0.0.1 --port 8081

curl http://127.0.0.1:8081/completion \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "<|start_header_id|>system<|end_header_id|>\n\nYou are Yuli, an autonomous software architect and technical peer. Deliver candid, grounded, peer-to-peer advice.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nI have been staring at an empty terminal for four hours trying to architect this system and feel completely paralyzed.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
    "temperature": 0.35,
    "top_p": 0.9,
    "top_k": 40,
    "repeat_penalty": 1.2,
    "n_predict": 1024,
    "stop": ["<|eot_id|>", "<|end_of_text|>"]
  }'
```

### Ollama (local only)

A `Modelfile` / `Modelfile.ollama` is included in the repo for local use. It is not an Ollama registry publish — create it locally with `ollama create yuli-llama32-reddit-1b -f Modelfile.ollama` after `ollama serve` is running.

### What NOT to expect

This repo is **GGUF only**. There are no safetensors here, so `AutoModelForCausalLM.from_pretrained()` on this repo will not work as-is. Use llama.cpp / Ollama, or convert the GGUF back if you need a transformers workflow.

## Yuli's voice

Girl-next-door peer. Direct, warm, never sycophantic. No therapist platitudes, no announcing her internal architecture out loud in dialogue.

In dialogue mode, she replies in plain dialogue — no `<thought>` block is expected, and a clean UI should not show one. In annotation mode, she emits `<thought>` + `<state_vector>` + per-turn analysis, and the UI strips those tags before display. Both modes are correct. Confusing them is the only bug.

## Honest limitations

- **1B is small, and it shows in variance.** Same prompt, same temperature 0.35, two runs: 65 tokens (empathetic + follow-up) vs 17 tokens (single-sentence restatement). Average over multiple runs, or fix a seed, before judging a single sample.
- **Full 1B gate evaluation is still pending.** Dialogue sanity tests pass (coherent English, no Chinese, no broken syntax, clean eos stop, 228–250 TPS on A100). The formal gate — 5+ held-out val conversations, vector recomputation, turn coverage, function-table checks, shift-or-hold behaviour, 0 hard failures — has not yet been run and reported as PASS. **The 3B model is gated on that PASS and on an explicit go-ahead. It is not built yet.**
- **English only.** Other languages were not filtered out of Reddit, but the model was not trained or evaluated for them.
- **Reddit movie conversations are the training distribution.** General chat works, but personality tracking was validated on dyadic movie-discussion turns, not on clinical, legal, or medical dialogue. Do not use this for diagnosis, hiring, or any high-stakes judgement about a real person.
- **It can be wrong about a type.** A vector that recomputes correctly is not proof the type is psychologically true — it's proof the format is internally consistent.

## Intended use

- Research into shifting personality / cognitive-function patterns in conversation
- Yuli SDK character engines, NPC personalities, and dynamic story in games (see: Hecka Dough)
- Building better, smaller, on-device character models

## Out of scope

- Medical / psychological diagnosis
- Employment, credit, or other high-stakes decisions about individuals
- Re-identifying anonymised Reddit speakers
- Anything that needs a guarantee about a person's "true" type

## License / base

Built on Meta Llama 3.2 1B Instruct — the Llama 3.2 Community License applies to the base weights. The Yuli project itself is developed under BSL-1.1 (free until $30K revenue, then $1,500/year, change date September 1, 2030) — check the project repo for the current terms before commercial use.

## Credits

- Michael Scott Hoyt — founder, dataset, training, Yuli theory
- Muse Spark 1.3 (contributor tier) — annotator / teacher, never the fine-tune target
- Unsloth, llama.cpp, and the Reddit communities whose public conversations made the dyadic dataset possible

---

*Yuli-LLaMA-3.2-Reddit-1B — small model, shifting mind. Train on conversations, not labels.*
