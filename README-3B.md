---
language:
- en
license: llama3.2
base_model: unsloth/Llama-3.2-3B-Instruct
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

# Yuli-LLaMA-3.2-Reddit-3B

> **Same Yuli. Bigger brain. Chattier mouth.**
>
> The 1B is Mobile. This is Desktop.

Yuli is an AI character / personality engine. This 3B model is her Desktop brain: fine-tuned on the same dyadic multi-turn Reddit dataset as the 1B, with ~2.6x the parameters to spend on fluency, follow-ups, and staying in character longer.

If the 1B answers you, the 3B *talks* to you. That chattiness is real — and it's a feature to steer, not just a stat to brag about.

**Base:** Meta Llama 3.2 3B Instruct, via Unsloth
**Built by:** [@hoytnix](https://huggingface.co/economyofdreams)
**Project:** Yuli — originally built for an MSc in AI, now the personality engine behind Hecka Dough and the Yuli SDK

> File sizes below are the actual sizes shown on HuggingFace
> (economyofdreams/Yuli-LLaMA-3.2-Reddit-3B, branch main, checked Oct 24, 2026).
> Repo total: 8.02 GB, 6 files, no subfolders. No README.md was present at that check.

---

## What makes this one different from the 1B

Same dataset. Same format. Same Yuli voice. Different headroom:

- **3.213B params, 28 layers** (vs 1.236B, 16 layers for 1B) — more capacity for multi-turn tracking, richer vocabulary, and longer, more natural replies
- **Chattier by design, and by observation.** User test, Oct 24 evening: 3B was "much chattier in the test than 1B." That's expected at this size — and it means prompt / stop discipline matters more here, not less. A chatty model that rambles past `<eot_id|>` or past the structured format is not a better Yuli, it's a louder one.
- **Desktop, not Mobile.** The 1B fits anywhere. The 3B is for machines with room to breathe — desktop, laptop with a decent GPU, or a server. On a phone, stick with the 1B Q4_K_M.
- **No 2GB discipline here.** That limit was for the 1B Mobile model only. For 3B Desktop, Q4_K_M is the recommended quant, and Q8_0 is also shipped. See Quants below.

## The dataset — same clean set as the 1B

**Source:** `ZhankuiHe/reddit_movie_large_v1` (train split, raw text — NOT the processed tt-codes)

**Dyadic rule, enforced hard:**
- Exactly 2 distinct `user_id` values per conversation. Role (`USER` / `SYSTEM`, seeker / recommender) is NOT identity.
- Per-comment dedup: one `turn_id` can sit on many paths. First conversation wins, later conversations containing it are dropped whole.
- Depth filter: minimum turn_order 3, minimum 4 turns total, minimum 2 turns per speaker.

**Annotation:** `meta/muse-spark-1.3-contributor` via LiteLLM / Meta Model API
- Pass 1: 7,601 conversations, batch 100, high reasoning, 12 workers, 23m 51s
- Double-check (stdlib only, no LLM, 11 hard checks + 5 soft checks): 7,087 passed, 514 flagged
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
| Base | `unsloth/Llama-3.2-3B-Instruct` |
| Method | 4-bit loading + QLoRA (Unsloth FastLanguageModel) — same as 1B |
| Train / Val | 7,203 / 380, split by `conv_id`, 0 overlap — same as 1B |
| Steps / Epochs | 2,700 / 3 |
| Time / Hardware | 28m 53s, NVIDIA A100-SXM4-40GB |
| Loss | **2.275 → 0.316** (final step 2,700: 0.315775, last-100 avg ~0.33, no NaN, no explosion) |
| Loss by epoch | End Epoch 1 (~step 900): ~0.458 — End Epoch 2 (~step 1,800): ~0.391 — End Epoch 3: 0.316 |
| Max seq length | 4,096 (same as 1B) |
| Language | English only |
| Code tokens | Kept — no language filter |
| Chat template | Llama 3.2 native `apply_chat_template`, system stays first-class |
| Response-only training | Loss on assistant tokens only |

## Quants — 2GB limit, and 3B makes it tight

Sizes below are exactly as shown on HuggingFace, branch main, Oct 24, 2026. Repo total 8.02 GB.

| File | Size (as shown on HF) | Ship? |
|---|---|---|
| `Yuli-LLaMA-3.2-Reddit-3B-IQ2_M.gguf` | 1.23 GB | Yes, fits |
| `Yuli-LLaMA-3.2-Reddit-3B-IQ3_XXS.gguf` | 1.35 GB | Yes, fits |
| `Yuli-LLaMA-3.2-Reddit-3B-Q4_K_M.gguf` | **2.02 GB** | **Yes — recommended, tight fit** |
| `Yuli-LLaMA-3.2-Reddit-3B-Q8_0.gguf` | 3.42 GB | **No — exceeds 2GB, but it IS in the repo** |
| `.gitattributes` | 1.81 kB | Repo metadata |
| `Modelfile` | 1.92 kB | Local Ollama use |

Note on the 2GB limit: that limit was for the 1B Mobile model only, NOT for this 3B Desktop model. Q4_K_M at 2.02 GB as displayed is the recommended quant, and Q8_0 at 3.42 GB is fine here — Desktop is not bound by the Mobile 2GB rule. The F16 base (~6.43 GB estimate) is an intermediate only, not shipped here.

## Quickstart

### llama.cpp (recommended)

```bash
llama.cpp/build/bin/llama-server \
  -m Yuli-LLaMA-3.2-Reddit-3B-Q4_K_M.gguf \
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

Keep the stops. With a chattier model, the stops are what keep a good reply from becoming a monologue.

### What NOT to expect

This repo is **GGUF only**. There are no safetensors here, so `AutoModelForCausalLM.from_pretrained()` on this repo will not work as-is. Use llama.cpp / Ollama, or convert the GGUF back if you need a transformers workflow.

## Yuli's voice

Girl-next-door peer. Direct, warm, never sycophantic. No therapist platitudes, no announcing her internal architecture out loud in dialogue.

In dialogue mode, she replies in plain dialogue — no `<thought>` block is expected, and a clean UI should not show one. In annotation mode, she emits `<thought>` + `<state_vector>` + per-turn analysis, and the UI strips those tags before display. Both modes are correct. Confusing them is the only bug.

With 3B, watch the voice, not just the word count: chattier should mean more follow-ups, more texture, more staying in character — not more filler, not repeating the user back, not drifting off-topic.

## Honest limitations

- **Chattier ≠ better, automatically.** The Oct 24 test observation is "much chattier than 1B" — that is a dialogue observation, not a gate result. A 3B that talks more but breaks `<thought>` / `<state_vector>` / turn-coverage / function-table format fails the Yuli annotation job even if it chats prettier.
- **Formal gate evaluation is still pending for both sizes.** Dialogue sanity tests pass for 1B (coherent English, no Chinese, no broken syntax, clean eos stop). The formal gate — 5+ held-out val conversations, vector recomputation, turn coverage, function-table checks, shift-or-hold behaviour, 0 hard failures — has not yet been run and reported as PASS for 1B or 3B. Do not claim a PASS in this README until that report exists.
- **The 2GB limit does not apply here.** That limit was for the 1B Mobile model only. Q8_0 at 3.42 GB is fine for this 3B Desktop model.
- **English only.** Other languages were not filtered out of Reddit, but the model was not trained or evaluated for them.
- **Reddit movie conversations are the training distribution.** General chat works, but personality tracking was validated on dyadic movie-discussion turns, not on clinical, legal, or medical dialogue. Do not use this for diagnosis, hiring, or any high-stakes judgement about a real person.
- **It can be wrong about a type.** A vector that recomputes correctly is not proof the type is psychologically true — it's proof the format is internally consistent.

## Intended use

- Research into shifting personality / cognitive-function patterns in conversation
- Yuli SDK character engines, NPC personalities, and dynamic story in games (see: Hecka Dough)
- Building better, smaller, on-device character models — with 3B as the Desktop step up from the 1B Mobile model

## Out of scope

- Medical / psychological diagnosis
- Employment, credit, or other high-stakes decisions about individuals
- Re-identifying anonymised Reddit speakers
- Anything that needs a guarantee about a person's "true" type

## License / base

Built on Meta Llama 3.2 3B Instruct — the Llama 3.2 Community License applies to the base weights. The Yuli project itself is developed under BSL-1.1 (free until $30K revenue, then $1,500/year, change date September 1, 2030) — check the project repo for the current terms before commercial use.

## Credits

- Michael Scott Hoyt — founder, dataset, training, Yuli theory
- Muse Spark 1.3 (contributor tier) — annotator
- Unsloth, llama.cpp, and the Reddit communities whose public conversations made the dyadic dataset possible

---

*Yuli-LLaMA-3.2-Reddit-3B — same mind, more mouth. Steer the chattiness, don't just admire it.*
