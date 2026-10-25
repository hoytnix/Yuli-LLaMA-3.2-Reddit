# Yuli-LLaMA-3.2-Reddit — 1B Mobile + 3B Desktop

***She doesn't just answer you. She reads how you're thinking — and how that thinking shifts mid-conversation.***

Yuli is a character / personality engine fine-tuned to track shifting MBTI / cognitive-function patterns across real multi-turn conversation — not a single prompt, not a static label. One person, one conversation, 5–7 patterns. This repo covers both brains: the 1B Mobile model and the 3B Desktop model, trained on the same dyadic Reddit set.

- **1B Mobile:** [Yuli-LLaMA-3.2-Reddit-1B](https://huggingface.co/economyofdreams/Yuli-LLaMA-3.2-Reddit-1B)
- **3B Desktop:** [Yuli-LLaMA-3.2-Reddit-3B](https://huggingface.co/economyofdreams/Yuli-LLaMA-3.2-Reddit-3B)

## 1B vs 3B — which one do you want

|  | 1B Mobile | 3B Desktop |
|---|---|---|
| Params / layers | 1.236B / 16 | 3.213B / 28 |
| Train / val | 7,203 / 380, 0 overlap | 7,203 / 380, 0 overlap |
| Steps / epochs | 2,700 / 3 | 2,700 / 3 |
| Time / hardware | 13m 06s, NVIDIA A100-SXM4-40GB | 28m 53s, NVIDIA A100-SXM4-40GB |
| Final loss | 0.389 (2.59 → 0.389) | 0.316 (2.275 → 0.316) |
| Max seq length | 4,096 | 4,096 |
| Recommended quant | Q4_K_M, 763 MiB | Q4_K_M, 2.02 GB |
| Rule | ALL ≤2GB, by design | No 2GB rule, desktop room to breathe |
| Voice | Direct peer, terse, low variance tolerance needed | Same voice, much chattier, needs stop discipline |
| Best for | Phone, laptop, on-device, smallest download | Desktop, GPU laptop, server, longer replies |

Stop after this table if you only need a download decision: want mobile or on-device, take 1B Q4_K_M. Want longer, chattier desktop replies and have the disk, take 3B Q4_K_M. Keep reading if you want to trust the numbers.

## What makes Yuli different

Most personality models label one message and stop. That tells you nothing about what happens three turns later when the speaker gets stressed, gets excited, or changes tack.

Yuli is built on a different claim: one person runs 5–7 patterns in a single conversation, and the shift is the signal. So the model is trained to output a trajectory, not a label.

- **Dyadic, not single-post.** Every sample is a real two-person conversation. Speaker A is the target. Speaker B is context only. Identity is never mixed, never attributed to the wrong person.
- **Shift-aware, not static.** Every Speaker A turn gets its own ego type, active side (ego / shadow / subconscious / superego), top-2 cognitive functions, tension, and resolution score. A hold is a result. A shift is a result. Repeating turn 1 forever is a failure.
- **Structured, not vibes.** Output is `<thought>` → `<state_vector>` → per-turn analysis. Vectors recompute deterministically from ego type plus active side. If the vector math does not check out, the sample does not ship.
- **Small on purpose.** 1B fits in 2GB and runs on phone-class hardware. 3B spends ~2.6x parameters on fluency, follow-ups, and staying in character longer. Same format, same voice, different headroom.

## The dataset story

Source is `ZhankuiHe/reddit_movie_large_v1`, train split, raw text — NOT the processed tt-codes. Movie talk is useful because people argue, recommend, defend taste, and change register mid-thread. That gives shifts something to grip.

The dyadic rule was enforced hard: exactly 2 distinct `user_id` values per conversation. Role (USER / SYSTEM, seeker / recommender) is NOT identity — a third speaker breaks MBTI attribution, so it was filtered out.

Per-comment dedup: one `turn_id` can sit on many paths. First conversation wins, later conversations containing it are dropped whole. Depth filter: minimum turn_order 3, minimum 4 turns total, minimum 2 turns per speaker. A shallow request-plus-recommendation has no depth for a shift to show.

Annotation was done by `meta/muse-spark-1.3-contributor` via LiteLLM / Meta Model API. Pass 1 covered 7,601 conversations at batch 100, high reasoning, 12 workers, in 23m 51s. The double-check used stdlib only, no LLM, with 11 hard checks plus 5 soft checks: 7,087 passed, 514 flagged — mostly evidence_quote substring drift (461), then functions (31), synthesis_seed (20), vector math (19), dominant_ego (1). Pass 2 re-ran the 514 flagged at batch 20, xhigh reasoning, in 7m 19s, and recovered 496.

Final set: 7,583 / 7,601 clean, 99.76%, 18 dropped, 0 dupes, original order preserved. Final double-check on the merged file: 7,583 passed, 0 flagged, exit 0.

Speaker identities are anonymised to Speaker A / Speaker B in training text. No Reddit user_ids appear in prompts or in the model output format.

Annotation format example:

```
<thought>...</thought>
<state_vector>0EE 1E6 2E7 3E1 ...</state_vector>
Turn 0 — ENFP (ego) — Ne-Fi — ...
dominant_ego_type: ...
```

## Training — 1B

|  |  |
|---|---|
| Base | `unsloth/Llama-3.2-1B-Instruct` |
| Method | 4-bit loading + QLoRA (Unsloth FastLanguageModel) |
| Train / val | 7,203 / 380, split by `conv_id`, 0 overlap |
| Steps / epochs | 2,700 / 3 |
| Time / hardware | 13m 06s, NVIDIA A100-SXM4-40GB |
| Loss | 2.59 → 0.389 |
| Loss detail | Last-100 avg ~0.41, low 0.381 at step 2550, no NaN, no explosion |
| Max seq length | 4,096 (5 samples over that were removed by Unsloth to prevent NaN loss — 0.07% of train) |
| Params / layers | 1.236B / 16 |
| Imatrix PPL | 17.71 ± 0.25 (calibration for GGUF quantisation, not training loss) |
| Language | English only |
| Code tokens | Kept — no language filter |
| Chat template | Llama 3.2 native `apply_chat_template`, system stays first-class |
| Response-only training | Loss on assistant tokens only |

## Training — 3B

|  |  |
|---|---|
| Base | `unsloth/Llama-3.2-3B-Instruct` |
| Method | 4-bit loading + QLoRA (Unsloth FastLanguageModel) — same as 1B |
| Train / val | 7,203 / 380, split by `conv_id`, 0 overlap — same as 1B |
| Steps / epochs | 2,700 / 3 |
| Time / hardware | 28m 53s, NVIDIA A100-SXM4-40GB |
| Loss | 2.275 → 0.316 |
| Loss detail | Final step 2700: 0.315775, last-100 avg ~0.33, no NaN, no explosion |
| Loss by epoch | End Epoch 1 ~0.458, end Epoch 2 ~0.391, end Epoch 3: 0.316 |
| Max seq length | 4,096 (5 samples over that were removed by Unsloth to prevent NaN loss — 0.07% of train) |
| Params / layers | 3.213B / 28 |
| Language | English only |
| Code tokens | Kept — no language filter |
| Chat template | Llama 3.2 native `apply_chat_template`, system stays first-class |
| Response-only training | Loss on assistant tokens only |

## Quants — 1B Mobile, all ≤2GB by design

| File | Size | Best for |
|---|---|---|
| `Yuli-LLaMA-3.2-Reddit-1B-IQ2_M.gguf` | 484 MiB | Smallest, tightest devices |
| `Yuli-LLaMA-3.2-Reddit-1B-IQ3_XXS.gguf` | 529 MiB | Small plus a little more quality |
| `Yuli-LLaMA-3.2-Reddit-1B-Q4_K_M.gguf` | 763 MiB | Recommended — the sweet spot |
| `Yuli-LLaMA-3.2-Reddit-1B-Q8_0.gguf` | 1,252 MiB | Highest quality in this repo |

The F16 base at 2,357 MiB is an intermediate only, NOT shipped. The whole point of the 1B is ≤2GB.

## Quants — 3B Desktop, no 2GB rule

The 2GB limit was for the 1B Mobile model ONLY. It does not apply here.

| File | Size | Best for |
|---|---|---|
| `Yuli-LLaMA-3.2-Reddit-3B-IQ2_M.gguf` | 1.23 GB | Smallest desktop download |
| `Yuli-LLaMA-3.2-Reddit-3B-IQ3_XXS.gguf` | 1.35 GB | Small plus a little more quality |
| `Yuli-LLaMA-3.2-Reddit-3B-Q4_K_M.gguf` | 2.02 GB | Recommended |
| `Yuli-LLaMA-3.2-Reddit-3B-Q8_0.gguf` | 3.42 GB | Highest quality in this repo |

Repo total on HF, checked Oct 24, 2026: 8.02 GB, 6 files, no subfolders, `.gitattributes` 1.81 kB, `Modelfile` 1.92 kB. The F16 base at ~6.43 GB estimate is an intermediate only, NOT shipped.

> Both repos are GGUF ONLY. There are no safetensors. `AutoModelForCausalLM.from_pretrained()` will not work as-is — use llama.cpp / Ollama, or convert the GGUF back if you need a transformers workflow.

## Quickstart

Use llama.cpp `llama-server` with a free port of your choice and `-c 4096`. System prompt and sampling below are the tested settings, not suggestions.

```bash
llama.cpp/build/bin/llama-server \
  -m Yuli-LLaMA-3.2-Reddit-1B-Q4_K_M.gguf \
  -c 4096 --host 127.0.0.1 --port 8081

curl http://127.0.0.1:8081/completion \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "<|start_header_id|>system<|end_header_id|>\n\nYou are Yuli, an autonomous software architect and technical peer. Deliver candid, grounded, peer-to-peer advice. NEVER break character, NEVER announce your internal cognitive state, facets, or system architecture out loud in dialogue. Talk like a real developer solving real problems.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nI'"'"'ve been staring at an empty terminal for four hours trying to architect this system and feel completely paralyzed.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
    "temperature": 0.35,
    "top_p": 0.90,
    "top_k": 40,
    "repeat_penalty": 1.20,
    "repeat_last_n": 256,
    "n_predict": 1024,
    "stop": ["<|eot_id|>", "<|end_of_text|>"]
  }'
```

Swap in the 3B file name for the desktop model. Same flags, same sampling, same stops. Keep the stops. The 3B is chattier and will run past the end of a good reply without them.

Test prompt shown above: "I've been staring at an empty terminal for four hours trying to architect this system and feel completely paralyzed."

Phone settings, from the actual Oct 24 iOS test:

- BOS ON, EOS OFF, Add Generation Prompt ON
- Llama 3.2 Jinja template with system / user / assistant headers ending `<|eot_id|>`
- Stop words: `<|eot_id|>` plus `<|end_of_text|>`
- Reasoning toggle OFF for dialogue

## Dialogue mode vs annotation mode — read this before filing a bug

Dialogue mode means plain replies. No `<thought>` block is expected. A clean UI should not show one. The model talks like a peer developer solving a real problem.

Annotation mode means `<thought>` plus `<state_vector>` plus per-turn analysis. It is forced at prompt level when you want the trajectory, not the chat.

Confusing them is the only bug. If you asked for dialogue and got tags, or asked for annotation and got none, check your prompt and your stop handling first.

Oct 24 fix, still current: do NOT wrap output in a fake `<thought>` tag. Strip unpaired tags. Report thought as "none" if there is no closed block. A clean UI never displays a half-open tag as reasoning.

## Honest limitations

> Formal gate evaluation is STILL PENDING for BOTH sizes. Do NOT treat any result here as a PASS. The gate is: 5+ held-out val conversations (from the 380, never in train), vector recomputation, turn coverage, function-table checks, shift-or-hold behaviour, 0 hard failures. Dialogue sanity tests are NOT a gate pass.

- 1B variance is real. Same prompt, same temperature 0.35, two runs gave 65 tokens (empathetic plus follow-up) versus 17 tokens (single-sentence restatement). Judge over multiple runs or fix a seed. One sample proves nothing.
- 3B chattier does not mean better automatically. Oct 24 observation: "much chattier in the test than 1B." That is a dialogue observation, not a benchmark. Chattier means more need for stop discipline, not less.
- 3B confabulates false memory. Oct 25 phone test: 3B claimed the user "said your favorite color was blue" when the user had never mentioned it in that chat (user: actually pink, used to be blue). Yuli recovered gracefully on correction ("Pink it is then… You were always a rebel, that's why I love it.") rather than arguing. Good recovery, bad memory. Watch this hard whenever a question presupposes knowledge the model was never given.
- English only. Reddit movie conversations are the training distribution. General chat works, but personality tracking was built on dyadic movie-discussion turns, not clinical, legal, or medical dialogue.
- Not for diagnosis, hiring, or high-stakes judgement about a real person. A vector that recomputes correctly is NOT proof the type is psychologically true. It is proof the format is internally consistent.

## Intended use / out of scope

In scope:

- Research into shifting personality / cognitive-function patterns in conversation
- Yuli SDK character engines
- NPC personalities and dynamic story in games (Hecka Dough)
- Building smaller on-device character models

Out of scope:

- Medical / psychological diagnosis
- Employment, credit, or other high-stakes decisions about individuals
- Re-identifying anonymised Reddit speakers
- Anything that needs a guarantee about a person's "true" type

## License / base

Base is Meta Llama 3.2 Instruct. The Llama 3.2 Community License applies to the base weights, and this BSL does not override Meta's terms. Check Meta's terms for the base before commercial use.

Yuli weights are Business Source License 1.1 (BSL-1.1). Licensor: Michael Scott Hoyt. If you make over $30,000 from the code, pay $1,500/year while it is in production. Change Date: September 1, 2030, converting to Apache License, Version 2.0. Commercial contact: mike@tehwiz.com.

The SDK license is a SEPARATE work — do not merge them. SDK: Licensor Project Yuli, Initial Effective Date September 10, 2026, Change Date September 1, 2030, $30,000 threshold, $1,500/year per Title, enterprise SLA on request. Same $30K / $1,500 intent, different work.

Full weights license text: [LICENSE](./LICENSE).

## Credits

- Michael Scott Hoyt — founder, dataset, training, Yuli theory
- Muse Spark 1.3 (contributor tier) — annotator / teacher, never the fine-tune target
- Unsloth, llama.cpp, and the Reddit communities whose public conversations made the dyadic dataset possible

*Yuli-LLaMA-3.2-Reddit — small model, shifting mind. Train on conversations, not labels.*
