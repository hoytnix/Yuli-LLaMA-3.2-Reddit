"""
Annotate DYADIC multi-turn conversations for Yuli with
Meta Muse Spark 1.3, contributor tier, via LiteLLM / Meta Model API.

Model: meta/muse-spark-1.3-contributor
  LiteLLM day-0 route for Muse Spark 1.3. Contributor pricing is
  $0.10 / $0.20 per 1M input / output tokens, 100 RPM per team,
  vs $1.25 / $4.25 standard. Contributor terms permit Meta to use
  prompts and completions for training -- acceptable here for public
  Reddit conversations, not for private / sensitive data.

API key: NEVER hard-coded in this file.
  Pass it as an option:  --api-key <key>
  Or export it: MODEL_API_KEY (Meta Model API name) or META_API_KEY
  (headless Muse Code name). The script checks --api-key first, then
  those env vars, in that order, and refuses to run with no key.
  Do not paste a key into this script, and do not commit a key.


Input:  reddit_movie_dyadic.jsonl, from extract_reddit_movie_dyadic.py
        One record per conversation, with a `turns` list sorted by
        turn_order, speakers A (seeker, target) and B (other participant).
Output: unsloth_4sides_dyadic.jsonl, one record per conversation.

What changed vs the old single-post batch_annotate_4sides.py:
  1. One Muse Spark call per CONVERSATION, not per post. The old script took
     one Reddit post and made one system/user/assistant sample out of it.
     That is single-turn, and it cannot teach Yuli to track a person
     shifting mid-conversation.
  2. The transcript is the whole dyadic conversation, in turn order,
     anonymised to Speaker A / Speaker B. Reddit user_ids (t2_...) are
     never put in the prompt or the training text.
  3. The analysis tracks Speaker A (the seeker) turn by turn. Speaker B
     is context only: the model must NOT attribute B's language to A,
     and must NOT merge the two speakers into one type. That is the
     whole point of the dyadic filter upstream.
  4. For each Speaker A turn the model returns ego_type, active_side,
     the deterministic 3-char hex vector [Side][Ego][Active], top-2
     ACTIVE functions, a verbatim evidence quote, tension, limbic
     anchor (S/T/A/neutral), resolution R, a rationale, and a shift
     note vs the previous A turn (same type / shifted, and what
     triggered it; short turns carry the prior label). Dominant type,
     overall R, an optional synthesis seed (->1 hook when R >= 0.75),
     and an optional archetype hint complete the sample. The training
     sample therefore shows a *trajectory*, not a single static label.
  5. Bug fixed from the old script: the FALLBACK_MODELS list was
     missing a comma after "gemini/gemini-3.1-pro", which silently
     concatenated the next model name into
     "gemini/gemini-3.1-progemini/gemini-3.1-flash-lite".
     That broken name is gone here.

Training sample format (Unsloth / chat SFT ready):
  conversations: [
    {role: system, content: "You are Yuli, ..."},
    {role: user, content: "Speaker A (turn 0): ...\nSpeaker B (turn 1): ..."},
    {role: assistant, content: "<thought>...</thought>\n\n"
        "<state_vector>...</state_vector>\n"
        "Turn 0 - Speaker A: Ego ..., Side ..., Vector ...\n"
        "  Functions: ...\n  Tension: ... | Limbic: ... | R: ...\n"
        '  Evidence: "..."\n  Shift: ...\n  Breakdown: ...\n...'}
  ]
  plus audit fields outside `conversations`: conv_id, turn_count,
  target_speaker, and the structured per-turn analysis. Trainers that
  only read `conversations` ignore the extra fields.

TRUE BATCHING, not one-call-per-conversation:
  --batch-size N packs N full transcripts into ONE Muse call.
  Default 25. Use --batch-size 50 for a bigger pack; both fit easily
  in Muse's 1,048,576-token context (a typical conversation here is
  ~600-900 input tokens + ~400-600 output tokens, so 25 packed is
  ~20k in / ~12k out, 50 packed is ~40k in / ~25k out).
  The speedup is NOT 25x: Muse still has to generate all N analyses,
  so output tokens dominate. Expect roughly 5-10x fewer calls and
  much less system-prompt / round-trip overhead, not 25x less time.
  Quality guard: each conversation in a batch is validated separately
  (conv_id + exact Speaker A turn_order coverage). Any conversation
  that fails validation in a batch is automatically retried as a
  single-conversation call, so one bad item does not lose the batch.

Usage (Colab):
    pip install litellm pydantic
    # Option A, preferred for a run:
    python batch_annotate_4sides_dyadic.py \
        --api-key "$MODEL_API_KEY" \
        --input-file reddit_movie_dyadic.jsonl \
        --output-file unsloth_4sides_dyadic.jsonl \
        --min-turn-order 3 --batch-size 25
    # Option B, env var, no key on the command line history in Colab:
    #   export MODEL_API_KEY=...   (or META_API_KEY)
    #   python batch_annotate_4sides_dyadic.py \
    #       --input-file reddit_movie_dyadic.jsonl \
    #       --output-file unsloth_4sides_dyadic.jsonl \
    #       --min-turn-order 3 --batch-size 25
    # NOTE: --min-turn-order must match the extractor. A 4-turn extract
    # has max turn_order 3, so --min-turn-order 3, NOT the default 4,
    # or annotate.py will skip everything.
"""

import argparse
import ast
import asyncio
import concurrent.futures
import json
import os
import re
import signal
import sys
import time
from typing import Any, Dict, List, Set

from pydantic import BaseModel, Field

# Lazy LiteLLM import, so --help works before `pip install litellm`.
# The actual annotation call fails clearly if litellm is missing.
try:
    import litellm
    from litellm import acompletion
    litellm.suppress_debug_info = True
except ImportError:  # pragma: no cover - only when litellm not installed
    litellm = None
    acompletion = None

# Single model, by request. No Gemini fallback cascade anymore.
# LiteLLM route for Meta Model API; the "meta/" prefix is required.
# If this call is rate-limited, the script quarantines it for 5 minutes
# and retries the SAME model, it does NOT silently switch tiers.
# (Switching to standard muse-spark-1.3 would change both price and
#  data-use terms, so that switch must be an explicit --model choice.)
DEFAULT_MODEL = "meta/muse-spark-1.3-contributor"

MODEL_COOLDOWNS: Dict[str, float] = {}
COOLDOWN_DURATION = 300.0  # 5 minutes

# MBTI 4-bit hypercube mapping (unchanged from the old script).
TYPE_TO_BITS = {
    "ISTP": 0x0, "ISTJ": 0x1, "ISFP": 0x2, "ISFJ": 0x3,
    "INTP": 0x4, "INTJ": 0x5, "INFP": 0x6, "INFJ": 0x7,
    "ESTP": 0x8, "ESTJ": 0x9, "ESFP": 0xA, "ESFJ": 0xB,
    "ENTP": 0xC, "ENTJ": 0xD, "ENFP": 0xE, "ENFJ": 0xF,
}
SIDE_OFFSETS = {"ego": 0, "shadow": 1, "subconscious": 2, "superego": 3}


def compute_expected_hex(ego_type: str, active_side: str) -> str:
    ego_val = TYPE_TO_BITS[ego_type.upper()]
    side_idx = SIDE_OFFSETS[active_side.lower()]
    if side_idx == 0:
        active_val = ego_val
    elif side_idx == 1:
        active_val = ego_val ^ 0x8  # Shadow
    elif side_idx == 2:
        active_val = ego_val ^ 0x9  # Subconscious
    else:
        active_val = ego_val ^ 0xF  # Superego
    return f"{side_idx:X}{ego_val:X}{active_val:X}"


VALID_EGO_TYPES = ("ISTP", "ISTJ", "ISFP", "ISFJ", "INTP", "INTJ",
                   "INFP", "INFJ", "ESTP", "ESTJ", "ESFP", "ESFJ",
                   "ENTP", "ENTJ", "ENFP", "ENFJ")
VALID_SIDES = ("ego", "shadow", "subconscious", "superego")
VALID_TENSION = ("low", "medium", "high")
# Limbic anchor per turn. NOTE: limbic "A" means Arousal, never Speaker A.
VALID_LIMBIC = ("S", "T", "A", "neutral")


class TurnAnalysis(BaseModel):
    turn_order: int = Field(description="turn_order of this Speaker A turn")
    ego_type: str = Field(description="Speaker A's ego MBTI type at THIS turn, "
        "one of the 16 types; short turns carry the prior turn forward")
    active_side: str = Field(description="Active quadrant at THIS turn: "
        "ego, shadow, subconscious, or superego")
    cognitive_vector: str = Field(description="3-character hex vector [Side][Ego][Active]; "
        "recomputed downstream via XOR, so emit your best-effort value")
    primary_functions_active: List[str] = Field(description="Top-2 functions of the ACTIVE "
        "type nibble only, from Ne Ni Se Si Te Ti Fe Fi "
        "(e.g. active ESFJ -> Fe, Si; active INFP -> Fi, Ne)")
    evidence_quote: str = Field(default="",
        description="Verbatim substring of THIS Speaker A turn, max 15 words, "
        "that justifies the label; empty only if the turn itself is empty")
    tension: str = Field(default="low",
        description="Interpersonal friction at this turn: low (banter/flow), "
        "medium (disagreement/stress), high (insult, flooding, boundary clash)")
    limbic_state: str = Field(default="neutral",
        description="Biological anchor: S safe/nurtured, T threatened/defensive, "
        "A aroused/activated, neutral flat info-seeking. "
        "Limbic A is arousal, never Speaker A.")
    resolution_r: float = Field(default=0.5,
        description="Conversational resolution 0.0 (open friction) to 1.0 "
        "(fully resolved) at this turn")
    shift_from_previous: str = Field(
        description="How this turn differs from Speaker A's previous turn, "
        "max 30 words: same type/side with 'no trigger, baseline holds', "
        "or shifted with the exact trigger quoted or named")
    psychological_rationale: str = Field(description="Final analytical synthesis "
        "for this turn, max 60 words, peer-direct tone, no flattery")


class ConversationAnalysis(BaseModel):
    target_speaker: str = Field(description="Always 'A' in this script")
    dominant_ego_type: str = Field(description="Speaker A's most frequent ego type "
        "across the conversation; ties break toward the earliest A turn")
    turns: List[TurnAnalysis] = Field(description="One entry per Speaker A turn, in turn order")
    chain_of_thought: str = Field(
        description="Step-by-step deduction, max 250 words, tracing Speaker A's "
        "function stack, triggers, stress axes, and side transitions across "
        "turns with turn numbers cited. Speaker B is context only and must "
        "not be given an ego type here. Peer-direct tone, no flattery.")
    conversation_resolution_r: float = Field(default=0.5,
        description="Overall resolution 0.0-1.0 at the final Speaker A turn")
    synthesis_seed: str = Field(default="",
        description="If resolution >= 0.75, one forward-tension hook, max 25 words, "
        "an open question or tangent sustaining momentum without sycophancy; "
        "else empty string with the open friction named in chain_of_thought")
    archetype_hint: str = Field(default="",
        description="Optional single resonance from the archetype library "
        "(e.g. Validation Seeker, Lonely Romantic, Corporate Savior, Debater, "
        "Silent Architect, Efficiency Expert, Protective Hero, Wounded Healer, "
        "High-Roller, Aesthetic Collector, Traditionalist, Lurker, Mid-Life "
        "Crisis, Nice-Guy Rage, Existential Dread, Hedonistic Spiral); else empty")


class BatchItem(BaseModel):
    """One conversation's analysis inside a packed batch reply."""
    conv_id: str = Field(description="Exact conv_id from the prompt, copied verbatim")
    target_speaker: str = Field(description="Always 'A' in this script")
    dominant_ego_type: str = Field(description="Speaker A's most frequent ego type "
        "across the conversation; ties break toward the earliest A turn")
    turns: List[TurnAnalysis] = Field(description="One entry per Speaker A turn, in turn order")
    chain_of_thought: str = Field(description="Deduction for this conversation only, "
        "max 250 words, turn numbers cited")
    conversation_resolution_r: float = Field(default=0.5,
        description="Overall resolution 0.0-1.0 at the final Speaker A turn")
    synthesis_seed: str = Field(default="",
        description="Forward-tension hook if resolution >= 0.75, else empty string")
    archetype_hint: str = Field(default="",
        description="Optional single archetype-library resonance, else empty")


class BatchAnalysis(BaseModel):
    items: List[BatchItem] = Field(
        description="One item per conversation in the prompt, in the same order, "
        "with conv_id copied exactly. Never merge conversations, never drop one.")


SYSTEM_PROMPT = """You are an expert cognitive psychometrician specializing in Jungian Analytical Psychology, MBTI, and John Beebe's 4-Sides-of-the-Mind framework (Ego, Shadow, Subconscious, Superego).

You annotate ONE dyadic conversation to produce training data for Yuli, an edge-native peer persona (LLaMA 3.2 1B/3B fine-tune) that tracks ONE speaker's cognitive state across turns. Precision and consistency matter more than cleverness: a 1B model learns whatever regularities you emit.

You are given ONE conversation between exactly two people: Speaker A (the seeker, your ONLY target) and Speaker B (context only), in turn order.

A. SPEAKER ISOLATION (violating this ruins the sample):
1. Track ONLY Speaker A. Never attribute Speaker B's wording, tone, movie titles, or arguments to A. If B suggests something A merely quotes back to reject, that is still A's turn, judged on A's stance toward it, not B's content.
2. Trap to avoid: when A quotes B (e.g. with ">" or "you said"), score A's reaction (accept / question / reject), never B's original claim.
3. No entries for Speaker B turns. No ego type for B anywhere, including chain_of_thought.

B. PER-TURN LABELS (one entry per Speaker A turn, in turn order):
For EACH Speaker A turn emit: ego_type, active_side, cognitive_vector, primary_functions_active (top-2 of the ACTIVE nibble only), evidence_quote (verbatim <=15-word substring of THIS A turn), tension (low/medium/high), limbic_state (S/T/A/neutral; limbic A means Arousal, never Speaker A), resolution_r (0.0 open friction to 1.0 resolved), shift_from_previous (<=30 words), psychological_rationale (<=60 words).

C. DYNAMICS:
1. Inertia first: a person does NOT change type every turn. Short A turns (<~8 words, e.g. "Ahm, are you sure?") carry the prior turn's type/side forward unless they contain an explicit insult, boundary, flooding, or topic rupture. First A turn sets the baseline; its shift reads "baseline".
2. Shift only on a named trigger: a B provocation, a rejected suggestion, mounting frustration, a boundary assertion, visible calming. Quote or name it. "No trigger, baseline holds" is a complete and preferred answer when nothing changed.
3. Tension scale: low = relaxed banter, cooperative flow, polite info exchange; medium = disagreement, correction, exasperation, firm restating; high = insult, contempt, flooding, ultimatum, walk-away.
4. Limbic anchor: S = feels safe, cared for, receptive; T = threatened, defensive, snapping back; A = activated/aroused (heated argument energy OR eager excitement); neutral = flat info-seeking with no charge.
5. resolution_r tracks whether THIS turn leaves things settled (>=0.75: thanks, agreement, warm close) or open (<0.75: restated disagreement, unanswered question, polite but firm refusal).
6. dominant_ego_type is A's most frequent ego type; ties break toward the earliest A turn.
7. conversation_resolution_r equals the final A turn's resolution_r. If it is >= 0.75, add synthesis_seed: one forward hook (open question or tangent, <=25 words) that sustains momentum WITHOUT flattery or "let me know if you need anything else" platitudes. If < 0.75, synthesis_seed is "" and chain_of_thought names the open friction.
8. archetype_hint is optional: at most one library resonance (Validation Seeker, Lonely Romantic, Corporate Savior, Debater, Silent Architect, Efficiency Expert, Protective Hero, Wounded Healer, High-Roller, Aesthetic Collector, Traditionalist, Lurker, Mid-Life Crisis, Nice-Guy Rage, Existential Dread, Hedonistic Spiral), else "".

D. HEX VECTOR (deterministic, self-check before emitting):
Side nibble: 0=Ego, 1=Shadow, 2=Subconscious, 3=Superego. Math on the ego nibble: Shadow = Ego XOR 8, Subconscious = Ego XOR 9, Superego = Ego XOR F, Ego = unchanged.
Nibbles: ISTP=0, ISTJ=1, ISFP=2, ISFJ=3, INTP=4, INTJ=5, INFP=6, INFJ=7, ESTP=8, ESTJ=9, ESFP=A, ESFJ=B, ENTP=C, ENTJ=D, ENFP=E, ENFJ=F.
Worked checks: ENFP base E -> Ego 0EE, Shadow 1E6 (E^8=6), Subconscious 2E7 (E^9=7), Superego 3E1 (E^F=1). ESFJ base B -> Ego 0BB, Shadow 1B3, Subconscious 2B2, Superego 3B4.
Vectors are recomputed downstream from your ego_type + active_side, so when unsure, get the TYPE and SIDE right and emit your best-effort vector.

E. FUNCTIONS (top-2 of the ACTIVE nibble, from Ne Ni Se Si Te Ti Fe Fi):
ISTP Ti-Se | ISTJ Si-Te | ISFP Fi-Se | ISFJ Si-Fe | INTP Ti-Ne | INTJ Ni-Te | INFP Fi-Ne | INFJ Ni-Fe | ESTP Se-Ti | ESTJ Te-Si | ESFP Se-Fi | ESFJ Fe-Si | ENTP Ne-Ti | ENTJ Te-Ni | ENFP Ne-Fi | ENFJ Fe-Ni.
Example: vector 1E6 means active INFP, so functions are Fi, Ne regardless of ego nibble. Side meaning: Ego = natural flow; Shadow = sharp, suspicious, controlling under attack; Subconscious = vulnerable, calming, de-escalating; Superego = firm boundary, reality check, playful but unsparing firmness.

F. YULI VOICE (how the analysis itself reads):
Girl-next-door peer, direct and warm, never sycophantic. Name flattery, people-pleasing, or boundary collapse in A plainly when you see it; never praise A to keep the peace. No therapist platitudes ("I hope that helps", "great question"). chain_of_thought (<=250 words) cites turn numbers, traces the trajectory (where A started, what B did, where A shifted or held, where A ended), and states the final resolution honestly.

G. OUTPUT DISCIPLINE:
Return strict JSON adhering to the schema, nothing else. Self-check before emitting: turn_orders exactly equal Speaker A's turn_orders (no B turns, none missing, none invented); every evidence_quote is a true substring of its turn; every vector recomputed via section D; tension/limbic/R mutually consistent (high tension cannot pair with R 1.0)."""


BATCH_SYSTEM_PROMPT = """You are an expert cognitive psychometrician specializing in Jungian Analytical Psychology, MBTI, and John Beebe's 4-Sides-of-the-Mind framework (Ego, Shadow, Subconscious, Superego).

You annotate MULTIPLE dyadic conversations in one prompt to produce training data for Yuli, an edge-native peer persona (LLaMA 3.2 1B/3B fine-tune) that tracks ONE speaker's cognitive state across turns. Precision and consistency matter more than cleverness: a 1B model learns whatever regularities you emit.

Conversations are separated by headers like:
=== CONVERSATION <conv_id> ===
... transcript ...
=== END CONVERSATION <conv_id> ===

A. ISOLATION RITUAL (read this first, follow it literally):
1. Finish conversation N completely (ALL its Speaker A turns plus its chain_of_thought) before reading conversation N+1. Never copy a type, function, quote, shift, or seed across headers.
2. Reset your prior completely at each === CONVERSATION === header: the new Speaker A is a stranger. The previous conversation's dominant type tells you nothing here.
3. Within a conversation, track ONLY Speaker A (the seeker). Speaker B is context. Never attribute B's wording, tone, or arguments to A. When A quotes B (">" or "you said"), score A's reaction (accept / question / reject), never B's claim. No B entries, no ego type for B anywhere.

B. PER-TURN LABELS (one entry per Speaker A turn, in turn order):
For EACH Speaker A turn emit: ego_type, active_side, cognitive_vector, primary_functions_active (top-2 of the ACTIVE nibble only), evidence_quote (verbatim <=15-word substring of THIS A turn), tension (low/medium/high), limbic_state (S/T/A/neutral; limbic A means Arousal, never Speaker A), resolution_r (0.0-1.0), shift_from_previous (<=30 words), psychological_rationale (<=60 words).

C. DYNAMICS:
1. Inertia first: no type-flipping turn to turn. Short A turns (<~8 words) carry the prior turn forward absent insult, boundary, flooding, or rupture. First A turn is the baseline ("baseline" shift).
2. Shift only on a named trigger (B provocation, rejected suggestion, mounting frustration, boundary, calming). "No trigger, baseline holds" is preferred when nothing changed.
3. Tension: low = banter/cooperative/polite exchange; medium = disagreement, correction, exasperation, firm restating; high = insult, contempt, flooding, ultimatum.
4. Limbic: S = safe/receptive; T = threatened/defensive; A = activated (heated OR eager); neutral = flat info-seeking.
5. resolution_r: >=0.75 means settled (thanks, agreement, warm close); <0.75 means open (restated refusal, unanswered question).
6. dominant_ego_type is the most frequent ego type; ties break toward the earliest A turn. conversation_resolution_r equals the final A turn's R. If >= 0.75, add synthesis_seed (one forward hook, <=25 words, no flattery, no "let me know if you need anything" platitudes); else "" with the open friction named in chain_of_thought.
7. archetype_hint: at most one library resonance (Validation Seeker, Lonely Romantic, Corporate Savior, Debater, Silent Architect, Efficiency Expert, Protective Hero, Wounded Healer, High-Roller, Aesthetic Collector, Traditionalist, Lurker, Mid-Life Crisis, Nice-Guy Rage, Existential Dread, Hedonistic Spiral), else "".

D. HEX VECTOR (self-check every item before emitting):
Side: 0=Ego, 1=Shadow, 2=Subconscious, 3=Superego. Shadow = Ego XOR 8, Subconscious = Ego XOR 9, Superego = Ego XOR F.
Nibbles: ISTP=0, ISTJ=1, ISFP=2, ISFJ=3, INTP=4, INTJ=5, INFP=6, INFJ=7, ESTP=8, ESTJ=9, ESFP=A, ESFJ=B, ENTP=C, ENTJ=D, ENFP=E, ENFJ=F.
Checks: ENFP E -> 0EE / 1E6 / 2E7 / 3E1. ESFJ B -> 0BB / 1B3 / 2B2 / 3B4. Vectors are recomputed downstream; get TYPE and SIDE right first.

E. FUNCTIONS (top-2 of the ACTIVE nibble, from Ne Ni Se Si Te Ti Fe Fi):
ISTP Ti-Se | ISTJ Si-Te | ISFP Fi-Se | ISFJ Si-Fe | INTP Ti-Ne | INTJ Ni-Te | INFP Fi-Ne | INFJ Ni-Fe | ESTP Se-Ti | ESTJ Te-Si | ESFP Se-Fi | ESFJ Fe-Si | ENTP Ne-Ti | ENTJ Te-Ni | ENFP Ne-Fi | ENFJ Fe-Ni.
Side meaning: Ego = natural flow; Shadow = sharp/suspicious/controlling under attack; Subconscious = vulnerable/calming; Superego = firm boundary, unsparing reality check.

F. YULI VOICE:
Girl-next-door peer, direct and warm, never sycophantic. Name A's flattery, people-pleasing, or boundary collapse plainly; never praise to keep the peace. No therapist platitudes. Each chain_of_thought (<=250 words) cites turn numbers and traces start -> B's moves -> shifts or holds -> honest ending resolution.

G. OUTPUT DISCIPLINE:
Return strict JSON: one `items` entry per conversation, in prompt order, `conv_id` copied EXACTLY. Per item: one `turns` entry per Speaker A turn (no B turns, none missing). If you genuinely cannot analyze a conversation, include it with an empty turns list rather than dropping it; never invent turns. Final self-check per item: orders match A's, quotes are true substrings, vectors recomputed, high tension never pairs with R 1.0."""


_STRINGIFIED_PAIR = re.compile(
    r"""^\[\s*['"](USER|SYSTEM)['"]\s*,\s*['"](.*)['"]\s*\]$""",
    re.DOTALL,
)
INVALID_USER_IDS = {"", "None", "null", "deleted", "[deleted]"}


def clean_text(text: Any) -> str:
    """Defensive clean, same as the fixed extractor.

    The old extractor leaked the HF stringified list into `text`,
    e.g. "['USER', 'Star wars']". If annotate.py is ever run on that
    old file, strip it here too, so Yuli never sees the literal.
    Also strips trailing [removed] / [deleted] markers.
    """
    if not isinstance(text, str):
        return ""
    text = text.strip()
    if text.startswith("[") and ("USER" in text[:20] or "SYSTEM" in text[:20]):
        try:
            parsed = ast.literal_eval(text)
            if isinstance(parsed, (list, tuple)) and len(parsed) >= 2:
                return clean_text(str(parsed[1]))
            if isinstance(parsed, (list, tuple)) and len(parsed) == 1:
                return clean_text(str(parsed[0]))
        except (ValueError, SyntaxError):
            pass
        match = _STRINGIFIED_PAIR.match(text)
        if match:
            return clean_text(match.group(2))
    for prefix in ("USER ", "SYSTEM ", "USER: ", "SYSTEM: "):
        if text.startswith(prefix):
            text = text[len(prefix):].strip()
            break
    for marker in ("[removed]", "[deleted]"):
        if text.endswith(marker):
            text = text[: -len(marker)].strip()
    if text.endswith("']") or text.endswith('"]'):
        text = text[:-2].strip()
    return text


def format_transcript(conversation: Dict[str, Any]) -> str:
    lines = []
    for turn in sorted(conversation["turns"], key=lambda t: t["turn_order"]):
        speaker = turn.get("speaker", "A" if turn.get("is_seeker") else "B")
        lines.append(f"Speaker {speaker} (turn {turn['turn_order']}): {clean_text(turn.get('text', ''))}")
    return "\n\n".join(lines)


def get_processed_ids(output_path: str) -> Set[str]:
    processed = set()
    if os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        record = json.loads(line)
                        if record.get("conv_id"):
                            processed.add(record["conv_id"])
                    except json.JSONDecodeError:
                        continue
    return processed


def _clamp_r(value: Any) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.5


def sanitize_analysis(analysis: Dict[str, Any]) -> str | None:
    """Normalize new label fields in place. Returns an error string, or None if OK.

    Tolerates model variance (case, surrounding whitespace) but rejects
    unknown ego types / sides / dominant types, since those poison the
    deterministic vector and the training text.
    """
    dom = str(analysis.get("dominant_ego_type", "")).upper().strip()
    if dom not in VALID_EGO_TYPES:
        return f"invalid dominant_ego_type {dom!r}"
    analysis["dominant_ego_type"] = dom
    for item in analysis.get("turns", []):
        ego = str(item.get("ego_type", "")).upper().strip()
        side = str(item.get("active_side", "")).lower().strip()
        if ego not in VALID_EGO_TYPES:
            return f"invalid ego_type {ego!r} at turn {item.get('turn_order')}"
        if side not in VALID_SIDES:
            return f"invalid active_side {side!r} at turn {item.get('turn_order')}"
        item["ego_type"] = ego
        item["active_side"] = side
        tension = str(item.get("tension", "low")).lower().strip()
        item["tension"] = tension if tension in VALID_TENSION else "low"
        limbic = str(item.get("limbic_state", "neutral")).strip()
        if limbic.upper() in ("S", "T", "A"):
            limbic = limbic.upper()
        elif limbic.lower() == "neutral":
            limbic = "neutral"
        else:
            limbic = "neutral"
        item["limbic_state"] = limbic
        item["resolution_r"] = _clamp_r(item.get("resolution_r", 0.5))
        funcs = [str(f).strip() for f in item.get("primary_functions_active", [])]
        item["primary_functions_active"] = [f for f in funcs if f]
    analysis["conversation_resolution_r"] = _clamp_r(
        analysis.get("conversation_resolution_r",
                     analysis.get("turns", [{}])[-1].get("resolution_r", 0.5)
                     if analysis.get("turns") else 0.5))
    if not isinstance(analysis.get("synthesis_seed", ""), str):
        analysis["synthesis_seed"] = ""
    if not isinstance(analysis.get("archetype_hint", ""), str):
        analysis["archetype_hint"] = ""
    return None


def build_training_sample(conversation, analysis):
    transcript = format_transcript(conversation)
    per_turn_lines = []
    vectors = []
    for item in analysis["turns"]:
        ego = item["ego_type"].upper()
        side = item["active_side"].lower()
        vector = (compute_expected_hex(ego, side)
                  if ego in TYPE_TO_BITS and side in SIDE_OFFSETS
                  else item.get("cognitive_vector", "000"))
        if str(item.get("cognitive_vector", "")).upper() not in ("", vector):
            print(f"      [i] Vector self-correct for turn {item.get('turn_order')}: "
                  f"model said {item.get('cognitive_vector')!r}, recomputed {vector}.")
        vectors.append(vector)
        tension = item.get("tension", "low")
        limbic = item.get("limbic_state", "neutral")
        r = _clamp_r(item.get("resolution_r", 0.5))
        evidence = item.get("evidence_quote", "")
        per_turn_lines.append(
            f"Turn {item['turn_order']} - Speaker A: "
            f"Ego {ego}, Side {side.title()}, Vector {vector}\n"
            f"  Functions: {', '.join(item.get('primary_functions_active', []))}\n"
            f"  Tension: {tension} | Limbic: {limbic} | R: {r:.2f}\n"
            f"  Evidence: \"{evidence}\"\n"
            f"  Shift: {item.get('shift_from_previous', '')}\n"
            f"  Breakdown: {item.get('psychological_rationale', '')}"
        )
    overall_r = _clamp_r(analysis.get("conversation_resolution_r",
                                     analysis.get("turns", [{}])[-1].get("resolution_r", 0.5)
                                     if analysis.get("turns") else 0.5))
    seed = (analysis.get("synthesis_seed", "") or "").strip()
    archetype = (analysis.get("archetype_hint", "") or "").strip()
    header = (
        f"<thought>\n{analysis.get('chain_of_thought', '').strip()}\n</thought>\n\n"
        f"<state_vector>{' '.join(vectors)}</state_vector>\n"
        f"**Target Speaker:** A (seeker)\n"
        f"**Dominant Ego Type:** {analysis.get('dominant_ego_type', '')}\n"
        f"**Resolution R:** {overall_r:.2f}"
    )
    if archetype:
        header += f"\n**Archetype Resonance:** {archetype}"
    if seed:
        header += f"\n**Synthesis Seed (->1):** {seed}"
    assistant_text = header + "\n\n" + "\n\n".join(per_turn_lines)
    return {
        "id": conversation["conv_id"],
        "conv_id": conversation["conv_id"],
        "thread_id": conversation.get("thread_id", ""),
        "turn_count": len(conversation["turns"]),
        "target_speaker": "A",
        "conversations": [
            {
                "role": "system",
                "content": "You are Yuli, a direct girl-next-door peer who reads "
                "people, not a sycophantic assistant. You diagnose 16-type "
                "MBTI dynamics and 4-Sides-of-the-Mind state transitions "
                "(Ego/Shadow/Subconscious/Superego, hex vector "
                "[Side][Ego][Active]) across a two-person conversation, "
                "tracking Speaker A turn by turn without confusing them "
                "with Speaker B. You name friction plainly, hold "
                "boundaries, and never flatter.",
            },
            {
                "role": "user",
                "content": "Analyze Speaker A's cognitive state and 4-sides "
                "dynamic across this two-person conversation. "
                "Speaker B is context only. For each Speaker A turn give "
                "ego type, side, vector, top-2 active functions, evidence "
                "quote, tension/limbic/R, and shift vs the previous A turn "
                "with its trigger. End with dominant type, overall R, and "
                "a synthesis seed if resolved.\n\n" + transcript,
            },
            {"role": "assistant", "content": assistant_text},
        ],
        "analysis": analysis,
    }


def _parse_json_response(content: str):
    """Parse a structured-output reply, tolerating wrapper text."""
    content = content.strip()
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        start = content.find("{")
        end = content.rfind("}")
        if start >= 0 and end > start:
            return json.loads(content[start:end + 1])
        raise


async def annotate_one(conversation, model_name, api_key, reasoning_effort):
    """Annotate one conversation. Returns a training sample or None."""
    user_prompt = (
        "Perform cognitive diagnosis and 4-sides state extraction for "
        "Speaker A across this two-person conversation. "
        "Speaker B is context only.\n"
        "Per Speaker A turn: ego type, side, hex vector, top-2 ACTIVE "
        "functions, verbatim evidence quote, tension/limbic/R, and shift "
        "vs the previous A turn with its trigger (or 'no trigger, "
        "baseline holds'). Short turns carry the prior label forward. "
        "Self-check vectors via Ego XOR 8/9/F before emitting.\n\n"
        + format_transcript(conversation)
    )
    if acompletion is None:
        raise RuntimeError("litellm is not installed. Run: pip install litellm pydantic")
    cooldown_expiry = MODEL_COOLDOWNS.get(model_name, 0.0)
    if time.time() < cooldown_expiry:
        print(f"   -> [Skipping {model_name}]: cooldown "
              f"({int(cooldown_expiry - time.time())}s remaining)...")
        return None
    print(f"   -> Querying {model_name} "
          f"(reasoning_effort: {reasoning_effort})...")
    try:
        # Muse Spark is a reasoning model. LiteLLM passes reasoning_effort
        # through in OpenAI shape: minimal / low / medium / high / xhigh.
        # There is no Gemini-style thinking-budget token parameter here.
        response = await acompletion(
            model=model_name,
            api_key=api_key,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            response_format=ConversationAnalysis,
            reasoning_effort=reasoning_effort,
        )
        msg = response.choices[0].message
        parsed = _parse_json_response(msg.content)
        # Normalize new label fields; reject unknown types/sides.
        clean_error = sanitize_analysis(parsed)
        if clean_error:
            print(f"      [!] Label validation failed on {model_name}: "
                  f"{clean_error}.")
            MODEL_COOLDOWNS[model_name] = time.time() + COOLDOWN_DURATION
            return None
        # Sanity: analysis must cover Speaker A turns, and only those.
        a_orders = sorted(
            t["turn_order"] for t in conversation["turns"]
            if t.get("speaker", "A" if t.get("is_seeker") else "B") == "A")
        got_orders = sorted(t["turn_order"] for t in parsed.get("turns", []))
        if got_orders != a_orders:
            print(f"      [!] Turn coverage mismatch on {model_name}: "
                  f"expected A turns {a_orders}, got {got_orders}.")
            MODEL_COOLDOWNS[model_name] = time.time() + COOLDOWN_DURATION
            return None
        # CPU-bound sample assembly runs in a worker thread so the
        # event loop stays free for in-flight API calls.
        return await asyncio.to_thread(
            build_training_sample, conversation, parsed)
    except Exception as e:
        MODEL_COOLDOWNS[model_name] = time.time() + COOLDOWN_DURATION
        print(f"      [!] Failure on {model_name}: {e}")
        print("      [!] Quarantining for 5 minutes.")
        return None


async def annotate_batch(conversations, model_name, api_key, reasoning_effort):
    """Annotate N conversations in ONE Muse call.

    Returns (samples, failed_conversations). Samples are validated
    individually: a conversation is only accepted if its item's
    conv_id matches exactly and its Speaker A turn_orders match exactly.
    Failed ones are returned for single-call fallback, not dropped.
    """
    if acompletion is None:
        raise RuntimeError("litellm is not installed. Run: pip install litellm pydantic")
    if len(conversations) == 1:
        sample = await annotate_one(
            conversations[0], model_name, api_key, reasoning_effort)
        return ([sample] if sample else []), ([] if sample else conversations)

    sections = []
    for conv in conversations:
        sections.append(
            f"=== CONVERSATION {conv['conv_id']} ===\n"
            f"{format_transcript(conv)}\n"
            f"=== END CONVERSATION {conv['conv_id']} ===")
    user_prompt = (
        "Perform cognitive diagnosis and 4-sides state extraction for "
        "Speaker A in EACH of the following conversations separately. "
        "Speaker B is context only in each one.\n"
        "Finish one conversation fully before starting the next; never "
        "carry types, quotes, or shifts across headers. Per Speaker A "
        "turn: ego type, side, hex vector, top-2 ACTIVE functions, "
        "verbatim evidence quote, tension/limbic/R, and shift vs the "
        "previous A turn ('no trigger, baseline holds' if unchanged). "
        "Short turns carry the prior label forward.\n\n"
        + "\n\n".join(sections))
    print(f"   -> Querying {model_name} with {len(conversations)} conversations "
          f"in one call (reasoning_effort: {reasoning_effort})...")
    try:
        response = await acompletion(
            model=model_name,
            api_key=api_key,
            messages=[
                {"role": "system", "content": BATCH_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            response_format=BatchAnalysis,
            reasoning_effort=reasoning_effort,
        )
        parsed = _parse_json_response(response.choices[0].message.content)
    except Exception as e:
        print(f"      [!] Batch call failed on {model_name}: {e}")
        print("      [!] Falling back to single calls for this batch.")
        return [], list(conversations)

    by_id = {}
    for item in parsed.get("items", []):
        by_id.setdefault(item.get("conv_id"), item)

    samples, failed = [], []
    for conv in conversations:
        item = by_id.get(conv["conv_id"])
        if item is None:
            print(f"      [!] Missing from batch reply: {conv['conv_id']}")
            failed.append(conv)
            continue
        clean_error = sanitize_analysis(item)
        if clean_error:
            print(f"      [!] Label validation failed in batch for "
                  f"{conv['conv_id']}: {clean_error}. Retrying singly.")
            failed.append(conv)
            continue
        a_orders = sorted(
            t["turn_order"] for t in conv["turns"]
            if t.get("speaker", "A" if t.get("is_seeker") else "B") == "A")
        got_orders = sorted(t["turn_order"] for t in item.get("turns", []))
        if got_orders != a_orders:
            print(f"      [!] Turn coverage mismatch in batch for "
                  f"{conv['conv_id']}: expected A turns {a_orders}, "
                  f"got {got_orders}. Retrying singly.")
            failed.append(conv)
            continue
        samples.append(await asyncio.to_thread(
            build_training_sample, conv, item))
    return samples, failed


def _format_duration(seconds: float) -> str:
    """Format a duration as 45s, 12m 30s, or 2h 15m for the TUI."""
    try:
        s = float(seconds)
    except (TypeError, ValueError):
        return "?"
    if s != s or s < 0:  # NaN or negative
        return "?"
    total = int(s)
    if total < 60:
        return f"{total}s"
    mins, secs = divmod(total, 60)
    if mins < 60:
        return f"{mins}m {secs:02d}s"
    hours, mins = divmod(mins, 60)
    return f"{hours}h {mins:02d}m"


def _format_progress(done: int, total: int, resolved_this_run: int,
                     elapsed_s: float, width: int = 30) -> str:
    """Render the overall progress bar: % done, elapsed, ETA, rate."""
    if total <= 0:
        return "[no work: nothing to annotate]"
    pct = max(0.0, min(100.0, 100.0 * done / total))
    filled = max(0, min(width, int(pct / 100.0 * width)))
    bar = "\u2588" * filled + "\u2591" * (width - filled)
    if elapsed_s > 0 and resolved_this_run > 0:
        rate = resolved_this_run / elapsed_s
        remaining = max(0, total - done)
        eta = remaining / rate if rate > 0 else float("nan")
        eta_str = f"ETA {_format_duration(eta)}"
        rate_str = f"{rate:.2f} conv/s"
    else:
        eta_str = "ETA ?"
        rate_str = "-- conv/s"
    return (f"[{bar}] {pct:5.1f}% ({done}/{total}) | "
            f"elapsed {_format_duration(elapsed_s)} | "
            f"{eta_str} | {rate_str}")


async def main():
    parser = argparse.ArgumentParser(
        description="Annotate dyadic multi-turn conversations with "
        "Meta Muse Spark 1.3 contributor, via LiteLLM / Meta Model API.")
    # API key is an OPTION, never hard-coded. --api-key wins, then env.
    # Meta's Model API name is MODEL_API_KEY; headless Muse Code uses
    # META_API_KEY. Both are checked so either export works.
    parser.add_argument("--api-key", dest="api_key", default=None,
                        help="Meta Model API key, as an option. If omitted, "
                        "MODEL_API_KEY, then META_API_KEY, from the "
                        "environment is used. No key is stored in this file.")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help="LiteLLM model id (default: "
                        "meta/muse-spark-1.3-contributor)")
    parser.add_argument("--input-file", default="reddit_movie_dyadic.jsonl",
                        help="Dyadic conversations from extract script")
    parser.add_argument("--output-file", default="unsloth_4sides_dyadic.jsonl")
    parser.add_argument("--batch-size", type=int, default=25,
                        help="Conversations packed into ONE Muse call "
                        "(default: 25; use 50 for a bigger pack, both fit "
                        "Muse's 1M-token context. Any conversation that "
                        "fails validation in a batch is retried singly.)")
    parser.add_argument("--min-turn-order", type=int, default=4,
                        help="Skip conversations that do not reach this "
                        "turn_order (default: 4, matching the extractor). "
                        "Shallow conversations give Yuli no depth to track "
                        "a shift in, so they are not worth annotating.")
    parser.add_argument("--reasoning-effort", default="high",
                        choices=["minimal", "low", "medium", "high", "xhigh"],
                        help="Muse reasoning effort (default: high; "
                        "xhigh is the current ceiling)")
    parser.add_argument("--interval", type=float, default=1.5,
                        help="Seconds between calls. Contributor limit is "
                        "100 RPM per team, so the old Gemini 20.5s interval "
                        "is wasteful here; 1.5s is a safe default.")
    parser.add_argument("--workers", type=int, default=4,
                        help="Concurrent batch workers sharing one event "
                        "loop (default: 4). Each worker awaits Muse calls "
                        "async while CPU work (sample assembly, JSON) runs "
                        "in threads. Keep workers * interval >= 0.6 to stay "
                        "under 100 RPM (e.g. 4 workers x 1.5s).")
    args = parser.parse_args()

    api_key = (args.api_key
               or os.getenv("MODEL_API_KEY")
               or os.getenv("META_API_KEY")
               or "")
    if not api_key:
        print("[!] Error: No Meta Model API key. Pass --api-key <key>, "
              "or export MODEL_API_KEY (or META_API_KEY). "
              "No key is hard-coded in this script.")
        sys.exit(1)
    # Pass the key only to LiteLLM for this run; do not write it anywhere.
    os.environ["MODEL_API_KEY"] = api_key

    if not os.path.exists(args.input_file):
        print(f"Error: Input file '{args.input_file}' does not exist.")
        return

    with open(args.input_file, "r", encoding="utf-8") as f:
        all_convs = [json.loads(line) for line in f if line.strip()]

    processed = get_processed_ids(args.output_file)
    remaining = []
    for c in all_convs:
        if c.get("conv_id") in processed:
            continue
        turns = c.get("turns", [])
        # Skip deleted-user conversations the old extractor let through.
        if any(str(t.get("user_id", "")) in INVALID_USER_IDS for t in turns):
            continue
        if turns and max(t.get("turn_order", 0) for t in turns) < args.min_turn_order:
            continue
        remaining.append(c)
    print(f"Total conversations in '{args.input_file}': {len(all_convs)}")
    print(f"Already completed in '{args.output_file}': {len(processed)}")
    print(f"Remaining to annotate: {len(remaining)}")

    def chunk_list(data, chunk_size):
        for i in range(0, len(data), chunk_size):
            yield data[i:i + chunk_size]

    batches = list(chunk_list(remaining, max(1, args.batch_size)))
    if args.workers < 1:
        args.workers = 1
    print(f"Workers: {args.workers}, batches: {len(batches)}, "
          f"batch-size: {args.batch_size}, interval: {args.interval}s")

    # ---- Event-driven pipeline: work queue -> workers -> result queue.
    # Workers share one event loop (async I/O for Muse calls) while
    # CPU/serialisation work runs in threads via asyncio.to_thread.
    work_queue: asyncio.Queue = asyncio.Queue()
    result_queue: asyncio.Queue = asyncio.Queue()
    shutdown_event = asyncio.Event()
    rate_lock = asyncio.Lock()
    rate_state = {"last_start": 0.0}
    single_sem = asyncio.Semaphore(max(1, args.workers))
    stats = {"packed_ok": 0, "single_ok": 0, "failed": 0}
    # Overall TUI progress: done counts everything already on disk plus
    # every conversation resolved this run (written or permanently
    # failed), so the bar always reaches 100% even with failures.
    total_target = len(processed) + len(remaining)
    base_done = len(processed)
    progress = {"written": 0, "failed": 0}
    run_start = time.monotonic()
    render_state = {"last_len": 0, "last_t": 0.0}

    def _render(force: bool = False) -> None:
        if total_target <= 0:
            return
        now = time.monotonic()
        if not force and (now - render_state["last_t"]) < 0.25:
            return
        done = base_done + progress["written"] + progress["failed"]
        resolved = progress["written"] + progress["failed"]
        line = _format_progress(done, total_target, resolved,
                                now - run_start)
        pad = max(0, render_state["last_len"] - len(line))
        sys.stdout.write("\r" + line + " " * pad)
        sys.stdout.flush()
        render_state["last_len"] = len(line)
        render_state["last_t"] = now

    try:
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, shutdown_event.set)
            except (NotImplementedError, RuntimeError, ValueError):
                pass
    except RuntimeError:
        pass

    async def _gate():
        """Global rate limiter: minimum --interval between call starts."""
        async with rate_lock:
            now = time.monotonic()
            wait = args.interval - (now - rate_state["last_start"])
            if wait > 0:
                await asyncio.sleep(wait)
            rate_state["last_start"] = time.monotonic()

    async def _retry_one(conv):
        async with single_sem:
            await _gate()
            return await annotate_one(
                conv, args.model, api_key, args.reasoning_effort)

    async def _worker(wid):
        while not shutdown_event.is_set():
            batch = await work_queue.get()
            try:
                if batch is None:
                    break
                print(f"[worker {wid}] Annotating {len(batch)} "
                      f"conversation(s) in one call...")
                await _gate()
                samples, failed = await annotate_batch(
                    batch, args.model, api_key, args.reasoning_effort)
                single_ok = 0
                if failed and not shutdown_event.is_set():
                    # Concurrent single-call fallback, bounded by
                    # single_sem so one bad batch cannot burst RPM.
                    retried = await asyncio.gather(
                        *(_retry_one(c) for c in failed))
                    for sample in retried:
                        if sample:
                            samples.append(sample)
                            single_ok += 1
                stats["packed_ok"] += len(samples) - single_ok
                stats["single_ok"] += single_ok
                stats["failed"] += len(failed) - single_ok
                progress["failed"] += len(failed) - single_ok
                for sample in samples:
                    line = await asyncio.to_thread(
                        json.dumps, sample)
                    await result_queue.put(line)
                print(f"[worker {wid}] Batch result: "
                      f"{len(samples) - single_ok} packed OK, "
                      f"{single_ok} recovered singly, "
                      f"{len(failed) - single_ok} still failed.")
            finally:
                work_queue.task_done()

    async def _writer():
        with open(args.output_file, "a", encoding="utf-8") as out_f:
            while True:
                line = await result_queue.get()
                try:
                    if line is None:
                        break
                    out_f.write(line + "\n")
                    out_f.flush()
                    progress["written"] += 1
                    _render()
                finally:
                    result_queue.task_done()

    _render(force=True)

    with concurrent.futures.ThreadPoolExecutor(
            max_workers=max(4, args.workers * 2),
            thread_name_prefix="annotate") as executor:
        loop = asyncio.get_running_loop()
        loop.set_default_executor(executor)
        writer_task = asyncio.create_task(_writer())
        worker_tasks = [asyncio.create_task(_worker(i))
                        for i in range(args.workers)]
        for batch in batches:
            if shutdown_event.is_set():
                break
            await work_queue.put(batch)
        for _ in worker_tasks:
            await work_queue.put(None)
        await work_queue.join()
        await asyncio.gather(*worker_tasks)
        await result_queue.put(None)
        await result_queue.join()
        await writer_task

    _render(force=True)
    sys.stdout.write("\n")
    sys.stdout.flush()
    elapsed_total = time.monotonic() - run_start
    resolved_total = progress["written"] + progress["failed"]
    rate_total = (resolved_total / elapsed_total) if elapsed_total > 0 else 0.0
    if shutdown_event.is_set():
        print("[!] Shutdown requested; partial progress is saved "
              f"(packed OK: {stats['packed_ok']}, "
              f"singly recovered: {stats['single_ok']}, "
              f"failed: {stats['failed']}, "
              f"elapsed: {_format_duration(elapsed_total)}).")
    total_now = len(get_processed_ids(args.output_file))
    print(f"Completed! '{args.output_file}' now contains {total_now} "
          f"annotated conversations in {_format_duration(elapsed_total)} "
          f"({rate_total:.2f} conv/s this run).")


if __name__ == "__main__":
    asyncio.run(main())
