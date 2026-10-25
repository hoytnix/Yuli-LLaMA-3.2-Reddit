#!/usr/bin/env python3
"""double-check.py -- deterministic, offline fact-checker for Yuli annotation output.

Reads (read-only, never modified):
  - input transcript file (reddit_movie_dyadic.jsonl from the extractor):
    one record per dyadic conversation: conv_id, turns[] sorted by
    turn_order, each turn with speaker "A" (seeker, target) or "B".
  - annotated file (unsloth_4sides_dyadic.jsonl from annotate.py):
    one record per conversation with conv_id, turn_count,
    target_speaker "A", conversations[] (system/user/assistant training
    sample), and analysis{} with per-Speaker-A-turn labels.

Writes (only these two files):
  - flagged.jsonl: one JSON object per flagged conversation, sorted by
    conv_id, with conv_id + failures (check name, turn_order where
    applicable, expected vs got). Never includes the full training text.
  - report file: same summary as stdout plus the flagged conv_id list.

Constants are copied exactly from annotate.py (same directory):
  TYPE_TO_BITS, SIDE_OFFSETS, XOR vector math, ACTIVE-nibble function
  table, archetype library. No LLM calls, no API keys, no network.
  Standard library only.

Exit code: 1 if any HARD check fails, 0 if all pass. Soft checks only
affect reported counts, never the exit code or flagged.jsonl.

Usage:
  python3 double-check.py --input-file reddit_movie_dyadic.jsonl \
      --annotated-file unsloth_4sides_dyadic.jsonl \
      --flagged-file flagged.jsonl \
      --report-file double-check-report.txt
"""

import argparse
import json
import re
import sys
from collections import Counter

# --- Exact copies of annotate.py constants (do NOT drift) ---

TYPE_TO_BITS = {
    "ISTP": 0x0, "ISTJ": 0x1, "ISFP": 0x2, "ISFJ": 0x3,
    "INTP": 0x4, "INTJ": 0x5, "INFP": 0x6, "INFJ": 0x7,
    "ESTP": 0x8, "ESTJ": 0x9, "ESFP": 0xA, "ESFJ": 0xB,
    "ENTP": 0xC, "ENTJ": 0xD, "ENFP": 0xE, "ENFJ": 0xF,
}
SIDE_OFFSETS = {"ego": 0, "shadow": 1, "subconscious": 2, "superego": 3}

BITS_TO_TYPE = {v: k for k, v in TYPE_TO_BITS.items()}

# Top-2 functions of the ACTIVE nibble only, in order (from annotate.py prompt E).
ACTIVE_FUNCTIONS = {
    "ISTP": ["Ti", "Se"], "ISTJ": ["Si", "Te"],
    "ISFP": ["Fi", "Se"], "ISFJ": ["Si", "Fe"],
    "INTP": ["Ti", "Ne"], "INTJ": ["Ni", "Te"],
    "INFP": ["Fi", "Ne"], "INFJ": ["Ni", "Fe"],
    "ESTP": ["Se", "Ti"], "ESTJ": ["Te", "Si"],
    "ESFP": ["Se", "Fi"], "ESFJ": ["Fe", "Si"],
    "ENTP": ["Ne", "Ti"], "ENTJ": ["Te", "Ni"],
    "ENFP": ["Ne", "Fi"], "ENFJ": ["Fe", "Ni"],
}

ARCHETYPE_LIBRARY = [
    "Validation Seeker", "Lonely Romantic", "Corporate Savior", "Debater",
    "Silent Architect", "Efficiency Expert", "Protective Hero",
    "Wounded Healer", "High-Roller", "Aesthetic Collector",
    "Traditionalist", "Lurker", "Mid-Life Crisis", "Nice-Guy Rage",
    "Existential Dread", "Hedonistic Spiral",
]

VALID_EGO_TYPES = tuple(TYPE_TO_BITS.keys())
VALID_SIDES = ("ego", "shadow", "subconscious", "superego")
VALID_TENSION = ("low", "medium", "high")
VALID_LIMBIC = ("S", "T", "A", "neutral")

HARD_CHECKS = [
    "count_identity",
    "turn_coverage",
    "evidence_quote",
    "vector",
    "functions",
    "ego_side_validity",
    "dominant_ego",
    "overall_resolution",
    "synthesis_seed",
    "archetype",
    "state_vector",
]

SOFT_CHECKS = [
    "first_shift_baseline",
    "shift_words<=30",
    "rationale_words<=60",
    "cot_words<=250",
    "tension_r_consistency",
]

STATE_VECTOR_RE = re.compile(r"<state_vector>(.*?)</state_vector>", re.DOTALL)
TURN_VECTOR_RE = re.compile(
    r"^Turn\s+(\d+)\s*-\s*Speaker A:.*?\bVector\s+([0-9A-Fa-f]{3})\b",
    re.MULTILINE,
)


def compute_expected_hex(ego_type, active_side):
    """Recompute [Side][Ego][Active] exactly like annotate.py.

    Raises KeyError/AttributeError on invalid input; callers turn that
    into a check failure.
    """
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


def active_type_of(ego_type, active_side):
    """Active MBTI type string for an ego type + side (via XOR math)."""
    ego_val = TYPE_TO_BITS[ego_type.upper()]
    side_idx = SIDE_OFFSETS[active_side.lower()]
    if side_idx == 0:
        active_val = ego_val
    elif side_idx == 1:
        active_val = ego_val ^ 0x8
    elif side_idx == 2:
        active_val = ego_val ^ 0x9
    else:
        active_val = ego_val ^ 0xF
    return BITS_TO_TYPE[active_val]


def normalize_ws(text):
    """Collapse all whitespace runs to one space and strip (no case-fold)."""
    return re.sub(r"\s+", " ", text).strip() if isinstance(text, str) else ""


def word_count(text):
    norm = normalize_ws(text)
    return len(norm.split()) if norm else 0


def speaker_of(turn):
    """Speaker label, same fallback as annotate.py (is_seeker -> A/B)."""
    if isinstance(turn, dict) and "speaker" in turn and turn["speaker"] in ("A", "B"):
        return turn["speaker"]
    return "A" if isinstance(turn, dict) and turn.get("is_seeker") else "B"


def parse_float(value):
    """Strict numeric parse; bools and non-numerics return None."""
    if isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if result != result:  # NaN
        return None
    return result


def json_safe(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    return str(value)


def make_failure(check, expected, got, turn_order=None, reason=None):
    failure = {"check": check, "expected": json_safe(expected),
               "got": json_safe(got)}
    if turn_order is not None:
        failure["turn_order"] = turn_order
    if reason is not None:
        failure["reason"] = reason
    return failure


def assistant_text(record):
    for msg in record.get("conversations", []) or []:
        if isinstance(msg, dict) and msg.get("role") == "assistant":
            content = msg.get("content", "")
            if isinstance(content, str):
                return content
    return None


def parse_state_vector_tokens(text):
    match = STATE_VECTOR_RE.search(text)
    if not match:
        return None
    return match.group(1).split()


def parse_training_vectors(text):
    """Map turn_order -> Vector token from per-turn training-text lines."""
    found = {}
    for order_str, vec in TURN_VECTOR_RE.findall(text):
        try:
            order = int(order_str)
        except ValueError:
            continue
        found.setdefault(order, vec)
    return found


def evaluate_conversation(conv_id, input_rec, ann_recs):
    """Run all checks for one conv_id.

    Returns (hard_failures, per_hard, per_soft) where per_hard maps each
    hard check name to 'pass'/'fail'/'skip' and per_soft maps each soft
    check name to 'pass'/'fail'/'skip'. Soft results never flag.
    """
    hard = []
    per_hard = {}
    per_soft = {}

    def record(check, failures_before):
        per_hard[check] = "fail" if len(hard) > failures_before else "pass"

    # --- Check 1: count and identity ---
    before = len(hard)
    if input_rec is None:
        n = len(ann_recs)
        hard.append(make_failure(
            "count_identity", "conv_id present in input file (no extras)",
            f"absent from input, present {n}x in annotated output",
            reason="extra conv_id not in input"))
    elif not ann_recs:
        hard.append(make_failure(
            "count_identity", "exactly 1 annotated record",
            "0 records (missing from annotated output)",
            reason="input conv_id missing from annotated output"))
    elif len(ann_recs) > 1:
        hard.append(make_failure(
            "count_identity", "exactly 1 annotated record",
            f"{len(ann_recs)} records (duplicate conv_id)",
            reason="duplicate conv_id in annotated output"))
    record("count_identity", before)
    if input_rec is None or not ann_recs:
        for check in HARD_CHECKS[1:]:
            per_hard[check] = "skip"
        for check in SOFT_CHECKS:
            per_soft[check] = "skip"
        return hard, per_hard, per_soft

    rec = ann_recs[0]
    analysis = rec.get("analysis") if isinstance(rec, dict) else None
    if not isinstance(analysis, dict):
        before = len(hard)
        hard.append(make_failure(
            "turn_coverage", "analysis object with per-turn labels",
            "missing/non-object analysis",
            reason="no analysis to check coverage against"))
        record("turn_coverage", before)
        for check in HARD_CHECKS[2:]:
            per_hard[check] = "skip"
        for check in SOFT_CHECKS:
            per_soft[check] = "fail" if check == "first_shift_baseline" else "pass"
        return hard, per_hard, per_soft

    a_turns = analysis.get("turns", [])
    if not isinstance(a_turns, list):
        before = len(hard)
        hard.append(make_failure(
            "turn_coverage", "analysis.turns list", type(a_turns).__name__,
            reason="analysis.turns is not a list"))
        record("turn_coverage", before)
        for check in HARD_CHECKS[2:]:
            per_hard[check] = "skip"
        for check in SOFT_CHECKS:
            per_soft[check] = "fail" if check == "first_shift_baseline" else "pass"
        return hard, per_hard, per_soft

    # Input Speaker A turns, sorted by turn_order.
    input_a = []
    for turn in input_rec.get("turns", []) or []:
        if not isinstance(turn, dict):
            continue
        if speaker_of(turn) == "A":
            input_a.append(turn)
    input_a.sort(key=lambda t: t.get("turn_order", 0))
    expected_orders = [t.get("turn_order") for t in input_a]
    input_by_order = {}
    for turn in input_a:
        input_by_order.setdefault(turn.get("turn_order"), turn)

    got_orders = [t.get("turn_order") if isinstance(t, dict) else None
                  for t in a_turns]
    analysis_by_order = {}
    for item in a_turns:
        if isinstance(item, dict):
            analysis_by_order.setdefault(item.get("turn_order"), item)

    # --- Check 2: turn coverage exact, in order ---
    before = len(hard)
    if got_orders != expected_orders:
        hard.append(make_failure(
            "turn_coverage", expected_orders, got_orders,
            reason="analysis turn_orders must exactly equal input Speaker A turn_orders in order"))
    record("turn_coverage", before)

    # --- Check 3: evidence quote ---
    before = len(hard)
    for item in a_turns:
        if not isinstance(item, dict):
            hard.append(make_failure("evidence_quote", "per-turn label object",
                                     type(item).__name__, reason="turn entry is not an object"))
            continue
        order = item.get("turn_order")
        quote = item.get("evidence_quote", "")
        if not isinstance(quote, str):
            hard.append(make_failure("evidence_quote", "string evidence_quote",
                                     type(quote).__name__, turn_order=order,
                                     reason="evidence_quote is not a string"))
            continue
        norm_q = normalize_ws(quote)
        src = input_by_order.get(order)
        if src is None:
            hard.append(make_failure(
                "evidence_quote", "turn_order present among input Speaker A turns",
                order, turn_order=order, reason="invented turn has no source text"))
            continue
        raw_text = src.get("text", "")
        norm_text = normalize_ws(raw_text if isinstance(raw_text, str) else "")
        if norm_text == "":
            if norm_q != "":
                hard.append(make_failure(
                    "evidence_quote", "empty quote for empty turn text",
                    quote, turn_order=order,
                    reason="turn text is empty so the quote must be empty"))
            continue
        if norm_q == "":
            hard.append(make_failure(
                "evidence_quote", "non-empty verbatim substring of the turn text",
                quote, turn_order=order, reason="empty quote for non-empty turn"))
            continue
        if norm_q not in norm_text:
            hard.append(make_failure(
                "evidence_quote", "verbatim substring of THIS Speaker A turn",
                quote, turn_order=order, reason="quote is not a substring (whitespace-normalized, case-sensitive)"))
        if word_count(norm_q) > 15:
            hard.append(make_failure(
                "evidence_quote", "max 15 words", f"{word_count(norm_q)} words: {quote}",
                turn_order=order, reason="quote exceeds 15 words"))
    record("evidence_quote", before)

    # Precompute recomputed vectors / actives for checks 4, 5, 11.
    recomputed = {}  # order -> hex string or None if uncomputable
    for item in a_turns:
        if not isinstance(item, dict):
            continue
        try:
            recomputed[item.get("turn_order")] = compute_expected_hex(
                item.get("ego_type", ""), item.get("active_side", ""))
        except (KeyError, AttributeError):
            recomputed[item.get("turn_order")] = None

    text = assistant_text(rec) if isinstance(rec, dict) else None
    text_vectors = parse_training_vectors(text) if text is not None else None

    # --- Check 4: vector in analysis AND training text ---
    before = len(hard)
    for item in a_turns:
        if not isinstance(item, dict):
            continue
        order = item.get("turn_order")
        expected = recomputed.get(order)
        if expected is None:
            hard.append(make_failure(
                "vector", "recomputable vector from ego_type + active_side",
                f"{item.get('ego_type')!r}/{item.get('active_side')!r}",
                turn_order=order, reason="invalid ego/side, vector uncomputable"))
            continue
        got_vec = item.get("cognitive_vector", "")
        got_norm = str(got_vec).upper().strip() if isinstance(got_vec, str) else ""
        if got_norm != expected:
            hard.append(make_failure(
                "vector", expected, got_vec, turn_order=order,
                reason="analysis cognitive_vector differs from recomputed vector"))
        if text is None:
            hard.append(make_failure(
                "vector", expected, "no assistant training text",
                turn_order=order, reason="no assistant message to check the training-text vector"))
        elif order not in text_vectors:
            hard.append(make_failure(
                "vector", expected, "turn missing from training text",
                turn_order=order, reason="no per-turn Vector line for this turn_order"))
        elif str(text_vectors[order]).upper().strip() != expected:
            hard.append(make_failure(
                "vector", expected, text_vectors[order], turn_order=order,
                reason="training-text Vector differs from recomputed vector"))
    record("vector", before)

    # --- Check 5: functions = top-2 of ACTIVE nibble, in order ---
    before = len(hard)
    for item in a_turns:
        if not isinstance(item, dict):
            continue
        order = item.get("turn_order")
        try:
            active_type = active_type_of(item.get("ego_type", ""),
                                        item.get("active_side", ""))
        except (KeyError, AttributeError):
            hard.append(make_failure(
                "functions", "recomputable active type from ego_type + active_side",
                f"{item.get('ego_type')!r}/{item.get('active_side')!r}",
                turn_order=order, reason="invalid ego/side, active type uncomputable"))
            continue
        expected = ACTIVE_FUNCTIONS[active_type]
        got_funcs = item.get("primary_functions_active", [])
        if not isinstance(got_funcs, list) or any(not isinstance(f, str) for f in got_funcs):
            hard.append(make_failure(
                "functions", expected, got_funcs, turn_order=order,
                reason="primary_functions_active must be a list of 2 strings"))
            continue
        got_stripped = [f.strip() for f in got_funcs]
        if got_stripped != expected:
            hard.append(make_failure(
                "functions", expected, got_funcs, turn_order=order,
                reason=f"must equal top-2 of ACTIVE nibble ({active_type}) in order"))
    record("functions", before)

    # --- Check 6: ego/side validity ---
    before = len(hard)
    for item in a_turns:
        if not isinstance(item, dict):
            hard.append(make_failure("ego_side_validity", "per-turn label object",
                                     type(item).__name__, reason="turn entry is not an object"))
            continue
        order = item.get("turn_order")
        ego = item.get("ego_type", "")
        side = item.get("active_side", "")
        tension = item.get("tension", "")
        limbic = item.get("limbic_state", "")
        r_val = parse_float(item.get("resolution_r"))
        if not isinstance(ego, str) or ego.upper().strip() not in VALID_EGO_TYPES:
            hard.append(make_failure(
                "ego_side_validity", f"one of {list(VALID_EGO_TYPES)}", ego,
                turn_order=order, reason="invalid ego_type"))
        if not isinstance(side, str) or side.lower().strip() not in VALID_SIDES:
            hard.append(make_failure(
                "ego_side_validity", f"one of {list(VALID_SIDES)}", side,
                turn_order=order, reason="invalid active_side"))
        if not isinstance(tension, str) or tension.lower().strip() not in VALID_TENSION:
            hard.append(make_failure(
                "ego_side_validity", f"one of {list(VALID_TENSION)}", tension,
                turn_order=order, reason="invalid tension"))
        if isinstance(limbic, str) and limbic.upper().strip() in ("S", "T", "A"):
            pass
        elif isinstance(limbic, str) and limbic.lower().strip() == "neutral":
            pass
        else:
            hard.append(make_failure(
                "ego_side_validity", 'one of "S", "T", "A", "neutral"', limbic,
                turn_order=order, reason="invalid limbic_state"))
        if r_val is None or not 0.0 <= r_val <= 1.0:
            hard.append(make_failure(
                "ego_side_validity", "resolution_r in 0.0-1.0",
                item.get("resolution_r"), turn_order=order,
                reason="resolution_r out of range or non-numeric"))
    record("ego_side_validity", before)

    # --- Check 7: dominant ego ---
    before = len(hard)
    counted = []
    first_idx = {}
    for i, item in enumerate(a_turns):
        if not isinstance(item, dict):
            continue
        ego = item.get("ego_type", "")
        if isinstance(ego, str) and ego.strip():
            key = ego.upper().strip()
            counted.append(key)
            first_idx.setdefault(key, i)
    if counted:
        counts = Counter(counted)
        top = max(counts.values())
        candidates = [k for k, v in counts.items() if v == top]
        expected_dom = min(candidates, key=lambda k: first_idx[k])
    else:
        expected_dom = ""
    got_dom = analysis.get("dominant_ego_type", "")
    got_dom_norm = got_dom.upper().strip() if isinstance(got_dom, str) else ""
    if not isinstance(got_dom, str) or got_dom_norm != expected_dom:
        hard.append(make_failure(
            "dominant_ego", expected_dom, got_dom,
            reason="must equal Speaker A's most frequent ego_type (ties to earliest A turn)"))
    record("dominant_ego", before)

    # --- Check 8: overall resolution = final A turn R (0.01 tolerance) ---
    before = len(hard)
    final_item = None
    for item in a_turns:
        if not isinstance(item, dict):
            continue
        if final_item is None or (isinstance(item.get("turn_order"), int)
                                  and item.get("turn_order") > final_item.get("turn_order", -1)):
            if isinstance(item.get("turn_order"), int):
                final_item = item
    if final_item is None:
        final_item = a_turns[-1] if a_turns and isinstance(a_turns[-1], dict) else None
    final_r = parse_float(final_item.get("resolution_r")) if final_item else None
    conv_r = parse_float(analysis.get("conversation_resolution_r"))
    if final_r is None:
        hard.append(make_failure(
            "overall_resolution", "numeric final Speaker A turn resolution_r",
            final_item.get("resolution_r") if final_item else "no turns",
            reason="final turn R is non-numeric"))
    elif conv_r is None:
        hard.append(make_failure(
            "overall_resolution", "numeric conversation_resolution_r",
            analysis.get("conversation_resolution_r"),
            reason="conversation R is non-numeric"))
    elif abs(conv_r - final_r) > 0.01 + 1e-9:
        hard.append(make_failure(
            "overall_resolution", final_r, analysis.get("conversation_resolution_r"),
            reason="conversation R must equal final Speaker A turn R within 0.01"))
    record("overall_resolution", before)

    # --- Check 9: synthesis seed gating + length ---
    before = len(hard)
    seed = analysis.get("synthesis_seed", "")
    if not isinstance(seed, str):
        hard.append(make_failure(
            "synthesis_seed", "string synthesis_seed", type(seed).__name__,
            reason="synthesis_seed is not a string"))
    elif conv_r is None:
        hard.append(make_failure(
            "synthesis_seed", "numeric conversation_resolution_r for gating",
            analysis.get("conversation_resolution_r"),
            reason="conversation R non-numeric, seed gating undefined"))
    elif conv_r >= 0.75:
        if normalize_ws(seed) == "":
            hard.append(make_failure(
                "synthesis_seed", "non-empty seed when R >= 0.75", seed,
                reason="resolved conversations (R>=0.75) require a synthesis seed"))
        elif word_count(seed) > 25:
            hard.append(make_failure(
                "synthesis_seed", "max 25 words", f"{word_count(seed)} words: {seed}",
                reason="synthesis_seed exceeds 25 words"))
    else:
        if normalize_ws(seed) != "":
            hard.append(make_failure(
                "synthesis_seed", "empty seed when R < 0.75", seed,
                reason="unresolved conversations (R<0.75) must have an empty seed"))
    record("synthesis_seed", before)

    # --- Check 10: archetype ---
    before = len(hard)
    hint = analysis.get("archetype_hint", "")
    if not isinstance(hint, str):
        hard.append(make_failure(
            "archetype", "string archetype_hint", type(hint).__name__,
            reason="archetype_hint is not a string"))
    elif hint.strip() != "" and hint.strip() not in ARCHETYPE_LIBRARY:
        hard.append(make_failure(
            "archetype", f"empty or exactly one of {ARCHETYPE_LIBRARY}", hint,
            reason="archetype_hint is not a library entry"))
    record("archetype", before)

    # --- Check 11: <state_vector> in training text ---
    before = len(hard)
    expected_vecs = []
    for item in a_turns:
        if not isinstance(item, dict):
            expected_vecs.append("???")
            continue
        val = recomputed.get(item.get("turn_order"))
        expected_vecs.append(val if val is not None else "???")
    if text is None:
        hard.append(make_failure(
            "state_vector", " ".join(expected_vecs), "no assistant training text",
            reason="no assistant message to check <state_vector>"))
    else:
        tokens = parse_state_vector_tokens(text)
        if tokens is None:
            hard.append(make_failure(
                "state_vector", " ".join(expected_vecs), "no <state_vector> tag",
                reason="assistant text has no <state_vector>...</state_vector>"))
        else:
            got_upper = [t.upper() for t in tokens]
            if got_upper != expected_vecs:
                hard.append(make_failure(
                    "state_vector", " ".join(expected_vecs), " ".join(tokens),
                    reason="space-joined vectors must equal recomputed per-turn vectors in turn order"))
    record("state_vector", before)

    # --- Soft checks (counts only, never flag) ---
    shifts = [it.get("shift_from_previous", "") if isinstance(it, dict) else ""
              for it in a_turns]
    rationales = [it.get("psychological_rationale", "") if isinstance(it, dict) else ""
                  for it in a_turns]
    cot = analysis.get("chain_of_thought", "")
    if a_turns and isinstance(shifts[0], str) and "baseline" in shifts[0].lower():
        per_soft["first_shift_baseline"] = "pass"
    else:
        per_soft["first_shift_baseline"] = "fail"
    per_soft["shift_words<=30"] = "pass" if all(
        word_count(s) <= 30 for s in shifts if isinstance(s, str)) and all(
        isinstance(s, str) for s in shifts) else "fail"
    per_soft["rationale_words<=60"] = "pass" if all(
        word_count(r) <= 60 for r in rationales if isinstance(r, str)) and all(
        isinstance(r, str) for r in rationales) else "fail"
    per_soft["cot_words<=250"] = ("pass" if isinstance(cot, str)
                                  and word_count(cot) <= 250 else "fail")
    consistent = True
    for item in a_turns:
        if not isinstance(item, dict):
            continue
        tension = item.get("tension", "")
        r_val = parse_float(item.get("resolution_r"))
        if not isinstance(tension, str) or r_val is None:
            continue
        t = tension.lower().strip()
        if (t == "high" and abs(r_val - 1.0) < 1e-9) or \
                (t == "low" and abs(r_val - 0.0) < 1e-9):
            consistent = False
            break
    per_soft["tension_r_consistency"] = "pass" if consistent else "fail"

    return hard, per_hard, per_soft


def load_jsonl(path):
    records = []
    errors = []
    with open(path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                errors.append(f"{path}:{lineno}: JSON error: {e}")
    return records, errors


def format_table(title, checks, counts):
    lines = [title]
    lines.append(f"{'check':<22}{'pass':>8}{'fail':>8}{'skip':>8}")
    for check in checks:
        c = counts[check]
        lines.append(f"{check:<22}{c['pass']:>8}{c['fail']:>8}{c['skip']:>8}")
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Deterministic offline fact-checker for Yuli annotation "
        "output. Read-only on both input files; writes only flagged.jsonl "
        "and the report. Exit 1 if any hard check fails, 0 if all pass.")
    parser.add_argument("--input-file", default="reddit_movie_dyadic.jsonl")
    parser.add_argument("--annotated-file", default="unsloth_4sides_dyadic.jsonl")
    parser.add_argument("--flagged-file", default="flagged.jsonl")
    parser.add_argument("--report-file", default="double-check-report.txt")
    args = parser.parse_args(argv)

    try:
        input_recs, input_errors = load_jsonl(args.input_file)
    except FileNotFoundError:
        print(f"[!] Input file not found: {args.input_file}", file=sys.stderr)
        return 2
    try:
        ann_recs, ann_errors = load_jsonl(args.annotated_file)
    except FileNotFoundError:
        print(f"[!] Annotated file not found: {args.annotated_file}", file=sys.stderr)
        return 2
    for err in input_errors + ann_errors:
        print(f"[!] {err}", file=sys.stderr)

    input_by_id = {}
    for rec in input_recs:
        if isinstance(rec, dict) and rec.get("conv_id"):
            input_by_id.setdefault(rec["conv_id"], rec)
        else:
            print("[!] input record without conv_id skipped", file=sys.stderr)
    ann_by_id = {}
    for rec in ann_recs:
        if isinstance(rec, dict) and rec.get("conv_id"):
            ann_by_id.setdefault(rec["conv_id"], []).append(rec)
        else:
            print("[!] annotated record without conv_id skipped", file=sys.stderr)

    all_ids = sorted(set(input_by_id) | set(ann_by_id))

    hard_counts = {c: {"pass": 0, "fail": 0, "skip": 0} for c in HARD_CHECKS}
    soft_counts = {c: {"pass": 0, "fail": 0, "skip": 0} for c in SOFT_CHECKS}
    flagged = {}  # conv_id -> failures

    for cid in all_ids:
        failures, per_hard, per_soft = evaluate_conversation(
            cid, input_by_id.get(cid), ann_by_id.get(cid, []))
        for check in HARD_CHECKS:
            hard_counts[check][per_hard.get(check, "skip")] += 1
        for check in SOFT_CHECKS:
            soft_counts[check][per_soft.get(check, "skip")] += 1
        if failures:
            flagged[cid] = failures

    total = len(all_ids)
    n_flagged = len(flagged)
    n_passed = total - n_flagged

    out = []
    out.append("Double-check summary (offline, deterministic)")
    out.append(f"Input file:     {args.input_file} ({len(input_recs)} records)")
    out.append(f"Annotated file: {args.annotated_file} ({len(ann_recs)} records)")
    out.append(f"Total checked:  {total}")
    out.append(f"Passed:         {n_passed}")
    out.append(f"Flagged:        {n_flagged}")
    out.append("")
    out.append(format_table("Hard checks (any fail flags the conversation):",
                            HARD_CHECKS, hard_counts))
    out.append("")
    out.append(format_table("Soft checks (counts only, never flag):",
                            SOFT_CHECKS, soft_counts))
    summary = "\n".join(out)
    print(summary)
    if n_flagged:
        print(f"\nFlagged conversations ({n_flagged}):")
        for cid in sorted(flagged):
            print(f"  - {cid} ({len(flagged[cid])} failure(s))")

    with open(args.flagged_file, "w", encoding="utf-8") as f:
        for cid in sorted(flagged):
            f.write(json.dumps({"conv_id": cid,
                                "failures": flagged[cid]}) + "\n")

    with open(args.report_file, "w", encoding="utf-8") as f:
        f.write(summary + "\n")
        f.write(f"\nFlagged conversations ({n_flagged}):\n")
        for cid in sorted(flagged):
            f.write(f"- {cid} ({len(flagged[cid])} failure(s))\n")

    return 1 if n_flagged else 0


if __name__ == "__main__":
    sys.exit(main())
