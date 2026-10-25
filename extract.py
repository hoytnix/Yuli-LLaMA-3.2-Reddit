"""
Extract DYADIC multi-turn conversations from ZhankuiHe/reddit_movie_large_v1.

Why this script exists, in one paragraph:
  The HF dataset is 1.67M *turns*, not conversations. Rows are grouped by
  conv_id, and conv_id is a *path* through a Reddit thread:
  e.g. t3_8vlkte_0/22 is path 0 of 22 paths extracted from thread t3_8vlkte.
  That means:
    - One thread -> many conversations.
    - One Reddit comment (turn_id) -> many conversations, because the same
      comment sits on many different paths.
  If you train on every conv_id naively you duplicate the same comment
  dozens of times, and you leak the same turns across train/validation
  splits and across training samples.

  This script therefore enforces, in order:
    1. Dyadic only: exactly 2 distinct user_id values in the whole conv_id.
       Role is NOT identity: is_seeker / USER / SYSTEM is a role label.
       SYSTEM is "any recommender", not one person. A conv_id with
       seeker + two different recommenders is 3 people, even if the
       turns are labelled USER/SYSTEM only. It is dropped.
    2. One use per comment: first kept conversation wins a turn_id.
       Any later conv_id that contains an already-used turn_id is
       dropped whole, not truncated. Each Reddit comment therefore
       appears in exactly one training conversation.
    3. Multi-turn, with both speakers actually speaking: by default
       >= 5 turns total, >= 2 turns per speaker, each kept turn >= 3 words,
       no [removed] / [deleted] / empty turns.
       Turn length is enforced as a turn_order minimum, not just a
       turn count: --min-turn-order (default 4, 0-indexed, i.e. turns
       0-4) is a configurable option, and the conversation must actually
       reach that turn_order. Raise it to 5 for a stricter cut (turns
       0-5, 6 turns total). A conversation with many rows that are all
       turn_order 0-1 duplicates does not count. The highest turn_order
       in the kept conversation is the check, enforced in both passes.
    4. Raw text, not processed text: `processed` replaces movie titles
       with IMDb IDs (tt3397884...). `raw` keeps the natural language
       ("Sicario, Seven..."). Yuli must learn language, not tt-codes.

  Output: reddit_movie_dyadic.jsonl, one JSON object per conversation:
    {
      "conv_id": "t3_8vlkte_0/22",
      "thread_id": "t3_8vlkte",
      "seeker_id": "t2_irzcd",          # hashed HF user id, audit only
      "participants": ["t2_irzcd", "t2_g77fz"],
      "turns": [
        {"turn_id": "t3_8vlkte", "turn_order": 0, "user_id": "t2_irzcd",
         "is_seeker": true, "speaker": "A",
         "text": "Looking for ...", "upvotes": null, "utc_time": 1530560352},
        ...
      ]
    }
  Speaker A is always the seeker (the conversation starter, is_seeker=true
  on turn 0). Speaker B is the single other participant. The annotator
  script maps A/B to anonymised labels and drops the user_ids entirely.

  Resumable: existing conv_ids in the output file are skipped, and their
  turn_ids are loaded into the used-turn set, so a resumed run does not
  re-use a comment that is already in the file.

  Two streaming passes are used on purpose: pass 1 only collects
  per-conv_id user sets and turn counts (small), pass 2 collects text
  only for kept conversations. Buffering all 1.67M rows would be wasteful.

Usage (Colab):
    pip install datasets
    python extract_reddit_movie_dyadic.py \
        --limit 1000 --min-turn-order 4 --min-turns-per-speaker 2 \
        --output reddit_movie_dyadic.jsonl
    # Stricter: --min-turn-order 5  (turns 0-5, at least 6 turns total)
"""

import argparse
import ast
import json
import os
import re
import sys
from collections import defaultdict


REMOVED_MARKERS = {"[removed]", "[deleted]", ""}
# user_ids that are not a real participant: deleted / anonymous accounts
# in the HF dump appear as the literal string "None".
INVALID_USER_IDS = {"", "None", "null", "deleted", "[deleted]"}

# Matches a stringified 2-element list, e.g.
#   "['USER', 'Looking for ...']"  or  '["SYSTEM", "Star wars"]'
# This is what the HF viewer shows, and what actually arrives in the
# `raw` / `processed` columns: a STRING, not a Python list. The old
# extract_text only stripped a leading "USER " prefix, so the whole
# "['USER', '...']" literal leaked into the output text field.
_STRINGIFIED_PAIR = re.compile(
    r"""^\[\s*['"](USER|SYSTEM)['"]\s*,\s*['"](.*)['"]\s*\]$""",
    re.DOTALL,
)


def _clean_text(text: str) -> str:
    """Strip role wrappers and trailing removed/deleted markers."""
    text = text.strip()
    if not text:
        return ""
    # Stringified list form first: "['USER', 'actual text']"
    if text.startswith("[") and ("USER" in text[:20] or "SYSTEM" in text[:20]):
        try:
            parsed = ast.literal_eval(text)
            if isinstance(parsed, (list, tuple)) and len(parsed) >= 2:
                return _clean_text(str(parsed[1]))
            if isinstance(parsed, (list, tuple)) and len(parsed) == 1:
                return _clean_text(str(parsed[0]))
        except (ValueError, SyntaxError):
            pass
        match = _STRINGIFIED_PAIR.match(text)
        if match:
            return _clean_text(match.group(2))
    # Plain role prefix form: "USER actual text" / "SYSTEM: actual text"
    for prefix in ("USER ", "SYSTEM ", "USER: ", "SYSTEM: "):
        if text.startswith(prefix):
            text = text[len(prefix):].strip()
            break
    # Trailing placeholder markers appended to a real post, e.g.
    # "Logan is so Great. [removed]" -- strip the marker, keep the post.
    for marker in ("[removed]", "[deleted]"):
        if text.endswith(marker):
            text = text[: -len(marker)].strip()
    # Unescape a stray trailing "']" left over from a malformed literal.
    if text.endswith("']") or text.endswith('"]'):
        text = text[:-2].strip()
    return text


def extract_text(raw, processed):
    """Return the natural-language utterance text from a dataset row.

    HF `raw` / `processed` columns arrive as a STRINGIFIED 2-element
    list, e.g. "['USER', 'Looking for ...']", not as a Python list.
    Handle list, dict, and plain-string shapes defensively, preferring raw,
    and always return clean text with no "['USER', '" prefix.
    """
    for field in (raw, processed):
        if field is None:
            continue
        if isinstance(field, (list, tuple)):
            if len(field) >= 2 and isinstance(field[1], str):
                return _clean_text(field[1])
            if len(field) == 1 and isinstance(field[0], str):
                return _clean_text(field[0])
            continue
        if isinstance(field, dict):
            for key in ("text", "content", "utterance"):
                if isinstance(field.get(key), str):
                    return _clean_text(field[key])
            continue
        if isinstance(field, str):
            return _clean_text(field)
    return ""


def thread_id_of(conv_id: str) -> str:
    """t3_8vlkte_0/22 -> t3_8vlkte ; t3_8upzwy_0/13 -> t3_8upzwy."""
    head = conv_id.split("/")[0]
    return head.rsplit("_", 1)[0] if "_" in head else head


def load_existing(output_path):
    """Return (existing conv_ids, used turn_ids) from a previous run."""
    conv_ids, turn_ids = set(), set()
    if os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if record.get("conv_id"):
                    conv_ids.add(record["conv_id"])
                for turn in record.get("turns", []):
                    if turn.get("turn_id"):
                        turn_ids.add(str(turn["turn_id"]))
    return conv_ids, turn_ids


def main():
    parser = argparse.ArgumentParser(
        description="Extract dyadic multi-turn conversations from "
        "ZhankuiHe/reddit_movie_large_v1."
    )
    parser.add_argument("--dataset", default="ZhankuiHe/reddit_movie_large_v1")
    parser.add_argument("--split", default="train",
                        help="HF split: train / validation / test")
    parser.add_argument("--limit", type=int, default=1000,
                        help="Max NEW conversations to write (default: 1000)")
    parser.add_argument("--min-turns", type=int, default=5,
                        help="Min total turns per conversation (default: 5, "
                        "kept in sync with --min-turn-order 4)")
    parser.add_argument("--min-turn-order", type=int, default=4,
                        help="Min highest turn_order a conversation must reach "
                        "(default: 4, i.e. turns 0-4, 0-indexed, at least "
                        "5 turns total). This is the primary depth check: "
                        "a 2-turn request + recommendation has no depth for "
                        "Yuli to track a shift in. Pass 5 for a stricter cut "
                        "(turns 0-5, at least 6 turns).")
    parser.add_argument("--min-turns-per-speaker", type=int, default=2,
                        help="Min turns per participant (default: 2)")
    parser.add_argument("--max-turns", type=int, default=16,
                        help="Truncate conversations longer than this (default: 16). "
                        "Set 0 for no truncation.")
    parser.add_argument("--min-words", type=int, default=3,
                        help="Min words per kept turn (default: 3)")
    parser.add_argument("--output", default="reddit_movie_dyadic.jsonl")
    # No API key is hard-coded in this file. This dataset is public, so
    # usually no token is needed at all. If one ever is, pass it as an
    # option (--hf-token) or via HF_TOKEN in the environment.
    parser.add_argument("--token", "--hf-token", dest="token",
                        default=os.getenv("HF_TOKEN"),
                        help="Hugging Face token, as an option only, or via "
                        "HF_TOKEN env var. Not hard-coded.")
    args = parser.parse_args()

    from datasets import load_dataset

    existing_conv_ids, used_turn_ids = load_existing(args.output)
    print(f"Existing conversations in '{args.output}': {len(existing_conv_ids)}")
    print(f"Turn ids already used: {len(used_turn_ids)}")
    print(f"Target: {args.limit} new dyadic conversations, "
          f"min_turn_order={args.min_turn_order}, "
          f"min_turns={args.min_turns}, "
          f"min_per_speaker={args.min_turns_per_speaker}, "
          f"min_words={args.min_words}")

    def load():
        kwargs = {"path": args.dataset, "split": args.split, "streaming": True}
        if args.token:
            kwargs["token"] = args.token
        return load_dataset(**kwargs)

    # ---- Pass 1: per-conv_id stats only, no text buffering. ----
    print("\nPass 1: counting participants per conv_id ...")
    users_per_conv = defaultdict(set)
    turns_per_conv = defaultdict(int)
    per_speaker = defaultdict(lambda: defaultdict(int))
    max_turn_order = defaultdict(int)
    n_rows = 0
    for entry in load():
        conv_id = str(entry.get("conv_id", ""))
        user_id = str(entry.get("user_id", ""))
        if not conv_id or user_id in INVALID_USER_IDS:
            continue
        users_per_conv[conv_id].add(user_id)
        turns_per_conv[conv_id] += 1
        per_speaker[conv_id][user_id] += 1
        try:
            order = int(entry.get("turn_order", 0))
        except (TypeError, ValueError):
            order = 0
        if order > max_turn_order[conv_id]:
            max_turn_order[conv_id] = order
        n_rows += 1
        if n_rows % 200000 == 0:
            print(f"  ... {n_rows} rows, {len(users_per_conv)} conv_ids seen")

    dyadic = {
        conv_id
        for conv_id, users in users_per_conv.items()
        if len(users) == 2
        and turns_per_conv[conv_id] >= args.min_turns
        and max_turn_order[conv_id] >= args.min_turn_order
        and all(count >= args.min_turns_per_speaker
                for count in per_speaker[conv_id].values())
        and conv_id not in existing_conv_ids
    }
    print(f"Pass 1 done: {n_rows} rows, {len(users_per_conv)} conv_ids, "
          f"{len(dyadic)} dyadic candidates (before turn-text filtering "
          f"and per-comment dedup)")

    # ---- Pass 2: collect text for dyadic candidates, first path wins. ----
    print("\nPass 2: collecting kept conversations (first path wins) ...")
    buffered = defaultdict(list)
    kept_order = []
    n_rows2 = 0
    for entry in load():
        n_rows2 += 1
        if n_rows2 % 200000 == 0:
            print(f"  ... {n_rows2} rows, {len(buffered)} dyadic convs buffered")
        conv_id = str(entry.get("conv_id", ""))
        if conv_id not in dyadic:
            continue
        if str(entry.get("user_id", "")) in INVALID_USER_IDS:
            continue
        text = extract_text(entry.get("raw"), entry.get("processed"))
        if text in REMOVED_MARKERS or len(text.split()) < args.min_words:
            continue
        buffered[conv_id].append({
            "turn_id": str(entry.get("turn_id", "")),
            "turn_order": int(entry.get("turn_order", 0)),
            "user_id": str(entry.get("user_id", "")),
            "is_seeker": bool(entry.get("is_seeker", False)),
            "text": text,
            "upvotes": entry.get("upvotes"),
            "utc_time": entry.get("utc_time"),
        })

    print(f"Pass 2 done: {n_rows2} rows streamed, "
          f"{len(buffered)} dyadic convs buffered, now writing ...")
    written = 0
    skipped_reuse = 0
    with open(args.output, "a", encoding="utf-8") as out_f:
        # Deterministic order: sort by thread, then by path number, so
        # "first path wins" is reproducible across runs.
        for conv_id in sorted(buffered):
            turns = sorted(buffered[conv_id], key=lambda t: t["turn_order"])
            if len(turns) < args.min_turns:
                continue
            # Re-enforce the turn_order minimum after text filtering:
            # dropping short / [removed] turns must not shrink a
            # conversation back below the depth threshold.
            if max(t["turn_order"] for t in turns) < args.min_turn_order:
                continue
            users = []
            for turn in turns:
                if turn["user_id"] not in users:
                    users.append(turn["user_id"])
            if len(users) != 2:
                continue
            # Both speakers must still have >= min turns after text filtering.
            counts = {user: sum(1 for t in turns if t["user_id"] == user)
                      for user in users}
            if any(c < args.min_turns_per_speaker for c in counts.values()):
                continue
            # One use per comment: if ANY turn was already used, drop the
            # whole conversation, do not truncate it into a partial thread.
            turn_ids = [t["turn_id"] for t in turns if t["turn_id"]]
            if any(tid in used_turn_ids for tid in turn_ids):
                skipped_reuse += 1
                continue
            if args.max_turns and len(turns) > args.max_turns:
                turns = turns[:args.max_turns]
                # Re-check dyadic + per-speaker counts after truncation.
                users = []
                for turn in turns:
                    if turn["user_id"] not in users:
                        users.append(turn["user_id"])
                if len(users) != 2:
                    continue
                turn_ids = [t["turn_id"] for t in turns if t["turn_id"]]

            seeker_id = next((t["user_id"] for t in turns if t["is_seeker"]),
                             users[0])
            # Speaker A = seeker (target for Yuli), Speaker B = the other.
            for turn in turns:
                turn["speaker"] = "A" if turn["user_id"] == seeker_id else "B"

            record = {
                "conv_id": conv_id,
                "thread_id": thread_id_of(conv_id),
                "seeker_id": seeker_id,
                "participants": users,
                "turns": turns,
            }
            out_f.write(json.dumps(record) + "\n")
            out_f.flush()
            for tid in turn_ids:
                used_turn_ids.add(tid)
            written += 1
            if written % 100 == 0:
                print(f"  ... wrote {written}/{args.limit}")
            if written >= args.limit:
                break

    print(f"\nDone. Wrote {written} new dyadic conversations to "
          f"'{args.output}'.")
    print(f"Skipped (turn already used in another path): {skipped_reuse}")
    print(f"Total conversations now: "
          f"{len(existing_conv_ids) + written}")


if __name__ == "__main__":
    main()

