"""Hand-crafted features from the transcripts and the audio timing stats.

Each feature is a simple signal tied to the grammar rubric: how much and how
fast the speaker talks, how long and complex the sentences are, how varied the
vocabulary is, how often they hesitate, and how many grammar errors a rule-based
checker (LanguageTool) finds.
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd

from src.data import check_alignment

WORD_RE = re.compile(r"[A-Za-z']+")
SENTENCE_END_RE = re.compile(r"(?<=[.!?])\s+")  # Whisper adds sentence punctuation

# Filled pauses. Whisper leaves out many of these, so this rate is a lower bound.
FILLERS = {"um", "umm", "uh", "uhm", "er", "erm", "ah", "hmm", "mm"}

# Words that usually open a subordinate clause. A rough proxy for the
# "complex structures" that separate rubric levels 4 and 5.
SUBORDINATORS = {"because", "although", "though", "while", "whereas", "if", "unless", "when",
                 "whenever", "whether", "since", "which", "who", "whom", "whose"}

# LanguageTool categories that say nothing about the speaker's grammar:
# spelling, casing, punctuation and typography come from the ASR output.
LT_IGNORED_CATEGORIES = {"TYPOS", "TYPOGRAPHY", "CASING", "PUNCTUATION"}


def transcript_features(text):
    """Length, sentence structure, vocabulary and disfluency features for one transcript."""
    words = [w.lower() for w in WORD_RE.findall(text)]
    n_words = len(words)
    n_sentences = max(sum(bool(WORD_RE.search(s)) for s in SENTENCE_END_RE.split(text.strip())), 1)
    per_100 = 100 / max(n_words, 1)
    return {
        "n_words": n_words,
        "words_per_sentence": n_words / n_sentences,
        "mean_word_length": float(np.mean([len(w) for w in words])) if words else 0.0,
        # Guiraud's index: unique words / sqrt(total words). Unlike the plain
        # type-token ratio it does not drop just because the speaker talks more.
        "lexical_diversity": len(set(words)) / np.sqrt(n_words) if words else 0.0,
        "filler_rate": sum(w in FILLERS for w in words) * per_100,
        # Immediate repeats such as "I I think" are a common sign of hesitation.
        "repetition_rate": sum(a == b for a, b in zip(words, words[1:])) * per_100,
        "subordinator_rate": sum(w in SUBORDINATORS for w in words) * per_100,
    }


def timing_features(transcripts, n_words):
    """Fluency features from the audio timing stats saved by the Kaggle notebook."""
    minutes = transcripts["duration_sec"].clip(lower=1) / 60
    return pd.DataFrame({
        "words_per_minute": n_words / minutes,
        "voiced_ratio": transcripts["voiced_sec"] / transcripts["duration_sec"].clip(lower=1),
        "long_pauses_per_minute": transcripts["n_long_pauses"] / minutes,
    })


def languagetool_features(transcripts, cache_path, language="en-US"):
    """LanguageTool error rates per 100 words, cached to CSV.

    Checking ~1000 transcripts takes a few minutes and needs Java, so the result
    is computed once and reused, like the transcripts themselves.
    """
    cache_path = Path(cache_path)
    if cache_path.exists():
        cached = pd.read_csv(cache_path, dtype={"file_id": str}, keep_default_na=False)
        check_alignment(cached, transcripts, cache_path.name)
        return cached

    import language_tool_python  # imported here so the cached path needs no Java

    tool = language_tool_python.LanguageTool(language)
    rows = []
    try:
        for i, text in enumerate(transcripts["transcript"]):
            matches = tool.check(text) if text else []
            kept = [m for m in matches if m.category not in LT_IGNORED_CATEGORIES]
            per_100 = 100 / max(len(WORD_RE.findall(text)), 1)
            rows.append({
                # GRAMMAR covers agreement, tense and article errors; the rest
                # (confused words, repeated words, style) goes into "other".
                "lt_grammar_rate": sum(m.category == "GRAMMAR" for m in kept) * per_100,
                "lt_other_rate": sum(m.category != "GRAMMAR" for m in kept) * per_100,
                "lt_rules": ";".join(m.rule_id for m in kept),  # kept for the report, not a model input
            })
            if (i + 1) % 200 == 0:
                print(f"LanguageTool: {i + 1}/{len(transcripts)}")
    finally:
        tool.close()

    result = pd.concat([transcripts[["file_id", "split"]].reset_index(drop=True), pd.DataFrame(rows)], axis=1)
    result.to_csv(cache_path, index=False)
    return result


def build_handcrafted(transcripts, cola, lt):
    """All hand-crafted features in one table, row-aligned with `transcripts`.

    `cola` holds the sentence acceptability scores from the Kaggle notebook and
    `lt` the LanguageTool rates. Missing values (clips with no sentences) are
    left as NaN and imputed inside the model pipeline.
    """
    text = pd.DataFrame([transcript_features(t) for t in transcripts["transcript"]])
    timing = timing_features(transcripts.reset_index(drop=True), text["n_words"])
    return pd.concat([
        text,
        timing,
        cola[["cola_mean", "cola_min", "cola_frac_bad"]].reset_index(drop=True),
        lt[["lt_grammar_rate", "lt_other_rate"]].reset_index(drop=True),
    ], axis=1)
