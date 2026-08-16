"""
Post-OCR Spell Correction Module for Hate Speech Detection

Cleans up OCR output by:
1. Replacing ALL confused characters simultaneously (handles multi-char errors like 4S→is, [0→no)
2. Running automatic spell correction on low-confidence words
3. Filtering out single-character garbage tokens (stray digits/symbols)
4. Preserving slang / intentional misspellings found in the model vocabulary
"""

import re
import wordsegment
wordsegment.load()


# ──────────────────────────────────────────────
# Common OCR character confusion map
# ──────────────────────────────────────────────
# EasyOCR frequently confuses visually similar characters.
# Map: OCR-output-char → likely-correct-char
OCR_CHAR_FIXES = {
    "0": "o",   # zero   → letter o
    "1": "l",   # one    → letter l
    "4": "i",   # four   → letter i  (common in Impact font)
    "5": "s",   # five   → letter s
    "8": "b",   # eight  → letter b
    "|": "l",   # pipe   → letter l
    "!": "i",   # bang   → letter i
    "@": "a",   # at     → letter a
    "$": "s",   # dollar → letter s
    "[": "n",   # bracket→ letter n  (vertical strokes confused)
    "]": "j",   # bracket→ letter j
    "{": "t",   # brace  → letter t
    "}": "j",   # brace  → letter j
}


def _build_speller():
    """Lazy-load autocorrect Speller (cached at module level)."""
    try:
        from autocorrect import Speller
        return Speller(lang="en", fast=True)
    except ImportError:
        return None


# Module-level cache so the speller is only built once per process.
_speller = None


def _get_speller():
    global _speller
    if _speller is None:
        _speller = _build_speller()
    return _speller


def _is_real_word(speller, word: str) -> bool:
    """Check if a word is recognised by the speller (i.e. it returns unchanged)."""
    if speller is None:
        return False
    return speller(word).lower() == word.lower()


def _fix_ocr_char_swaps(word: str) -> str:
    """
    Replace ALL OCR-confused characters at once, then verify the
    result is a real English word.  Falls back to the original if not.

    This handles cases like  4S → is,  [0 → no,  vou → you  where
    multiple characters in the same word are mangled.
    """
    speller = _get_speller()
    if speller is None:
        return word

    if len(word) > 20 or len(word) < 2:
        return word

    # Lowercase first — OCR often produces mixed case like '4S' which
    # should become 'is' not 'iS'
    word_lower = word.lower()

    # --- Strategy 1: replace ALL confused chars at once ---
    all_replaced = []
    for ch in word_lower:
        all_replaced.append(OCR_CHAR_FIXES.get(ch, ch))
    candidate_all = "".join(all_replaced)

    if candidate_all != word_lower and _is_real_word(speller, candidate_all):
        return candidate_all

    # --- Strategy 2: replace all, then spell-correct the result ---
    spell_fixed = speller(candidate_all)
    if spell_fixed.lower() != word_lower and _is_real_word(speller, spell_fixed):
        return spell_fixed

    # --- Strategy 3: try one-at-a-time swaps (single char errors) ---
    for pos, ch in enumerate(word_lower):
        replacement = OCR_CHAR_FIXES.get(ch)
        if replacement:
            candidate = word_lower[:pos] + replacement + word_lower[pos + 1:]
            if _is_real_word(speller, candidate):
                return candidate
            # Also try spell-correcting after single swap
            spell_single = speller(candidate)
            if _is_real_word(speller, spell_single):
                return spell_single

    return word


def _is_garbage_token(word: str) -> bool:
    """
    Detect single-character (or very short) garbage tokens that are
    clearly OCR noise — lone digits, lone symbols, etc.
    """
    if len(word) > 2:
        return False
    # A single digit or symbol on its own is almost always noise
    if len(word) == 1 and not word.isalpha():
        return True
    # Two-char tokens that are all digits/symbols
    if len(word) == 2 and not any(c.isalpha() for c in word):
        return True
    return False


def correct_ocr_text(raw_text: str, word_confidences: list = None,
                     confidence_threshold: float = 0.80,
                     model_vocab: set = None) -> str:
    """
    Apply spell correction to OCR-extracted text.

    Parameters
    ----------
    raw_text : str
        The raw text extracted by OCR.
    word_confidences : list[float] | None
        Per-word confidence scores from OCR (same order as words in raw_text).
        If provided, only words below *confidence_threshold* are corrected.
        If None, all words are considered for correction.
    confidence_threshold : float
        Words with OCR confidence >= this value are assumed correct.
    model_vocab : set | None
        Set of words known to the hate-speech model's tokenizer.
        Words present here are *never* corrected (preserves slang).

    Returns
    -------
    str
        The corrected text.
    """
    speller = _get_speller()

    if not raw_text or not raw_text.strip():
        return raw_text

    # Process line-by-line to preserve line structure
    output_lines = []
    word_index = 0  # tracks position in the flat word_confidences list

    for line in raw_text.split("\n"):
        words = line.split()

        if word_confidences is None:
            line_confs = [0.0] * len(words)
        else:
            line_confs = []
            for _ in words:
                if word_index < len(word_confidences):
                    line_confs.append(word_confidences[word_index])
                else:
                    line_confs.append(0.0)
                word_index += 1

        model_vocab_set = model_vocab or set()

        corrected_words = []
        for word, conf in zip(words, line_confs):
            # Step 1: FIRST try to fix OCR character swaps
            # (must happen before garbage filtering — e.g. '[0' looks like
            #  garbage but is really 'no' with both chars mangled)
            fixed = _fix_ocr_char_swaps(word)
            corrected_tokens.append(candidate)

    return " ".join(corrected_tokens)
                    continue
            new_tokens.append(tok)

        out_paras.append(' '.join(new_tokens))

    # Rejoin paragraphs with double newline
    return '\n\n'.join(out_paras)

    for line in raw_text.split("\n"):
        words = line.split()

        if word_confidences is None:
            line_confs = [0.0] * len(words)
        else:
            line_confs = []
            for _ in words:
                if word_index < len(word_confidences):
                    line_confs.append(word_confidences[word_index])
                else:
                    line_confs.append(0.0)
                word_index += 1

        corrected_words = []
        for word, conf in zip(words, line_confs):
            # 0. NEW: Preserve common slang/internet words (don't correct these)
            if word.lower() in PRESERVE_WORDS:
                corrected_words.append(word)
                continue
            
            # 1. Very basic char swaps
            fixed = _fix_ocr_char_swaps(word)

            # 1b. NEW: Check if fixed version is also in preserve list
            if fixed.lower() in PRESERVE_WORDS:
                corrected_words.append(fixed)
                continue

            # 2. Skip obvious garbage after char swap
            if fixed == word and _is_garbage_token(word):
                continue
            if _is_garbage_token(fixed):
                continue

            # 3. If in vocab, keep it!
            if fixed.lower() in model_vocab_set:
                corrected_words.append(fixed)
                continue

            # 4. Try segmenting first before trusting spellcheck fake-positives!
            # If word is at least 5 chars and segments into multiple semantic words (like Fornoreason)
            if len(fixed) >= 5:
                segmented = wordsegment.segment(fixed)
                # If it successfully split into multiple distinct words
                if len(segmented) > 1:
                    corrected_words.extend(segmented)
                    continue

            # 5. If high confidence and real word, keep it
            if speller and _is_real_word(speller, fixed) and conf >= confidence_threshold:
                corrected_words.append(fixed)
                continue

            # 6. Spellcheck
            if speller is not None and not _is_real_word(speller, fixed):
                fixed = speller(fixed)

            corrected_words.append(fixed)

        if corrected_words:
            output_lines.append(" ".join(corrected_words))

    return "\n".join(output_lines)
