"""Brazilian Portuguese front end: text -> phoneme tokens with a stress feature.

Text first goes through ``normalize_ptbr`` (digits, money, dates, times and
abbreviations become words). eSpeak NG with its ``pt-br`` voice then turns
the words into phonemes.

* eSpeak NG is an external rule-based program, called as a separate process.
  It is not part of this repository; install it with your package manager.
* A token is one eSpeak phoneme as eSpeak segments it (``--sep``), for
  example ``t``, ``tʃ``, ``eɪ`` or ``ɐ̃ʊ̃``. Phonemes outside the 51-phone
  inventory (rare ones, or phonemes of a foreign word) map to the closest
  inventory phones (``FALLBACK``).
* Lexical stress is a parallel feature (0 none, 1 primary, 2 secondary), not a
  token, so it never needs its own audio frames.
* Punctuation, word gaps and the utterance edges are explicit tokens.
* Every symbol is stored decomposed (Unicode NFD), the way eSpeak writes nasal
  vowels (o + U+0303).

The model was trained on tokens made by exactly this code. ``frontend_fingerprint``
hashes the inventory and the fallback rules; the model metadata stores the
same hash, and ``tiny_voca_ptbr.voice.Voice`` refuses to run when they differ.

Origin: ``src/tiny_voca/text.py`` of tiny-voca by Muhammed Nazeem
(https://github.com/n4ze3m/tiny-voca, Apache-2.0). Changed for this port: the
pt-br eSpeak voice, the phoneme inventory, the fallback table, the
abbreviations, NFD symbols and the pt-BR normalizer. For this release the
English-only "dizione" front end was removed. Tokens are unchanged.
"""

from dataclasses import dataclass
import hashlib
import json
import re
import shutil
import subprocess
import unicodedata

FRONTEND = "espeak"
FRONTEND_VERSION = "ptbr-1"
ESPEAK_VOICE = "pt-br"
SEPARATOR = "_"

PAD, BOS, EOS, SP, UNK = "<pad>", "<bos>", "<eos>", "<sp>", "<unk>"
SPECIALS = (PAD, BOS, EOS, SP, UNK)
PUNCTUATION = (",", ".", "?", "!", ";", ":", '"', "'", "(", ")", "—", "…")
# Every eSpeak NG 1.51 pt-br phoneme seen at least 100 times over 41k Common
# Voice sentences (all are labels of the forced aligner). Nasal vowels are
# written as vowel + ŋ by eSpeak; final unstressed e/a come out as y/æ.
CONSONANTS = ("p", "b", "t", "d", "k", "ɡ", "f", "v", "s", "z", "ʃ", "ʒ", "x", "m", "n", "ɲ", "ŋ",
              "l", "ʎ", "ɾ", "r", "j", "w", "tʃ", "dʒ")
VOWELS = ("a", "aː", "e", "ɛ", "i", "o", "ɔ", "u", "ə", "æ", "y", "ʊ", "ɐ̃", "ũ", "õ",
          "eɪ", "aɪ", "oɪ", "aʊ", "eʊ", "iʊ", "ɛʊ", "uɪ", "ɛɪ", "ɔɪ", "ɐ̃ʊ̃")
# eSpeak and the aligner write nasal vowels decomposed (o + U+0303); so do we.
CONSONANTS = tuple(unicodedata.normalize("NFD", s) for s in CONSONANTS)
VOWELS = tuple(unicodedata.normalize("NFD", s) for s in VOWELS)
PHONES = CONSONANTS + VOWELS
# Append-only: changing the order or removing a symbol invalidates the model.
SYMBOLS = SPECIALS + PUNCTUATION + PHONES
SYMBOL_TO_ID = {symbol: index for index, symbol in enumerate(SYMBOLS)}
PHONE_SET = frozenset(PHONES)
VOWEL_SET = frozenset(VOWELS)
STRESS_MARKS = {"ˈ": 1, "ˌ": 2}

# Rare pt-br phonemes and the phonemes eSpeak uses when it switches to another
# language for a foreign word map to the closest inventory phone(s).
FALLBACK = {
    "ɪ": ("i",), "iː": ("i",), "ts": ("t", "s"), "dz": ("d", "z"), "h": ("x",), "ɹ": ("ɾ",), "θ": ("s",), "ð": ("d",),
    "c": ("k",), "ɔø": ("ɔ",), "ɜ": ("ɛ",), "ɜː": ("ɛ",), "ʌ": ("a",), "ɑ": ("a",), "ɑː": ("a",), "ɐ": ("a",),
    "ɒ": ("ɔ",), "ɔː": ("ɔ",), "uː": ("u",), "ɛː": ("ɛ",), "eː": ("e",), "oː": ("o",), "ɚ": ("e", "r"),
    "oʊ": ("o", "w"), "əʊ": ("o", "w"), "ʔ": (), "ç": ("x",), "χ": ("x",), "ʁ": ("x",), "ɣ": ("ɡ",), "β": ("v",),
    "ɫ": ("l",), "ɬ": ("l",), "ɻ": ("ɾ",), "ʀ": ("x",), "ø": ("e",), "œ": ("ɛ",), "ɨ": ("i",), "ʉ": ("u",),
    "ɑ̃": ("ɐ̃",), "ɔ̃": ("õ",), "ɛ̃": ("e", "ŋ"), "ĩ": ("i", "ŋ"), "ẽ": ("e", "ŋ"), "ã": ("ɐ̃",),
    "dʑ": ("dʒ",), "tɕ": ("tʃ",), "ɕ": ("ʃ",), "ʑ": ("ʒ",), "ʝ": ("j",), "ɟ": ("dʒ",),
    "aɪə": ("a", "j", "a"), "aɪɚ": ("a", "j", "e", "r"), "iə": ("i", "a"), "ʊə": ("u", "a"), "eə": ("e", "a"),
    "l̩": ("u", "w"), "n̩": ("e", "n"), "m̩": ("e", "m"), "əl": ("e", "w"),
}
FALLBACK = {unicodedata.normalize("NFD", k): tuple(unicodedata.normalize("NFD", x) for x in v) for k, v in FALLBACK.items()}

_QUOTES = str.maketrans({"‘": "'", "’": "'", "‚": "'", "‛": "'", "`": "'", "´": "'",
                         "“": '"', "”": '"', "„": '"', "‟": '"', "«": '"', "»": '"',
                         "–": "—", "―": "—", "[": "(", "{": "(", "]": ")", "}": ")"})
_PUNCTUATION_CHARS = frozenset(PUNCTUATION)
ABBREVIATIONS = {"sr", "sra", "srta", "dr", "dra", "prof", "profa", "av", "etc", "vs", "jr", "pág", "tel"}
_LANGUAGE_TAG = re.compile(r"\((?:_?[a-z0-9]_?){2,3}(?:-(?:_?[a-z0-9]_?)+)*\)")


class NothingToPronounce(ValueError):
    """The text has no word, number or letter to say: only marks, emoji and the like."""


@dataclass(frozen=True)
class Phonemized:
    symbols: tuple
    ids: tuple
    stress: tuple
    unknown: tuple  # eSpeak phonemes that had no inventory mapping

    def __len__(self):
        return len(self.ids)


def frontend_identity():
    return {"frontend_version": FRONTEND_VERSION, "voice": ESPEAK_VOICE, "separator": SEPARATOR,
            "symbols": list(SYMBOLS), "fallback": {key: list(value) for key, value in sorted(FALLBACK.items())}}


def frontend_fingerprint():
    return hashlib.sha256(json.dumps(frontend_identity(), sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def normalize_text(text):
    from .normalize_ptbr import normalize as normalize_ptbr
    # Before NFKC, which turns º/ª into o/a ("nº", "1º", "2ª").
    text = unicodedata.normalize("NFKC", normalize_ptbr(text)).translate(_QUOTES)
    text = re.sub(r"\.{3,}", "…", text)
    text = re.sub(r"-{2,}|\s-\s", "—", text)
    return re.sub(r"\s+", " ", text).strip()


def chunks(text):
    """Yield ``(is_punctuation, piece)``. Decimals, clock times, digit groups,
    contractions, title abbreviations and initialisms stay inside the text piece
    for eSpeak; every other punctuation mark becomes its own token."""
    start = 0
    for i, char in enumerate(text):
        if char not in _PUNCTUATION_CHARS:
            continue
        previous = text[i-1] if i else ""
        following = text[i+1] if i+1 < len(text) else ""
        if char == "'" and previous.isalnum() and following.isalnum():
            continue
        if char in ".,:" and previous.isdigit() and following.isdigit():
            continue
        if char == ".":
            word = re.search(r"([A-Za-zÀ-ÿ]+)$", text[:i])
            if word and word[1].lower() in ABBREVIATIONS:
                continue
            if previous.isalpha() and following.isalpha():
                continue
        if text[start:i].strip():
            yield False, text[start:i].strip()
        yield True, char
        start = i+1
    if text[start:].strip():
        yield False, text[start:].strip()


def espeak_executable():
    executable = shutil.which("espeak-ng")
    if executable is None:
        raise RuntimeError("eSpeak NG is not installed: install the espeak-ng package (for example: sudo apt install espeak-ng)")
    return executable


def espeak_version():
    return subprocess.run([espeak_executable(), "--version"], capture_output=True, text=True, check=True, timeout=10).stdout.strip()


def espeak_words(piece):
    """Phonemize one text piece. Returns words as lists of ``(phone, stress)``.

    The input must end with a newline: eSpeak NG 1.51 silently drops the last
    character of unterminated standard input.
    """
    result = subprocess.run(
        [espeak_executable(), "-q", "--ipa", "--sep=" + SEPARATOR, "-v", ESPEAK_VOICE, "--stdin"],
        input=piece.rstrip("\n") + "\n", text=True, capture_output=True, check=True, timeout=30)
    return parse_espeak(result.stdout)


def parse_espeak(output):
    words = []
    for word in _LANGUAGE_TAG.sub("", output).split():
        units = []
        for unit in word.split(SEPARATOR):
            stress = 0
            for mark, level in STRESS_MARKS.items():
                if mark in unit:
                    stress = stress or level
                    unit = unit.replace(mark, "")
            unit = unicodedata.normalize("NFD", "".join(c for c in unit if unicodedata.category(c) != "Cf" and c not in "()"))
            if unit:
                units.append((unit, stress))
        if units:
            words.append(units)
    return words


def canonical(phone):
    """Map an eSpeak phoneme to inventory phones; ``None`` when unknown."""
    if phone in PHONE_SET:
        return (phone,)
    if phone in FALLBACK:
        return FALLBACK[phone]
    stripped = "".join(c for c in unicodedata.normalize("NFD", phone) if not unicodedata.combining(c))
    for candidate in (stripped, stripped.replace("ː", ""), stripped + "ː"):
        if candidate != phone and candidate in PHONE_SET:
            return (candidate,)
        if candidate != phone and candidate in FALLBACK:
            return FALLBACK[candidate]
    return None


def tokenize(pieces):
    """Build the token sequence from ``(is_punctuation, value)`` pieces where a
    text piece's value is the parsed eSpeak word list. Pure function (testable
    without eSpeak)."""
    symbols, stress, unknown = [BOS], [0], []
    for punctuation, value in pieces:
        if punctuation:
            if symbols[-1] != value:  # collapse repeats such as "!!"
                symbols.append(value)
                stress.append(0)
            continue
        for word in value:
            if symbols[-1] in PHONE_SET or symbols[-1] == UNK:
                symbols.append(SP)
                stress.append(0)
            for phone, level in word:
                mapped = canonical(phone)
                if mapped is None:
                    unknown.append(phone)
                    mapped = (UNK,)
                # A fallback may expand to several phones: stress goes to its first vowel.
                if not mapped:
                    continue
                carrier = next((i for i, item in enumerate(mapped) if item in VOWEL_SET), 0)
                for index, item in enumerate(mapped):
                    symbols.append(item)
                    stress.append(level if index == carrier else 0)
    symbols.append(EOS)
    stress.append(0)
    if not any(s in PHONE_SET or s == UNK for s in symbols):
        raise NothingToPronounce("the text has nothing to pronounce")
    return Phonemized(tuple(symbols), tuple(SYMBOL_TO_ID[s] for s in symbols), tuple(stress), tuple(unknown))


def phonemize(text):
    if not isinstance(text, str) or not text.strip():
        raise ValueError("text must not be empty")
    pieces = []
    for punctuation, piece in chunks(normalize_text(text)):
        pieces.append((True, piece) if punctuation else (False, espeak_words(piece)))
    return tokenize(pieces)
