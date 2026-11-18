"""The tiny-voca pt-BR ONNX model as a small Python voice, without PyTorch.

    from tiny_voca_ptbr import Voice
    voice = Voice()                                   # model/tiny-voca-ptbr.onnx
    wave, spoken = voice.speak("Oi, tudo bem?", "F")  # float32 samples at voice.sample_rate

Text of any length is read sentence by sentence and the audio is joined. A
full stop is added when the text ends without a mark. A sentence too long for
one call of the model (``max_tokens`` phonemes or ``max_frames`` frames) is
cut after commas, then between words. Sentences with nothing to pronounce are
skipped.

Needs numpy and onnxruntime, plus eSpeak NG (``espeak-ng``) on the PATH.

Origin: the sentence splitting is copied from ``src/tiny_voca/infer.py`` of
tiny-voca by Muhammed Nazeem (https://github.com/n4ze3m/tiny-voca, Apache-2.0),
which needs PyTorch. Changed for Portuguese: the abbreviations, accented
capitals and dashes at a sentence start, and the message when nothing can be
read. The ONNX calls use the input and output names of upstream ``export.py``.
"""
import json
import math
from pathlib import Path
import re
import warnings

import numpy as np
import onnxruntime as ort

from .text import FRONTEND, NothingToPronounce, espeak_version, frontend_fingerprint, phonemize

MODEL = Path(__file__).resolve().parents[1] / "model" / "tiny-voca-ptbr.onnx"
PEAK = 0.99                                          # loudest sample of the returned audio
SEMITONE = math.log(2) / 12                          # the model's pitch_shift is a log-frequency offset
# The training data was phonemized with this eSpeak NG release (eval/annotations.json).
TRAINED_ESPEAK = "1.51"
PORTUGUESE_ONLY = "tiny-voca pt-BR só lê português"

# Copied from tiny_voca/infer.py.
_SPACE = re.compile("[\\s" + chr(0xFEFF) + "]+")
_SENTENCE_END = re.compile(r"""(?:(?<=[.!?…])|(?<=[.!?…]["')\]]))\s+(?=["'(\[—-]?[A-ZÀ-Ý0-9])""")
_NOT_AN_END = re.compile(r"(?:\b(?:sr|sra|srta|dr|dra|prof|profa|av|jr|vs|etc|pág|págs|tel|cel|obs|aprox|ltda|sto|sta|exmo|exma|p\.ex)|\b[A-ZÀ-Ý])\.$",
                         re.IGNORECASE)
_CLAUSE_END = re.compile(r"(?<=[,;:—])\s+")
_TEXT_END = re.compile("[\\s" + chr(0xFEFF) + "]+$")
_CLOSING = "\"'”’)]»"


def sentences(text):
    """The sentences of ``text``, without a break after a title, an initial or inside a number."""
    parts, current = [], ""
    for piece in _SENTENCE_END.split(_SPACE.sub(" ", text).strip()):
        current = f"{current} {piece}" if current else piece
        if not _NOT_AN_END.search(current):
            parts.append(current)
            current = ""
    if current:
        parts.append(current)
    return [part for part in parts if any(char.isalnum() for char in part)]


def finished(text):
    """``text`` with a full stop added when it ends without a mark, so the last word does not break off."""
    body = _TEXT_END.sub("", text)
    if not body:
        return text
    last = _SPACE.split(body)[-1]
    if any(char.isalnum() for char in last):
        marked = last.rstrip(_CLOSING)[-1:] in tuple(".!?…,;:")
    else:
        marked = any(char in ".!?…" for char in last) or last in (",", ";", ":")
    return text if marked else body + "."


def shorter(text):
    """``text`` in smaller pieces: after commas and the like if it has any, else in two between words."""
    pieces = _CLAUSE_END.split(text)
    if len(pieces) > 1:
        return pieces
    words = text.split(" ")
    if len(words) < 2:
        return []
    half = (len(words) + 1) // 2
    return [" ".join(words[:half]), " ".join(words[half:])]


def espeak_release():
    """``"1.51"`` from ``eSpeak NG text-to-speech: 1.51  Data at: ...``."""
    found = re.search(r"text-to-speech:\s*(\S+)", espeak_version())
    return found.group(1) if found else "unknown"


class Voice:
    def __init__(self, model=MODEL, threads=1):
        model = Path(model)
        self.meta = json.loads(model.with_suffix(".json").read_text(encoding="utf-8"))
        if self.meta["frontend"] != FRONTEND or frontend_fingerprint() != self.meta["frontend_fingerprint"]:
            raise RuntimeError("tiny_voca_ptbr/text.py does not make the tokens this model was trained on")
        release = espeak_release()
        if release != TRAINED_ESPEAK:
            warnings.warn(f"eSpeak NG {release} is installed, but the model was trained on eSpeak NG {TRAINED_ESPEAK} "
                          "phonemes; some words may be pronounced differently", stacklevel=2)
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(model), options, providers=["CPUExecutionProvider"])
        self.sample_rate = self.meta["sample_rate"]
        self.voices = list(self.meta["voices"])

    def _run(self, phones, voice, speed, pitch_shift, pitch_range):
        one = lambda value: np.array([value], dtype=np.float32)
        audio, durations = self.session.run(None, {
            "tokens": np.array([phones.ids], dtype=np.int64),
            "stress": np.array([phones.stress], dtype=np.int64),
            "voice": np.array([self.voices.index(voice)], dtype=np.int64),
            "speed": one(speed), "pitch_shift": one(pitch_shift), "pitch_range": one(pitch_range)})
        return audio[0], int(durations.sum())

    def speak(self, text, voice="F", speed=1.0, semitones=0.0, pitch_range=1.0, peak=PEAK):
        """Text of any length -> (float32 waveform, phonemized pieces).

        ``voice`` is ``"F"`` (woman) or ``"M"`` (man). ``speed`` 1.0 is the
        predicted rate, larger is faster. ``semitones`` moves the whole voice
        up or down. ``pitch_range`` scales the melody around its mean (1.0 as
        predicted, below 1 flatter, above 1 livelier). The audio is scaled down
        as a whole when it peaks above ``peak`` (``None`` keeps it as it is)."""
        if voice not in self.voices:
            raise ValueError(f"voice must be one of {self.voices}")
        limit = self.meta["max_tokens"] - 12
        waves, spoken = [], []

        def read(piece):
            try:
                phones = phonemize(piece)
            except NothingToPronounce:
                return
            if len(phones) <= limit:
                wave, frames = self._run(phones, voice, speed, semitones * SEMITONE, pitch_range)
                if frames <= self.meta["max_frames"]:
                    waves.append(wave)
                    spoken.append(phones)
                    return
            pieces = shorter(piece)
            if not pieces:
                raise ValueError(f"one word is too long to speak: {piece[:40]!r}")
            for part in pieces:
                read(part)

        for sentence in sentences(finished(text)):
            read(sentence)
        if not waves:
            raise NothingToPronounce(f"nothing in the text could be read: {PORTUGUESE_ONLY}")
        wave = np.concatenate(waves).astype(np.float32)
        if not np.isfinite(wave).all():
            raise RuntimeError("model produced non-finite audio")
        loudest = float(np.abs(wave).max())
        if peak is not None and loudest > peak:
            wave = wave * (peak / loudest)
        return wave, spoken
