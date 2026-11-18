"""Checks for the packaged model and front end. From the repository root:

    python -m unittest discover -s tests -v
"""
import json
from pathlib import Path
import unittest
import warnings

import numpy as np

from tiny_voca_ptbr import MODEL, Voice
from tiny_voca_ptbr.normalize_ptbr import normalize
from tiny_voca_ptbr.text import SYMBOLS, NothingToPronounce, frontend_fingerprint, phonemize
from tiny_voca_ptbr.voice import TRAINED_ESPEAK, espeak_release, finished, sentences

ROOT = Path(__file__).resolve().parents[1]
META = json.loads(MODEL.with_suffix(".json").read_text(encoding="utf-8"))


class FrontEnd(unittest.TestCase):
    def test_matches_the_model(self):
        self.assertEqual(frontend_fingerprint(), META["frontend_fingerprint"])
        self.assertEqual(list(SYMBOLS), META["symbols"])

    def test_normalizer(self):
        cases = {
            "R$ 19,90": "dezenove reais e noventa centavos",
            "R$ 1,5 milhão": "um milhão e meio de reais",
            "16/11/1990": "dezesseis de novembro de mil novecentos e noventa",
            "14h30": "catorze e meia",
            "(11) 98765-4321": "onze, nove oito sete seis cinco, quatro três dois um",
            "2.200 pessoas": "duas mil e duzentas pessoas",
            "1º lugar, 2ª vez": "primeiro lugar, segunda vez",
            "25°C, 5 km, 50%": "vinte e cinco graus, cinco quilômetros, cinquenta por cento",
        }
        for written, spoken in cases.items():
            self.assertEqual(normalize(written), spoken)

    def test_rebuilds_the_evaluation_phonemes(self):
        if espeak_release() != TRAINED_ESPEAK:
            self.skipTest(f"eSpeak NG {espeak_release()} installed; the recorded phonemes come from {TRAINED_ESPEAK}")
        samples = json.loads((ROOT / "eval" / "report.json").read_text(encoding="utf-8"))["samples"]
        for sample in samples:
            self.assertEqual(" ".join(phonemize(sample["text"]).symbols), sample["phonemes"], sample["id"])

    def test_sentences(self):
        self.assertEqual(finished("Oi"), "Oi.")
        self.assertEqual(sentences("O Dr. Silva chegou. Sente-se!"), ["O Dr. Silva chegou.", "Sente-se!"])


class Speech(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cls.voice = Voice(threads=1)

    def test_both_voices(self):
        for name in ("F", "M"):
            wave, spoken = self.voice.speak("Oi, tudo bem? Custa R$ 7,50.", name)
            self.assertEqual(wave.dtype, np.float32)
            self.assertTrue(np.isfinite(wave).all())
            self.assertGreater(len(wave), self.voice.sample_rate)          # more than a second
            self.assertLessEqual(float(np.abs(wave).max()), 0.99 + 1e-6)
            self.assertEqual(len(spoken), 2)                               # two sentences

    def test_nothing_to_say(self):
        with self.assertRaises(NothingToPronounce):
            self.voice.speak("🙂 :)", "F")


if __name__ == "__main__":
    unittest.main()
