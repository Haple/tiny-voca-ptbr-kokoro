"""Speak Brazilian Portuguese with tiny-voca pt-BR on a CPU.

    python speak.py --text "Oi, tudo bem?" --voice F --output oi.wav
    python speak.py --text-file historia.txt --voice M --output historia.wav

Needs numpy, onnxruntime and soundfile (requirements.txt), plus eSpeak NG
(espeak-ng) on the PATH. No GPU, no PyTorch, no network.
"""
import argparse
from pathlib import Path
import time

import soundfile as sf

from tiny_voca_ptbr import MODEL, Voice
from tiny_voca_ptbr.text import NothingToPronounce


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--text", help="text of any length; it is read sentence by sentence")
    source.add_argument("--text-file", type=Path, help="read the text from a UTF-8 file")
    parser.add_argument("--voice", choices=("F", "M"), default="F", help="F: woman, M: man")
    parser.add_argument("--speed", type=float, default=1.0, help="1.0 = as predicted, larger is faster")
    parser.add_argument("--semitones", type=float, default=0.0, help="move the whole voice up (+) or down (-)")
    parser.add_argument("--pitch-range", type=float, default=1.0, help="melody size: 1.0 as predicted, <1 flatter, >1 livelier")
    parser.add_argument("--threads", type=int, default=1, help="CPU threads for ONNX Runtime")
    parser.add_argument("--phonemes", action="store_true", help="print the phonemes the model was given")
    parser.add_argument("--model", type=Path, default=MODEL)
    parser.add_argument("--output", type=Path, required=True, help="WAV file to write (24 kHz, 16-bit)")
    args = parser.parse_args()
    text = args.text if args.text is not None else args.text_file.read_text(encoding="utf-8")
    voice = Voice(args.model, threads=args.threads)
    start = time.perf_counter()
    try:
        wave, spoken = voice.speak(text, args.voice, args.speed, args.semitones, args.pitch_range)
    except NothingToPronounce as error:
        parser.exit(1, f"{error}. No file was written.\n")
    took = time.perf_counter() - start
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sf.write(args.output, wave, voice.sample_rate, subtype="PCM_16")
    if args.phonemes:
        for phones in spoken:
            print(" ".join(phones.symbols))
    print(f"{len(wave) / voice.sample_rate:.2f} s of speech in {took:.2f} s on {args.threads} CPU thread(s) -> {args.output}")


if __name__ == "__main__":
    main()
