# tiny-voca-ptbr-kokoro

A small Brazilian Portuguese text-to-speech model with two voices, a woman (F)
and a man (M). It has 4.5M parameters, ships as one 21 MB ONNX file, and runs
offline on a single CPU thread more than 30 times faster than real time. No
GPU and no PyTorch are needed.

It is a port of [tiny-voca](https://github.com/n4ze3m/tiny-voca) by Muhammed
Nazeem, an English model. The architecture (here with a 68-symbol
vocabulary) and the training recipe are upstream's. New in this port: a Brazilian front end (a pt-BR text normalizer
and eSpeak NG `pt-br` phonemes), a sentence corpus, and
[Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) as the teacher that
read it aloud. The model was trained from scratch on that synthetic speech.

On Hugging Face: [the model](https://huggingface.co/Haple/tiny-voca-ptbr-kokoro),
with the samples playable on the page.

Not to be confused with [SemFence/tiny-voca-ptbr](https://github.com/SemFence/tiny-voca-ptbr),
another Brazilian Portuguese port made independently in the same week, with
its own phonemizer, a Chatterbox teacher and a browser version.

**Em português.** O tiny-voca-ptbr-kokoro é um modelo pequeno de síntese de voz em
português do Brasil, com duas vozes: uma mulher (F) e um homem (M). São 4,5
milhões de parâmetros num arquivo ONNX de 21 MB, que roda offline num único
núcleo de CPU, mais de 30 vezes mais rápido que o tempo real, sem GPU. Ele
aprendeu a falar ouvindo o Kokoro-82M ler 8.945 frases brasileiras e foi
treinado do zero. Números, valores em reais, datas, horas e telefones são
escritos por extenso antes da leitura ("R$ 19,90" vira "dezenove reais e
noventa centavos"). Código e pesos sob a licença Apache-2.0.

## Samples

Two sentences that are not in the training data, each in both voices, made
with `speak.py` on a CPU (`samples/samples.json` has the phonemes and timings).

| Text | Woman (F) | Man (M) |
|---|---|---|
| Bom dia! A padaria da esquina abre às 6h30, então ainda dá tempo de comprar pão quentinho. | [bom-dia-F.wav](samples/bom-dia-F.wav) | [bom-dia-M.wav](samples/bom-dia-M.wav) |
| Você viu que o conserto do carro ficou em R$ 350? O mecânico disse que a peça só chega na quarta-feira. | [conserto-F.wav](samples/conserto-F.wav) | [conserto-M.wav](samples/conserto-M.wav) |

As a quick check, a small speech recognizer (Whisper small, on the CPU) wrote
the same words back for all four files, apart from two that sound alike: "às"
came back as "a" (before "seis" the two sound the same), and "conserto" as its
homophone "concerto". It also wrote the question as a statement (see
Limitations).

## Results

Every number here comes from a file in this repository: the model files, and
[`eval/`](eval), which holds files copied from the training run and the checks
run for this release.

| | |
|---|---|
| Parameters | 4,535,755 |
| Model file | `model/tiny-voca-ptbr.onnx`, 21.4 MB, float32 |
| Output | 24 kHz, mono |
| Word error rate, 16 new sentences x 2 voices (32 clips) | 1.44% (F 1.14%, M 1.74%) |
| Word error rate, held-out test clips (8) | 0.0% (teacher's recordings: 0.0%) |
| Word error rate, validation clips (8) | 0.78% (teacher's recordings: 0.78%) |
| Speed, 1 CPU thread, training machine | 31.0x real time with ONNX, 28.6x with PyTorch |
| Speed, 1 CPU thread, AMD EPYC 9354P, this package | 34.4x real time |
| Pitch error against the teacher, validation | 0.66 semitones, melody range ratio 1.01 |

- **Word error rate** ([`eval/report.json`](eval/report.json)). Whisper
  large-v3-turbo (via faster-whisper) transcribed every clip; the text and the
  transcript went through the same pt-BR normalizer, so "19" and "dezenove"
  match. The 16 new sentences are not in the training data. They have a
  price, a time, questions, nasal vowels, lh and nh. The slips: the man's "à
  tarde" lost the "à", the man's "sabiá" was heard as "Xabiá", and in both
  voices "para a manhã" was heard as "para amanhã", which sounds the same.
  These sets are small, so this is a sanity check, not a benchmark.
- **Speed** is seconds of speech per second of computing, with phonemization
  included. On the training machine, 6 texts in both voices (12 clips) gave
  84.1 s of speech in 2.71 s
  ([`eval/final_samples.json`](eval/final_samples.json); the CPU model was
  not recorded). This package's code on the same 12 texts:
  84.15 s in 2.44 s, the median of three passes on an AMD EPYC 9354P
  ([`eval/vps_check.json`](eval/vps_check.json)).
- **Pitch** is from the training log at step 100,000
  ([`eval/train_final_step.json`](eval/train_final_step.json)): the mean pitch
  error on validation clips, and how much the melody moves compared with the
  teacher's (1.0 = the same). On the test clips the student speaks 1.6%
  longer than the teacher (duration ratio 1.0155, `eval/report.json`).
- **Listening** ([`eval/listening.json`](eval/listening.json)). Aleph Santos,
  a Brazilian Portuguese speaker and the author, rated both final voices 4 of
  5. In a blind A/B test on 6 sentences against an unreleased variant (see
  below) he preferred this model 4 times, with 2 ties. That is one listener,
  not a MOS study.
- **The package matches the training run**
  ([`eval/frontend_check.json`](eval/frontend_check.json),
  [`eval/vps_check.json`](eval/vps_check.json)). Its front end gives the same
  tokens as the training code on 1,568 texts and rebuilds all 56 phoneme
  strings in `eval/report.json`. On the 12 speed-test clips, its audio has
  the same length as the training machine's, and the median spectral
  difference from it is 2.93 dB, the same as between two runs on one machine
  (2.92 dB): the hiss is random (see Limitations).

## Quick start

```bash
sudo apt install espeak-ng python3-venv   # Ubuntu 24.04 has eSpeak NG 1.51, the version used in training
git clone https://github.com/Haple/tiny-voca-ptbr-kokoro.git
cd tiny-voca-ptbr-kokoro
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
python speak.py --text "Oi! Tudo bem? O café custa R$ 7,50." --voice F --output oi.wav
```

The same files are on Hugging Face:
`hf download Haple/tiny-voca-ptbr-kokoro --local-dir tiny-voca-ptbr-kokoro`.

| Option | Default | Meaning |
|---|---|---|
| `--text`, `--text-file` | | Text of any length; it is read sentence by sentence |
| `--voice` | `F` | `F` (woman) or `M` (man) |
| `--speed` | 1.0 | Larger is faster |
| `--semitones` | 0 | Move the whole voice up or down |
| `--pitch-range` | 1.0 | Below 1 a flatter melody, above 1 a livelier one |
| `--threads` | 1 | CPU threads for ONNX Runtime |
| `--phonemes` | off | Print the phonemes the model was given |

From Python:

```python
import soundfile as sf
from tiny_voca_ptbr import Voice

voice = Voice()                       # model/tiny-voca-ptbr.onnx on 1 CPU thread
wave, phonemes = voice.speak("Bom dia! Hoje é dia 18/11.", "M")
sf.write("bom-dia.wav", wave, voice.sample_rate)
```

The normalizer writes out what a Brazilian would say:

| Written | Read as |
|---|---|
| R$ 19,90 | dezenove reais e noventa centavos |
| 14h30 | catorze e meia |
| 16/11/1990 | dezesseis de novembro de mil novecentos e noventa |
| (11) 98765-4321 | onze, nove oito sete seis cinco, quatro três dois um |
| 2.200 pessoas | duas mil e duzentas pessoas |
| 1º lugar, 2ª vez | primeiro lugar, segunda vez |
| Dr. Silva, Av. Paulista | Doutor Silva, avenida Paulista |

Requirements: Python with numpy, onnxruntime and soundfile
(`requirements.txt`; tested on Python 3.12 with numpy 1.26.4, onnxruntime
1.19.0 and soundfile 0.12.1, and with numpy 2.5.5, onnxruntime 1.32.0 and
soundfile 0.14.0), plus the `espeak-ng` program. The model learned eSpeak
NG 1.51's phonemes, and the code warns when another version is installed. On
9,013 corpus and evaluation texts, Debian 12's eSpeak NG 1.51 gives exactly
the training tokens; Debian 13's 1.52 gives different tokens for 42 of them,
almost all in names ([`eval/espeak_versions.json`](eval/espeak_versions.json)).

Tests: `python -m unittest discover -s tests -v`. They check that the front
end still makes the tokens the model was trained on and that both voices
speak.

## How it was made

1. **Choosing the teacher by ear.** Voices that upstream's teacher, Qwen3-TTS,
   designed from English or Portuguese descriptions did not sound Brazilian to
   a Brazilian listener, and most voice clones made with Qwen3-TTS and
   Chatterbox failed too. Kokoro-82M's own Brazilian voices passed. The woman
   is a mix of two Kokoro voices, 70% `pf_dora` and 30% `af_heart` (a weighted
   average of their style vectors); the man is Kokoro's `pm_santa`
   ([`model/voices.json`](model/voices.json)). Kokoro also turns Portuguese
   into sounds with eSpeak NG `pt-br`, so teacher and student follow the same
   pronunciation rules.
2. **Sentences** ([`eval/corpus_summary.json`](eval/corpus_summary.json)).
   8,945 sentences with 81,704 words: 3,600 from the Portuguese sentence
   collector of Mozilla Common Voice (CC0), filtered to Brazilian usage; 900
   templates for money, dates, times, phone numbers, ordinals and names; and
   4,445 sentences written for this project with a language model (everyday
   talk, dialogue, narration, explanations). 1,897 are questions. The split is
   by sentence (train 8,322, validation 357, test 266), so both voices of a
   sentence are in the same split.
3. **Words and phonemes.** `normalize_ptbr.py` writes out numbers, money,
   dates, times, phone numbers, ordinals and abbreviations, with gender
   agreement ("duas pessoas", "duzentas casas"). eSpeak NG then gives the
   phonemes. The model knows 68 symbols: 51 phones, 12 punctuation marks and 5
   special tokens. Symbols are stored decomposed (Unicode NFD), like eSpeak's
   output; with the composed "õ", nasal vowels had silently turned into oral
   ones.
4. **The teacher reads** ([`eval/teacher_generation.json`](eval/teacher_generation.json)).
   Kokoro read every sentence in both voices, 17,890 clips. Whisper
   large-v3-turbo transcribed each clip. The 457 clips that failed the checks
   (more than 15% word errors, with one wrong word always allowed, or an
   implausible length, speaking rate or silence) were left out, which kept
   16.37 hours.
5. **Alignment** ([`eval/annotations.json`](eval/annotations.json),
   [`eval/annotation_summary.json`](eval/annotation_summary.json)). The
   training code measures pitch (pYIN), loudness and mel spectrograms, then
   finds each phoneme's duration with the forced aligner
   `facebook/wav2vec2-lv-60-espeak-cv-ft`. That aligner labels Portuguese
   with European Portuguese phonemes, so for alignment only, Brazilian
   symbols are mapped to the labels it expects (æ to ɐ, y to ɨ, tʃ to t, dʒ to
   d, x to ʁ, r to ɾ, j to i, and four more). Clips with weak alignments were
   dropped: 15,539 training clips remained (woman 7,899, man 7,640; 14.78
   hours), plus 671 validation and 491 test clips.
6. **Training** ([`eval/recipe.json`](eval/recipe.json),
   [`eval/stages.txt`](eval/stages.txt)). Upstream's training settings, for
   100,000 steps from random weights: critics (GAN losses) from step 8,000,
   and teacher forcing faded out by step 10,000. It took 5 hours 10 minutes
   on one rented GPU on RunPod, including a 12-minute stop when the machine
   restarted; training then resumed exactly from a checkpoint.
7. **Evaluation and export** with upstream's scripts
   ([`eval/report.json`](eval/report.json)).

**Tried and not released.** A variant fine-tuned for 60,000 more steps on
speech from NVIDIA Magpie TTS had a lower word error rate on the same 16 new
sentences (1.14% against 1.44%), but it lost the blind listening test above:
it was never preferred ([`eval/magpie_variant.json`](eval/magpie_variant.json),
[`eval/listening.json`](eval/listening.json)). Here the lower error rate did
not predict the listener's choice.

## Limitations

- **Questions.** The rise at the end of a question may be weak. Whisper kept
  the question mark for 7 of the 12 questions in `eval/report.json` (4 of 6
  on the teacher's own recordings) and for 1 of 4 in the ONNX clips of
  `eval/final_samples.json`. Punctuation in a transcript is a rough proxy,
  so take this as a hint.
- **A lone "Oi!" in the man's voice** at the start of a text can sound like
  "Pois": Whisper heard "Pois" (`eval/final_samples.json`).
- **Pronunciation is eSpeak NG's.** English names and loanwords are read
  with Portuguese rules: "Instagram" ends like "grão" (`i ŋ s t a ɡ r ɐ̃ʊ̃`).
  `--phonemes` shows what the model was given.
- **No top of the band.** The teacher's audio has almost no energy above 10
  kHz (median 0.035% of the total in the evaluation clips), and the student
  copies that (0.033%) (`eval/vps_check.json`).
- **The hiss is random.** The vocoder's noise branch draws new noise on every
  call, so the same text never gives exactly the same samples. Timing and
  pitch do not change.
- **Synthetic teacher.** The model never heard a person. It copies Kokoro's
  accent and mistakes and cannot sound better than Kokoro.
- Two fixed voices, no voice cloning, no streaming.
- No in-browser version and no live demo yet. The phonemizer is the eSpeak NG
  program, so a browser would need eSpeak compiled to WebAssembly.
- Small evaluation sets (8 to 32 clips per group).

## Files

| Path | What |
|---|---|
| `model/tiny-voca-ptbr.onnx` | The model (float32). Inputs: tokens, stress, voice, speed, pitch_shift, pitch_range. Outputs: audio, durations |
| `model/tiny-voca-ptbr.json` | Sample rate, voices, the 68 symbols, limits and the front-end fingerprint |
| `model/voices.json` | Which Kokoro voices taught F and M |
| `model/SHA256SUMS` | Checksums |
| `tiny_voca_ptbr/` | Normalizer, front end and ONNX runtime |
| `speak.py` | Command-line tool |
| `tests/` | Checks that the front end matches the model and that both voices speak |
| `samples/` | The samples above |
| `eval/` | Evaluation report, training logs and the checks behind the numbers in this file |

## License

- Code and model weights: Apache-2.0 ([`LICENSE`](LICENSE), [`NOTICE`](NOTICE)).
  The repository includes code from tiny-voca by Muhammed Nazeem
  (Apache-2.0); `tiny_voca_ptbr/text.py` and `tiny_voca_ptbr/voice.py` are
  modified copies of upstream files and say what changed.
- The weights were trained from random initialization on speech generated by
  Kokoro-82M (Apache-2.0) with its voices `pf_dora`, `af_heart` and
  `pm_santa`. No Kokoro weights or voice files are included. Kokoro's model
  card describes its own training data.
- eSpeak NG (GPL-3.0-or-later) is called as a separate program and is not
  included.
- Used only to build the model and not included: the aligner
  `facebook/wav2vec2-lv-60-espeak-cv-ft` (Apache-2.0), Whisper large-v3-turbo
  (MIT) with faster-whisper (MIT), and Common Voice sentence-collector text
  (CC0-1.0).

## Credits

- [tiny-voca](https://github.com/n4ze3m/tiny-voca) by Muhammed Nazeem: the
  architecture, the training and evaluation code and the recipe.
- [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) by hexgrad: the teacher.
- [eSpeak NG](https://github.com/espeak-ng/espeak-ng): the phonemes.
- [Mozilla Common Voice](https://commonvoice.mozilla.org): Portuguese sentences.
- [wav2vec2-lv-60-espeak-cv-ft](https://huggingface.co/facebook/wav2vec2-lv-60-espeak-cv-ft)
  (Meta): forced alignment.
- [Whisper](https://github.com/openai/whisper) (OpenAI) and
  [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (SYSTRAN):
  screening and evaluation.

## How this was built

Aleph Santos set the goals, made the decisions and judged the outputs. He
asked for Brazilian voices that run free and offline, chose to have one
woman's and one man's voice, picked the teacher and the voice mix by
listening, rated the results, and chose this model over the Magpie variant in
a blind test.

Claude Code (Anthropic's coding agent) wrote the code as the implementer:
the normalizer, the front-end and aligner changes, the sentence pipeline, the
scripts that generated the audio and trained and evaluated the model on the
rented GPU (Claude Code also ran them), the ONNX runtime and this package.
