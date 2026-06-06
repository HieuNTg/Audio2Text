# Audio2Text — Vietnamese ASR System (Gemma 3N)

End-to-end toolkit for **Vietnamese automatic speech recognition** built on a fine-tuned
**Gemma 3N** model. The repository consolidates **training**, **evaluation**, and a
**production inference pipeline** into a clean, reproducible codebase.

> **Result:** **7.21% WER** on a 5,000-sample test set (0 empty predictions, ~97K reference words).

---

## Table of Contents
- [Highlights](#highlights)
- [Architecture](#architecture)
- [Inference Pipeline](#inference-pipeline)
- [Engineering Decisions & Trade-offs](#engineering-decisions--trade-offs)
- [Results](#results)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Tech Stack](#tech-stack)

---

## Highlights

- **Parameter-efficient fine-tuning** of Gemma 3N with **PEFT/LoRA + 4-bit quantization** via
  `unsloth.FastModel`, trainable on a single consumer GPU.
- **Robust inference pipeline** that handles real-world long-form audio/video:
  `Demucs → denoise → VAD → overlap-aware chunking → context-aware decoding`.
- **Rigorous evaluation harness** reporting WER alongside diagnostic signals
  (empty predictions, word-count drift, normalized text pairs for manual error analysis).
- **Reproducible, modular codebase** — separate `train` / `evaluate` / `predict` entry points
  with CLI wrappers that run from the project root without environment tweaking.

---

## Architecture

```
                          ┌─────────────────────────┐
   Raw audio / video ───▶ │   Inference Pipeline    │ ───▶  Transcript
                          └─────────────────────────┘
                                      ▲
                                      │ adapters (LoRA)
                           ┌─────────────────────────┐
   Vietnamese dataset ─▶   │   PEFT Fine-tuning      │
                           │   Gemma 3N + 4-bit      │
                           └─────────────────────────┘
```

---

## Inference Pipeline

The pipeline (`pipline_predict.py`) transforms raw, noisy, long-form input into clean text
through ordered stages — each stage exists to fix a specific real-world failure mode:

| # | Stage | Purpose |
|---|-------|---------|
| 1 | **Media extraction** (FFmpeg) | Pull a normalized audio track from video or mixed media |
| 2 | **Demucs vocal separation** *(optional)* | Strip background music/noise so speech dominates |
| 3 | **Denoising** | Further suppress residual noise before detection |
| 4 | **VAD** (Silero / WebRTC) | Detect speech regions, drop silence and non-speech |
| 5 | **Overlap-aware chunking** | Split long audio into model-sized windows with overlap + merge |
| 6 | **Context-aware decoding** | Decode each chunk with surrounding context to keep coherence |

---

## Engineering Decisions & Trade-offs

These are the deliberate choices behind the pipeline — and the reasoning for each:

- **Why Demucs *before* VAD?**
  Background music and noise cause VAD to mis-fire (false "speech" on instrumental sections).
  Separating vocals first makes voice-activity detection far cleaner, which improves every
  downstream stage.

- **Why 4-bit quantization + LoRA instead of full fine-tuning?**
  Full fine-tuning of Gemma 3N needs more VRAM than a single consumer GPU offers. 4-bit loading
  plus LoRA adapters cuts memory dramatically while keeping quality high — the **7.21% WER**
  confirms the accuracy trade-off is negligible for this task.

- **Why overlap-aware chunking?**
  Naïvely slicing long audio cuts words/sentences at boundaries and loses them. Overlapping
  windows with a merge step prevent boundary word-loss and keep transcripts continuous.

- **Why track *empty predictions* and *word-count drift* — not just WER?**
  A single aggregate WER can hide systematic failures (e.g. the model silently emitting nothing).
  Logging 0 empty predictions and near-matched word counts (96,794 predicted vs 97,279 reference)
  proves the model is actually transcribing, not gaming the metric.

---

## Results

| Metric | Value |
|--------|-------|
| **Word Error Rate (WER)** | **7.21%** (0.0721) |
| Test samples | 5,000 |
| Empty predictions | 0 |
| Total reference words | 97,279 |
| Total predicted words | 96,794 |
| Avg reference length | 19.46 words |
| Avg predicted length | 19.36 words |

Evaluation reports include normalized reference/prediction text pairs for manual error analysis.

---

## Project Structure

```
Audio2Text/
├── train.py             # PEFT fine-tuning entry point
├── evaluate.py          # WER + diagnostic evaluation
├── pipline_predict.py   # End-to-end inference pipeline
├── src/                 # Reusable modules (data, model, pipeline)
├── scripts/             # CLI wrappers (run_*.py)
├── results/             # Evaluation outputs & reports
└── requirements.txt
```

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Fine-tune
python train.py            # or: python scripts/run_train.py

# 3. Evaluate
python evaluate.py         # reports WER + diagnostics

# 4. Transcribe audio/video
python pipline_predict.py --input path/to/media.mp4
```

---

## Tech Stack

- **Model / Training:** Gemma 3N, Unsloth (`FastModel`), PEFT/LoRA, 4-bit quantization, Hugging Face
- **Speech / Audio:** Demucs, Silero VAD, WebRTC-VAD, FFmpeg, Librosa, SoundFile
- **Language:** Python

---

*Built and maintained by [Nguyen Trung Hieu](https://github.com/HieuNTg).*
