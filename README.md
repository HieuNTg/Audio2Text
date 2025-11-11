# Audio2Text

Vietnamese end‑to‑end toolkit for fine‑tuning, evaluating, and deploying Gemma 3N automatic speech recognition. This repository consolidates training, evaluation, and inference pipelines together with ready‑made scripts so others can reproduce the workflow or extend it for their own datasets.

## Project Goals
- Provide a transparent reference implementation for Gemma 3N fine‑tuning on Vietnamese audio.
- Ship a production‑ready transcription pipeline (Demucs → VAD → smart chunking → context‑aware decoding).
- Keep results, scripts, and code organized so the repo doubles as a living portfolio piece for ASR projects on GitHub.

## Key Features
- **Training**: PEFT fine‑tuning with `unsloth.FastModel`, chat‑style prompts, 4‑bit loading, dataset helpers.
- **Evaluation**: Automated WER computation, configurable dataset slices, pretty report saved to `results/`.
- **Inference**: Full audio/video pipeline with optional Demucs vocal separation, denoise, Silero/WebRTC VAD, and overlap‑aware merging.
- **CLI wrappers**: `scripts/run_*.py` let you run everything from the project root without tweaking `PYTHONPATH`.

## Repository Layout
```
Audio2Text/
├─ README.md
├─ LICENSE
├─ requirements.txt
├─ .gitignore
├─ results/
│  └─ evaluation_results.txt      # Sample WER report (5k utterances)
├─ scripts/
│  ├─ run_train.py                # Calls audiototext.train
│  ├─ run_evaluate.py             # Calls audiototext.evaluate
│  └─ run_transcribe.py           # Calls audiototext.pipline_predict
├─ train.py / evaluate.py / pipline_predict.py  # Legacy wrappers (still runnable)
└─ src/audiototext/
   ├─ __init__.py
   ├─ train.py                    # Core training pipeline
   ├─ evaluate.py                 # Evaluation + metrics report writer
   └─ pipline_predict.py          # Advanced inference pipeline
```

Legacy scripts at the project root simply forward to the package implementation so older commands like `python train.py` continue to work.

## Quick Start (Windows PowerShell)
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt

# Required for extracting audio from video
choco install ffmpeg -y

# Optional but recommended for vocal separation
pip install demucs
```

> Tip: if VRAM is tight, keep the default 4‑bit configuration in `src/audiototext/train.py` and reduce `batch_size`.

## Usage

### Fine-tune Gemma 3N
```powershell
python scripts/run_train.py
```
Adjust dataset name, sample count, or PEFT settings in `src/audiototext/train.py`.

### Evaluate (WER report)
```powershell
# Point to your checkpoint (defaults to ./checkpoints/final_model1 if not set)
setx A2T_CHECKPOINT_DIR "D:\models\final_model1"

python scripts/run_evaluate.py
```
The script processes 5k samples (configurable) and writes `results/evaluation_results.txt`.

### Transcribe audio/video
```powershell
python scripts/run_transcribe.py
```
The pipeline handles video extraction → Demucs (optional) → VAD → chunking → transcription → smart merging. Edit the config block near the bottom of `src/audiototext/pipline_predict.py` to toggle features or change defaults.

## Sample Evaluation Results
(`results/evaluation_results.txt`)

| Metric                | Value        |
|-----------------------|------------- |
| Word Error Rate       | 0.0721 (7.21%) |
| Samples               | 5,000        |
| Empty predictions     | 0            |
| Total ref words       | 97,279       |
| Total pred words      | 96,794       |
| Avg ref length        | 19.46 words  |
| Avg pred length       | 19.36 words  |

The report file also includes every REF/PRED pair plus normalized text for manual error scanning.

## Configuration Notes
- **Checkpoints**: place your fine‑tuned weights under `checkpoints/final_model1` or set `A2T_CHECKPOINT_DIR`.
- **Datasets**: update `prepare_dataset()` (train) and `prepare_evaluation_dataset()` (evaluate) for alternative corpora.
- **Results**: everything is saved under `results/` so it can be committed or shared as evidence of performance.
- **External deps**: `ffmpeg`, `demucs`, `webrtcvad`, `unsloth`, `noisereduce` (optional). Install them before running the respective feature.

## Troubleshooting
- **CUDA OOM**: lower `batch_size`, enable gradient accumulation, or run inference on smaller chunks.
- **Demucs not found**: install via `pip install demucs` and ensure it is on PATH (restart PowerShell if needed).
- **Results not written**: Windows may lock the `results/` folder if opened in Explorer—close the folder or run PowerShell as admin.

## Contributing
Issues and pull requests are welcome. Please describe the dataset/config you used so others can reproduce the change.

## License
[MIT](LICENSE)
