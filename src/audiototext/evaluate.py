"""
Moved evaluation script into package.
"""

import os
import re
from pathlib import Path
import numpy as np
import torch
from datasets import load_dataset, Audio
from unsloth import FastModel
import evaluate
from tqdm import tqdm

# --- Cấu hình ---
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "results"
DEFAULT_CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints" / "final_model1"
CHECKPOINT_DIR = os.environ.get("A2T_CHECKPOINT_DIR", str(DEFAULT_CHECKPOINT_DIR))
DATASET_NAME = "doof-ferb/vlsp2020_vinai_100h"
NUM_SAMPLES = 5000
MAX_NEW_TOKENS = 2024

def setup_gpu():
    if torch.cuda.is_available():
        print(f"✓ CUDA available")
        print(f"✓ GPU: {torch.cuda.get_device_name()}")
        torch.cuda.empty_cache()
        device = "cuda"
    else:
        print("⚠ Using CPU")
        device = "cpu"
    return device

def load_model_and_processor(checkpoint_path):
    print(f"\n📦 Loading model from: {checkpoint_path}")
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    model, processor = FastModel.from_pretrained(
        model_name=checkpoint_path,
        max_seq_length=1024,
        load_in_4bit=True,
        device_map={"": "cuda:0"} if torch.cuda.is_available() else {"": "cpu"}
    )
    
    model.eval()
    print("✓ Model loaded successfully!")
    return model, processor

def prepare_evaluation_dataset(dataset_name, num_samples, offset=40001):
    print(f"\n📚 Loading dataset: {dataset_name}")
    dataset = load_dataset(dataset_name, split="train")
    total_length = len(dataset)
    start_index = offset
    end_index = offset + num_samples
    if start_index >= total_length:
        print(f"🚨 Lỗi: Offset ({start_index}) nằm ngoài độ dài dataset ({total_length}). Sẽ trả về một dataset rỗng.")
        return dataset.select(range(0))
    if end_index > total_length:
        print(f"⚠️ Cảnh báo: Yêu cầu {num_samples} mẫu, nhưng chỉ còn {total_length - start_index} mẫu. Sẽ lấy các mẫu đến cuối dataset.")
        end_index = total_length
    print(f"Selecting samples from index {start_index} to {end_index - 1}...")
    dataset = dataset.select(range(start_index, end_index))
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
    print(f"✓ Prepared {len(dataset)} samples")
    return dataset

def normalize_text(text):
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def transcribe_sample(model, processor, audio_array):
    if not isinstance(audio_array, np.ndarray):
        audio_array = np.array(audio_array, dtype=np.float32)
    elif audio_array.dtype != np.float32:
        audio_array = audio_array.astype(np.float32)
    messages = [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": "You are an assistant that transcribes Vietnamese speech accurately."
                }
            ]
        },
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio": audio_array},
                {"type": "text", "text": "Please transcribe this audio."}
            ]
        }
    ]
    prompt = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    inputs = processor(
        text=[prompt],
        audio=[audio_array],
        return_tensors="pt",
        padding=True,
        truncation=True
    )
    for key in inputs.keys():
        if isinstance(inputs[key], torch.Tensor):
            inputs[key] = inputs[key].to(model.device)
    try:
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=processor.tokenizer.pad_token_id,
                eos_token_id=processor.tokenizer.eos_token_id,
            )
        decoded_output = processor.tokenizer.decode(outputs[0], skip_special_tokens=False)
        patterns = [
            r"<start_of_turn>model\s*\n\s*(.*?)(?:<end_of_turn>|<eos>|$)",
            r"assistant\s*\n\s*(.*?)(?:<end_of_turn>|<eos>|$)",
        ]
        prediction = ""
        for pattern in patterns:
            match = re.search(pattern, decoded_output, re.DOTALL | re.IGNORECASE)
            if match:
                prediction = match.group(1).strip()
                prediction = re.sub(r'<[^>]+>', '', prediction).strip()
                break
        if not prediction:
            input_length = inputs['input_ids'].shape[1]
            generated_tokens = outputs[0][input_length:]
            prediction = processor.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
        return prediction
    except Exception as e:
        print(f"\n⚠ Error during generation: {e}")
        return ""

def calculate_wer(predictions, references):
    wer_metric = evaluate.load("wer")
    norm_preds = [normalize_text(p) for p in predictions]
    norm_refs = [normalize_text(r) for r in references]
    wer_score = wer_metric.compute(predictions=norm_preds, references=norm_refs)
    empty_preds = sum(1 for p in predictions if not p.strip())
    total_ref_words = sum(len(r.split()) for r in norm_refs)
    total_pred_words = sum(len(p.split()) for p in norm_preds)
    return {
        'wer': wer_score,
        'empty_predictions': empty_preds,
        'total_ref_words': total_ref_words,
        'total_pred_words': total_pred_words,
        'avg_ref_length': total_ref_words / len(norm_refs) if norm_refs else 0,
        'avg_pred_length': total_pred_words / len(norm_preds) if norm_preds else 0,
    }

def save_detailed_results(output_dir: Path, metrics, predictions, references):
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "evaluation_results.txt"
    try:
        with output_file.open("w", encoding="utf-8") as f:
            f.write("="*70 + "\n")
            f.write("EVALUATION RESULTS - GEMMA 3N ASR\n")
            f.write("="*70 + "\n\n")
            f.write(f"WER Score: {metrics['wer']:.4f} ({metrics['wer']*100:.2f}%)\n")
            f.write(f"Total Samples: {len(predictions)}\n")
            f.write(f"Empty Predictions: {metrics['empty_predictions']}\n")
            f.write(f"Total Reference Words: {metrics['total_ref_words']}\n")
            f.write(f"Total Predicted Words: {metrics['total_pred_words']}\n")
            f.write(f"Avg Reference Length: {metrics['avg_ref_length']:.2f} words\n")
            f.write(f"Avg Prediction Length: {metrics['avg_pred_length']:.2f} words\n")
            f.write("\n" + "="*70 + "\n\n")
            for i, (ref, pred) in enumerate(zip(references, predictions), 1):
                f.write(f"Sample {i}:\n")
                f.write(f"  REF:  {ref}\n")
                f.write(f"  PRED: {pred}\n")
                f.write(f"  NORM REF:  {normalize_text(ref)}\n")
                f.write(f"  NORM PRED: {normalize_text(pred)}\n")
                f.write("-" * 70 + "\n")
        print(f"\nResults saved to: {output_file}")
    except Exception as e:
        print(f"\nCould not save results: {e}")

def main():
    print("="*70)
    print("GEMMA 3N ASR - EVALUATION SCRIPT")
    print("="*70)
    device = setup_gpu()
    try:
        model, processor = load_model_and_processor(CHECKPOINT_DIR)
    except Exception as e:
        print(f"\n✗ Error loading model: {e}")
        return
    try:
        eval_dataset = prepare_evaluation_dataset(DATASET_NAME, NUM_SAMPLES)
    except Exception as e:
        print(f"\n✗ Error loading dataset: {e}")
        return
    predictions = []
    references = []
    print("\n" + "="*70)
    print("STARTING EVALUATION")
    print("="*70 + "\n")
    for idx, sample in enumerate(tqdm(eval_dataset, desc="Transcribing")):
        try:
            audio_array = sample["audio"]["array"]
            reference_text = sample["transcription"]
            predicted_text = transcribe_sample(model, processor, audio_array)
            predictions.append(predicted_text)
            references.append(reference_text)
            if idx < 3:
                print(f"\n[Debug Sample {idx+1}]")
                print(f"Audio shape: {audio_array.shape}")
                print(f"Ref:  {reference_text[:80]}...")
                print(f"Pred: {predicted_text[:80]}...")
        except Exception as e:
            print(f"\n⚠ Error at sample {idx}: {e}")
            predictions.append("")
            references.append(sample.get("transcription", ""))
    print("\n" + "="*70)
    print("CALCULATING METRICS")
    print("="*70)
    metrics = calculate_wer(predictions, references)
    print("\n" + "="*70)
    print("FINAL RESULTS")
    print("="*70)
    print(f"📊 WER Score: {metrics['wer']:.4f} ({metrics['wer']*100:.2f}%)")
    print(f"📝 Total Samples: {len(predictions)}")
    print(f"⚠ Empty Predictions: {metrics['empty_predictions']}")
    print(f"📏 Avg Reference Length: {metrics['avg_ref_length']:.2f} words")
    print(f"📏 Avg Prediction Length: {metrics['avg_pred_length']:.2f} words")
    print("="*70)
    print("\n--- SAMPLE PREDICTIONS ---")
    for i in range(min(5, len(predictions))):
        print(f"\n[Sample {i+1}]")
        print(f"Reference:  {references[i]}")
        print(f"Prediction: {predictions[i]}")
        print("-" * 50)
    save_detailed_results(RESULTS_DIR, metrics, predictions, references)
    print("\n✓ Evaluation complete!")

if __name__ == "__main__":
    main()
