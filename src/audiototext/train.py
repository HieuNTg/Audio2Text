"""
Moved training script (originally at project root).
"""

# Original file moved into package for better repo layout.

import os
import re
import numpy as np
import torch
from datasets import load_dataset, Audio
from unsloth import FastModel
from trl import SFTTrainer, SFTConfig


def load_model_and_processor(model_name="unsloth/gemma-3n-E4B-it", max_seq_length=1024):
    """Load the pretrained model and processor"""
    print("Loading model and processor...")
    
    model, processor = FastModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        load_in_4bit=True,  # 4-bit quantization for memory efficiency
        full_finetuning=False
    )
    
    print("Model and processor loaded successfully!")
    return model, processor


def prepare_dataset(dataset_name="doof-ferb/vlsp2020_vinai_100h", num_samples=40000):
    """Load and prepare the audio dataset"""
    print(f"Loading dataset: {dataset_name}")
    
    # Load dataset
    dataset = load_dataset(dataset_name, split="train")
    
    # Select subset of samples
    dataset = dataset.select(range(num_samples))
    
    # Cast audio column to proper format
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
    
    print(f"Dataset prepared with {len(dataset)} samples")
    return dataset


def format_intersection_data(samples: dict) -> dict:
    """Format dataset samples into chat-style messages with audio"""
    formatted_samples = {"messages": [], "audio": []}

    for idx in range(len(samples["audio"])):
        # Process audio
        audio = samples["audio"][idx]["array"]
        if not isinstance(audio, np.ndarray):
            audio = np.array(audio, dtype=np.float32)
        elif audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        # Get transcription label
        label = str(samples["transcription"][idx]).strip()

        # Create chat message format
        message = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "You are an assistant that transcribes Vietnamese speech accurately.",
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio": audio},
                    {"type": "text", "text": "Please transcribe this audio."}
                ]
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": label}]
            }
        ]
        
        formatted_samples["messages"].append(message)
        formatted_samples["audio"].append(samples["audio"][idx])

    return formatted_samples


def setup_peft_model(model):
    """Configure PEFT (Parameter-Efficient Fine-Tuning) for the model"""
    print("Setting up PEFT configuration...")
    
    model = FastModel.get_peft_model(
        model,
        finetune_vision_layers=False,  # Don't fine-tune audio encoder
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        r=16,
        lora_alpha=32,
        lora_dropout=0,
        use_gradient_checkpointing="unsloth",
        bias="none",
        random_state=3407,
        use_rslora=True,
        loftq_config=None,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
            "post", "linear_start", "linear_end",
            "embedding_projection",
        ],
    )
    
    print("PEFT model configured successfully!")
    return model


def create_collate_fn(processor, model):
    """
    Create custom collate function for data batching.
    Đảm bảo masking chính xác: Chỉ gán nhãn cho phần assistant (bản dịch).
    """
    
    def collate_fn(examples):
        texts = []
        audios = []

        for example in examples:
            text = processor.apply_chat_template(
                example["messages"], 
                tokenize=False, 
                add_generation_prompt=False
            ).strip()
            texts.append(text)

            audio = example["audio"]["array"]
            if isinstance(audio, np.ndarray):
                if audio.dtype != np.float32:
                    audio = audio.astype(np.float32)
            audios.append(audio)

        batch = processor(
            text=texts, 
            audio=audios, 
            return_tensors="pt", 
            padding=True, 
            truncation=True
        )
        
        device = model.device
        for key in batch.keys():
             if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(device)
                if key == "input_features" and batch[key].dtype == torch.float32:
                    batch[key] = batch[key].to(dtype=torch.float16)
                elif key == "input_features_mask" or key in ["input_ids", "attention_mask", "token_type_ids"]:
                    pass
                elif batch[key].dtype in [torch.float32, torch.float64]:
                    batch[key] = batch[key].to(dtype=torch.float16)

        labels = batch["input_ids"].clone()
        prompt_messages = examples[0]["messages"][:2]
        prompt_text_template = processor.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True
        ).strip()
        prompt_tokens = processor.tokenizer(
            prompt_text_template, 
            return_tensors="pt"
        )
        prompt_len = prompt_tokens["input_ids"].shape[1]

        labels[labels == processor.tokenizer.pad_token_id] = -100
        for i in range(labels.shape[0]):
            if labels.shape[1] > prompt_len:
                labels[i, :prompt_len] = -100

        if hasattr(processor.tokenizer, 'image_token_id'):
            labels[labels == processor.tokenizer.image_token_id] = -100
        if hasattr(processor.tokenizer, 'audio_token_id'):
            labels[labels == processor.tokenizer.audio_token_id] = -100
        if hasattr(processor.tokenizer, 'boi_token_id'):
            labels[labels == processor.tokenizer.boi_token_id] = -100

        batch["labels"] = labels
        return batch
    
    return collate_fn


def create_trainer(model, processor, formatted_dataset, collate_fn, output_dir="outputs"):
    print("Creating trainer...")
    
    trainer = SFTTrainer(
        model=model,
        train_dataset=formatted_dataset,
        
        processing_class=processor.tokenizer,
        data_collator=collate_fn,
        
        args=SFTConfig(
            per_device_train_batch_size=2,
            per_device_eval_batch_size=2,
            gradient_accumulation_steps=8,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            warmup_ratio=0.1,
            num_train_epochs=3,
            learning_rate=5e-6,
            logging_steps=10,
            save_strategy="steps",
            save_steps = 100,
            optim="adamw_8bit",
            weight_decay=0.05,
            lr_scheduler_type="cosine",
            seed=3407,
            output_dir=output_dir,
            use_cpu=False,
            dataloader_pin_memory=False,
            report_to="none",
            remove_unused_columns=False,
            dataset_text_field="",
            dataset_kwargs={"skip_prepare_dataset": True},
            dataset_num_proc=2,
            max_length=1024,
            fp16=False,
            bf16=True,
        )
    )
    
    print("Trainer created successfully!")
    return trainer


def print_memory_stats():
    gpu_stats = torch.cuda.get_device_properties(0)
    start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
    print(f"\nGPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
    print(f"{start_gpu_memory} GB of memory reserved.")


def debug_batch(collate_fn, dataset, model):
    print("\n=== Debug Info ===")
    print(f"Model device: {model.device}")
    print(f"Model dtype: {next(model.parameters()).dtype}")

    print("\n=== Checking first batch ===")
    first_batch = collate_fn([dataset[0]])
    for key, value in first_batch.items():
        if isinstance(value, torch.Tensor):
            print(f"{key}: shape={value.shape}, dtype={value.dtype}, device={value.device}")


def main():
    model, processor = load_model_and_processor()
    dataset = prepare_dataset()
    print("Formatting dataset...")
    formatted_dataset = dataset.map(
        format_intersection_data, 
        batched=True, 
        batch_size=8, 
        num_proc=4
    )
    model = setup_peft_model(model)
    collate_fn = create_collate_fn(processor, model)
    print_memory_stats()
    trainer = create_trainer(model, processor, formatted_dataset, collate_fn)
    torch._dynamo.config.suppress_errors = True
    print("\n" + "="*50)
    print("Starting training...")
    print("="*50 + "\n")
    trainer_stats = trainer.train(resume_from_checkpoint = True)
    print("\n" + "="*50)
    print("Training completed!")
    print(f"Final loss: {trainer_stats.training_loss}")
    print("="*50)
    model.save_pretrained("final_model1")
    processor.save_pretrained("final_model1")
    print("\nModel saved to 'final_model' directory")


if __name__ == "__main__":
    main()
