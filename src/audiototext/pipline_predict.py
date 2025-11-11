"""
Audio/Video to Text Transcription Script - IMPROVED VERSION
Version: 4.1.0-VAD-Enhanced-Vocals

Features:
- (MỚI) Vocal separation (Demucs) để tách giọng khỏi nhạc nền
- (MỚI) Tùy chọn khử noise sau khi tách giọng
- VAD thông minh loại bỏ khoảng lặng
- Smart chunking chỉ chứa đoạn nói
- Xử lý gap dài, merge text thông minh, context accumulation
- Hỗ trợ file dài, Colab + local, GPU/CPU
"""

import os
import re
import gc
import subprocess
import difflib
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import torch
import librosa
from tqdm import tqdm
import webrtcvad

from unsloth import FastModel


class VoiceActivityDetector:
    """
    Enhanced Voice Activity Detection for smart audio segmentation.
    """

    def __init__(
        self,
        aggressiveness: int = 2,
        frame_duration_ms: int = 30,
        min_speech_duration_ms: int = 300,
        max_silence_duration_ms: int = 800,
        padding_duration_ms: int = 150
    ):
        if frame_duration_ms not in [10, 20, 30]:
            raise ValueError("frame_duration_ms must be 10, 20, or 30")

        self.aggressiveness = aggressiveness
        self.frame_duration_ms = frame_duration_ms
        self.min_speech_duration_ms = min_speech_duration_ms
        self.max_silence_duration_ms = max_silence_duration_ms
        self.padding_duration_ms = padding_duration_ms

        self.vad = webrtcvad.Vad(aggressiveness)

    def _frame_generator(self, audio: np.ndarray, sample_rate: int) -> List[bytes]:
        n = int(sample_rate * (self.frame_duration_ms / 1000.0))
        offset = 0
        duration = float(n) / sample_rate

        frames = []
        while offset + n <= len(audio):
            frame = audio[offset:offset + n]
            frame_int16 = (frame * 32767).astype(np.int16)
            frames.append(frame_int16.tobytes())
            offset += n

        return frames

    def _vad_collector(self, frames: List[bytes]) -> List[Tuple[float, float]]:
        if not frames:
            return []

        segment_start = None
        segments = []

        frame_duration_sec = self.frame_duration_ms / 1000.0
        max_silence_frames = int(self.max_silence_duration_ms / self.frame_duration_ms)
        min_speech_frames = int(self.min_speech_duration_ms / self.frame_duration_ms)

        silence_counter = 0
        speech_frames = 0

        for i, frame in enumerate(frames):
            is_speech = self.vad.is_speech(frame, 16000)

            if is_speech:
                if segment_start is None:
                    segment_start = i * frame_duration_sec
                speech_frames += 1
                silence_counter = 0
            else:
                if segment_start is not None:
                    silence_counter += 1
                    if silence_counter >= max_silence_frames:
                        segment_end = (i - silence_counter) * frame_duration_sec
                        if speech_frames >= min_speech_frames:
                            segments.append((segment_start, segment_end + frame_duration_sec))
                        segment_start = None
                        speech_frames = 0
                        silence_counter = 0

        if segment_start is not None and speech_frames >= min_speech_frames:
            segment_end = len(frames) * frame_duration_sec
            segments.append((segment_start, segment_end))

        return segments

    def _add_padding(self, segments: List[Tuple[float, float]], total_duration: float) -> List[Tuple[float, float]]:
        padding_sec = self.padding_duration_ms / 1000.0
        padded = []
        for start, end in segments:
            s = max(0, start - padding_sec)
            e = min(total_duration, end + padding_sec)
            padded.append((s, e))
        return padded

    def _merge_overlapping_segments(self, segments: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        if not segments:
            return []
        segments = sorted(segments, key=lambda x: x[0])
        merged = [segments[0]]
        for cs, ce in segments[1:]:
            ls, le = merged[-1]
            if cs <= le + 0.1:
                merged[-1] = (ls, max(le, ce))
            else:
                merged.append((cs, ce))
        return merged

    def detect_speech_segments(self, audio: np.ndarray, sample_rate: int = 16000) -> List[Tuple[float, float]]:
        if sample_rate != 16000:
            print("⚠️ VAD requires 16kHz audio, resampling...")
            audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=16000)
            sample_rate = 16000

        frames = self._frame_generator(audio, sample_rate)
        segments = self._vad_collector(frames)
        segments = self._add_padding(segments, len(audio) / sample_rate)
        segments = self._merge_overlapping_segments(segments)
        return segments

class AudioTranscriber:
    """
    Enhanced Audio/Video transcription with:
    - Vocal separation (optional)
    - VAD-based speech-only segmentation
    - Smart chunking + context-aware transcription
    """

    def __init__(
        self,
        checkpoint_path: str,
        sampling_rate: int = 16000,
        max_new_tokens: int = 8192,
        chunk_duration_sec: int = 30,
        overlap_sec: float = 0.5,
        use_context_limit: bool = False,
        max_context_chars: int = 200_000,
        context_summary_threshold: int = 300_000,
        # VAD parameters
        vad_aggressiveness: int = 2,
        vad_min_speech_duration_ms: int = 300,
        vad_max_silence_duration_ms: int = 800,
        vad_padding_duration_ms: int = 150,
        # Smart chunking
        min_chunk_duration_sec: float = 2.0,
        context_duplicate_threshold: float = 0.85,
        # NEW: Vocal separation + denoise
        use_vocal_separation: bool = True,
        vocal_separation_backend: str = "demucs",
        demucs_model: str = "htdemucs_ft",
        use_denoise: bool = False,
    ) -> None:

        self.checkpoint_path = checkpoint_path
        self.sampling_rate = sampling_rate
        self.max_new_tokens = max_new_tokens
        self.chunk_duration_sec = min(chunk_duration_sec, 30)
        self.overlap_sec = overlap_sec

        self.use_context_limit = use_context_limit
        self.max_context_chars = max_context_chars
        self.context_summary_threshold = context_summary_threshold
        self.context_duplicate_threshold = context_duplicate_threshold

        self.vad_aggressiveness = vad_aggressiveness
        self.min_chunk_duration_sec = min_chunk_duration_sec
        self.context_leak_prefix_chars = 80
        self.context_leak_suffix_slack = 10
        self.context_leak_min_chars = 80

        # NEW:
        self.use_vocal_separation = use_vocal_separation
        self.vocal_separation_backend = vocal_separation_backend
        self.demucs_model = demucs_model
        self.use_denoise = use_denoise

        # VAD
        self.vad = VoiceActivityDetector(
            aggressiveness=vad_aggressiveness,
            min_speech_duration_ms=vad_min_speech_duration_ms,
            max_silence_duration_ms=vad_max_silence_duration_ms,
            padding_duration_ms=vad_padding_duration_ms,
        )

        # State
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.processor = None
        self._silero_model = None
        self._silero_get_timestamps = None
        self._silero_available = False
        self._load_silero_vad()

        self._print_config()

    # ==================== Config / Device ====================

    def _print_config(self) -> None:
        print("🔧 Enhanced AudioTranscriber Configuration:")
        print(f"   • Device: {self.device}")
        print(f"   • VAD: ENABLED (aggr: {self.vad_aggressiveness})")
        print(f"   • Chunk: {self.chunk_duration_sec}s, min: {self.min_chunk_duration_sec}s, overlap: {self.overlap_sec}s")
        print(f"   • Context limit: {self.use_context_limit}")
        if self.use_context_limit:
            print(f"     ↳ Max context: {self.max_context_chars}, truncate at: {self.context_summary_threshold}")
        print(f"   • Vocal separation: {self.use_vocal_separation} (backend={self.vocal_separation_backend}, model={self.demucs_model})")
        print(f"   • Denoise: {self.use_denoise}")

    def load_model(self) -> None:
        print("\n" + "=" * 60)
        print("STEP 1: LOADING SPEECH-TO-TEXT MODEL")
        print("=" * 60)

        self._cleanup_memory()
        self._print_device_info()

        print(f"\n📦 Loading model from: {self.checkpoint_path}")
        try:
            self.model, self.processor = FastModel.from_pretrained(
                model_name=self.checkpoint_path,
                max_seq_length=65536,
                load_in_4bit=True,
                device_map={"": self.device},
            )
            if hasattr(self.model, "gradient_checkpointing_enable"):
                try:
                    self.model.gradient_checkpointing_enable()
                    print("✓ Gradient checkpointing enabled")
                except Exception:
                    pass

            self.model.eval()
            print("✅ Model loaded successfully!\n")
        except Exception as e:
            print(f"\n❌ Error loading model: {e}")
            raise

    def _cleanup_memory(self) -> None:
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        gc.collect()

    def _print_device_info(self) -> None:
        if "cuda" in self.device:
            gpu_name = torch.cuda.get_device_name()
            total_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"✓ GPU: {gpu_name}")
            print(f"✓ VRAM: {total_mem:.1f} GB")
        else:
            print("⚠️ No GPU detected, using CPU (slower)")

    # ==================== Silero VAD ====================

    def _load_silero_vad(self) -> None:
        try:
            model, utils = torch.hub.load(
                repo_or_dir="snakers4/silero-vad",
                model="silero_vad",
                force_reload=False
            )
            get_speech_timestamps = utils[0]
            self._silero_model = model.to("cpu")
            self._silero_get_timestamps = get_speech_timestamps
            self._silero_available = True
            print("✓ Silero VAD loaded successfully.")
        except Exception as e:
            print(f"⚠️ Could not load Silero VAD (using WebRTC fallback): {e}")
            self._silero_model = None
            self._silero_get_timestamps = None
            self._silero_available = False

    def _detect_speech_segments(self, audio_array: np.ndarray) -> List[Tuple[float, float]]:
        if self._silero_available and self._silero_get_timestamps:
            try:
                return self._detect_speech_segments_silero(audio_array)
            except Exception as e:
                print(f"   ⚠️ Silero VAD failed: {e}. Falling back to WebRTC.")
                self._silero_available = False
        return self.vad.detect_speech_segments(audio_array, self.sampling_rate)

    def _detect_speech_segments_silero(self, audio_array: np.ndarray) -> List[Tuple[float, float]]:
        if self._silero_model is None or self._silero_get_timestamps is None:
            return []
        waveform = torch.from_numpy(audio_array).float().to("cpu")
        if waveform.ndim > 1:
            waveform = waveform.mean(dim=-1)

        timestamps = self._silero_get_timestamps(
            waveform,
            self._silero_model,
            sampling_rate=self.sampling_rate
        )
        if not timestamps:
            return []

        segments = []
        for ts in timestamps:
            start = max(0.0, ts.get("start", 0) / self.sampling_rate)
            end = max(start, ts.get("end", 0) / self.sampling_rate)
            if end - start > 0:
                segments.append((start, end))

        total_duration = len(audio_array) / self.sampling_rate
        segments = self.vad._add_padding(segments, total_duration)
        segments = self.vad._merge_overlapping_segments(segments)
        return segments
    # ==================== Vocal Separation & Denoise ====================

    def _check_demucs(self) -> bool:
        try:
            result = subprocess.run(
                ["demucs", "--help"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            return result.returncode == 0
        except FileNotFoundError:
            return False

    def _apply_vocal_separation(self, input_path: str) -> Optional[str]:
        if not self.use_vocal_separation:
            return None
        if self.vocal_separation_backend != "demucs":
            print("⚠️ Only 'demucs' backend is implemented currently.")
            return None
        if not self._check_demucs():
            print("⚠️ demucs not found in environment. Skipping vocal separation.")
            return None

        print("   🎼 Running Demucs vocal separation (vocals vs accompaniment)...")
        try:
            subprocess.run(
                [
                    "demucs",
                    "--two-stems=vocals",
                    "-n", self.demucs_model,
                    input_path,
                ],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except Exception as e:
            print(f"   ❌ Demucs failed: {e}")
            return None

        base = os.path.splitext(os.path.basename(input_path))[0]
        out_dir = os.path.join("separated", self.demucs_model, base)
        vocals_path = os.path.join(out_dir, "vocals.wav")

        if not os.path.exists(vocals_path):
            print(f"   ❌ Demucs output not found: {vocals_path}")
            return None

        print(f"   ✅ Vocal track extracted: {vocals_path}")
        return vocals_path

    def _apply_denoise_if_enabled(self, audio_array: np.ndarray) -> np.ndarray:
        if not self.use_denoise:
            return audio_array
        try:
            import noisereduce as nr
            reduced = nr.reduce_noise(y=audio_array, sr=self.sampling_rate)
            print("   ✓ Simple denoise applied.")
            return reduced.astype(np.float32)
        except Exception as e:
            print(f"   ⚠️ Denoise skipped (noisereduce missing or error: {e})")
            return audio_array

    # ==================== Audio I/O ====================

    def _check_ffmpeg(self) -> bool:
        result = subprocess.run("which ffmpeg", shell=True, capture_output=True)
        if result.returncode != 0:
            print("⚠️ ffmpeg not found, trying to install...")
            try:
                subprocess.run(
                    "apt-get update && apt-get install -y ffmpeg",
                    shell=True,
                    check=False,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                return True
            except Exception as e:
                print(f"❌ Failed to install ffmpeg: {e}")
                return False
        return True

    def _extract_audio_with_ffmpeg(self, video_path: str, output_path: str) -> Optional[str]:
        ffmpeg_cmd = (
            f'ffmpeg -y -i "{video_path}" -vn -acodec pcm_s16le '
            f'-ar {self.sampling_rate} -ac 1 "{output_path}"'
        )
        print("   ⏳ Extracting audio from video...")
        try:
            subprocess.run(
                ffmpeg_cmd,
                shell=True,
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            return output_path
        except subprocess.CalledProcessError as e:
            print(f"❌ ffmpeg error: {e}")
            return None

    def _extract_audio_from_file(self, file_path: str) -> Optional[np.ndarray]:
        print(f"🎬 Processing file: {os.path.basename(file_path)}")

        if not os.path.exists(file_path):
            print(f"❌ File not found: {file_path}")
            return None

        file_ext = os.path.splitext(file_path)[1].lower()
        audio_temp = "/tmp/extracted_audio.wav"
        file_to_load = file_path

        video_formats = [".mp4", ".avi", ".mkv", ".mov", ".flv", ".webm"]
        if file_ext in video_formats:
            if not self._check_ffmpeg():
                return None
            tmp = self._extract_audio_with_ffmpeg(file_path, audio_temp)
            if tmp is None:
                return None
            file_to_load = tmp

        # Vocal separation (nếu bật)
        if self.use_vocal_separation:
            sep_path = self._apply_vocal_separation(file_to_load)
            if sep_path:
                file_to_load = sep_path
            else:
                print("   ⚠️ Using original mixed audio (vocal separation unavailable).")

        try:
            print("   ⏳ Loading audio (mono, 16kHz)...")
            audio_array, sr = librosa.load(
                file_to_load,
                sr=self.sampling_rate,
                mono=True
            )
            duration_sec = len(audio_array) / sr
            print(f"   ✅ Audio loaded: {duration_sec:.1f}s ({len(audio_array)} samples, {sr}Hz)")

            # Optional denoise sau khi đã là giọng
            audio_array = self._apply_denoise_if_enabled(audio_array)

            return audio_array.astype(np.float32)

        except Exception as e:
            print(f"❌ Error loading audio: {e}")
            return None

        finally:
            if os.path.exists(audio_temp):
                try:
                    os.remove(audio_temp)
                except Exception:
                    pass
    # ==================== Chunking (VAD-first) ====================

    def _split_audio_with_vad(self, audio_array: np.ndarray) -> List[np.ndarray]:
        total_duration = len(audio_array) / self.sampling_rate
        print(f"\n🎯 Using VAD to detect speech segments...")

        speech_segments = self._detect_speech_segments(audio_array)
        if not speech_segments:
            print("   ❌ No speech detected in audio!")
            return [audio_array]

        total_speech = sum(e - s for s, e in speech_segments)
        speech_ratio = total_speech / total_duration
        print(f"   ✅ Found {len(speech_segments)} segments")
        print(f"   📊 Speech ratio: {speech_ratio:.1%} ({total_speech:.1f}s / {total_duration:.1f}s)")

        sps = self.sampling_rate
        parts = []
        for s, e in speech_segments:
            ss = int(s * sps)
            ee = int(e * sps)
            if ee > ss:
                parts.append(audio_array[ss:ee])

        if not parts:
            print("   ❌ Empty after VAD trim, fallback to full audio.")
            return [audio_array]

        speech_audio = np.concatenate(parts)
        speech_dur = len(speech_audio) / sps

        if speech_dur <= self.chunk_duration_sec:
            print(f"   ✅ Speech-only audio {speech_dur:.1f}s <= {self.chunk_duration_sec}s → 1 chunk\n")
            return [speech_audio]

        samples_per_chunk = int(sps * self.chunk_duration_sec)
        overlap_samples = int(sps * self.overlap_sec)
        if overlap_samples >= samples_per_chunk:
            overlap_samples = samples_per_chunk // 3
            print(f"   ⚠️ Overlap too large, adjusted to {overlap_samples / sps:.2f}s")

        step = max(1, samples_per_chunk - overlap_samples)
        final_chunks = []
        idx = 0

        while idx < len(speech_audio):
            end = min(idx + samples_per_chunk, len(speech_audio))
            chunk = speech_audio[idx:end]
            dur = len(chunk) / sps

            if dur < self.min_chunk_duration_sec and final_chunks:
                merged = np.concatenate([final_chunks[-1], chunk])
                if len(merged) / sps <= self.chunk_duration_sec:
                    final_chunks[-1] = merged
                else:
                    final_chunks.append(chunk)
                break

            final_chunks.append(chunk)
            if end == len(speech_audio):
                break
            idx += step

        print(f"   ✅ Final: {len(final_chunks)} speech-only chunk(s)\n")
        return final_chunks

    def _split_audio_simple(self, audio_array: np.ndarray) -> List[np.ndarray]:
        total_duration = len(audio_array) / self.sampling_rate
        if total_duration <= self.chunk_duration_sec:
            print(f"\n✂️   Audio {total_duration:.1f}s <= {self.chunk_duration_sec}s → 1 chunk\n")
            return [audio_array]

        samples_per_chunk = int(self.sampling_rate * self.chunk_duration_sec)
        overlap_samples = int(self.sampling_rate * self.overlap_sec)
        if overlap_samples >= samples_per_chunk:
            overlap_samples = samples_per_chunk // 3
            print(f"   ⚠️ Overlap adjusted to {overlap_samples / self.sampling_rate:.2f}s")

        step = samples_per_chunk - overlap_samples
        chunks = []

        print(f"\n✂️   Simple split into ~{self.chunk_duration_sec}s chunks, overlap {self.overlap_sec}s")

        start = 0
        while start < len(audio_array):
            end = min(start + samples_per_chunk, len(audio_array))
            chunks.append(audio_array[start:end])
            if end == len(audio_array):
                break
            start += step

        print(f"   ✅ Total chunks: {len(chunks)}\n")
        return chunks

    def _split_audio_smart(self, audio_array: np.ndarray) -> List[np.ndarray]:
        try:
            return self._split_audio_with_vad(audio_array)
        except Exception as e:
            print(f"   ⚠️ VAD splitting failed: {e}")
            print("   🔄 Falling back to simple splitting...")
            return self._split_audio_simple(audio_array)

    # ==================== Chunk validation & speech check ====================

    def _validate_chunks(self, chunks: List[np.ndarray]) -> bool:
        print("\n🔍 Validating chunks...")
        issues = []
        for i, c in enumerate(chunks):
            dur = len(c) / self.sampling_rate
            if len(c) == 0:
                issues.append(f"Chunk {i+1}: Empty")
            elif dur < 0.1:
                issues.append(f"Chunk {i+1}: Too short ({dur:.2f}s)")
        if issues:
            print("   ❌ Issues:")
            for m in issues:
                print("     -", m)
            return False
        print("   ✅ All chunks valid")
        return True

    def _chunk_has_speech(self, chunk: np.ndarray) -> bool:
        if chunk.size == 0:
            return False
        energy = float(np.mean(chunk ** 2))
        if energy < 1e-6:
            return False
        try:
            segs = self._detect_speech_segments(chunk)
            return any((e - s) >= 0.2 for s, e in segs)
        except Exception:
            return energy > 1e-4
    # ==================== Transcription core ====================

    def _truncate_context(self, context_text: str) -> str:
        if len(context_text) <= self.max_context_chars:
            return context_text
        truncated = "..." + context_text[-(self.max_context_chars - 3):]
        first_period = truncated.find(". ", 3)
        if 0 < first_period < 100:
            truncated = "..." + truncated[first_period + 2:]
        return truncated

    def _manage_context(self, context_text: str) -> str:
        if self.use_context_limit and len(context_text) > self.context_summary_threshold:
            print(f"   ⚙️ Context too long ({len(context_text)} chars) → truncating...")
            new_ctx = self._truncate_context(context_text)
            print(f"   ✓ Context after truncation: {len(new_ctx)} chars")
            return new_ctx
        return context_text

    def _transcribe_chunk(self, audio_chunk: np.ndarray, context_text: str = "") -> str:
        if context_text:
            context_to_use = self._truncate_context(context_text) if self.use_context_limit else context_text
            user_prompt = (
                "Ngữ cảnh trước đó (KHÔNG chép lại, chỉ dùng để giữ mạch):\n"
                f"{context_to_use}\n"
                "-----\n"
                "YÊU CẦU:\n"
                "- Chỉ ghi nội dung đoạn audio mới.\n"
                "- Giữ đúng tiếng Việt, đúng ngữ pháp.\n"
                "- Không đoán nếu không nghe rõ (ghi [không nghe rõ])."
            )
        else:
            user_prompt = (
                "Hãy chép lại chính xác nội dung tiếng Việt trong đoạn audio này.\n"
                "- Giữ nguyên tiếng Việt, không dịch.\n"
                "- Trình bày liền mạch, đúng chính tả.\n"
                "- Nếu câu dang dở, giữ nguyên dang dở."
            )

        messages = [
            {
                "role": "system",
                "content": [{
                    "type": "text",
                    "text": "Bạn là chuyên gia gỡ băng tiếng Việt, ưu tiên độ chính xác, "
                            "không bỏ sót nội dung và không thêm bịa."
                }],
            },
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio": audio_chunk},
                    {"type": "text", "text": user_prompt},
                ],
            },
        ]

        try:
            prompt = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        except Exception as e:
            print(f"⚠️ Error applying chat template: {e}")
            return "[PROCESSING ERROR]"

        try:
            inputs = self.processor(
                text=[prompt],
                audio=[audio_chunk],
                return_tensors="pt",
                padding=True,
                truncation=False,
            ).to(self.device)
        except Exception as e:
            print(f"⚠️ Error processing input: {e}")
            return "[PROCESSING ERROR]"

        try:
            pad_id = getattr(getattr(self.processor, "tokenizer", None), "pad_token_id", 0)
            eos_id = getattr(getattr(self.processor, "tokenizer", None), "eos_token_id", 2)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    min_new_tokens=0,
                    do_sample=False,
                    num_beams=3,
                    pad_token_id=pad_id,
                    eos_token_id=eos_id,
                )

            decoded = self.processor.tokenizer.decode(
                outputs[0],
                skip_special_tokens=False
            )
            prediction = self._extract_model_response(decoded, inputs, outputs)
            prediction = re.sub(r"<[^>]+>", "", prediction).strip()
            return prediction

        except torch.cuda.OutOfMemoryError:
            dur = len(audio_chunk) / self.sampling_rate
            print(f"\n❌ OOM with {dur:.1f}s chunk. Giảm chunk_duration_sec hoặc context.")
            return "[OOM ERROR]"
        except Exception as e:
            print(f"\n⚠️ Transcription error: {e}")
            return f"[ERROR: {str(e)[:120]}]"
        finally:
            if "outputs" in locals():
                del outputs
            if "inputs" in locals():
                del inputs
            self._cleanup_memory()

    def _extract_model_response(self, decoded: str, inputs: Dict, outputs: torch.Tensor) -> str:
        pattern = r"<start_of_turn>model\s*\n\s*(.*?)(?=<end_of_turn>|<eos>|$)"
        match = re.search(pattern, decoded, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
        input_len = inputs.get("input_ids", torch.tensor([[0]])).shape[1]
        return self.processor.tokenizer.decode(
            outputs[0][input_len:],
            skip_special_tokens=True
        ).strip()

    # ==================== Merge & duplicate handling ====================

    def _tokenize_with_spans(self, text: str) -> Tuple[List[str], List[Tuple[int, int]]]:
        tokens = []
        spans = []
        for m in re.finditer(r"\S+", text):
            tokens.append(m.group())
            spans.append(m.span())
        return tokens, spans

    def _normalize_text(self, text: str) -> str:
        return re.sub(r"\s+", " ", text).strip().lower()

    def _is_context_duplicate(self, text: str, context_text: str) -> bool:
        nt = self._normalize_text(text)
        nc = self._normalize_text(context_text)
        if not nt:
            return True
        if not nc:
            return False
        if nt in nc:
            return True
        matcher = difflib.SequenceMatcher(None, nc, nt)
        match = matcher.find_longest_match(0, len(nc), 0, len(nt))
        coverage = match.size / max(1, len(nt))
        return coverage >= self.context_duplicate_threshold

    def _trim_context_leak(self, text: str, context_text: str) -> Tuple[str, bool]:
        if not text or not context_text:
            return text, False

        tl = text.lower()
        cl = context_text.lower()
        matcher = difflib.SequenceMatcher(None, tl, cl)
        match = matcher.find_longest_match(0, len(tl), 0, len(cl))
        if match.size == 0:
            return text, False

        suffix_ratio = match.size / max(1, len(tl))
        starts_after_new = match.a >= self.context_leak_prefix_chars
        reaches_end = match.a + match.size >= len(tl) - self.context_leak_suffix_slack
        large_enough = match.size >= self.context_leak_min_chars
        ratio_ok = suffix_ratio >= self.context_duplicate_threshold * 0.6

        if starts_after_new and reaches_end and large_enough and ratio_ok:
            trimmed = text[:match.a].rstrip()
            return trimmed, True
        return text, False

    def _find_best_overlap(self, text1: str, text2: str) -> int:
        if not text1 or not text2:
            return 0
        t1, _ = self._tokenize_with_spans(text1)
        t2, spans2 = self._tokenize_with_spans(text2)
        if not t1 or not t2:
            return 0
        max_window = min(len(t1), len(t2), 80)
        suffix = t1[-max_window:]
        best = 0
        for size in range(max_window, 0, -1):
            if suffix[-size:] == t2[:size]:
                best = size
                break
        if best < 3:
            return 0
        return spans2[best - 1][1]

    def _merge_dynamic_overlap(self, chunks: List[str]) -> str:
        cleaned = [c.strip() for c in chunks if c and c.strip()]
        if not cleaned:
            return ""
        merged = cleaned[0]
        for c in cleaned[1:]:
            drop = self._find_best_overlap(merged, c)
            addition = c[drop:].lstrip() if drop else c
            if not addition:
                continue
            if merged and not merged.endswith((" ", "\n")) and not addition.startswith((" ", "\n")):
                merged += " "
            merged += addition
        merged = re.sub(r"\s+\n", "\n", merged)
        merged = re.sub(r"\n{3,}", "\n\n", merged)
        merged = re.sub(r"[ \t]{2,}", " ", merged)
        return merged.strip()
    # ==================== Main Workflow ====================

    def transcribe_file(self, file_path: str) -> Tuple[str, List[int], List[int], List[int]]:
        print("\n" + "=" * 60)
        print("STEP 3: ENHANCED TRANSCRIPTION WITH VAD + VOCAL SEPARATION")
        print("=" * 60)

        audio_array = self._extract_audio_from_file(file_path)
        if audio_array is None:
            return "[AUDIO EXTRACTION FAILED]", [], [], []

        chunks = self._split_audio_smart(audio_array)
        if not self._validate_chunks(chunks):
            return "[CHUNK VALIDATION FAILED]", [], [], []

        print(f"🔄 Transcribing {len(chunks)} chunk(s) with context accumulation...\n")

        transcriptions = []
        failed_chunks = []
        skipped_chunks = []
        duplicate_chunks = []
        accumulated_context = ""

        for i, chunk in enumerate(tqdm(chunks, desc="Transcribing", ncols=80)):
            dur = len(chunk) / self.sampling_rate
            tqdm.write(f"\n📍 Chunk {i+1}/{len(chunks)} ({dur:.1f}s)")
            if accumulated_context:
                tqdm.write(f"   📚 Context length: {len(accumulated_context)} chars")

            if not self._chunk_has_speech(chunk):
                skipped_chunks.append(i + 1)
                tqdm.write("   ⏭️  No speech detected, skipping chunk.")
                continue

            text = self._transcribe_chunk(chunk, context_text=accumulated_context).strip()

            if "[ERROR" in text or "[OOM" in text or text == "[PROCESSING ERROR]":
                failed_chunks.append(i + 1)
                tqdm.write("   ❌ Chunk failed.")
                continue
            if not text:
                duplicate_chunks.append(i + 1)
                tqdm.write("   ⏭️  Empty transcription.")
                continue

            trimmed_text, trimmed = self._trim_context_leak(text, accumulated_context)
            if not trimmed_text:
                duplicate_chunks.append(i + 1)
                tqdm.write("   ⏭️  Fully overlapped after trim.")
                continue
            if trimmed:
                tqdm.write("   ✂️  Trimmed repeated context from output.")

            transcriptions.append(trimmed_text)
            tqdm.write(f"   ✅ OK: {len(trimmed_text)} chars")
            tqdm.write(f"   ▶️ Predict: {trimmed_text}")

            accumulated_context += " " + trimmed_text
            accumulated_context = self._manage_context(accumulated_context)

        print("\n🔄 Merging final text...")
        if not transcriptions:
            print("❌ No successful transcriptions to merge.")
            final_text = "[NO TRANSCRIPTION DATA]"
        else:
            final_text = self._merge_dynamic_overlap(transcriptions)
            print("✅ Merging complete.")

        total_duration = len(audio_array) / self.sampling_rate
        self._print_summary(
            duration=total_duration,
            num_chunks=len(chunks),
            failed_chunks=failed_chunks,
            skipped_chunks=skipped_chunks,
            duplicate_chunks=duplicate_chunks,
            text=final_text
        )

        return final_text, failed_chunks, skipped_chunks, duplicate_chunks

    def _print_summary(
        self,
        duration: float,
        num_chunks: int,
        failed_chunks: List[int],
        skipped_chunks: List[int],
        duplicate_chunks: List[int],
        text: str
    ) -> None:
        print("\n" + "=" * 60)
        print("✅ ENHANCED TRANSCRIPTION COMPLETE!")
        print("=" * 60)
        print(f"📊 Audio Duration: {duration:.1f}s")
        print(
            f"📊 Total Chunks: {num_chunks} "
            f"({len(failed_chunks)} failed, {len(skipped_chunks)} skipped, {len(duplicate_chunks)} duplicate)"
        )
        print("📊 VAD-Optimized: Yes (speech-only chunks)")
        print(f"📊 Output: {len(text)} chars, ~{len(text.split())} words")
        if failed_chunks:
            print(f"   ⚠️ Failed chunks: {failed_chunks}")
        if skipped_chunks:
            print(f"   ⏭️  Skipped (no speech): {skipped_chunks}")
        if duplicate_chunks:
            print(f"   🔁 Duplicate predictions skipped: {duplicate_chunks}")
        print("\n" + "-" * 60)
        print("📄 PREVIEW (first 1000 chars):\n")
        print(text[:1000] + ("..." if len(text) > 1000 else ""))
        print("\n" + "=" * 60)

# ==================== Main ====================

def main() -> None:
    print("""
╔══════════════════════════════════════════════════════════════╗
║   🎙️  VIETNAMESE AUDIO/VIDEO → TEXT - V4.1 VOCALS+VAD  🎙️   ║
╚══════════════════════════════════════════════════════════════╝
""")

    config = {
        "checkpoint_path": "/content/drive/MyDrive/Audio2Text",
        "sampling_rate": 16000,
        "max_new_tokens": 8192,
        "chunk_duration_sec": 30,
        "overlap_sec": 0.5,
        "use_context_limit": False,
        "max_context_chars": 200_000,
        "context_summary_threshold": 300_000,
        # VAD
        "vad_aggressiveness": 2,
        "vad_min_speech_duration_ms": 300,
        "vad_max_silence_duration_ms": 800,
        "vad_padding_duration_ms": 150,
        # Smart chunk
        "min_chunk_duration_sec": 2.0,
        "context_duplicate_threshold": 0.85,
        # NEW: vocal separation + denoise
        "use_vocal_separation": True,          # bật tách giọng
        "vocal_separation_backend": "demucs",  # dùng Demucs
        "demucs_model": "htdemucs_ft",         # hoặc "htdemucs"
        "use_denoise": True,                  # nếu muốn khử noise thêm, chuyển True (cần noisereduce)
    }

    try:
        transcriber = AudioTranscriber(**config)
    except Exception as e:
        print(f"❌ Initialization error: {e}")
        return

    try:
        transcriber.load_model()
    except Exception as e:
        print(f"\n❌ Failed to load model: {e}")
        return

    print("\n" + "=" * 60)
    print("STEP 2: SELECT/UPLOAD FILE")
    print("=" * 60)
    print("📤 Supported: MP4, MP3, WAV, M4A, AVI, MKV, FLAC, etc.")
    print("🎯 Pipeline: [Video/Audio] → (Demucs vocals) → VAD → Chunk → STT\n")

    try:
        inp = input("Enter audio/video file path: ").strip().strip('"').strip("'")
        if not inp:
            print("❌ Empty path!")
            return
        if not os.path.exists(inp):
            print(f"❌ File not found: {inp}")
            return
        file_path = inp
    except KeyboardInterrupt:
        print("\n⛔ Cancelled.")
        return

    try:
        result, failed_chunks, skipped_chunks, duplicate_chunks = transcriber.transcribe_file(file_path)

        if result and not result.startswith("["):
            print("\n" + "=" * 60)
            print("STEP 4: SAVING RESULTS")
            print("=" * 60)

            base_name = os.path.splitext(os.path.basename(file_path))[0]
            output_dir = os.getcwd()
            output_file = os.path.join(output_dir, f"{base_name}_transcript_v4_1_vad_vocals.txt")

            with open(output_file, "w", encoding="utf-8") as f:
                f.write(f"FILE: {os.path.basename(file_path)}\n")
                f.write("VERSION: 4.1.0-VAD-Enhanced-Vocals\n")
                f.write("PIPELINE: Demucs (vocals) → VAD → smart chunk → context-aware STT\n")
                f.write(f"CONFIG: {config}\n")
                f.write("=" * 60 + "\n\n")
                f.write(result)

                if failed_chunks:
                    f.write("\n\n" + "=" * 60 + "\n")
                    f.write(f"⚠️ FAILED CHUNKS: {failed_chunks}\n")
                if skipped_chunks:
                    f.write("\n\n" + "=" * 60 + "\n")
                    f.write(f"⏭️  SKIPPED (no speech): {skipped_chunks}\n")
                if duplicate_chunks:
                    f.write("\n\n" + "=" * 60 + "\n")
                    f.write(f"🔁 DUPLICATE (context repetition): {duplicate_chunks}\n")

            print(f"✅ Saved to: {output_file}")
            try:
                size_kb = os.path.getsize(output_file) / 1e3
                print(f"📊 File size: {size_kb:.1f} KB")
            except Exception:
                pass

        print("\n🎉 DONE!")

    except Exception as e:
        print(f"\n❌ Critical error during transcription: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
