import os
import time
from trainer import Trainer, TrainerArgs
from TTS.config.shared_configs import BaseDatasetConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.layers.xtts.trainer.gpt_trainer import GPTArgs, GPTTrainer, GPTTrainerConfig, XttsAudioConfig
from TTS.utils.manage import ModelManager

# Logging parameters
RUN_NAME = "GPT_XTTS_LJSpeech_FT"
PROJECT_NAME = "XTTS_trainer"
DASHBOARD_LOGGER = "tensorboard"
LOGGER_URI = None

# Output path
OUT_PATH = "finetune/outputs"

# Training Parameters
OPTIMIZER_WD_ONLY_ON_WEIGHTS = True
START_WITH_EVAL = False
BATCH_SIZE = 4
GRAD_ACUMM_STEPS = 4  # BATCH_SIZE * GRAD_ACUMM_STEPS ≈ 252 recommended

# Dataset
config_dataset = BaseDatasetConfig(
    formatter="ljspeech",
    dataset_name="ljspeech",
    path="data/",
    meta_file_train="metadata1.csv",
    language="ru",
)
DATASETS_CONFIG_LIST = [config_dataset]

# Checkpoint paths
CHECKPOINTS_OUT_PATH = os.path.join(OUT_PATH, "XTTS_v1.1_original_model_files/")
os.makedirs(CHECKPOINTS_OUT_PATH, exist_ok=True)

# Download helper with retry logic
def safe_download(links, path, retries=3, wait=5):
    """Download files with retries and progress bar"""
    for attempt in range(1, retries + 1):
        try:
            ModelManager._download_model_files(links, path, progress_bar=True)
            return
        except Exception as e:
            print(f"⚠️ Download attempt {attempt}/{retries} failed: {e}")
            if attempt < retries:
                print(f"⏳ Retrying in {wait} seconds...")
                time.sleep(wait)
            else:
                raise RuntimeError(f"❌ Failed to download after {retries} attempts. Please try manually.")

# DVAE + mel stats
DVAE_CHECKPOINT_LINK = "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v1/v1.1.2/dvae.pth"
MEL_NORM_LINK = "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v1/v1.1.2/mel_stats.pth"
DVAE_CHECKPOINT = os.path.join(CHECKPOINTS_OUT_PATH, os.path.basename(DVAE_CHECKPOINT_LINK))
MEL_NORM_FILE = os.path.join(CHECKPOINTS_OUT_PATH, os.path.basename(MEL_NORM_LINK))

if not (os.path.isfile(DVAE_CHECKPOINT) and os.path.isfile(MEL_NORM_FILE)):
    print(" > Downloading DVAE model files...")
    safe_download([MEL_NORM_LINK, DVAE_CHECKPOINT_LINK], CHECKPOINTS_OUT_PATH)
else:
    print("✅ DVAE model files found — skipping download.")

# XTTS model + tokenizer
TOKENIZER_FILE_LINK = "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v1/v1.1.2/vocab.json"
XTTS_CHECKPOINT_LINK = "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v1/v1.1.2/model.pth"
TOKENIZER_FILE = os.path.join(CHECKPOINTS_OUT_PATH, os.path.basename(TOKENIZER_FILE_LINK))
XTTS_CHECKPOINT = os.path.join(CHECKPOINTS_OUT_PATH, os.path.basename(XTTS_CHECKPOINT_LINK))

if not (os.path.isfile(TOKENIZER_FILE) and os.path.isfile(XTTS_CHECKPOINT)):
    print(" > Downloading XTTS model files...")
    safe_download([TOKENIZER_FILE_LINK, XTTS_CHECKPOINT_LINK], CHECKPOINTS_OUT_PATH)
else:
    print("✅ XTTS model files found — skipping download.")

# Training speaker reference
SPEAKER_REFERENCE = ["./data/wavs/01.wav"]
LANGUAGE = config_dataset.language

def main():
    model_args = GPTArgs(
        max_conditioning_length=132300,
        min_conditioning_length=66150,
        debug_loading_failures=True,
        max_wav_length=220500,
        max_text_length=200,
        mel_norm_file=MEL_NORM_FILE,
        dvae_checkpoint=DVAE_CHECKPOINT,
        xtts_checkpoint=XTTS_CHECKPOINT,
        tokenizer_file=TOKENIZER_FILE,
        gpt_num_audio_tokens=8194,
        gpt_start_audio_token=8192,
        gpt_stop_audio_token=8193,
    )

    audio_config = XttsAudioConfig(
        sample_rate=22050,
        dvae_sample_rate=22050,
        output_sample_rate=24000,
    )

    config = GPTTrainerConfig(
        output_path=OUT_PATH,
        model_args=model_args,
        run_name=RUN_NAME,
        project_name=PROJECT_NAME,
        run_description="GPT XTTS fine-tuning - NO EVAL",
        dashboard_logger=DASHBOARD_LOGGER,
        logger_uri=LOGGER_URI,
        audio=audio_config,
        batch_size=BATCH_SIZE,
        batch_group_size=0,
        eval_batch_size=BATCH_SIZE,
        num_loader_workers=0,
        eval_split_max_size=0,  
        print_step=1,
        plot_step=10,
        log_model_step=10,
        epochs=3,
        save_step=20,
        save_n_checkpoints=1,
        save_checkpoints=True,
        run_eval=False,
        optimizer="AdamW",
        optimizer_wd_only_on_weights=OPTIMIZER_WD_ONLY_ON_WEIGHTS,
        optimizer_params={"betas": [0.9, 0.96], "eps": 1e-8, "weight_decay": 1e-2},
        lr=5e-06,
        lr_scheduler="MultiStepLR",
        lr_scheduler_params={"milestones": [50000 * 18, 150000 * 18, 300000 * 18], "gamma": 0.5, "last_epoch": -1},
        test_sentences=[],  # ← Remove test sentences too
    )

    model = GPTTrainer.init_from_config(config)
    
    print("📊 Loading training data...")
    
    # Manual data loading to avoid tuple issues
    train_samples = []
    metadata_path = os.path.join("data", "metadata.csv")
    
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    parts = line.split('|')
                    if len(parts) >= 2:
                        audio_file = parts[0].strip()
                        text = parts[1].strip()
                        
                        # Ensure audio file has correct path
                        if not audio_file.startswith("wavs/"):
                            audio_file = os.path.join("wavs", audio_file)
                        if not audio_file.endswith(".wav"):
                            audio_file += ".wav"
                        
                        full_audio_path = os.path.join("data", audio_file)
                        
                        if os.path.exists(full_audio_path):
                            train_samples.append({
                                "audio_file": full_audio_path,
                                "text": text,
                                "speaker_name": "speaker",
                                "language": "ru"
                            })
    
    print(f"✅ Loaded {len(train_samples)} training samples")
    
    print("🚫 Evaluation completely disabled")

    trainer = Trainer(
        TrainerArgs(
            restore_path=None,
            skip_train_epoch=False,
            start_with_eval=False,  # ← No eval
            grad_accum_steps=GRAD_ACUMM_STEPS,
            
        ),
        config,
        output_path=OUT_PATH,
        model=model,
        train_samples=train_samples,
        eval_samples=[],  # ← Empty eval samples
    )
    trainer.fit()

if __name__ == "__main__":
    main()
