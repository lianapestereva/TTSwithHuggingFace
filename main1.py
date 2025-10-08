import os
import torch
import torchaudio
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
from utils import read_text_file

# ================================
# Paths
# ================================
RUN_PATH = "finetune/outputs/GPT_XTTS_LJSpeech_FT-October-07-2025_09+19PM-ec3246a"
CHECKPOINT_DIR = RUN_PATH 
CONFIG_PATH = os.path.join(RUN_PATH, "config.json")
XTTS_CHECKPOINT = os.path.join(RUN_PATH, "best_model.pth")
TOKENIZER_PATH = "finetune/outputs/XTTS_v1.1_original_model_files/vocab.json"

SPEAKER_REFERENCE = "voice.wav"
OUTPUT_WAV_PATH = "output13.wav"

# ================================
# Load model
# ================================
print("Loading fine-tuned XTTS model...")

torch.cuda.empty_cache()  

config = XttsConfig()
config.load_json(CONFIG_PATH)
model = Xtts.init_from_config(config)
model.load_checkpoint(
    config=config,
    checkpoint_dir=RUN_PATH,
    vocab_path=TOKENIZER_PATH,
    use_deepspeed=False
)
model.to("cpu")

# ================================
# Get conditioning latents
# ================================

print("Computing speaker latents...")
gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(audio_path=[SPEAKER_REFERENCE])

# ================================
# Inference
# ================================

text = read_text_file("to_read.txt")
print(f"Generating speech for:\n{text}")
model.eval()
with torch.no_grad():
    
    out = model.inference(
        text,
        "ru",
        gpt_cond_latent,
        speaker_embedding,
        temperature=0.31,          
        length_penalty=1.0,      
        repetition_penalty=11.0,  
        top_k=40,                 
        top_p=0.81,              
        speed=1.01,                
        enable_text_splitting=False

    )
torchaudio.save(OUTPUT_WAV_PATH, torch.tensor(out["wav"]).unsqueeze(0), 24000)
print(f"Done! Audio saved at: {OUTPUT_WAV_PATH}")
