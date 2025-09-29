from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
import torch
import scipy.io.wavfile
from utils import read_text_file, convert_to_xtts_reference

config = XttsConfig()
config.load_json(".venv/Lib/site-packages/TTS/tts/configs/xtts_config.json")  # or use model.config
model = Xtts.init_from_config(config)
model.load_checkpoint(config, checkpoint_dir="tts_models/multilingual/multi-dataset/xtts_v2", eval=True)
model.cuda() if torch.cuda.is_available() else model.cpu()
 
text = read_text_file(to_read.txt)
speach_path = convert_to_xtts_reference(video.mp4)

tts.tts_to_file(
    text=text,
    file_path="output_high_quality.wav",
    config=config,
    speaker_wav="reference.wav",  # ваш 6+ секундный файл
    language="ru",                # обязательно "ru" для русского
    split_sentences=True,         # разбивает на предложения — улучшает интонацию
    length_scale=1.0,             # 1.0 = нормальная скорость; попробуйте 0.95–1.05
    temperature=0.65,             # ниже = стабильнее и естественнее (0.5–0.75 оптимально)
    top_k=50,                     # фильтрация токенов (40–70)
    top_p=0.85,                   # nucleus sampling (0.8–0.95)
    repetition_penalty=2.0,        # снижает повторы (1.5–3.0)
    max_ref_len=15,
    gpt_cond_len=6,
    sound_norm_refs=True,
)


scipy.io.wavfile.write("output.wav", 24000, outputs["wav"])