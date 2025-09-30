from TTS.api import TTS
import scipy.io.wavfile
from utils import read_text_file
import torch
from TTS.api import TTS

print("Загрузка модели...")
tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2", gpu=False)
 
text = read_text_file("to_read.txt")

tts.tts_to_file(
 
    text=text,
    file_path="output.wav",
    speaker_wav="voice.wav",
    language="ru",
    split_sentences=True,      
    #length_scale=0.9,           
    
    temperature=0.3,
    top_k=20, 
    top_p=0.5,
    repetition_penalty=4.0,        
    
    speed=0.97,                 
    enable_text_splitting=True,  

)


