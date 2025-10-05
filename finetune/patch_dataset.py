import numpy as np
import librosa

_original_load_wav = None

def patched_load_wav(self, filename):
    """Fallback load_wav using librosa if self.ap is None"""
    if self.ap is not None:
        return self.ap.load_wav(filename)
    
    if hasattr(self, 'config') and 'audio' in self.config:
        from TTS.utils.audio import AudioProcessor
        audio_config = self.config['audio']
        self.ap = AudioProcessor(**audio_config)
        return self.ap.load_wav(filename)
    
    wav, _ = librosa.load(filename, sr=22050, mono=True)
    return wav.astype(np.float32)

def apply_patch():
    """Apply the patch to TTS dataset"""
    from TTS.tts.datasets.dataset import TTSDataset  
    global _original_load_wav
    if _original_load_wav is None:
        _original_load_wav = TTSDataset.load_wav
        TTSDataset.load_wav = patched_load_wav