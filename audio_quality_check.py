from pathlib import Path
import librosa

def deep_check_audio_files():
    base = Path("data/wavs")
    broken_files = []
    
    for wav_path in base.glob("*.wav"):
        try:
            file_size = wav_path.stat().st_size
            if file_size == 0:
                print(f"❌ {wav_path.name} - ПУСТОЙ ФАЙЛ (0 bytes)")
                broken_files.append(wav_path)
                continue
                
            try:
                audio, sr = librosa.load(str(wav_path), sr=22050)
                print(f"✓ {wav_path.name} - OK ({len(audio)} samples)")
            except Exception as e:
                print(f"❌ {wav_path.name} - ОШИБКА: {e}")
                broken_files.append(wav_path)
                
        except Exception as e:
            print(f"❌ {wav_path.name} - КРИТИЧЕСКАЯ ОШИБКА: {e}")
            broken_files.append(wav_path)
    
    return broken_files



if __name__=="__main__":
    print("Проверка аудиофайлов...")
    broken = deep_check_audio_files()
    print(f"\nНайдено проблемных файлов: {len(broken)}")

    
    y, sr = librosa.load("data/wavs/45.wav", sr=None, mono=True)
    print(f"Sample rate: {sr}, Duration: {len(y)/sr:.2f}s, Mono: {y.ndim == 1}")