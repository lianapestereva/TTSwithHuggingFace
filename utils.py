from pydub import AudioSegment
import sys
import os


def read_text_file(filename: str) -> str:

    if not os.path.isfile(filename):
        print(f"Ошибка: файл '{filename}' не найден в текущей папке.")
        sys.exit(1)

    try:
        with open(filename, encoding="utf-8") as f:
            text = f.read().strip()
    except UnicodeDecodeError:
        print(f"Ошибка: файл '{filename}' не в кодировке UTF-8.")
        sys.exit(1)

    if not text:
        print(f"Ошибка: файл '{filename}' пустой или содержит только пробелы/переносы строк.")
        sys.exit(1)

    print("Текст загружен...")
    return text


def convert_to_xtts_reference(input_path: str, output_path: str = "reference.wav") -> None:

    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"Файл не найден: {input_path}")
    
    try:
        sound = AudioSegment.from_file(input_path)
        sound = sound.set_frame_rate(24000).set_channels(1)
        
        sound.export(output_path, format="wav")
        
        print(f"Конвертация завершена: '{input_path}' → '{output_path}'")
        
    except Exception as e:
        print(f"Ошибка при конвертации: {e}")
        raise