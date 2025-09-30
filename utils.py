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

