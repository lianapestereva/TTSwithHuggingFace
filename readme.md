# Генерация русской речи с клонированием голоса (Coqui XTTS-v2)

Этот проект позволяет клонировать любой голос из короткого аудио/видео (от 6 секунд) и генерировать естественную речь на русском языке.
Используется модель [coqui/XTTS-v2](https://huggingface.co/coqui/XTTS-v2).


## Требования
Python 3.9–3.11 


## Использование
1) Установка
   ```bash
python -m -3.11 venv .venv
source venv/bin/activate  # Linux/macOS или
venv\Scripts\activate  # Windows
pip install -r requirements.txt
   ```     
2) Добавить в директорию проекта файл .wav с 7-15 секнудным отрезком чистой и качественной записи голоса и .txt файл с текстом, который нужно озвучить. Сгенерированный файл будет в формате .wav в той же директории.


https://github.com/ishine/open_tts
