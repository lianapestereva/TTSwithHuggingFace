
import json
import os
from TTS.utils.manage import ModelManager


def prepare_config(config_path, model_dir):
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)

    config["dataset_config"]["eval_split_size"] = 0.02
    config["eval_split_size"] = 0.02


    model_file = os.path.join(model_dir, "model_file.pth")
    if not os.path.exists(model_file):
        model_file = os.path.join(model_dir, "model.pth")
    if not os.path.exists(model_file):
        raise FileNotFoundError(f"Не найден .pth файл в {model_dir}")

    config["restore_path"] = model_file

    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
    data_dir = os.path.normpath(data_dir)  
    config["datasets"][0]["path"] = data_dir

    config["output_path"] = os.path.abspath("./finetune/outputs")
    config["output_path"] = os.path.normpath(config["output_path"])

    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    print(f"Путь к данным: {data_dir}")
    print(f"Полный путь к metadata.csv: {os.path.join(data_dir, 'metadata.csv')}")
    return config


if __name__ == "__main__":
    os.environ["TRAINER_TELEMETRY"] = "0"
    os.environ["WANDB_MODE"] = "offline"
    os.environ["HTTP_PROXY"] = ""
    os.environ["HTTPS_PROXY"] = ""
    os.environ["TTS_TELEMETRY"] = "0"
    os.environ["COQUI_TOS_AGREED"] = "0"
    
    os.environ["CURL_CA_BUNDLE"] = ""
    os.environ["REQUESTS_CA_BUNDLE"] = ""
    manager = ModelManager()
    model_dir, _, _ = manager.download_model("tts_models/multilingual/multi-dataset/xtts_v2")
    print(f"Модель: {model_dir}")

    config_file = "./finetune/config.json"
    prepare_config(config_file, model_dir)

    cmd = "python -m TTS.bin.train_tts --config_path ./finetune/config.json --use_cuda false --enable_webui false"
    print(f"\n{cmd}")
    os.system(cmd)
