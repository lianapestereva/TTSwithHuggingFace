import json
import os
from TTS.utils.manage import ModelManager
from patch_dataset import apply_patch
apply_patch()

def prepare_config(config_path, model_dir):
    original_config_path = os.path.join(model_dir, "config.json")
    with open(original_config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)

    config["run_eval"] = True
    config["eval_split_size"] = 0.03 
    config["eval_split_max_size"] = 0.03
    config["eval_batch_size"] = 1

    config["batch_size"] = 2
    config["epochs"] = 20
    config["lr"] = 0.0001
    config["print_step"] = 10
    config["save_step"] = 500
    config["num_loader_workers"] = 0

    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
    config["datasets"] = [{
        "formatter": "simple",
        "path": data_dir,
        "meta_file_train": "metadata.csv",
        "language": "ru"
    }]

    config["output_path"] = "./finetune/outputs"
    config["restore_path"] = os.path.join(model_dir, "model.pth")

    config["gpt_checkpoint"] = os.path.join(model_dir, "model.pth")
    config["dvae_checkpoint"] = os.path.join(model_dir, "dvae.pth")
    config["tokenizer_file"] = os.path.join(model_dir, "vocab.json")

    if "model_args" not in config:
        config["model_args"] = {}
    config["model_args"]["audio"] = config["audio"]

    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    print(f"Файл модели: {config['restore_path']}")
    print(f"Путь к данным: {data_dir}")
    return config

if __name__ == "__main__":
    os.environ["TRAINER_TELEMETRY"] = "0"
    os.environ["WANDB_MODE"] = "offline"
    os.environ["TTS_TELEMETRY"] = "0"
    
    manager = ModelManager()
    model_dir, _, _ = manager.download_model("tts_models/multilingual/multi-dataset/xtts_v2")
    print(f"Модель: {model_dir}")

    config_file = "./finetune/config.json"
    prepare_config(config_file, model_dir)

    cmd = "python -m TTS.bin.train_tts --config_path ./finetune/config.json --use_cuda false --enable_webui false"
    
    print(f"\n{cmd}")
    os.system(cmd)