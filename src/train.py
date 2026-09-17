"""
Обучение детектора сорняков (YOLOv8) на автоматически размеченном
датасете (см. prepare_dataset.py).

По умолчанию сам находит GPU, если он есть (CUDA) — на слабом CPU-only
окружении без флагов запустится на CPU с уменьшенными epochs/imgsz,
на машине с видеокартой подхватит GPU и параметры для полноценного
обучения. Явно переопределить можно через --device cpu / --device 0.

Запуск:
    python src/train.py
    python src/train.py --epochs 150 --imgsz 640 --batch 32 --model yolov8s.pt
"""
import argparse
import shutil
from pathlib import Path

import torch
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
DATA_YAML = ROOT / "data" / "data.yaml"
MODELS_DIR = ROOT / "models"


def main():
    has_gpu = torch.cuda.is_available()

    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=150 if has_gpu else 25,
                         help="По умолчанию: 150 на GPU, 25 на CPU (иначе обучение "
                              "растянется на часы).")
    parser.add_argument("--imgsz", type=int, default=640 if has_gpu else 512)
    parser.add_argument("--batch", type=int, default=16 if has_gpu else 8)
    parser.add_argument("--model", type=str, default="yolov8n.pt",
                         help="yolov8n/s/m/l/x.pt — крупнее модель = точнее, но "
                              "медленнее. На GPU можно смело брать yolov8s или yolov8m.")
    parser.add_argument("--device", type=str, default=None,
                         help="Явно: 'cpu' или '0' (первая видеокарта). "
                              "По умолчанию — автоопределение.")
    args = parser.parse_args()

    if not DATA_YAML.exists():
        raise SystemExit(
            f"{DATA_YAML} не найден. Сначала запусти: python src/prepare_dataset.py"
        )

    device = args.device
    if device is None:
        device = 0 if has_gpu else "cpu"

    print(f"[train] CUDA доступна: {has_gpu} | device={device} | "
          f"epochs={args.epochs} imgsz={args.imgsz} batch={args.batch} model={args.model}")
    if not has_gpu:
        print("[train] GPU не найден — если он должен быть, см. README, раздел "
              "«Обучение на мощном ПК» (обычный `pip install torch` на Windows "
              "часто ставит CPU-сборку без CUDA).")

    MODELS_DIR.mkdir(exist_ok=True)

    model = YOLO(args.model)
    results = model.train(
        data=str(DATA_YAML),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        project=str(ROOT / "runs"),
        name="weed_detect",
        exist_ok=True,
        patience=15,
        verbose=True,
    )

    best_weights = Path(results.save_dir) / "weights" / "best.pt"
    if best_weights.exists():
        dest = MODELS_DIR / "best.pt"
        shutil.copy(best_weights, dest)
        print(f"\nЛучшие веса скопированы в: {dest}")
    else:
        print("\nВНИМАНИЕ: best.pt не найден в", best_weights)


if __name__ == "__main__":
    main()
