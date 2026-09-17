"""
Обучение baseline-детектора сорняков (YOLOv8n) на автоматически
размеченном датасете (см. prepare_dataset.py).

Запуск:
    python src/train.py
    python src/train.py --epochs 30 --imgsz 640
"""
import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
DATA_YAML = ROOT / "data" / "data.yaml"
MODELS_DIR = ROOT / "models"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--model", type=str, default="yolov8n.pt")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    if not DATA_YAML.exists():
        raise SystemExit(
            f"{DATA_YAML} не найден. Сначала запусти: python src/prepare_dataset.py"
        )

    MODELS_DIR.mkdir(exist_ok=True)

    model = YOLO(args.model)
    results = model.train(
        data=str(DATA_YAML),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
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
