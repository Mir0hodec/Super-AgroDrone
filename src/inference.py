"""Загрузка модели и единая точка входа для standard/tiled инференса."""
import time
from pathlib import Path

import yaml
from ultralytics import YOLO
from ultralytics.utils.patches import imread, imwrite

from tiled_inference import tiled_predict, standard_predict
from visualize import draw_detections, compute_stats

ROOT = Path(__file__).resolve().parent.parent


def load_class_names(data_yaml_path=None):
    path = data_yaml_path or (ROOT / "data" / "data.yaml")
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    names = data["names"]
    if isinstance(names, dict):
        return [names[i] for i in sorted(names)]
    return list(names)


def load_model(weights_path=None):
    weights_path = weights_path or (ROOT / "models" / "best.pt")
    return YOLO(str(weights_path))


def run_inference(model, class_names, image_bgr, mode="tiled", conf=0.25,
                   tile_size=640, overlap=0.2, iou=0.5):
    """
    mode: "tiled" (по умолчанию, для больших UAV-снимков) или "standard".
    Возвращает (annotated_image_bgr, stats_dict, elapsed_seconds, n_tiles).
    """
    t0 = time.perf_counter()

    if mode == "tiled":
        dets, n_tiles = tiled_predict(
            model, image_bgr, conf=conf, tile_size=tile_size, overlap=overlap, iou=iou
        )
    else:
        dets, n_tiles = standard_predict(model, image_bgr, conf=conf)

    elapsed = time.perf_counter() - t0

    annotated = draw_detections(image_bgr, dets, class_names)
    stats = compute_stats(dets, class_names, image_bgr.shape)

    return annotated, stats, elapsed, n_tiles


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Использование: python src/inference.py <путь_к_изображению> [conf] [mode]")
        sys.exit(1)

    img_path = sys.argv[1]
    conf = float(sys.argv[2]) if len(sys.argv) > 2 else 0.25
    mode = sys.argv[3] if len(sys.argv) > 3 else "tiled"

    class_names = load_class_names()
    model = load_model()
    image = imread(img_path)
    if image is None:
        raise SystemExit(f"Не удалось прочитать {img_path}")

    annotated, stats, elapsed, n_tiles = run_inference(
        model, class_names, image, mode=mode, conf=conf
    )

    out_path = str(Path(img_path).with_name(Path(img_path).stem + "_pred.jpg"))
    imwrite(out_path, annotated)

    print("Режим:", mode, "| тайлов:", n_tiles, "| время:", f"{elapsed:.3f}s")
    print("Total detections:", stats["total"])
    print("По классам:", stats["counts"])
    print("Weed coverage %:", f"{stats['coverage_pct']:.2f}")
    print("Сохранено:", out_path)
