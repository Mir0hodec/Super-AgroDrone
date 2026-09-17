"""
Конвертация исходного classification-датасета "Сорняки" (папки вида
<Вид сорняка>/<Фаза роста>/*.jpg) в YOLO detection-формат.

Исходные данные НЕ содержат bounding box разметки — это набор кропов
отдельных растений на почве. Чтобы получить рабочий детектор, для каждого
фото автоматически строится псевдо-bbox: растение сегментируется по
индексу избыточной зелени (ExG = 2G - R - B) с порогом Отсу, берётся
bounding rect наибольшего связного компонента. Если сегментация не
удалась (нет зелёного объекта, шум) — используется bbox на центральные
90% кадра как безопасный fallback.

Классы = вид сорняка (3 шт.), фаза роста в имя класса не включается,
т.к. в ТЗ требуется "количество каждого вида сорняка".
"""
import cv2
import numpy as np
import random
import shutil
import yaml
from pathlib import Path

RAW_DIR = Path(__file__).resolve().parent.parent / "Сорняки"
OUT_DIR = Path(__file__).resolve().parent.parent / "data"
VAL_FRACTION = 0.15
SEED = 42

CLASS_NAMES = sorted([p.name for p in RAW_DIR.iterdir() if p.is_dir()])


def imread_unicode(path):
    """cv2.imread не умеет читать пути с кириллицей на Windows — обходим через imdecode."""
    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def segment_plant_bbox(img_bgr):
    """Возвращает (x1, y1, x2, y2) в пикселях для наибольшего зелёного blob-а."""
    h, w = img_bgr.shape[:2]
    img_f = img_bgr.astype(np.float32)
    b, g, r = img_f[:, :, 0], img_f[:, :, 1], img_f[:, :, 2]
    exg = 2 * g - r - b
    exg_norm = cv2.normalize(exg, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    exg_blur = cv2.GaussianBlur(exg_norm, (5, 5), 0)
    _, mask = cv2.threshold(exg_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel = np.ones((7, 7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return _fallback_bbox(w, h)

    biggest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(biggest)
    if area < 0.01 * w * h:
        return _fallback_bbox(w, h)

    x, y, bw, bh = cv2.boundingRect(biggest)
    # небольшой запас по краям
    pad_x, pad_y = int(bw * 0.05), int(bh * 0.05)
    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(w, x + bw + pad_x)
    y2 = min(h, y + bh + pad_y)

    # если найденный бокс подозрительно мал (< 5% площади кадра) — fallback
    if (x2 - x1) * (y2 - y1) < 0.05 * w * h:
        return _fallback_bbox(w, h)

    return x1, y1, x2, y2


def _fallback_bbox(w, h):
    x1, y1 = int(w * 0.05), int(h * 0.05)
    x2, y2 = int(w * 0.95), int(h * 0.95)
    return x1, y1, x2, y2


def to_yolo_line(class_id, x1, y1, x2, y2, w, h):
    cx = (x1 + x2) / 2 / w
    cy = (y1 + y2) / 2 / h
    bw = (x2 - x1) / w
    bh = (y2 - y1) / h
    return f"{class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"


def collect_images():
    items = []
    for class_name in CLASS_NAMES:
        class_dir = RAW_DIR / class_name
        for stage_dir in class_dir.iterdir():
            if not stage_dir.is_dir():
                continue
            for img_path in stage_dir.glob("*.jpg"):
                items.append((img_path, class_name))
    return items


def main():
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    for split in ("train", "val"):
        (OUT_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUT_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)

    items = collect_images()
    random.Random(SEED).shuffle(items)

    # стратифицированный сплит по классам
    by_class = {c: [] for c in CLASS_NAMES}
    for path, cls in items:
        by_class[cls].append(path)

    train_items, val_items = [], []
    for cls, paths in by_class.items():
        n_val = max(1, int(len(paths) * VAL_FRACTION))
        val_items += [(p, cls) for p in paths[:n_val]]
        train_items += [(p, cls) for p in paths[n_val:]]

    stats = {"train": {c: 0 for c in CLASS_NAMES}, "val": {c: 0 for c in CLASS_NAMES}}
    failed = 0

    for split, split_items in (("train", train_items), ("val", val_items)):
        for img_path, cls in split_items:
            img = imread_unicode(img_path)
            if img is None:
                failed += 1
                continue
            h, w = img.shape[:2]
            x1, y1, x2, y2 = segment_plant_bbox(img)
            class_id = CLASS_NAMES.index(cls)
            line = to_yolo_line(class_id, x1, y1, x2, y2, w, h)

            stem = f"{cls}__{img_path.stem}".replace(" ", "_")
            out_img = OUT_DIR / "images" / split / f"{stem}.jpg"
            out_lbl = OUT_DIR / "labels" / split / f"{stem}.txt"
            shutil.copy(img_path, out_img)
            out_lbl.write_text(line + "\n", encoding="utf-8")
            stats[split][cls] += 1

    data_yaml = {
        "path": str(OUT_DIR.resolve()),
        "train": "images/train",
        "val": "images/val",
        "names": {i: name for i, name in enumerate(CLASS_NAMES)},
    }
    with open(OUT_DIR / "data.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(data_yaml, f, allow_unicode=True, sort_keys=False)

    print("Классы:", CLASS_NAMES)
    print("Train:", stats["train"], "=", sum(stats["train"].values()))
    print("Val:", stats["val"], "=", sum(stats["val"].values()))
    if failed:
        print("Не удалось прочитать изображений:", failed)
    print("data.yaml ->", OUT_DIR / "data.yaml")


if __name__ == "__main__":
    main()
