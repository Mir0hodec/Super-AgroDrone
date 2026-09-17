"""Отрисовка детекций и расчёт статистики (per-class counts, coverage %)."""
import cv2
import numpy as np

# Фиксированные BGR-цвета по классам, чтобы виды визуально не путались.
PALETTE = [
    (60, 60, 230),    # красный
    (230, 160, 30),   # синий/голубой
    (40, 200, 120),   # зелёный
    (200, 60, 200),   # пурпурный
    (30, 200, 230),   # жёлтый
]


def color_for_class(cls_id):
    return PALETTE[cls_id % len(PALETTE)]


def draw_detections(image_bgr, detections, class_names):
    img = image_bgr.copy()
    for d in detections:
        color = color_for_class(d.cls_id)
        p1 = (int(d.x1), int(d.y1))
        p2 = (int(d.x2), int(d.y2))
        cv2.rectangle(img, p1, p2, color, 2)
        label = f"{class_names[d.cls_id]} {d.conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        ty = max(p1[1], th + 6)
        cv2.rectangle(img, (p1[0], ty - th - 6), (p1[0] + tw + 4, ty), color, -1)
        cv2.putText(img, label, (p1[0] + 2, ty - 4), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def compute_stats(detections, class_names, image_shape):
    h, w = image_shape[:2]
    counts = {name: 0 for name in class_names}
    for d in detections:
        counts[class_names[d.cls_id]] += 1

    coverage_mask = np.zeros((h, w), dtype=np.uint8)
    for d in detections:
        x1, y1 = max(0, int(d.x1)), max(0, int(d.y1))
        x2, y2 = min(w, int(d.x2)), min(h, int(d.y2))
        if x2 > x1 and y2 > y1:
            coverage_mask[y1:y2, x1:x2] = 1

    coverage_pct = 100.0 * coverage_mask.sum() / (w * h) if w * h > 0 else 0.0

    return {
        "counts": counts,
        "total": len(detections),
        "coverage_pct": coverage_pct,
    }
