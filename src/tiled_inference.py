"""
Tiled inference для больших снимков с дрона (UAV).

Изображение режется на перекрывающиеся тайлы фиксированного размера,
детекция выполняется на каждом тайле отдельно, координаты боксов
пересчитываются в систему координат исходного изображения, после чего
пересекающиеся дубликаты (возникающие на стыках тайлов) убираются NMS.
"""
from dataclasses import dataclass

import numpy as np


@dataclass
class Detection:
    x1: float
    y1: float
    x2: float
    y2: float
    conf: float
    cls_id: int


def make_tiles(width, height, tile_size=640, overlap=0.2):
    """Возвращает список (x0, y0, x1, y1) окон тайлов, покрывающих всё изображение."""
    stride = max(1, int(tile_size * (1 - overlap)))
    xs = list(range(0, max(width - tile_size, 0) + 1, stride))
    ys = list(range(0, max(height - tile_size, 0) + 1, stride))
    if not xs or xs[-1] + tile_size < width:
        xs.append(max(width - tile_size, 0))
    if not ys or ys[-1] + tile_size < height:
        ys.append(max(height - tile_size, 0))

    tiles = []
    for y0 in ys:
        for x0 in xs:
            x1 = min(x0 + tile_size, width)
            y1 = min(y0 + tile_size, height)
            tiles.append((x0, y0, x1, y1))
    return tiles


def nms(detections, iou_threshold=0.5):
    """Class-aware NMS для объединения детекций со стыков тайлов."""
    if not detections:
        return []

    by_class = {}
    for d in detections:
        by_class.setdefault(d.cls_id, []).append(d)

    kept = []
    for cls_id, dets in by_class.items():
        dets = sorted(dets, key=lambda d: d.conf, reverse=True)
        boxes = np.array([[d.x1, d.y1, d.x2, d.y2] for d in dets])
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        suppressed = np.zeros(len(dets), dtype=bool)

        for i in range(len(dets)):
            if suppressed[i]:
                continue
            kept.append(dets[i])
            xx1 = np.maximum(boxes[i, 0], boxes[i + 1:, 0])
            yy1 = np.maximum(boxes[i, 1], boxes[i + 1:, 1])
            xx2 = np.minimum(boxes[i, 2], boxes[i + 1:, 2])
            yy2 = np.minimum(boxes[i, 3], boxes[i + 1:, 3])
            w = np.maximum(0, xx2 - xx1)
            h = np.maximum(0, yy2 - yy1)
            inter = w * h
            union = areas[i] + areas[i + 1:] - inter
            iou = np.where(union > 0, inter / union, 0)
            for j_offset, val in enumerate(iou):
                j = i + 1 + j_offset
                if val > iou_threshold:
                    suppressed[j] = True

    return kept


def tiled_predict(model, image_bgr, conf=0.25, tile_size=640, overlap=0.2, iou=0.5):
    """
    image_bgr: numpy array HxWx3 (BGR, как из cv2.imread).
    Возвращает список Detection в координатах исходного изображения.
    """
    h, w = image_bgr.shape[:2]

    # Небольшие изображения гонять тайлингом смысла нет — один проход.
    if w <= tile_size and h <= tile_size:
        tiles = [(0, 0, w, h)]
    else:
        tiles = make_tiles(w, h, tile_size=tile_size, overlap=overlap)

    all_dets = []
    for (x0, y0, x1, y1) in tiles:
        crop = image_bgr[y0:y1, x0:x1]
        if crop.size == 0:
            continue
        results = model.predict(crop, conf=conf, verbose=False)[0]
        if results.boxes is None:
            continue
        for box in results.boxes:
            bx1, by1, bx2, by2 = box.xyxy[0].tolist()
            all_dets.append(
                Detection(
                    x1=bx1 + x0,
                    y1=by1 + y0,
                    x2=bx2 + x0,
                    y2=by2 + y0,
                    conf=float(box.conf[0]),
                    cls_id=int(box.cls[0]),
                )
            )

    return nms(all_dets, iou_threshold=iou), len(tiles)


def standard_predict(model, image_bgr, conf=0.25):
    """Обычный однопроходный inference на всём изображении (для сравнения / мелких фото)."""
    results = model.predict(image_bgr, conf=conf, verbose=False)[0]
    dets = []
    if results.boxes is not None:
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            dets.append(
                Detection(
                    x1=x1, y1=y1, x2=x2, y2=y2,
                    conf=float(box.conf[0]),
                    cls_id=int(box.cls[0]),
                )
            )
    return dets, 1
