"""
Web-демо детекции сорняков на снимках с дрона.

Запуск:
    streamlit run app.py
"""
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from inference import load_class_names, load_model, run_inference  # noqa: E402
from visualize import color_for_class  # noqa: E402

st.set_page_config(page_title="Weed Detector", layout="wide")

MODEL_PATH = ROOT / "models" / "best.pt"
DATA_YAML = ROOT / "data" / "data.yaml"


@st.cache_resource
def get_model_and_classes():
    class_names = load_class_names(DATA_YAML)
    model = load_model(MODEL_PATH)
    return model, class_names


st.title("🌱 Детектор сорняков на снимках с дрона")
st.caption(
    "Загрузите изображение поля/растения. Для больших снимков с дрона используется "
    "тайловый инференс (по умолчанию), для обычных фото можно переключиться на standard."
)

if not MODEL_PATH.exists():
    st.error(
        f"Модель не найдена: {MODEL_PATH}\n\n"
        "Сначала выполните:\n\n"
        "```\npython src/prepare_dataset.py\npython src/train.py\n```"
    )
    st.stop()

model, class_names = get_model_and_classes()

with st.sidebar:
    st.header("Настройки")
    mode_label = st.radio(
        "Режим инференса",
        options=["Tiled UAV inference", "Standard inference"],
        index=0,
        help="Tiled — изображение режется на перекрывающиеся тайлы, детекция "
             "выполняется на каждом тайле и объединяется. Рекомендуется для "
             "больших снимков с дрона. Standard — один проход по всему кадру.",
    )
    mode = "tiled" if mode_label.startswith("Tiled") else "standard"

    conf = st.slider("Confidence threshold", min_value=0.05, max_value=0.95,
                      value=0.25, step=0.05)

    if mode == "tiled":
        tile_size = st.select_slider("Размер тайла (px)",
                                      options=[320, 416, 512, 640, 768, 1024],
                                      value=640)
        overlap = st.slider("Перекрытие тайлов", min_value=0.0, max_value=0.5,
                             value=0.2, step=0.05)
    else:
        tile_size, overlap = 640, 0.2

    st.markdown("---")
    st.subheader("Классы")
    for i, name in enumerate(class_names):
        color = color_for_class(i)[::-1]  # BGR -> RGB
        st.markdown(
            f"<span style='display:inline-block;width:12px;height:12px;"
            f"background:rgb{color};margin-right:6px;border-radius:2px;'></span>{name}",
            unsafe_allow_html=True,
        )

uploaded = st.file_uploader("Изображение", type=["jpg", "jpeg", "png"])

if uploaded is not None:
    file_bytes = np.frombuffer(uploaded.read(), np.uint8)
    image_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if image_bgr is None:
        st.error("Не удалось декодировать изображение.")
        st.stop()

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Оригинал")
        st.image(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB), use_container_width=True)

    with st.spinner("Выполняется инференс..."):
        annotated, stats, elapsed, n_tiles = run_inference(
            model, class_names, image_bgr,
            mode=mode, conf=conf, tile_size=tile_size, overlap=overlap,
        )

    with col2:
        st.subheader("Результат детекции")
        st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), use_container_width=True)

    st.markdown("---")
    st.subheader("Статистика")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total detections", stats["total"])
    m2.metric("Weed coverage %", f"{stats['coverage_pct']:.1f}%")
    m3.metric("Время обработки", f"{elapsed:.2f} с")
    m4.metric("Тайлов обработано", n_tiles)

    st.subheader("Количество по видам сорняков")
    cols = st.columns(max(1, len(class_names)))
    for i, name in enumerate(class_names):
        cols[i].metric(name, stats["counts"].get(name, 0))
else:
    st.info("Загрузите изображение, чтобы запустить детекцию.")
