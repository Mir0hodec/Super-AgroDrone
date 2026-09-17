"""
Web-демо детекции сорняков на снимках с дрона: вид сорняка, фаза
вегетации, количество, % засорённости и рекомендация по обработке.

Запуск:
    streamlit run app.py
"""
import sys
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from inference import load_class_names, load_model, run_inference  # noqa: E402
from visualize import color_for_class  # noqa: E402
from weed_info import get_recommendation, split_class_name  # noqa: E402

st.set_page_config(page_title="AgroDrone Weed Detector", page_icon="🌾", layout="wide")

MODEL_PATH = ROOT / "models" / "best.pt"
DATA_YAML = ROOT / "data" / "data.yaml"

CUSTOM_CSS = """
<style>
.block-container { padding-top: 1.6rem; padding-bottom: 3rem; }

.hero {
    background: linear-gradient(120deg, #0d2f1c 0%, #17603a 55%, #2bab66 100%);
    padding: 30px 36px;
    border-radius: 20px;
    margin-bottom: 24px;
    box-shadow: 0 10px 30px rgba(0,0,0,0.35);
    border: 1px solid rgba(255,255,255,0.08);
}
.hero h1 { margin: 0; font-size: 2.1rem; color: #ffffff; letter-spacing: -0.01em; }
.hero p { margin: 8px 0 0; color: #d9ffe6; font-size: 0.98rem; max-width: 760px; }
.hero .badge-row { margin-top: 14px; display:flex; gap:10px; flex-wrap:wrap; }
.pill {
    display: inline-block;
    padding: 5px 13px;
    border-radius: 999px;
    background: rgba(255,255,255,0.12);
    color: #eafff2;
    font-size: 0.78rem;
    border: 1px solid rgba(255,255,255,0.18);
}

[data-testid="stMetric"] {
    background: linear-gradient(160deg, #14301f, #0c1f14);
    border: 1px solid #24512f;
    padding: 14px 16px 10px;
    border-radius: 14px;
    box-shadow: 0 4px 16px rgba(0,0,0,0.3);
}
[data-testid="stMetricLabel"] { color: #8fd6a8 !important; font-size: 0.82rem; }
[data-testid="stMetricValue"] { color: #f2fff6 !important; }

.class-badge {
    display: flex; align-items: center; gap: 9px;
    padding: 7px 12px; border-radius: 12px;
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.07);
    margin-bottom: 7px; font-size: 0.86rem;
}
.class-dot { width: 11px; height: 11px; border-radius: 50%; box-shadow: 0 0 6px rgba(0,0,0,0.4); flex-shrink: 0; }

.section-title { font-size: 1.05rem; font-weight: 700; margin: 6px 0 12px; color: #eafff2; }

.rec-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-left: 4px solid var(--accent, #2bab66);
    border-radius: 12px;
    padding: 14px 18px;
    margin-bottom: 12px;
}
.rec-card .rec-title { font-weight: 700; font-size: 0.98rem; color: #f2fff6; margin-bottom: 2px; }
.rec-card .rec-sub { font-size: 0.8rem; color: #9fd9b4; margin-bottom: 8px; }
.rec-card .rec-herb { font-size: 0.85rem; color: #eafff2; margin-bottom: 6px; }
.rec-card .rec-note { font-size: 0.82rem; color: #b8ccbe; }
.herb-tag {
    display: inline-block; padding: 3px 10px; margin: 2px 4px 2px 0;
    border-radius: 999px; background: rgba(43,171,102,0.18);
    border: 1px solid rgba(43,171,102,0.4); font-size: 0.78rem; color: #d3ffe3;
}

.disclaimer {
    background: rgba(255,193,7,0.08);
    border: 1px dashed #d4a017;
    border-radius: 12px;
    padding: 10px 16px;
    font-size: 0.8rem;
    color: #e9d6a0;
    margin-top: 6px;
}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

st.markdown(
    """
    <div class="hero">
        <h1>🌾 AgroDrone Weed Detector</h1>
        <p>Дрон фотографирует поле → снимок передаётся на компьютер → модель определяет
        вид сорняка, фазу вегетации, количество, % засорённости и даёт рекомендацию
        по обработке.</p>
        <div class="badge-row">
            <span class="pill">YOLOv8</span>
            <span class="pill">Tiled UAV inference</span>
            <span class="pill">Вид + фаза вегетации</span>
            <span class="pill">Рекомендации по обработке</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

if not MODEL_PATH.exists():
    st.error(
        f"Модель не найдена: {MODEL_PATH}\n\n"
        "Сначала выполните:\n\n"
        "```\npython src/prepare_dataset.py\npython src/train.py\n```"
    )
    st.stop()


@st.cache_resource
def get_model_and_classes():
    class_names = load_class_names(DATA_YAML)
    model = load_model(MODEL_PATH)
    return model, class_names


model, class_names = get_model_and_classes()

with st.sidebar:
    st.markdown("### ⚙️ Настройки инференса")
    mode_label = st.radio(
        "Режим инференса",
        options=["Tiled UAV inference", "Standard inference"],
        index=0,
        help="Tiled — изображение режется на перекрывающиеся тайлы, детекция "
             "выполняется на каждом тайле и объединяется через NMS. Рекомендуется "
             "для больших снимков с дрона. Standard — один проход по всему кадру.",
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
    st.markdown("### 🌿 Классы (вид / фаза)")
    for i, name in enumerate(class_names):
        r, g, b = color_for_class(i)[::-1]  # BGR -> RGB
        st.markdown(
            f"<div class='class-badge'>"
            f"<span class='class-dot' style='background: rgb({r},{g},{b});'></span>{name}"
            f"</div>",
            unsafe_allow_html=True,
        )

uploaded = st.file_uploader("Снимок с дрона", type=["jpg", "jpeg", "png"])

if uploaded is not None:
    file_bytes = np.frombuffer(uploaded.read(), np.uint8)
    image_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if image_bgr is None:
        st.error("Не удалось декодировать изображение.")
        st.stop()

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("<div class='section-title'>Оригинал</div>", unsafe_allow_html=True)
        st.image(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB), use_container_width=True)

    with st.spinner("Выполняется инференс..."):
        annotated, stats, elapsed, n_tiles = run_inference(
            model, class_names, image_bgr,
            mode=mode, conf=conf, tile_size=tile_size, overlap=overlap,
        )

    with col2:
        st.markdown("<div class='section-title'>Результат детекции</div>", unsafe_allow_html=True)
        st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), use_container_width=True)

    st.markdown("---")
    st.markdown("<div class='section-title'>Статистика</div>", unsafe_allow_html=True)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total detections", stats["total"])
    m2.metric("Weed coverage %", f"{stats['coverage_pct']:.1f}%")
    m3.metric("Время обработки", f"{elapsed:.2f} с")
    m4.metric("Тайлов обработано", n_tiles)

    st.markdown("<div class='section-title'>Количество по видам / фазам</div>",
                unsafe_allow_html=True)
    present = [name for name in class_names if stats["counts"].get(name, 0) > 0]
    if present:
        cols = st.columns(max(1, len(present)))
        for i, name in enumerate(present):
            cols[i].metric(name, stats["counts"][name])
    else:
        st.info("Сорняки не обнаружены при текущем пороге confidence.")

    if present:
        st.markdown("<div class='section-title'>🧪 Рекомендации по обработке</div>",
                    unsafe_allow_html=True)
        for name in present:
            rec = get_recommendation(name)
            species, stage = split_class_name(name)
            class_id = class_names.index(name)
            r, g, b = color_for_class(class_id)[::-1]
            herb_tags = "".join(f"<span class='herb-tag'>{h}</span>" for h in rec["herbicides"]) \
                or "<span class='herb-tag'>нет данных</span>"
            st.markdown(
                f"""
                <div class="rec-card" style="--accent: rgb({r},{g},{b});">
                    <div class="rec-title">{species} — {stats['counts'][name]} шт.
                        <span style="font-weight:400; color:#9fd9b4;">
                        (фаза: {stage or '—'}, {rec['type']})</span>
                    </div>
                    <div class="rec-sub">Рекомендуемые группы гербицидов:</div>
                    <div class="rec-herb">{herb_tags}</div>
                    <div class="rec-note">{rec['note']}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        st.markdown(
            "<div class='disclaimer'>⚠️ Это общая агрономическая справка для демо, "
            "не инструкция по применению конкретного препарата. Перед обработкой "
            "сверяйтесь с этикеткой препарата и действующими регламентами.</div>",
            unsafe_allow_html=True,
        )
else:
    st.info("Загрузите снимок с дрона, чтобы запустить детекцию.")
