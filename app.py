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
SAMPLES_DIR = ROOT / "Поля"
THUMBS_DIR = ROOT / "assets" / "thumbs"
IMG_EXTS = (".jpg", ".jpeg", ".png")

CUSTOM_CSS = """
<style>
.block-container { padding-top: 1.6rem; padding-bottom: 3rem; }

.hero {
    background: linear-gradient(120deg, #0d2f1c 0%, #17603a 55%, #2bab66 100%);
    padding: 28px 34px;
    border-radius: 20px;
    margin-bottom: 22px;
    box-shadow: 0 10px 30px rgba(0,0,0,0.35);
    border: 1px solid rgba(255,255,255,0.08);
}
.hero h1 { margin: 0; font-size: 2rem; color: #ffffff; letter-spacing: -0.01em; }
.hero p { margin: 8px 0 0; color: #d9ffe6; font-size: 0.97rem; max-width: 760px; }

.steps {
    display: flex; gap: 10px; margin-top: 18px; flex-wrap: wrap;
}
.step {
    display: flex; align-items: center; gap: 8px;
    background: rgba(255,255,255,0.1);
    border: 1px solid rgba(255,255,255,0.16);
    border-radius: 999px; padding: 6px 14px 6px 10px;
    color: #eafff2; font-size: 0.82rem;
}
.step b {
    width: 20px; height: 20px; border-radius: 50%;
    background: rgba(255,255,255,0.22);
    display: flex; align-items: center; justify-content: center;
    font-size: 0.72rem;
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

.section-title { font-size: 1.05rem; font-weight: 700; margin: 6px 0 12px; color: inherit; }
.hint { font-size: 0.85rem; color: #8a8a8a; margin: -6px 0 16px; }

.rec-card {
    background: rgba(120,120,120,0.06);
    border: 1px solid rgba(120,120,120,0.16);
    border-left: 4px solid var(--accent, #2bab66);
    border-radius: 12px;
    padding: 14px 18px;
    margin-bottom: 12px;
}
.rec-card .rec-title { font-weight: 700; font-size: 0.98rem; margin-bottom: 2px; }
.rec-card .rec-sub { font-size: 0.8rem; opacity: 0.75; margin-bottom: 8px; }
.rec-card .rec-herb { font-size: 0.85rem; margin-bottom: 6px; }
.rec-card .rec-note { font-size: 0.82rem; opacity: 0.75; }
.herb-tag {
    display: inline-block; padding: 3px 10px; margin: 2px 4px 2px 0;
    border-radius: 999px; background: rgba(43,171,102,0.16);
    border: 1px solid rgba(43,171,102,0.4); font-size: 0.78rem;
}

.disclaimer {
    background: rgba(212,160,23,0.08);
    border: 1px dashed #d4a017;
    border-radius: 12px;
    padding: 10px 16px;
    font-size: 0.8rem;
    margin-top: 6px;
}

.sample-card { text-align: center; }
.sample-card img { border-radius: 10px; border: 1px solid rgba(120,120,120,0.25); }
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

st.markdown(
    """
    <div class="hero">
        <h1>🌾 AgroDrone — поиск сорняков по снимку с дрона</h1>
        <p>Загрузите снимок поля — приложение найдёт сорняки, определит вид и фазу
        роста, посчитает засорённость и подскажет, чем обработать.</p>
        <div class="steps">
            <div class="step"><b>1</b> Дрон фотографирует поле</div>
            <div class="step"><b>2</b> Снимок попадает на этот компьютер</div>
            <div class="step"><b>3</b> Приложение анализирует и даёт рекомендацию</div>
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
    st.markdown("### ⚙️ Настройки")
    mode_label = st.radio(
        "Как искать сорняки на снимке",
        options=["Крупный снимок с дрона (по частям)", "Обычное фото (целиком)"],
        index=0,
        help="Снимок с дрона режется на перекрывающиеся квадраты, каждый "
             "проверяется отдельно, результаты склеиваются обратно — иначе "
             "мелкий сорняк на большом кадре модель просто не разглядит. "
             "«Целиком» — один проход по всему фото, подходит для некрупных снимков.",
    )
    mode = "tiled" if mode_label.startswith("Крупный") else "standard"

    conf = st.slider(
        "Чувствительность обнаружения", min_value=0.05, max_value=0.95,
        value=0.25, step=0.05,
        help="Выше — меньше ложных срабатываний, но можно пропустить слабо заметные "
             "сорняки. Ниже — находит больше, но и ошибок больше.",
    )

    if mode == "tiled":
        tile_size = st.select_slider("Размер квадрата разбивки, px",
                                      options=[320, 416, 512, 640, 768, 1024],
                                      value=640)
        overlap = st.slider("Перекрытие квадратов", min_value=0.0, max_value=0.5,
                             value=0.2, step=0.05)
    else:
        tile_size, overlap = 640, 0.2

    st.markdown("---")
    st.markdown("### 🌿 Что умеет находить")
    for i, name in enumerate(class_names):
        r, g, b = color_for_class(i)[::-1]  # BGR -> RGB
        st.markdown(
            f"<div class='class-badge'>"
            f"<span class='class-dot' style='background: rgb({r},{g},{b});'></span>{name}"
            f"</div>",
            unsafe_allow_html=True,
        )


def load_bgr_from_path(path: Path):
    from ultralytics.utils.patches import imread
    return imread(str(path))


def load_bgr_from_upload(uploaded_file):
    file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
    return cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)


def render_result(image_bgr, source_label):
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("<div class='section-title'>Снимок</div>", unsafe_allow_html=True)
        st.image(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB), use_container_width=True)

    with st.spinner("Ищу сорняки на снимке..."):
        annotated, stats, elapsed, n_tiles = run_inference(
            model, class_names, image_bgr,
            mode=mode, conf=conf, tile_size=tile_size, overlap=overlap,
        )

    with col2:
        st.markdown("<div class='section-title'>Результат</div>", unsafe_allow_html=True)
        st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), use_container_width=True)

    st.markdown("---")
    st.markdown("<div class='section-title'>Статистика</div>", unsafe_allow_html=True)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Найдено сорняков", stats["total"])
    m2.metric("Засорённость поля", f"{stats['coverage_pct']:.1f}%")
    m3.metric("Время обработки", f"{elapsed:.2f} с")
    m4.metric("Участков проверено", n_tiles)

    present = [name for name in class_names if stats["counts"].get(name, 0) > 0]

    st.markdown("<div class='section-title'>По видам и фазам роста</div>",
                unsafe_allow_html=True)
    if present:
        cols = st.columns(max(1, len(present)))
        for i, name in enumerate(present):
            cols[i].metric(name, stats["counts"][name])
    else:
        st.info("При текущей чувствительности сорняки не обнаружены. "
                "Попробуйте снизить порог слева.")

    if present:
        st.markdown("<div class='section-title'>🧪 Чем обработать</div>",
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
                        <span style="font-weight:400; opacity:0.7;">
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


st.markdown("### Откуда взять снимок")
source = st.radio(
    "Источник снимка",
    ["📸 Примеры с дрона", "⬆️ Загрузить со своего компьютера", "🗂️ Папка на компьютере"],
    horizontal=True,
    label_visibility="collapsed",
)

image_to_process = None
source_label = None

if source == "📸 Примеры с дрона":
    st.markdown(
        "<div class='hint'>Реальные снимки с дрона DJI, сделанные во время полёта — "
        "чтобы посмотреть, как приложение работает, без своих файлов.</div>",
        unsafe_allow_html=True,
    )
    samples = sorted(SAMPLES_DIR.glob("*.JPG")) if SAMPLES_DIR.exists() else []
    if not samples:
        st.warning("Папка с примерами не найдена.")
    else:
        cols = st.columns(len(samples))
        for i, sample_path in enumerate(samples):
            with cols[i]:
                thumb_path = THUMBS_DIR / f"{sample_path.stem}.jpg"
                st.markdown("<div class='sample-card'>", unsafe_allow_html=True)
                if thumb_path.exists():
                    st.image(str(thumb_path), use_container_width=True)
                if st.button("Выбрать", key=f"sample_{i}", use_container_width=True):
                    st.session_state["chosen_sample"] = str(sample_path)
                st.markdown("</div>", unsafe_allow_html=True)

        chosen = st.session_state.get("chosen_sample")
        if chosen:
            image_to_process = load_bgr_from_path(Path(chosen))
            source_label = Path(chosen).name

elif source == "⬆️ Загрузить со своего компьютера":
    uploaded = st.file_uploader("Снимок с дрона", type=["jpg", "jpeg", "png"],
                                 label_visibility="collapsed")
    if uploaded is not None:
        image_to_process = load_bgr_from_upload(uploaded)
        source_label = uploaded.name

else:  # Папка на компьютере
    st.markdown(
        "<div class='hint'>Реальный сценарий работы: снимки с SD-карты дрона "
        "скопированы в папку на ноутбуке — укажите путь к ней, приложение покажет "
        "найденные фото.</div>",
        unsafe_allow_html=True,
    )
    folder = st.text_input("Путь к папке", placeholder=r"C:\Users\...\Фото с дрона")
    if folder:
        folder_path = Path(folder)
        if not folder_path.is_dir():
            st.error("Такой папки не найдено.")
        else:
            found = sorted(p for p in folder_path.iterdir()
                            if p.suffix.lower() in IMG_EXTS)
            if not found:
                st.warning("В этой папке нет изображений (.jpg / .png).")
            else:
                names = [p.name for p in found]
                picked = st.selectbox(f"Найдено файлов: {len(found)}", names)
                image_to_process = load_bgr_from_path(folder_path / picked)
                source_label = picked

st.markdown("---")

if image_to_process is not None:
    if image_to_process is None or image_to_process.size == 0:
        st.error("Не удалось прочитать изображение.")
    else:
        render_result(image_to_process, source_label)
else:
    st.info("Выберите пример, загрузите файл или укажите папку, чтобы запустить поиск сорняков.")
