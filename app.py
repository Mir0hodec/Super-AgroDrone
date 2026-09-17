"""
Web-демо детекции сорняков: фото/снимок с дрона + live-видео (веб-камера,
карта видеозахвата, видеофайл — имитация наземного приёма сигнала с DJI O4).

Запуск:
    streamlit run app.py
"""
import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from inference import load_class_names, load_model, run_inference  # noqa: E402
from visualize import color_for_class  # noqa: E402

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
.hero h1 {
    margin: 0;
    font-size: 2.1rem;
    color: #ffffff;
    letter-spacing: -0.01em;
}
.hero p {
    margin: 8px 0 0;
    color: #d9ffe6;
    font-size: 0.98rem;
    max-width: 760px;
}
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
    display: flex;
    align-items: center;
    gap: 9px;
    padding: 7px 12px;
    border-radius: 12px;
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.07);
    margin-bottom: 7px;
    font-size: 0.86rem;
}
.class-dot {
    width: 11px; height: 11px; border-radius: 50%;
    box-shadow: 0 0 6px rgba(0,0,0,0.4);
    flex-shrink: 0;
}

.section-title {
    font-size: 1.05rem;
    font-weight: 700;
    margin: 6px 0 12px;
    color: #eafff2;
}

.arch-box {
    background: rgba(43,171,102,0.08);
    border: 1px dashed #2bab66;
    border-radius: 14px;
    padding: 14px 18px;
    font-size: 0.86rem;
    color: #cfead9;
    margin-bottom: 10px;
}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

st.markdown(
    """
    <div class="hero">
        <h1>🌾 AgroDrone Weed Detector</h1>
        <p>Детекция сорняков на снимках и видео с дрона: тайловый инференс для больших
        UAV-кадров, распознавание по видам, оценка засорённости поля.</p>
        <div class="badge-row">
            <span class="pill">YOLOv8</span>
            <span class="pill">Tiled UAV inference</span>
            <span class="pill">Live video</span>
            <span class="pill">DJI O4 ground-station совместимо</span>
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
    st.markdown("### 🌿 Классы")
    for i, name in enumerate(class_names):
        r, g, b = color_for_class(i)[::-1]  # BGR -> RGB
        st.markdown(
            f"<div class='class-badge'>"
            f"<span class='class-dot' style='background: rgb({r},{g},{b});'></span>{name}"
            f"</div>",
            unsafe_allow_html=True,
        )


def render_stats(stats, elapsed, n_tiles):
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total detections", stats["total"])
    m2.metric("Weed coverage %", f"{stats['coverage_pct']:.1f}%")
    m3.metric("Время обработки", f"{elapsed:.2f} с")
    m4.metric("Тайлов обработано", n_tiles)

    st.markdown("<div class='section-title'>Количество по видам сорняков</div>",
                unsafe_allow_html=True)
    cols = st.columns(max(1, len(class_names)))
    for i, name in enumerate(class_names):
        cols[i].metric(name, stats["counts"].get(name, 0))


tab_photo, tab_video, tab_about = st.tabs(
    ["📷 Фото / снимок", "🎥 Видео / камера (наземная станция)", "ℹ️ Архитектура для дрона"]
)

# ---------------------------------------------------------------- Фото ----
with tab_photo:
    uploaded = st.file_uploader("Изображение", type=["jpg", "jpeg", "png"])

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
        render_stats(stats, elapsed, n_tiles)
    else:
        st.info("Загрузите изображение, чтобы запустить детекцию.")

# --------------------------------------------------------------- Видео ----
with tab_video:
    st.markdown(
        """
        <div class="arch-box">
        🛰️ <b>DJI O4 передаёт только картинку</b> (это видеоканал, не компьютер).
        На реальном полёте наземная станция (ноутбук с картой видеозахвата HDMI/USB,
        подключённой к пульту/очкам, либо поток из DJI SDK) видит этот сигнал как
        обычное видео-устройство или видеофайл — ровно то, что принимает этот режим.
        </div>
        """,
        unsafe_allow_html=True,
    )

    source_type = st.radio("Источник видео", ["Видеофайл", "Камера / карта видеозахвата"],
                            horizontal=True)

    if source_type == "Видеофайл":
        video_file = st.file_uploader("Видео", type=["mp4", "mov", "avi", "mkv"],
                                       key="video_upload")
        video_source = None
        if video_file is not None:
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=Path(video_file.name).suffix)
            tmp.write(video_file.read())
            tmp.close()
            video_source = tmp.name
    else:
        device_index = st.text_input(
            "Индекс видеоустройства",
            value="0",
            help="Номер камеры/карты видеозахвата в системе (обычно 0 — первое "
                 "подключённое устройство). Карта захвата с HDMI-выхода пульта/очков "
                 "DJI определяется системой так же, как веб-камера.",
        )
        video_source = int(device_index) if device_index.strip().isdigit() else device_index

    c1, c2 = st.columns(2)
    frame_skip = c1.number_input("Обрабатывать каждый N-й кадр", min_value=1, max_value=60,
                                  value=5, help="Больше N — быстрее, но реже обновление.")
    max_frames = c2.number_input("Максимум кадров за сеанс", min_value=1, max_value=300,
                                  value=20, help="Ограничение на CPU, чтобы сеанс не висел бесконечно.")

    start = st.button("▶ Запустить сеанс детекции", type="primary")

    if start:
        if video_source is None:
            st.warning("Сначала выберите видеофайл или укажите камеру.")
        else:
            cap = cv2.VideoCapture(video_source)
            if not cap.isOpened():
                st.error(
                    "Не удалось открыть видеоисточник. Если это камера/карта захвата — "
                    "проверьте, что устройство подключено к этой машине и индекс указан "
                    "верно (в облачной/тестовой среде физических камер обычно нет)."
                )
            else:
                frame_slot = st.empty()
                stats_slot = st.empty()
                progress_slot = st.empty()

                processed = 0
                frame_idx = 0
                total_counts = {name: 0 for name in class_names}
                total_detections = 0
                total_elapsed = 0.0

                while processed < max_frames:
                    ok, frame = cap.read()
                    if not ok:
                        break
                    frame_idx += 1
                    if (frame_idx - 1) % frame_skip != 0:
                        continue

                    annotated, stats, elapsed, n_tiles = run_inference(
                        model, class_names, frame,
                        mode=mode, conf=conf, tile_size=tile_size, overlap=overlap,
                    )
                    processed += 1
                    total_detections += stats["total"]
                    total_elapsed += elapsed
                    for name in class_names:
                        total_counts[name] += stats["counts"].get(name, 0)

                    frame_slot.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                                      use_container_width=True,
                                      caption=f"Кадр {frame_idx} | обработано {processed}/{max_frames}")

                    with stats_slot.container():
                        m1, m2, m3 = st.columns(3)
                        m1.metric("Детекций в кадре", stats["total"])
                        m2.metric("Coverage %", f"{stats['coverage_pct']:.1f}%")
                        m3.metric("Время кадра", f"{elapsed:.2f} с")

                    progress_slot.progress(processed / max_frames)

                cap.release()

                if processed == 0:
                    st.warning("Не удалось прочитать ни одного кадра из источника.")
                else:
                    st.success(f"Сеанс завершён: обработано {processed} кадров.")
                    st.markdown("<div class='section-title'>Итог по сеансу</div>",
                                unsafe_allow_html=True)
                    s1, s2, s3 = st.columns(3)
                    s1.metric("Всего детекций", total_detections)
                    s2.metric("Среднее время/кадр", f"{total_elapsed / processed:.2f} с")
                    s3.metric("Обработано кадров", processed)

                    cols = st.columns(max(1, len(class_names)))
                    for i, name in enumerate(class_names):
                        cols[i].metric(name, total_counts[name])

# ------------------------------------------------------------- О дроне ----
with tab_about:
    st.markdown(
        """
### Как это разворачивается на реальный дрон

**Важно:** DJI O4 — это система видеопередачи (4-е поколение OcuSync), у неё нет
собственного вычислительного модуля. Запустить модель детекции "внутри" O4
физически невозможно — там негде исполнять код.

Реальная рабочая схема:

1. **Дрон** снимает поле и передаёт видео через O4 на пульт/очки.
2. **Наземная станция** (ноутбук/мини-ПК) получает этот сигнал — либо через карту
   видеозахвата (HDMI-выход пульта → USB-капture card → выглядит в системе как
   обычная веб-камера), либо программно через DJI SDK/RTMP-поток.
3. Этот **веб-демо** в режиме **«🎥 Видео / камера»** принимает ровно такой источник
   (индекс видеоустройства или видеопоток) и гоняет через него ту же модель
   детекции, что и для фото — покадрово, с тем же tiled-инференсом для больших
   кадров.
4. Результат (боксы, статистика, % засорённости) показывается на земле в реальном
   времени — так же, как в этом интерфейсе.

Это стандартная и единственно реалистичная для хакатон-MVP архитектура: **вычисления
на земле, а не на борту**. Перенос модели непосредственно на дрон потребовал бы
отдельного бортового компьютера (Jetson/аналог) — сознательно вне периметра этого
MVP согласно приоритетам задачи.
        """
    )
