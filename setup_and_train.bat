@echo off
REM Один клик: ставит зависимости, готовит датасет, запускает обучение.
REM Использование: setup_and_train.bat [доп. флаги для train.py]
REM Пример:        setup_and_train.bat --epochs 200 --model yolov8s.pt

echo === 1/3: pip install -r requirements.txt ===
pip install -r requirements.txt
if errorlevel 1 goto :error

echo.
echo === Проверка GPU ===
python -c "import torch; print('torch', torch.__version__, '| CUDA available:', torch.cuda.is_available())"
echo Если CUDA available: False, а видеокарта есть — смотри README, раздел
echo "Обучение на мощном ПК" (нужно переустановить torch с CUDA-сборкой).
echo.

echo === 2/3: python src\prepare_dataset.py ===
python src\prepare_dataset.py
if errorlevel 1 goto :error

echo.
echo === 3/3: python src\train.py %* ===
python src\train.py %*
if errorlevel 1 goto :error

echo.
echo Готово. Веса: models\best.pt
echo Запуск демо: streamlit run app.py
goto :eof

:error
echo.
echo Что-то упало на предыдущем шаге — смотри вывод выше.
exit /b 1
