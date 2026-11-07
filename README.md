# Photonika — нейросетевой синтез фазового ДОЭ

Решение кейса хакатона НИЯУ МИФИ «Разработка нейросетевого метода синтеза дифракционных оптических элементов».
Модель **совместно обрабатывает 4 изображения 128×128** и формирует **один фазовый ДОЭ 128×128**.
Из него рассчитываются четыре восстановления по формуле дифракции Френеля.

Основной файл для проверки — `doe.png`: **PNG, 128×128 пикселей, 8 бит, Grayscale**
(один канал, режим `L`, без альфа-канала). Значения 0…255 кодируют фазу от −π до +π.

## Сдача решения

По регламенту организаторов:

- **До 09:30** — загрузить и отправить все материалы через [форму сдачи](https://forms.yandex.ru/u/6aedea5f6f1ca49998bfe6e7).
- **09:30 — Stop Coding:** разработка заканчивается, после этого изменять решение нельзя.
- **13:00** — публикация списка команд, прошедших в финал.
- **До 13:30** — завершить презентацию и загрузить её в GitHub-репозиторий. Презентацию можно готовить после Stop Coding.
- **14:00** — очная защита; онлайн-командам организаторы пришлют ссылку в чат.
- Защита: **5 минут презентации + 5 минут ответов на вопросы**. Слушать выступления могут все участники.

Для сдачи нужен GitHub-репозиторий, содержащий:

1. Исходный код нейросети — `ml/model.py`, `ml/physics_model.py`, `ml/pdr_model.py`; обучение — `ml/training.py`.
2. **Обученные веса итоговой модели**. В примерах ниже используется `data/weights/doe_multiplane.pt`.
3. Dockerfile — [`docker/Dockerfile`](docker/Dockerfile).
4. Список зависимостей — [`requirements.txt`](requirements.txt). PyTorch устанавливается автоматически при сборке Docker-образа.
5. Этот `README.md` с инструкцией по сборке, запуску и проверке.

**Статус весов:** сейчас `.gitignore` исключает `data/weights/` и файлы моделей, а `.dockerignore`
исключает их из образа. В отслеживаемых Git-файлах весов нет. До сдачи необходимо включить
выбранный обученный checkpoint в репозиторий (например, через Git LFS), согласовав исключения
в `.gitignore`, и проверить его доступность после клонирования. При использовании Git LFS
должен быть получен сам файл весов, а не текстовый указатель. Локальные экспериментальные
checkpoint не означают, что финальные веса уже опубликованы.

Docker-команды ниже монтируют веса из репозитория в контейнер. **Пока финальные веса не включены
в сдаваемый репозиторий, требование к комплектности решения не выполнено.** Если итоговый файл
называется иначе, во всех командах замените значение `--weights`. Для нейросетевой проверки
метод задаётся явно: без весов команда завершится ошибкой, а не переключится на итерационный метод.

## Сборка Docker

Все команды выполняются из корня репозитория. Требуется работающий Docker с Linux-контейнерами;
библиотеки Python, Node.js и системные зависимости устанавливаются внутри образа автоматически.
При первой сборке нужен доступ к сети. Устанавливать Python, PyTorch или npm на хосте не требуется.

Рекомендуемый переносимый вариант — **CPU**, без настройки GPU:

```powershell
docker build -f docker/Dockerfile --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cpu -t photonika:cpu .
```

Команды с `${PWD}` ниже подходят для PowerShell и Bash, если выполнять их из корня проекта.

## Запуск на одном тестовом комплекте

Поместите четыре тестовых изображения в `submission_inputs/set01/` под именами
`1.png`, `2.png`, `3.png`, `4.png`. Каждый вход должен иметь размер 128×128.
Порядок `--inputs` определяет соответствие расстояниям восстановления:

- `1.png` → 24,1 мм;
- `2.png` → 25,3 мм;
- `3.png` → 26,5 мм;
- `4.png` → 27,7 мм.

Длина волны — 532 нм, шаг пикселя ДОЭ — 10 мкм. Параметры заданы в `ml/config.py`.
Все четыре входа передаются **одним вызовом**, который создаёт один общий ДОЭ.

```powershell
docker run --rm -v "${PWD}/submission_inputs:/inputs:ro" -v "${PWD}/data/weights:/app/data/weights:ro" -v "${PWD}/results:/app/results" photonika:cpu python -m ml.cli synth --inputs /inputs/set01/1.png /inputs/set01/2.png /inputs/set01/3.png /inputs/set01/4.png --weights /app/data/weights/doe_multiplane.pt --method net+gs --refine-steps 30 --device cpu --out /app/results/set01
```

Результат появится на хосте в `results/set01/`:

- **`doe.png`** — фазовый ДОЭ в требуемом формате; именно его используют для восстановления.
- `recon_1.png` … `recon_4.png` — восстановления из сохранённого 8-битного ДОЭ.
- `source_1.png` … `source_4.png` — входы после загрузки и приведения к Grayscale.
- `metrics.json`, `metrics.csv` — метрики по каждой плоскости и средние по комплекту.
- `preview.png` — визуальное сравнение входов, ДОЭ и восстановлений.

`net+gs` выполняет проход сети и 30 итераций многоплоскостного GS. Для измерения одного прохода
нейросети используйте `--method net`. Выбранный метод и параметры сохраняются в `metrics.json`.
Изображения другого размера загрузчик обрезает и масштабирует с предупреждением; для официальной
проверки следует передавать исходные 128×128 без изменения размеров.

Проверка формата выходного файла внутри контейнера:

```powershell
docker run --rm -v "${PWD}/results:/app/results:ro" photonika:cpu python -c "from PIL import Image; im = Image.open('/app/results/set01/doe.png'); assert im.format == 'PNG' and im.size == (128, 128) and im.mode == 'L'; print(im.format, im.size, im.mode)"
```

Независимый пересчёт дифракции и метрик по готовому ДОЭ:

```powershell
docker run --rm -v "${PWD}/submission_inputs:/inputs:ro" -v "${PWD}/results:/app/results:ro" photonika:cpu python tools/verify_submission.py --doe /app/results/set01/doe.png --inputs /inputs/set01/1.png /inputs/set01/2.png /inputs/set01/3.png /inputs/set01/4.png
```

Скрипт использует NumPy, Pillow и scikit-image и не импортирует код модели.

## Тестирование и оценка

Организаторы проверяют решение **не менее чем на 5 комплектах**, каждый из которых содержит
4 изображения 128×128 с различными типами объектов. Для каждого комплекта синтезируется
отдельный общий ДОЭ, затем выполняется расчёт дифракции Френеля на четырёх заданных расстояниях.

- **SSIM** — основная метрика структурного сходства.
- **Pearson (PCC)** — дополнительная метрика корреляции.
- Обе метрики вычисляются отдельно для каждой из четырёх пар «вход — восстановление»,
  затем усредняются по четырём плоскостям.
- Итоговые SSIM и Pearson — средние соответствующих оценок по всем тестовым комплектам.
- Дополнительно фиксируется время работы модели, поскольку задача включает ускорение синтеза ДОЭ.

Локальные отчёты содержат `planes` (оценки каждой плоскости), `mean.ssim`, `mean.pcc` и
`synthesis_time_ms`. Последнее — время синтеза, включая доуточнение выбранного метода;
загрузка весов, чтение входов, восстановление для отчёта и сохранение файлов в него не входят.
На CUDA замер выполняется с синхронизацией. Для сравнения скорости следует фиксировать устройство,
метод, число итераций и отдельно учитывать полный запуск контейнера.

Локальный SSIM реализован через `skimage.metrics.structural_similarity`: окно 7×7,
`data_range=1`, каждое изображение нормируется на свой максимум. Дополнительный `ssim_global`
оставлен для диагностики. Окончательные оценки определяются реализацией проверки организаторов;
сам регламент не уточняет параметры SSIM и нормализации.

Для проверки пяти внешних комплектов создайте папки `submission_inputs/set01` … `set05`,
в каждой — `1.png` … `4.png`, и выполните в **PowerShell**:

```powershell
foreach ($set in @('set01', 'set02', 'set03', 'set04', 'set05')) {
    docker run --rm -v "${PWD}/submission_inputs:/inputs:ro" -v "${PWD}/data/weights:/app/data/weights:ro" -v "${PWD}/results:/app/results" photonika:cpu python -m ml.cli synth --inputs "/inputs/$set/1.png" "/inputs/$set/2.png" "/inputs/$set/3.png" "/inputs/$set/4.png" --weights /app/data/weights/doe_multiplane.pt --method net+gs --refine-steps 30 --device cpu --out "/app/results/$set"
    if ($LASTEXITCODE -ne 0) { throw "Ошибка обработки комплекта $set" }
}
```

Каждый комплект получит собственный `results/setXX/doe.png` и отчёт. Для итоговых значений
усредните `mean.ssim` и `mean.pcc` из пяти отчётов `metrics.json` по отдельности.

Для предварительной проверки сборка образа автоматически создаёт демонстрационные наборы
в `/app/data/testsets/`: `photos_classic`, `text_binary`, `mephi_letters`, `digits`,
`random_photos_0` и другие. Они служат локальными примерами и не заменяют тесты организаторов.
Быстрая проверка контейнера без весов:

```powershell
docker run --rm photonika:cpu python -m ml.cli synth --inputs data/testsets/photos_classic/1.png data/testsets/photos_classic/2.png data/testsets/photos_classic/3.png data/testsets/photos_classic/4.png --method gs --gs-iters 2 --device cpu --out /tmp/photonika-smoke --no-preview
```

Этот короткий итерационный запуск проверяет работоспособность конвейера, но не качество нейросети.

## GPU и веб-демо

Для GPU нужны совместимый драйвер NVIDIA и доступ Docker к GPU (NVIDIA Container Toolkit).
CUDA-библиотеки и PyTorch устанавливаются внутри образа:

```powershell
docker build -f docker/Dockerfile -t photonika:cuda .
```

В команде синтеза добавьте `--gpus all` после `docker run`, замените образ на `photonika:cuda`,
а `--device cpu` — на `--device cuda`. Для прогрева перед замером можно добавить `--warmup`.

Веб-интерфейс (FastAPI + React), CPU, <http://localhost:8000>:

```powershell
docker compose --profile cpu up --build web-cpu
```

Веб-интерфейс с GPU, тот же адрес:

```powershell
docker compose up --build web
```

Альтернативное Streamlit-демо, <http://localhost:8501>:

```powershell
docker compose --profile streamlit up --build streamlit
```

Останавливайте один веб-сервис перед запуском другого на порту 8000. Веб-сервисы монтируют
веса из `data/weights/`; для воспроизводимого тестирования используйте CLI-команду выше
с явно заданными весами и методом.

## Устройство решения и обучение

- `ml/model.py` — создание и загрузка моделей; архитектура хранится в checkpoint.
- `ml/physics_model.py` — U-Net с attention, дифференцируемой оптикой и обучаемыми стадиями GS (`physics-unrolled`).
- `ml/pdr_model.py` — комплексная архитектура `pdr`; `ml/legacy_model.py` — исходная `unet`.
- `ml/fresnel.py` — распространение Френеля: NumPy для пересчёта из PNG и PyTorch для обучения.
- `ml/pipeline.py` — совместная обработка четырёх входов, квантование фазы, сохранение и повторное чтение ДОЭ перед оценкой.
- `ml/metrics.py`, `ml/losses.py` — метрики и дифференцируемые функции потерь.
- `ml/data.py` — генерация обучающих данных и тестовых четвёрок.
- `backend/`, `frontend/` — API и веб-интерфейс.

Доступные методы: `net`, `net+gs`, `net+adam`, `net+gs+adam`, `gs`, `gs+adam`, `adam`.
Без явного `--method` CLI выбирает `net+gs`, если найден указанный файл весов, иначе `gs+adam`.
При сдаче нейросетевого решения используйте явный метод `net` или `net+gs`.

Подробности архитектуры и обучения: [physics-unrolled](docs/physics-unrolled.md).
Для RTX A6000: [подготовка пакета, обучение и выбор checkpoint](docs/a6000.md).
Обучение не требуется для запуска с уже предоставленными финальными весами.

Для разработки вне Docker (PowerShell; Python 3.12):

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m ml.cli info
.\.venv\Scripts\python.exe -m pytest -q
```

Пример обучения кандидата (на устройстве с CUDA):

```powershell
.\.venv\Scripts\python.exe -m ml.training --architecture physics-unrolled --base 16 --unroll 4 --stage-base 8 --batch 4 --lr 1e-4 --minutes 120 --output data/runs/physics_b16_u4 --weights data/weights/physics_unrolled.pt
```

Для такого обучения установите CUDA-сборку PyTorch вместо CPU-сборки. Полученный checkpoint
нужно оценить на отдельных тестовых комплектах перед выбором финальных весов.

## Проверка перед отправкой

- [ ] В GitHub доступны исходники, обученные финальные веса, `docker/Dockerfile`, `requirements.txt` и `README.md`.
- [ ] README указывает фактический файл финальных весов; он загружается явно выбранным нейросетевым методом.
- [ ] После чистого клонирования Docker собирается и запускает решение без ручной установки библиотек и настройки Python-окружения.
- [ ] На каждом из не менее пяти комплектов четыре входа обрабатываются совместно и создаётся один `doe.png`.
- [ ] Каждый ДОЭ — PNG 128×128, 8 бит, Grayscale (`L`).
- [ ] Проверены восстановления Френеля, SSIM, Pearson и время работы; отчёты сохранены по комплектам.
- [ ] Все материалы отправлены через форму до 09:30; после Stop Coding код и веса не меняются.
- [ ] Презентация загружена в GitHub до 13:30 и рассчитана на 5 минут выступления.
