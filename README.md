# Photonika — нейросетевой синтез фазового ДОЭ

Кейс хакатона НИЯУ МИФИ «Разработка нейросетевого метода синтеза дифракционных оптических
элементов». Сеть получает **4 изображения 128×128** и выдаёт **один фазовый ДОЭ 128×128**,
из которого по формуле Френеля с одним FFT на расстояниях **24,1 / 25,3 / 26,5 / 27,7 мм**
восстанавливается каждое из изображений (λ = 532 нм, пиксель ПВМС 10 мкм).

Результат в формате сдачи: `doe.png` (0 = −π, 255 = +π), `recon_1..4.png` (модуль поля, 8 бит),
`metrics.json` / `metrics.csv` (PCC и SSIM для каждой из 4 пар).

---

## Быстрый старт

```powershell
python -m venv .venv
# GPU (NVIDIA, драйвер 535+):
.\.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cu128
# или только CPU:
# .\.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements.txt

.\.venv\Scripts\python.exe -m app.cli info            # параметры физики, CUDA, наличие весов
.\.venv\Scripts\python.exe -m pytest -q               # тесты физики, метрик, методов, UI
```

Синтез в формате сдачи (i-е изображение восстанавливается на i-м расстоянии):

```powershell
python -m app.cli synth --inputs 1.png 2.png 3.png 4.png --out results/run1
python -m app.cli synth --inputs ... --out results/run1 --method gs+adam   # без нейросети
```

Метод по умолчанию: `net+gs`, если есть веса `data/weights/doe_multiplane.pt`, иначе `gs+adam`.

> **Веса в репозиторий не попадают** (`.gitignore`: `data/weights/`, `*.pt`, `*.pth`, `*.ckpt`,
> `*.safetensors`, `*.onnx`; в Docker-образ тоже не копируются). Файл весов передаётся внутри
> команды напрямую и кладётся в `data/weights/doe_multiplane.pt`. Не добавляйте его через `git add -f`.
Все методы: `net` (один проход сети), `net+gs` (сеть + K итераций GS, `--refine-steps`),
`net+adam` (сеть + K шагов Adam), `gs` (многоплоскостный Gerchberg–Saxton, `--gs-iters`),
`gs+adam`, `adam`.

Что создаётся в `--out`: `doe.png`, `recon_1..4.png`, `source_1..4.png` (входы после приведения
к 8 бит 128×128), `metrics.json`, `metrics.csv`, `preview.png` (исходные / ДОЭ / восстановления).

Проверка готового результата:

```powershell
python -m app.cli verify --doe results/run1/doe.png --inputs 1.png 2.png 3.png 4.png
python tools/verify_submission.py --doe results/run1/doe.png --inputs 1.png 2.png 3.png 4.png --recon results/run1/recon_1.png results/run1/recon_2.png results/run1/recon_3.png results/run1/recon_4.png
```

`tools/verify_submission.py` — самостоятельный скрипт (numpy + Pillow + scikit-image), не импортирует
код проекта: им можно независимо пересчитать восстановление и метрики по PNG.

---

## Физика

Формула из презентации организаторов:

$$K(x,y)=\frac{e^{jkz}}{j\lambda z}\,e^{\frac{j\pi}{\lambda z}(x^2+y^2)}\;\mathrm{FFT}\!\left[H(u,v)\,e^{\frac{j\pi}{\lambda z}(u^2+v^2)}\right]$$

Внешние множители не влияют на модуль, поэтому `|K_i| = |FFT(exp(jφ) · Q_i)|`, где
`Q_i = exp(jπ(u²+v²)/(λ z_i))`. Координаты `(u, v)` отсчитываются от центра ДОЭ с шагом 10 мкм,
результат центрируется `fftshift`, FFT нормирован как `ortho` (энергия сохраняется).
Реализация: `app/fresnel.py` (numpy — эталон для сдачи, torch — дифференцируемая версия).

* **Критическое расстояние** `N·Δx²/λ = 24,06 мм`. Все расстояния из условия чуть больше него, поэтому
  квадратичная фаза дискретизирована без алиасинга и формула с одним FFT корректна.
* **Шаг пикселя в плоскости восстановления** `λz/(NΔx)` растёт от 10,02 до 11,51 мкм. Изображения
  сравниваются на сетке 128×128 без пересчёта масштаба — как в условии.
* Тесты (`tests/test_fresnel.py`): FFT совпадает с прямым суммированием по формуле (N = 16), torch
  совпадает с numpy, энергия сохраняется, фаза ↔ PNG без потерь сверх шага квантования, линза
  `−π r²/(λ z_i)` фокусируется в центр именно в плоскости `z_i`.

## Методы

* **Многоплоскостный GS** (`app/baselines.py`): поле во всех 4 плоскостях → модуль заменяется на
  целевой при сохранении фазы → каждая плоскость обратно распространяется к ДОЭ → сумма → фаза.
* **Adam по фазе**: прямая оптимизация фазы ДОЭ по той же функции потерь, что и у сети.
* **Нейросеть** (`app/model.py`, `MultiPlaneUNet`): U-Net на 4 уровня (128→64→32→16→8), свёртки
  с циклическим паддингом (FFT периодичен), GroupNorm. Выход — комплексная поправка `(a, b)`,
  фаза получается через `atan2` (без разрыва на ±π) и сразу лежит в [−π, π]. Голова
  инициализирована так, что **необученная сеть выдаёт ровно ∠S = одна итерация GS** — обучение
  улучшает этот старт. Режимы входа (`--input-mode`):
  * `imgphase` (по умолчанию) — U-Net в плоскости изображений (признаки и цели совпадают
    пространственно) предсказывает фазу поля в каждой из 4 плоскостей, затем одно обратное
    распространение и сумма в плоскости ДОЭ — по духу DeepCGH;
  * `backprop` — U-Net в плоскости ДОЭ; вход — изображения + обратное распространение каждой цели
    `B_i = IFFT(A_i e^{jψ_i})·Q_i*` (Re/Im) + cos/sin ∠ΣB_i;
  * `raw` — только 4 изображения (контроль для A/B).
  * `--unroll K` — после U-Net ещё K стадий «шаг GS (4 FFT) + маленькая U-Net-поправка». При
    инициализации это ровно GS-K; время инференса фиксировано (никаких итераций до сходимости).
* **Функция потерь** (`app/losses.py`) для каждой плоскости:
  `MSE(s·|K|, A)/mean(A²) + (1 − PCC) + (1 − SSIM*)`, `s` — оптимальный масштаб (MSE оценивает
  форму, а не яркость), `SSIM*` — смесь локального SSIM и ИСС со слайда 3 по изображениям,
  нормированным на максимум (как у жюри, см. выше). Дифференцируемые SSIM совпадают с skimage
  и с формулой слайда (проверено тестами). Плоскости складываются как
  `(1 − w)·среднее + w·soft-max-взвешенная сумма` (`--worst`, по умолчанию 0,5): худшая плоскость
  получает больший вес и не «умирает» (иначе страдает низкоконтрастная Луна).
* **Доуточнение**: выход сети — стартовая фаза для K итераций GS (`net+gs`, дёшево) или K шагов
  Adam (`net+adam`). Шаг Adam в несколько раз дороже итерации GS (обратный проход через FFT и SSIM),
  поэтому основной запасной режим — `net+gs`.

## Данные

`app/data.py`, пул 20 000 изображений 128×128 (кэшируется в `data/pool/`):
символы из TTF-шрифтов (кириллица, латиница, цифры; 17 шрифтов Windows или DejaVu/Liberation в
Docker) — 25 %, геометрические фигуры — 15 %, фото — 40 %, 1/f-текстуры — 20 % (организаторы:
«изображения могут быть разнообразные, но не совсем простые» — упор на фото). Четвёрки
собираются случайно на лету: в половине случаев из одной категории, иначе смешанные;
аугментации — инверсия и гамма. Фото: 15 встроенных изображений scikit-image (случайные кропы)
и, если есть, папка `data/photos/`:

```powershell
python -m app.data fetch-div2k          # DIV2K valid HR, 100 фото, ≈450 МБ → data/photos/div2k
python -m app.data build-pool --count 20000
python -m app.data preview              # data/pool_preview.png
```

После добавления фото удалите `data/pool/` или смените `--pool-size`, чтобы пул пересобрался.

Тестовые наборы (`data/testsets/`, `python -m app.cli make-testsets`) в обучение не входят:
`photos_classic` (camera, astronaut, coins, moon), `text_binary` (text, page, horse, фигуры),
`mephi_letters` («МИФИ»), `digits` («2026»), `random_*` — случайные четвёрки с test-seed.
Фиксированные val-четвёрки (буквы / фото / смешанные) генерируются по отдельному seed.

## Обучение

```powershell
# фаза 1: A/B-прогоны (4 × 20 мин): backprop / imgphase, без unroll и с unroll 4
.\scripts\ab-test.ps1
.\scripts\ab-test.ps1 -Minutes 15 -Only ab_backprop_b32,ab_imgphase_b32   # часть вариантов
python -m app.compare_runs "data/runs/ab_*"          # таблица + data/runs/compare.png

# фаза 2: ночной прогон выбранного варианта в фоне (окно можно закрыть)
.\scripts\train-local.ps1 -Minutes 480 -InputMode imgphase -Base 32   # вариант, выбранный по A/B
.\scripts\training-status.ps1                        # статус, val PCC/SSIM по категориям
.\scripts\stop-training.ps1                          # мягкая остановка с сохранением
.\scripts\train-local.ps1 -Resume -Minutes 120       # продолжить

# фаза 3: дообучение v2 (меньше lr, больше вес SSIM)
.\scripts\train-local.ps1 -Minutes 240 -InitWeights data\weights\doe_multiplane.pt -Lr 3e-5 -WSsim 2 -RunName training_v2 -WeightsName doe_multiplane_v2.pt
```

Напрямую: `python -m app.training --help`. Сохраняется: `data/weights/doe_multiplane.pt` (лучшие веса
по val PCC + SSIM), `data/training/{last.pt, metrics.jsonl, status.json, config.json,
final_report.json, preview.png, training.log}`. Возобновление проверяет совпадение версии
физики и конфигурации сети. Финальный отчёт сравнивает сеть, сеть+20 и GS-50 на test-наборах
по 8-битным PNG.

**Критерий go/no-go в T+16** (из плана): если PCC сети на «фото» ≥ 0,35 и не хуже GS-50 —
основной режим `net`, иначе `net+gs` (сеть как старт для K итераций GS).

## Бенчмарк «качество / время»

```powershell
python -m app.cli bench --sets data/testsets/photos_classic data/testsets/mephi_letters --device cuda
```

Время — медиана 3 запусков после прогрева, с синхронизацией CUDA. Результат: `results/bench/bench.md`,
`bench.json`, `quality_vs_time.png`.

## Демо

```powershell
streamlit run app/ui.py
```

Загрузка 4 PNG или тестового набора, выбор метода, ДОЭ, 4 восстановления, таблица метрик,
сравнение по времени с GS-50, ползунок z от 23 до 29 мм (изображения фокусируются по очереди),
скачивание `doe.png`/`recon_*.png`/`metrics.json` архивом.

## Docker

```powershell
docker compose up --build                                   # демо с GPU, http://localhost:8501
docker compose --profile cpu up --build doe-demo-cpu        # без GPU, http://localhost:8502
```

Веса монтируются из `data/weights`, результаты — в `results/`.

