# Photonika — нейросетевой синтез фазового ДОЭ

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

Веб-интерфейс <http://localhost:8000>
