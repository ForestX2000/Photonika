# Photonika — нейросетевой синтез фазового ДОЭ

## Сборка Docker

Все команды выполняются из корня репозитория. Требуется работающий Docker с Linux-контейнерами;
библиотеки Python, Node.js и системные зависимости устанавливаются внутри образа автоматически.
При первой сборке нужен доступ к сети. Устанавливать Python, PyTorch или npm на хосте не требуется.

Классический вариант сборки:

```powershell
docker compose up -d -build
```

Веб-интерфейс будет доступен на <http://localhost:8000>
Или [уже развёрнутое решение](https://photonika.forestinc.ru)

Веса для папки data/weights и презентацию можно найти на [гугл диске](https://drive.google.com/drive/folders/1--rE2HqjoPtzeWC8t4qfpRXWkelBIaSt?usp=sharing)
