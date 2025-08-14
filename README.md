## ML Ops – Домашнее задание 1

Датасеты предоставлены в рамках соревнования [Teta ML 1 2025](https://www.kaggle.com/competitions/teta-ml-1-2025).

Сервис для автоматического обнаружения мошеннических транзакций в режиме батчевого скоринга.  
Обрабатывает CSV-файлы из указанной директории с использованием предобученной модели **RandomForestClassifier**.  
В дополнение к `sample_submission.csv` сервис сохраняет:
- JSON-файл с топ-5 важнейшими признаками модели,
- PNG-график плотности распределения предсказанных вероятностей.

---

## Архитектура решения
```
├── .gitignore
├── Dockerfile
├── README.md
├── app/
│   └── app.py               # Ядро сервиса с обработчиком файлов
├── src/
│   ├── preprocessing.py     # Пайплайн обработки данных
│   └── scorer.py            # Модуль прогнозирования
├── models/
│   ├── model.pkl            # Сериализованная модель RandomForest
│   ├── encoders.pkl         # Словари для кодирования категориальных признаков
│   ├── feature_names.json   # Список признаков в порядке обучения
│   └── threshold.json       # Оптимальный порог классификации по F1
├── train_data/
│   └── train.csv            # Данные для обучения (скачать из соревнования)
├── input/
│   └── test.csv             # Данные для скоринга
└── output/                  # Результаты работы сервиса
    ├── sample_submission.csv  # Предсказанные классы
    ├── top_features.json      # Топ-5 фич по важности
    └── density_plot.png       # График распределения вероятностей
```


---

## Быстрый старт

### Требования
- **Docker** версии 20.10+
- Не менее 2 ГБ свободного места
- Порты: не требуется проброс, используется файловая система

---

### Подготовка данных
1. Скачайте файл `train.csv` из соревнования  
   [Teta ML 1 2025](https://www.kaggle.com/competitions/teta-ml-1-2025)  
   и поместите его в директорию: ./train_data/train.csv

2. Аналогично скачайте `test.csv` и положите в: ./input/test.csv


---

### Обучение модели (опционально)
Если модель еще не обучена, запустите:
```bash
python -u train_and_export.py --train train_data/train.csv --test input/test.csv --out models
```
Скрипт:

1. выполнит предобработку данных,
2. обучит модель RandomForest с class_weight="balanced",
3. подберет оптимальный порог по метрике F1,
4. сохранит модель и артефакты в ./models.

---

### Сборка Docker - образа
```bash
docker build -t fraud_rf_service .
```
```bash
docker run --rm \
  -v "$(pwd)/input:/app/input" \
  -v "$(pwd)/output:/app/output" \
  -v "$(pwd)/logs:/app/logs" \
  -v "$(pwd)/models:/app/models" \
  fraud_rf_service
```
После запуска в логах появится сообщение: \

__main__ - INFO - File observer started

### Результаты работы

1. Сервис обработает input/test.csv и сохранит в ./output:
2. sample_submission.csv — предсказанные метки классов,
3. top_features.json — топ-5 признаков по важности,
4. density_plot.png — график плотности вероятностей класса 1.

### Пример проверки количества положительных прогнозов:
```bash
grep -c ",1$" output/sample_submission.csv
```

### Пример лога работы
```bash
2025-08-14 21:06:06,449 - preprocessing - INFO - Loading data from /app/input/test.csv
2025-08-14 21:06:07,250 - preprocessing - INFO - Preprocessed shape: (262144, 13)
2025-08-14 21:06:08,926 - scorer - INFO - Prediction complete: 262144 rows, pos_rate=0.0050, thr=0.310
2025-08-14 21:06:09,373 - __main__ - INFO - Saved timestamped: /app/output/predictions_20250814_210608_test.csv, /app/output/top_features_20250814_210608_test.json, /app/output/density_plot_20250814_210608_test.png
2025-08-14 21:06:09,795 - __main__ - INFO - Saved fixed: /app/output/sample_submission.csv, /app/output/top_features.json, /app/output/density_plot.png
2025-08-14 21:06:09,796 - __main__ - INFO - Done for: /app/input/test.csv
```