# MLOps Lab 1: Twitter Sentiment Analysis with MLflow

## Опис проєкту

Лабораторна робота №1 з курсу MLOps. Проєкт реалізує повний цикл навчання моделі машинного навчання для аналізу тональності Twitter-повідомлень (виявлення мови ненависті) з використанням **MLflow** для відстеження експериментів.

## Структура проєкту

```
mlops/
├── .gitignore           # Файл виключень Git
├── requirements.txt     # Список залежностей
├── README.md            # Опис проєкту
├── run_experiments.sh   # Скрипт для запуску серії експериментів
├── data/
│   └── raw/             # Сирі дані (не в Git)
│       └── twitter.csv
├── notebooks/
│   └── 01_eda.ipynb     # Ноутбук з EDA аналізом
├── src/
│   └── train.py         # Скрипт навчання з MLflow
├── mlruns/              # Логи MLflow (не в Git)
└── models/              # Збережені моделі (не в Git)
```

## Встановлення

### 1. Клонування репозиторію
```bash
git clone <url>
cd mlops
```

### 2. Створення віртуального середовища
```bash
python3 -m venv venv
source venv/bin/activate  # Linux/macOS
# .\\venv\\Scripts\\activate  # Windows
```

### 3. Встановлення залежностей
```bash
pip install -r requirements.txt
```

### 4. Завантаження даних
Завантажте датасет [Twitter Sentiment Analysis](https://www.kaggle.com/datasets/arkhoshghalb/twitter-sentiment-analysis-hatred-speech) та розмістіть CSV файл у `data/raw/twitter.csv`.

## Використання

### Запуск EDA ноутбука
```bash
jupyter notebook notebooks/01_eda.ipynb
```

### Запуск навчання моделі
```bash
# Базовий запуск (Logistic Regression)
python src/train.py

# З кастомними параметрами
python src/train.py --model_type logistic_regression --C 0.5 --max_features 10000

# Random Forest
python src/train.py --model_type random_forest --n_estimators 200

# SVM
python src/train.py --model_type svm --C 2.0
```

### CLI Аргументи
| Аргумент | Опис | За замовчуванням |
|---|---|---|
| `--model_type` | Тип моделі: `logistic_regression`, `random_forest`, `svm` | `logistic_regression` |
| `--max_features` | Максимальна кількість ознак TF-IDF | `5000` |
| `--ngram_max` | Максимальний розмір n-грам | `2` |
| `--C` | Параметр регуляризації (LR/SVM) | `1.0` |
| `--n_estimators` | Кількість дерев (Random Forest) | `100` |
| `--test_size` | Частка тестової вибірки | `0.2` |

### Запуск серії експериментів
```bash
chmod +x run_experiments.sh
./run_experiments.sh
```

### Перегляд результатів у MLflow UI
```bash
mlflow ui
# Відкрийте http://127.0.0.1:5000
```

## Технології
- **Python 3.x**
- **scikit-learn** — моделі ML та TF-IDF
- **MLflow** — відстеження експериментів
- **pandas / numpy** — обробка даних
- **matplotlib / seaborn** — візуалізація
- **NLTK** — обробка тексту
