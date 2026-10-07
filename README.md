# Heart Failure Prediction

Предсказание риска смерти от сердечной недостаточности с помощью классического ML.

## О проекте

Модель предсказывает, умрёт ли пациент от сердечной недостаточности (`DEATH_EVENT`: 0 или 1).
Используется датасет **Heart Failure Clinical Records** (299 записей, 12 признаков).

## Технологии

- Python 3.x
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Imbalanced-learn (SMOTE)

## Как запустить

1. Установи зависимости:
```bash
pip install -r requirements.txt
Скачай датасет:
```


# Скачай с Kaggle: Heart Failure Prediction
# Положи файл 'heart_failure.csv' в папку проекта


Запусти:

bash
python main.py
Данные
Источник: Kaggle — Heart Failure Prediction (COMP3220 UWI)

Признаки:

age — возраст

anaemia — анемия (0/1)

creatinine_phosphokinase — уровень фермента CPK

diabetes — диабет (0/1)

ejection_fraction — фракция выброса

high_blood_pressure — высокое давление (0/1)

platelets — тромбоциты

serum_creatinine — креатинин

serum_sodium — натрий

sex — пол (0/1)

smoking — курение (0/1)

time — время наблюдения (исключён)

DEATH_EVENT — целевая переменная (0/1)

Feature Engineering
Созданы 4 новых признака:

Age_fraction = age × ejection_fraction

Serum_cretinine_ser = serum_creatinine / serum_sodium

Level_of_creat_platles = platelets / creatinine_phosphokinase

is_old = (age > 60)

## Результаты моделей

| Модель | Accuracy | Precision | Recall | F1 | CV |
|--------|:--------:|:---------:|:------:|:--:|:--:|
| RandomForest | 68% | 65% | **52%** | **58%** | 78.3% |
| XGBoost | 65% | 61% | 44% | 51% | **83.6%** |
| **LogisticRegression** | **70%** | **71%** | 48% | 57% | 78.0% |

**Лучшая модель:** RandomForest (по Recall и F1 — важно для медицины).




## Автор
Abdulla - 
