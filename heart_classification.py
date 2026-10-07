import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE

#1. ЗАГРУЗКА И ПРОСМОТР ДАННЫХ
df = pd.read_csv('heart_failure.csv')

print(df.head())
print(df.shape)
print(df.describe())

#1.1 EDA
print(df['DEATH_EVENT'].value_counts())
print(df.corr()['DEATH_EVENT'].sort_values(ascending=False))


# 2. Создание новых признаков

df['Age_fraction'] = df['age'] * df['ejection_fraction']

df['Serum_cretinine_ser'] = df['serum_creatinine'] / df["serum_sodium"]

df['Level_of_creat_platles'] = df['platelets'] / df['creatinine_phosphokinase']

df['is_old'] = (df['age'] > 60).astype(int)

print(df['is_old'].head())
print("---")
print(df['DEATH_EVENT'].value_counts())
#2. ВЫБОР ДАННЫХ
x = df.drop('DEATH_EVENT', axis=1)
y = df['DEATH_EVENT']
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

#3. ОПРЕДЕЛЕНИЕ ПРИЗНАКОВ

# Числовые колонки
numeric_cols = ['age', 'creatinine_phosphokinase', 'ejection_fraction',
    'platelets', 'serum_creatinine', 'serum_sodium', 'Age_fraction', 'Serum_cretinine_ser', 'Level_of_creat_platles']

# Бинарные колонки (0/1)
binary_cols = ['anaemia', 'diabetes', 'high_blood_pressure', 'sex', 'smoking', 'is_old']


#4. МАСШТАБИРОВАНИЕ ДАННЫХ
numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

binary_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
])

preprocessor = ColumnTransformer([
    ('num', numeric_transformer, numeric_cols),
    ('bin', binary_transformer, binary_cols)
])

#Исп. Smote
smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

#5. УСТАНОВКА PIPELINE

pipeline1 = Pipeline([
    ('preprocessor', preprocessor),
    ('rf', RandomForestClassifier(
        n_estimators = 180,
        max_depth = 4,
        min_samples_split = 4,
        min_samples_leaf = 4,
        class_weight='balanced'
    ))
])

pipeline2 = Pipeline([
    ('preprocessor', preprocessor),
    ('xg', XGBClassifier(
        n_estimators=80,
        learning_rate=0.35,
        max_depth=5,
        reg_alpha=0.2,
        reg_lambda=6.0,
        subsample=0.65,
        scale_pos_weight=2.1
    ))
])

pipeline3 = Pipeline([
    ('preprocessor', preprocessor),
    ('lr', LogisticRegression(
        C = 0.1,
        class_weight='balanced',
        max_iter=1000,
        penalty='l2',
        solver='lbfgs',

    ))
])

#6. ОБУЧЕНИЕ
pipeline1.fit(X_train, y_train)

pipeline2.fit(X_train, y_train)

pipeline3.fit(X_train, y_train)

#7. ПЕРВИЧНАЯ ОЦЕНКА

accuracy_of_ran = accuracy_score(y_test, np.round(pipeline1.predict(X_test)))
precision_of_ran = precision_score(y_test, np.round(pipeline1.predict(X_test)))
recall_of_ran = recall_score(y_test, np.round(pipeline1.predict(X_test)))
f1_of_ran = f1_score(y_test, np.round(pipeline1.predict(X_test)))

print('==========================================')

print(f'=== РЕЗУЛЬТАТЫ RAN  ===\n'
      f'1) Общая точность модели: {np.round(accuracy_of_ran, 2) * 100}%\n'
      f'2) Точность положительных предсказаний: {np.round(precision_of_ran, 2) * 100}%\n'
      f'3) Сколько смогла правильно предсказать из ПРАВИЛЬНЫХ: {np.round(recall_of_ran, 2) * 100}%\n'
      f'4) Баланс между точсноть и полнотой: {np.round(f1_of_ran, 2) * 100}%\n')




accuracy_of_xG = accuracy_score(y_test, np.round(pipeline2.predict(X_test)))
precision_of_xG = precision_score(y_test, np.round(pipeline2.predict(X_test)))
recall_of_xG = recall_score(y_test, np.round(pipeline2.predict(X_test)))
f1_of_xG = f1_score(y_test, np.round(pipeline2.predict(X_test)))
print('==========================================')

print(f'=== РЕЗУЛЬТАТЫ XGBClassifier ===\n'
      f'1) Общая точность модели: {np.round(accuracy_of_xG, 2) * 100}%\n'
      f'2) Точность положительных предсказаний: {np.round(precision_of_xG, 2) * 100}%\n'
      f'3) Сколько смогла правильно предсказать из ПРАВИЛЬНЫХ: {np.round(recall_of_xG, 2) * 100}%\n'
      f'4) Баланс между точсноть и полнотой: {np.round(f1_of_xG, 2) * 100}%\n')





accuracy_of_lr = accuracy_score(y_test, np.round(pipeline3.predict(X_test)))
precision_of_lr = precision_score(y_test, np.round(pipeline3.predict(X_test)))
recall_of_lr = recall_score(y_test, np.round(pipeline3.predict(X_test)))
f1_of_lr = f1_score(y_test, np.round(pipeline3.predict(X_test)))
print('==========================================')

print(f'=== РЕЗУЛЬТАТЫ LR  ===\n'
      f'1) Общая точность модели: {np.round(accuracy_of_lr, 2) * 100}%\n'
      f'2) Точность положительных предсказаний: {np.round(precision_of_lr, 2) * 100}%\n'
      f'3) Сколько смогла правильно предсказать из ПРАВИЛЬНЫХ: {np.round(recall_of_lr, 2) * 100}%\n'
      f'4) Баланс между точсноть и полнотой: {np.round(f1_of_lr, 2) * 100}%\n')




#8. ПОДБОР ПАРАМЕТРОВ С GridSearchCV

# #ДЛЯ RF
# param_grid_rf = {
#     'rf__n_estimators': [120, 150, 180],
#     'rf__max_depth': [2, 3, 4],
#     'rf__min_samples_split': [4, 6, 8],
#     'rf__min_samples_leaf': [1, 2, 4, 5, 6]
# }
#
# grid_rf = GridSearchCV(pipeline1, param_grid=param_grid_rf, cv=3, n_jobs=-1, scoring='accuracy')
# grid_rf.fit(X_train, y_train)
#
# print('==========================================')
#
# print(f'ЛУЧШИЕ ПАРАМЕТРЫ ДЛЯ RandomForest: {grid_rf.best_params_}\n'
#       f'ЛУЧШАЯ ТОЧНОСТЬ: {np.round(grid_rf.best_score_, 2)}')
#
#
# #ДЛЯ XGB
# param_grid_xgb = {
#     'xg__n_estimators': [50, 80, 120],
#     'xg__learning_rate': [0.15, 0.2, 0.35],
#     'xg__max_depth': [3, 5, 8, 10],
#     'xg__reg_alpha': [0.2, 0.5, 0.75],
#     'xg__reg_lambda': [4.0, 6.0],
#     'xg__subsample': [0.6, 0.8, 0.95]
# }
#
# grid_xg = GridSearchCV(pipeline2, param_grid=param_grid_xgb, cv=3, n_jobs=-1, scoring='accuracy')
# grid_xg.fit(X_train, y_train)
#
# print('==========================================')
#
# print(f'ЛУЧШИЕ ПАРАМЕТРЫ ДЛЯ Xgboost: {grid_xg.best_params_}\n'
#       f'ЛУЧШАЯ ТОЧНОСТЬ: {np.round(grid_xg.best_score_, 2)}')

 #ДЛЯ LR
# param_grid_lr = {
#     'lr__C': [0.01, 0.1, 1.0, 10.0],
#     'lr__penalty': ['l2', 'l1'],
#     'lr__solver': ['liblinear', 'lbfgs'],
#     'lr__class_weight': ['balanced', None],
#     'lr__max_iter': [1000, 1500, 2000]
# }
#
# grid_lr = GridSearchCV(pipeline3, param_grid=param_grid_lr, cv=5, n_jobs=-1, scoring='accuracy')
# grid_lr.fit(X_train, y_train)
# print('==========================================')
#
# print(f'ЛУЧШИЕ ПАРАМЕТРЫ ДЛЯ LR: {grid_lr.best_params_}\n'
#       f'ЛУЧШАЯ ТОЧНОСТЬ: {np.round(grid_lr.best_score_, 2)}')



#9. ОЦЕНКА С cros_val_score
#ДЛЯ RF
scores_ran = cross_val_score(pipeline1, X_train, y_train, cv=5)
print(f'CV: {scores_ran.mean() * 100}% +- {scores_ran.std() * 100}%')

#ДЛЯ XGB
scores_xgb = cross_val_score(pipeline2, X_train, y_train, cv=5)
print(f'CV: {scores_xgb.mean() * 100}% +- {scores_xgb.std() * 100}%')

#ДЛЯ LR
scores_lr = cross_val_score(pipeline3, X_train, y_train, cv=5)
print(f'CV: {scores_lr.mean() * 100}% +- {scores_lr.std() * 100}%')

























