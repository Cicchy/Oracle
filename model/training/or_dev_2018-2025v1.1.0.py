import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
import pickle

# Cargar el conjunto de datos
data = pd.read_csv('frc-match-history.csv')

# Seleccionar las características relevantes
features = [
    'year', 'event_key', 'comp_level', 'match_number', 'epa_diff', 'epa_ratio',
    'red_1_win_rate', 'red_2_win_rate', 'red_3_win_rate', 'blue_1_win_rate',
    'blue_2_win_rate', 'blue_3_win_rate', 'red_1_epa_trend', 'red_2_epa_trend',
    'red_3_epa_trend', 'blue_1_epa_trend', 'blue_2_epa_trend', 'blue_3_epa_trend',
    'red_auto_points', 'blue_auto_points', 'red_teleop_points', 'blue_teleop_points'
]

# Crear la variable de clasificación
data['red_win'] = (data['score_dif'] > 0).astype(int)

# Seleccionar la variable objetivo
target = 'red_win'

X = data[features]
y = data[target]

# Codificar variables categóricas
X = pd.get_dummies(X, columns=['event_key', 'comp_level'], drop_first=True)

# Manejar valores faltantes
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

# Escalar características
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

# Guardar los nombres de las columnas antes de la transformación
column_names = X.columns

# Reconstruir el DataFrame con los nombres de las columnas
X_transformed = pd.DataFrame(X_scaled, columns=column_names)

# Dividir el conjunto de datos
X_train, X_test, y_train, y_test = train_test_split(X_transformed, y, test_size=0.2, random_state=42)

# Crear y entrenar el modelo XGBClassifier
xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
xgb_model.fit(X_train, y_train)

# Realizar predicciones de probabilidad
y_pred_proba = xgb_model.predict_proba(X_test)[:, 1]

# Realizar predicciones de clase
y_pred = (y_pred_proba > 0.5).astype(int)

# Evaluar el rendimiento
accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)

print(f'Accuracy: {accuracy}')
print(f'ROC AUC: {roc_auc}')

# Guardar el modelo
with open('or_dev_2018-2025v1.0.0.pkl', 'wb') as f:
    pickle.dump(xgb_model, f)

# Función para predecir probabilidades
def predict_win_probability(input_data):
    input_df = pd.DataFrame([input_data], columns=features)
    input_df = pd.get_dummies(input_df, columns=['event_key', 'comp_level'], drop_first=True)
    input_df = input_df.reindex(columns=X_train.columns, fill_value=0)
    input_imputed = imputer.transform(input_df)
    input_scaled = scaler.transform(input_imputed)
    win_probability = xgb_model.predict_proba(input_scaled)[:, 1][0]
    return win_probability

# Ejemplo de uso
input_data = {
    'year': 2018, 'event_key': '2018abca', 'comp_level': 'qm', 'match_number': 6,
    'epa_diff': 100, 'epa_ratio': 1.2, 'red_1_win_rate': 0.8, 'red_2_win_rate': 0.7,
    'red_3_win_rate': 0.9, 'blue_1_win_rate': 0.6, 'blue_2_win_rate': 0.5,
    'blue_3_win_rate': 0.8, 'red_1_epa_trend': 0.1, 'red_2_epa_trend': -0.2,
    'red_3_epa_trend': 0.3, 'blue_1_epa_trend': -0.1, 'blue_2_epa_trend': 0.2,
    'blue_3_epa_trend': -0.3, 'red_auto_points': 20, 'blue_auto_points': 15,
    'red_teleop_points': 100, 'blue_teleop_points': 90
}

win_probability = predict_win_probability(input_data)
print(f'Probability of Red Alliance Winning: {win_probability}')
