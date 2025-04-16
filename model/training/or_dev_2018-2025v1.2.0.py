import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pickle

# Cargar el conjunto de datos
data = pd.read_csv('datasets/frc-match-history.csv')

# Seleccionar las características relevantes
features = [
    'year', 'event_key', 'comp_level', 'match_number',
    'red_1_epa', 'red_1_epa_auto', 'red_1_epa_teleop', 'red_1_epa_endgame',
    'red_2_epa', 'red_2_epa_auto', 'red_2_epa_teleop', 'red_2_epa_endgame',
    'red_3_epa', 'red_3_epa_auto', 'red_3_epa_teleop', 'red_3_epa_endgame',
    'blue_1_epa', 'blue_1_epa_auto', 'blue_1_epa_teleop', 'blue_1_epa_endgame',
    'blue_2_epa', 'blue_2_epa_auto', 'blue_2_epa_teleop', 'blue_2_epa_endgame',
    'blue_3_epa', 'blue_3_epa_auto', 'blue_3_epa_teleop', 'blue_3_epa_endgame',
    'red_1_win_rate', 'red_2_win_rate', 'red_3_win_rate', 'blue_1_win_rate',
    'blue_2_win_rate', 'blue_3_win_rate', 'red_1_epa_trend', 'red_2_epa_trend',
    'red_3_epa_trend', 'blue_1_epa_trend', 'blue_2_epa_trend', 'blue_3_epa_trend',
    'red_auto_points', 'blue_auto_points', 'red_teleop_points', 'blue_teleop_points'
]

# Seleccionar las variables objetivo (descomposición de puntuaciones)
target = [
    'red_score', 'blue_score', 'score_dif', 'red_auto_points', 'blue_auto_points',
    'red_teleop_points', 'blue_teleop_points', 'red_endgame_points',
    'blue_endgame_points', 'red_foul_points', 'blue_foul_points', 'red_rp',
    'blue_rp'
]

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

# Crear y entrenar el modelo XGBoost
xgb_model = XGBRegressor(n_estimators=100, random_state=42)
xgb_model.fit(X_train, y_train)

# Realizar predicciones
y_pred = xgb_model.predict(X_test)

# Evaluar el rendimiento
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'Mean Absolute Error: {mae}')

# Guardar el modelo
with open('or_dev_2018-2025v1.2.0.pkl', 'wb') as f:
    pickle.dump(xgb_model, f)

# Función para realizar predicciones
def predict_scores(input_data):
    input_df = pd.DataFrame([input_data], columns=features)
    input_df = pd.get_dummies(input_df, columns=['event_key', 'comp_level'], drop_first=True)
    input_df = input_df.reindex(columns=X_train.columns, fill_value=0)
    input_imputed = imputer.transform(input_df)
    input_scaled = scaler.transform(input_imputed)
    predictions = xgb_model.predict(input_scaled)
    predictions_df = pd.DataFrame(predictions, columns=target)
    return predictions_df

# Ejemplo de uso
input_data = {
    'year': 2018, 'event_key': '2018abca', 'comp_level': 'qm', 'match_number': 6,
    'red_1_epa': 1500, 'red_1_epa_auto': 200, 'red_1_epa_teleop': 1000, 'red_1_epa_endgame': 300,
    'red_2_epa': 1400, 'red_2_epa_auto': 180, 'red_2_epa_teleop': 950, 'red_2_epa_endgame': 270,
    'red_3_epa': 1600, 'red_3_epa_auto': 220, 'red_3_epa_teleop': 1050, 'red_3_epa_endgame': 330,
    'blue_1_epa': 1300, 'blue_1_epa_auto': 160, 'blue_1_epa_teleop': 900, 'blue_1_epa_endgame': 240,
    'blue_2_epa': 1200, 'blue_2_epa_auto': 140, 'blue_2_epa_teleop': 850, 'blue_2_epa_endgame': 210,
    'blue_3_epa': 1500, 'blue_3_epa_auto': 200, 'blue_3_epa_teleop': 1000, 'blue_3_epa_endgame': 300,
    'red_1_win_rate': 0.8, 'red_2_win_rate': 0.7, 'red_3_win_rate': 0.9,
    'blue_1_win_rate': 0.6, 'blue_2_win_rate': 0.5, 'blue_3_win_rate': 0.8,
    'red_1_epa_trend': 0.1, 'red_2_epa_trend': -0.2, 'red_3_epa_trend': 0.3,
    'blue_1_epa_trend': -0.1, 'blue_2_epa_trend': 0.2, 'blue_3_epa_trend': -0.3,
    'red_auto_points': 20, 'blue_auto_points': 15, 'red_teleop_points': 100, 'blue_teleop_points': 90
}

predicted_scores = predict_scores(input_data)
print(predicted_scores)