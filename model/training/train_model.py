import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import pickle

# Cargar el conjunto de datos
data = pd.read_csv('frc-match-history.csv')

# Seleccionar las características relevantes
features = [
    'year', 'comp_level', 'match_number', 'red_1_epa', 'red_2_epa', 'red_3_epa',
    'blue_1_epa', 'blue_2_epa', 'blue_3_epa', 'red_1_win_rate', 'red_2_win_rate',
    'red_3_win_rate', 'blue_1_win_rate', 'blue_2_win_rate', 'blue_3_win_rate'
]

target = [
    'red_score', 'blue_score', 'score_dif', 'red_auto_points', 'blue_auto_points',
    'red_teleop_points', 'blue_teleop_points', 'red_endgame_points',
    'blue_endgame_points', 'red_foul_points', 'blue_foul_points', 'red_rp',
    'blue_rp'
]

X = data[features]
y = data[target]

# Codificar variables categóricas
X = pd.get_dummies(X, columns=['comp_level'], drop_first=True)

# Guardar los nombres de las columnas antes de la transformación
column_names = X.columns

# Manejar valores faltantes
imputer = SimpleImputer(strategy='mean')
X = imputer.fit_transform(X)

# Escalar características
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Reconstruir el DataFrame con los nombres de las columnas
X = pd.DataFrame(X, columns=column_names)

# Dividir el conjunto de datos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.ensemble import RandomForestRegressor

# Crear y entrenar el modelo
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

from sklearn.metrics import mean_squared_error, mean_absolute_error

# Realizar predicciones
y_pred = model.predict(X_test)

# Evaluar el rendimiento
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'Mean Absolute Error: {mae}')

# Guardar el modelo
with open('prediction_model.pkl', 'wb') as f:
    pickle.dump(model, f)

# Función para realizar predicciones
def predict_scores(input_data):
    input_df = pd.DataFrame([input_data], columns=features)
    input_df = pd.get_dummies(input_df, columns=['comp_level'], drop_first=True)
    input_df = input_df.reindex(columns=X_train.columns, fill_value=0)
    input_imputed = imputer.transform(input_df)
    input_scaled = scaler.transform(input_imputed)
    predictions = model.predict(input_scaled)
    # Convertir las predicciones a un DataFrame con nombres de columna
    predictions_df = pd.DataFrame(predictions, columns=target)
    return predictions_df

# Ejemplo de uso
input_data = {
    'year': 2018, 'comp_level': 'qm', 'match_number': 6, 'red_1_epa': 144.07,
    'red_2_epa': 188.45, 'red_3_epa': 202.31, 'blue_1_epa': 99.49,
    'blue_2_epa': 61.38, 'blue_3_epa': 161.57, 'red_1_win_rate': 0.83,
    'red_2_win_rate': 0.72, 'red_3_win_rate': 1.0, 'blue_1_win_rate': 0.6,
    'blue_2_win_rate': 0.12, 'blue_3_win_rate': 0.9
}

predicted_scores = predict_scores(input_data)
print(f'Predicted Scores:\n {predicted_scores}')