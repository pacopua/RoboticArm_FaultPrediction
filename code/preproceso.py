import pandas as pd

# ==========================================
# CONFIGURACIÓN
# ==========================================
NUM_LAGS = 3  # Número de pasos de tiempo anteriores
COLUMNAS_A_ELIMINAR = ['Num', 'Timestamp'] 
# ==========================================

# 1. Cargar el dataset
df = pd.read_csv('../dataset/robot_dataset.csv')

# 2. Eliminar columnas innecesarias (Num y Timestamp)
# Usamos errors='ignore' por si el código se ejecuta sobre un archivo que ya no las tiene
df = df.drop(columns=COLUMNAS_A_ELIMINAR, errors='ignore')

# 3. Limpieza de nulos
df_prep = df.dropna().copy()

# 4. Crear la columna de estado combinada (0-3)
mapping = {'False': 0, 'True': 1, False: 0, True: 1}
stop = df_prep['Robot_ProtectiveStop'].map(mapping)
grip = df_prep['grip_lost'].map(mapping)
df_prep['Status_Combined'] = (stop * 2) + grip

# 5. Identificar sensores para los lags
# Filtramos para no hacer lags de las columnas de estado ni de 'cycle'
sensor_cols = [col for col in df_prep.columns if any(x in col for x in ['Current', 'Temperature', 'Speed', 'Tool_current'])]

# 6. Generación dinámica de Lags
print(f"Generando {NUM_LAGS} lags para {len(sensor_cols)} sensores...")
for col in sensor_cols:
    for i in range(1, NUM_LAGS + 1):
        df_prep[f'{col}_lag_{i}'] = df_prep[col].shift(i)

# 7. Limpieza final
# Eliminamos las filas iniciales con NaN por los lags
df_final = df_prep.dropna().copy()

# Guardar el resultado
#df_final.to_csv('robot_dataset_limpio_ml.csv', index=False)

print("--- PROCESO COMPLETADO ---")
print(f"Columnas eliminadas: {COLUMNAS_A_ELIMINAR}")
print(f"Total columnas finales: {len(df_final.columns)}")
print(f"Ejemplo de columnas: {list(df_final.columns[:10])} ...")



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import GroupShuffleSplit


# Limpiar espacios en nombres de columnas si los hay
df_final.columns = df_final.columns.str.strip()

# ==========================================
# 2. MODELADO (Split correcto)
# ==========================================
target_col = 'Status_Combined'
# Mantenemos 'cycle' temporalmente para hacer el split
features_to_drop = ['Robot_ProtectiveStop', 'grip_lost', 'Status_Combined']
X = df_final.drop(columns=features_to_drop)
y = df_final[target_col]
groups = df_final['cycle'] # Guardamos los grupos

# --- CORRECCIÓN 2: Split por Grupos (Ciclos) ---
# Esto asegura que un ciclo entero esté en train O en test, nunca mezclado
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=2)
train_idx, test_idx = next(gss.split(X, y, groups))

# Ahora sí podemos quitar la columna cycle de las features
X_train = X.iloc[train_idx].drop(columns=['cycle'])
X_test = X.iloc[test_idx].drop(columns=['cycle'])
y_train = y.iloc[train_idx]
y_test = y.iloc[test_idx]

# Escalado
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Entrenamiento
model = LogisticRegression(max_iter=2000, class_weight='balanced', random_state=2)
model.fit(X_train_scaled, y_train)

# Evaluación
y_pred = model.predict(X_test_scaled)
print("\nReporte de Clasificación (Split por Ciclos):")
print(classification_report(y_test, y_pred))

# ==========================================
# 3. INTERPRETACIÓN DE CARACTERÍSTICAS
# ==========================================
# --- CORRECCIÓN 3: Manejo de Multiclase ---
# Vamos a promediar el valor absoluto de los coeficientes de todas las clases
# para ver la "importancia global" de la característica.
avg_importance = np.mean(np.abs(model.coef_), axis=0)

importance_df = pd.DataFrame({
    'Feature': X_train.columns, 
    'Importance': avg_importance
})
top_10 = importance_df.sort_values(by='Importance', ascending=False).head(10)

plt.figure(figsize=(10, 6))
plt.barh(top_10['Feature'], top_10['Importance'], color='coral')
plt.xlabel('Importancia Media Absoluta (todas las clases)')
plt.title('Top 10 Sensores más influyentes (Global)')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

