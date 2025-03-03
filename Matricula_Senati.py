import pandas as pd

# creamos el DataFrame inicial
data = {
    'Carrera': ['Ingeniería de Software IA', 'Ciberseguridad', 'Diseño Gráfico'],
    'Matricula': [315, 600, 300]
}

df = pd.DataFrame(data)

# ajustamos la numeración del índice desde 1
df.index = df.index + 1

# imprimir los datos iniciales
print("\n Matrícula inicial (SENATI):\n", df)

# calculamos los nuevos valores
df['Matricula_Ajustado'] = df['Matricula'] * 1.10
df['Descuento'] = df['Matricula_Ajustado'] * 0.05
df['Matricula_Final'] = df['Matricula_Ajustado'] - df['Descuento']

# imprimimos los datos con ajustes
print("\n Datos con Matrícula Ajustada y Descuento:\n", df)

# agregamos la nueva carrera
nueva_carrera = pd.DataFrame({
    'Carrera': ['Mecatrónica Industrial'],
    'Matricula': [450]
})
nueva_carrera.index = [len(df) + 1]  # mantener la numeración en orden
df = pd.concat([df, nueva_carrera])

# recalculamos los valores despues de agregar la nueva carrera
df['Matricula_Ajustado'] = df['Matricula'] * 1.10
df['Descuento'] = df['Matricula_Ajustado'] * 0.05
df['Matricula_Final'] = df['Matricula_Ajustado'] - df['Descuento']

# imprimimos el DataFrame con los datos actualizado
print("\n Datos después de agregar 'Mecatrónica Industrial':\n", df)


# Filtramos carreras con matrícula >= 300 (para mayor demanda)
df_filtrado = df[df['Matricula'] >= 300]

# Ordenamos las carreras, asegurando que Ingeniería de Software con IA sea la primera
df_filtrado['Ranking'] = [1] + list(range(2, len(df_filtrado) + 1))
df_filtrado = df_filtrado.sort_values(by='Ranking')

# Mostramos el ranking
print("\nRanking de carreras con más demanda laboral:\n", df_filtrado)


# ordenamos el DataFrame por la Matrícula Final
df_ordenado = df.sort_values(by='Matricula_Final', ascending=False)
df_ordenado.index = range(1, len(df_ordenado) + 1)  # Ajustamos la numeración para que cuente del 1

print("\n Datos ordenado por matrícula final:\n", df_ordenado)


# guardamos el archivo en un CSV
df.to_csv('Matricula_Senati.csv', index=False)
print("\n Datos guardado en 'Matricula_Senati.csv'")