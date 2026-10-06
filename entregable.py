from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


SEMILLA = 42
TOTAL_REGISTROS = 600
CARPETA_GRAFICOS = Path("graficos")


def imprimir_titulo(numero, titulo):
    print("\n")
    print("=" * 60)
    print(f"{numero}. {titulo}")
    print("=" * 60)


def calcular_varianza_manual(datos):
    datos = np.array(datos.dropna())
    media = np.mean(datos)
    suma_cuadrados = np.sum((datos - media) ** 2)
    return suma_cuadrados / len(datos)


def calcular_desviacion_manual(datos):
    return np.sqrt(calcular_varianza_manual(datos))


def generar_dataset(n=TOTAL_REGISTROS, semilla=SEMILLA):
    np.random.seed(semilla)

    temperatura = np.clip(np.random.normal(72, 9, n), 45, 105)
    vibracion = np.clip(np.random.normal(4.2, 1.4, n), 0.5, 9)
    presion = np.clip(np.random.normal(6.5, 1.0, n), 3, 10)
    horas_operacion = np.random.randint(200, 10001, n)
    carga = np.clip(np.random.normal(70, 15, n), 25, 100)

    tipo_maquina = np.random.choice(
        ["Torno", "Compresor", "Bomba"],
        size=n,
        p=[0.4, 0.3, 0.3],
    )

    z = (
        -4
        + 0.065 * (temperatura - 70)
        + 0.75 * (vibracion - 4)
        + 0.45 * (presion - 6)
        + 0.00022 * (horas_operacion - 4000)
        + 0.035 * (carga - 65)
    )
    z += np.where(tipo_maquina == "Compresor", 0.25, 0)
    z += np.where(tipo_maquina == "Bomba", -0.05, 0)

    probabilidad_falla = 1 / (1 + np.exp(-z))
    falla = np.random.binomial(1, probabilidad_falla)

    df = pd.DataFrame(
        {
            "Temperatura_C": temperatura,
            "Vibracion_mm_s": vibracion,
            "Presion_bar": presion,
            "Horas_Operacion": horas_operacion,
            "Carga_pct": carga,
            "Tipo_Maquina": tipo_maquina,
            "Falla": falla,
        }
    )

    rng = np.random.default_rng(semilla)

    for columna in ["Temperatura_C", "Vibracion_mm_s", "Presion_bar", "Carga_pct"]:
        indices = rng.choice(df.index, size=4, replace=False)
        df.loc[indices, columna] = np.nan

    indices_tipo = rng.choice(df.index, size=8, replace=False)
    df.loc[indices_tipo, "Tipo_Maquina"] = np.nan

    indices_atipicos = rng.choice(df.index, size=6, replace=False)
    df.loc[indices_atipicos, "Temperatura_C"] = [115, 120, 125, 130, 135, 140]

    return df


def generar_graficos(df):
    CARPETA_GRAFICOS.mkdir(exist_ok=True)

    print("\nSe generará el histograma de temperatura.")
    plt.figure(figsize=(9, 5))
    sns.histplot(data=df, x="Temperatura_C", bins=20, kde=True)
    plt.title("Distribución de la temperatura de las máquinas")
    plt.xlabel("Temperatura (°C)")
    plt.ylabel("Cantidad de registros")
    plt.tight_layout()
    plt.savefig(CARPETA_GRAFICOS / "histograma_temperatura.png", dpi=150)
    plt.show()

    print("\nSe generará el gráfico de dispersión.")
    plt.figure(figsize=(9, 5))
    sns.scatterplot(data=df, x="Temperatura_C", y="Vibracion_mm_s", hue="Falla")
    plt.title("Temperatura vs. Vibración")
    plt.xlabel("Temperatura (°C)")
    plt.ylabel("Vibración (mm/s)")
    plt.legend(title="Falla")
    plt.tight_layout()
    plt.savefig(CARPETA_GRAFICOS / "dispersion_temperatura_vibracion.png", dpi=150)
    plt.show()


def entrenar_modelo(df):
    X = df.drop(columns=["Falla"])
    y = df["Falla"]

    variables_numericas = [
        "Temperatura_C",
        "Vibracion_mm_s",
        "Presion_bar",
        "Horas_Operacion",
        "Carga_pct",
    ]
    variables_categoricas = ["Tipo_Maquina"]

    procesamiento = ColumnTransformer(
        transformers=[
            (
                "numericas",
                Pipeline(
                    steps=[
                        ("imputador", SimpleImputer(strategy="median")),
                        ("escalador", StandardScaler()),
                    ]
                ),
                variables_numericas,
            ),
            (
                "categoricas",
                Pipeline(
                    steps=[
                        ("imputador", SimpleImputer(strategy="most_frequent")),
                        ("codificador", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                variables_categoricas,
            ),
        ]
    )

    modelo = Pipeline(
        steps=[
            ("preprocesamiento", procesamiento),
            (
                "clasificador",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=SEMILLA,
                ),
            ),
        ]
    )

    X_entrenamiento, X_prueba, y_entrenamiento, y_prueba = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=SEMILLA,
        stratify=y,
    )

    print(f"\nRegistros para entrenamiento: {len(X_entrenamiento)}")
    print(f"Registros para prueba: {len(X_prueba)}")

    imprimir_titulo(12, "ENTRENAMIENTO DEL MODELO")
    modelo.fit(X_entrenamiento, y_entrenamiento)
    print("\nModelo entrenado correctamente.")

    predicciones = modelo.predict(X_prueba)
    exactitud = accuracy_score(y_prueba, predicciones)
    matriz = confusion_matrix(y_prueba, predicciones)
    reporte = classification_report(y_prueba, predicciones, digits=4)

    return exactitud, matriz, reporte


def main():
    df = generar_dataset()
    df.to_csv("fallas_maquinas.csv", index=False)

    imprimir_titulo(1, "EXPLORACIÓN DEL DATASET")
    print("\nPrimeros registros:")
    print(df.head())
    print("\nDimensiones del dataset:")
    print(df.shape)
    print("\nInformación del dataset:")
    print(df.info())
    print("\nEstadísticas descriptivas:")
    print(df.describe())

    imprimir_titulo(2, "VALORES NULOS")
    print(df.isnull().sum())

    imprimir_titulo(3, "DISTRIBUCIÓN DE LA VARIABLE OBJETIVO")
    print("\nCantidad de registros por clase:")
    print(df["Falla"].value_counts().sort_index())
    print("\nInterpretación:")
    print("0 = Funcionamiento normal")
    print("1 = Posible falla")

    imprimir_titulo(4, "ANÁLISIS ESTADÍSTICO")
    media_temperatura = df["Temperatura_C"].mean()
    mediana_temperatura = df["Temperatura_C"].median()
    varianza_temperatura = df["Temperatura_C"].var(ddof=0)
    desviacion_temperatura = df["Temperatura_C"].std(ddof=0)

    print(f"\nMedia de temperatura: {media_temperatura:.2f} °C")
    print(f"Mediana de temperatura: {mediana_temperatura:.2f} °C")
    print(f"Varianza de temperatura: {varianza_temperatura:.2f}")
    print(f"Desviación estándar de temperatura: {desviacion_temperatura:.2f} °C")

    varianza_manual = calcular_varianza_manual(df["Temperatura_C"])
    desviacion_manual = calcular_desviacion_manual(df["Temperatura_C"])
    varianza_numpy = np.nanvar(df["Temperatura_C"])
    desviacion_numpy = np.nanstd(df["Temperatura_C"])

    imprimir_titulo(5, "COMPARACIÓN DE CÁLCULOS")
    print(f"\nVarianza manual: {varianza_manual:.2f}")
    print(f"Varianza con NumPy: {varianza_numpy:.2f}")
    print(f"\nDesviación estándar manual: {desviacion_manual:.2f}")
    print(f"Desviación estándar con NumPy: {desviacion_numpy:.2f}")

    limite_inferior = media_temperatura - 3 * desviacion_temperatura
    limite_superior = media_temperatura + 3 * desviacion_temperatura
    mascara_atipicos = (
        (df["Temperatura_C"] < limite_inferior)
        | (df["Temperatura_C"] > limite_superior)
    )
    valores_atipicos = df.loc[mascara_atipicos, "Temperatura_C"]

    imprimir_titulo(6, "DETECCIÓN DE VALORES ATÍPICOS")
    print(f"\nLímite inferior: {limite_inferior:.2f}")
    print(f"Límite superior: {limite_superior:.2f}")
    print(f"\nCantidad de posibles valores atípicos: {len(valores_atipicos)}")
    print("\nValores identificados:")
    print(valores_atipicos.sort_values().to_list())

    imprimir_titulo(7, "ÁLGEBRA LINEAL - VECTORES")
    vector_1 = np.array([72, 4.3, 6.5])
    vector_2 = np.array([1.1, 0.5, 0.2])
    print("\nVector 1:")
    print(vector_1)
    print("\nVector 2:")
    print(vector_2)
    print(f"\nProducto punto: {np.dot(vector_1, vector_2):.2f}")
    print(f"\nNorma del vector 1: {np.linalg.norm(vector_1):.2f}")

    imprimir_titulo(8, "ÁLGEBRA LINEAL - MATRICES")
    matriz = np.array([[72, 4.3, 6.5], [75, 5.1, 7.0], [68, 3.8, 6.1]])
    print("\nMatriz:")
    print(matriz)
    print("\nTranspuesta de la matriz:")
    print(matriz.T)
    print("\nMatriz multiplicada por 2:")
    print(matriz * 2)

    imprimir_titulo(9, "SISTEMA DE ECUACIONES LINEALES")
    A = np.array([[2, 1], [1, 3]])
    b = np.array([8, 13])
    solucion = np.linalg.solve(A, b)
    print("\nMatriz A:")
    print(A)
    print("\nVector b:")
    print(b)
    print("\nSolución:")
    print(f"x = {solucion[0]:.2f}")
    print(f"y = {solucion[1]:.2f}")

    imprimir_titulo(10, "GENERACIÓN DE GRÁFICOS")
    generar_graficos(df)

    imprimir_titulo(11, "PREPARACIÓN DEL MODELO")
    exactitud, matriz_confusion, reporte = entrenar_modelo(df)

    imprimir_titulo(13, "EVALUACIÓN DEL MODELO")
    print(f"\nExactitud del modelo: {exactitud * 100:.2f}%")
    print("\nMatriz de confusión:")
    print(matriz_confusion)
    print("\nReporte de clasificación:")
    print(reporte)

    imprimir_titulo(14, "INTERPRETACIÓN DE LOS RESULTADOS")
    print(
        "\nEl modelo fue utilizado para clasificar máquinas en funcionamiento "
        "normal o posible falla."
    )
    print(f"\nLa exactitud obtenida fue de {exactitud * 100:.2f}%.")
    print(
        "\nLa matriz de confusión permite observar qué cantidad de registros "
        "fueron clasificados correcta e incorrectamente."
    )
    print(
        "\nEl modelo sirve como una herramienta de apoyo para identificar "
        "posibles condiciones anormales, pero no reemplaza una inspección "
        "técnica de la máquina."
    )

    print("\n")
    print("=" * 60)
    print("PROCESO FINALIZADO")
    print("=" * 60)


if __name__ == "__main__":
    main()
