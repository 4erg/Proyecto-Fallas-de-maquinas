# Entregable: analisis predictivo de fallas en maquinas

Este proyecto contiene un script en Python que genera un dataset simulado de maquinas industriales, realiza analisis estadistico, crea graficos y entrena un modelo de regresion logistica para predecir posibles fallas.

## Archivos del proyecto

- `entregable.py`: script principal del trabajo.
- `requirements.txt`: lista de librerias necesarias para ejecutar el script.
- `graficos/`: carpeta donde se guardan los graficos generados al ejecutar el programa.

## Requisitos

Antes de ejecutar el proyecto, cada computadora debe tener instalado:

- Python 3.10 o superior.
- pip, que normalmente viene incluido con Python.

Para instalar Python en Windows, descargarlo desde:

https://www.python.org/downloads/

Durante la instalacion, marcar la opcion:

```text
Add Python to PATH
```

## Instalacion en Windows

Abrir PowerShell o CMD en la carpeta del proyecto.

Ejemplo:

```powershell
cd "C:\Users\user\Downloads\tareaç"
```

Crear un entorno virtual:

```powershell
py -m venv .venv
```

Activar el entorno virtual:

```powershell
.\.venv\Scripts\activate
```

Instalar las dependencias:

```powershell
py -m pip install -r requirements.txt
```

Ejecutar el programa:

```powershell
py entregable.py
```

## Instalacion en macOS o Linux

Abrir una terminal en la carpeta del proyecto.

Crear un entorno virtual:

```bash
python3 -m venv .venv
```

Activar el entorno virtual:

```bash
source .venv/bin/activate
```

Instalar las dependencias:

```bash
python3 -m pip install -r requirements.txt
```

Ejecutar el programa:

```bash
python3 entregable.py
```

## Que hace el programa

El script realiza los siguientes pasos:

1. Genera un dataset simulado de 600 registros.
2. Incluye variables como temperatura, vibracion, presion, horas de operacion, carga y tipo de maquina.
3. Simula una variable objetivo llamada `Falla`.
4. Agrega valores nulos y valores atipicos para el analisis.
5. Calcula media, mediana, varianza y desviacion estandar.
6. Compara calculos manuales con funciones de NumPy.
7. Detecta posibles valores atipicos.
8. Realiza operaciones basicas de algebra lineal.
9. Genera graficos con Matplotlib y Seaborn.
10. Entrena un modelo de regresion logistica.
11. Evalua el modelo con exactitud, matriz de confusion y reporte de clasificacion.

## Salida esperada

Al ejecutar el archivo, se mostrara en consola:

- Informacion general del dataset.
- Conteo de valores nulos.
- Distribucion de la variable objetivo.
- Estadisticas descriptivas.
- Valores atipicos detectados.
- Resultados de algebra lineal.
- Metricas del modelo de clasificacion.

Tambien se crearan dos graficos dentro de la carpeta `graficos/`:

- `histograma_temperatura.png`
- `dispersion_temperatura_vibracion.png`

## Problemas comunes

Si aparece un mensaje como:

```text
Python no se encontro
```

significa que Python no esta instalado o no fue agregado al PATH. La solucion es reinstalar Python desde `python.org` y marcar la opcion `Add Python to PATH`.

Si aparece un error indicando que falta una libreria, ejecutar nuevamente:

```powershell
py -m pip install -r requirements.txt
```

Si el comando `py` no funciona, probar con:

```powershell
python entregable.py
```

o:

```powershell
python3 entregable.py
```
