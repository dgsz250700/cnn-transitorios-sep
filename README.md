# Red neuronal convolucional para clasificar fenómenos transitorios electromagnéticos en SEP

Proyecto de grado de **Daniela Gutierrez** para la Universidad de los Andes.

## Descripción
Este repositorio contiene el código y el documento de tesis del proyecto enfocado en la clasificación de fenómenos transitorios electromagnéticos en sistemas eléctricos de potencia mediante una **red convolucional 1D (CNN)**.

## Alcance del proyecto
Se trabaja con dos clases principales:
- **Banco**: transitorios asociados a energización/desenergización de bancos de condensadores.
- **Línea**: transitorios asociados a energización de línea de transmisión.

## Contenido sugerido del repositorio
```text
.
├── README.md
├── tesis.pdf
├── src/
│   └── cnn_transients.py
├── data/
│   ├── prueba_banco.csv
│   ├── test_banco.csv
│   ├── prueba_linea.csv
│   └── test_linea.csv
├── results/
│   └── .gitkeep
├── requirements.txt
└── .gitignore
```

## Requisitos
Instala las dependencias con:
```bash
pip install -r requirements.txt
```

## Ejecución
Desde la raíz del repositorio:
```bash
python src/cnn_transients.py
```

## Notas
- El script asume que los archivos CSV están disponibles.
- Puedes ajustar hiperparámetros como número de filtros, tamaño del kernel, épocas y batch size.
- Se guarda automáticamente el mejor modelo durante el entrenamiento.


