# Sistema de reconocimiento de huellas dactilares controlado por reconocimiento de voz

Trabajo Final de **Inteligencia Artificial I** · Ingeniería en Mecatrónica · Facultad de Ingeniería, UNCuyo · 2026.

Agente de control de acceso con interfaz gráfica. Con comandos de voz ("Reconocer", "Identificar" y "Salir", clasificados con un K-NN) decide si una huella dactilar sintética pertenece a una de tres identidades autorizadas o a un intruso (K-means con k = 3, umbral de distancia calibrado en validación y análisis PCA). PCA, K-means y K-NN están programados desde cero en NumPy; scikit-learn se usa solo como oráculo en los tests.

Estado: Fase 0 (setup) · tag `v0.0-setup`.

## Instalación

Requisitos: Windows 10 u 11, [Python 3.12](https://www.python.org/downloads/) y Git.

```powershell
git clone https://github.com/Carula99/tf-ia-huellas-voz.git
cd tf-ia-huellas-voz
py -3.12 -m venv venv
venv\Scripts\python -m pip install --upgrade pip
venv\Scripts\python -m pip install -r requirements.txt
```

No hace falta activar el venv: todos los comandos usan `venv\Scripts\python`. Si preferís activarlo (`venv\Scripts\Activate.ps1`) y PowerShell avisa que la ejecución de scripts está deshabilitada, se habilita solo para tu usuario con `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`.

En VS Code: abrir la carpeta del repo y elegir el intérprete `venv\Scripts\python.exe` (Ctrl+Shift+P → *Python: Select Interpreter*). Para los notebooks, elegir el mismo entorno como kernel.

## Tests

Cada etapa tiene un contrato en `docs/contratos/` y sus tests en `tests/`. Los tests se escriben en otro chat solo a partir del contrato y se congelan (commit) antes de programar: si un test falla, se corrige el código, nunca el test.

```powershell
venv\Scripts\python -m pytest
```

`pytest` a secas corre los tests rápidos. Los de aceptación (metas sobre el dataset completo) se corren aparte:

```powershell
venv\Scripts\python -m pytest -m aceptacion
```

## Estructura del proyecto

```text
tf-ia-huellas-voz/
├── README.md
├── requirements.txt     dependencias con versiones fijadas
├── pyproject.toml       configuración de pytest
├── config.yaml          semillas, rutas, tamaños e hiperparámetros
├── src/
│   ├── config.py        lectura de config.yaml
│   ├── core_ml/         PCA, K-means++, K-NN y validación dejando una persona afuera (NumPy)
│   ├── huellas/         H1 generador · H2 preprocesamiento · H3 características · H4 umbral
│   ├── voz/             V1 ingesta · V2 preprocesamiento · V3 MFCC · V4 K-NN y rechazo
│   ├── agente/          máquina de estados del agente
│   └── app/             interfaz CustomTkinter
├── tests/               un archivo de tests por etapa
├── notebooks/           exploración y figuras del informe
├── data/                fuera del repo: se regenera con las semillas de config.yaml
├── models/              fuera del repo: centroides, escaladores y umbrales (.joblib)
└── docs/
    ├── contratos/       contrato y prompt de los tests de cada etapa
    └── figuras/         diagramas del plan y figuras del informe
```

`data/` y `models/` no se suben al repo (solo su `.gitkeep`). Las grabaciones de voz de terceros no se publican.

## Licencias y créditos

- **Huellas:** generador propio inspirado en SFinGe (Cappelli et al., 2000), con el modelo de orientación de Sherlock y Monro (1993). No se usan huellas reales.
- **Voces sintéticas:** edge-tts, que usa el servicio de texto a voz de Microsoft Edge.
- **Librerías**, cada una con su licencia: NumPy, SciPy, scikit-image, scikit-learn, soundfile, joblib e ipykernel (BSD); OpenCV (Apache 2.0); librosa (ISC); sounddevice, PyYAML y pytest (MIT); edge-tts (LGPL-3.0); CustomTkinter (CC0-1.0 / MIT según los metadatos del paquete); Matplotlib (licencia de Matplotlib, basada en la PSF).
- **IA generativa:** su uso (herramientas, prompts y cómo se verificó cada resultado) se documenta en el informe.
- **Código:** licencia a definir antes de hacer público el repo.
