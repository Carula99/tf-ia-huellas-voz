# Instrucciones para agentes (VS Code: Claude Code, Copilot, Cline, Continue, Gemini…)

Trabajo Final de Inteligencia Artificial I (UNCuyo, 2026) de Santiago: reconocimiento de huellas dactilares sintéticas controlado por comandos de voz. Presentación: semana del 10/11/2026 (informe PDF + demo + coloquio). Se habla en español rioplatense, informal y al grano; explicaciones cortas, con las ecuaciones cuando hacen falta.

## Antes de hacer nada

1. Leé `docs/traspaso/tf-ia-estado.md`: estado de cada etapa, decisiones, lo que sigue y dónde está cada cosa. Es la versión vigente del traspaso.
2. Leé el contrato de la etapa en la que vas a trabajar (`docs/contratos/`).
3. Mirá `git log --oneline -10` y `git status`.

## El método (siempre igual)

1. Contrato de la etapa en `docs/contratos/<etapa>_*.md` (firmas, esquema y criterios medibles) y su prompt de tests en `docs/contratos/<etapa>_prompt_tests.md`. Commit propio.
2. Tests escritos por **otro** agente o chat que lee solo el contrato (ver "Escribir tests"). Se congelan con un commit antes de programar.
3. Implementación hasta pytest verde **sin tocar los tests**. Si un test o el contrato están mal, se cambian de forma explícita (versión nueva del contrato) y se anota en el Documento 2.
4. Verificación manual (sección 8 del contrato; la hace Santiago), tag anotado de la etapa con push y actualización del Documento 2.

PCA, K-means y K-NN se programan desde cero con NumPy (`src/core_ml/`); scikit-learn solo se usa como oráculo en los tests.

## Entorno (Windows 11)

- PowerShell con la ejecución de scripts deshabilitada: **no se activa el venv**. Todo se corre desde la raíz del repo con `venv\Scripts\python …`. Python 3.12.10 en `venv`; el `python` del sistema es 3.14 y no se usa.
- Tests rápidos: `venv\Scripts\python -m pytest tests\test_setup.py tests\test_v1.py -q` (los de tu carril y los de sus etapas anteriores). Aceptación (lenta): `venv\Scripts\python -m pytest -m aceptacion tests\test_v1.py`.
- No correr al mismo tiempo dos procesos pesados (generar un dataset, la aceptación o un entrenamiento).
- La carpeta `IA 1` es `C:\Users\Santiago\OneDrive\Escritorio\IA 1` (OneDrive). El repo está fuera de OneDrive: `C:\Users\Santiago\source\repos\tf-ia-huellas-voz`.

## Git

- `git add` con rutas explícitas. **Nunca** `git add -A`, `git add .`, `git checkout .`, `git reset --hard`, `git stash` ni `git clean`: puede haber trabajo a medio hacer de otro carril.
- Antes de cada commit, `git log` y `git status` (otros chats también commitean).
- Mensaje en un archivo UTF-8 (`git commit -F .git\MSG.txt`; el archivo se borra después), con el estilo de los anteriores (`V2: …`) y una línea `Co-Authored-By:` del agente que lo hizo. Tags anotados con el mensaje en un archivo (`git tag -a <tag> -F …`).
- Push de `main` (y del tag) cada vez que se cierra algo.

## Carriles

Huellas (H1–H4) está terminado; voz (V1–V4) va por V2; después viene la Fase 5 (integración y app). Si trabajan dos agentes a la vez, cada uno toca solo lo suyo:

- Voz: `src/voz/`, `scripts/` de voz, `tests/test_v*.py` y `docs/contratos/v*`.
- Huellas: `src/huellas/`, `tests/test_h*.py` y `docs/contratos/h*`.
- `src/core_ml/` es compartido: lo que fija un contrato no se cambia sin cambiar ese contrato.
- `config.yaml`, `src/config.py`, `README.md` y `requirements.txt`: solo cambios chicos en el lugar, leyendo el archivo justo antes; nunca se reemplazan enteros.
- Los tests congelados no se editan nunca.

## Escribir tests (paso 2)

El agente que escribe los tests no puede ver la implementación ni los prototipos. En VS Code, lo más seguro es abrir una ventana nueva con una carpeta que tenga **solo** el contrato y el prompt (por ejemplo `IA 1\v2_para_tests`), pegarle el bloque de texto de `docs/contratos/<etapa>_prompt_tests.md` y traer después el `test_<etapa>.py` al repo con un commit propio ("tests congelados"). Si se hace con el repo abierto, el prompt tiene que prohibir leer `src/`, `docs/respaldo/` (tiene los prototipos) y `docs/traspaso/`. Anotar la herramienta y el modelo en el prompt (campos del principio del archivo).

## Datos y privacidad

- `data/` y `models/` no están en Git: se regeneran con los scripts y las semillas de `config.yaml` (los comandos están en el traspaso). Los audios crudos están solo en la notebook (`data/voz/`) y en OneDrive (`IA 1\Trabajo final\Respaldos`).
- Las personas que grabaron se nombran solo por código (p01 a p29). No se escriben nombres en ningún archivo del repo (tampoco los de las subcarpetas de WhatsApp), no se abre `IA 1\Trabajo final\Privado\` y no se suben audios.

## Documentos

- Los dos Word vigentes están en OneDrive, en `IA 1\Trabajo final`: el Plan de desarrollo v0.3.2 y el Documento 2 (Documentación del desarrollo, v0.6.9). Se generan con los `.js` de `IA 1\Trabajo final\Fuentes de los Word` (Node con `docx@9.7.1`; ver su `LEEME.txt`): se copia el último `doc2_vXXX.js` con otro número, se edita y se regenera. Las decisiones y los prompts siguen numerándose desde el último (hoy, decisión 74 y prompt P22).
- El Documento 2 lo actualiza un agente por vez, al cerrar cada etapa: ficha, decisiones, bitácora y uso de IA (con el prompt usado).
- `docs/respaldo/` es una **copia de respaldo** del 23/10 (Word, fuentes, scripts de prototipos, enunciado y transcripciones): no se edita; lo vigente está en OneDrive. Ver `docs/respaldo/LEEME.md`.
- Al terminar una sesión de trabajo, actualizar `docs/traspaso/tf-ia-estado.md` (leerlo antes y conservar lo del otro carril) y commitearlo.
