# Prompt para escribir tests/test_v3.py

Se usa en un chat o ventana aislada, sin el contexto del desarrollo, con `v3_caracteristicas.md` adjunto.

- **Fecha:** 24/10/2026
- **Herramienta y modelo:** [a completar al congelar `tests/test_v3.py`]

```text
Actuá como un ingeniero de testing en Python. Voy a implementar la etapa V3 (extracción de características acústicas de voz por bloques —MFCC con CMN, Δ, energía RMS, cruces por cero y duración—, escalador ponderado y PCA propio) del trabajo final de IA I (Python 3.12, Windows, VS Code). El contrato completo es el archivo adjunto v3_caracteristicas.md: firmas exactas, definiciones y criterios de aceptación. Escribí un único archivo, tests/test_v3.py, con pytest, que verifique cada punto de la sección 6 del contrato (la 7 es manual: no la automatices).

Reglas:
- Leé solo v3_caracteristicas.md. No abras ningún otro archivo de la carpeta IA 1 ni del repo (hay documentos del desarrollo, prototipos y datos privados): los tests salen únicamente del contrato.
- Cuando termines, guardá el archivo en IA 1\v3_para_tests\test_v3.py (o entregalo como test_v3.py).
- No implementes nada de src/ ni propongas cómo hacerlo: solo los tests. El oráculo de la sección 4.2 se escribe dentro del test a partir del contrato.
- Raíz del repo: RAIZ = Path(__file__).resolve().parents[1]. La configuración real se carga con src.config.cargar_config().
- Importá src.config, src.core_ml.pca, src.core_ml.metricas y src.voz.caracteristicas dentro de cada test o de un fixture, no arriba del archivo: si algo todavía no existe, que fallen esos tests y no la colección entera.
- Usá solo la biblioteca estándar, pytest, numpy, librosa, soundfile, PyYAML y scikit-learn (solo como oráculo: sklearn.decomposition.PCA y sklearn.metrics.silhouette_score).
- Para probar que cargar_config exige las claves nuevas, escribí copias modificadas de config.yaml en tmp_path; nunca toques el config.yaml del repo.
- Lo que escriba archivos va en tmp_path o tmp_path_factory: nunca escribas en data/ ni en docs/ del repo. En el test de aceptación (6.6) pasale salida=tmp_path / "caracteristicas.npz" a generar_caracteristicas.
- Armá la muestra sintética de la sección 5 y sus bloques una sola vez con fixtures de scope="module", para que la corrida rápida (sin @pytest.mark.aceptacion) tarde pocos segundos.
- El test de la sección 6.6 lleva @pytest.mark.aceptacion y se saltea con pytest.skip si no existe data/voz/prepro/metadatos.csv; todos los demás corren con pytest a secas.
- Un test por criterio, con nombres descriptivos en español y mensajes de error claros que muestren el valor medido y el esperado. Usá parametrize donde ayude.
- Si algo del contrato es ambiguo o contradictorio, preguntame antes de escribir el archivo.
```

