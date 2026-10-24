# Prompt para escribir tests/test_v4.py

Se usa en un chat o ventana aislada (con un modelo distinto de Gemini, como Claude Sonnet 3.7 / Opus o GPT-4o), sin el contexto del desarrollo ni acceso a los archivos de prototipos ni de código previo, con `v4_clasificacion.md` adjunto o pegado a continuación.

- **Fecha:** 24/10/2026
- **Herramienta y modelo:** Ventana aislada / modelo externo (distinto de Gemini). Lee únicamente `v4_clasificacion.md` (versión 1.1). Genera `tests/test_v4.py` congelado antes de iniciar la implementación en `src/`.

```text
Actuá como un ingeniero de testing en Python. Voy a implementar la etapa V4 (clasificador K-NN propio con voto ponderado por distancia, selección de hiperparámetros por validación cruzada dejando una persona afuera en train, reponderación de bloques y proyección PCA, normalización L2 unitaria por fila, calibración del umbral de rechazo tau* en val, y reconocimiento completo de audio) del trabajo final de IA I (Python 3.12, Windows, VS Code).

Las etapas previas (Fase 0 de configuración, H3 con PCA propio en src/core_ml/pca.py, H4 con matriz_confusion en src/core_ml/metricas.py, V2 con preprocesamiento y V3 con características acústicas por bloques, escalador ponderado y caracteristicas.npz) ya están aprobadas: los tests pueden apoyarse en ellas según lo describe el contrato.
Esta etapa agrega src/core_ml/knn.py y src/voz/clasificacion.py.

El contrato completo es el archivo adjunto v4_clasificacion.md: firmas exactas, definiciones y criterios de aceptación. Escribí un único archivo, tests/test_v4.py, con pytest, que verifique cada punto de la sección 6 del contrato (la sección 7 es manual: no la automatices).

Reglas operativas estrictas:
1. Leé solo v4_clasificacion.md. No abras ni leas ningún otro archivo del repositorio ni del entorno (no abras prototipos, ni código previo, ni transcripts ni documentos de traspaso): los tests deben salir exclusivamente de las especificaciones del contrato.
2. No implementes nada de src/ ni propongas cómo resolverlo: tu única entrega es el archivo de tests. Los oráculos de las secciones 4 y 6 (oráculo de K-NN, de calibración, de validación cruzada y de evaluación) se escriben dentro del archivo de test.
3. Raíz del repositorio: RAIZ = Path(__file__).resolve().parents[1]. La configuración real se carga con src.config.cargar_config().
4. Importá los módulos de src/ dentro de cada test o dentro de fixtures de pytest, nunca en el tope del archivo: si una función o clase aún no existe o falla su importación, deben fallar solo los tests que la usan y no la recolección global de pytest. Módulos involucrados: src.config, src.core_ml.knn, src.core_ml.pca, src.core_ml.metricas, src.voz.preprocesamiento, src.voz.caracteristicas, src.voz.clasificacion.
5. Prohibición estricta de scikit-learn en src/: el test (sección 6.2) debe verificar explícitamente que los archivos src/core_ml/knn.py y src/voz/clasificacion.py no contengan la palabra "sklearn". En cambio, dentro de tests/test_v4.py podés y debés usar scikit-learn libremente como oráculo comparativo (sklearn.neighbors.KNeighborsClassifier, sklearn.decomposition.PCA, sklearn.metrics.confusion_matrix).
6. Bibliotecas permitidas en el test: biblioteca estándar de Python 3.12, pytest, numpy, scipy, soundfile, librosa, PyYAML, matplotlib y scikit-learn (solo como oráculo).
7. Aislamiento de disco: todo archivo temporal (archivos de modelo, copias de config, datasets chicos, figuras temporales) debe crearse en tmp_path o tmp_path_factory. Nunca escribas ni modifiques archivos dentro de data/, models/ ni docs/ del repositorio.
8. Rendimiento: armá los datos sintéticos (sección 6.2 a 6.7) y el dataset chico (sección 6.8 con clips sintéticos y metadatos) usando fixtures de pytest con scope="module", para que la corrida habitual rápida (sin @pytest.mark.aceptacion) complete en pocos segundos.
9. Marcas y aceptación: el test de la sección 6.9 debe llevar obligatoriamente la marca @pytest.mark.aceptacion y saltearse con pytest.skip si no existe data/voz/prepro/caracteristicas.npz. Todos los demás tests (secciones 6.1 a 6.8) deben correr con pytest a secas sin flags especiales.
10. Estilo de tests: un test por criterio con nombres descriptivos en español (test_knn_vecinos_oraculo, test_seleccionar_hiperparametros_criterio_suma, etc.), aserciones con mensajes de error informativos que detallen el valor obtenido vs esperado, y uso de @pytest.mark.parametrize donde simplifique la cobertura.
11. Entregá el código completo de tests/test_v4.py listo para guardar y ejecutar. Si detectás alguna inconsistencia o ambigüedad en el contrato, preguntame antes de asumir un comportamiento.
```

