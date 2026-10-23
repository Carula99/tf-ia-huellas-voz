# Prompt para escribir tests/test_h4.py

Se pega en un chat nuevo, sin el contexto del desarrollo, seguido del contenido completo de `h4_clasificacion.md`.

- **Fecha:** 23/10/2026
- **Herramienta y modelo:** Claude, en otro chat, con el contrato versión 1. Revisó el contrato contra los de H1 a H3 y el prototipo, armó el dataset chico y comprobó los τ\* de la sección 8. Hizo tres preguntas, que Santiago respondió: si podía probar el archivo contra una implementación descartable, si `far_max_val` se valida también con `eer` y qué tipos acepta para los escalares. Esas respuestas y sus lecturas del contrato pasaron a la versión 1.1. Probó el archivo contra la implementación descartable, que no se entrega, y con 36 errores plantados, de los que detecta 35. Entregó `tests/test_h4.py` el 23/10/2026: 174 tests (169 normales y 5 de aceptación).

```text
Actuá como un ingeniero de testing en Python. Voy a implementar la etapa H4 (K-means propio, distancia normalizada por la dispersión de cada cluster, umbral de rechazo calibrado en validación, matriz de confusión propia, archivo del modelo y reconocimiento de una imagen) del trabajo final de IA I (Python 3.12, Windows, VS Code). Las etapas H1 (generador de huellas sintéticas y dataset), H2 (preprocesamiento) y H3 (características por bloques, escalador, PCA propio y caracteristicas.npz) ya están implementadas y probadas: los tests las usan para armar los datasets. Abajo va el contrato completo de H4: firmas exactas, definiciones y criterios de aceptación. Escribí un único archivo, tests/test_h4.py, con pytest, que verifique cada punto de la sección 6 del contrato (la 7 es manual: no la automatices).

Reglas:
- No implementes nada de src/ ni propongas cómo hacerlo: solo los tests. Los oráculos de la calibración, la evaluación y las dispersiones se escriben dentro del test, a partir de las secciones 0 y 4 del contrato.
- Raíz del repo: RAIZ = Path(__file__).resolve().parents[1]. La configuración real se carga con src.config.cargar_config().
- Importá src.config, src.huellas.dataset, src.huellas.generador, src.huellas.variantes, src.huellas.preprocesamiento, src.huellas.caracteristicas, src.huellas.clasificacion, src.core_ml.kmeans y src.core_ml.metricas dentro de cada test o de un fixture, no arriba del archivo: si algo todavía no existe, que fallen esos tests y no la colección entera.
- De H1, H2 y H3 usá solo lo que nombra el contrato: plan_dataset, generar_dataset, la receta de la sección 4 de H1 (generar_base, sortear_parametros, aplicar_variante y el campo imagen de la variante) para el dataset chico, preprocesar, BLOQUES, caracteristicas, Escalador, ajustar_escalador y generar_caracteristicas.
- Usá solo la biblioteca estándar, pytest, numpy, scipy, opencv (cv2), PyYAML y scikit-learn (solo como oráculo: sklearn.cluster.KMeans y sklearn.metrics.confusion_matrix).
- Para probar que cargar_config exige la clave nueva, escribí copias modificadas de config.yaml en tmp_path; nunca toques el config.yaml del repo.
- Lo que escriba archivos va en tmp_path o tmp_path_factory: nunca escribas en data/, models/ ni docs/ del repo.
- Armá el dataset chico de la sección 5 (PNG, metadatos.csv y caracteristicas.npz) una sola vez, con un fixture de scope="module", y que el archivo completo (sin el test de aceptación) tarde a lo sumo un par de minutos.
- El test de la sección 6.9 lleva @pytest.mark.aceptacion; todos los demás corren con pytest a secas.
- Un test por criterio, con nombres descriptivos en español y mensajes de error que digan qué falló (valor medido y esperado). Usá parametrize donde ayude.
- Si algo del contrato es ambiguo o contradictorio, preguntame antes de escribir el archivo.

=== CONTRATO ===
[contenido de docs/contratos/h4_clasificacion.md]
```
