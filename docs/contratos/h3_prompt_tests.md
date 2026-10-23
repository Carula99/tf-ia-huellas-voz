
# Prompt para escribir tests/test_h3.py

Se pega en un chat nuevo, sin el contexto del desarrollo, seguido del contenido completo de `h3_caracteristicas.md`.

- **Fecha:** 22/10/2026
- **Herramienta y modelo:** Claude, en otro chat (anotar el modelo cuando entregue el archivo).

```text
Actuá como un ingeniero de testing en Python. Voy a implementar la etapa H3 (características de huellas por bloques —orientación, Gabor y minucias—, escalador con imputación, PCA propio y silhouette propia) del trabajo final de IA I (Python 3.12, Windows, VS Code). Las etapas H1 (generador de huellas sintéticas y variantes) y H2 (preprocesamiento: ventana de 250 × 250 alineada, realce, binarización y esqueleto) ya están implementadas y probadas: los tests las usan para generar y preprocesar las huellas. Abajo va el contrato completo de H3: firmas exactas, definiciones y criterios de aceptación. Escribí un único archivo, tests/test_h3.py, con pytest, que verifique cada punto de la sección 6 del contrato (la 7 es manual: no la automatices).

Reglas:
- No implementes nada de src/ ni propongas cómo hacerlo: solo los tests. Los oráculos de la sección 4 (bloques y minucias) se escriben dentro del test, a partir del contrato.
- Raíz del repo: RAIZ = Path(__file__).resolve().parents[1]. La configuración real se carga con src.config.cargar_config().
- Importá src.config, src.huellas.generador, src.huellas.variantes, src.huellas.dataset, src.huellas.preprocesamiento, src.huellas.caracteristicas, src.core_ml.pca y src.core_ml.metricas dentro de cada test o de un fixture, no arriba del archivo: si algo todavía no existe, que fallen esos tests y no la colección entera.
- De H1 y H2 usá solo lo que nombra el contrato: generar_base, ParametrosVariante, sortear_parametros, aplicar_variante, plan_dataset, generar_dataset, preprocesar y HuellaPreprocesada, y los campos de Variante (imagen, parametros).
- Usá solo la biblioteca estándar, pytest, numpy, scipy, opencv (cv2), PyYAML y scikit-learn (solo como oráculo: sklearn.decomposition.PCA y sklearn.metrics.silhouette_score).
- Para probar que cargar_config exige las claves nuevas, escribí copias modificadas de config.yaml en tmp_path; nunca toques el config.yaml del repo.
- Lo que escriba archivos va en tmp_path o tmp_path_factory: nunca escribas en data/ ni en docs/ del repo.
- Generar, preprocesar y calcular las características lleva tiempo: armá la muestra de la sección 5, sus resultados de H2, sus bloques y el escalador una sola vez, con fixtures de scope="module", y que el archivo completo (sin el test de aceptación) tarde a lo sumo un par de minutos.
- El test de la sección 6.9 lleva @pytest.mark.aceptacion; todos los demás corren con pytest a secas.
- Un test por criterio, con nombres descriptivos en español y mensajes de error que digan qué falló: valor medido y esperado y, en los criterios por identidad, el valor de cada identidad (y en los de cada variante, cuáles fallaron). Usá parametrize donde ayude.
- Si algo del contrato es ambiguo o contradictorio, preguntame antes de escribir el archivo.

=== CONTRATO ===
[contenido de docs/contratos/h3_caracteristicas.md]
```
