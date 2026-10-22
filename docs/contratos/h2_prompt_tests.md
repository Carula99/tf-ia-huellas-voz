
# Prompt para escribir tests/test_h2.py

Se pega en un chat nuevo, sin el contexto del desarrollo, seguido del contenido completo de `h2_preprocesamiento.md`.

- **Fecha:** [completar el día que se use]
- **Herramienta y modelo:** [completar]

```text
Actuá como un ingeniero de testing en Python. Voy a implementar la etapa H2 (preprocesamiento de huellas: segmentación, punto de referencia, enderezado, ventana de 250 × 250, realce, binarización y esqueleto) del trabajo final de IA I (Python 3.12, Windows, VS Code). La etapa H1 (generador de huellas sintéticas y variantes) ya está implementada y probada: los tests la usan para generar las huellas. Abajo va el contrato completo de H2: firmas exactas, tipos y criterios de aceptación. Escribí un único archivo, tests/test_h2.py, con pytest, que verifique cada punto de la sección 5 del contrato (la 6 es manual: no la automatices).

Reglas:
- No implementes nada de src/ ni propongas cómo hacerlo: solo los tests.
- Raíz del repo: RAIZ = Path(__file__).resolve().parents[1]. La configuración real se carga con src.config.cargar_config().
- Importá src.config, src.huellas.generador, src.huellas.variantes, src.huellas.dataset y src.huellas.preprocesamiento dentro de cada test o de un fixture, no arriba del archivo: si algo todavía no existe, que fallen esos tests y no la colección entera.
- De H1 usá solo lo que nombra el contrato: generar_base, ParametrosVariante, sortear_parametros, aplicar_variante, plan_dataset y los campos de HuellaBase (imagen, mascara, orientacion, singularidades, periodo_px) y de Variante (imagen, mascara, singularidades, punto_referencia, parametros, mapa_x, mapa_y).
- Usá solo la biblioteca estándar, pytest, numpy, scipy, opencv (cv2) y PyYAML.
- Para probar que cargar_config exige las claves nuevas, escribí copias modificadas de config.yaml en tmp_path; nunca toques el config.yaml del repo.
- Lo que escriba archivos va en tmp_path o tmp_path_factory: nunca escribas en data/ ni en docs/ del repo.
- Generar y preprocesar lleva tiempo: generá la muestra de la sección 4 y sus resultados una sola vez, con fixtures de scope="module", y que el archivo completo (sin el test de aceptación) tarde a lo sumo un par de minutos.
- El test de la sección 5.9 lleva @pytest.mark.aceptacion; todos los demás corren con pytest a secas.
- Un test por criterio, con nombres descriptivos en español y mensajes de error que digan qué falló: valor medido y esperado y, en los criterios del 95 %, la proporción, cuántas variantes fallaron y cuáles (archivo del plan, error medido). Usá parametrize para los tipos.
- Si algo del contrato es ambiguo o contradictorio, preguntame antes de escribir el archivo.

=== CONTRATO ===
[contenido de docs/contratos/h2_preprocesamiento.md]
```
