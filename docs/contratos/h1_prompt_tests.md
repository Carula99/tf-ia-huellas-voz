# Prompt usado para escribir tests/test_h1.py

Se pega en un chat nuevo, sin el contexto del desarrollo, seguido del contenido completo de `h1_generador.md`.

- **Fecha:** 16/10/2026
- **Herramienta y modelo:** Claude (claude-opus-5-5)

```text
Actuá como un ingeniero de testing en Python. Voy a implementar la etapa H1 (generador de huellas sintéticas y dataset) del trabajo final de IA I (Python 3.12, Windows, VS Code). Abajo va el contrato completo: firmas exactas, tipos y criterios de aceptación. Escribí un único archivo, tests/test_h1.py, con pytest, que verifique cada punto de la sección 5 del contrato (la 6 es manual: no la automatices).

Reglas:
- No implementes nada de src/ ni propongas cómo hacerlo: solo los tests.
- Raíz del repo: RAIZ = Path(__file__).resolve().parents[1]. La configuración real se carga con src.config.cargar_config().
- Importá src.config, src.huellas.generador, src.huellas.variantes y src.huellas.dataset dentro de cada test o de un fixture, no arriba del archivo: si algo todavía no existe, que fallen esos tests y no la colección entera.
- Usá solo la biblioteca estándar, pytest, numpy, scipy, opencv (cv2) y PyYAML.
- Para la configuración reducida, partí de una copia profunda del cfg real y cambiá solo los conteos.
- Lo que escriba archivos va en tmp_path o tmp_path_factory: nunca escribas en data/ ni en docs/ del repo.
- Generar bases y datasets lleva tiempo: usá fixtures con scope="module" para no repetirlos, y que el archivo completo (sin el test de aceptación) tarde a lo sumo un par de minutos.
- El test de la sección 5.8 lleva @pytest.mark.aceptacion; todos los demás corren con pytest a secas.
- Un test por criterio, con nombres descriptivos en español y mensajes de error que digan qué falló (valor medido y esperado). Usá parametrize para tipos y semillas.
- Si algo del contrato es ambiguo o contradictorio, preguntame antes de escribir el archivo.

=== CONTRATO ===
[contenido de docs/contratos/h1_generador.md]
```
