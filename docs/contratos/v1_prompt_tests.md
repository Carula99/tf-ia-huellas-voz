# Prompt para escribir tests/test_v1.py

Se pega en un chat nuevo, sin el contexto del desarrollo, seguido del contenido completo de `v1_dataset.md`.

- **Fecha:** [completar]
- **Herramienta y modelo:** [completar]

```text
Actuá como un ingeniero de testing en Python. Voy a implementar la etapa V1 (dataset de voz) del trabajo final de IA I (Python 3.12, Windows, VS Code). Abajo va el contrato completo: firmas exactas, tipos y criterios de aceptación. Escribí un único archivo, tests/test_v1.py, con pytest, que verifique cada punto de la sección 7 del contrato (la 8 es manual: no la automatices).

Reglas:
- No implementes nada de src/ ni propongas cómo hacerlo: solo los tests.
- Raíz del repo: RAIZ = Path(__file__).resolve().parents[1]. La configuración real se carga con src.config.cargar_config().
- Importá src.config, src.voz.audio, src.voz.segmentacion, src.voz.tts y src.voz.dataset dentro de cada test o de un fixture, no arriba del archivo: si algo todavía no existe, que fallen esos tests y no la colección entera.
- Usá solo la biblioteca estándar, pytest, numpy, scipy, soundfile, av (PyAV) y PyYAML.
- Los audios de prueba se sintetizan en el test (tonos con armónicos y rampas, ruido gaussiano, silencios) y se escriben con soundfile: WAV a 44,1 kHz para la notebook, OGG/Opus para WhatsApp y MP3 para las voces sintéticas. El M4A (AAC) se escribe con PyAV.
- Nunca uses internet: las voces sintéticas se prueban con un sintetizador falso que se pasa como argumento.
- Para la configuración reducida, partí de una copia profunda del cfg real y cambiá solo lo necesario (particiones, voces, ruido).
- Lo que escriba archivos va en tmp_path o tmp_path_factory: nunca escribas en data/ ni en docs/ del repo.
- Armar la fuente de prueba y generar el dataset lleva tiempo: usá fixtures con scope="module" para no repetirlos, y que el archivo completo (sin el test de aceptación) tarde a lo sumo un par de minutos.
- El test de la sección 7.8 lleva @pytest.mark.aceptacion y se saltea con pytest.skip si no están los datos reales; todos los demás corren con pytest a secas.
- Un test por criterio, con nombres descriptivos en español y mensajes de error que digan qué falló (valor medido y esperado). Usá parametrize para frecuencias, formatos y motivos.
- Si algo del contrato es ambiguo o contradictorio, preguntame antes de escribir el archivo.

=== CONTRATO ===
[contenido de docs/contratos/v1_dataset.md]
```
