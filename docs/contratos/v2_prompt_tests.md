# Prompt para escribir tests/test_v2.py

Se usa en un chat nuevo, sin el contexto del desarrollo, con `v2_preprocesamiento.md` adjunto.

- **Fecha:** <se completa al usarlo>
- **Herramienta y modelo:** <se completa al usarlo>

```text
Actuá como un ingeniero de testing en Python. Voy a implementar la etapa V2 (preprocesamiento de voz) del trabajo final de IA I (Python 3.12, Windows, VS Code). El contrato completo es el archivo adjunto v2_preprocesamiento.md: firmas exactas, tipos y criterios de aceptación. Escribí un único archivo, tests/test_v2.py, con pytest, que verifique cada punto de la sección 7 del contrato (la 8 es manual: no la automatices).

Reglas:
- Leé solo v2_preprocesamiento.md. No abras ningún otro archivo de la carpeta IA 1 (hay documentos del desarrollo y datos privados): los tests salen solo del contrato.
- Cuando termines, guardá el archivo en IA 1\test_v2.py.
- No implementes nada de src/ ni propongas cómo hacerlo: solo los tests.
- Raíz del repo: RAIZ = Path(__file__).resolve().parents[1]. La configuración real se carga con src.config.cargar_config().
- De V1 podés usar src.voz.audio.remuestrear y src.voz.segmentacion.envolvente_db y detectar_palabras, que ya existen y tienen sus propios tests; no los vuelvas a probar.
- Importá src.config, src.voz.preprocesamiento y src.voz.aumento dentro de cada test o de un fixture, no arriba del archivo: si algo todavía no existe, que fallen esos tests y no la colección entera.
- Usá solo la biblioteca estándar, pytest, numpy, scipy, soundfile y PyYAML.
- Los audios de prueba se sintetizan en el test (tonos con armónicos y rampas, ruido filtrado, silencios) a 16 kHz, y el dataset de V1 de prueba se arma en tmp_path con un metadatos.csv de las 7 columnas de la sección 1 y WAV escritos con soundfile.
- Para la configuración reducida, partí de una copia profunda del cfg real y cambiá solo lo necesario.
- Lo que escriba archivos va en tmp_path o tmp_path_factory: nunca escribas en data/ ni en docs/ del repo.
- Usá fixtures con scope="module" para no repetir lo que tarda, y que el archivo completo (sin el test de aceptación) tarde a lo sumo un par de minutos.
- El test de la sección 7.6 lleva @pytest.mark.aceptacion y se saltea con pytest.skip si no están los datos reales; todos los demás corren con pytest a secas.
- Un test por criterio, con nombres descriptivos en español y mensajes de error que digan qué falló (valor medido y esperado). Usá parametrize para frecuencias, factores y motivos.
- Si algo del contrato es ambiguo o contradictorio, preguntame antes de escribir el archivo.
```
