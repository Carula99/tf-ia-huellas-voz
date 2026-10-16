# Prompt usado para escribir tests/test_setup.py

Se pegó en un chat nuevo, sin el contexto del desarrollo, seguido del contenido completo de `fase0_setup.md`.

- **Fecha:** 16/10/2026
- **Herramienta y modelo:** Claude Opus 5.5 y Gemini Argon

```text
Actuá como un ingeniero de testing en Python. Voy a implementar la Fase 0 (setup) del trabajo final de Inteligencia Artificial I: un sistema de reconocimiento de huellas dactilares controlado por voz, en Python 3.12 sobre Windows 11 y VS Code. Abajo va el contrato completo. Escribí un único archivo, tests/test_setup.py, con pytest, que verifique cada punto de las secciones 1 a 8 del contrato (la 9 es manual: no la automatices).

Reglas:
- No implementes nada de src/ ni propongas cómo hacerlo: solo los tests.
- Raíz del repo: RAIZ = Path(__file__).resolve().parents[1]. Que nada dependa del directorio de trabajo.
- Usá solo la biblioteca estándar, pytest y PyYAML (yaml.safe_load).
- Importá las librerías del entorno y src.config dentro de cada test (con importlib.import_module o un fixture), no arriba del archivo: si algo todavía no existe, que fallen esos tests y no la colección entera.
- Abrí todos los archivos de texto con encoding="utf-8" (en Windows el default no es UTF-8).
- Lo que escriba archivos va en tmp_path. Nunca modifiques el config.yaml real ni otros archivos del repo.
- Para las claves obligatorias, escribí dentro del test la lista de caminos con punto que sale del esqueleto de la sección 7 y probá con parametrize que sacar cada una (de una copia del config real guardada en tmp_path) haga que cargar_config lance ValueError con ese camino en el mensaje.
- Un test por criterio, con nombres descriptivos en español y mensajes de error que digan qué falló. Usá parametrize para las listas (imports, carpetas, rutas del .gitignore, claves).
- Ningún test con @pytest.mark.aceptacion: todos corren con pytest a secas.
- Si algo del contrato es ambiguo o contradictorio, preguntame antes de escribir el archivo.

=== CONTRATO ===
[contenido de docs/contratos/fase0_setup.md]
```
