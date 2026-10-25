# Informe final (Fase 6)

- `informe.md`: el texto del informe, con las secciones de la consigna en su orden. Es la fuente: los números salen del Documento 2 v0.7.5, de los contratos y de lo que imprimen los módulos. Lo que está entre corchetes lo completa Santiago.
- `defensa.md`: lista de la entrega (sección 2 del Plan) con su evidencia, guion de los ensayos de la demo y preguntas probables del coloquio.
- `armar.py`: arma `salida/informe.docx` y, si está LibreOffice, `salida/informe.pdf` (la carpeta `salida/` no va a Git).
- `referencia.docx`: la plantilla de estilos que usa pandoc (A4, márgenes de 2,5 cm, Calibri, tablas con bordes). La arma `estilo_referencia.py`; solo hace falta volver a correrlo para cambiar el estilo.
- Las figuras son las de `docs/figuras/`. Los dos diagramas nuevos (`maquina_estados` y `agente_arquitectura`) se dibujan con Graphviz desde su `.dot` (el comando está al principio de cada archivo).

## Cómo se arma

Hace falta pandoc 3.1 o más nuevo y `lxml` (viene con `python-docx`); para el PDF, LibreOffice con su módulo de fórmulas. En la nube de Claude están los dos (`apt-get install libreoffice-math` si faltan las fórmulas). Desde la raíz del repo:

```text
python docs/informe/armar.py
```

En la notebook, si no están pandoc ni LibreOffice: abrir `informe.docx` (el que mandó Claude o uno nuevo armado en la nube), editarlo en Word y exportar el PDF desde Word, que dibuja bien las fórmulas. `armar.py` retoca el Word para LibreOffice (paréntesis comunes en las fórmulas y la doble barra de la norma); en Word se ve igual.

## Qué falta para la versión final

- Los corchetes: apellido y legajo, fecha de la presentación, edición de Russell y Norvig y la fila de "Demo sin cierres inesperados" de la tabla 9 (después de los 3 ensayos).
- Leerlo entero para sacar lo que suene a relleno y pasarlo a palabras propias donde haga falta (pedido de la cátedra).
- Revisar las dos referencias que se sumaron respecto del Documento 2: Zuiderveld (1994), por CLAHE, y Rousseeuw (1987), por la silhouette; y las de software (NumPy, SciPy, Matplotlib, scikit-image).
- Si se vuelve a medir algo (por ejemplo, una V4.1), cambiar los números en el resumen, en las secciones 6.6 y 6.7 y en las conclusiones.
