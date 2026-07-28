Caras de BMO (animadas por frames)
==================================

Cada expresion es una CARPETA con uno o varios frames numerados. Si la carpeta
tiene un solo frame, la cara queda fija; si tiene varios, la web los cicla y
BMO se anima (la boca al hablar, el parpadeo al pensar).

Estructura:

  faces/
    <expresion>/
      01.png
      02.png
      ...

Los frames se llaman con numeros correlativos: 01.png, 02.png, 03.png...
Se aceptan estos formatos: png, gif, webp, jpg, jpeg.

Expresiones que usa BMO (el nombre de la carpeta = valor del enum Expression):

  neutral/    -> cara en reposo (esperando)
  listening/  -> te esta escuchando
  thinking/   -> esta pensando
  talking/    -> te esta respondiendo
  happy/      -> contento
  sad/        -> algo salio mal / se colgo
  capturing/  -> mirando por la camara (tool look)
  warmup/     -> arrancando / calentando motores

Las caras actuales vienen del repo brenpoly/be-more-agent (carpeta faces).
Mapeo aplicado: idle->neutral y happy, speaking->talking, listen->listening,
thinking->thinking, error->sad, capturing->capturing, warmup->warmup.

Si falta la carpeta de una expresion (o esta vacia), la web muestra el NOMBRE
en grande como fallback, asi podes probar sin imagenes todavia.

Para cambiar los fps de la animacion, edita _DEFAULT_FPS en face_web.py.
