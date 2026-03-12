# Ideas de Marketing y Redes Sociales - Escriba

## Estrategia General: "Build in Public"
No hacer tutoriales genéricos. Mostrar problemas reales, soluciones técnicas y herramientas propias.

---

## 🧵 Hilo de Twitter: Lanzamiento / Showcasing

**Objetivo:** Mostrar la herramienta, enfatizar privacidad y ahorro de dinero (vs Otter.ai).

### Tweet 1 (El Gancho + Video)
Cancelé mi suscripción a Otter.ai y construí mi propia solución local. 🐻🚫💸

Les presento `escriba`:
- 🎙️ Transcripción en tiempo real en macOS.
- 🔒 100% Privado (todo corre en tu máquina).
- 🎧 Combina audio del sistema (Zoom/Meet) + Micrófono.
- ⚡️ Gratis (gracias a Whisper).

Aquí el código 👇 
github.com/skalas/escriba

*(Adjuntar video de 30s mostrando la terminal transcribiendo un video de YouTube o una call)*

### Tweet 2 (El "Cómo" Técnico)
El reto principal: macOS no te deja grabar audio del sistema fácil. 🍎😤

Tuve que usar `ScreenCaptureKit` (nativo de Apple) con un wrapper en Swift, y mandarlo a un proceso de Python que corre `faster-whisper`.

Nada de drivers virtuales raros como BlackHole. Todo nativo y limpio.

### Tweet 3 (Features "Cool")
Lo mejor es que detecta cuando entro a una call y empieza a grabar solo. 🤖
Al final, tengo un TXT/Markdown con todo lo que se dijo.

Próximo paso: Meterle diarización (para saber quién habla) y resúmenes automáticos con LLMs locales.

### Tweet 4 (Call to Action)
Si les interesa la privacidad y tener sus propias herramientas, denle una estrella ⭐ en GitHub.
Acepto PRs (me faltan tests, no me juzguen 😅).

github.com/skalas/escriba

---

## 📹 Ideas para Video (YouTube / TikTok)

**Título:** "Cómo hackear el audio de Mac con Swift y Python"
**Contenido:**
1. El problema: Grabar audio de sistema en Mac es difícil.
2. La solución: ScreenCaptureKit (Swift) pipeado a Python.
3. Demo: Corriendo el script.
4. Código: Breve tour por la arquitectura (Swift CLI -> Stdout -> Python Popen).

---

## 📝 Siguientes Pasos
1. **Grabar la demo:** 30 segundos, sin editar mucho. Que se vea real.
2. **Publicar el hilo:** Martes/Miércoles a las 9-10 AM suele ser buena hora.
