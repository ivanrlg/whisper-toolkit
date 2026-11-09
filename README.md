# Transcripción de reuniones largas

## Whisper sencillo, sin API de pago — recomendado para este uso

[![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ivanrlg/whisper-toolkit/blob/simplify-long-audio/Whisper_Simple_Gratis.ipynb)

**[Whisper_Simple_Gratis.ipynb](Whisper_Simple_Gratis.ipynb)** mejora el flujo original:
una sola celda para instalar, subir el audio, transcribir y descargar TXT/JSON/SRT/VTT.
Usa faster-whisper `large-v3-turbo` dentro de Colab. **Transcribir no requiere ninguna clave.**

1. Pulsa **Abrir en Colab** arriba. También puedes descargar y subir el notebook manualmente.
2. Activa una **T4 GPU** en el menú de entorno de ejecución si tu cuenta gratuita tiene una
   disponible. No requiere Colab Pro. En CPU también funciona, pero será más lento.
3. Pulsa ▶, conecta Drive para conservar avances y selecciona el audio. Deja
   `SEPARAR_HABLANTES = False` si solo necesitas el texto.

La diarización es opcional y tampoco usa una API de pago. Para activarla, acepta las condiciones
de [pyannote Community-1](https://huggingface.co/pyannote/speaker-diarization-community-1),
crea un [token de lectura de Hugging Face](https://huggingface.co/settings/tokens) y guárdalo
como `HF_TOKEN` en Secretos de Colab. También reconoce `HUGGINGFACE_TOKEN` y
`HUGGINGFACE_HUB_TOKEN`. El token permite descargar el modelo; el audio se procesa en Colab.

Esta versión elimina la transcripción duplicada de WhisperX y ejecuta las etapas en procesos
separados para liberar la memoria GPU. Usa un entorno Python 3.12 independiente, versiones
fijadas y archivos de dependencias resueltos para Linux; no hay que reiniciar el kernel.
La preparación inicial y la descarga de modelos requieren internet y pueden tardar varios minutos.

Los avances se guardan cada cinco minutos de audio en **Mi unidad → Transcripciones_Whisper**.
Reejecutar con el mismo audio/configuración retoma los bloques terminados. Dos segundos de
contexto a cada lado de los bloques ayudan a reducir cortes; la selección de palabras se hace
por su punto medio. Las uniones aún pueden contener errores de reconocimiento y deben revisarse.
La diarización procesa la grabación completa para mantener las etiquetas de hablante globales.
Si falla, se conserva la transcripción sin hablantes. Los resultados parciales están señalados
en TXT/JSON. Al desactivar Drive, la recuperación solo dura lo que sobreviva esa sesión.

**Validación:** 14 pruebas locales del motor y el notebook, y transcripción real corta en CPU
con `tiny.en`, procesada en tres bloques y recuperada sin repetir inferencia. No se ha probado
la GPU de Colab ni la diarización con token autorizado, ni medido calidad/tiempo en una reunión
real de 60–90 minutos. La cuenta gratuita de Colab tiene [recursos variables y no garantizados](https://research.google.com/colaboratory/faq.html).

Para regenerar y verificar esta versión:

```sh
python build_free_notebook.py
python -m unittest discover -s tests -p test_whisper_gratis.py -v
```

`free-asr.lock` y `free-diarization.lock` fijan también dependencias transitivas. CTranslate2 4.6.0
necesita `pkg_resources`, por eso se fija `setuptools==80.9.0` (las versiones 81+ lo eliminan).
El notebook incorpora los locks y el motor completo; no descarga código de una rama del repo.

### Sobre el notebook anterior

La versión original publicada tiene los campos de claves vacíos. Su diarización necesita
un **token de Hugging Face**, no una clave de Whisper. Puedes revisar los Secretos de tu
copia de Colab o gestionar el token en [Hugging Face](https://huggingface.co/settings/tokens).
No guardes credenciales dentro del notebook ni en Git.

La revisión encontró trabajo duplicado (faster-whisper y después WhisperX), varios modelos
residentes a la vez, instalaciones sin versiones fijas y una exportación que usa los segmentos
originales en lugar de los segmentos diarizados. Esas son causas y riesgos identificables en
el código; no se dispone de tus logs para atribuirles cada fallo observado.

Se conserva abajo la documentación histórica y el notebook original como referencia.

---

# Modern Whisper Transcription Pipeline (versión original)

Pipeline moderno de transcripción de audio/video, pensado para reuniones largas. Diseñado para ejecutarse en **Google Colab con GPU T4** (gratis), aunque también funciona en local si tienes GPU NVIDIA.

## Qué incluye

- **`faster-whisper`** — Motor principal (2-4x más rápido que el Whisper original, misma precisión)
- **`WhisperX`** — Timestamps por palabra y diarización de hablantes (quién habló y cuándo)
- **Exportadores** — `.txt`, `.json`, `.srt`, `.vtt`
- **Post-procesado opcional con Groq** — limpieza de texto, resúmenes, actas de reunión
- **Alternativa sin GPU** — transcripción directa con la API de Groq (hasta 25 MB, ~216x en tiempo real)

## Cómo usar (Colab)

1. Abre `Whisper_Modern_Pipeline.ipynb` en [Google Colab](https://colab.research.google.com/)
2. Activa la GPU: `Entorno de ejecución → Cambiar tipo de entorno de ejecución → T4 GPU`
3. Ejecuta las celdas en orden:
   - **Sección 1** — Instala dependencias
   - **Sección 2** — Ajusta modelo, idioma, diarización
   - **Sección 3** — Sube tu audio/video
   - **Sección 4** — Transcribe
   - **Sección 5** — *(opcional)* Diarización con WhisperX
   - **Sección 6** — Preview con timestamps
   - **Sección 7** — Exporta a `.txt`, `.json`, `.srt`, `.vtt`
   - **Sección 8-9** — *(opcional)* Post-procesado o transcripción directa con Groq

## Configuración rápida

Edita la celda de configuración (Sección 2):

```python
MODEL_SIZE = "large-v3-turbo"   # Mejor balance velocidad/precisión
LANGUAGE = None                  # None = autodetectar
ENABLE_DIARIZATION = False       # True para identificar hablantes
```

## Modelos disponibles

| Modelo | VRAM | Velocidad | Calidad |
|---|---|---|---|
| `tiny` | ~1 GB | Extrema | Baja |
| `base` | ~1 GB | Muy alta | Media-baja |
| `small` | ~2 GB | Alta | Media |
| `medium` | ~5 GB | Media | Alta |
| `large-v3` | ~10 GB | Lenta | Muy alta |
| `large-v3-turbo` | ~6 GB | Alta | Muy alta |
| `distil-large-v3` | ~6 GB | Muy alta | Alta |

En Colab T4 (16 GB VRAM), `large-v3-turbo` es la mejor opción por defecto.

## Diarización (opcional)

Para identificar hablantes necesitas:

1. Cuenta en [huggingface.co](https://huggingface.co)
2. Aceptar los términos en [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)
3. Pegar tu token en `HUGGINGFACE_TOKEN` (Sección 2)
4. Poner `ENABLE_DIARIZATION = True`

## Post-procesado con Groq (opcional)

Tres tareas disponibles:
- `clean` — limpia puntuación, quita muletillas
- `summary` — resumen en bullets
- `minutes` — acta de reunión estructurada

Obtén una key gratis en [console.groq.com](https://console.groq.com) y pégala en `GROQ_API_KEY` (Sección 8).

## Recursos

- [faster-whisper](https://github.com/SYSTRAN/faster-whisper)
- [WhisperX](https://github.com/m-bain/whisperX)
- [Groq Console](https://console.groq.com)
- [pyannote models](https://huggingface.co/pyannote)

## Licencia

MIT
