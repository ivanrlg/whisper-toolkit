# Modern Whisper Transcription Pipeline

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
