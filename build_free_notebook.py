"""Genera la entrega autónoma para Colab sin servicios de pago."""
import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
INTRO = """# Whisper sencillo y gratis

[![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ivanrlg/whisper-toolkit/blob/simplify-long-audio/Whisper_Simple_Gratis.ipynb)

**Tu transcripción de siempre con menos pasos, sin contratar una API ni introducir datos de pago.**
Para audios de una hora o una hora y media. Ejecuta la única celda: instala lo necesario,
te pide el audio, transcribe y descarga un ZIP con **TXT, JSON, SRT y VTT**.

## Cómo usarlo

1. En Colab, ve a **Entorno de ejecución → Cambiar tipo de entorno de ejecución → T4 GPU**,
   si está disponible en tu cuenta gratuita. No necesitas Colab Pro. También puede usar CPU,
   pero con audios largos será más lento.
2. Deja `SEPARAR_HABLANTES` desactivado al empezar: **transcribir no necesita ninguna clave**.
3. Pulsa **▶**, autoriza Drive para guardar avances y selecciona tu audio.
   Para un archivo que ya está en Drive, puedes poner su ruta en `RUTA_AUDIO`, por ejemplo
   `/content/drive/MyDrive/Audios/reunion.m4a`.

`IDIOMA = es` está preparado para español. Usa `auto` si no sabes el idioma, o `en` para inglés.
Para grabaciones que alternan idiomas, revisa especialmente el resultado: se usa un idioma principal.

## Si quieres separar los hablantes — también sin API de pago

Esta función es opcional. Usa **pyannote Community-1**, ejecutado en el entorno de Colab.

1. Inicia sesión en Hugging Face y acepta las condiciones de
   [pyannote/speaker-diarization-community-1](https://huggingface.co/pyannote/speaker-diarization-community-1).
2. Crea un [token de lectura gratuito](https://huggingface.co/settings/tokens).
   Si usas permisos detallados, permite leer los modelos restringidos a los que tienes acceso.
   No necesitas una cuenta de pago ni una clave de pyannoteAI.
3. En el icono de llave de Colab (**Secretos**), guarda el token como `HF_TOKEN` y permite
   su acceso a este notebook. También reconoce secretos anteriores llamados `HUGGINGFACE_TOKEN`
   y `HUGGINGFACE_HUB_TOKEN`.
4. Activa `SEPARAR_HABLANTES` y vuelve a ejecutar con el mismo audio.
   **La transcripción ya guardada se reutiliza.**

No se escribe el token dentro del notebook ni de los resultados. Si falta o la diarización falla,
recibirás la transcripción sin hablantes con un mensaje explicando lo que falta.

## Qué se ha simplificado

- **Una sola transcripción** con faster-whisper `large-v3-turbo`. Sin segunda pasada de WhisperX,
  sin modelo de alineación adicional y sin las secciones de Groq.
- Instalación aislada del kernel, Python 3.12 y versiones fijadas: no hace falta reiniciar Colab
  tras instalar. La primera preparación y la descarga de modelos pueden tardar varios minutos.
- Se guarda cada bloque de aproximadamente cinco minutos. Si algo se interrumpe, vuelve a ejecutar
  con el mismo archivo y se recuperan los bloques terminados. El bloque interrumpido se repite.
- Cada bloque tiene dos segundos de contexto adicional; se conservan las palabras por su punto medio
  para reducir cortes y repeticiones en las uniones. Los timestamps siguen referidos al audio original.
- La diarización usa **la grabación completa** para mantener las etiquetas de hablante a lo largo
  de toda la reunión. Transcripción y diarización corren en procesos separados para liberar la memoria.
- Los hablantes se escriben en TXT, JSON, SRT y VTT. Si la diarización falla, no se pierde el texto.

## Dónde queda guardado

Con `GUARDAR_EN_DRIVE` activado: **Mi unidad → Transcripciones_Whisper**.
Cada audio/configuración tiene una carpeta de recuperación. Conserva esa carpeta y usa la misma
configuración para retomar. Si desactivas Drive, los avances desaparecen cuando Colab elimina la sesión.
Los resultados parciales se identifican como tales en TXT y JSON; verifica `complete` en JSON.

Los audios se procesan en Colab, sin enviarlos a una API de transcripción de pago.
Se requiere internet para instalar paquetes y descargar los modelos. Colab gratuito tiene
límites variables y no garantiza disponibilidad de GPU o duración de sesión.

Los tiempos de palabras y las voces detectadas pueden contener errores, sobre todo con ruido,
voces simultáneas y en las uniones de bloques. `SPEAKER_00`, etc. distingue voces, no nombres reales.

**Validación:** pruebas locales de recuperación y exportaciones, más una transcripción real corta
en CPU con un modelo pequeño. Falta probar GPU de Colab, diarización con un token autorizado y
un audio real de 60–90 minutos; no se garantiza un tiempo de ejecución.

Fuentes: [faster-whisper](https://github.com/SYSTRAN/faster-whisper),
[Community-1](https://huggingface.co/pyannote/speaker-diarization-community-1),
[límites de Colab](https://research.google.com/colaboratory/faq.html).
"""


def build():
    launcher = (ROOT / "colab_gratis.py").read_text(encoding="utf-8")
    controls, body = launcher.split("\n\ndef install_environment", 1)
    embedded = "\n\nimport hashlib\nimport os\nfrom pathlib import Path\n"
    for variable, filename in (("WORKER_SOURCE", "whisper_gratis.py"),
                               ("ASR_LOCK", "free-asr.lock"),
                               ("DIARIZATION_LOCK", "free-diarization.lock")):
        embedded += f"{variable} = {repr((ROOT / filename).read_text(encoding='utf-8'))}\n"
    code = controls + embedded + "\n\ndef install_environment" + body
    ast.parse(code)
    notebook = {
        "nbformat": 4, "nbformat_minor": 5,
        "metadata": {"colab": {"provenance": [], "name": "Whisper_Simple_Gratis.ipynb"},
                     "accelerator": "GPU",
                     "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                     "language_info": {"name": "python"}},
        "cells": [
            {"cell_type": "markdown", "id": "guia-gratis", "metadata": {}, "source": INTRO.splitlines(True)},
            {"cell_type": "code", "id": "transcribir-gratis", "metadata": {"cellView": "form"},
             "execution_count": None, "outputs": [], "source": code.splitlines(True)},
        ],
    }
    target = ROOT / "Whisper_Simple_Gratis.ipynb"
    target.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(f"Generado: {target.name}")


if __name__ == "__main__":
    build()
