"""Whisper en Colab, sin API de pago. Motor autocontenido del notebook."""
import argparse
from bisect import bisect_right
import hashlib
import html
import json
import math
import os
from pathlib import Path
import zipfile

SAMPLE_RATE = 16000
DIARIZATION_MODEL = "pyannote/speaker-diarization-community-1"
ALGORITHM_VERSION = 1


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def prepare_job(audio_path, output_root, model="large-v3-turbo", language="es", chunk_seconds=300):
    path = Path(audio_path)
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError("Selecciona un archivo de audio o video que no esté vacío.")
    config = {"model": model, "language": language, "chunk_seconds": chunk_seconds,
              "algorithm_version": ALGORITHM_VERSION, "faster_whisper_version": "1.2.1"}
    digest = hashlib.sha256(json.dumps(config, sort_keys=True).encode())
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    folder = Path(output_root) / digest.hexdigest()[:24]
    folder.mkdir(parents=True, exist_ok=True)
    save_json(folder / "config.json", config)
    return folder


def chunk_plan(duration, size=300, context=2):
    if duration <= 0 or size <= 0 or context < 0:
        raise ValueError("Duración o tamaño de bloque no válido.")
    return [dict(index=i, start=i * size, end=min(duration, (i + 1) * size),
                 read_start=max(0, i * size - context),
                 read_end=min(duration, (i + 1) * size + context))
            for i in range(math.ceil(duration / size))]


def keep_core_segments(segments, chunk):
    """Conserva cada palabra en el bloque al que pertenece su punto medio."""
    kept = []
    offset = chunk["read_start"]
    for segment in segments:
        words = []
        for word in segment.get("words") or []:
            start, end = word["start"] + offset, word["end"] + offset
            midpoint = (start + end) / 2
            if chunk["start"] <= midpoint < chunk["end"]:
                words.append({**word, "start": max(0, start), "end": end})
        if words:
            kept.append({"start": words[0]["start"], "end": words[-1]["end"],
                         "text": "".join(w["word"] for w in words).strip(), "words": words})
        elif not segment.get("words"):
            start, end = segment["start"] + offset, segment["end"] + offset
            if chunk["start"] <= (start + end) / 2 < chunk["end"]:
                kept.append({"start": start, "end": end, "text": segment["text"].strip(), "words": []})
    return kept


def timestamp(seconds, separator="."):
    total = max(0, round(seconds * 1000))
    seconds, millis = divmod(total, 1000)
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}{separator}{millis:03d}"


def subtitle_cues(segments):
    for segment in segments:
        words = segment.get("words") or []
        if not words:
            yield segment
            continue
        group = []
        for word in words:
            if group and (len(group) >= 12 or word["end"] - group[0]["start"] > 7):
                yield {**segment, "start": group[0]["start"], "end": group[-1]["end"],
                       "text": "".join(w["word"] for w in group).strip()}
                group = []
            group.append(word)
        if group:
            yield {**segment, "start": group[0]["start"], "end": group[-1]["end"],
                   "text": "".join(w["word"] for w in group).strip()}


def labelled_text(segment):
    prefix = f"[{segment['speaker']}] " if "speaker" in segment else ""
    return prefix + segment["text"].strip()


def write_outputs(result, folder):
    folder = Path(folder)
    segments = result["segments"]
    partial = "TRANSCRIPCIÓN PARCIAL — vuelve a ejecutar para completarla.\n\n" if not result["complete"] else ""
    text = "\n\n".join(f"[{timestamp(s['start'])}] {labelled_text(s)}" for s in segments)
    if not segments:
        text = "No se detectó voz en los bloques procesados."
    (folder / "transcripcion.txt").write_text(partial + text + "\n", encoding="utf-8")
    save_json(folder / "transcripcion.json", {**result, "timestamp_unit": "seconds"})
    cues = list(subtitle_cues(segments))
    for extension, separator in (("srt", ","), ("vtt", ".")):
        blocks = []
        for index, cue in enumerate(cues, 1):
            start, end = cue["start"], max(cue["end"], cue["start"] + 0.001)
            blocks.append(f"{index}\n{timestamp(start, separator)} --> {timestamp(end, separator)}\n"
                          + html.escape(labelled_text(cue)))
        prefix = "WEBVTT\n\n" if extension == "vtt" else ""
        (folder / f"subtitulos.{extension}").write_text(prefix + "\n\n".join(blocks) + "\n", encoding="utf-8")
    archive = folder / "transcripcion_whisper.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as bundle:
        for name in ("transcripcion.txt", "transcripcion.json", "subtitulos.srt", "subtitulos.vtt"):
            bundle.write(folder / name, arcname=name)
    return archive


def transcribe(audio_path, folder, device="auto", model_factory=None, decoder=None):
    folder = Path(folder)
    config = read_json(folder / "config.json")
    base_path = folder / "transcripcion_base.json"
    if base_path.exists():
        previous = read_json(base_path)
        if previous.get("complete"):
            print("Transcripción ya guardada: no se repetirá.", flush=True)
            write_outputs(previous, folder)
            return previous
    if decoder is None:
        from faster_whisper.audio import decode_audio
        decoder = decode_audio
    audio = decoder(str(audio_path), sampling_rate=SAMPLE_RATE)
    duration = len(audio) / SAMPLE_RATE
    plan = chunk_plan(duration, config["chunk_seconds"])
    if model_factory is None:
        from faster_whisper import WhisperModel
        model_factory = WhisperModel
    if device == "auto":
        import ctranslate2
        device = "cuda" if ctranslate2.get_cuda_device_count() else "cpu"
    print(f"Motor: {config['model']} · {device} · audio de {duration / 60:.1f} minutos.", flush=True)
    if device == "cpu":
        print("Sin GPU: la transcripción puede tardar bastante. T4 gratuita es recomendable si está disponible.", flush=True)
    model = None
    detected_language = config["language"]
    completed_segments = []
    result = None
    for chunk in plan:
        cache_path = folder / f"bloque_{chunk['index']:04d}.json"
        if cache_path.exists():
            cached = read_json(cache_path)
            if cached.get("chunk") != chunk:
                raise RuntimeError("Un bloque guardado no corresponde al audio. Conserva la carpeta para revisarlo.")
            print(f"Bloque {chunk['index'] + 1}/{len(plan)} recuperado.", flush=True)
        else:
            if model is None:
                model = model_factory(config["model"], device=device,
                                      compute_type="int8_float16" if device == "cuda" else "int8")
            print(f"Transcribiendo bloque {chunk['index'] + 1}/{len(plan)}…", flush=True)
            excerpt = audio[round(chunk["read_start"] * SAMPLE_RATE):round(chunk["read_end"] * SAMPLE_RATE)]
            generator, info = model.transcribe(
                excerpt, language=detected_language, beam_size=5, word_timestamps=True,
                vad_filter=True, vad_parameters={"min_silence_duration_ms": 500},
                condition_on_previous_text=False,
            )
            raw = []
            for segment in generator:
                raw.append({"start": segment.start, "end": segment.end, "text": segment.text,
                            "words": [{"start": w.start, "end": w.end, "word": w.word,
                                       "probability": w.probability} for w in segment.words or []]})
            retained = keep_core_segments(raw, chunk)
            # No fijar un idioma detectado en un bloque sin voz.
            language = detected_language or (info.language if retained else None)
            cached = {"chunk": chunk, "language": language, "segments": retained}
            save_json(cache_path, cached)
        detected_language = detected_language or cached.get("language")
        completed_segments.extend(cached["segments"])
        result = {"model": config["model"], "language": detected_language,
                  "duration": duration, "complete": chunk["index"] == len(plan) - 1,
                  "diarized": False, "segments": completed_segments,
                  "completed_chunks": chunk["index"] + 1, "total_chunks": len(plan)}
        save_json(base_path, result)
        write_outputs(result, folder)
    print("Transcripción completa y archivos guardados.", flush=True)
    return result


def assign_speakers(segments, turns):
    """Asigna por solapamiento temporal usando una diarización global exclusiva."""
    turns = sorted(turns, key=lambda t: (t["start"], t["end"]))
    ends = [t["end"] for t in turns]

    def label(start, end):
        # Exclusive diarization no tiene turnos solapados: ends está ordenado.
        index = bisect_right(ends, start)
        best, chosen = 0.0, "SIN_IDENTIFICAR"
        while index < len(turns) and turns[index]["start"] < max(end, start + 0.001):
            turn = turns[index]
            overlap = max(0, min(max(end, start + 0.001), turn["end"]) - max(start, turn["start"]))
            if overlap > best:
                best, chosen = overlap, turn["speaker"]
            index += 1
        return chosen

    output = []
    for segment in segments:
        words = segment.get("words") or []
        if not words:
            output.append({**segment, "speaker": label(segment["start"], segment["end"])})
            continue
        groups = []
        for word in words:
            speaker = label(word["start"], word["end"])
            if not groups or groups[-1]["speaker"] != speaker:
                groups.append({"speaker": speaker, "words": []})
            groups[-1]["words"].append({**word, "speaker": speaker})
        for group in groups:
            words = group["words"]
            output.append({**group, "start": words[0]["start"], "end": words[-1]["end"],
                           "text": "".join(w["word"] for w in words).strip()})
    return output


def diarize(audio_path, folder, device="auto"):
    folder = Path(folder)
    result = read_json(folder / "transcripcion_base.json")
    if not result["complete"]:
        raise RuntimeError("Completa la transcripción antes de separar hablantes.")
    cached_path = folder / "hablantes.json"
    if cached_path.exists():
        cached = read_json(cached_path)
        if cached.get("model") != DIARIZATION_MODEL:
            raise RuntimeError("El archivo de hablantes guardado usa otro modelo.")
        turns = cached["turns"]
    else:
        token = os.environ.get("HF_TOKEN", "").strip()
        if not token:
            raise RuntimeError("Falta HF_TOKEN para la diarización opcional.")
        import torch
        from faster_whisper.audio import decode_audio
        from pyannote.audio import Pipeline
        from pyannote.audio.pipelines.utils.hook import ProgressHook
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        pipeline = Pipeline.from_pretrained(DIARIZATION_MODEL, token=token)
        if pipeline is None:
            raise RuntimeError("No se pudo descargar el modelo. Revisa el token y los permisos de Hugging Face.")
        pipeline.to(torch.device(device))
        waveform = torch.from_numpy(decode_audio(str(audio_path), sampling_rate=SAMPLE_RATE)).unsqueeze(0)
        print("Separando hablantes sobre la grabación completa…", flush=True)
        with ProgressHook() as hook:
            output = pipeline({"waveform": waveform, "sample_rate": SAMPLE_RATE}, hook=hook)
        turns = [{"start": turn.start, "end": turn.end, "speaker": speaker}
                 for turn, _, speaker in output.exclusive_speaker_diarization.itertracks(yield_label=True)]
        save_json(cached_path, {"model": DIARIZATION_MODEL, "turns": turns})
    labelled = {**result, "diarized": True, "diarization_model": DIARIZATION_MODEL,
                "segments": assign_speakers(result["segments"], turns)}
    write_outputs(labelled, folder)
    print("Diarización completa. Los archivos descargables incluyen los hablantes.", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("transcribe", "diarize"))
    parser.add_argument("--audio", required=True)
    parser.add_argument("--folder", required=True)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    args = parser.parse_args()
    try:
        if args.stage == "transcribe":
            transcribe(args.audio, args.folder, args.device)
        else:
            diarize(args.audio, args.folder, args.device)
    except Exception as error:
        token = os.environ.get("HF_TOKEN", "")
        message = str(error)
        if token:
            message = message.replace(token, "[TOKEN OCULTO]")
        print(f"No se completó {args.stage}: {message[:1500]}", flush=True)
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
