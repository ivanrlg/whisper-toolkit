# @title Transcribir con Whisper — sin API de pago
# @markdown Activa **Entorno de ejecución → Cambiar tipo → T4 GPU**, si está disponible.
# @markdown Deja la ruta vacía para subir el archivo desde tu computadora.
RUTA_AUDIO = "" # @param {type:"string"}
IDIOMA = "es" # @param ["es", "auto", "en"]
SEPARAR_HABLANTES = False # @param {type:"boolean"}
GUARDAR_EN_DRIVE = True # @param {type:"boolean"}


def install_environment(work, lock_text, stage):
    """Entorno separado del kernel de Colab: no hay que reiniciarlo."""
    import subprocess
    import sys
    work = Path(work)
    bootstrap = work / "herramientas"
    bootstrap_env = os.environ.copy()
    bootstrap_env["PYTHONPATH"] = str(bootstrap)
    if not (bootstrap / "uv").is_dir():
        subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", "--target",
                        str(bootstrap), "uv==0.13.7"], check=True)
    uv = [sys.executable, "-m", "uv"]
    python = work / "entorno" / "bin" / "python"
    if not python.exists():
        subprocess.run(uv + ["venv", "--python", "3.12", str(work / "entorno")],
                       env=bootstrap_env, check=True)
    requirements = work / f"{stage}.lock"
    requirements.write_text(lock_text, encoding="utf-8")
    marker = work / f"{stage}.instalado"
    fingerprint = hashlib.sha256(lock_text.encode()).hexdigest()
    if not marker.exists() or marker.read_text() != fingerprint:
        print(f"Preparando {stage}; la primera vez puede tardar varios minutos…", flush=True)
        subprocess.run(uv + ["pip", "install", "--python", str(python), "-r", str(requirements)],
                       env=bootstrap_env, check=True)
        marker.write_text(fingerprint)
    return python


def worker_environment(python, token=""):
    import subprocess
    env = os.environ.copy()
    # CTranslate2 debe encontrar CUDA antes de arrancar el proceso.
    library_code = (
        "import site,pathlib; print(':'.join(str(p) for s in site.getsitepackages() "
        "for p in pathlib.Path(s).glob('nvidia/*/lib')))"
    )
    libraries = subprocess.check_output([str(python), "-c", library_code], text=True).strip()
    env["LD_LIBRARY_PATH"] = libraries + ":" + env.get("LD_LIBRARY_PATH", "")
    env["HF_HUB_DISABLE_TELEMETRY"] = "1"
    env["PYANNOTE_METRICS_ENABLED"] = "0"
    if token:
        env["HF_TOKEN"] = token
    return env


def run_free_colab():
    import subprocess
    import shutil
    import types
    from google.colab import drive, files, userdata

    print("Whisper local en Colab: no hay API de pago ni coste por minuto.", flush=True)
    if GUARDAR_EN_DRIVE:
        drive.mount("/content/drive")
        output_root = Path("/content/drive/MyDrive/Transcripciones_Whisper")
    else:
        output_root = Path("/content/Transcripciones_Whisper")
        print("Sin Drive, los avances se pierden cuando Colab elimina la sesión.", flush=True)
    source = RUTA_AUDIO.strip()
    if not source:
        uploaded = files.upload()
        if len(uploaded) != 1:
            raise ValueError("Selecciona un solo archivo de audio o video.")
        source = next(iter(uploaded))
        del uploaded
    audio = Path(source).expanduser().resolve()

    # El código está dentro de este notebook; no se descarga desde una rama mutable.
    work = Path("/content/whisper_gratis_runtime")
    work.mkdir(parents=True, exist_ok=True)
    worker = work / "whisper_gratis.py"
    worker.write_text(WORKER_SOURCE, encoding="utf-8")
    engine = types.ModuleType("whisper_gratis_notebook")
    exec(compile(WORKER_SOURCE, str(worker), "exec"), engine.__dict__)
    folder = engine.prepare_job(audio, output_root, language=None if IDIOMA == "auto" else IDIOMA)
    print(f"Avances y resultados: {folder}", flush=True)

    gpu = bool(shutil.which("nvidia-smi")) and subprocess.run(
        ["nvidia-smi", "-L"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    ).returncode == 0
    lock = ASR_LOCK
    if gpu:
        lock += "\nnvidia-cublas-cu12==12.8.4.1\nnvidia-cudnn-cu12==9.10.2.21\n"
    python = install_environment(work, lock, "transcripcion")
    command = [str(python), "-u", str(worker)]
    common = ["--audio", str(audio), "--folder", str(folder), "--device", "cuda" if gpu else "cpu"]
    # Cada etapa usa un proceso distinto y libera toda su memoria al terminar.
    completed = subprocess.run(command + ["transcribe"] + common, env=worker_environment(python))
    archive = folder / "transcripcion_whisper.zip"
    if completed.returncode:
        print("La transcripción se interrumpió. Reejecuta con el mismo audio para retomar los bloques guardados.")
        if archive.exists():
            print("Puedes descargar lo que se alcanzó a transcribir; está marcado como parcial.")
            files.download(str(archive))
        return

    if SEPARAR_HABLANTES:
        token = os.environ.get("HF_TOKEN", "").strip()
        if not token:
            for name in ("HF_TOKEN", "HUGGINGFACE_TOKEN", "HUGGINGFACE_HUB_TOKEN"):
                try:
                    token = (userdata.get(name) or "").strip()
                except (userdata.SecretNotFoundError, userdata.NotebookAccessError):
                    token = ""
                if token:
                    break
        # Hablantes ya guardados no requieren volver a usar la credencial.
        if not token and not (folder / "hablantes.json").exists():
            print("El texto está completo. Falta el token gratuito para separar hablantes.")
            print("Crea un secreto HF_TOKEN y acepta las condiciones del modelo Community-1.")
            print("https://huggingface.co/settings/tokens")
            print("https://huggingface.co/pyannote/speaker-diarization-community-1")
        else:
            try:
                if not (folder / "hablantes.json").exists():
                    python = install_environment(work, DIARIZATION_LOCK, "diarizacion")
                diarized = subprocess.run(command + ["diarize"] + common,
                                          env=worker_environment(python, token))
                if diarized.returncode:
                    print("La diarización falló; se conserva y descarga la transcripción sin hablantes.")
                    print("Revisa HF_TOKEN y los permisos de Community-1. Reejecutar no repite la transcripción.")
            except subprocess.CalledProcessError:
                print("No se pudo preparar la diarización. La transcripción está completa y conservada.")
            finally:
                token = ""
    print(f"Archivos listos en {folder}")
    files.download(str(archive))


try:
    run_free_colab()
except Exception as error:
    print(f"No se completó: {type(error).__name__}: {error}")
    print("Revisa el mensaje anterior. Los bloques ya guardados se recuperan al reejecutar con el mismo audio.")
