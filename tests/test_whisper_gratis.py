import ast
import json
from pathlib import Path
import tempfile
import types
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zipfile

import whisper_gratis as app


class Audio:
    def __init__(self, length):
        self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, slice_):
        return Audio(slice_.stop - slice_.start)


class FakeModel:
    def __init__(self, fail_on=None):
        self.calls = 0
        self.fail_on = fail_on

    def transcribe(self, audio, **options):
        self.calls += 1
        if self.calls == self.fail_on:
            raise RuntimeError("interrupción simulada")
        word = SimpleNamespace(start=2.1, end=2.5, word=" Hola", probability=0.9)
        segment = SimpleNamespace(start=2.1, end=2.5, text=" Hola", words=[word])
        return iter([segment]), SimpleNamespace(language="es")


class FreeWhisperTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "audio.wav"
        self.source.write_bytes(b"fake recording for mock tests")
        self.folder = app.prepare_job(self.source, self.root / "output", chunk_seconds=300)

    def transcribe(self, model, duration=650):
        return app.transcribe(self.source, self.folder, device="cpu", model_factory=lambda *a, **k: model,
                              decoder=lambda *a, **k: Audio(duration * app.SAMPLE_RATE))

    def test_ninety_minutes_plan_covers_all_audio(self):
        plan = app.chunk_plan(5400)
        self.assertEqual(len(plan), 18)
        self.assertEqual(sum(c["end"] - c["start"] for c in plan), 5400)
        self.assertEqual(plan[0]["read_start"], 0)
        self.assertEqual(plan[-1]["read_end"], 5400)
        self.assertEqual(plan[1]["read_start"], 298)
        self.assertEqual(plan[0]["read_end"], 302)
        self.assertTrue(all(a["end"] == b["start"] for a, b in zip(plan, plan[1:])))

    def test_word_at_boundary_owned_by_exactly_one_chunk(self):
        plan = app.chunk_plan(600)
        source = {"start": 299, "end": 301, "text": " hola mundo", "words": [
            {"start": 299.5, "end": 300, "word": " hola"},
            {"start": 300, "end": 300.5, "word": " mundo"},
        ]}
        kept = []
        for chunk in plan:
            offset = chunk["read_start"]
            local = {**source, "words": [{**w, "start": w["start"] - offset, "end": w["end"] - offset}
                                          for w in source["words"]]}
            kept += app.keep_core_segments([local], chunk)
        self.assertEqual([w["word"] for s in kept for w in s["words"]], [" hola", " mundo"])
        self.assertEqual(kept[-1]["end"], 300.5)

    def test_resume_after_failure_keeps_completed_chunks(self):
        with self.assertRaisesRegex(RuntimeError, "interrupción"):
            self.transcribe(FakeModel(fail_on=2))
        saved = app.read_json(self.folder / "transcripcion_base.json")
        self.assertFalse(saved["complete"])
        self.assertEqual(saved["completed_chunks"], 1)
        self.assertTrue((self.folder / "transcripcion_whisper.zip").exists())
        self.assertIn("PARCIAL", (self.folder / "transcripcion.txt").read_text(encoding="utf-8"))
        model = FakeModel()
        result = self.transcribe(model)
        self.assertTrue(result["complete"])
        self.assertEqual(model.calls, 2)
        self.assertEqual(len(result["segments"]), 3)

    def test_completed_transcription_does_not_reload_audio_or_model(self):
        self.transcribe(FakeModel())
        with patch.object(app, "write_outputs"):
            result = app.transcribe(self.source, self.folder,
                                    decoder=lambda *a, **k: self.fail("Audio recargado"),
                                    model_factory=lambda *a, **k: self.fail("Modelo recargado"))
        self.assertTrue(result["complete"])

    def test_no_token_is_needed_for_transcription(self):
        with patch.dict("os.environ", {}, clear=True):
            self.assertTrue(self.transcribe(FakeModel())["complete"])

    def test_cache_identity_depends_on_audio_and_asr_options(self):
        renamed = self.source.with_name("new-name.wav")
        renamed.write_bytes(self.source.read_bytes())
        self.assertEqual(self.folder, app.prepare_job(renamed, self.root / "output"))
        self.assertNotEqual(self.folder, app.prepare_job(renamed, self.root / "output", language="en"))
        self.assertNotEqual(self.folder, app.prepare_job(renamed, self.root / "output", model="medium"))
        renamed.write_bytes(b"another recording")
        self.assertNotEqual(self.folder, app.prepare_job(renamed, self.root / "output"))

    def test_speakers_can_change_inside_a_segment_and_remain_global(self):
        segments = [{"start": 0, "end": 5400, "text": " Hola Sí Adiós", "words": [
            {"start": 0, "end": 1, "word": " Hola"},
            {"start": 1, "end": 2, "word": " Sí"},
            {"start": 5399, "end": 5400, "word": " Adiós"},
        ]}]
        turns = [{"start": 0, "end": 1, "speaker": "SPEAKER_00"},
                 {"start": 1, "end": 2, "speaker": "SPEAKER_01"},
                 {"start": 5399, "end": 5400, "speaker": "SPEAKER_00"}]
        labelled = app.assign_speakers(segments, turns)
        self.assertEqual([s["speaker"] for s in labelled], ["SPEAKER_00", "SPEAKER_01", "SPEAKER_00"])
        self.assertEqual([s["text"] for s in labelled], ["Hola", "Sí", "Adiós"])
        result = {"complete": True, "diarized": True, "segments": labelled}
        archive = app.write_outputs(result, self.folder)
        with zipfile.ZipFile(archive) as bundle:
            for name in ("transcripcion.txt", "transcripcion.json", "subtitulos.srt", "subtitulos.vtt"):
                text = bundle.read(name).decode()
                self.assertIn("SPEAKER_00", text)
                self.assertIn("SPEAKER_01", text)
            self.assertIn("01:30:00,000", bundle.read("subtitulos.srt").decode())
        self.assertEqual(app.timestamp(3599.9999), "01:00:00.000")

    def test_missing_speaker_is_not_invented(self):
        output = app.assign_speakers([{"start": 10, "end": 11, "text": "hola", "words": []}],
                                    [{"start": 0, "end": 1, "speaker": "SPEAKER_00"}])
        self.assertEqual(output[0]["speaker"], "SIN_IDENTIFICAR")

    def test_diarization_failure_preserves_plain_transcript(self):
        self.transcribe(FakeModel())
        original = (self.folder / "transcripcion.txt").read_bytes()
        with patch.dict("os.environ", {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "HF_TOKEN"):
                app.diarize(self.source, self.folder)
        self.assertEqual(original, (self.folder / "transcripcion.txt").read_bytes())

    def test_cached_diarization_needs_no_token_or_heavy_imports(self):
        self.transcribe(FakeModel())
        app.save_json(self.folder / "hablantes.json", {"model": app.DIARIZATION_MODEL,
                      "turns": [{"start": 0, "end": 650, "speaker": "SPEAKER_00"}]})
        with patch.dict("os.environ", {}, clear=True):
            app.diarize(self.source, self.folder)
        self.assertTrue(app.read_json(self.folder / "transcripcion.json")["diarized"])

    def test_subtitle_split_preserves_every_word(self):
        words = [{"start": i, "end": i + 0.5, "word": f" w{i}"} for i in range(20)]
        cues = list(app.subtitle_cues([{"start": 0, "end": 20, "text": "", "words": words}]))
        self.assertEqual(" ".join(c["text"] for c in cues), " ".join(f"w{i}" for i in range(20)))
        self.assertTrue(all(c["end"] - c["start"] <= 7 for c in cues))

    def test_notebook_is_self_contained_and_one_cell(self):
        root = Path(__file__).resolve().parents[1]
        notebook = json.loads((root / "Whisper_Simple_Gratis.ipynb").read_text(encoding="utf-8"))
        cells = [c for c in notebook["cells"] if c["cell_type"] == "code"]
        self.assertEqual(len(cells), 1)
        code = "".join(cells[0]["source"])
        tree = ast.parse(code)
        constants = {}
        for node in tree.body:
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant):
                constants[node.targets[0].id] = node.value.value
        self.assertEqual(constants["WORKER_SOURCE"], (root / "whisper_gratis.py").read_text(encoding="utf-8"))
        self.assertFalse(constants["SEPARAR_HABLANTES"])
        self.assertTrue(constants["GUARDAR_EN_DRIVE"])
        self.assertIn("faster-whisper==1.2.1", constants["ASR_LOCK"])
        self.assertNotIn("assemblyai", code.lower())
        self.assertNotIn("api.openai.com", code)
        self.assertEqual(cells[0]["outputs"], [])

    def execute_notebook_with_fake_colab(self, diarization=False):
        root = Path(__file__).resolve().parents[1]
        notebook = json.loads((root / "Whisper_Simple_Gratis.ipynb").read_text(encoding="utf-8"))
        code = "".join(notebook["cells"][1]["source"])
        code = code.replace("GUARDAR_EN_DRIVE = True", "GUARDAR_EN_DRIVE = False")
        if diarization:
            code = code.replace("SEPARAR_HABLANTES = False", "SEPARAR_HABLANTES = True")
        for original, target in (("/content/Transcripciones_Whisper", self.root / "outputs"),
                                 ("/content/whisper_gratis_runtime", self.root / "runtime")):
            code = code.replace(f'Path("{original}")', f"Path({str(target)!r})")
        downloads, stages = [], []
        colab = types.ModuleType("google.colab")
        colab.files = SimpleNamespace(upload=lambda: {str(self.source): b"audio"}, download=downloads.append)
        colab.drive = SimpleNamespace(mount=lambda _: self.fail("Drive desactivado"))
        colab.userdata = SimpleNamespace(
            get=(lambda _: "fake-test-token") if diarization else (lambda _: self.fail("No necesita clave")),
            SecretNotFoundError=KeyError, NotebookAccessError=PermissionError,
        )
        google = types.ModuleType("google")
        google.colab = colab

        def fake_run(command, **options):
            if "transcribe" in command:
                stages.append("transcribe")
                folder = Path(command[command.index("--folder") + 1])
                app.transcribe(self.source, folder, "cpu", model_factory=lambda *a, **k: FakeModel(),
                               decoder=lambda *a, **k: Audio(650 * app.SAMPLE_RATE))
            if "diarize" in command:
                stages.append("diarize")
                self.assertEqual(options["env"]["HF_TOKEN"], "fake-test-token")
                self.assertNotIn("fake-test-token", " ".join(command))
                return SimpleNamespace(returncode=1)
            return SimpleNamespace(returncode=0)

        with patch.dict("sys.modules", {"google": google, "google.colab": colab}):
            with patch.dict("os.environ", {}, clear=True):
                with patch("subprocess.run", side_effect=fake_run), patch("subprocess.check_output", return_value=""):
                    with patch("shutil.which", return_value=None):
                        exec(compile(code, "Whisper_Simple_Gratis.ipynb", "exec"), {"__name__": "__main__"})
        self.assertEqual(len(downloads), 1)
        self.assertTrue(Path(downloads[0]).exists())
        with zipfile.ZipFile(downloads[0]) as archive:
            output = json.loads(archive.read("transcripcion.json"))
            self.assertTrue(output["complete"])
            self.assertFalse(output["diarized"])
            self.assertNotIn("fake-test-token", archive.read("transcripcion.json").decode())
        return stages

    def test_entire_notebook_default_flow_needs_no_key(self):
        self.assertEqual(self.execute_notebook_with_fake_colab(), ["transcribe"])

    def test_entire_notebook_downloads_text_when_diarization_fails(self):
        self.assertEqual(self.execute_notebook_with_fake_colab(diarization=True), ["transcribe", "diarize"])


if __name__ == "__main__":
    unittest.main()
