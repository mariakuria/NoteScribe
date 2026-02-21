import os
import uuid
import tempfile
import traceback
from pathlib import Path
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app) 

UPLOAD_FOLDER = Path(tempfile.gettempdir()) / "notescribe_uploads"
OUTPUT_FOLDER = Path(tempfile.gettempdir()) / "notescribe_outputs"
UPLOAD_FOLDER.mkdir(exist_ok=True)
OUTPUT_FOLDER.mkdir(exist_ok=True)

ALLOWED_EXTENSIONS = {".mp3", ".wav", ".m4a", ".ogg", ".flac", ".aac"}
MAX_FILE_SIZE = 50 * 1024 * 1024  


def allowed_file(filename: str) -> bool: 
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS




#Creates midi file from inputted audio (using BasicPitch)
def audio_to_midi(audio_path: Path, output_dir: Path, onset_threshold: float = 0.5,
                  frame_threshold: float = 0.3, min_note_length: float = 0.05) -> Path:
    from basic_pitch.inference import predict_and_save
    from basic_pitch import ICASSP_2022_MODEL_PATH

    predict_and_save(
        audio_path_list=[str(audio_path)],
        output_directory=str(output_dir),
        save_midi=True, 
        sonify_midi=False, 
        save_model_outputs=False,
        save_notes=False, 
        model_or_model_path=ICASSP_2022_MODEL_PATH,
        onset_threshold=onset_threshold, 
        frame_threshold=frame_threshold, 
        minimum_note_length=min_note_length, 
    )

    stem = audio_path.stem
    midi_path = output_dir / f"{stem}_basic_pitch.mid"
    if not midi_path.exists():
        candidates = list(output_dir.glob("*.mid")) 
        if not candidates:
            raise FileNotFoundError("Basic Pitch did not produce a MIDI file.")
        midi_path = candidates[0] 

    return midi_path 

#Creates sheet music from midi file (Using music21)

def midi_to_sheet(midi_path: Path, output_dir: Path,
                  key_sig: str = "auto", time_sig: str = "auto",
                  output_format: str = "musicxml") -> dict:
    
    from music21 import converter, stream, tempo, meter, key as m21key, analysis

    score: stream.Score = converter.parse(str(midi_path))

    if key_sig == "auto":
        detected_key = score.analyze("key")
        key_name = str(detected_key)
    else:
        detected_key = m21key.Key(key_sig)
        key_name = key_sig + " Major"
        for part in score.parts:
            part.insert(0, detected_key)

    if time_sig != "auto":
        ts = meter.TimeSignature(time_sig)
        for part in score.parts:
            for existing in part.getElementsByClass(meter.TimeSignature):
                part.remove(existing)
            part.insert(0, ts)
        detected_time = time_sig
    else:
        ts_list = score.flat.getElementsByClass(meter.TimeSignature)
        detected_time = str(ts_list[0]) if ts_list else "4/4"

    tempo_list = score.flat.getElementsByClass(tempo.MetronomeMark)
    bpm = int(tempo_list[0].number) if tempo_list else 120

    notes = score.flat.notes
    note_count = len(notes)
    note_names = []
    for n in notes:
        if hasattr(n, "pitch"):
            note_names.append(n.pitch.nameWithOctave)
        elif hasattr(n, "pitches"):
            note_names.extend([p.nameWithOctave for p in n.pitches])

    measures = score.parts[0].getElementsByClass("Measure") if score.parts else []
    measure_count = len(measures)

    stem = midi_path.stem.replace("_basic_pitch", "")
    outputs = {}

    xml_path = output_dir / f"{stem}.xml"
    score.write("musicxml", fp=str(xml_path))
    outputs["musicxml"] = str(xml_path)

    import shutil
    midi_out = output_dir / f"{stem}.mid"
    shutil.copy(midi_path, midi_out)
    outputs["midi"] = str(midi_out)

    if output_format == "pdf":
        try:
            pdf_path = output_dir / f"{stem}.pdf"
            score.write("musicxml.pdf", fp=str(pdf_path))
            outputs["pdf"] = str(pdf_path)
        except Exception as e:
            logger.warning(f"PDF export failed (MuseScore may not be installed): {e}")

    return {
        "paths": outputs,
        "metadata": {
            "key": key_name,
            "time_signature": detected_time,
            "tempo_bpm": bpm,
            "note_count": note_count,
            "measure_count": measure_count,
            "notes_sample": note_names[:60], 
        }
    }

#Routes section 
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "service": "NoteScribe"})


@app.route("/transcribe", methods=["POST"])
def transcribe():
    """
    Accepts multipart/form-data:
      - file        : audio file
      - key_sig     : e.g. "C", "G", "auto"  (default: auto)
      - time_sig    : e.g. "4/4", "3/4", "auto" (default: auto)
      - output_format: "musicxml" | "midi" | "pdf" (default: musicxml)
      - onset_threshold : float 0–1 (default 0.5)
      - frame_threshold : float 0–1 (default 0.3)

    Returns JSON with metadata + job_id for downloading files.
    """
    if "file" not in request.files:
        return jsonify({"error": "No audio file provided. Use field name 'file'."}), 400

    f = request.files["file"]
    if not f.filename:
        return jsonify({"error": "Empty filename."}), 400
    if not allowed_file(f.filename):
        return jsonify({"error": f"Unsupported format. Allowed: {ALLOWED_EXTENSIONS}"}), 400

    key_sig        = request.form.get("key_sig", "auto")
    time_sig       = request.form.get("time_sig", "auto")
    output_format  = request.form.get("output_format", "musicxml")
    onset_thresh   = float(request.form.get("onset_threshold", 0.5))
    frame_thresh   = float(request.form.get("frame_threshold", 0.3))

    job_id = str(uuid.uuid4())
    job_dir = OUTPUT_FOLDER / job_id
    job_dir.mkdir(parents=True)

    suffix = Path(f.filename).suffix.lower()
    audio_path = UPLOAD_FOLDER / f"{job_id}{suffix}"
    f.save(str(audio_path))
    logger.info(f"[{job_id}] Saved upload: {audio_path} ({audio_path.stat().st_size} bytes)")

#Attempts to actually convert from audio to sheet music
    try:
        logger.info(f"[{job_id}] Running Basic Pitch…")
        midi_path = audio_to_midi(audio_path, job_dir, onset_thresh, frame_thresh)
        logger.info(f"[{job_id}] MIDI saved: {midi_path}")

        logger.info(f"[{job_id}] Converting MIDI → sheet music…")
        result = midi_to_sheet(midi_path, job_dir, key_sig, time_sig, output_format)
        logger.info(f"[{job_id}] Done. Notes: {result['metadata']['note_count']}")

        return jsonify({
            "job_id": job_id,
            "status": "success",
            "metadata": result["metadata"],
            "download_urls": {
                fmt: f"/download/{job_id}/{fmt}"
                for fmt in result["paths"]
            }
        })

    except Exception as e:
        logger.error(f"[{job_id}] Transcription failed: {traceback.format_exc()}")
        return jsonify({"error": str(e), "job_id": job_id}), 500

    finally:
        try:
            audio_path.unlink()
        except Exception:
            pass


@app.route("/download/<job_id>/<fmt>", methods=["GET"])
def download(job_id: str, fmt: str):
    """Download a generated file by job_id and format (musicxml | midi | pdf)."""
    if ".." in job_id or "/" in job_id:
        return jsonify({"error": "Invalid job ID"}), 400

    job_dir = OUTPUT_FOLDER / job_id
    if not job_dir.exists():
        return jsonify({"error": "Job not found or expired"}), 404

    ext_map = {"musicxml": ".xml", "midi": ".mid", "pdf": ".pdf"}
    ext = ext_map.get(fmt)
    if not ext:
        return jsonify({"error": f"Unknown format '{fmt}'"}), 400

    candidates = list(job_dir.glob(f"*{ext}"))
    if not candidates:
        return jsonify({"error": f"No {fmt} file found for this job"}), 404

    file_path = candidates[0]
    mime_map = {
        ".xml": "application/vnd.recordare.musicxml+xml",
        ".mid": "audio/midi",
        ".pdf": "application/pdf",
    }
    return send_file(
        str(file_path),
        mimetype=mime_map.get(ext, "application/octet-stream"),
        as_attachment=True,
        download_name=file_path.name,
    )


@app.route("/job/<job_id>", methods=["GET"])
def job_status(job_id: str):
    """Check if a job's files still exist."""
    if ".." in job_id or "/" in job_id:
        return jsonify({"error": "Invalid job ID"}), 400
    job_dir = OUTPUT_FOLDER / job_id
    if not job_dir.exists():
        return jsonify({"status": "not_found"}), 404
    files = [f.name for f in job_dir.iterdir()]
    return jsonify({"status": "exists", "files": files})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)