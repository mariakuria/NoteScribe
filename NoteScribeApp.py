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
CORS(app)  # Allow frontend to call this API

UPLOAD_FOLDER = Path(tempfile.gettempdir()) / "notescribe_uploads"
OUTPUT_FOLDER = Path(tempfile.gettempdir()) / "notescribe_outputs"
UPLOAD_FOLDER.mkdir(exist_ok=True)
OUTPUT_FOLDER.mkdir(exist_ok=True)

ALLOWED_EXTENSIONS = {".mp3", ".wav", ".m4a", ".ogg", ".flac", ".aac"}
MAX_FILE_SIZE = 50 * 1024 * 1024  #caps uploads at 50MB.


def allowed_file(filename: str) -> bool: #convert suffix to lower case and check if type is allowed
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS



#---------------------------------------------------------------------------------------------------------------
# CONVERT AUDIO TO MIDI
def audio_to_midi(audio_path: Path, output_dir: Path, onset_threshold: float = 0.5,
                  frame_threshold: float = 0.3, min_note_length: float = 0.05) -> Path:
    #Runs Spotify's Basic Pitch to convert audio to MIDI.
    from basic_pitch.inference import predict_and_save
    from basic_pitch import ICASSP_2022_MODEL_PATH

    predict_and_save(
        audio_path_list=[str(audio_path)],
        output_directory=str(output_dir),
        save_midi=True, #true, output a MIDI file
        sonify_midi=False, #don't also create an audio version of the MIDI
        save_model_outputs=False,
        save_notes=False, # skip extra output files you don't need
        model_or_model_path=ICASSP_2022_MODEL_PATH,
        onset_threshold=onset_threshold, #how confident the model needs to be that a note starts
        frame_threshold=frame_threshold, #how confident it needs to be that a note is sustaining
        minimum_note_length=min_note_length, #ignores very short blips (in seconds)
    )

    # Basic Pitch names the MIDI file after the input audio file
    stem = audio_path.stem
    midi_path = output_dir / f"{stem}_basic_pitch.mid" #names the output file after input file, so if input was song.mp3, it looks for song_basic_pitch.mid
    if not midi_path.exists():
        # Try alternate naming
        candidates = list(output_dir.glob("*.mid")) #if path doesn't exist check in candidates array
        if not candidates:
            raise FileNotFoundError("Basic Pitch did not produce a MIDI file.")
        midi_path = candidates[0] #add file to candidates

    return midi_path #returns the path to the MIDI file



#------------------------------------------------------------------------------------------
# CONVERT MIDI TO SHEET MUSIC using Music21

def midi_to_sheet(midi_path: Path, output_dir: Path,
                  key_sig: str = "auto", time_sig: str = "auto",
                  output_format: str = "musicxml") -> dict:
    
    from music21 import converter, stream, tempo, meter, key as m21key, analysis

    score: stream.Score = converter.parse(str(midi_path))

    # ── Key signature ──
    if key_sig == "auto":
        detected_key = score.analyze("key")
        key_name = str(detected_key)
    else:
        detected_key = m21key.Key(key_sig)
        key_name = key_sig + " Major"
        for part in score.parts:
            part.insert(0, detected_key)

    # ── Time signature ──
    if time_sig != "auto":
        ts = meter.TimeSignature(time_sig)
        for part in score.parts:
            # Remove existing time signatures then add the chosen one
            for existing in part.getElementsByClass(meter.TimeSignature):
                part.remove(existing)
            part.insert(0, ts)
        detected_time = time_sig
    else:
        ts_list = score.flat.getElementsByClass(meter.TimeSignature)
        detected_time = str(ts_list[0]) if ts_list else "4/4"

    # ── Tempo ──
    tempo_list = score.flat.getElementsByClass(tempo.MetronomeMark)
    bpm = int(tempo_list[0].number) if tempo_list else 120

    # ── Note stats ──
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

    # ── Export ──
    stem = midi_path.stem.replace("_basic_pitch", "")
    outputs = {}

    # Always export MusicXML (used for re-rendering)
    xml_path = output_dir / f"{stem}.xml"
    score.write("musicxml", fp=str(xml_path))
    outputs["musicxml"] = str(xml_path)

    # Copy MIDI as well
    import shutil
    midi_out = output_dir / f"{stem}.mid"
    shutil.copy(midi_path, midi_out)
    outputs["midi"] = str(midi_out)

    # PDF via MuseScore if available, otherwise skip gracefully
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
            "notes_sample": note_names[:60],  # first 60 for display
        }
    }