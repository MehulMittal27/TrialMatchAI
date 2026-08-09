from __future__ import annotations

import json

from trialmatchai.cli.import_patient import main


def test_import_patient_cli_writes_profile_and_summary(tmp_path, monkeypatch):
    note = tmp_path / "patient.txt"
    note.write_text("Patient has breast cancer.", encoding="utf-8")
    profile_dir = tmp_path / "profiles"
    summary_dir = tmp_path / "summaries"

    monkeypatch.setattr(
        "sys.argv",
        [
            "trialmatchai import-patient",
            "--input",
            str(note),
            "--format",
            "text",
            "--output-dir",
            str(profile_dir),
            "--summary-dir",
            str(summary_dir),
            "--no-entities",
        ],
    )

    assert main() == 0

    profile = json.loads((profile_dir / "patient.json").read_text(encoding="utf-8"))
    summary = json.loads((summary_dir / "patient.json").read_text(encoding="utf-8"))
    assert profile["patient_id"] == "patient"
    assert summary["patient_id"] == "patient"
    assert summary["main_conditions"] == ["Patient has breast cancer."]
    assert summary["patient_narrative"]


def test_import_patient_cli_reuses_one_annotator_for_multiple_inputs(
    tmp_path, monkeypatch
):
    first = tmp_path / "1.txt"
    second = tmp_path / "2.txt"
    first.write_text("Patient one.", encoding="utf-8")
    second.write_text("Patient two.", encoding="utf-8")
    profile_dir = tmp_path / "profiles"
    summary_dir = tmp_path / "summaries"
    annotators = []
    monkeypatch.setattr(
        "trialmatchai.cli.import_patient._try_build_entity_annotator",
        lambda _config: annotators.append(object()) or annotators[-1],
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "trialmatchai import-patient",
            "--input",
            str(first),
            "--input",
            str(second),
            "--format",
            "text",
            "--output-dir",
            str(profile_dir),
            "--summary-dir",
            str(summary_dir),
        ],
    )

    assert main() == 0
    assert sorted(path.stem for path in profile_dir.glob("*.json")) == ["1", "2"]
    assert len(annotators) == 1
