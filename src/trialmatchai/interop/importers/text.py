from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from trialmatchai.interop.models import Demographics, PatientNote, PatientProfile, Provenance
from trialmatchai.interop.utils import (
    make_fact,
    normalize_gender,
    safe_patient_id,
    source_path_string,
)


ENTITY_GROUP_TO_CATEGORY = {
    "disease": "condition",
    "condition": "condition",
    "drug": "medication",
    "medication": "medication",
    "procedure": "procedure",
    "diagnostic_test": "observation",
    "laboratory_test": "observation",
    "radiology": "diagnostic_report",
    "sign_symptom": "phenotype",
    "gene": "genomic_finding",
    "cell_type": "phenotype",
    "species": "phenotype",
}


def import_text_note(
    path: str | Path,
    *,
    entity_annotator: Any | None = None,
) -> PatientProfile:
    note_path = Path(path)
    text = note_path.read_text(encoding="utf-8")
    patient_id = safe_patient_id(note_path.stem, note_path.stem)
    provenance = Provenance(
        source_format="text",
        source_id=patient_id,
        source_path=source_path_string(note_path),
        source_field="note_text",
    )
    entities = _annotate(text, entity_annotator)
    profile = PatientProfile(
        patient_id=patient_id,
        demographics=_extract_demographics(text),
        notes=[
            PatientNote(
                note_id=f"{patient_id}-note",
                text=text,
                entities=entities,
                provenance=provenance,
            )
        ],
        provenance=[provenance],
    )
    for entity in entities:
        fact = _entity_to_fact(entity, provenance)
        if fact is not None:
            profile.add_fact(fact)
    return profile


# A free-text note carries age and sex only in prose, so they must be read out of it: the
# structured importers get them from Phenopacket ``subject``, FHIR ``Patient`` or OMOP ``person``,
# and a text note has no equivalent field. Without this the profile's demographics stay empty and
# ``profile_to_matching_summary`` falls back to "all", which silently disables the ``age`` and
# ``sex`` entries of ``search.first_level.hard_filters`` for every text-ingested patient.
#
# Patterns are taken verbatim from the reference implementation's own extraction
# (``utils/gpt/gpt-generate-summaries.py`` in the v0.01 release), which reads age and sex from the
# raw description and writes them as the flat ``age``/``gender`` keys the matcher consumes. The
# deterministic regex is used rather than that script's LLM fallback: it needs no external model,
# it normalises to the same two values, and it keeps ingestion reproducible.
_AGE_PATTERN = re.compile(
    r"(\b\d{1,3}\b)[-\s]?(?:year-old|yr-old|years old|year old)", re.IGNORECASE
)
# TREC topics are admission notes and write the same fact as "22yo", "75 yo", "70 y/o". The
# reference regex matches none of those, but its own LLM fallback is instructed to "Normalize the
# Age to an integer number and Gender to either Male or Female", so recovering them is that
# fallback's job done deterministically rather than a departure from the method.
_AGE_SHORTHAND_PATTERN = re.compile(
    r"(\d{1,3})\s*(?:y\.?o\.?\b|y/o|yrs?\.?[-\s]?old|years?[-\s]?old)", re.IGNORECASE
)
_SEX_PATTERN = re.compile(
    r"\b(male|female|man|woman|boy|girl|gentleman|lady)\b", re.IGNORECASE
)
# "48 M", "74M", "60 yo M" carry age and sex with no word for either. A bare digit-letter pair is
# far too loose to trust anywhere in a note - "2 M" is a concentration, "5 F" a catheter size - so
# it is honoured only in the note's opening, where these notes state demographics and nowhere else
# does. A wrong age is NOT recall-safe (it filters on the trial's min/max age), which is why this
# one is positional and case-sensitive rather than permissive.
_DEMOGRAPHIC_SHORTHAND_PATTERN = re.compile(
    r"\b(\d{1,3})\s*(?:y\.?o\.?|y/o|yrs?|years?)?\s*(M|F)\b"
)
_SHORTHAND_WINDOW = 60
_SEX_NORMALIZATION = {
    "male": "male",
    "man": "male",
    "boy": "male",
    "gentleman": "male",
    "m": "male",
    "female": "female",
    "woman": "female",
    "girl": "female",
    "lady": "female",
    "f": "female",
}
_MAX_PLAUSIBLE_AGE = 120


def _extract_demographics(text: str) -> Demographics:
    """Read age and sex out of a free-text note, as the reference implementation does."""

    opening = text[:_SHORTHAND_WINDOW]
    shorthand = _DEMOGRAPHIC_SHORTHAND_PATTERN.search(opening)

    age_years: float | None = None
    for match in (_AGE_PATTERN.search(text), _AGE_SHORTHAND_PATTERN.search(text), shorthand):
        if match is None:
            continue
        candidate = int(match.group(1))
        # A three-digit match can be a typo or a stray number; keep only plausible human ages so a
        # bad parse cannot narrow retrieval. Out-of-range values leave the filter disabled.
        if 0 < candidate <= _MAX_PLAUSIBLE_AGE:
            age_years = float(candidate)
            break

    sex: str | None = None
    sex_match = _SEX_PATTERN.search(text)
    token = sex_match.group(1) if sex_match is not None else (
        shorthand.group(2) if shorthand is not None else None
    )
    if token is not None:
        sex = normalize_gender(_SEX_NORMALIZATION[token.casefold()])

    return Demographics(age_years=age_years, sex=sex)

def _annotate(text: str, entity_annotator: Any | None) -> list[dict]:
    if entity_annotator is None:
        return []
    if hasattr(entity_annotator, "annotate_texts_in_parallel"):
        result = entity_annotator.annotate_texts_in_parallel([text], max_workers=1)
        return list(result[0]) if result else []
    if hasattr(entity_annotator, "annotate_texts"):
        result = entity_annotator.annotate_texts([text])
        annotations = result[0] if result else []
        return [
            annotation.to_dict() if hasattr(annotation, "to_dict") else dict(annotation)
            for annotation in annotations
        ]
    return []


def _entity_to_fact(entity: dict, provenance: Provenance):
    if entity.get("error_code"):
        return None
    group = str(entity.get("entity_group") or entity.get("class") or "").casefold()
    # Groups may arrive space-separated ("sign symptom") but map keys are underscore-joined;
    # normalize so they don't fall through to the "observation" default (e.g. -> phenotype).
    category = ENTITY_GROUP_TO_CATEGORY.get(group.replace(" ", "_"), "observation")
    text = str(entity.get("text") or entity.get("entity") or "").strip()
    if not text:
        return None
    normalized_codes = []
    for normalized_id in entity.get("normalized_id") or []:
        if normalized_id == "CUI-less" or ":" not in normalized_id:
            continue
        vocabulary, code = normalized_id.split(":", 1)
        normalized_codes.append(
            {
                "vocabulary": vocabulary,
                "code": code,
                "label": text,
                "confidence": entity.get("linker_score") or entity.get("score"),
                "mapping_status": "normalized",
            }
        )
    from trialmatchai.interop.models import NormalizedCode

    return make_fact(
        category=category,
        label=text,
        provenance=provenance,
        normalized_codes=[NormalizedCode.model_validate(code) for code in normalized_codes],
        evidence_text=text,
        evidence_start=entity.get("start"),
        evidence_end=entity.get("end"),
        confidence=entity.get("score"),
        extra={
            "entity_group": entity.get("entity_group"),
            "synonyms": entity.get("synonyms") or [],
            "concept_candidates": entity.get("concept_candidates") or [],
            "linker_status": entity.get("linker_status"),
        },
    )
