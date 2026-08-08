"""P2: exercise the PRODUCTION concept-retrieval path (LanceDBConceptStore + the
RRF merge + the SQL filter), which the suite previously tested only through the
lexical-only InMemoryConceptStore stand-in (test/codebase audit)."""

from __future__ import annotations

from trialmatchai.entities.linker import (
    ConceptLinker,
    LanceDBConceptStore,
    _lancedb_filter,
    _rrf_merge,
    _sql_escape,
    lexical_reranker,
)
from trialmatchai.entities.types import ConceptCandidate, EntityAnnotation, EntitySchema


def _candidate(concept_id, vocab="", code="", name="x", domain=""):
    return ConceptCandidate(
        concept_id=concept_id,
        vocabulary_id=vocab,
        concept_code=code,
        concept_name=name,
        domain_id=domain,
    )


class _FixedCandidateLanceStore(LanceDBConceptStore):
    """Production search/fusion with deterministic channel results and no database."""

    def __init__(self, fts, vector):
        self.fts = fts
        self.vector = vector

    def _embed_query(self, query):
        return [1.0]

    def _search_fts(self, query, vocabularies, domain_hints, limit):
        return self.fts[:limit]

    def _search_vector(self, vector, vocabularies, domain_hints, limit):
        return self.vector[:limit]


def _linker(store, *, entity_group, vocabulary, domain):
    schema = EntitySchema(
        id=entity_group,
        label=entity_group,
        entity_group=entity_group,
        description=entity_group,
        target_vocabularies=(vocabulary,),
        domain_hints=(domain,),
    )
    return ConceptLinker(
        store,
        (schema,),
        reranker=lexical_reranker,
        search_limit=10,
        retrieval_limit=50,
    )


def _annotation(entity_group, text):
    return EntityAnnotation(
        entity_group=entity_group,
        schema_id=entity_group,
        text=text,
        start=0,
        end=len(text),
        score=1.0,
    )


def test_rrf_merge_keeps_distinct_cui_less_candidates():
    # Two distinct concepts with empty vocab/code share the constant normalized_id
    # "CUI-less"; the old dedup key collapsed them to one. Now keyed on concept_id.
    a = _candidate("C1", name="alpha")
    b = _candidate("C2", name="beta")
    merged = _rrf_merge([a, b], [], limit=10)
    assert {c.concept_id for c in merged} == {"C1", "C2"}


def test_rrf_merge_combines_same_concept_across_channels():
    fts = _candidate("SNOMED:1", vocab="SNOMED", code="1", name="diabetes", domain="Condition")
    vec = _candidate("SNOMED:1", vocab="SNOMED", code="1", name="diabetes", domain="Condition")
    merged = _rrf_merge([fts], [vec], limit=10)
    assert len(merged) == 1  # same concept_id -> one combined result
    assert set(merged[0].source_scores) == {"fts", "vector"}


def test_linker_reranks_deeper_pool_before_final_candidate_limit():
    # Reproduces the full-store EGFR miss: the exact FTS match is rank 8, while
    # three approximate candidates occur in both channels and dominate RRF's
    # first ten merged results. The exact match must survive until reranking.
    fts = [
        _candidate("NCBIGene:100507500", "NCBIGene", "100507500", "EGFR-AS1", "Gene"),
        _candidate("NCBIGene:116435281", "NCBIGene", "116435281", "EGILA", "Gene"),
        _candidate("NCBIGene:144093424", "NCBIGene", "144093424", "EUDAL", "Gene"),
        _candidate("NCBIGene:102725541", "NCBIGene", "102725541", "ELDR", "Gene"),
        _candidate("NCBIGene:55220", "NCBIGene", "55220", "KLHDC8A", "Gene"),
        _candidate("NCBIGene:2059", "NCBIGene", "2059", "EPS8", "Gene"),
        _candidate("NCBIGene:100996668", "NCBIGene", "100996668", "LINC02381", "Gene"),
        _candidate("NCBIGene:1956", "NCBIGene", "1956", "EGFR", "Gene"),
        _candidate("NCBIGene:900000009", "NCBIGene", "900000009", "fts filler 9", "Gene"),
        _candidate("NCBIGene:900000010", "NCBIGene", "900000010", "fts filler 10", "Gene"),
        *[
            _candidate(
                f"NCBIGene:9000000{rank}",
                "NCBIGene",
                f"9000000{rank}",
                f"fts filler {rank}",
                "Gene",
            )
            for rank in range(11, 51)
        ],
    ]
    vector = [
        fts[1],
        fts[2],
        _candidate("NCBIGene:800000003", "NCBIGene", "800000003", "vector filler 3", "Gene"),
        fts[0],
        *[
            _candidate(
                f"NCBIGene:80000000{rank}",
                "NCBIGene",
                f"80000000{rank}",
                f"vector filler {rank}",
                "Gene",
            )
            for rank in range(5, 44)
        ],
        fts[7],
        *[
            _candidate(
                f"NCBIGene:80000000{rank}",
                "NCBIGene",
                f"80000000{rank}",
                f"vector filler {rank}",
                "Gene",
            )
            for rank in range(45, 51)
        ],
    ]

    result = _linker(
        _FixedCandidateLanceStore(fts, vector),
        entity_group="gene",
        vocabulary="NCBIGene",
        domain="Gene",
    ).link_annotation(
        _annotation("gene", "EGFR")
    )

    assert result.linker_status == "accepted"
    assert result.normalized_id == ("EntrezGene:1956",)
    assert len(result.concept_candidates) == 10


def test_linker_retrieves_exact_match_beyond_final_candidate_limit():
    # Reproduces the full-store Fever miss: the exact preferred term is around
    # FTS rank 50 and vector rank 13, so final limit ten must not cap retrieval.
    fts = [
        _candidate(
            f"SNOMED:{rank}",
            "SNOMED",
            str(rank),
            f"fever candidate {rank}",
            "Condition",
        )
        for rank in range(1, 51)
    ]
    exact = _candidate("SNOMED:386661006", "SNOMED", "386661006", "Fever", "Condition")
    fts[49] = exact
    vector = [
        _candidate(
            f"SNOMED:800000{rank}",
            "SNOMED",
            f"800000{rank}",
            f"vector candidate {rank}",
            "Condition",
        )
        for rank in range(1, 51)
    ]
    vector[12] = exact

    result = _linker(
        _FixedCandidateLanceStore(fts, vector),
        entity_group="sign symptom",
        vocabulary="SNOMED",
        domain="Condition",
    ).link_annotation(
        _annotation("sign symptom", "fever")
    )

    assert result.linker_status == "accepted"
    assert result.normalized_id == ("SNOMED:386661006",)
    assert len(result.concept_candidates) == 10


def test_lancedb_filter_and_sql_escape():
    assert _lancedb_filter(["DOID", "SNOMED"], ["Disease"]) == (
        "vocabulary_id IN ('DOID', 'SNOMED') AND domain_id IN ('Disease')"
    )
    assert _lancedb_filter(["DOID"], []) == "vocabulary_id IN ('DOID')"
    assert _lancedb_filter([], []) == ""
    assert _sql_escape("O'Brien") == "O''Brien"  # quote is escaped, not injected


def test_lancedb_concept_store_search_applies_vocabulary_filter(tmp_path):
    from trialmatchai.entities.builder import write_lancedb_table

    rows = [
        {
            "concept_id": "DOID:1", "vocabulary_id": "DOID", "concept_code": "1",
            "concept_name": "diabetes mellitus", "domain_id": "Disease",
            "concept_class_id": "", "standard_concept": "", "synonyms": ["diabetes"],
            "fts_text": "diabetes mellitus diabetes",
        },
        {
            "concept_id": "RxNorm:1", "vocabulary_id": "RxNorm", "concept_code": "1",
            "concept_name": "metformin", "domain_id": "Drug",
            "concept_class_id": "", "standard_concept": "", "synonyms": [],
            "fts_text": "metformin",
        },
    ]
    db = str(tmp_path / "concepts")
    write_lancedb_table(rows, db_path=db, table_name="concepts", embeddings=[[1.0, 0, 0, 0], [0, 1.0, 0, 0]])

    class FakeEmbedder:
        def embed_text(self, text):
            return [1.0, 0, 0, 0]  # nearest the DOID row

    store = LanceDBConceptStore(db, table_name="concepts", embedder=FakeEmbedder())

    # The disease vocabulary is searched and returns the disease concept.
    hits = store.search("diabetes", vocabularies=["DOID"], domain_hints=["Disease"], limit=5)
    assert hits and hits[0].concept_id == "DOID:1"
    # Filtering to a different vocabulary excludes it entirely.
    other = store.search("diabetes", vocabularies=["RxNorm"], domain_hints=["Drug"], limit=5)
    assert all(h.concept_id != "DOID:1" for h in other)
