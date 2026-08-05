# Bootstrap data directory normalization

## Issues

The published TrialMatchAI data archive `processed_trials.tar.gz` currently
contains a top-level directory named `processed_docs/`. TrialMatchAI v0.7.0's
bootstrap code and downstream commands expect the canonical directory
`data/processed_trials/`.

Without normalization, `bootstrap-data` extracts the trial files successfully
but fails while writing its completion marker:

```text
FileNotFoundError: .../data/processed_trials/.bootstrap_complete
```

This is a packaging/path mismatch, not a corrupted trial corpus.

The criteria ZIP archives also currently contain a redundant top-level
`processed_criteria/` directory. Extracting them into `data/processed_criteria/`
would otherwise produce `data/processed_criteria/processed_criteria/`, while
the indexer expects trial-ID directories directly below `data/processed_criteria/`.

## Fix

`src/trialmatchai/cli/bootstrap_data.py` now normalizes the archive layout at
the extraction boundary:

```text
data/processed_docs/  ->  data/processed_trials/
data/processed_criteria/processed_criteria/  ->  data/processed_criteria/
```

The rest of the pipeline therefore keeps one stable path. If an archive
contains the canonical directory already, it is left unchanged. If both names
exist, bootstrap fails closed instead of merging ambiguous corpora. If neither
exists, bootstrap reports an invalid archive layout.

## Verification

The regression tests are in `tests/test_bootstrap_data.py` and cover both
published archive layouts. Verification performed locally:

```text
7 bootstrap-data tests passed
The full suite passed before the criteria-layout fix; rerun it after applying
the fix before committing.
```

## Reproduction and recovery

From the TrialMatchAI checkout, run:

```bash
uv run trialmatchai bootstrap-data
```

The command should finish using the already downloaded archives, normalize the
directory, and remove temporary archives after successful extraction. Confirm
the result with:

```bash
find data/processed_trials -maxdepth 1 -type f | head
find data/processed_criteria -maxdepth 2 -type f | head
```

## Future review

When the upstream Zenodo archive is refreshed, inspect its top-level members
before changing this normalization. If the upstream archive adopts the
canonical paths, the compatibility paths can remain harmlessly in place or be
removed in a separately reviewed cleanup. Do not silently merge conflicting
directories.

## Large-corpus indexing note

The prepared corpus contains more than two million criteria JSON files. The
indexer previously materialized every criteria record in one Python list before
writing LanceDB, which could leave only the trial table on a 24 GB Mac if the
process ran out of memory. Criteria indexing now uses bounded 8,192-record
batches and creates the FTS/vector indexes only after the final batch.

If a run stops after reporting the trial count, check for both tables before
retrying:

```bash
uv run trialmatchai healthcheck --require-tables
```

The retry is safe: the trial table is recreated and criteria batches are
written with bounded memory. Do not use `--reindex` unless an intentional full
rebuild is needed.
