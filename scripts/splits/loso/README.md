# LOSO splits

These files are generated from `data/official-phase-large/demographics.csv` by:

```bash
python scripts/make_loso_splits.py
python scripts/verify_loso_splits.py
```

Each split contains the exact train/validation record lists, held-out site, site
composition, raw and CAISR-available counts, and a record-content SHA-256.
Reported dataset sizes for model training must use the CAISR-available counts;
the raw split retains all 6,600 labelled records for exact bookkeeping.
`manifest.json` stores the SHA-256 of each file and a digest of the labelled
records **including CAISR availability**; changing the data root in a way that
silently changes the usable counts therefore invalidates the manifest.
`scripts/verify_loso_splits.py` checks that digest, every manifest count and
the exact record sets, while `scripts/lo_site_sched.py` repeats the safety
checks immediately before training.

The older `scripts/split_I0006.json` is retained for historical reference but is
not used by the current LOSO runner.
