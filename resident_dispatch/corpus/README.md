# Resident-dispatch input corpus

Eight fixed document pairs provide fresh inputs for the conditional resident-dispatch experiment. Every source ID, URL, raw-file hash, and cleaned-text hash differs from all 40 documents in `followup/corpus/sources.json` and `optimization/corpus/sources.json`. All new blocks are labeled `final`; no source was chosen using routing or performance measurements.

| Block | Public-domain prose | CPython file |
|---|---|---|
| rd-final-01 | [Emma](https://www.gutenberg.org/ebooks/158) | Lib/tokenize.py |
| rd-final-02 | [Great Expectations](https://www.gutenberg.org/ebooks/1400) | Lib/inspect.py |
| rd-final-03 | [The Adventures of Tom Sawyer](https://www.gutenberg.org/ebooks/74) | Lib/ast.py |
| rd-final-04 | [Treasure Island](https://www.gutenberg.org/ebooks/120) | Lib/dis.py |
| rd-final-05 | [The Wonderful Wizard of Oz](https://www.gutenberg.org/ebooks/55) | Lib/subprocess.py |
| rd-final-06 | [The Odyssey, Samuel Butler prose translation](https://www.gutenberg.org/ebooks/1727) | Lib/zipfile/__init__.py |
| rd-final-07 | [The War of the Worlds](https://www.gutenberg.org/ebooks/36) | Lib/tarfile.py |
| rd-final-08 | [The Yellow Wallpaper](https://www.gutenberg.org/ebooks/1952) | Lib/uuid.py |

The complete original text, its Gutenberg notice, and the catalog page declaring US public-domain status are archived. Capture removes only the Gutenberg wrapper between its explicit start/end markers. CPython files are unchanged, pinned to commit `0fb18b02c8ad56299d6a2910be0bab8ad601ef24`; the complete CPython license is retained in `raw/cpython_LICENSE`. `sources.json` contains exact URLs, revisions, raw/cleaned hashes, and prior-manifest pins. `download-inventory.json` records download URLs, resolved URLs, file sizes, and hashes.

Capture uses the first 2,048 tokenizer tokens of each source, without a chat template or added special tokens. Each block yields 4,096 records. The pretrained Qwen layer-0 embedding, causal attention, residual, normalization, and native router computation remain byte-for-byte identical to `capture_optimization_tapes.py`; CPU tests protect those sections. The script checks the installed Transformers source hashes, actual loaded checkpoint tensors, native decoder boundary, native router recomputation, router formula and BF16 rounding, and exact BF16 serialization. No synthetic routing or substituted hidden states are eligible as capture data.

Composition samples recorded token rows without replacement within a step and reuses them across steps. The fixed geometry is raw544 = warm32 + measured512, batches128/1024, stable/transient/persistent/switching, phase120 (switching112), duration16, repeat1024, and seed20471. The transient changes domain only at raw steps152–167. These are constructed replay schedules of teacher-forced prefill records, not natural online demand or generated decoding sessions. Document pairs are finite corpus blocks; shared authors and one CPython project prevent interpreting source-disjointness as a random production sample.

CPU-only provenance check (does not create either output directory or import the GPU model runtime):

```bash
python scripts/capture_resident_tapes.py --validate-sources-only \
  --output-dir work/unused-resident-capture --cache-dir work/unused-model-cache
python -m pytest tests/test_resident_inputs.py -q
```

Only after the experiment's headroom decision authorizes GPU capture:

```bash
python scripts/capture_resident_tapes.py --split final \
  --output-dir /fresh/capture-directory --cache-dir /existing/pinned-model-cache --compose
```

Alternatively, compose the same geometry from already captured banks:

```bash
python scripts/compose_resident_tapes.py --banks-dir /capture-directory/banks \
  --output-dir /fresh/tape-directory --split final
```

Both paths preserve capture metadata and refuse to overwrite authenticated banks or tapes. The separate composer requires this study's capture-script, validation-module, source-text helper, predecessor-script, router-tape module, source-manifest, and per-prompt native-attestation pins. Include `resident_dispatch/corpus/validation.py` in any experiment source fingerprint; it is nested below the package root. Production capture has not been performed merely by preparing this corpus.
