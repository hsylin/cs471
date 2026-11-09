This is a new input corpus for the kernel optimization study. It contains ten
new prose/code document pairs: two pilot pairs and eight final pairs. All twenty
document IDs, URLs, raw contents, and cleaned contents are disjoint from the
entire earlier payback corpus. The exclusion manifest is hash-pinned in
`sources.json`; `optimization.input_sources` verifies the exclusion and every
source checksum before model capture. Selection preceded new routing capture
and timing inspection.

The prose consists of complete Project Gutenberg texts. Each original notice
is retained, alongside the retrieved catalog HTML declaring public-domain US
status and its source URL. The authors died by 1942; the War and Peace edition
credits translators Aylmer Maude (1938) and Louise Maude (1939). CPython source
files use commit `0fb18b02c8ad56299d6a2910be0bab8ad601ef24`; the complete unchanged
LICENSE and all file-level notices are archived. `download-inventory.json`
records actual retrieval URLs, byte counts and hashes. None of these Python
files is executed as part of corpus preparation or capture.

The fixed pairs are:

| Split/block | Prose | CPython file |
|---|---|---|
| opt-pilot-01 | Wuthering Heights | Lib/random.py |
| opt-pilot-02 | The Scarlet Letter | Lib/string.py |
| opt-final-01 | Jane Eyre | Lib/calendar.py |
| opt-final-02 | Middlemarch | Lib/textwrap.py |
| opt-final-03 | Walden | Lib/configparser.py |
| opt-final-04 | Anne of Green Gables | Lib/pprint.py |
| opt-final-05 | Little Women | Lib/shlex.py |
| opt-final-06 | Huckleberry Finn | Lib/enum.py |
| opt-final-07 | Ulysses | Lib/ipaddress.py |
| opt-final-08 | War and Peace | Lib/_pydecimal.py |

The new capture entry point preserves the exact pinned predecessor computation:
pretrained embedding, layer-0 causal attention, residual/norm, and original Qwen
router from revision `ad44e777bcd18fa416d9da3bd8f70d33ebb85d39`, under the same
hash-verified Transformers 5.17.0 implementation. Only expert computation is
replaced by a recorder after routing. Each source supplies its first 2,048
actual tokenizer tokens as a causal prefill prefix. No chat template or special
tokens are added. Native boundary/router, formula/rounding, and BF16 round-trip
checks remain unchanged. Banks record the new capture/validation script hashes,
excluded-manifest pin, and exact source/model provenance.

Run on the existing capture environment:

```bash
python scripts/capture_optimization_tapes.py \
  --sources optimization/corpus/sources.json \
  --output-dir /path/to/new-optimization-capture \
  --cache-dir /path/to/existing-hf-cache \
  --local-files-only --split all --compose
```

`--validate-sources-only` checks corpus integrity on CPU without model loading.
Capture creates `banks/`, `tapes/` and `capture-index.json` and refuses to
replace existing banks. All final inputs may be captured before freezing the
optimization, but final routing statistics and runtime outcomes must not inform
kernel selection. Tuning uses only the two new pilot blocks.

The constructed replay has 544 raw inputs: 32 observation warmup inputs followed
by 512 measured inputs. Transient and persistent switch domains at measured120;
transient returns at136 with no recurrence (`repeat_steps=1024`). Switching
begins at112 in 16-step phases. Replay seed is10471; token batches are128/1024.
The common first scheduled decision63 precedes the regime change, while127
observes the change. These schedules retain the prior decision geometry and do
not represent natural serving arrivals. Use `scripts/compose_optimization_tapes.py`
with a new output directory to reproduce these schedules from authenticated
banks without changing their original capture metadata.

The finite, hand-selected document pairs are the comparison blocks. Authors,
projects, and language conventions can still be related; disjoint documents do
not establish independent draws from a production population. Python prefixes
may include comments and docstrings. Replay reuses actual captured token records
across steps; it is constructed repeated-prefill replay, not autoregressive
serving demand. Technical repetitions or resampling seeds create no new source
blocks. The complete frozen prior capture code and corpus are unchanged.
