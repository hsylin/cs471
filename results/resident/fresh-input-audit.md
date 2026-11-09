Resident-dispatch input audit: verified all 8 banks and 64 replay tapes.

All file/content hashes, source exclusions, capture provenance, and array-validity checks passed. The capture checkpoint loading and native router computation match the frozen predecessor exactly. All replay indices were independently regenerated from seed 20471; raw 544/warm 32/phase 120/switch 112/duration 16/repeat 1024 match. Transient/persistent histories match through raw167 and then diverge as declared.

No expert-demand distributions, runtime results, or tuning choices were inspected.

- This is a same-team CPU audit, not external-lab reproduction. It verifies actual artifacts and recorded GPU capture attestations; it does not independently rerun pretrained attention/router computation.
- Valid unique top-k IDs and rounded combining weights do not independently prove their learned-router origin; pinned capture computation and its exactness attestations supply that evidence.
- Document disjointness concerns document identities and whole-content hashes, not absence of shared language, comments, authors, or project structure.
- Constructed repeated-prefill replay and fixed document blocks are not natural serving traffic or random production samples.
