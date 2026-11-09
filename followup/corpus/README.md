This archive supplies ten fixed pairs of public literary text and CPython source
documents for authentic Qwen layer-0 routing capture. Two pairs are pilot data;
eight disjoint pairs are final data. The source selection was made before any
capture or runtime inspection. Source IDs never overlap across pairs or splits.

The literary works are old public-domain works distributed by Project Gutenberg.
The complete downloaded source, including its Gutenberg license and notices, is
retained. The capture script excludes the Gutenberg wrapper from model input.
CPython files come from commit `0fb18b02c8ad56299d6a2910be0bab8ad601ef24`
(Python 3.12.0). Its complete LICENSE is archived as `raw/cpython_LICENSE`.
Copyright and license notices within the individual source files are retained.
The manifest records source URLs, authors, licenses, and exact archived hashes.

Capture uses the first 2,048 tokenizer tokens of each cleaned document, without
a chat template or added special tokens, as one causal prefill prefix. A Python
source prefix may contain docstrings and comments; it is not pure program code.
Replay retains each token's actual normalized hidden state, logical experts,
combining weights, source document, and original position. Domain switching and
token reuse are explicitly constructed; these are not natural request arrivals,
autoregressive decode traces, or representative samples of production traffic.
The finite document pairs are the comparison blocks. Additional random sampling
seeds or repeat executions of a pair do not create independent source blocks.
