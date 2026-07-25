# Standalone MuSiQue Late Chunking

This folder contains the native Late Chunking implementation plus the complete
frozen MuSiQue 60K experiment. It can be copied away from SAADI and includes 66
expanded validation contexts, exact group membership, and 300 queries for each
of 2-hop, 3-hop, and 4-hop.

The implementation keeps the method's native sequence:

1. create fixed token chunks and retain their token spans;
2. encode the largest permitted contextual window;
3. mean-pool contextual token embeddings over each chunk span;
4. retrieve ten chunks within the query's expanded group;
5. generate independent gold/silver labels and evaluate at 5 and 10.

Documents larger than a model forward pass are windowed with the configured
overlap; they are not silently truncated.

## Setup and validation

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
VALIDATE_ONLY=1 bash run_musique_late_chunking.sh
```

## Historical Late Chunking grid

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 \
bash run_musique_late_chunking.sh
```

Defaults:

- hops: `2,3,4`;
- retrievers: `jina,qwen`;
- chunk sizes: `200,300,500`;
- overlaps: `0,50,100`;
- maximum tokens per encoder forward: 8192;
- encoder-window overlap: 256;
- retrieval depth: 10.

For only the canonical MuSiQue 250/0 comparison:

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 \
CHUNK_SIZES_CSV=250 \
OVERLAPS_CSV=0 \
bash run_musique_late_chunking.sh
```

Completed retrieval and evaluation artifacts are skipped unless
`FORCE_RERUN=1` is set. Outputs are local:

```text
late_chunk_runs/
late_chunk_evaluations/
tables/late_chunking_mega_table.txt
```

Supporting paragraphs remain independent label targets. A paragraph contained
in one chunk is gold; a paragraph spanning chunks becomes a silver group.
Evidence paragraphs from different locations are never joined into an
artificial envelope.
