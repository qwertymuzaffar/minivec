# minivec

[![npm version](https://img.shields.io/npm/v/minivec)](https://www.npmjs.com/package/minivec)
[![CI](https://github.com/qwertymuzaffar/minivec/actions/workflows/ci.yml/badge.svg)](https://github.com/qwertymuzaffar/minivec/actions/workflows/ci.yml)
[![license: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

Tiny embedded vector store with **HNSW approximate search implemented from scratch** - zero dependencies, no native bindings, runs anywhere JavaScript does: Node, browsers, edge runtimes.

On 20,000 x 384-dim vectors, HNSW answers queries **40-67x faster than exact search** at 99-100% recall@10 (see [Benchmarks](#benchmarks)).

## Install

```bash
npm i minivec
```

## Quick start

```ts
import { MiniVec } from 'minivec';

const store = new MiniVec<{ title: string }>({ dim: 384 });

store.add('doc-1', embedding1, { title: 'Getting started' });
store.add('doc-2', embedding2, { title: 'Deployment guide' });

const results = store.search(queryEmbedding, { k: 5 });
// [{ id: 'doc-2', score: 0.87, meta: { title: 'Deployment guide' } }, ...]
```

Scores are **higher-is-better for every metric**: cosine similarity, dot product, or negative L2 distance.

## API

### `new MiniVec(options)`

| Option | Default | Description |
|---|---|---|
| `dim` | required | Vector dimensionality |
| `metric` | `'cosine'` | `'cosine'` (inputs auto-normalized), `'dot'`, `'euclidean'` |
| `M` | `16` | HNSW: max links per node per layer |
| `efConstruction` | `200` | HNSW: build-time search width |

### Methods

| Method | Description |
|---|---|
| `add(id, vector, meta?)` | Insert; an existing id is upserted |
| `search(vector, { k, ef, filter, exact })` | Nearest neighbors, higher score first |
| `get(id)` / `has(id)` / `remove(id)` | Record access; `size` counts live records |
| `toJSON()` / `MiniVec.fromJSON(snapshot)` | Whole-store persistence (vectors base64-packed) |

### Search options

| Option | Default | Description |
|---|---|---|
| `k` | `10` | Number of results |
| `ef` | `max(k, 50)` | Search width - the recall/speed dial (see below) |
| `filter` | - | `(meta, id) => boolean`; raise `ef` for selective filters |
| `exact` | `false` | Brute-force scan: exact results, O(n) |

## How the index works

minivec implements the HNSW graph (Malkov & Yashunin, 2016): every vector becomes a node in a multi-layer proximity graph. Upper layers are sparse express lanes; a query greedily descends to the bottom layer, then runs a beam search of width `ef` among the candidates. Neighbor selection uses the paper's diversity heuristic, which spreads links across directions - the property that keeps high-dimensional graphs navigable.

Practical tuning:

- **`ef` (query time)** - the only dial most apps need. `16` is fast, `50+` is near-exact on realistic embedding data.
- **`M` / `efConstruction` (build time)** - raise for harder datasets (more clusters, higher intrinsic dimension) at the cost of memory and build speed.
- **`exact: true`** - the honest fallback for small stores (under ~2k vectors it is often just as fast).

Deletes are tombstones: removed records never appear in results, but the graph keeps routing through them until you rebuild (serialize live records into a fresh store).

## Persistence

`toJSON()` returns a plain JSON-safe snapshot with all vectors packed into one base64 string; `MiniVec.fromJSON()` restores the store *including* the built graph - no re-indexing on load.

```ts
// Node
import { writeFile, readFile } from 'node:fs/promises';
await writeFile('index.json', JSON.stringify(store.toJSON()));
const revived = MiniVec.fromJSON(JSON.parse(await readFile('index.json', 'utf8')));

// Browser (IndexedDB via idb-keyval, or any storage you like)
await set('index', store.toJSON());
const revived = MiniVec.fromJSON(await get('index'));
```

## Benchmarks

`npm run bench` - 20,000 vectors, 384 dimensions (all-MiniLM-L6-v2 size), cosine, clustered data mirroring real embedding structure, Apple silicon, Node 20:

| Search | Throughput | recall@10 |
|---|---|---|
| exact (baseline) | 54 qps | 1.000 |
| HNSW `ef=16` | 3,631 qps | 0.990 |
| HNSW `ef=50` | 2,165 qps | 1.000 |
| HNSW `ef=100` | 1,342 qps | 1.000 |

Build: ~288 adds/s at `efConstruction: 200`. Snapshot: 42 MB JSON for 20k x 384d.

A note on honesty: uniform random high-dimensional vectors are the ANN worst case (distance concentration) and no library does well on them; the clustered generator models what actual text/image embeddings look like. Run the bench on your own data for numbers that matter.

## Alternatives

- [hnswlib-node](https://www.npmjs.com/package/hnswlib-node) - native bindings to the reference C++ implementation: faster raw throughput, Node-only, requires compilation. minivec trades peak speed for zero dependencies and running in the browser.
- [vectra](https://www.npmjs.com/package/vectra) - file-based local vector store with exact search; no ANN index.
- A real vector database (pgvector, Qdrant, ...) - the right call beyond a few hundred thousand vectors or when you need multi-process access.

## License

MIT (c) Muzaffar Qosimov
