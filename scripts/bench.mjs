import { MiniVec } from '../dist/index.js';

function rng(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

const N = 20000;
const DIM = 384; // all-MiniLM-L6-v2 size
const QUERIES = 200;
const K = 10;

const random = rng(1234);
// Clustered vectors (mixture around 200 centers) - mirrors the manifold
// structure of real text/image embeddings. Uniform random high-dim data is
// the known ANN worst case (distance concentration) and represents no real
// workload.
const CENTERS = 200;
const centers = Array.from({ length: CENTERS }, () => {
  const c = new Float32Array(DIM);
  for (let i = 0; i < DIM; i++) c[i] = random() * 2 - 1;
  return c;
});
const vec = () => {
  const c = centers[Math.floor(random() * CENTERS)];
  const v = new Float32Array(DIM);
  for (let i = 0; i < DIM; i++) v[i] = c[i] + (random() * 2 - 1) * 0.35;
  return v;
};

console.log(`minivec bench - N=${N}, dim=${DIM}, k=${K}, ${QUERIES} queries, cosine, clustered`);

const store = new MiniVec({ dim: DIM, random: rng(99) });
const t0 = performance.now();
for (let i = 0; i < N; i++) store.add(`v${i}`, vec());
const buildMs = performance.now() - t0;
console.log(`build: ${(buildMs / 1000).toFixed(1)}s (${(N / (buildMs / 1000)).toFixed(0)} adds/s)`);

const queries = Array.from({ length: QUERIES }, vec);

const tExact = performance.now();
const exactResults = queries.map((q) => store.search(q, { k: K, exact: true }).map((r) => r.id));
const exactMs = performance.now() - tExact;
console.log(`exact:      ${(QUERIES / (exactMs / 1000)).toFixed(0).padStart(6)} qps (baseline)`);

for (const ef of [16, 50, 100, 200]) {
  const t = performance.now();
  const results = queries.map((q) => store.search(q, { k: K, ef }).map((r) => r.id));
  const ms = performance.now() - t;
  let hits = 0;
  for (let i = 0; i < QUERIES; i++) {
    const truth = new Set(exactResults[i]);
    hits += results[i].filter((id) => truth.has(id)).length;
  }
  const recall = hits / (QUERIES * K);
  console.log(`hnsw ef=${String(ef).padEnd(3)} ${(QUERIES / (ms / 1000)).toFixed(0).padStart(6)} qps  recall@${K}=${recall.toFixed(3)}`);
}

const snapshotBytes = JSON.stringify(store.toJSON()).length;
console.log(`snapshot: ${(snapshotBytes / 1024 / 1024).toFixed(1)} MB JSON for ${N} x ${DIM}d`);
