import { MiniVec, cosineSimilarity } from './index';
import { MinHeap } from './heap';
import { base64ToFloats, floatsToBase64 } from './b64';
import { distanceToScore, dot, l2Squared, normalize } from './metrics';

/** Deterministic RNG (mulberry32) for reproducible index construction. */
function rng(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function randomVectors(count: number, dim: number, random: () => number): Float32Array[] {
  return Array.from({ length: count }, () => {
    const v = new Float32Array(dim);
    for (let i = 0; i < dim; i++) v[i] = random() * 2 - 1;
    return v;
  });
}

describe('MinHeap', () => {
  it('pops in ascending key order', () => {
    const heap = new MinHeap<string>();
    const keys = [5, 1, 4, 1.5, 9, 0, 7];
    for (const k of keys) heap.push(k, `v${k}`);
    const popped: number[] = [];
    for (let item = heap.pop(); item; item = heap.pop()) popped.push(item.key);
    expect(popped).toEqual([...keys].sort((a, b) => a - b));
    expect(heap.pop()).toBeUndefined();
  });
});

describe('metrics', () => {
  it('dot, l2, normalize behave as expected', () => {
    const a = Float32Array.from([3, 4]);
    const b = Float32Array.from([1, 0]);
    expect(dot(a, b)).toBe(3);
    expect(l2Squared(a, b)).toBe(20);
    const n = normalize(a);
    expect(n[0]).toBeCloseTo(0.6, 5);
    expect(n[1]).toBeCloseTo(0.8, 5);
    const zero = Float32Array.from([0, 0]);
    expect(normalize(zero)).toBe(zero);
  });

  it('distanceToScore gives higher-is-better for every metric', () => {
    expect(distanceToScore('cosine', 0)).toBe(1);
    expect(distanceToScore('dot', -5)).toBe(5);
    expect(distanceToScore('euclidean', 4)).toBe(-2);
  });

  it('cosineSimilarity helper matches expectations', () => {
    expect(cosineSimilarity([1, 0], [1, 0])).toBeCloseTo(1, 6);
    expect(cosineSimilarity([1, 0], [0, 1])).toBeCloseTo(0, 6);
    expect(() => cosineSimilarity([1], [1, 2])).toThrow(RangeError);
  });
});

describe('base64 vectors', () => {
  it('roundtrips Float32Arrays exactly', () => {
    const v = Float32Array.from([0.25, -1.5, 3.14159, 1e-7, 42]);
    expect(Array.from(base64ToFloats(floatsToBase64(v)))).toEqual(Array.from(v));
  });
});

describe('MiniVec basics', () => {
  it('validates options and vector dimensions', () => {
    expect(() => new MiniVec({ dim: 0 })).toThrow(RangeError);
    const store = new MiniVec({ dim: 3 });
    expect(() => store.add('a', [1, 2])).toThrow(/length 2, expected 3/);
    expect(() => store.search([1, 2])).toThrow(RangeError);
  });

  it('add/get/has/remove/size work and empty search returns []', () => {
    const store = new MiniVec<{ tag: string }>({ dim: 2 });
    expect(store.search([1, 0])).toEqual([]);
    store.add('a', [1, 0], { tag: 'x' });
    store.add('b', [0, 1]);
    expect(store.size).toBe(2);
    expect(store.has('a')).toBe(true);
    expect(store.get('a')?.meta).toEqual({ tag: 'x' });
    expect(store.remove('a')).toBe(true);
    expect(store.remove('a')).toBe(false);
    expect(store.size).toBe(1);
    expect(store.has('a')).toBe(false);
  });

  it('finds nearest neighbors with scores (cosine)', () => {
    const store = new MiniVec({ dim: 2, random: rng(1) });
    store.add('east', [1, 0]);
    store.add('north', [0, 1]);
    store.add('northeast', [1, 1]);
    const results = store.search([0.9, 0.1], { k: 2 });
    expect(results.map((r) => r.id)).toEqual(['east', 'northeast']);
    expect(results[0].score).toBeGreaterThan(results[1].score);
    expect(results[0].score).toBeLessThanOrEqual(1.0001);
  });

  it('supports dot and euclidean metrics with higher-is-better scores', () => {
    const dotStore = new MiniVec({ dim: 2, metric: 'dot', random: rng(2) });
    dotStore.add('big', [10, 0]);
    dotStore.add('small', [1, 0]);
    expect(dotStore.search([1, 0], { k: 2 }).map((r) => r.id)).toEqual(['big', 'small']);

    const l2 = new MiniVec({ dim: 1, metric: 'euclidean', random: rng(3) });
    l2.add('near', [1]);
    l2.add('far', [10]);
    const res = l2.search([0], { k: 2 });
    expect(res.map((r) => r.id)).toEqual(['near', 'far']);
    expect(res[0].score).toBeCloseTo(-1, 5);
  });

  it('tombstoned records never come back from search', () => {
    const store = new MiniVec({ dim: 2, random: rng(4) });
    store.add('keep', [1, 0]);
    store.add('drop', [0.99, 0.01]);
    store.remove('drop');
    const results = store.search([1, 0], { k: 10 });
    expect(results.map((r) => r.id)).toEqual(['keep']);
  });

  it('upserts an existing id', () => {
    const store = new MiniVec<{ v: number }>({ dim: 2, random: rng(5) });
    store.add('x', [1, 0], { v: 1 });
    store.add('x', [0, 1], { v: 2 });
    expect(store.size).toBe(1);
    expect(store.get('x')?.meta).toEqual({ v: 2 });
    expect(store.search([0, 1], { k: 1 })[0].id).toBe('x');
    expect(store.search([0, 1], { k: 1 })[0].score).toBeCloseTo(1, 5);
  });

  it('applies metadata filters', () => {
    const store = new MiniVec<{ lang: string }>({ dim: 2, random: rng(6) });
    store.add('a', [1, 0], { lang: 'en' });
    store.add('b', [0.99, 0.14], { lang: 'de' });
    store.add('c', [0.98, 0.2], { lang: 'en' });
    const results = store.search([1, 0], { k: 5, filter: (meta) => meta?.lang === 'en' });
    expect(results.map((r) => r.id)).toEqual(['a', 'c']);
  });
});

describe('HNSW vs exact search', () => {
  it('achieves high recall@10 on random vectors', () => {
    const random = rng(42);
    const dim = 32;
    const store = new MiniVec({ dim, random: rng(7) });
    const vectors = randomVectors(2000, dim, random);
    vectors.forEach((v, i) => store.add(`v${i}`, v));

    const queries = randomVectors(50, dim, random);
    let hits = 0;
    let total = 0;
    for (const q of queries) {
      const exact = store.search(q, { k: 10, exact: true }).map((r) => r.id);
      const approx = store.search(q, { k: 10, ef: 100 }).map((r) => r.id);
      const exactSet = new Set(exact);
      hits += approx.filter((id) => exactSet.has(id)).length;
      total += exact.length;
    }
    const recall = hits / total;
    expect(recall).toBeGreaterThan(0.9);
  });

  it('exact search returns perfectly ordered results', () => {
    const random = rng(8);
    const dim = 8;
    const store = new MiniVec({ dim, random: rng(9) });
    const vectors = randomVectors(300, dim, random);
    vectors.forEach((v, i) => store.add(`v${i}`, v));
    const q = randomVectors(1, dim, random)[0];
    const results = store.search(q, { k: 20, exact: true });
    for (let i = 1; i < results.length; i++) {
      expect(results[i - 1].score).toBeGreaterThanOrEqual(results[i].score);
    }
  });
});

describe('serialization', () => {
  it('roundtrips through JSON with identical search results', () => {
    const random = rng(10);
    const dim = 16;
    const store = new MiniVec<{ n: number }>({ dim, random: rng(11) });
    randomVectors(500, dim, random).forEach((v, i) => store.add(`v${i}`, v, { n: i }));
    store.remove('v42');

    const revived = MiniVec.fromJSON<{ n: number }>(JSON.parse(JSON.stringify(store.toJSON())));
    expect(revived.size).toBe(store.size);
    expect(revived.has('v42')).toBe(false);
    expect(revived.get('v7')?.meta).toEqual({ n: 7 });

    const q = randomVectors(1, dim, random)[0];
    const a = store.search(q, { k: 15, ef: 80 });
    const b = revived.search(q, { k: 15, ef: 80 });
    expect(b).toEqual(a);
  });

  it('rejects unknown snapshot versions', () => {
    const snapshot = new MiniVec({ dim: 2 }).toJSON();
    expect(() => MiniVec.fromJSON({ ...snapshot, version: 2 as 1 })).toThrow(/version/);
  });
});
