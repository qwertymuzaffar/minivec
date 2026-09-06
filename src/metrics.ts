export type Metric = 'cosine' | 'dot' | 'euclidean';

export function dot(a: Float32Array, b: Float32Array): number {
  let sum = 0;
  for (let i = 0; i < a.length; i++) sum += a[i] * b[i];
  return sum;
}

export function l2Squared(a: Float32Array, b: Float32Array): number {
  let sum = 0;
  for (let i = 0; i < a.length; i++) {
    const d = a[i] - b[i];
    sum += d * d;
  }
  return sum;
}

/** Returns a normalized copy; a zero vector is returned unchanged. */
export function normalize(v: Float32Array): Float32Array {
  let sum = 0;
  for (let i = 0; i < v.length; i++) sum += v[i] * v[i];
  const norm = Math.sqrt(sum);
  if (norm === 0) return v;
  const out = new Float32Array(v.length);
  for (let i = 0; i < v.length; i++) out[i] = v[i] / norm;
  return out;
}

/**
 * Distance for graph traversal: smaller is closer.
 * cosine assumes vectors were normalized on insert (1 - dot).
 */
export function distanceFn(metric: Metric): (a: Float32Array, b: Float32Array) => number {
  switch (metric) {
    case 'cosine':
      return (a, b) => 1 - dot(a, b);
    case 'dot':
      return (a, b) => -dot(a, b);
    case 'euclidean':
      return l2Squared;
  }
}

/** User-facing score: higher is better, for every metric. */
export function distanceToScore(metric: Metric, distance: number): number {
  switch (metric) {
    case 'cosine':
      return 1 - distance; // cosine similarity
    case 'dot':
      return -distance; // the dot product
    case 'euclidean':
      return -Math.sqrt(Math.max(0, distance)); // negative L2 distance
  }
}
