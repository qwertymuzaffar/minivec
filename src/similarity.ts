import { dot, normalize } from './metrics';

/** Cosine similarity of two vectors, without needing a store. */
export function cosineSimilarity(a: ArrayLike<number>, b: ArrayLike<number>): number {
  if (a.length !== b.length || a.length === 0) {
    throw new RangeError(`cosineSimilarity: incompatible lengths ${a.length} and ${b.length}`);
  }
  const fa = normalize(a instanceof Float32Array ? a : Float32Array.from(a));
  const fb = normalize(b instanceof Float32Array ? b : Float32Array.from(b));
  return dot(fa, fb);
}
