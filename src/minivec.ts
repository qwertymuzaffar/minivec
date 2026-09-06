import { base64ToFloats, floatsToBase64 } from './b64';
import { Hnsw, type HnswState } from './hnsw';
import { distanceFn, distanceToScore, normalize, type Metric } from './metrics';

export interface MiniVecOptions {
  /** Vector dimensionality; every vector must match. */
  dim: number;
  /** 'cosine' (default), 'dot', or 'euclidean'. */
  metric?: Metric;
  /** HNSW: max links per node per layer (default 16). */
  M?: number;
  /** HNSW: build-time search width (default 200). */
  efConstruction?: number;
  /** Random source for level assignment (injectable for reproducibility). */
  random?: () => number;
}

export interface SearchOptions<M> {
  /** Number of results (default 10). */
  k?: number;
  /** Query-time search width; higher = better recall (default max(k, 50)). */
  ef?: number;
  /** Keep only records the predicate accepts. Raise ef for selective filters. */
  filter?: (meta: M | undefined, id: string) => boolean;
  /** Brute-force scan instead of the index - exact results, O(n). */
  exact?: boolean;
}

export interface SearchResult<M> {
  id: string;
  /** Higher is better for every metric (cosine sim, dot product, -L2). */
  score: number;
  meta?: M;
}

export interface MiniVecSnapshot {
  version: 1;
  dim: number;
  metric: Metric;
  M: number;
  efConstruction: number;
  ids: (string | null)[];
  metas: unknown[];
  deleted: number[];
  /** All vectors concatenated as one base64-encoded Float32Array. */
  vectors: string;
  hnsw: HnswState;
}

/**
 * Embedded vector store with exact and approximate (HNSW) search.
 * Runs anywhere JavaScript does - Node, browsers, edge - with zero
 * dependencies. Persistence via toJSON()/MiniVec.fromJSON().
 */
export class MiniVec<M = Record<string, unknown>> {
  private readonly dim: number;
  private readonly metric: Metric;
  private readonly M: number;
  private readonly efConstruction: number;

  private vectors: Float32Array[] = [];
  private ids: (string | null)[] = [];
  private metas: (M | undefined)[] = [];
  private deleted: boolean[] = [];
  private idToNode = new Map<string, number>();
  private live = 0;

  private readonly dist: (a: Float32Array, b: Float32Array) => number;
  private readonly graph: Hnsw;

  constructor(options: MiniVecOptions) {
    if (!Number.isInteger(options.dim) || options.dim < 1) {
      throw new RangeError(`dim must be a positive integer, got ${options.dim}`);
    }
    this.dim = options.dim;
    this.metric = options.metric ?? 'cosine';
    this.M = options.M ?? 16;
    this.efConstruction = options.efConstruction ?? 200;
    this.dist = distanceFn(this.metric);
    this.graph = new Hnsw(
      (a, b) => this.dist(this.vectors[a], this.vectors[b]),
      (q, node) => this.dist(q as Float32Array, this.vectors[node]),
      { M: this.M, efConstruction: this.efConstruction, random: options.random },
    );
  }

  get size(): number {
    return this.live;
  }

  /** Adds a record. An existing id is upserted (old entry tombstoned). */
  add(id: string, vector: ArrayLike<number>, meta?: M): void {
    if (vector.length !== this.dim) {
      throw new RangeError(`vector for "${id}" has length ${vector.length}, expected ${this.dim}`);
    }
    const existing = this.idToNode.get(id);
    if (existing !== undefined) this.tombstone(existing);

    const raw = vector instanceof Float32Array ? vector : Float32Array.from(vector);
    const stored = this.metric === 'cosine' ? normalize(raw) : raw.slice();
    const node = this.vectors.length;
    this.vectors.push(stored);
    this.ids.push(id);
    this.metas.push(meta);
    this.deleted.push(false);
    this.idToNode.set(id, node);
    this.live++;
    this.graph.insert(node);
  }

  get(id: string): { id: string; vector: Float32Array; meta?: M } | undefined {
    const node = this.idToNode.get(id);
    if (node === undefined) return undefined;
    return { id, vector: this.vectors[node], meta: this.metas[node] };
  }

  has(id: string): boolean {
    return this.idToNode.has(id);
  }

  /** Removes a record (tombstoned; the graph keeps routing through it). */
  remove(id: string): boolean {
    const node = this.idToNode.get(id);
    if (node === undefined) return false;
    this.tombstone(node);
    this.idToNode.delete(id);
    return true;
  }

  search(vector: ArrayLike<number>, options: SearchOptions<M> = {}): SearchResult<M>[] {
    if (vector.length !== this.dim) {
      throw new RangeError(`query vector has length ${vector.length}, expected ${this.dim}`);
    }
    const k = options.k ?? 10;
    if (this.live === 0 || k < 1) return [];
    const raw = vector instanceof Float32Array ? vector : Float32Array.from(vector);
    const query = this.metric === 'cosine' ? normalize(raw) : raw;

    if (options.exact) return this.exactSearch(query, k, options.filter);

    const ef = Math.max(options.ef ?? 50, k);
    const nodes = this.graph.search(query, ef);
    const out: SearchResult<M>[] = [];
    for (const node of nodes) {
      if (this.deleted[node]) continue;
      const id = this.ids[node]!;
      const meta = this.metas[node];
      if (options.filter && !options.filter(meta, id)) continue;
      out.push({ id, score: distanceToScore(this.metric, this.dist(query, this.vectors[node])), ...(meta !== undefined ? { meta } : {}) });
      if (out.length === k) break;
    }
    return out;
  }

  private exactSearch(query: Float32Array, k: number, filter?: (meta: M | undefined, id: string) => boolean): SearchResult<M>[] {
    const all: SearchResult<M>[] = [];
    for (let node = 0; node < this.vectors.length; node++) {
      if (this.deleted[node]) continue;
      const id = this.ids[node]!;
      const meta = this.metas[node];
      if (filter && !filter(meta, id)) continue;
      all.push({ id, score: distanceToScore(this.metric, this.dist(query, this.vectors[node])), ...(meta !== undefined ? { meta } : {}) });
    }
    all.sort((a, b) => b.score - a.score);
    return all.slice(0, k);
  }

  toJSON(): MiniVecSnapshot {
    const flat = new Float32Array(this.vectors.length * this.dim);
    this.vectors.forEach((v, i) => flat.set(v, i * this.dim));
    return {
      version: 1,
      dim: this.dim,
      metric: this.metric,
      M: this.M,
      efConstruction: this.efConstruction,
      ids: this.ids,
      metas: this.metas as unknown[],
      deleted: this.deleted.flatMap((d, i) => (d ? [i] : [])),
      vectors: floatsToBase64(flat),
      hnsw: this.graph.toState(),
    };
  }

  static fromJSON<M = Record<string, unknown>>(snapshot: MiniVecSnapshot): MiniVec<M> {
    if (snapshot.version !== 1) {
      throw new Error(`unsupported minivec snapshot version: ${snapshot.version}`);
    }
    const store = new MiniVec<M>({
      dim: snapshot.dim,
      metric: snapshot.metric,
      M: snapshot.M,
      efConstruction: snapshot.efConstruction,
    });
    const flat = base64ToFloats(snapshot.vectors);
    const count = snapshot.ids.length;
    const deletedSet = new Set(snapshot.deleted);
    for (let node = 0; node < count; node++) {
      store.vectors.push(flat.subarray(node * snapshot.dim, (node + 1) * snapshot.dim));
      store.ids.push(snapshot.ids[node]);
      store.metas.push(snapshot.metas[node] as M | undefined);
      const dead = deletedSet.has(node);
      store.deleted.push(dead);
      const id = snapshot.ids[node];
      if (!dead && id !== null) {
        store.idToNode.set(id, node);
        store.live++;
      }
    }
    store.graph.loadState(snapshot.hnsw);
    return store;
  }

  private tombstone(node: number): void {
    if (!this.deleted[node]) {
      this.deleted[node] = true;
      this.live--;
    }
  }
}
