import { MinHeap } from './heap';

export interface HnswOptions {
  /** Max links per node per layer (level 0 allows 2M). Default 16. */
  M: number;
  /** Search width while building. Higher = better graph, slower add. Default 200. */
  efConstruction: number;
  /** Random source (injectable for reproducible tests). */
  random?: () => number;
}

export interface HnswState {
  entry: number;
  maxLevel: number;
  levels: number[];
  /** links[node][level] = neighbor node indexes. */
  links: number[][][];
}

/**
 * Hierarchical Navigable Small World graph (Malkov & Yashunin 2016),
 * implemented from scratch over integer node ids. Vectors and deletion
 * state live with the caller; this class owns only the graph.
 */
export class Hnsw {
  private readonly M: number;
  private readonly efConstruction: number;
  private readonly levelMult: number;
  private readonly random: () => number;

  entry = -1;
  maxLevel = -1;
  /** Top layer of each node. */
  levels: number[] = [];
  /** links[node][level] = neighbor node indexes. */
  links: number[][][] = [];

  constructor(
    private readonly distance: (a: number, b: number) => number,
    private readonly distanceToQuery: (q: unknown, node: number) => number,
    options: HnswOptions,
  ) {
    this.M = options.M;
    this.efConstruction = options.efConstruction;
    this.levelMult = 1 / Math.log(options.M);
    this.random = options.random ?? Math.random;
  }

  private capacity(level: number): number {
    return level === 0 ? this.M * 2 : this.M;
  }

  /** Inserts a node index that the caller has already registered a vector for. */
  insert(node: number): void {
    const level = Math.floor(-Math.log(Math.max(this.random(), 1e-12)) * this.levelMult);
    this.levels[node] = level;
    this.links[node] = Array.from({ length: level + 1 }, () => []);

    if (this.entry === -1) {
      this.entry = node;
      this.maxLevel = level;
      return;
    }

    let ep = this.entry;
    for (let lc = this.maxLevel; lc > level; lc--) {
      ep = this.greedyClosest(node, ep, lc);
    }

    for (let lc = Math.min(level, this.maxLevel); lc >= 0; lc--) {
      const nearest = this.searchLayerByNode(node, ep, this.efConstruction, lc);
      const neighbors = this.selectNeighbors(node, nearest, this.M);
      this.links[node][lc] = neighbors.slice();
      for (const neighbor of neighbors) {
        const list = this.links[neighbor][lc];
        list.push(node);
        if (list.length > this.capacity(lc)) {
          this.links[neighbor][lc] = this.selectNeighbors(neighbor, list, this.capacity(lc));
        }
      }
      if (nearest.length > 0) ep = nearest[0];
    }

    if (level > this.maxLevel) {
      this.maxLevel = level;
      this.entry = node;
    }
  }

  /**
   * Diversity heuristic from the HNSW paper (Algorithm 4): a candidate is
   * selected only if it is closer to the base node than to every neighbor
   * already selected - spreading links across directions instead of
   * clustering them, which is what keeps high-dimensional graphs
   * navigable. Rejected candidates backfill any remaining capacity.
   */
  private selectNeighbors(base: number, candidates: number[], max: number): number[] {
    if (candidates.length <= max) return candidates.slice();
    const sorted = candidates
      .map((c) => ({ c, d: this.distance(base, c) }))
      .sort((a, b) => a.d - b.d);
    const selected: { c: number; d: number }[] = [];
    const rejected: number[] = [];
    for (const item of sorted) {
      if (selected.length >= max) break;
      let diverse = true;
      for (const s of selected) {
        if (this.distance(item.c, s.c) < item.d) {
          diverse = false;
          break;
        }
      }
      if (diverse) selected.push(item);
      else rejected.push(item.c);
    }
    const out = selected.map((s) => s.c);
    for (const r of rejected) {
      if (out.length >= max) break;
      out.push(r);
    }
    return out;
  }

  /** Greedy descent step used above the target layer. */
  private greedyClosest(node: number, ep: number, level: number): number {
    let current = ep;
    let currentDist = this.distance(node, current);
    for (;;) {
      let improved = false;
      for (const neighbor of this.links[current][level]) {
        const d = this.distance(node, neighbor);
        if (d < currentDist) {
          current = neighbor;
          currentDist = d;
          improved = true;
        }
      }
      if (!improved) return current;
    }
  }

  /** Beam search on one layer for an existing node (build time). */
  private searchLayerByNode(node: number, ep: number, ef: number, level: number): number[] {
    return this.beam((other) => this.distance(node, other), ep, ef, level);
  }

  /** Beam search on one layer for an external query (query time). */
  searchLayer(query: unknown, ep: number, ef: number, level: number): number[] {
    return this.beam((node) => this.distanceToQuery(query, node), ep, ef, level);
  }

  /** Query-time entry: descend greedily to layer 1, beam at layer 0. */
  search(query: unknown, ef: number): number[] {
    if (this.entry === -1) return [];
    let ep = this.entry;
    let epDist = this.distanceToQuery(query, ep);
    for (let lc = this.maxLevel; lc >= 1; lc--) {
      for (;;) {
        let improved = false;
        for (const neighbor of this.links[ep][lc]) {
          const d = this.distanceToQuery(query, neighbor);
          if (d < epDist) {
            ep = neighbor;
            epDist = d;
            improved = true;
          }
        }
        if (!improved) break;
      }
    }
    return this.beam((node) => this.distanceToQuery(query, node), ep, ef, 0);
  }

  private beam(dist: (node: number) => number, ep: number, ef: number, level: number): number[] {
    const visited = new Set<number>([ep]);
    const candidates = new MinHeap<number>(); // closest first
    const results = new MinHeap<number>(); // negated dist: worst result on top
    const epDist = dist(ep);
    candidates.push(epDist, ep);
    results.push(-epDist, ep);

    while (candidates.size > 0) {
      const current = candidates.pop()!;
      const worst = -results.peekKey();
      if (current.key > worst && results.size >= ef) break;
      for (const neighbor of this.links[current.value][level]) {
        if (visited.has(neighbor)) continue;
        visited.add(neighbor);
        const d = dist(neighbor);
        if (results.size < ef || d < -results.peekKey()) {
          candidates.push(d, neighbor);
          results.push(-d, neighbor);
          if (results.size > ef) results.pop();
        }
      }
    }

    const out: { node: number; d: number }[] = [];
    for (let item = results.pop(); item; item = results.pop()) {
      out.push({ node: item.value, d: -item.key });
    }
    out.sort((a, b) => a.d - b.d);
    return out.map((x) => x.node);
  }

  toState(): HnswState {
    return { entry: this.entry, maxLevel: this.maxLevel, levels: this.levels, links: this.links };
  }

  loadState(state: HnswState): void {
    this.entry = state.entry;
    this.maxLevel = state.maxLevel;
    this.levels = state.levels;
    this.links = state.links;
  }
}
