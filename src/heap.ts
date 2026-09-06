/** Binary min-heap keyed by number; pop() returns the smallest key. */
export class MinHeap<T> {
  private keys: number[] = [];
  private values: T[] = [];

  get size(): number {
    return this.keys.length;
  }

  peekKey(): number {
    return this.keys[0];
  }

  push(key: number, value: T): void {
    const { keys, values } = this;
    let i = keys.length;
    keys.push(key);
    values.push(value);
    while (i > 0) {
      const parent = (i - 1) >> 1;
      if (keys[parent] <= keys[i]) break;
      this.swap(i, parent);
      i = parent;
    }
  }

  pop(): { key: number; value: T } | undefined {
    const { keys, values } = this;
    if (keys.length === 0) return undefined;
    const top = { key: keys[0], value: values[0] };
    const lastKey = keys.pop()!;
    const lastValue = values.pop()!;
    if (keys.length > 0) {
      keys[0] = lastKey;
      values[0] = lastValue;
      let i = 0;
      for (;;) {
        const left = 2 * i + 1;
        const right = left + 1;
        let smallest = i;
        if (left < keys.length && keys[left] < keys[smallest]) smallest = left;
        if (right < keys.length && keys[right] < keys[smallest]) smallest = right;
        if (smallest === i) break;
        this.swap(i, smallest);
        i = smallest;
      }
    }
    return top;
  }

  private swap(a: number, b: number): void {
    [this.keys[a], this.keys[b]] = [this.keys[b], this.keys[a]];
    [this.values[a], this.values[b]] = [this.values[b], this.values[a]];
  }
}
