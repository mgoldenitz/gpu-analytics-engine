"""Hash group-by on the GPU: SUM(value) and COUNT(*) per distinct int32 key.

Open addressing with linear probing in one flat table, the same idea as
static_map in NVIDIA/cuCollections (include/cuco/static_map.cuh and its
linear_probing probing scheme). Our version is much smaller: one key type,
no cooperative groups, no window loads. The hash is the 32-bit finalizer
(fmix32) of MurmurHash3 by Austin Appleby.

strategy="block" first adds up rows in a per-block hash table in shared memory.
cuCollections puts a hash table in shared memory in
examples/static_set/shared_memory_example.cu; we do the same with a map. The
CUDA C++ Programming Guide describes atomic functions on shared memory
(section "Atomic Functions": they work on 32- and 64-bit words in global or
shared memory, and are atomic only within the thread block for shared memory).
"""
import operator

import cupy as cp

from engine.scan import _scan  # 0/1 flags: total <= n, so no int32 check (it costs a full pass)

THREADS = 256
EMPTY = -2**31  # INT_MIN marks an empty slot, so it cannot be a real key
# Capacity stays at or below 2^30, inside int32 scan positions. The table uses 24 bytes
# per slot, so the top of this range needs about 24 GB; group_by_sum_count checks free
# GPU memory first and raises MemoryError (about 2^27 groups fit on an 8 GB card).
MAX_GROUPS = 2**29
ROWS_PER_BLOCK = 4096  # 16 rows per thread, so a block can see more keys than SLOTS

_SOURCE = f"#define ROWS_PER_BLOCK {ROWS_PER_BLOCK}\n" + r'''
#define EMPTY (-2147483647 - 1)

// MurmurHash3 fmix32: mixes all 32 bits, so near keys land far apart.
__device__ unsigned int hash32(unsigned int h) {
  h ^= h >> 16;
  h *= 0x85ebca6bu;
  h ^= h >> 13;
  h *= 0xc2b2ae35u;
  h ^= h >> 16;
  return h;
}

// Add (sum, count) into key's slot of the global table. Used by both strategies.
__device__ void add_global(int key, unsigned long long sum, int count,
                           int* t_keys, unsigned long long* t_sums, int* t_counts,
                           unsigned int mask, int* overflow) {
  unsigned int slot = hash32((unsigned int)key) & mask;  // capacity is a power of two
  for (unsigned int probe = 0; probe <= mask; probe++) {
    int prev = atomicCAS(&t_keys[slot], EMPTY, key);
    if (prev == EMPTY || prev == key) {
      atomicAdd(&t_sums[slot], sum);
      atomicAdd(&t_counts[slot], count);
      return;
    }
    slot = (slot + 1) & mask;  // linear probing, wraps at the end
  }
  *overflow = 1;  // every slot holds another key
}

// Step 1, strategy="global": one thread per row adds its value into its key's slot.
extern "C" __global__ void insert(const int* keys, const int* vals, int n,
                                  int* t_keys, unsigned long long* t_sums, int* t_counts,
                                  unsigned int mask, int* overflow) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  add_global(keys[i], (unsigned long long)(long long)vals[i], 1,  // sign-extend
             t_keys, t_sums, t_counts, mask, overflow);
}

#define SLOTS 1024           // shared table per block: 1024 * (4 + 8 + 4) bytes = 16 KB
#define SHARED_PROBES 32     // after this many taken slots, the row goes to global

// Step 1, strategy="block": add up the block's rows in shared memory, then flush once per key.
extern "C" __global__ void insert_block(const int* keys, const int* vals, int n,
                                        int* t_keys, unsigned long long* t_sums, int* t_counts,
                                        unsigned int mask, int* overflow) {
  __shared__ int s_keys[SLOTS];
  __shared__ unsigned long long s_sums[SLOTS];
  __shared__ int s_counts[SLOTS];
  for (int s = threadIdx.x; s < SLOTS; s += blockDim.x) {
    s_keys[s] = EMPTY;
    s_sums[s] = 0;
    s_counts[s] = 0;
  }
  __syncthreads();  // barrier 1: the table is empty before any thread inserts

  int start = blockIdx.x * ROWS_PER_BLOCK;
  int end = min(start + ROWS_PER_BLOCK, n);
  for (int i = start + threadIdx.x; i < end; i += blockDim.x) {  // no early return: all reach barrier 2
    int key = keys[i];
    unsigned long long v = (unsigned long long)(long long)vals[i];  // sign-extend
    unsigned int slot = hash32((unsigned int)key) & (SLOTS - 1);
    bool placed = false;
    for (int probe = 0; probe < SHARED_PROBES; probe++) {
      int prev = atomicCAS(&s_keys[slot], EMPTY, key);
      if (prev == EMPTY || prev == key) {
        atomicAdd(&s_sums[slot], v);
        atomicAdd(&s_counts[slot], 1);
        placed = true;
        break;
      }
      slot = (slot + 1) & (SLOTS - 1);
    }
    if (!placed) add_global(key, v, 1, t_keys, t_sums, t_counts, mask, overflow);  // fallback
  }
  __syncthreads();  // barrier 2: every shared add is done before any thread reads a slot

  for (int s = threadIdx.x; s < SLOTS; s += blockDim.x)  // flush: one global update per key
    if (s_keys[s] != EMPTY)
      add_global(s_keys[s], s_sums[s], s_counts[s], t_keys, t_sums, t_counts, mask, overflow);
}

// Step 2, flag: same as the filter predicate, flags[s] = 1 if slot s is used.
extern "C" __global__ void flag_used(const int* t_keys, int* flags, int cap) {
  int s = blockIdx.x * blockDim.x + threadIdx.x;
  if (s < cap) flags[s] = t_keys[s] != EMPTY;
}

// Step 4, scatter: each used slot writes its three columns to position pos[s].
extern "C" __global__ void scatter(const int* t_keys, const unsigned long long* t_sums,
                                   const int* t_counts, const int* flags, const int* pos,
                                   int* out_keys, unsigned long long* out_sums,
                                   long long* out_counts, int cap) {
  int s = blockIdx.x * blockDim.x + threadIdx.x;
  if (s < cap && flags[s]) {
    int p = pos[s];
    out_keys[p] = t_keys[s];
    out_sums[p] = t_sums[s];
    out_counts[p] = t_counts[s];
  }
}
'''

_module = cp.RawModule(code=_SOURCE)
_insert = _module.get_function('insert')
_insert_block = _module.get_function('insert_block')
_flag_used = _module.get_function('flag_used')
_scatter = _module.get_function('scatter')


def _grid(n):
    return ((n + THREADS - 1) // THREADS,)


def group_by_sum_count(keys, values, expected_groups, strategy="global"):
    """Return (keys int32, sums int64, counts int64), one row per distinct key, in no order.

    expected_groups sets the table size: the next power of two >= 2 * expected_groups.
    strategy="global" does one global atomic update per row; "block" adds up each block's
    rows in shared memory first, then does one global update per key per block.
    Raises RuntimeError if the input has more distinct keys than the table holds.
    """
    if strategy not in ("global", "block"):
        raise ValueError('strategy must be "global" or "block"')
    for name, a in (("keys", keys), ("values", values)):
        if not isinstance(a, cp.ndarray) or a.dtype != cp.int32 or a.ndim != 1:
            raise TypeError(f"{name} must be a 1-D cupy int32 array")
    if keys.size != values.size:
        raise ValueError("keys and values must have the same length")
    expected_groups = operator.index(expected_groups)  # np.int64 has no .bit_length()
    if not 1 <= expected_groups <= MAX_GROUPS:
        raise ValueError(f"expected_groups must be in 1..{MAX_GROUPS}")
    keys = cp.ascontiguousarray(keys)
    values = cp.ascontiguousarray(values)
    n = keys.size
    if n == 0:
        return cp.empty(0, cp.int32), cp.empty(0, cp.int64), cp.empty(0, cp.int64)
    if bool((keys == EMPTY).any()):
        raise ValueError("keys must not contain INT_MIN (it marks empty slots)")

    cap = 1 << (2 * expected_groups - 1).bit_length()
    # 24 bytes per slot: table keys, sums, counts (16) plus scan flags and positions (8).
    need = cap * 24
    free = cp.cuda.Device().mem_info[0] + cp.get_default_memory_pool().free_bytes()
    if need > free:
        raise MemoryError(f"expected_groups={expected_groups:,} needs {need / 2**30:.1f} GB "
                          f"of GPU memory for the table; {free / 2**30:.1f} GB is free")
    t_keys = cp.full(cap, EMPTY, dtype=cp.int32)
    t_sums = cp.zeros(cap, dtype=cp.uint64)
    t_counts = cp.zeros(cap, dtype=cp.int32)
    overflow = cp.zeros(1, dtype=cp.int32)
    args = (keys, values, cp.int32(n), t_keys, t_sums, t_counts, cp.uint32(cap - 1), overflow)
    if strategy == "global":
        _insert(_grid(n), (THREADS,), args)
    else:
        _insert_block(((n + ROWS_PER_BLOCK - 1) // ROWS_PER_BLOCK,), (THREADS,), args)
    if int(overflow[0]):
        raise RuntimeError(f"hash table full: more than {cap} distinct keys; "
                           "raise expected_groups")

    # Steps 2-4: compact the used slots, as in engine/filter.py with scan="own".
    flags = cp.empty(cap, dtype=cp.int32)
    _flag_used(_grid(cap), (THREADS,), (t_keys, flags, cp.int32(cap)))
    pos = _scan(flags)
    count = int(pos[-1] + flags[-1])

    out_keys = cp.empty(count, dtype=cp.int32)
    out_sums = cp.empty(count, dtype=cp.uint64)
    out_counts = cp.empty(count, dtype=cp.int64)
    _scatter(_grid(cap), (THREADS,), (t_keys, t_sums, t_counts, flags, pos,
                                      out_keys, out_sums, out_counts, cp.int32(cap)))
    return out_keys, out_sums.view(cp.int64), out_counts  # two's complement: same bits
