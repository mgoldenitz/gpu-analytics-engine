"""Hash group-by on the GPU: SUM(value) and COUNT(*) per distinct int32 key.

Open addressing with linear probing in one flat table, the same idea as
static_map in NVIDIA/cuCollections (include/cuco/static_map.cuh and its
linear_probing probing scheme). Our version is much smaller: one key type,
no cooperative groups, no window loads. The hash is the 32-bit finalizer
(fmix32) of MurmurHash3 by Austin Appleby.
"""
import cupy as cp

from engine.scan import exclusive_scan

THREADS = 256
EMPTY = -2**31  # INT_MIN marks an empty slot, so it cannot be a real key
MAX_GROUPS = 2**29  # capacity stays at or below 2^30, inside int32 scan positions

_SOURCE = r'''
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

// Step 1, insert: one thread per row adds its value into its key's slot.
extern "C" __global__ void insert(const int* keys, const int* vals, int n,
                                  int* t_keys, unsigned long long* t_sums, int* t_counts,
                                  unsigned int mask, int* overflow) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  int key = keys[i];
  unsigned int slot = hash32((unsigned int)key) & mask;  // capacity is a power of two
  for (unsigned int probe = 0; probe <= mask; probe++) {
    int prev = atomicCAS(&t_keys[slot], EMPTY, key);
    if (prev == EMPTY || prev == key) {
      atomicAdd(&t_sums[slot], (unsigned long long)(long long)vals[i]);  // sign-extend
      atomicAdd(&t_counts[slot], 1);
      return;
    }
    slot = (slot + 1) & mask;  // linear probing, wraps at the end
  }
  *overflow = 1;  // every slot holds another key
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
_flag_used = _module.get_function('flag_used')
_scatter = _module.get_function('scatter')


def _grid(n):
    return ((n + THREADS - 1) // THREADS,)


def group_by_sum_count(keys, values, expected_groups):
    """Return (keys int32, sums int64, counts int64), one row per distinct key, in no order.

    expected_groups sets the table size: the next power of two >= 2 * expected_groups.
    Raises RuntimeError if the input has more distinct keys than the table holds.
    """
    for name, a in (("keys", keys), ("values", values)):
        if not isinstance(a, cp.ndarray) or a.dtype != cp.int32 or a.ndim != 1:
            raise TypeError(f"{name} must be a 1-D cupy int32 array")
    if keys.size != values.size:
        raise ValueError("keys and values must have the same length")
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
    t_keys = cp.full(cap, EMPTY, dtype=cp.int32)
    t_sums = cp.zeros(cap, dtype=cp.uint64)
    t_counts = cp.zeros(cap, dtype=cp.int32)
    overflow = cp.zeros(1, dtype=cp.int32)
    _insert(_grid(n), (THREADS,), (keys, values, cp.int32(n), t_keys, t_sums, t_counts,
                                   cp.uint32(cap - 1), overflow))
    if int(overflow[0]):
        raise RuntimeError(f"hash table full: more than {cap} distinct keys; "
                           "raise expected_groups")

    # Steps 2-4: compact the used slots, as in engine/filter.py with scan="own".
    flags = cp.empty(cap, dtype=cp.int32)
    _flag_used(_grid(cap), (THREADS,), (t_keys, flags, cp.int32(cap)))
    pos = exclusive_scan(flags)
    count = int(pos[-1] + flags[-1])

    out_keys = cp.empty(count, dtype=cp.int32)
    out_sums = cp.empty(count, dtype=cp.uint64)
    out_counts = cp.empty(count, dtype=cp.int64)
    _scatter(_grid(cap), (THREADS,), (t_keys, t_sums, t_counts, flags, pos,
                                      out_keys, out_sums, out_counts, cp.int32(cap)))
    return out_keys, out_sums.view(cp.int64), out_counts  # two's complement: same bits
