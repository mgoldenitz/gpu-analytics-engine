"""TPC-H Q1 in one fused kernel: filter on shipdate, then sum 6 aggregates for each group.

The group id (returnflag, linestatus) has only 6 possible values, known before the run, so
each block keeps its sums in shared memory, indexed directly by group id. No hash table.
This is the block pre-aggregation idea of insert_block in engine/groupby.py.

Input: the encoded columns of benchmarks/tpch_data.encode(), as cupy arrays on the GPU.
Units: price in cents, qty/disc/tax in hundredths. So sum_disc_price = price*(100-disc)
has 4 decimals and sum_charge = price*(100-disc)*(100+tax) has 6, the same scales DuckDB
prints. The 3 averages are computed on the host from the sums.
"""
import cupy as cp
import numpy as np

GROUPS = 6
AGGS = 6  # count, sum_qty, sum_base_price, sum_disc_price, sum_charge, sum_disc
ROWS_PER_BLOCK = 4096  # same as engine/groupby.py
THREADS = 256
NAMES = [(r, s) for r in "ANR" for s in "FO"]  # index = group id (tpch_data.encode)
DTYPES = {"price": np.int64, "qty": np.int16, "disc": np.int16, "tax": np.int16,
          "ship": np.int32, "gid": np.uint8}

_SOURCE = r'''
#define SLOTS 36
#define GROUPS 6
#define AGGS 6
#define ROWS_PER_BLOCK 4096

extern "C" __global__
void q1_v1(const long long* price, const short* qty, const short* disc, const short* tax,
              const int* ship, const unsigned char* gid, long long n, int cutoff,
              unsigned long long* out)
{
    __shared__ unsigned long long acc[SLOTS];
    for (int s = threadIdx.x; s < SLOTS; s += blockDim.x) acc[s] = 0;
    __syncthreads();

    long long start = (long long)blockIdx.x * ROWS_PER_BLOCK;
    long long end = min(start + ROWS_PER_BLOCK, n);
    for (long long i = start + threadIdx.x; i < end; i += blockDim.x) {
        if (ship[i] > cutoff) continue;
        unsigned long long p = (unsigned long long)price[i];
        unsigned long long dp = p * (unsigned long long)(100 - disc[i]);
        unsigned long long* a = acc + gid[i] * AGGS;
        atomicAdd(a + 0, 1ULL);
        atomicAdd(a + 1, (unsigned long long)qty[i]);
        atomicAdd(a + 2, p);
        atomicAdd(a + 3, dp);
        atomicAdd(a + 4, dp * (unsigned long long)(100 + tax[i]));
        atomicAdd(a + 5, (unsigned long long)disc[i]);
    }
    __syncthreads();

    for (int s = threadIdx.x; s < SLOTS; s += blockDim.x)
        if (acc[s] != 0) atomicAdd(out + s, acc[s]);
}

// v2: each thread sums its rows in registers, then warps and blocks combine the sums.
// v1 did 6 shared atomics per passing row; v2 does 36 per warp and 36 global per block.
extern "C" __global__
void q1_v2(const long long* price, const short* qty, const short* disc, const short* tax,
           const int* ship, const unsigned char* gid, long long n, int cutoff,
           unsigned long long* out)
{
    __shared__ unsigned long long acc[SLOTS];
    for (int s = threadIdx.x; s < SLOTS; s += blockDim.x) acc[s] = 0;
    __syncthreads();

    unsigned long long r[SLOTS];  // registers: the unrolled loops below use fixed indexes
    #pragma unroll
    for (int s = 0; s < SLOTS; s++) r[s] = 0;

    long long stride = (long long)gridDim.x * blockDim.x;
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
        if (ship[i] > cutoff) continue;
        unsigned long long p = (unsigned long long)price[i];
        unsigned long long dp = p * (unsigned long long)(100 - disc[i]);
        unsigned long long v[AGGS] = {1ULL, (unsigned long long)qty[i], p, dp,
                                      dp * (unsigned long long)(100 + tax[i]),
                                      (unsigned long long)disc[i]};
        int g = gid[i];
        #pragma unroll
        for (int k = 0; k < GROUPS; k++) {
            unsigned long long m = (g == k) ? ~0ULL : 0ULL;  // all ones only for this row's group
            #pragma unroll
            for (int a = 0; a < AGGS; a++) r[k * AGGS + a] += v[a] & m;
        }
    }

    #pragma unroll
    for (int s = 0; s < SLOTS; s++) {
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) r[s] += __shfl_down_sync(0xffffffffu, r[s], off);
    }
    if ((threadIdx.x & 31) == 0) {
        #pragma unroll
        for (int s = 0; s < SLOTS; s++) if (r[s] != 0) atomicAdd(acc + s, r[s]);
    }
    __syncthreads();

    for (int s = threadIdx.x; s < SLOTS; s += blockDim.x)
        if (acc[s] != 0) atomicAdd(out + s, acc[s]);
}
'''

_module = cp.RawModule(code=_SOURCE)
_kernels = {"v1": _module.get_function('q1_v1'), "v2": _module.get_function('q1_v2')}
_SMS = cp.cuda.Device().attributes["MultiProcessorCount"]


def v2_blocks(n):
    """Grid for q1_v2: enough blocks to fill every SM, never more blocks than 256-row chunks."""
    per_sm = cp.cuda.driver.occupancyMaxActiveBlocksPerMultiprocessor(
        _kernels["v2"].kernel.ptr, THREADS, 0)
    return max(1, min(_SMS * per_sm, (n + THREADS - 1) // THREADS))


def q1_sums(price, qty, disc, tax, ship, gid, cutoff, out=None, kernel="v1"):
    """Launch kernel "v1" or "v2". Returns the (6 groups x 6 aggregates) uint64 sums, still on the GPU."""
    cols = {"price": price, "qty": qty, "disc": disc, "tax": tax, "ship": ship, "gid": gid}
    n = price.size
    for name, a in cols.items():
        if not isinstance(a, cp.ndarray) or a.ndim != 1 or a.dtype != DTYPES[name]:
            raise TypeError(f"{name} must be a 1-D cupy {np.dtype(DTYPES[name]).name} array")
        if a.size != n:
            raise ValueError("all columns must have the same length")
    if out is None:
        out = cp.zeros(GROUPS * AGGS, dtype=cp.uint64)
    else:
        out.fill(0)
    if n:
        blocks = v2_blocks(n) if kernel == "v2" else (n + ROWS_PER_BLOCK - 1) // ROWS_PER_BLOCK
        _kernels[kernel]((blocks,), (THREADS,), (price, qty, disc, tax, ship, gid,
                                          np.int64(n), np.int32(cutoff), out))
    return out.reshape(GROUPS, AGGS)


def q1_rows(sums):
    """Host side: the Q1 rows from the (6 x 6) sums, in Q1 order, empty groups left out."""
    rows = []
    for g, (cnt, qty, price, disc_price, charge, disc) in enumerate(sums.tolist()):
        if cnt == 0:
            continue
        rows.append({"l_returnflag": NAMES[g][0], "l_linestatus": NAMES[g][1],
                     "sum_qty": qty, "sum_base_price": price, "sum_disc_price": disc_price,
                     "sum_charge": charge, "avg_qty": qty / 100 / cnt,
                     "avg_price": price / 100 / cnt, "avg_disc": disc / 100 / cnt,
                     "count_order": cnt})
    return rows


def q1(price, qty, disc, tax, ship, gid, cutoff, kernel="v1"):
    """Q1 on the GPU: the kernel, one small D2H copy (288 bytes), then the rows on the host."""
    return q1_rows(cp.asnumpy(q1_sums(price, qty, disc, tax, ship, gid, cutoff, kernel=kernel)))
