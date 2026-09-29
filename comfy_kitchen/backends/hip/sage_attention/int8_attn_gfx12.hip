// SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Unmasked D=128 INT8 attention over the layouts from launch_gfx12_qk_quant and
// launch_gfx12_tile_channel_v_quant. Separate so only it builds -fno-fast-math.
#include <hip/hip_runtime.h>

#include <cfloat>
#include <cstdint>
#include <stdexcept>

#include "../mma.h"
#include "sage_common.h"

namespace comfy::hip_backend::sage {
namespace {

// Schedule: BM128/BN64, eight waves, padded LDS tiles, next K/V tile prefetched
// through registers, P packed straight from the score accumulator.

constexpr int kLongHeadDim = 128;
constexpr int kLongBlockM = 128;
constexpr int kLongBlockN = 64;
constexpr int kLongWarps = 8;
constexpr int kLongThreads = kLongWarps * 32;
constexpr int kLongKStride = kLongHeadDim + 8;
constexpr int kLongVStride = kLongBlockN + 8;

__forceinline__ __device__ int long_fresh_lane_id() {
    int lane;
    // Recomputed after the loop; carrying it through makes LLVM spill.
    asm volatile("v_mbcnt_lo_u32_b32 %0, -1, 0" : "=v"(lane));
    return lane;
}

__forceinline__ __device__ void long_store_b128_padded(int8_t* dst, uint4 value) {
    // Eight-byte padding spreads consecutive WMMA fragment rows over banks.
    *reinterpret_cast<uint2*>(dst) = make_uint2(value.x, value.y);
    *reinterpret_cast<uint2*>(dst + 8) = make_uint2(value.z, value.w);
}

// K and V^T are padded to whole 64-key tiles: 256 threads load exactly 512
// 16-byte chunks per operand, with no bounds checks.
struct LongKVTileRegs {
    uint4 k[2];
    uint4 v[2];

    __forceinline__ __device__ void load(const int8_t* __restrict__ k_head,
                                         const int8_t* __restrict__ v_head,
                                         int tile) {
        const int tid = threadIdx.x;
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            const int chunk = tid + i * kLongThreads;
            const int krow = chunk >> 3;
            const int kd16 = (chunk & 7) << 4;
            k[i] = *reinterpret_cast<const uint4*>(
                k_head + static_cast<int64_t>(tile * kLongBlockN + krow) * kLongHeadDim + kd16);

            const int vd = chunk >> 2;
            const int vn16 = (chunk & 3) << 4;
            v[i] = *reinterpret_cast<const uint4*>(
                v_head + static_cast<int64_t>(tile) * kLongHeadDim * kLongBlockN +
                vd * kLongBlockN + vn16);
        }
    }

    __forceinline__ __device__ void store(int8_t* ks, int8_t* vs) const {
        const int tid = threadIdx.x;
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            const int chunk = tid + i * kLongThreads;
            const int krow = chunk >> 3;
            const int kd16 = (chunk & 7) << 4;
            long_store_b128_padded(ks + krow * kLongKStride + kd16, k[i]);

            const int vd = chunk >> 2;
            const int vn16 = (chunk & 3) << 4;
            long_store_b128_padded(vs + vd * kLongVStride + vn16, v[i]);
        }
    }
};

template <bool NoQueryTail>
__forceinline__ __device__ void long_stage_q_tile(
    const int8_t* __restrict__ q_head, int8_t* __restrict__ qs,
    int query_block0, int qo_len) {
    const int tid = threadIdx.x;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int chunk = tid + i * kLongThreads;
        const int row = chunk >> 3;
        const int d16 = (chunk & 7) << 4;
        uint4 value = make_uint4(0, 0, 0, 0);
        if (NoQueryTail || query_block0 + row < qo_len) {
            value = *reinterpret_cast<const uint4*>(
                q_head + static_cast<int64_t>(query_block0 + row) * kLongHeadDim + d16);
        }
        long_store_b128_padded(qs + row * kLongKStride + d16, value);
    }
}

template <typename OutT, bool NoQueryTail, bool FullScoreScale>
__global__ __launch_bounds__(kLongThreads)
__attribute__((amdgpu_waves_per_eu(1, 1)))
void gfx12_int8_attention_kernel(
    const int8_t* __restrict__ q,
    const int8_t* __restrict__ k,
    const int8_t* __restrict__ v,
    OutT* __restrict__ output,
    const float* __restrict__ q_descale,
    const float* __restrict__ k_descale,
    const float* __restrict__ v_descale,
    int qo_len, int kv_len, int padded_q, int padded_k,
    int k_groups_per_head, int query_block_offset,
    int64_t output_stride_h, int64_t output_stride_n, float sm_scale) {

    __shared__ __align__(16) int8_t ks[2][kLongBlockN * kLongKStride];
    __shared__ __align__(16) int8_t vs[2][kLongHeadDim * kLongVStride];
    __shared__ __align__(16) int8_t qs[kLongBlockM * kLongKStride];

    const int lane = threadIdx.x & 31;
    const int wave = threadIdx.x >> 5;
    const int head = blockIdx.y;
    const int query_block = blockIdx.x + query_block_offset;
    const int query0 = query_block * kLongBlockM + wave * 16;
    const bool active_query_wave = NoQueryTail || query0 < qo_len;
    const int tiles = padded_k / kLongBlockN;
    const int8_t* const q_head =
        q + static_cast<int64_t>(head) * qo_len * kLongHeadDim;
    const int8_t* const k_head =
        k + static_cast<int64_t>(head) * padded_k * kLongHeadDim;
    const int8_t* const v_head =
        v + static_cast<int64_t>(head) * kLongHeadDim * padded_k;

    const int query_lane = query0 + acc_col(lane);
    const float q_scale = active_query_wave && query_lane < qo_len
        ? q_descale[head * padded_q + query_lane]
        : 0.0f;
    const float qk_scale = sm_scale * kLog2e;
    // Same finite softmax state and P quantization as the generic kernel.
    float row_max = kMaskedScore;
    float row_sum = 1.0f;
    v8f out_acc[8];
#pragma unroll
    for (int dt = 0; dt < 8; ++dt) {
        out_acc[dt] = v8f{0, 0, 0, 0, 0, 0, 0, 0};
    }

    LongKVTileRegs next;
    int kv_buf = 0;
    long_stage_q_tile<NoQueryTail>(q_head, qs, query_block * kLongBlockM, qo_len);
    next.load(k_head, v_head, 0);
    next.store(ks[kv_buf], vs[kv_buf]);
    __syncthreads();

    for (int tile = 0; tile < tiles; ++tile) {
        const bool has_next = tile + 1 < tiles;
        if (has_next) {
            next.load(k_head, v_head, tile + 1);
        }

        // Idle tail waves still join K/V staging and barriers but skip the math.
        if (active_query_wave) {
            float key_scale[4];
#pragma unroll
            for (int ct = 0; ct < 4; ++ct) {
                key_scale[ct] =
                    k_descale[head * k_groups_per_head + tile * 4 + ct];
            }
            float score_scale[4];
#pragma unroll
            for (int ct = 0; ct < 4; ++ct) {
                // Scale association is picked per grid by the launcher.
                if constexpr (FullScoreScale) {
                    score_scale[ct] = (q_scale * key_scale[ct]) * qk_scale;
                } else {
                    score_scale[ct] = key_scale[ct] * qk_scale;
                }
            }

            v8f score[4];
            {
                v8i score_i32[4];
#pragma unroll
                for (int ct = 0; ct < 4; ++ct) score_i32[ct] = MmaInt8::zero();
#pragma unroll
                for (int dt = 0; dt < 8; ++dt) {
                    const MmaInt8::Frag q_frag = MmaInt8::load(
                        qs, wave * 16 + frag_row(lane), dt * 16, kLongKStride, lane);
#pragma unroll
                    for (int ct = 0; ct < 4; ++ct) {
                        const MmaInt8::Frag k_frag = MmaInt8::load(
                            ks[kv_buf], ct * 16 + frag_row(lane), dt * 16,
                            kLongKStride, lane);
                        // K x Q^T yields scores in P x V's A-operand order.
                        score_i32[ct] = MmaInt8::mma(k_frag, q_frag, score_i32[ct]);
                    }
                }
#pragma unroll
                for (int ct = 0; ct < 4; ++ct) {
#pragma unroll
                    for (int e = 0; e < 8; ++e) {
                        float value;
                        if constexpr (FullScoreScale) {
                            value = static_cast<float>(score_i32[ct][e]) * score_scale[ct];
                        } else {
                            value = (static_cast<float>(score_i32[ct][e]) * q_scale) *
                                    score_scale[ct];
                        }
                        const int key = tile * kLongBlockN + ct * 16 + acc_row(lane, e);
                        if (key >= kv_len) value = kMaskedScore;
                        score[ct][e] = value;
                    }
                }
            }

            float local_max = -FLT_MAX;
#pragma unroll
            for (int ct = 0; ct < 4; ++ct) {
#pragma unroll
                for (int e = 0; e < 8; ++e) local_max = fmaxf(local_max, score[ct][e]);
            }
            float tile_max = fmaxf(local_max, __shfl_xor(local_max, 16, 32));

            // Shift so P fills u8 [0, 255]; tile_scale carries the gap to row_max.
            tile_max -= kProbU8Offset;
            const float previous_max = row_max;
            const float smaller = __builtin_amdgcn_exp2f(-fabsf(previous_max - tile_max));
            const float output_scale = previous_max < tile_max ? smaller : 1.0f;
            const float tile_scale = tile_max < previous_max ? smaller : 1.0f;
            row_max = fmaxf(previous_max, tile_max);
            row_sum *= output_scale;
#pragma unroll
            for (int dt = 0; dt < 8; ++dt) {
#pragma unroll
                for (int e = 0; e < 8; ++e) {
                    out_acc[dt][e] *=
                        __shfl(output_scale, acc_row(lane, e), 32);
                }
            }

            MmaInt8::Frag direct_p_frag[4];
#pragma unroll
            for (int ct = 0; ct < 4; ++ct) {
                uint32_t packed[8];
#pragma unroll
                for (int e = 0; e < 8; ++e) {
                    const float probability = __builtin_amdgcn_exp2f(score[ct][e] - tile_max);
                    score[ct][e] = probability;
                    packed[e] = prob_to_u8(probability);
                }
                direct_p_frag[ct] = pack_prob_frag(packed, lane);
            }

            // Per-half-wave denominator, summed in a fixed balanced order.
            const float tile_sum =
                (sum8_balanced(score[0]) + sum8_balanced(score[2])) +
                (sum8_balanced(score[1]) + sum8_balanced(score[3]));
            row_sum += tile_sum * tile_scale;

#pragma unroll
            for (int dt = 0; dt < 8; ++dt) {
                v8i product = MmaInt8::zero();
#pragma unroll
                for (int ct = 0; ct < 4; ++ct) {
                    const MmaInt8::Frag v_frag = MmaInt8::load(
                        vs[kv_buf], dt * 16 + frag_row(lane), ct * 16,
                        kLongVStride, lane);
                    product = MmaInt8::mma_ua(direct_p_frag[ct], v_frag, product);
                }
                const int d = dt * 16 + acc_col(lane);
                const float value_scale =
                    v_descale[(head * tiles + tile) * kLongHeadDim + d];
#pragma unroll
                for (int e = 0; e < 8; ++e) {
                    const float row_tile_scale =
                        __shfl(tile_scale, acc_row(lane, e), 32);
                    out_acc[dt][e] = fmaf(
                        static_cast<float>(product[e]),
                        value_scale * row_tile_scale, out_acc[dt][e]);
                }
            }
        }

        if (has_next) {
            next.store(ks[kv_buf ^ 1], vs[kv_buf ^ 1]);
            __syncthreads();
            kv_buf ^= 1;
        }
    }

    row_sum += __shfl_xor(row_sum, 16, 32);

    if (active_query_wave) {
        const int output_lane = long_fresh_lane_id();
#pragma unroll
        for (int dt = 0; dt < 8; ++dt) {
            const int d = dt * 16 + acc_col(output_lane);
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                const int row = query0 + acc_row(output_lane, e);
                if (NoQueryTail || row < qo_len) {
                    const float reciprocal = 1.0f / __shfl(
                        row_sum, acc_row(output_lane, e), 32);
                    output[static_cast<int64_t>(head) * output_stride_h +
                           static_cast<int64_t>(row) * output_stride_n + d] =
                        static_cast<OutT>(out_acc[dt][e] * reciprocal);
                }
            }
        }
    }
}

template <typename OutT>
void launch_gfx12_int8_attention_typed(
    const int8_t* q, const int8_t* k, const int8_t* v, OutT* output,
    const float* q_descale, const float* k_descale, const float* v_descale,
    int num_heads, int qo_len, int kv_len, int padded_q, int padded_k,
    int k_groups_per_head,
    int64_t output_stride_h, int64_t output_stride_n,
    float sm_scale, hipStream_t stream) {
    const int full_blocks = qo_len / kLongBlockM;
    if (full_blocks) {
        // A full grid scales as (s * q_scale) * (k_scale * qk_scale). With a
        // query tail, all blocks use the tail order so every row rounds alike.
        if (qo_len % kLongBlockM != 0) {
            gfx12_int8_attention_kernel<OutT, true, true>
                <<<dim3(full_blocks, num_heads), kLongThreads, 0, stream>>>(
                    q, k, v, output, q_descale, k_descale, v_descale,
                    qo_len, kv_len, padded_q, padded_k, k_groups_per_head,
                    0, output_stride_h, output_stride_n, sm_scale);
        } else {
            gfx12_int8_attention_kernel<OutT, true, false>
                <<<dim3(full_blocks, num_heads), kLongThreads, 0, stream>>>(
                    q, k, v, output, q_descale, k_descale, v_descale,
                    qo_len, kv_len, padded_q, padded_k, k_groups_per_head,
                    0, output_stride_h, output_stride_n, sm_scale);
        }
    }
    if (full_blocks * kLongBlockM != qo_len) {
        gfx12_int8_attention_kernel<OutT, false, true>
            <<<dim3(1, num_heads), kLongThreads, 0, stream>>>(
                q, k, v, output, q_descale, k_descale, v_descale,
                qo_len, kv_len, padded_q, padded_k, k_groups_per_head,
                full_blocks, output_stride_h, output_stride_n, sm_scale);
    }
}
}  // namespace
}  // namespace comfy::hip_backend::sage

using namespace comfy::hip_backend::sage;

extern "C" void launch_gfx12_int8_attention(
    const void* q, const void* k, const void* v, void* o,
    const void* q_descale, const void* k_descale, const void* v_descale,
    int num_heads, int qo_len, int kv_len, int padded_q, int padded_k,
    int k_groups_per_head,
    int64_t output_stride_h, int64_t output_stride_n,
    float sm_scale, int output_dtype_code, hipStream_t stream) {
    if (!current_device_has_nonduplicated_wmma()) {
        throw std::runtime_error(
            "gfx12 INT8 attention requires a gfx12-class (gfx12/gfx117) device");
    }
    if (num_heads <= 0 || qo_len <= 0 || kv_len <= 0) {
        throw std::runtime_error(
            "gfx12 INT8 attention requires positive heads and lengths");
    }
    if (padded_q % 128 != 0 || padded_q < qo_len ||
        padded_k % 64 != 0 || padded_k < kv_len) {
        throw std::runtime_error(
            "gfx12 INT8 attention received invalid Q or K/V padding");
    }
    if (k_groups_per_head != padded_k / 16) {
        throw std::runtime_error(
            "gfx12 INT8 attention received invalid K scale grouping");
    }
    if (reinterpret_cast<uintptr_t>(q) % 16 != 0 ||
        reinterpret_cast<uintptr_t>(k) % 16 != 0 ||
        reinterpret_cast<uintptr_t>(v) % 16 != 0 ||
        reinterpret_cast<uintptr_t>(o) % 16 != 0) {
        throw std::runtime_error(
            "gfx12 INT8 attention buffers must be 16-byte aligned");
    }
    const auto* qi = static_cast<const int8_t*>(q);
    const auto* ki = static_cast<const int8_t*>(k);
    const auto* vi = static_cast<const int8_t*>(v);
    const auto* qs = static_cast<const float*>(q_descale);
    const auto* ks = static_cast<const float*>(k_descale);
    const auto* vs = static_cast<const float*>(v_descale);
    if (output_dtype_code == 1) {
        launch_gfx12_int8_attention_typed(
            qi, ki, vi, static_cast<__half*>(o), qs, ks, vs, num_heads,
            qo_len, kv_len, padded_q, padded_k, k_groups_per_head,
            output_stride_h, output_stride_n, sm_scale, stream);
    } else if (output_dtype_code == 2) {
        launch_gfx12_int8_attention_typed(
            qi, ki, vi, static_cast<__bf16*>(o), qs, ks, vs, num_heads,
            qo_len, kv_len, padded_q, padded_k, k_groups_per_head,
            output_stride_h, output_stride_n, sm_scale, stream);
    } else {
        throw std::runtime_error(
            "gfx12 INT8 attention supports float16 and bfloat16 output");
    }
}
