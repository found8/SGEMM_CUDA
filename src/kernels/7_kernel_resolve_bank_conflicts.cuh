#pragma once

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))

template <const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void sgemmResolveBankConflicts(int M, int N, int K, float alpha,
                                          float *A, float *B, float beta,
                                          float *C) {
  const uint cRow = blockIdx.y;
  const uint cCol = blockIdx.x;

  // BN/TN are the number of threads to span a column
  const int threadCol = threadIdx.x % (BN / TN);
  const int threadRow = threadIdx.x / (BN / TN);

  // allocate space for the current blocktile in smem
  __shared__ float As[BM * BK];
  __shared__ float Bs[BK * BN];

  // Move blocktile to beginning of A's row and B's column
  A += cRow * BM * K;
  B += cCol * BN;
  C += cRow * BM * N + cCol * BN;

  // calculating the indices that this thread will load into SMEM
  // we'll load 128bit / 32bit = 4 elements per thread at each step
  const uint innerRowA = threadIdx.x / (BK / 4);
  const uint innerColA = threadIdx.x % (BK / 4);
  const uint innerRowB = threadIdx.x / (BN / 4);
  const uint innerColB = threadIdx.x % (BN / 4);

  // allocate thread-local cache for results in registerfile
  float threadResults[TM * TN] = {0.0};
  float regM[TM] = {0.0};
  float regN[TN] = {0.0};

  // outer-most loop over block tiles
  for (uint bkIdx = 0; bkIdx < K; bkIdx += BK) {
    // populate the SMEM caches
    // transpose A while loading it
    float4 tmp =
        reinterpret_cast<float4 *>(&A[innerRowA * K + innerColA * 4])[0];

    /* 每两个线程innerRowA相同，innerColA差1，访问的元素差 4*BM=4*128=512
     * 会访问同一个bank，所以此处会有bank conflict
     */
    As[(innerColA * 4 + 0) * BM + innerRowA] = tmp.x;
    As[(innerColA * 4 + 1) * BM + innerRowA] = tmp.y;
    As[(innerColA * 4 + 2) * BM + innerRowA] = tmp.z;
    As[(innerColA * 4 + 3) * BM + innerRowA] = tmp.w;

    // "linearize" Bs while storing it
    /* 不同线程可能读取B的同一行，也可能读取不同行，最终将整个block的所有行全部读入Bs
     * 每个线程读取一行的4个元素到寄存器，转存到Bs的4行中(将Bs看作每行16个元素)
     */
    tmp = reinterpret_cast<float4 *>(&B[innerRowB * N + innerColB * 4])[0]; // 当前线程innerRowB和innerColB是确定的

    /* (innerColB % 2) * 4 表示每次读取B中4个元素都转存到Bs的4行中(将Bs看作每行16个元素)
     * innerRowB * 8 表示B中每行(128元素)存储到Bs中8行(8*16元素)
     * innerColB / 2 表示每两个元素保存到同一列，结合上面两行，表示B中连续的8个元素转存到Bs同一列8行中
     * 每两个线程会写入Bs同一列，相差64个元素，有bank conflict
     */
    Bs[((innerColB % 2) * 4 + innerRowB * 8 + 0) * 16 + innerColB / 2] = tmp.x;
    Bs[((innerColB % 2) * 4 + innerRowB * 8 + 1) * 16 + innerColB / 2] = tmp.y;
    Bs[((innerColB % 2) * 4 + innerRowB * 8 + 2) * 16 + innerColB / 2] = tmp.z;
    Bs[((innerColB % 2) * 4 + innerRowB * 8 + 3) * 16 + innerColB / 2] = tmp.w;
    __syncthreads();

    // advance blocktile
    A += BK;     // move BK columns to right
    B += BK * N; // move BK rows down

    // calculate per-thread results
    for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {
      // block into registers
      for (uint i = 0; i < TM; ++i) {
        /* threadRow每(BN/TN=16)加1，即在16个线程内同一时刻可广播读取As
         * 每16个相差TM个bank，所以16个之间同一时刻不会访问同一bank
         */
        regM[i] = As[dotIdx * BM + threadRow * TM + i];
      }
      for (uint i = 0; i < TN; ++i) {
        /* 前例的Bs读取方式如下：
         * regN[i] = Bs[dotIdx * BN + threadCol * TN + i];
         * threadCol相邻线程间加1，所以相邻线程地址间隔为TN=8个bank
         * 间隔4个线程后地址间隔8*4=32个bank，访问同一个bank，从而bank conflict
         *
         * 新例中不同线程threadCol不同(BN/TN=16个线程重复)
         * 即warp中前16个线程对应读取16个不同bank，后16个线程与前16个线程地址相同(广播？)
         */
         regN[i] = Bs[(dotIdx * 8 + i) * 16 + threadCol];
      }
      /* 上述操作最终目的要实现：
       * 1. A中一列连续TM个元素转存到regM寄存器中
       * 2. B中一行连续TN个元素转存到regN寄存器中
       */
      for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {
        for (uint resIdxN = 0; resIdxN < TN; ++resIdxN) {
          threadResults[resIdxM * TN + resIdxN] +=
              regM[resIdxM] * regN[resIdxN];
        }
      }
    }
    __syncthreads();
  }

  // write out the results
  for (uint resIdxM = 0; resIdxM < TM; resIdxM += 1) {
    for (uint resIdxN = 0; resIdxN < TN; resIdxN += 4) {
      // load C vector into registers
      float4 tmp = reinterpret_cast<float4 *>(
          &C[(threadRow * TM + resIdxM) * N + threadCol * TN + resIdxN])[0];
      // perform GEMM update in reg
      tmp.x = alpha * threadResults[resIdxM * TN + resIdxN] + beta * tmp.x;
      tmp.y = alpha * threadResults[resIdxM * TN + resIdxN + 1] + beta * tmp.y;
      tmp.z = alpha * threadResults[resIdxM * TN + resIdxN + 2] + beta * tmp.z;
      tmp.w = alpha * threadResults[resIdxM * TN + resIdxN + 3] + beta * tmp.w;
      // write back
      reinterpret_cast<float4 *>(
          &C[(threadRow * TM + resIdxM) * N + threadCol * TN + resIdxN])[0] =
          tmp;
    }
  }
}