#pragma once

#include <iostream>
#include <torch/extension.h>

#include "cutlass_common.cuh"

using ElementAccumulator = float;                   // <- data type of accumulator
using ElementComputeEpilogue = ElementAccumulator;  // <- data type of epilogue operations
using ElementInputA = cutlass::half_t;              // <- data type of elements in input matrix A
using ElementInputB = cutlass::half_t;              // <- data type of elements in input matrix B
using ElementOutput = float;                        // <- data type of elements in output matrix D

using LayoutInputA = cutlass::layout::ColumnMajor;
using LayoutInputB = cutlass::layout::RowMajor;
using LayoutOutput = cutlass::layout::RowMajor;

using MMAOp = cutlass::arch::OpClassTensorOp;
using SmArch = cutlass::arch::Sm80;
using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<128, 128, 32>;  // <- threadblock tile M = 128, N = 128, K = 32
// This code section describes tile size a warp will compute
using ShapeMMAWarp = cutlass::gemm::GemmShape<64, 64, 32>;  // <- warp tile M = 64, N = 64, K = 32
// This code section describes the size of MMA op
using ShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 16>;  // <- MMA Op tile M = 8, N = 8, K = 4

// This code section describes ?
using EpilogueOp = cutlass::epilogue::thread::LinearCombination<ElementOutput,                                     // <- data type of output matrix
                                                                128 / cutlass::sizeof_bits<ElementOutput>::value,  // <- This is the number of elements per
                                                                                                                // vectorized memory access. For half
                                                                                                                // precision, it's 8 elements. This becomes
                                                                                                                // the vector width of math instructions in
                                                                                                                // epilogue too
                                                                ElementAccumulator,                                // <- data type of accumulator
                                                                ElementComputeEpilogue>;  // <- data type for alpha/beta in linear combination function
using Gemm = cutlass::gemm::device::GemmSplitKParallel<ElementInputA, LayoutInputA,
                                                        ElementInputB, LayoutInputB,
                                                        ElementOutput, LayoutOutput,
                                                        ElementAccumulator,
                                                        MMAOp,
                                                        SmArch,
                                                        ShapeMMAThreadBlock,
                                                        ShapeMMAWarp,
                                                        ShapeMMAOp,
                                                        EpilogueOp>;

using Gemm_ref = cutlass::reference::device::Gemm<ElementInputA,
                                                    LayoutInputA,
                                                    ElementInputB,
                                                    LayoutInputB,
                                                    ElementOutput,
                                                    LayoutOutput,
                                                    ElementComputeEpilogue,
                                                    ElementComputeEpilogue>;
struct CutlassGEMMSplitk {
public:
    int M, N, K, split_k_slices;
    Gemm gemm_op;
    typename Gemm::Arguments args;
    cutlass::device_memory::allocation<uint8_t> workspace;

    cutlass::HostTensor<ElementInputA, LayoutInputA> A;
    cutlass::HostTensor<ElementInputB, LayoutInputB> B;
    cutlass::HostTensor<ElementOutput, LayoutOutput> C;
    cutlass::HostTensor<ElementOutput, LayoutOutput> D;
    cutlass::HostTensor<ElementOutput, LayoutOutput> D_ref;

    CutlassGEMMSplitk(int M_, int N_, int K_, int splitK)
        : M(M_), N(N_), K(K_), split_k_slices(splitK),
          A({M, K}), B({K, N}), C({M, N}), D({M, N}), D_ref({M, N}) {}

    void init();

    bool correctness_check();

    float benchmark(int iterations = 10);
};





                                                                                                          