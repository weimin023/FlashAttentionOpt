#include "cutlass/gemm/device/gemm.h"
#include "cutlass_common.cuh"

using ElementInputA = float;
using LayoutInputA = cutlass::layout::RowMajor;
using ElementInputB = float;
using LayoutInputB = cutlass::layout::RowMajor;
using ElementOutput = float;
using LayoutOutput = cutlass::layout::RowMajor;
using ElementAccumulator = float;                   // <- data type of accumulator
using ElementComputeEpilogue = ElementAccumulator;

using Gemm = cutlass::gemm::device::Gemm<ElementInputA, LayoutInputA,
                                            ElementInputB, LayoutInputB,
                                            ElementOutput,LayoutOutput>;

using Gemm_ref = cutlass::reference::device::Gemm<ElementInputA,
                                            LayoutInputA,
                                            ElementInputB,
                                            LayoutInputB,
                                            ElementOutput,
                                            LayoutOutput,
                                            ElementComputeEpilogue,
                                            ElementComputeEpilogue>;
struct CutlassGEMMDefault {
public:
    int M, N, K;
    Gemm gemm_op;
    typename Gemm::Arguments args;

    cutlass::HostTensor<ElementInputA, LayoutInputA> A;
    cutlass::HostTensor<ElementInputB, LayoutInputB> B;
    cutlass::HostTensor<ElementOutput, LayoutOutput> out;
    cutlass::HostTensor<ElementOutput, LayoutOutput> out_ref;

    CutlassGEMMDefault(int M, int N, int K): M(M), N(N), K(K), A({M, K}), B({K, N}), out({M, N}), out_ref({M, N}) {}

    void init();
    bool correctness_check();

    float benchmark(int iterations = 10);
};