#include <vector>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/kernel/gemm_grouped.h"
#include "cutlass/gemm/kernel/default_gemm_grouped.h"
#include "cutlass/gemm/device/gemm_grouped.h"
#include "cutlass/gemm/device/gemm_universal.h"

#include "cutlass/util/command_line.h"
#include "cutlass/util/distribution.h"
#include "cutlass/util/device_memory.h"
#include "cutlass/util/tensor_view_io.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/reference/host/gemm_complex.h"
#include "cutlass/util/reference/device/gemm_complex.h"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_copy.h"
#include "cutlass/util/reference/device/tensor_fill.h"
#include "cutlass/util/reference/host/tensor_norm.h"

#include "cutlass/layout/matrix.h"
#include "cutlass/gemm/kernel/gemm_grouped.h"
#include "cutlass/gemm/kernel/gemm_transpose_operands.h"
#include "cutlass/gemm/kernel/default_gemm.h"
#include "cutlass/gemm/kernel/default_gemm_complex.h"
#include "cutlass/gemm/device/default_gemm_configuration.h"
#include "cutlass/gemm/gemm.h"

#include "cutlass/epilogue/threadblock/epilogue_with_visitor.h"
#include "cutlass/fast_math.h"
#include "kernel_forward.h"

struct Options {
    Options():
        help(false),
        error(false),
        alignment(1),
        reference_check(true),
        head_number(12),
        batch_size(16),
        head_size(64),
        head_size_v(64),
        seq_length(1024),
        seq_length_kv(1024),
        use_mask(false),
        iterations(20),
        causal(false) { }
    
    bool help;
    bool error;
    bool reference_check;
    bool use_mask;
    bool causal;
    
    std::vector<cutlass::gemm::GemmCoord> problem_sizes0;
    std::vector<cutlass::gemm::GemmCoord> problem_sizes1;
    
    std::vector<cutlass::gemm::GemmCoord> problem_sizes0_real;
    std::vector<cutlass::gemm::GemmCoord> problem_sizes1_real;
    
    int alignment;
    int head_number;
    int batch_size;
    int head_size;
    int head_size_v;
    int seq_length;
    int seq_length_kv;
    int iterations;
    
    // alpha0, alpha1 and beta are fixed 
    // in this multi-head attention example
    float alpha0;
    float alpha1;
    float beta;

    void parse(int argc, char const **args) {
        cutlass::CommandLine cmd(argc, args);

        if (cmd.check_cmd_line_flag("help")) {
            help = true;
            return;
        }

        cmd.get_cmd_line_argument("alignment", alignment, 1);
        cmd.get_cmd_line_argument("head_number", head_number, 12);
        cmd.get_cmd_line_argument("batch_size", batch_size, 16);
        cmd.get_cmd_line_argument("head_size", head_size, 64);
        cmd.get_cmd_line_argument("head_size_v", head_size_v, head_size);
        cmd.get_cmd_line_argument("seq_length", seq_length, 1024);
        cmd.get_cmd_line_argument("seq_length_kv", seq_length_kv, seq_length);
        cmd.get_cmd_line_argument("use_mask", use_mask, false);
        cmd.get_cmd_line_argument("iterations", iterations, 20);
        cmd.get_cmd_line_argument("reference-check", reference_check, true);
        cmd.get_cmd_line_argument("causal", causal, true);

        randomize_problems();
    }
}

void test_cu_fmha_kernel_multihead_cutlass() {

}