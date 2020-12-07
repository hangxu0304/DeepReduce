#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "./compression_utils.hpp"

#include <cmath>
#include <string>

using namespace tensorflow;

REGISTER_OP("Logger")
.Attr("bloom_logs_path: string")
.Attr("gradient_id: int")
.Attr("rank: int")
.Attr("verbosity_frequency: int")
.Attr("verbosity: int")
.Input("initial_tensor: float32")
.Input("coefficients: double")
.Input("step: int64")
.Doc(R"doc()doc");

class LoggerOp : public OpKernel {

public:

    explicit LoggerOp(OpKernelConstruction *context) : OpKernel(context) {
        OP_REQUIRES_OK(context, context->GetAttr("bloom_logs_path", &bloom_logs_path));
        OP_REQUIRES_OK(context, context->GetAttr("gradient_id", &gradient_id));
        OP_REQUIRES_OK(context, context->GetAttr("rank", &rank));
        OP_REQUIRES_OK(context, context->GetAttr("verbosity", &verbosity));
        OP_REQUIRES_OK(context, context->GetAttr("verbosity_frequency", &verbosity_frequency));
    }

    void Compute(OpKernelContext *context) override {

        // Retrieving Inputs
        const Tensor &initial_tensor = context->input(0);
        int N = initial_tensor.flat<float>().size();
        const Tensor &coefficients_tensor = context->input(1);
        // auto coefficients_tensor_flat = coefficients_tensor.flat<float64>();
        int64 step = context->input(2).flat<int64>()(0);

        // *********************** Logging ********************** //
        if (verbosity_frequency != 0 && step % verbosity_frequency == 0 ) {
            CompressionUtilities::logging(N, initial_tensor, coefficients_tensor, bloom_logs_path, gradient_id,
                                        step, rank, verbosity);
        }
        // *********************** Logging ********************** //
    }

private:
    // Logging
    string bloom_logs_path;
    int gradient_id;
    int rank;
    int verbosity_frequency;
    int verbosity;
};
REGISTER_KERNEL_BUILDER(Name("Logger").Device(DEVICE_CPU), LoggerOp);
