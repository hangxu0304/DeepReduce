#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "../../third_party/bloomfilter/inc/OrdinaryBloomFilter.hpp"
#include "./policies.hpp"
#include "./compression_utils.hpp"

#include <cmath>
#include <string>
#include <cstdlib>
#include <algorithm>
#include <functional>

using namespace tensorflow;

REGISTER_OP("BloomCompressor")
.Attr("T: {int32, int64, float16, float32, float64}")
.Attr("false_positives_aware: bool")
.Attr("policy: string")
.Attr("fpr: float")
.Attr("hash_num: int")
.Attr("bloom_size: int")
.Attr("bloom_logs_path: string")
.Attr("gradient_id: int")
.Attr("rank: int")
.Attr("verbosity_frequency: int")
.Attr("verbosity: int")
.Input("values: T")
.Input("indices: int32")
.Input("initial_tensor: int32")
.Input("step: int64")
.Output("compressed_tensor: int8")
.Doc(R"doc()doc");

REGISTER_OP("BloomDecompressor")
.Attr("policy: string")
.Attr("mem_mode: int")
.Attr("hash_num: int")
.Attr("bloom_size: int")
.Attr("bloom_logs_path: string")
.Attr("gradient_id: int")
.Attr("rank: int")
.Attr("suffix: int")
.Attr("verbosity_frequency: int")
.Attr("verbosity: int")
.Input("compressed_tensor: int8")
.Input("decompressed_size: int32")
.Input("step: int64")
.Output("decompressed_tensor: int32")
.Doc(R"doc()doc");

class BloomCompressorOp : public OpKernel {

public:

    explicit BloomCompressorOp(OpKernelConstruction *context) : OpKernel(context) {
        OP_REQUIRES_OK(context, context->GetAttr("false_positives_aware", &false_positives_aware));
        OP_REQUIRES_OK(context, context->GetAttr("policy", &policy));
        OP_REQUIRES_OK(context, context->GetAttr("fpr", &fpr));
        OP_REQUIRES_OK(context, context->GetAttr("hash_num", &hash_num));
        OP_REQUIRES_OK(context, context->GetAttr("bloom_size", &bloom_size));
        OP_REQUIRES_OK(context, context->GetAttr("bloom_logs_path", &bloom_logs_path));
        OP_REQUIRES_OK(context, context->GetAttr("gradient_id", &gradient_id));
        OP_REQUIRES_OK(context, context->GetAttr("rank", &rank));
        OP_REQUIRES_OK(context, context->GetAttr("verbosity", &verbosity));
        OP_REQUIRES_OK(context, context->GetAttr("verbosity_frequency", &verbosity_frequency));
    }

    void Compute(OpKernelContext *context) override {



         // Retrieving Inputs
        const Tensor &values = context->input(0);  auto values_flat = values.flat<int>();
        const Tensor &indices = context->input(1);  auto indices_flat = indices.flat<int>();
        const Tensor &initial_tensor = context->input(2); auto initial_flat = initial_tensor.flat<int>();
        int64 step = context->input(3).flat<int64>()(0);

        int N = initial_flat.size();
        int K = values_flat.size();

        // Given FPR compute M and H
        int m,rem,h;
        float temp_m,temp_h;
        temp_m = (K*fabs(log(fpr))) / (pow(log(2), 2));
        // Give bloom size in number of bytes ; bloom size must be a multiple of 8
        m = int(temp_m/8);
        rem = m % 8;
        if (rem != 0 || m == 0){
            m += 1;
        }
        temp_h = (m*8 / K)*log(2);
        h = int(ceil(temp_h));

        bloom_size=m;
        hash_num=h;

        // Building Bloom Filter
        bloom::OrdinaryBloomFilter<uint32_t> bloom(hash_num, bloom_size);
        for (int i=0; i<K; ++i) {
            bloom.Insert(indices_flat(i));
        }

        // Select Indices using a Policy, and update K by the size of selected_indices
        std::vector<int> selected_indices;
        Policies::select_indices(policy, N, K, step, bloom, selected_indices);
        K = selected_indices.size();

        int output_concat_dim = 2*sizeof(int) + K*sizeof(int) + bloom_size;
        // Create an output tensor
        TensorShape output_shape;
        output_shape.AddDim(output_concat_dim);
        Tensor *output = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output));

        auto output_flat = output->template flat<int8>();
        int8* out_ptr = output_flat.data();

        // copy the bloom_size and hash_num in the output tensor
        memcpy(out_ptr, &bloom_size, sizeof(int));
        memcpy(out_ptr+sizeof(int), &hash_num, sizeof(int));

        // Copy the values in the output tensor
        std::vector<int> new_values;
        if (false_positives_aware) {
            for (int i=0; i<K; i++) {
                int chosen_index = selected_indices[i];
                new_values.push_back(initial_flat(chosen_index));
            }
            memcpy(out_ptr+2*sizeof(int), new_values.data(), K*sizeof(int));
        } else {
            const void* values_ptr = values_flat.data();
            memcpy(out_ptr+2*sizeof(int), values_ptr, K*sizeof(int));
        }

        // Copy the bloom filter in the output tensor
        std::vector<unsigned char> &bloom_vec = bloom.Get_bloom();
        std::copy(bloom_vec.begin(), bloom_vec.end(), out_ptr+(K+2)*sizeof(int));

        // *********************** For Debugging ********************** //
        if (verbosity_frequency != 0 && step % verbosity_frequency == 0 ) {
            if (!false_positives_aware) {
                // Select Indices using a Policy
                Policies::select_indices(policy, N, K, step, bloom, selected_indices);
            }
            CompressionUtilities::logging_compressor(bloom, N, K, output_concat_dim, initial_tensor, indices, values,
                            new_values, selected_indices, bloom_logs_path, gradient_id, step, policy, rank, verbosity);
        }
        // *********************** For Debugging ********************** //
    }

private:
    float fpr;
    int hash_num;
    int bloom_size;
    string policy;
    bool false_positives_aware;
    // Logging
    string bloom_logs_path;
    int gradient_id;
    int rank;
    int verbosity_frequency;
    int verbosity;
};

class BloomDecompressorOp : public OpKernel {

public:

    explicit BloomDecompressorOp(OpKernelConstruction *context) : OpKernel(context) {
        OP_REQUIRES_OK(context, context->GetAttr("policy", &policy));
        OP_REQUIRES_OK(context, context->GetAttr("mem_mode", &mem_mode));
        OP_REQUIRES_OK(context, context->GetAttr("hash_num", &hash_num));
        OP_REQUIRES_OK(context, context->GetAttr("bloom_size", &bloom_size));
        OP_REQUIRES_OK(context, context->GetAttr("bloom_logs_path", &bloom_logs_path));
        OP_REQUIRES_OK(context, context->GetAttr("gradient_id", &gradient_id));
        OP_REQUIRES_OK(context, context->GetAttr("rank", &rank));
        OP_REQUIRES_OK(context, context->GetAttr("suffix", &suffix));
        OP_REQUIRES_OK(context, context->GetAttr("verbosity", &verbosity));
        OP_REQUIRES_OK(context, context->GetAttr("verbosity_frequency", &verbosity_frequency));
    }

    void Compute(OpKernelContext *context) override {

        // Retrieving Inputs
        const Tensor &compressed_tensor = context->input(0);
        auto compressed_tensor_flat = compressed_tensor.flat<int8>();
        int N = *context->input(1).flat<int>().data();
        int64 step = context->input(2).flat<int64>()(0);

        // Reconstruct the bloom filter
        const int8 *ptr = compressed_tensor_flat.data();           // Note: int8 is 1 byte

        memcpy(&bloom_size, ptr, sizeof(int));
        memcpy(&hash_num, ptr+sizeof(int), sizeof(int));

        int K = (compressed_tensor_flat.size()-bloom_size)/sizeof(int) - 2;
        int values_bytes = K*sizeof(int);
        int *values_vec = (int*) malloc(values_bytes);
        memcpy(values_vec, ptr+2*sizeof(int), values_bytes);

        ptr += (values_bytes + 2*sizeof(int));
        bloom::OrdinaryBloomFilter<uint32_t> bloom(hash_num, bloom_size, ptr);

        // Create an output tensor
        TensorShape decompressed_tensor_shape;
        decompressed_tensor_shape.AddDim(N);
        Tensor *decompressed_tensor = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(0, decompressed_tensor_shape, &decompressed_tensor));
        auto decompressed_tensor_flat = decompressed_tensor->template flat<int>();
        memset(decompressed_tensor_flat.data(), 0, N*sizeof(int));

        // Select Indices using a Policy
        std::vector<int> selected_indices;
        Policies::select_indices(policy, N, K, step, bloom, selected_indices);

        // Map values to the selected indices
        for (int i=0; i<K; i++) {
            decompressed_tensor_flat(selected_indices[i]) = values_vec[i];
        }

        // *********************** For Debugging ********************** //
        if (verbosity_frequency != 0 && step % verbosity_frequency == 0 && mem_mode == 0) {
            CompressionUtilities::logging_decompressor(bloom, N, K, values_vec, selected_indices, bloom_logs_path,
                                gradient_id, suffix, step, decompressed_tensor, policy, rank, verbosity);
        }
        // *********************** For Debugging ********************** //

        free(values_vec);
    }

private:
    string policy;
    int mem_mode;
    int hash_num;
    int bloom_size;
    // Logging
    string bloom_logs_path;
    int gradient_id;
    int rank;
    int suffix;
    int verbosity_frequency;
    int verbosity;
};

REGISTER_KERNEL_BUILDER(Name("BloomCompressor").Device(DEVICE_CPU), BloomCompressorOp);
REGISTER_KERNEL_BUILDER(Name("BloomDecompressor").Device(DEVICE_CPU), BloomDecompressorOp);
