
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_types.h"

#include "../../third_party/FastPFor/headers/codecfactory.h"
#include "../../third_party/FastPFor/headers/deltautil.h"

#include <assert.h>
#include <string>
#include <cstdlib>

using namespace tensorflow;

using namespace FastPForLib;

REGISTER_OP("IntegerCompressor")
//.Attr("T: {int32, int64, float16, float32, float64}")
.Attr("logfile_suffix: int")       // For debugging
.Attr("logs_path_suffix: int")     // For debugging
.Attr("verbosity: int")            // For debugging
.Attr("code: string")
.Input("input: uint32")
.Input("step: int64")              // For debugging
.Output("intcompressed_tensor: uint32")
.Doc(R"doc( Integer compression )doc");

REGISTER_OP("IntegerDecompressor")
//.Attr("T: {int32, int64, float16, float32, float64}")
.Attr("logfile_suffix: int")       // For debugging
.Attr("logs_path_suffix: int")     // For debugging
.Attr("suffix: int")                    // For debugging
.Attr("verbosity: int")            // For debugging
.Attr("code: string")
.Input("input: uint32")
.Input("decompressed_size: int32")
.Input("step: int64")              // For debugging
.Output("decompressed_tensor: uint32")
.Doc(R"doc( Integer decompression )doc");


class IntegerCompressorOp : public OpKernel {

public:

    explicit IntegerCompressorOp(OpKernelConstruction *context) : OpKernel(context) {
        OP_REQUIRES_OK(context, context->GetAttr("logfile_suffix", &logfile_suffix));       // For debugging
        OP_REQUIRES_OK(context, context->GetAttr("logs_path_suffix", &logs_path_suffix));   // For debugging
        OP_REQUIRES_OK(context, context->GetAttr("verbosity", &verbosity));                 // For debugging
        OP_REQUIRES_OK(context, context->GetAttr("code", &code));
    }

    void Compute(OpKernelContext *context) override {

        const Tensor &input_tensor = context->input(0);
        auto input_tensor_flat = input_tensor.flat<uint32_t>();
        size_t input_tensor_size = input_tensor_flat.size();

        IntegerCODEC &codec = *CODECFactory::getFromName(code);   // Pick a CODEC
        std::vector<uint32> intcompressed_output(input_tensor_size + 262144);
        size_t intcompressed_size = intcompressed_output.size();
        codec.encodeArray(input_tensor_flat.data(), input_tensor_size, intcompressed_output.data(), intcompressed_size);
        // Shrink back the array:
        intcompressed_output.resize(intcompressed_size);
        intcompressed_output.shrink_to_fit();

        // display compression rate:
        std::cout << std::setprecision(3);
        std::cout << "You are using " << 32.0 * static_cast<double>(intcompressed_output.size()) /
                     static_cast<double>(input_tensor_flat.size()) << " ints per integer. " << std::endl;

        // Create an output tensor
        int output_concat_dim = intcompressed_output.size() ;
        printf("output_concat_dim %d\n", output_concat_dim);
        TensorShape output_shape;
        output_shape.AddDim(output_concat_dim);
        Tensor *output = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output));

        auto output_flat = output->template flat<uint32>();
        uint32* out_ptr = output_flat.data();

        std::copy(intcompressed_output.begin(), intcompressed_output.end(), out_ptr);


//        //////
//        std::vector<uint32> init(input_tensor_size);
//        const uint32_t *ptr = input_tensor_flat.data();
//        memcpy(init.data(), ptr, input_tensor_size*sizeof(int));
//        std::vector<uint32_t> decompressed_output(input_tensor_size);
//        codec.decodeArray(intcompressed_output.data(), output_concat_dim, decompressed_output.data(), input_tensor_size);
//        decompressed_output.resize(input_tensor_size);
//
//        assert(std::equal(init.begin(), init.end(), decompressed_output.begin()) == 1);
//        /////

        // *********************** For Debugging ********************** //
        const Tensor &step_tensor = context->input(1);
        auto step = step_tensor.flat<int64>();

        if (verbosity != 0 && step(0) % verbosity == 0 ) {
            std::string suffix = std::to_string(logfile_suffix);
            std::string logs_suffix = std::to_string(logs_path_suffix);
            std::string str_step = std::to_string(step(0));

            std::string cmd = "mkdir -p logs" + logs_suffix + "/step_" + str_step + "/" + suffix + "/";
            int systemRet = system(cmd.c_str());
            if(systemRet == -1){
                perror("mkdir failed");
            }
            std::string str = "logs" + logs_suffix + "/step_" + str_step + "/" + suffix + "/intcompressor_logs_" + suffix + ".txt";
            FILE* f = fopen(str.c_str(),"w");
            fprintf(f, "input_tensor: %s\n", input_tensor.DebugString(input_tensor_flat.size()).c_str());
            fprintf(f, "Output_concat_size: = %d\n\n", output_concat_dim);
            fprintf(f, "intcompressed_tensor: %s\n", output->DebugString(output_flat.size()).c_str());
            fprintf(f, "\n\n########################################################################################\n\n");
            fclose(f);

            std::string str1 = "logs" + logs_suffix + "/step_" + str_step + "/" + suffix + "/stats" + suffix + ".txt";
            f = fopen(str1.c_str(),"w");
            fprintf(f, "Initial_Size: %d  Final_Size: %d\n", input_tensor_flat.size()*32,  output_concat_dim*32 + 32);
            fclose(f);
        }
        // *********************** For Debugging ********************** //

    }

private:
    int logfile_suffix;     // For debugging
    int logs_path_suffix;   // For debugging
    int verbosity;          // For debugging
    string code;
};

class IntegerDecompressorOp : public OpKernel {

public:

    explicit IntegerDecompressorOp(OpKernelConstruction *context) : OpKernel(context) {
        OP_REQUIRES_OK(context, context->GetAttr("logfile_suffix", &logfile_suffix));       // For debugging
        OP_REQUIRES_OK(context, context->GetAttr("logs_path_suffix", &logs_path_suffix));   // For debugging
        OP_REQUIRES_OK(context, context->GetAttr("suffix", &suffix));                       // For debugging
        OP_REQUIRES_OK(context, context->GetAttr("verbosity", &verbosity));                 // For debugging
        OP_REQUIRES_OK(context, context->GetAttr("code", &code));

    }

    void Compute(OpKernelContext *context) override {

        const Tensor &input_tensor = context->input(0);
        auto input_tensor_flat = input_tensor.flat<uint32_t>();
        const size_t input_tensor_size = input_tensor_flat.size();

        const Tensor &decompressed_size_tensor = context->input(1);
        auto decompressed_size_flat = decompressed_size_tensor.flat<int>();
        int decompressed_size = *decompressed_size_flat.data();

        IntegerCODEC &codec = *CODECFactory::getFromName(code);   // Pick a CODEC
        std::vector<uint32_t> decompressed_output(decompressed_size);
        size_t recoveredsize = decompressed_output.size();

        codec.decodeArray(input_tensor_flat.data(), input_tensor_size, decompressed_output.data(), recoveredsize);
        decompressed_output.resize(recoveredsize);

        // Create an output tensor
        int output_concat_dim = recoveredsize;
        printf("output_concat_dim %d\n", output_concat_dim);
        TensorShape output_shape;
        output_shape.AddDim(output_concat_dim);
        Tensor *output = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output));

        auto output_flat = output->template flat<uint32>();
        uint32* out_ptr = output_flat.data();

        std::copy(decompressed_output.begin(), decompressed_output.end(), out_ptr);

        // *********************** For Debugging ********************** //
        const Tensor &step_tensor = context->input(2);
        auto step = step_tensor.flat<int64>();
        if (verbosity != 0 && step(0) % verbosity == 0 ) {
            std::string str_suffix = std::to_string(logfile_suffix);
            std::string logs_suffix = std::to_string(logs_path_suffix);
            std::string str_step = std::to_string(step(0));
            std::string str = "logs" + logs_suffix + "/step_" + str_step + "/" + str_suffix + "/intdecompressor_logs_" + str_suffix + "_" + std::to_string(suffix) + ".txt";
            FILE* f = fopen(str.c_str(),"w");
            fprintf(f, "input_tensor: %s\n", input_tensor.DebugString(input_tensor_flat.size()).c_str());
            fprintf(f, "Output_concat_size: = %d\n\n", output_concat_dim);
            fprintf(f, "recoveredsize: = %d\n\n", recoveredsize);
            fprintf(f, "intdecompressed_tensor: %s\n", output->DebugString(output_flat.size()).c_str());
            fprintf(f, "\n\n########################################################################################\n\n");
            fclose(f);
        }
        // *********************** For Debugging ********************** //

    }

private:
    int logfile_suffix;     // For debugging
    int logs_path_suffix;   // For debugging
    int suffix;             // For debugging
    int verbosity;          // For debugging
    string code;
};

REGISTER_KERNEL_BUILDER(Name("IntegerCompressor").Device(DEVICE_CPU), IntegerCompressorOp);

REGISTER_KERNEL_BUILDER(Name("IntegerDecompressor").Device(DEVICE_CPU), IntegerDecompressorOp);

