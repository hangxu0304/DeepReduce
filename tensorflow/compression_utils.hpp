#ifndef compression_utils_hpp
#define compression_utils_hpp

#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/op.h"

#include "../../third_party/bloomfilter/inc/OrdinaryBloomFilter.hpp"
#include "../../third_party/bloomfilter/inc/MurmurHash.hpp"
#include "../../third_party/bloomfilter/inc/FnvHash.hpp"
#include "policies.hpp"

#include <iostream>

class CompressionUtilities {

public:

    explicit
    CompressionUtilities(){}

    static void fprint(FILE* f, int size, std::vector<uint8_t> ptr) {
        unsigned int bit_pos, byte_pos, value, byte;
        fprintf(f, "Bitstream Array: \n [ ");

        for (byte_pos=0; byte_pos<size; byte_pos++) {
            for (bit_pos=0; bit_pos<8; bit_pos++) {
                byte = ptr[byte_pos];
                value = 1;
                value = value << bit_pos;
                value = value | byte;
                if ((value^byte) != 0)
                    fprintf(f, "0 ");
                else fprintf(f, "1 ");
            }
        }
        fprintf(f, "]\n\n");
    }

    static void fprint(FILE* f, int size, std::vector<int8_t> ptr) {
        unsigned int bit_pos, byte_pos, value, byte;
        fprintf(f, "Bitstream Array: \n [ ");

        for (byte_pos=0; byte_pos<size; byte_pos++) {
            for (bit_pos=0; bit_pos<8; bit_pos++) {
                byte = ptr[byte_pos];
                value = 1;
                value = value << bit_pos;
                value = value | byte;
                if ((value^byte) != 0)
                    fprintf(f, "0 ");
                else fprintf(f, "1 ");
            }
        }
        fprintf(f, "]\n\n");
    }

    static void print_vector(int* vec, int size, FILE* f) {
        fprintf(f, "\n[");
        int i=0;
        for (i = 0; i < size-1; i++) {
            fprintf(f, "%d, ", (int) vec[i]);
        }
        fprintf(f, "%d]\n\n", (int) vec[i]);
    }

    static void print_vector(int* vec, int size) {
        printf("\n[");
        int i=0;
        for (i = 0; i < size-1; i++) {
            printf("%d, ", (int) vec[i]);
        }
        printf("%d]\n\n", (int) vec[i]);
    }

    static void print_2d_vector(std::vector<std::vector<int>>& vec, FILE* f) {
        for (auto& it: vec) {
             fprintf(f, "{");
             for (auto& itt : it)
                fprintf(f, "%d, ", itt);
             fprintf(f, "}\n");
        }
    }

    static void print_map(std::map<int, std::vector<int>>& map, FILE* f) {
        for (auto& it: map) {
            fprintf(f, "Key: %d, Values: ", it.first);
             for (auto& itt : it.second)
                fprintf(f, "%d, ", itt);
             fprintf(f, "\n");
        }
    }

    // Logging for values approximation
    static void logging_compressor(bloom::OrdinaryBloomFilter<uint32_t>& bloom, int N, int K, int output_concat_dim,
    const Tensor& initial_tensor, const Tensor& indices, const Tensor& values, std::vector<int>& new_values, std::vector<int>& selected_indices,
    std::string bloom_logs_path, int gradient_id, int64 step, std::string policy, int rank, int verbosity) {

        FILE* f;
        std::string str;
        int false_positives = bloom.Compute_False_Positives(N, indices);
        int policy_errors = Policies::get_policy_errors(K, indices, selected_indices);
        std::string str_gradient_id = std::to_string(gradient_id);
        std::string str_step = std::to_string(step);
        std::string str_rank = std::to_string(rank);
        int bloom_size = bloom.Get_numBytes();

        std::string path = bloom_logs_path + "/" + str_rank + "/step_" + str_step + "/" + str_gradient_id + "/";
        std::string cmd = "mkdir -p " + path;
        int systemRet = system(cmd.c_str());
        if(systemRet == -1){
            perror("mkdir failed");
        }
        if (verbosity > 1) {
            str = path + "compressor_logs_" + policy + ".txt";
            f = fopen(str.c_str(),"w");
            fprintf(f, "\nInitial Tensor: %s\n\n", initial_tensor.DebugString(N).c_str());
            fprintf(f, "Values: %s\n", values.DebugString(K).c_str());
            fprintf(f, "\nIndices: %s\n\n", indices.DebugString(K).c_str());
//            fprintf(f, "Step: = %d\n\n", step);
            fprintf(f, "Bloom size: = %d\n", bloom_size);
            bloom.fprint(f);
            fprintf(f, "\nIndices Chosen:");
            print_vector(selected_indices.data(), K, f);
            if (new_values.size() > 0) {
                fprintf(f, "\nValues-Sent:");
                print_vector(new_values.data(), K, f);
            }
            fprintf(f, "Output_concat_size: = %d\n\n", output_concat_dim);
            fprintf(f, "FalsePositives: %d\n", false_positives);
            fprintf(f, "Total: %d\n", N);
            fprintf(f, "\n\n########################################################################################\n\n");
            fclose(f);
        }

        str = path + "fpr.txt";
        f = fopen(str.c_str(),"w");
        fprintf(f, "FalsePositives: %d  Total: %d\n", false_positives,  N);
        fclose(f);
        str = path + "policy_errors.txt";
        f = fopen(str.c_str(),"w");
        fprintf(f, "PolicyErrors: %d  Total: %d\n", policy_errors,  K);
        fclose(f);
        str = path + "stats.txt";
        f = fopen(str.c_str(),"w");
        fprintf(f, "Initial_Size: %d  Final_Size: %d\n", N /*in bits*/,  bloom_size*8 /*in bits*/);
        fclose(f);
    }

    static void logging_decompressor(bloom::OrdinaryBloomFilter<uint32_t>& bloom, int N, int K,
    int* values_vec, std::vector<int>& selected_indices, std::string bloom_logs_path, int gradient_id,
    int suffix, int64 step, Tensor* decompressed_tensor, std::string policy, int rank, int verbosity) {
        if (verbosity > 1) {
            FILE* f;
            std::string str_gradient_id = std::to_string(gradient_id);
            std::string str_step = std::to_string(step);
            std::string str_rank = std::to_string(rank);

            std::string path = bloom_logs_path + "/" + str_rank + "/step_" + str_step + "/" + str_gradient_id;
            std::string str = path + "/decompressor_logs_" + policy + "_" + std::to_string(suffix) + ".txt";
            f = fopen(str.c_str(),"w");
            fprintf(f, "decompressed size: %d\n\n", N);
//            fprintf(f, "Step: = %d\n\n", step);

            int bloom_size = bloom.Get_numBytes();
            fprintf(f, "Bloom size: = %d\n", bloom_size);
            bloom.fprint(f);
            fprintf(f, "\nIndices Chosen:");
            print_vector(selected_indices.data(), K, f);
            fprintf(f, "Values Received:"); print_vector(values_vec, K, f);
            fprintf(f, "Decompressed_tensor: %s\n", decompressed_tensor->DebugString(N).c_str());
            fprintf(f, "########################################################################################\n\n");
            fclose (f);
        }
    }

    // Logging for values approximation
    static void logging(int N, const Tensor& initial_tensor, const Tensor& coefficients,
    std::string bloom_logs_path, int gradient_id, int64 step, int rank, int verbosity) {

        FILE* f;
        std::string str;
        std::string str_gradient_id = std::to_string(gradient_id);
        std::string str_step = std::to_string(step);
        std::string str_rank = std::to_string(rank);

        std::string path = bloom_logs_path + "/" + str_rank + "/step_" + str_step + "/" + str_gradient_id + "/";
        std::string cmd = "mkdir -p " + path;
        int systemRet = system(cmd.c_str());
        if(systemRet == -1){
            perror("mkdir failed");
        }
//        if (verbosity > 1) {
//            str = path + "log.txt";
//            f = fopen(str.c_str(),"w");
//            fprintf(f, "\nInitial Tensor: %s\n\n", initial_tensor.DebugString(N).c_str());
//            fprintf(f, "\nEstimated Tensor: %s\n\n", coefficients.DebugString(N).c_str());
//            fprintf(f, "\nCoefficients: %s\n\n", coefficients.DebugString(num_of_coefficients).c_str());
//            fprintf(f, "Step: = %d\n\n", step);
//        }
        auto initial_flat = initial_tensor.flat<float>();
        auto coefficients_flat = coefficients.flat<double>();
        str = path + "values.csv";
        f = fopen(str.c_str(),"w");
        for(int i=0; i<N; i++) {
            fprintf(f, "%.40g\n", initial_flat(i));
        }
        fclose(f);

        str = path + "coefficients.csv";
        f = fopen(str.c_str(),"w");
        for(int i=0; i<coefficients_flat.size(); i++) {
            fprintf(f, "%.40g\n", coefficients_flat(i));
        }
        fclose(f);
    }

    // Logging for bitstream compression
    static void logging_bitstream_compressor(const Tensor& indices_tensor, int output_concat_dim, int tensor_size_bytes,
    std::vector<uint8_t>& bitstream, std::vector<int>* lengths, const Tensor* output, int initial_tensor_size,
    std::string logs_path, int gradient_id, int64 step, int rank, int verbosity) {

        FILE* f;
        std::string str;
        std::string str_gradient_id = std::to_string(gradient_id);
        std::string str_step = std::to_string(step);
        std::string str_rank = std::to_string(rank);

        std::string path = logs_path + "/" + str_rank + "/step_" + str_step + "/" + str_gradient_id + "/";
        std::string cmd = "mkdir -p " + path;
        int systemRet = system(cmd.c_str());
        if(systemRet == -1){
            perror("mkdir failed");
        }
        if (verbosity > 1) {
            str = path + "RleCompressor_logs.txt";
            f = fopen(str.c_str(),"w");
            auto indices_tensor_flat = indices_tensor.flat<int32_t>();
            fprintf(f, "indices_tensor: %s\n", indices_tensor.DebugString(indices_tensor_flat.size()).c_str());
            fprintf(f, "Output_concat_size: = %d\n\n", output_concat_dim);
            CompressionUtilities::fprint(f, tensor_size_bytes, bitstream);
            if (lengths != NULL) {
                fprintf(f, "Lengths:\n");
                CompressionUtilities::print_vector(lengths->data(), lengths->size(), f);
            }
            auto output_flat = output->template flat<uint8>();
            fprintf(f, "Encoded lengths: %s\n", output->DebugString(output_flat.size()).c_str());
            fprintf(f, "\n\n########################################################################################\n\n");
            fclose(f);
        }

        str = path + "stats.txt";
        f = fopen(str.c_str(),"w");
        fprintf(f, "Initial_Size: %d  Final_Size: %d\n", initial_tensor_size,  output_concat_dim*8 + 32); // in bits // worker sends the size of the encoded tensor
        fclose(f);
    }

    static void logging_bitstream_decompressor(const Tensor& encoding, int output_concat_dim, std::vector<int>* lengths,
    const Tensor* output, std::string logs_path, int gradient_id, int64 step, int rank, int suffix, int verbosity) {

        FILE* f;
        std::string str;
        std::string str_gradient_id = std::to_string(gradient_id);
        std::string str_step = std::to_string(step);
        std::string str_rank = std::to_string(rank);

        std::string path = logs_path + "/" + str_rank + "/step_" + str_step + "/" + str_gradient_id + "/";
        std::string cmd = "mkdir -p " + path;
        int systemRet = system(cmd.c_str());
        if(systemRet == -1){
            perror("mkdir failed");
        }
        if (verbosity > 1) {
            str = path + "RleDecompressor_logs" + "_" + std::to_string(suffix) + ".txt";
            f = fopen(str.c_str(),"w");

            auto encoding_flat = encoding.flat<uint8_t>();
            fprintf(f, "encoding_flat: %s\n", encoding.DebugString(encoding_flat.size()).c_str());
            fprintf(f, "Output_concat_size: = %d\n\n", output_concat_dim);
            if (lengths != NULL) {
                fprintf(f, "Lengths:\n");
                CompressionUtilities::print_vector(lengths->data(), lengths->size(), f);
            }
            auto output_flat = output->template flat<int32>();
            fprintf(f, "Indices: %s\n", output->DebugString(output_flat.size()).c_str());
            fprintf(f, "\n\n########################################################################################\n\n");
            fclose(f);
        }
    }

};

#endif
