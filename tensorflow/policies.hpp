#ifndef policies_hpp
#define policies_hpp

#include <iostream>

#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/op.h"
#include "../../third_party/bloomfilter/inc/OrdinaryBloomFilter.hpp"

using namespace tensorflow;
#include <random>


class Policies {

public:

    explicit
    Policies(){}

    static int find(const Tensor& indices, int x) {
        auto indices_flat = indices.flat<int>();
        for (int i=0; i<indices_flat.size(); ++i) {   // Dummy lookup
            if (indices_flat(i) == x)
                return 1;
        }
        return 0;
    }

    static int get_policy_errors(const int K, const Tensor& indices, const std::vector<int>& selected_indices) {
        int policy_errors = 0;
        for (int i=0; i<K; i++) {
            int chosen_index = selected_indices[i];
            if (!find(indices, chosen_index)) {
                policy_errors++;
            }
        }
        return policy_errors;
    }

    static void build_conflict_sets(int N, bloom::OrdinaryBloomFilter<uint32_t>& bloom, std::map<int, std::unordered_set<int>>& conflict_sets) {
        // Iterating over the universe and collecting the conflict sets
        uint8_t hash_num = bloom.Get_numHashes();
        for (size_t i=0; i<N; i++) {
            if (bloom.Query(i)) {  // If it is positive
                for (uint8_t j=0; j<hash_num; j++) {
                    int hash = bloom.Get_Hash(i,j);
//                    std::set<int>& cs = conflict_sets[hash];
//                    if (std::find(cs.begin(), cs.end(), i) == cs.end()) {
                    conflict_sets[hash].insert(i);
//                    }
                }
            }
        }
    }

    static void transform_and_sort(std::map<int, std::unordered_set<int>>& conflict_sets, std::vector<std::unordered_set<int>>& conflict_sets_ordered) {

        typedef std::function<std::unordered_set<int>(std::pair<int, std::unordered_set<int>>)> Transformator;
        Transformator transformator = [](std::pair<int, std::unordered_set<int>> i) {
            return i.second;
        };
        std::transform(conflict_sets.begin(), conflict_sets.end(), conflict_sets_ordered.begin(), transformator);
        std::sort(conflict_sets_ordered.begin(), conflict_sets_ordered.end(), [](const std::unordered_set<int>& l, const std::unordered_set<int>& r) {
            return l.size() < r.size();
        });
    }

//    static void print_map(std::map<int, std::unordered_set<int>>& map) {
//        for (auto& it: map) {
//             printf("Key: %d, Values: ", it.first);
//             for (auto& itt : it.second)
//                printf("%d, ", itt);
//             printf("\n");
//        }
//    }
//    static void print_2d_vector(std::vector<std::unordered_set<int>>& vec) {
//        printf("Conflict Sets:\n");
//        for (auto& it: vec) {
//             printf("       {");
//             for (auto& itt : it) {
//                printf("%d, ", itt);
//             }
//             printf("}\n");
//        }
//    }
//    static void print_vector(std::vector<int>& vec) {
//        printf("\n[");
//        int i=0;
//        for (i = 0; i < vec.size()-1; i++) {
//            printf("%d, ", (int) vec[i]);
//        }
//        printf("%d]\n\n", (int) vec[i]);
//    }

    static int erase_intersection(std::unordered_set<int>& a, std::unordered_set<int>& b) {
        bool compromised=false;
        std::unordered_set<int>::iterator it;
        for (it = a.begin(); it != a.end();) {
            if (b.find(*it) != b.end()) {
                it = a.erase(it);
                compromised = true;
            } else {
                it++;
            }
        }
        return compromised;
    }

    static void choose_indices_from_conflict_sets(int K, int seed, std::vector<std::unordered_set<int>>& conflict_sets_ordered, std::vector<int>& selected_indices) {
        std::default_random_engine generator;
        generator.seed(seed);
        bool compromised;
        int random, left = K;
        std::unordered_set<int> selected_indices_set;
        while (left > 0) {             // Don't stop until you have selected K positives
            for (int i=0; i<conflict_sets_ordered.size() && left>0; i++) {
                std::unordered_set<int>& cset = conflict_sets_ordered[i];
                compromised = erase_intersection(cset, selected_indices_set);
                if (!compromised && cset.size()>0) {
                    std::uniform_int_distribution<int> distribution(0, cset.size()-1);
                    random = distribution(generator);
                    auto it = std::begin(cset); std::advance(it, random);
                    selected_indices_set.insert(*it);
                    cset.erase(it);
                    left--;
                }
            }
        }
        std::copy(selected_indices_set.begin(), selected_indices_set.end(), std::back_inserter(selected_indices));
        std::sort(selected_indices.begin(), selected_indices.end());
    }

    static void conflict_sets_policy(int N, int K, int seed, bloom::OrdinaryBloomFilter<uint32_t>& bloom, std::vector<int>& selected_indices) {
        std::map<int, std::unordered_set<int>> conflict_sets;
        build_conflict_sets(N, bloom, conflict_sets);
//print_map(conflict_sets);
        // Sort the conflict sets by their size
        std::vector<std::unordered_set<int>> conflict_sets_ordered;
        conflict_sets_ordered.resize(conflict_sets.size());
        transform_and_sort(conflict_sets, conflict_sets_ordered);
        // Collect selected indices
        choose_indices_from_conflict_sets(K, seed, conflict_sets_ordered, selected_indices);
     }

    static void leftmostK(int N, int K, bloom::OrdinaryBloomFilter<uint32_t>& bloom,
                                    std::vector<int>& selected_indices) {
        // Iterating over the universe and collecting the first K positives
        for (size_t i=0, left=K; i<N && left>0; i++) {
            if (bloom.Query(i)) {  // If it is positive
                selected_indices.push_back(i);
                left--;
            }
        }
     }


    static void randomK(int N, int K, int64 step, bloom::OrdinaryBloomFilter<uint32_t>& bloom,
                                    std::vector<int>& selected_indices) {
        // Iterating over the universe and creating P
        std::vector<int> P;
        for (size_t i=0; i<N; i++) {
            if (bloom.Query(i)) {  // If it is positive
                P.push_back(i);
            }
        }
        // Randomly choose K indices from P
        std::default_random_engine generator;
        generator.seed(step);
        int random;
        for (int i=0; i<K; i++) {
            std::uniform_int_distribution<int> distribution(0, P.size()-1);
            random = distribution(generator);
            auto it = std::begin(P); std::advance(it, random);
            selected_indices.push_back(*it);
            P.erase(it);
        }
     }

     static void select_indices(std::string policy, int N, int K, int64 step,
                                bloom::OrdinaryBloomFilter<uint32_t>& bloom,
                                std::vector<int>& selected_indices) {
        if (policy == "conflict_sets") {
            conflict_sets_policy(N, K, step, bloom, selected_indices);
        } else if (policy == "leftmostK") {
            leftmostK(N, K, bloom, selected_indices);
        } else if (policy == "randomK") {
            randomK(N, K, step, bloom, selected_indices);
        } else if (policy == "policy_zero") {
            leftmostK(N, N, bloom, selected_indices);
        }
    }

};

#endif
