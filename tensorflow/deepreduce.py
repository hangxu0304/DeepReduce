"""
DeepReduce compression algorithms.
Author:Kelly Kostopoulou
"""

from __future__ import division
import tensorflow as tf
from tensorflow.python.framework import load_library
from tensorflow.python.platform import resource_loader
import math
from horovod.tensorflow.mpi_ops import rank

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


class Compressor(object):
    """Interface for compressing and decompressing a given tensor."""

    residuals = {}

    @staticmethod
    def compress(tensor, params):
        """Compresses a tensor and returns a list of compressed tensors with the context needed to decompress it."""
        pass

    @staticmethod
    def decompress(tensors, ctx, params):
        """Decompress a list of compressed tensors with the given context."""
        pass

    @classmethod
    def memory_compensate(cls, tensor, params):
        """Update the tensor with the residuals."""
        use_memory = params['use_memory']
        beta = params['beta']
        gamma = params['gamma']
        if use_memory:
            name = tensor.name
            cls.residuals[tensor.name] = tf.Variable(tf.zeros_like(tensor), trainable=False)
            tensor = beta * cls.residuals[name] + gamma * tensor
        return tensor

    @classmethod
    def memory_update(cls, tensor, tensor_compensate, tensor_compressed, ctx, params):
        """Update the residuals."""
        use_memory = params['use_memory']
        if use_memory:
            name = tensor.name
            tensor_decompressed = cls.decompress(tensor_compressed, ctx, params)
            delta = tensor_compensate - tensor_decompressed
            memory_update_op = cls.residuals[name].assign(delta)
        return [memory_update_op] if use_memory else []

    @staticmethod
    def aggregate(tensors, params):
        """Aggregate a list of tensors."""
        average = params['average']
        agged_tensor = tf.math.add_n(tensors)
        horovod_size = tf.cast(params["horovod_size"], dtype=agged_tensor.dtype)
        agged_tensor = (agged_tensor / horovod_size) if average else agged_tensor
        return agged_tensor

    
class Values_Approximation_Helper(Compressor):

    @staticmethod
    def double_exponential_fit(X_, Y_, K):

        # S, SS initialization
        Ysum = Y_ + tf.roll(Y_, shift=-1, axis=0)
        Xsum = tf.roll(X_, shift=-1, axis=0) - X_
        S = tf.tensor_scatter_nd_update(tf.roll(0.5 * Ysum * Xsum, shift=1, axis=0), [[0]], tf.zeros(1, tf.float64))
        S = tf.math.cumsum(S)
        Ssum = S + tf.roll(S, shift=-1, axis=0)
        SS = tf.tensor_scatter_nd_update(tf.roll(0.5 * Ssum * Xsum, shift=1, axis=0), [[0]], tf.zeros(1, tf.float64))
        SS = tf.math.cumsum(SS)

        sum_SSk_squared = tf.math.reduce_sum(tf.math.pow(SS, 2))
        sum_SSk_Sk = tf.math.reduce_sum(S * SS)
        sum_SSk_xk = tf.math.reduce_sum(SS * X_)
        sum_SSk = tf.math.reduce_sum(SS)
        sum_Sk_squared = tf.math.reduce_sum(tf.math.pow(S, 2))
        sum_Sk_xk = tf.math.reduce_sum(S * X_)
        sum_Sk = tf.math.reduce_sum(S)
        sum_data_x = tf.cast(K * (K + 1) / 2, tf.float64)
        sum_data_x_squared = tf.cast(K * (K + 1) * (2 * K + 1) / 6, tf.float64)
        K = tf.cast(K, tf.float64)

        # Form the first system
        values = tf.stack([sum_SSk_squared, sum_Sk_squared, sum_data_x_squared, K,
                            sum_SSk_Sk, sum_SSk_xk, sum_SSk, sum_Sk_xk, sum_Sk, sum_data_x], axis=0)

        A_LS_1 = tf.scatter_nd([[0, 0], [1, 1], [2, 2], [3, 3],
                                [0, 1], [0, 2], [0, 3],
                                [1, 2], [1, 3],
                                [2, 3]],
                               values, [4, 4])
        A_LS_1 = tf.tensor_scatter_nd_update(A_LS_1,
                                             [[0, 0], [1, 1], [2, 2], [3, 3],
                                              [1, 0], [2, 0], [3, 0],
                                              [2, 1], [3, 1],
                                              [3, 2]],
                                             values)

        a = tf.math.reduce_sum(tf.transpose(SS) * Y_)
        b = tf.math.reduce_sum(tf.transpose(S) * Y_)
        c = tf.math.reduce_sum(tf.transpose(X_) * Y_)
        d = tf.math.reduce_sum(Y_)

        b_vector_1 = tf.stack([a, b, c, d], axis=0)
        b_vector_1 = tf.reshape(b_vector_1, [4, 1])

        # Solve the first system
        Coefficient_vector_1 = tf.linalg.solve(A_LS_1, b_vector_1)

        # Calculate p1 and q1
        p1 = 0.5 * (Coefficient_vector_1[1] + tf.math.sqrt(
            tf.math.pow(Coefficient_vector_1[1], 2) + 4 * Coefficient_vector_1[0]))
        q1 = 0.5 * (Coefficient_vector_1[1] - tf.math.sqrt(
            tf.math.pow(Coefficient_vector_1[1], 2) + 4 * Coefficient_vector_1[0]))

        beta_k = tf.math.exp(p1 * X_)
        eta_k = tf.math.exp(q1 * X_)

        sum_betak_square = tf.math.reduce_sum(tf.math.pow(beta_k, 2))
        sum_etak_square = tf.math.reduce_sum(tf.math.pow(eta_k, 2))
        sum_betak_etak = tf.math.reduce_sum(beta_k * eta_k)

        # Form the second system
        A_LS_2 = tf.stack([sum_betak_square, sum_betak_etak, sum_betak_etak, sum_etak_square], axis=0)
        A_LS_2 = tf.reshape(A_LS_2, [2, 2])
        a = tf.reshape(tf.math.reduce_sum(tf.transpose(beta_k) * Y_), [1, ])
        b = tf.reshape(tf.math.reduce_sum(tf.transpose(eta_k) * Y_), [1, ])
        b_vector_2 = tf.stack([a, b], axis=0)
        b_vector_2 = tf.reshape(b_vector_2, [2, 1])

        # Solve the second system
        Coefficient_vector_2 = tf.linalg.solve(A_LS_2, b_vector_2)

        # print("Coefficient_vector_1: \n", Coefficient_vector_1)
        # print("p1:\n", p1)
        # print("Coefficient_vector_2:\n", Coefficient_vector_2)
        # print("q1:\n", q1)
        return Coefficient_vector_2[0], Coefficient_vector_2[1], p1, q1

    @staticmethod
    def logit_basis(X, a, N):  # log(p/(1-p))
        return tf.cast(a * tf.math.log(X / ((N + 1) - X)), dtype=tf.float64)

    @staticmethod
    def exp_basis(X, b, c):
        return tf.cast(b * tf.math.exp(c * X), dtype=tf.float64)

    @staticmethod
    def polynomial_basis(X, a):
        return tf.cast(tf.pow(X, a), dtype=tf.float64)

    @staticmethod
    def GetInputMatrix_Polynomial(xcol, x):
        N = tf.size(x)
        Xtrans = tf.ones([1, N], tf.float64)
        for i in range(1, xcol):
            basis = Values_Approximation_Helper.polynomial_basis(x, i)
            Xtrans = tf.concat([Xtrans, basis], axis=0)
        return tf.transpose(Xtrans)

    @staticmethod
    def find_breaks(y_train, num_of_segments, N):
        b = tf.constant(0, shape=(1,), dtype=tf.int32)
        N = tf.constant(N, shape=(1,), dtype=tf.int32)
        y = y_train
        break_points = tf.zeros(num_of_segments + 1, tf.int32)
        for i in range(num_of_segments - 1):
            a = tf.math.argmax(tf.abs(tf.linspace(y[0], y[-1], tf.size(y)) - y))
            b = b + tf.cast(a, tf.int32)
            break_points = tf.tensor_scatter_nd_update(break_points, [[i + 1]], b)
            y = tf.gather(y_train, tf.range(b[0], N[0]))
        break_points = tf.tensor_scatter_nd_update(break_points, [[num_of_segments]], N)
        sizes = [break_points[i + 1] - break_points[i] for i in range(num_of_segments)]
        return break_points, sizes

    @staticmethod
    def get_breaks(model, N):
        if model == "resnet20_v2":
            breaks = {
                432 : [0, 353, 432],
                2304 : [0, 1847, 2229, 2304],
                4608 : [0, 4073, 4544, 4608],
                9216 : [0, 8164, 9012, 9216],
                18432 : [0, 16094, 18060, 18432],
                36864 : [0, 33742, 36595, 36864]}
        elif model == "vgg16":
            breaks = {
                1728 : [0, 1443, 1663, 1728],
                36864 : [0, 34097, 36467, 36815, 36864],
                73728 : [0, 67595, 73032, 73630, 73728],
                147456 : [0, 132193, 145286, 147125, 147456],
                294912 : [0, 272485, 292623, 294580, 294844, 294912],
                589824 : [0, 553577, 586620, 589431, 589764, 589824],
                1179648 : [0, 1099105, 1172811, 1179005, 1179543, 1179648],
                2359296 : [0, 2195844, 2343594, 2357633, 2359102, 2359296]}
        elif model == "resnet50":
            breaks = {
                4096 : [0, 3656, 4018, 4096],
                9408 : [0, 8476, 9165, 9408],
                16384 : [0, 14406, 16145, 16327, 16384],
                36864 : [0, 32238, 36292, 36726, 36864],
                131072 : [0, 121069, 130381, 130989, 131072],
                32768 : [0, 29429, 32320, 32692, 32768],
                147456 : [0, 133258, 145944, 147255, 147456],
                65536 : [0, 58690, 64507, 65371, 65536],
                524288 : [0, 494762, 522078, 524067, 524238, 524288],
                589824 : [0, 539407, 584654, 589214, 589738, 589824],
                262144 : [0, 237433, 259437, 261782, 262062, 262144],
                2097152 : [0, 1990620, 2088919, 2096322, 2097036, 2097152],
                2359296 : [0, 2188168, 2341896, 2356580, 2358793, 2359296],
                1048576 : [0, 981145, 1041707, 1047784, 1048461, 1048576],
                2050048 : [0, 1980923, 2044274, 2049225, 2049929, 2050048]}
        return breaks[N]


    @staticmethod
    def LeastSquares(X, y):  # returns (X'X)^-1 X'y
        Xtrans = tf.transpose(X)
        tmp = tf.matmul(Xtrans, X)
        inverse = tf.linalg.inv(tmp)
        theta_estimates = tf.matmul(tf.matmul(inverse, Xtrans), y)
        return theta_estimates

    @staticmethod
    def is_convolutional(model, N):
        # print(model)
        if model == "resnet20_v2":
            conv_sizes = [432, 2304, 4608, 9216, 18432, 36864]
        elif model == "vgg16":
            conv_sizes = [1728, 36864, 73728, 147456, 294912, 589824, 1179648, 2359296]
        elif model == "resnet50":
            conv_sizes = [4096, 9408, 16384, 36864, 131072, 32768, 147456, 65536, 524288,
                          589824, 262144, 2097152, 2359296, 1048576, 2050048]
        if N in conv_sizes:
            return True
        return False

    @staticmethod
    def get_num_of_segments(model, N):
        if model == "resnet20_v2":
            segments = {432:2, 2304:3, 4608:3, 9216:3, 18432:3, 36864:3}
        elif model == "vgg16":
            segments = {1728:3, 36864:4, 73728:4, 147456:4, 294912:5, 589824:5, 1179648:5, 2359296:5}
        elif model == "resnet50":
            segments = {  4096:3, 9408:3, 16384:4, 36864:4, 131072:4, 32768:4, 147456:4, 65536:4, 524288:5,
                          589824:5, 262144:5, 2097152:5, 2359296:5, 1048576:5, 2050048:5}
        return segments[N]
    

# Bloom on CPU
class BloomFilterCompressor(Compressor):
    global_step = 0
    """"""
    @staticmethod
    def bloom_configuration(k, fpr):
        # Given FPR compute M and H
        m = (k*abs(math.log(fpr))) / (math.pow(math.log(2), 2))
        # Give bloom size in number of bytes ; bloom size must be a multiple of 8
        m = int(m/8)
        rem = m % 8
        if rem != 0 or m == 0:
            m += 1
        h = (m*8 / k)*math.log(2)
        h = int(math.ceil(h))
        return m, h

    @staticmethod
    def topk_indices(tensor, K):
        _, indices = tf.math.top_k(tf.math.abs(tensor), K, sorted=False)
        indices = tf.sort(indices, axis=0, direction='ASCENDING')
        return indices

    @staticmethod
    def threshold_indices(tensor, params):
        tensor_flatten = tf.reshape(tensor, [-1])
        threshold_val = tf.constant(params["threshold_val"], dtype=tf.float32)
        threshold_val = tf.math.minimum(threshold_val, tf.reduce_max(tf.abs(tensor_flatten)))
        thr_mask = tf.math.greater_equal(tf.math.abs(tensor_flatten), threshold_val)
        indices = tf.reshape(tf.where(thr_mask), [-1])
        indices = tf.sort(indices, axis=0, direction='ASCENDING')
        indices = tf.cast(indices, dtype=tf.int32)
        return indices

    @staticmethod
    def randomk_indices(tensor_name, N, K):
        all_indices = tf.range(N, dtype=tf.int32)
        h = hash(tensor_name) + BloomFilterCompressor.global_step
        tf.compat.v1.set_random_seed(1)
        indices = tf.random.shuffle(all_indices, seed=h)[:K]
        indices = tf.sort(indices, axis=0, direction='ASCENDING')
        BloomFilterCompressor.global_step += 1
        return indices

    @staticmethod
    def compress(tensor, params):

        tensor_shape = tf.shape(tensor)
        tensor_flatten = tf.reshape(tensor, [-1])
        elemnum = tensor_flatten.get_shape().as_list()[0]

        compress_ratio = params["compress_ratio"]
        k = max(1, int(elemnum * compress_ratio))

        # Configure bloom filter's m, k values
        assert params["bloom_fpr"] is not None, "False Positive Rate is None"
        params['m'], params['k'] = BloomFilterCompressor.bloom_configuration(k, params["bloom_fpr"])
        assert params['k'] < 256, "Number of hash functions too big"

        # params["bloom_config"].add_data(k, params['m']*8, params['k'], params["bloom_fpr"])
        # params["throughput_info"].add_data(elemnum, elemnum/8,  params['m']*8, (params['m']*8)/8, elemnum-params['m']*8, (elemnum-params['m']*8)/8)

        if params['bloom_on'] == "topk":            # Topk Sparsification Method
            indices = BloomFilterCompressor.topk_indices(tensor_flatten, k)
        elif params['bloom_on'] == "randomk":       # Randomk Sparsification Method
            indices = BloomFilterCompressor.randomk_indices(tensor.name, elemnum, k)
        elif params['bloom_on'] == "threshold":       # threshold Sparsification Method
            indices = BloomFilterCompressor.threshold_indices(tensor, params)

        values = tf.gather(tensor_flatten, indices)
        values = tf.bitcast(values, tf.int32)

        filename = resource_loader.get_path_to_datafile('bloom_filter_compression.so')
        library = load_library.load_op_library(filename)
        bloom_compressor = library.bloom_compressor

        log_initial_tensor = tf.bitcast(tensor_flatten, tf.int32)
        compressed_tensor = bloom_compressor(values, indices, log_initial_tensor, tf.train.get_or_create_global_step(),
                                             false_positives_aware=params['bloom_false_positives_aware'],
                                             policy=params['bloom_policy'],
                                             fpr=params["bloom_fpr"],
                                             hash_num=params['k'],
                                             bloom_size=params['m'],
                                             bloom_logs_path=params['bloom_logs_path'],
                                             gradient_id=params['gradient_id'],
                                             verbosity_frequency=params['bloom_verbosity_frequency'],
                                             verbosity=params['bloom_verbosity'],
                                             rank=rank())
        ctx = tensor_shape
        params['tensors_size_are_same'] = False
        return compressed_tensor, ctx

    @staticmethod
    def decompress(compressed_tensor, ctx, params):

        tensor_shape = ctx
        tensor_size = tf.math.reduce_prod(tensor_shape)

        filename = resource_loader.get_path_to_datafile('bloom_filter_compression.so')
        library = load_library.load_op_library(filename)
        bloom_decompressor = library.bloom_decompressor

        decompressed_tensor = bloom_decompressor(compressed_tensor, tensor_size,
                                                 tf.train.get_or_create_global_step(),
                                                 policy=params['bloom_policy'],
                                                 mem_mode=params['mem_mode'],
                                                 hash_num=params['k'],
                                                 bloom_size=params['m'],
                                                 bloom_logs_path=params['bloom_logs_path'],
                                                 gradient_id=params['gradient_id'],
                                                 verbosity_frequency=params['bloom_verbosity_frequency'],
                                                 verbosity=params['bloom_verbosity'],
                                                 suffix=params['suffix'],
                                                 rank=rank())

        decompressed_tensor = tf.bitcast(decompressed_tensor, tf.float32)
        decompressed_tensor = tf.reshape(decompressed_tensor, tensor_shape)
        return decompressed_tensor
    

# Double exponential on GPU
class DoubleExpCompressor(Compressor):
    # Values_Approximation_Compressor + TopK

    @staticmethod
    def compress(tensor, params):
        tensor_shape = tf.shape(tensor)
        tensor_flatten = tf.reshape(tensor, [-1])
        N = tensor_flatten.get_shape().as_list()[0]
        params['N'] = int(N)
        # print("Tensor", tensor, "size:", params['N'])

        elemnum = tensor_flatten.get_shape().as_list()[0]
        compress_ratio = params["compress_ratio"]

        k = max(1, int(elemnum * compress_ratio))
        params['K'] = k
        abs_vals, indices = tf.math.top_k(tf.math.abs(tensor_flatten), k, )  # sorted by default, desending order
        values = tf.gather(tensor_flatten, indices)
        # Values Approximation
        if elemnum>9000:
            #  Indices have a negative sign if they correspond to negative values and positive otherwise
            indices = tf.cast(indices, dtype=tf.int32)
            indices = (indices + 1) * (tf.cast(values > 0, dtype=tf.int32) * 2 - 1)

            mapping = tf.argsort(abs_vals, axis=0, direction='ASCENDING')
            abs_vals = tf.gather(abs_vals, mapping)
            indices = tf.gather(indices, mapping)

            # Fitting the curve
            X = tf.cast(tf.range(1, k + 1), tf.float64)
            coefficients = Values_Approximation_Helper.double_exponential_fit(X, tf.cast(abs_vals, tf.float64), k)

            coefficients = tf.reshape(coefficients, [-1])

            tensor_compressed = indices, coefficients
            params['X_train'] = X

        else:       # No approximation
            tensor_compressed = indices, values

        ctx = tensor_shape, elemnum
        params['tensors_size_are_same'] = True
        return tensor_compressed, ctx

    @staticmethod
    def decompress(tensor_compressed, ctx, params):
        tensor_shape, elemnum = ctx
        tensor_size = tf.math.reduce_prod(tensor_shape)

        if elemnum>9000:
            indices, message = tensor_compressed

            y_estimates = message[0] * tf.math.exp(message[2] * params['X_train']) + \
                          message[1] * tf.math.exp(message[3] * params['X_train'])

            values = tf.reshape(y_estimates, [-1])

            values = tf.cast(values, dtype=tf.float32) * tf.cast(tf.math.sign(indices), dtype=tf.float32)
            indices = tf.math.abs(indices) - 1

        else:
            indices, values = tensor_compressed

        tensor_decompressed = tf.scatter_nd(tf.expand_dims(indices, 1), values, [tensor_size])
        tensor_decompressed = tf.reshape(tensor_decompressed, tensor_shape)
        return tensor_decompressed
    

# PolyFit on GPU
class PolySegCompressor(Compressor):

    @staticmethod
    def compress(tensor, params):
        tensor_shape = tf.shape(tensor)
        tensor_flatten = tf.reshape(tensor, [-1])
        N = tensor_flatten.get_shape().as_list()[0]
        compress_ratio = params["compress_ratio"]
        params['N'] = params['K'] = int(N)

        # print("Tensor", tensor, "size:", params['N'])

        if Values_Approximation_Helper.is_convolutional(params['model_name'], params['N']):

            abs_values = tf.math.abs(tensor_flatten)

            if params['approximation_mode'] == "topk":
                K = max(1, int(N * compress_ratio))
                params['K'] = K
                top_values, mapping = tf.math.top_k(abs_values, K, sorted=False)
                sorted_mapping = tf.argsort(top_values, axis=0, direction='ASCENDING')
                values = tf.gather(top_values, sorted_mapping)
                mapping = tf.gather(mapping, sorted_mapping)
            else:
                mapping = tf.argsort(abs_values, axis=0, direction='ASCENDING')
                values = tf.gather(abs_values, mapping)

            # Indices have a negative sign if they correspond to negative values and positive otherwise
            negative_indices = tf.where(tf.less(tf.gather(tensor_flatten, mapping), 0))
            Nneg = tf.size(negative_indices)
            mask = tf.tensor_scatter_nd_update(tf.ones([params['K']], dtype=tf.int32), negative_indices,
                                               -tf.ones(Nneg, dtype=tf.int32))
            mapping = (mapping + 1) * mask

            # Fitting the curve segments
            params['num_of_segments'] = Values_Approximation_Helper.get_num_of_segments(params['model_name'], params['N'])
            # break_points, sizes = Values_Approximation_Helper.find_breaks(values, params['num_of_segments'], params['K'])
            break_points = Values_Approximation_Helper.get_breaks(params['model_name'], N)
            sizes = [break_points[i+1]-break_points[i] for i in range(params['num_of_segments'])]

            # params['X'] = {}
            coefficients = []
            for i in range(params['num_of_segments']):
                x = tf.reshape(tf.cast(tf.range(0, sizes[i]), tf.float64), [1, sizes[i]])
                X = Values_Approximation_Helper.GetInputMatrix_Polynomial(params['polynomial_degree'], x)
                y = tf.reshape(values[break_points[i]: break_points[i+1]], [sizes[i], 1])
                coefficients += [Values_Approximation_Helper.LeastSquares(X, tf.cast(y, tf.float64))]
                # params['X'][i] = X
            coefficients = tf.convert_to_tensor(coefficients)
            coefficients = tf.reshape(coefficients, [-1])

            ##################### Logging #####################
            # filename = resource_loader.get_path_to_datafile('mpi_lib.cpython-36m-x86_64-linux-gnu.so')
            # library = load_library.load_op_library(filename)
            # logger = library.logger
            # logger = logger(tensor_flatten, coefficients, tf.train.get_or_create_global_step(),
            #                 bloom_logs_path=params['bloom_logs_path'],
            #                 gradient_id=params['gradient_id'],
            #                 verbosity_frequency=params['bloom_verbosity_frequency'],
            #                 verbosity=params['bloom_verbosity'],
            #                 rank=rank())
            ##################### / Logging #####################

            compressed_indices = mapping  # Possible indices compression
            # with tf.control_dependencies([logger]):
            sizes = tf.cast(sizes, tf.float64)
            compressed_indices = tf.cast(compressed_indices, tf.float64)
            tensor_compressed = tf.concat([sizes, coefficients, compressed_indices], 0)
                # tensor_compressed = tf.concat([coefficients, compressed_indices], 0)
        else:
            tensor_compressed = tensor

        ctx = tensor_shape
        params['tensors_size_are_same'] = True
        return tensor_compressed, ctx

    @staticmethod
    def decompress(tensor_compressed, ctx, params):
        tensor_shape = ctx

        if Values_Approximation_Helper.is_convolutional(params['model_name'], params['N']):
            sizes, coefficients, indices = tf.split(tensor_compressed, [params['num_of_segments'],
                                                                        params['polynomial_degree'] * params[
                                                                            'num_of_segments'],
                                                                        params['K']])
            # coefficients, indices = tf.split(tensor_compressed, [params['polynomial_degree'] *
            #                                                             params['num_of_segments'],
            #                                                             params['N']])
            coefficients = tf.reshape(coefficients, [params['num_of_segments'], params['polynomial_degree']])
            decompressed_indices = tf.cast(indices, tf.int32)
            sizes = tf.cast(sizes, tf.int32)
            negative_indices = tf.where(tf.less(decompressed_indices, 0))
            decompressed_indices = tf.math.abs(decompressed_indices)
            decompressed_indices = decompressed_indices - 1
            Nneg = tf.size(negative_indices)
            mask = tf.tensor_scatter_nd_update(tf.ones(tf.shape(decompressed_indices), dtype=tf.int32), negative_indices,
                                               -tf.ones(Nneg, dtype=tf.int32))
            y_segments = []
            for i in range(params['num_of_segments']):
                x = tf.reshape(tf.cast(tf.range(0, sizes[i]), tf.float64), [1, sizes[i]])
                X = Values_Approximation_Helper.GetInputMatrix_Polynomial(params['polynomial_degree'], x)
                # X = params['X'][i]
                y_segments += [tf.matmul(X, tf.reshape(coefficients[i], [params['polynomial_degree'], 1]))]
            values = tf.reshape(tf.concat(y_segments, axis=0), [params['K']])
            values = values * tf.cast(mask, tf.float64)

            decompressed_indices = tf.expand_dims(decompressed_indices, 1)
            tensor_decompressed = tf.scatter_nd(decompressed_indices, tf.cast(values, tf.float32), [params['N']])
            tensor_decompressed = tf.reshape(tensor_decompressed, tensor_shape)
        else:
            tensor_decompressed = tensor_compressed

        return tensor_decompressed