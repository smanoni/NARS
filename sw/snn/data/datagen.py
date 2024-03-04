# Copyright 2023 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Data Generation file for SNNs 0.1v
# TODO: Rewrite everything

import numpy as np
import random
import sys
from scipy.sparse import csr_matrix
import scipy.sparse
np.set_printoptions(threshold=sys.maxsize)

# Set Ifmap size, sparsity level and convolution params
A_H = 16
A_W = 16
A_C = 32
K   = 3
S   = 1
SPARSITY = 0.6 
NNZ_PERC = 1 - SPARSITY

# Define dimensions for ofmaps and membrane potential tensors
OF_ROWS = (A_H - K) // S + 1 
OF_COLS = (A_W - K) // S + 1
OF_CHAN = 16

# Define dimensions for weights tensor
W_HEIGHT = K * K * A_C
W_WIDTH  = OF_CHAN

# Calculate the number of non-zero values for matrix A based on the percentage
num_non_zero_a = int((A_H * A_W * A_C) * NNZ_PERC)

# Create a random 3D tensor with float values for spike ifmap
# to use SSRs on dense MatMul we need to use the float vals
spikes = np.zeros((A_C, A_H, A_W), dtype=np.float32)

# Get random indices for non-zero elements
indices_a = np.random.choice(range(A_H * A_W * A_C), num_non_zero_a, replace=False)
random.seed(42)
sorted_indices_a = sorted(indices_a)

# Set each non-zero element to a different random value, value 1 is set to spikes,
# for debugging is easier to use values different from 1 and 0.
for index in indices_a:
    #value = random.randint(1, 2000)  # Generate a random value between 1 and 2000
    #sign = random.choice([-1, 1])
    #value = sign * random.uniform(0, 1.2)
    value = 1
    spikes.flat[index] = value

# Flatten the 3D matrix into a 1D vector for matrix A
spikes_flat = []
for i in range(A_H):
    for j in range(A_W):
        for k in range(A_C):
            spikes_flat.append(spikes[k][i][j])

# Calculate the number of non-zero values for matrix B based on the percentage
num_non_zero_v = int((OF_ROWS * OF_COLS * OF_CHAN) * 0.6)

# Create a random 3D tensor with double values for matrix B (v_vals)
v_vals = np.zeros((OF_ROWS, OF_COLS, OF_CHAN), dtype=np.float64)

# Set a random subset of elements to non-zero values for matrix B (v_vals)
indices_v = np.random.choice(range(OF_ROWS * OF_COLS * OF_CHAN), num_non_zero_v, replace=False)
v_vals.flat[indices_v] = np.random.rand(num_non_zero_v)  # Random double values between 0 and 1

# Flatten the 3D matrix into a 1D vector for matrix B (v_vals)
v_vals_flat = v_vals.flatten()

# Calculate the number of non-zero values for matrix W based on the percentage
num_non_zero_w = int(W_HEIGHT * W_WIDTH * 0.6)

# Create a random 2D tensor with double values for matrix W (w_vals)
w_vals = np.zeros((W_HEIGHT, W_WIDTH), dtype=np.float64)

# Set a random subset of elements to non-zero values for matrix W (w_vals)
indices_w = np.random.choice(range(W_HEIGHT * W_WIDTH), num_non_zero_w, replace=False)
indices_w = range(W_HEIGHT * W_WIDTH)
w_val = 1
for index in indices_w:
    w_vals.flat[index] = np.random.uniform(-1, 1)

w_vals_flat = w_vals.flatten()

im2row_flat = []

# Generate im2row matrix golden model
im2row = np.empty([0,K*K*A_C])
for i in range(0, A_H - K + 1, S): # Move along rows
    for j in range(0, A_W - K + 1, S): # Move along cols
        for g in range(0, K):
            for h in range(0, K):   
                for y in range(0, A_C): # Move along channels   
                    im2row_flat.append(spikes[y][i+g][j+h])                    

im2row_gold_len = len(im2row_flat)/(K*K*A_C)
im2row = np.reshape(im2row_flat,(OF_ROWS * OF_COLS, K*K*A_C))
im2row_csr = csr_matrix(im2row)
np.set_printoptions(linewidth=1000)

w_vals_reshaped = w_vals_flat.reshape((W_HEIGHT, W_WIDTH))

# Perform matrix multiplication between im2row and w_vals_reshaped
mm = np.dot(im2row, w_vals_reshaped)

# Generate a matrix with random float numbers of the same size as mm
vth = np.random.rand(mm.shape[0], mm.shape[1])

# Save all four matrices as vectors to a .h file
with open('data.h', 'w') as f:
    f.write('#ifndef PROB_SNN_SNITCH_CC_H\n')
    f.write('#define PROB_SNN_SNITCH_CC_H\n\n')
    f.write('#define K ' + repr(K) + '\n')
    f.write('#define A_H ' + repr(A_H) + '\n')
    f.write('#define A_W ' + repr(A_W) + '\n')
    f.write('#define A_C ' + repr(A_C) + '\n')
    f.write('#define S ' + repr(S) + '\n\n')
    f.write('double __attribute__((section(".l1"))) spikes[A_H * A_W * A_C] = {\n')
    for i, val in enumerate(spikes_flat):
        f.write('    ' + str(val))
        if i < len(spikes_flat) - 1:
            f.write(',')
        f.write('\n')
    f.write('};\n\n')

    f.write('#define OF_ROWS ' + repr(OF_ROWS) + '\n')
    f.write('#define OF_COLS ' + repr(OF_COLS) + '\n')
    f.write('#define OF_CHAN ' + repr(OF_CHAN) + '// equal to the number of filters\n\n')
    f.write('double __attribute__((section(".l1"))) v_vals[OF_ROWS * OF_COLS * OF_CHAN] = {\n')
    for i, val in enumerate(v_vals_flat):
        f.write('    ' + str(val))
        if i < len(v_vals_flat) - 1:
            f.write(',')
        f.write('\n')
    f.write('};\n\n')

    f.write('double __attribute__((section(".l1"))) v_vals_bstatic[OF_ROWS * OF_COLS * OF_CHAN] = {\n')
    for i, val in enumerate(v_vals_flat):
        f.write('    ' + str(val))
        if i < len(v_vals_flat) - 1:
            f.write(',')
        f.write('\n')
    f.write('};\n\n')

    f.write('double __attribute__((section(".l1"))) v_vals_dense[OF_ROWS * OF_COLS * OF_CHAN] = {\n')
    for i, val in enumerate(v_vals_flat):
        f.write('    ' + str(val))
        if i < len(v_vals_flat) - 1:
            f.write(',')
        f.write('\n')
    f.write('};\n\n')

    f.write('double __attribute__((section(".l1"))) v_vals_dense_ssr[OF_ROWS * OF_COLS * OF_CHAN] = {\n')
    for i, val in enumerate(v_vals_flat):
        f.write('    ' + str(val))
        if i < len(v_vals_flat) - 1:
            f.write(',')
        f.write('\n')
    f.write('};\n\n')

    f.write('double __attribute__((section(".l1"))) v_vals_c_bstatic[OF_ROWS * OF_COLS * OF_CHAN] = {\n')
    for i, val in enumerate(v_vals_flat):
        f.write('    ' + str(val))
        if i < len(v_vals_flat) - 1:
            f.write(',')
        f.write('\n')
    f.write('};\n\n')

    f.write('#define W_HEIGHT ' + repr(W_HEIGHT) + '\n')
    f.write('#define W_WIDTH ' + repr(W_WIDTH) + '\n\n')
    f.write('double __attribute__((section(".l1"))) w_vals[W_HEIGHT * W_WIDTH] = {\n')
    for i, val in enumerate(w_vals.flat):
        f.write('    ' + str(val))
        if i < len(w_vals.flat) - 1:
            f.write(',')
        f.write('\n')
    f.write('};\n\n')

    f.write('#define VTH_ROWS ' + repr(vth.shape[0]) + '\n')
    f.write('#define VTH_COLS ' + repr(vth.shape[1]) + '\n\n')
    f.write('double __attribute__((section(".l1"))) vth[' + repr(vth.size) + '] = {\n')
    for i, val in enumerate(vth.flatten()):
        f.write('    ' + str(val))
        if i < len(vth.flatten()) - 1:
            f.write(',')
        f.write('\n')
    f.write('};\n\n')

    f.write('#define IM2ROW_GOLD_ROWS ' + repr(OF_ROWS * OF_COLS) + '\n')
    f.write('#define IM2ROW_GOLD_COLS ' + repr(K * K * A_C) + '\n')
    f.write('#define IM2ROW_GOLD_TOT ' + repr((K * K * A_C)* OF_ROWS * OF_COLS) + '\n')
    f.write('double __attribute__((section(".l1"))) im2row_gold[' + repr(im2row.shape[0]*im2row.shape[1]) + '] = {\n')
    for i, val in enumerate(im2row_flat):
        f.write('    ' + str(val))
        if i < len(im2row_flat) - 1:
            f.write(',')
        f.write('\n')
    f.write('};\n\n')

    f.write('#define CSR_IDX_GOLD_LEN ' + repr(len(im2row_csr.indices)) + '\n')
    f.write('#define CSR_RPTR_GOLD_LEN ' + repr(len(im2row_csr.indptr)) + '\n')

    f.write('double __attribute__((section(".l1"))) row_csr_val_gold[] = {\n')
    f.write(', '.join(map(str, im2row_csr.data)))
    f.write('};\n\n')

    f.write('int __attribute__((section(".l1"))) row_csr_idx_gold[] = {\n')
    f.write(', '.join(map(str, im2row_csr.indices)))
    f.write('};\n\n')

    f.write('int __attribute__((section(".l1"))) row_csr_rptr_gold[] = {\n')
    f.write(', '.join(map(str, im2row_csr.indptr)))
    f.write('};\n\n')
    
    f.write('#define RESULT_ROWS ' + repr(mm.shape[0]) + '\n')
    f.write('#define RESULT_COLS ' + repr(mm.shape[1]) + '\n\n')
    f.write('double __attribute__((section(".l1"))) mm_gold[' + repr(mm.shape[0]*mm.shape[1]) + '] = {\n')
    for i, val in enumerate(mm.flatten()):
        f.write('    ' + str(val))
        if i < len(mm.flatten()) - 1:
            f.write(',')
        f.write('\n')
    f.write('};\n\n')

    f.write('#endif // PROB_SNN_SNITCH_CC_H\n')

#print("Matrix data saved as vectors in matrices.h")