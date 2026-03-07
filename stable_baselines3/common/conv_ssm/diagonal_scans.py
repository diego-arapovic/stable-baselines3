# ------------------------------------------------------------------------------
# Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is made available under the Nvidia Source Code License.
# To view a copy of this license, visit
# https://github.com/NVlabs/ConvSSM/blob/main/LICENSE
#
# Written by Jimmy Smith
# ------------------------------------------------------------------------------

import jax
from jax import lax, numpy as np
from .conv_ops import vmap_conv


# Scan functions
@jax.vmap
def conv_binary_operator(q_i, q_j):
    """Assumes 1x1 kernels
       :inputs q_i an q_j are tuples containing (A_i, BU_i) and (A_j, BU_j)
       :inputs A_i and A_j are (P,)
       :inputs BU_i and BU_j are bszxH_UxW_UxP
       :returns tuple where first entry AA is (P,)
                and second entry is bszxH_UxW_UxP"""

    A_i, BU_i = q_i
    A_j, BU_j = q_j

    # AA = convolve_1x1_kernels(A_j, A_i)
    AA = A_j * A_i
    A_jBU_i = np.expand_dims(A_j, (0, 1, 2)) * BU_i

    return AA, A_jBU_i + BU_j

@jax.vmap
def conv_binary_operator_reset(q_i, q_j):
    """ Binary operator for parallel scan of ConvSSM with resets.
        Args:
            q_i: tuple containing A_i (bsz, P), BU_i (bsz, H, W, P), c_i (bsz,)
            q_j: tuple containing A_j (bsz, P), BU_j (bsz, H, W, P), c_j (bsz,)
        Returns:
            new element (AA_out, BU_out, c_out)
    """
    A_i, BU_i, c_i = q_i
    A_j, BU_j, c_j = q_j

    # Expand reset flags for broadcasting
    c_j_A = np.expand_dims(c_j, -1)           # (bsz, 1) for A
    c_j_BU = np.expand_dims(c_j, (1, 2, 3))   # (bsz, 1, 1, 1) for BU

    # State transition (wipes A_i if reset)
    AA = A_j * (A_i * (1 - c_j_A) + c_j_A) # equally AA = (A_j * A_i) * (1 - c_j_A) + A_j * c_j_A

    # Accumulated input/hidden state
    A_j_exp = np.expand_dims(A_j, (1, 2))     # (bsz, 1, 1, P)
    A_jBU_i = A_j_exp * BU_i
    BU_out = A_jBU_i * (1 - c_j_BU) + BU_j # equally BU_out = (A_jBU_i + BU_j) * (1 - c_j_BU) + BU_j * c_j_BU

    # Reset propagation
    c_out = c_i * (1 - c_j) + c_j

    return AA, BU_out, c_out

def apply_convSSM_parallel(A, B, C, us, x0, d):
    """Compute the output sequence of the convolutional SSM
        given the input sequence using a parallel scan.
        Computes x_k = A * x_{k-1} + B * u_k
                 y_k = C * x_k     + D * U_k
        where * is a convolution operator.
    Args:
        A (complex64): Conv kernel A                (P,)
        B (complex64): input-to-state conv kernel   (k_B,k_B,U,P)
        C (complex64): state-to-output conv kernel  (k_c,k_c, P, U)
        us (float32): input sequence of features  (L,bsz,H, W, U)
        x0 (complex64): initial state               (bsz, H, W, P)
        d (float32/bool): optional sequence resets  (L, bsz)
    Returns:
        x_L (complex64): the last state of the SSM  (bsz, H, W, P)
        ys (float32): the conv SSM outputs        (L,bsz, H, W, U)
    """
    L = us.shape[0]
    bsz = us.shape[1]
    
    # Real-valued decomposition: conv(us, B) = conv(us, B.real) + 1j*conv(us, B.imag)
    # Avoids complex64 cast of us (halves memory) and uses 2 real convs instead of 4
    Bus = vmap_conv(B.real, us) + 1j * vmap_conv(B.imag, us)
    x0_expanded = np.expand_dims(A, (0, 1, 2)) * x0

    if d is None:
        Bus = Bus.at[0].add(x0_expanded)
        As = A * np.ones((L,) + A.shape)
        _, xs = lax.associative_scan(conv_binary_operator, (As, Bus))
    else:
        if d.shape != (L, bsz):
            raise ValueError(f"Reset array 'd' must have shape {(L, bsz)}, but got {d.shape}")
        
        # d[0] has shape (bsz,) -> expand to match x0 (bsz, 1, 1, 1)
        reset_mask = np.expand_dims(1 - d[0], (1, 2, 3))
        Bus = Bus.at[0].add(x0_expanded * reset_mask)
        
        # batch (L, bsz, P) for independent batch resets
        As = np.broadcast_to(A, (L, bsz, A.shape[0]))
        
        _, xs, _ = lax.associative_scan(conv_binary_operator_reset, (As, Bus, d))

    # Real-valued decomposition: conv(C, xs).real = conv(C.real, xs.real) - conv(C.imag, xs.imag)
    ys = 2 * (vmap_conv(C.real, xs.real) - vmap_conv(C.imag, xs.imag))

    return xs[-1], ys


def apply_convSSM_sequential(A, B, C, us, x0, d):
    """Compute the output sequence of the convolutional SSM
        given the input sequence sequentially. For testing purposes.
    Args:
        A (complex64): Conv kernel A                (P,)
        B (complex64): input-to-state conv kernel   (k_B,k_B,U,P)
        C (complex64): state-to-output conv kernel  (k_c,k_c, P, U)
        us (float32): input sequence of features  (L,bsz,H, W, U)
        x0 (complex64): initial state               (bsz, H, W, P)
    Returns:
        x_L (complex64): the last state of the SSM  (bsz, H, W, P)
        ys (float32): the conv SSM outputs        (L,bsz, H, W, U)
    """
    def step(x_k_1, inputs):
        if d is not None:
            u_k, c_k = inputs
            # Apply reset: Wipe the previous hidden state if c_k == 1
            x_k_1 = x_k_1 * (1 - c_k)
        else:
            u_k = inputs

        # Real-valued decomposition for B: conv(u, B) = conv(u, B.real) + 1j*conv(u, B.imag)
        Bu = (lax.conv_general_dilated(u_k, B.real, (1, 1), 'SAME',
               dimension_numbers=('NHWC', 'HWIO', 'NHWC'))
              + 1j * lax.conv_general_dilated(u_k, B.imag, (1, 1), 'SAME',
                 dimension_numbers=('NHWC', 'HWIO', 'NHWC')))
        x_k = np.expand_dims(A, (0, 1, 2)) * x_k_1 + Bu
        # Real-valued decomposition for C: conv(C, x).real = conv(C.real, x.real) - conv(C.imag, x.imag)
        y_k = 2 * (lax.conv_general_dilated(x_k.real, C.real, (1, 1), 'SAME',
                 dimension_numbers=('NHWC', 'HWIO', 'NHWC'))
                 - lax.conv_general_dilated(x_k.imag, C.imag, (1, 1), 'SAME',
                   dimension_numbers=('NHWC', 'HWIO', 'NHWC')))
        return x_k, y_k
    scan_inputs = (us, d) if d is not None else us
    return lax.scan(step, np.complex64(x0), scan_inputs)
