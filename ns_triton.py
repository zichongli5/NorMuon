import torch
import triton
import triton.language as tl
from torch import Tensor

TRITON_AVAILABLE = True

# Triton's tl.dot input_precision is backend-specific: CUDA accepts
# "ieee"/"tf32"/"tf32x3", while ROCm only accepts "ieee" (see issue #61).
_IS_HIP = torch.version.hip is not None


def _reject_output_alias(out: Tensor, *inputs: Tensor) -> None:
    if out is not None and any(
        out.untyped_storage().data_ptr() == value.untyped_storage().data_ptr()
        for value in inputs
    ):
        raise ValueError("out must not share storage with an input tensor")


def _get_autotune_configs():
    # Symmetric kernels (line_1/line_2): square tiles only. With a
    # rectangular BM x BN tile the mirrored store region overlaps the
    # own-write region of neighboring programs; the values coincide only
    # because the Gram matrix is bitwise symmetric, which is a fragile
    # invariant and redundant work. Square tiles plus the LOWER_UPPER
    # above-diagonal skip make every output element written by exactly one
    # program. Small, safe search space to keep first-call autotune cheap.
    return [
        triton.Config(
            {
                "BLOCK_SIZE_M": bs,
                "BLOCK_SIZE_N": bs,
                "BLOCK_SIZE_K": bk,
                "GROUP_SIZE_M": 8,
                "LOWER_UPPER": 1,
            },
            num_stages=stages,
            num_warps=warps,
        )
        for bs in [64, 128]
        for bk in [64, 128]
        for stages, warps in [(3, 4), (4, 4)]
    ]


@triton.jit
def _batch_offset(batch_idx, batch_stride):
    return batch_idx.to(tl.int64) * batch_stride


@triton.jit
def _pid_to_block(
    pid,
    M,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    """
    Helper function to map Triton program ID to (batch, row, col) of the output matrix.
    """
    # Split output matrix into blocks of size (BLOCK_SIZE_M, BLOCK_SIZE_N)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(M, BLOCK_SIZE_N)

    # Map PID to a single matrix in batch
    batch_idx = pid // (num_pid_m * num_pid_n)
    pid = pid % (num_pid_m * num_pid_n)

    # Map PID to 2D grid of blocks
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n
    pid_m, pid_n = tl.swizzle2d(pid_m, pid_n, num_pid_m, num_pid_n, GROUP_SIZE_M)

    m_idx = pid_m * BLOCK_SIZE_M
    n_idx = pid_n * BLOCK_SIZE_N

    return batch_idx, m_idx, n_idx


@triton.autotune(
    configs=_get_autotune_configs(),
    key=["M", "K", "a_stride_r", "a_stride_c", "c_stride_r", "c_stride_c"],
)
@triton.jit
def ns_line_1_kernel(
    A_ptr,
    C_ptr,
    M,
    K,
    a_stride_b,
    a_stride_r,
    a_stride_c,
    c_stride_b,
    c_stride_r,
    c_stride_c,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    LOWER_UPPER: tl.constexpr,
    INPUT_PRECISION: tl.constexpr = "tf32",
):
    """
    Input A has shape (M, K)
    Output C has shape (M, M)
    Compute C = A @ A.T
    """

    pid = tl.program_id(axis=0)
    batch_idx, m_idx, n_idx = _pid_to_block(
        pid, M, BLOCK_SIZE_M, BLOCK_SIZE_N, GROUP_SIZE_M
    )

    # Skip blocks that don't need to be computed
    skip_block_below_diag = (LOWER_UPPER == 0) and (n_idx + BLOCK_SIZE_N <= m_idx)
    skip_block_above_diag = (LOWER_UPPER != 0) and (m_idx + BLOCK_SIZE_M <= n_idx)
    if skip_block_below_diag or skip_block_above_diag:
        return

    # Index into one matrix of batch
    A_ptr += _batch_offset(batch_idx, a_stride_b)
    C_ptr += _batch_offset(batch_idx, c_stride_b)

    # Create pointer arrays for A and A.T
    offs_m = (m_idx + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_n = (n_idx + tl.arange(0, BLOCK_SIZE_N)) % M
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = A_ptr + (offs_m[:, None] * a_stride_r + offs_k[None, :] * a_stride_c)
    at_ptrs = A_ptr + (offs_k[:, None] * a_stride_c + offs_n[None, :] * a_stride_r)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Accumulate over blocks of K
    for k in tl.range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        at = tl.load(at_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        accumulator = tl.dot(a, at, accumulator, input_precision=INPUT_PRECISION)
        a_ptrs += BLOCK_SIZE_K * a_stride_c
        at_ptrs += BLOCK_SIZE_K * a_stride_c

    out_dtype = C_ptr.dtype.element_ty
    output = accumulator.to(out_dtype)

    # Store block of C
    offs_cm = m_idx + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = n_idx + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = C_ptr + (offs_cm[:, None] * c_stride_r + offs_cn[None, :] * c_stride_c)
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < M)
    tl.store(c_ptrs, output, mask=c_mask)

    # Store block of C mirrored across the diagonal. With square tiles the
    # mirror region of a diagonal tile is the tile itself (m_idx == n_idx),
    # so that redundant write is skipped.
    if m_idx != n_idx:
        c_ptrs_t = C_ptr + (offs_cn[:, None] * c_stride_r + offs_cm[None, :] * c_stride_c)
        c_mask_t = (offs_cn[:, None] < M) & (offs_cm[None, :] < M)
        tl.store(c_ptrs_t, output.T, mask=c_mask_t)


def ns_line_1(A: Tensor, *, out: Tensor = None):
    """
    Launch Triton kernel to compute C = A @ A.T
    """
    if not TRITON_AVAILABLE:
        raise RuntimeError("ns_line_1 requires the 'triton' package")
    if A.ndim > 3 or A.ndim < 2:
        raise ValueError(f"Input tensor must be 2D or 3D, but got {A.ndim}D tensor.")
    _reject_output_alias(out, A)

    M, K = A.shape[-2:]

    expected = (*A.shape[:-1], M)
    if out is None:
        out = torch.empty(expected, device=A.device, dtype=A.dtype)
    if out.shape != expected:
        raise ValueError(f"out must have shape {expected}, got {out.shape}")
    if out.device != A.device:
        raise ValueError(f"out must be on the same device as A ({A.device}), got {out.device}")
    if out.dtype != A.dtype:
        raise ValueError(f"out dtype must match A dtype ({A.dtype}), got {out.dtype}")

    batch_size = A.size(0) if A.ndim == 3 else 1
    input_batch_stride = A.stride(0) if A.ndim == 3 else 0
    output_batch_stride = out.stride(0) if out.ndim == 3 else 0
    input_precision = "ieee" if (_IS_HIP or A.dtype == torch.float32) else "tf32"

    def grid(meta):
        return (
            batch_size
            * triton.cdiv(M, meta["BLOCK_SIZE_M"])
            * triton.cdiv(M, meta["BLOCK_SIZE_N"]),
        )
    ns_line_1_kernel[grid](
        A_ptr=A,
        C_ptr=out,
        M=M,
        K=K,
        a_stride_b=input_batch_stride,
        a_stride_r=A.stride(-2),
        a_stride_c=A.stride(-1),
        c_stride_b=output_batch_stride,
        c_stride_r=out.stride(-2),
        c_stride_c=out.stride(-1),
        INPUT_PRECISION=input_precision,
    )

    return out


@triton.autotune(
    configs=_get_autotune_configs(),
    key=["M", "a_stride_r", "a_stride_c", "c_stride_r", "c_stride_c"],
)
@triton.jit
def ns_line_2_kernel(
    A_ptr,
    C_ptr,
    M,
    a_stride_b,
    a_stride_r,
    a_stride_c,
    c_stride_b,
    c_stride_r,
    c_stride_c,
    alpha,
    beta,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    LOWER_UPPER: tl.constexpr,
    INPUT_PRECISION: tl.constexpr = "tf32",
):
    """
    Input A is square matrix with shape (M, M)
    Output C has shape (M, M)
    Compute C = alpha * A @ A.T + beta * A
    """

    pid = tl.program_id(axis=0)
    batch_idx, m_idx, n_idx = _pid_to_block(
        pid, M, BLOCK_SIZE_M, BLOCK_SIZE_N, GROUP_SIZE_M
    )

    # Skip blocks that don't need to be computed
    skip_block_below_diag = (LOWER_UPPER == 0) and (n_idx + BLOCK_SIZE_N <= m_idx)
    skip_block_above_diag = (LOWER_UPPER != 0) and (m_idx + BLOCK_SIZE_M <= n_idx)
    if skip_block_below_diag or skip_block_above_diag:
        return

    # Index into one matrix of batch
    A_ptr += _batch_offset(batch_idx, a_stride_b)
    C_ptr += _batch_offset(batch_idx, c_stride_b)

    # Create pointer arrays for A and A.T
    offs_m = (m_idx + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_n = (n_idx + tl.arange(0, BLOCK_SIZE_N)) % M
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = A_ptr + (offs_m[:, None] * a_stride_r + offs_k[None, :] * a_stride_c)
    at_ptrs = A_ptr + (offs_k[:, None] * a_stride_c + offs_n[None, :] * a_stride_r)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Accumulate over blocks of K
    for k in tl.range(0, tl.cdiv(M, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < M - k * BLOCK_SIZE_K, other=0.0)
        at = tl.load(at_ptrs, mask=offs_k[:, None] < M - k * BLOCK_SIZE_K, other=0.0)
        accumulator = tl.dot(a, at, accumulator, input_precision=INPUT_PRECISION)
        a_ptrs += BLOCK_SIZE_K * a_stride_c
        at_ptrs += BLOCK_SIZE_K * a_stride_c

    # Load block of A to add (corresponds to the current block of C)
    offs_am = m_idx + tl.arange(0, BLOCK_SIZE_M)
    offs_an = n_idx + tl.arange(0, BLOCK_SIZE_N)
    a_add_ptrs = A_ptr + (offs_am[:, None] * a_stride_r + offs_an[None, :] * a_stride_c)
    a_add_mask = (offs_am[:, None] < M) & (offs_an[None, :] < M)
    a_add = tl.load(a_add_ptrs, mask=a_add_mask, other=0.0).to(tl.float32)

    # Apply alpha and beta
    accumulator *= alpha
    accumulator += a_add * beta

    out_dtype = C_ptr.dtype.element_ty
    output = accumulator.to(out_dtype)

    # Store block of C
    offs_cm = m_idx + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = n_idx + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = C_ptr + (offs_cm[:, None] * c_stride_r + offs_cn[None, :] * c_stride_c)
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < M)
    tl.store(c_ptrs, output, mask=c_mask)

    # Store block of C mirrored across the diagonal. With square tiles the
    # mirror region of a diagonal tile is the tile itself (m_idx == n_idx),
    # so that redundant write is skipped.
    if m_idx != n_idx:
        # The mirrored element C[ln, lm] must carry alpha*G[ln, lm] + beta*A[ln, lm].
        # output[lm, ln] = alpha*G[lm, ln] + beta*A[lm, ln]; the Gram part is
        # bitwise symmetric (same products, same order), but the beta*A term is
        # not, so the transposed value must be corrected by beta*(A[ln,lm]-A[lm,ln]).
        # The correction is applied in fp32 before the store cast, and it is
        # exactly zero for symmetric inputs, keeping the NS pipeline path
        # bitwise unchanged.
        a_add_t_ptrs = A_ptr + offs_an[:, None] * a_stride_r + offs_am[None, :] * a_stride_c
        # Mask index order is transposed relative to a_add_mask because this
        # load has shape (BN, BM): element [i, j] is A[offs_an[i], offs_am[j]].
        a_add_t_mask = (offs_an[:, None] < M) & (offs_am[None, :] < M)
        a_add_t = tl.load(a_add_t_ptrs, mask=a_add_t_mask, other=0.0).to(tl.float32)
        mirrored = accumulator.T + (a_add_t - a_add.T) * beta
        output_t = mirrored.to(out_dtype)
        c_ptrs_t = C_ptr + (offs_cn[:, None] * c_stride_r + offs_cm[None, :] * c_stride_c)
        c_mask_t = (offs_cn[:, None] < M) & (offs_cm[None, :] < M)
        tl.store(c_ptrs_t, output_t, mask=c_mask_t)


def ns_line_2(A: Tensor, alpha: float, beta: float, *, out: Tensor = None):
    """
    Launch Triton kernel to compute C = alpha * A @ A.T + beta * A
    """
    if not TRITON_AVAILABLE:
        raise RuntimeError("ns_line_2 requires the 'triton' package")
    if A.ndim > 3 or A.ndim < 2:
        raise ValueError(f"Input tensor must be 2D or 3D, but got {A.ndim}D tensor.")
    _reject_output_alias(out, A)

    M, K = A.shape[-2:]
    if M != K:
        raise ValueError(
            f"Input must be symmetric square matrix, but got shape {A.shape}"
        )

    expected = (*A.shape[:-1], M)
    if out is None:
        out = torch.empty(expected, device=A.device, dtype=A.dtype)
    if out.shape != expected:
        raise ValueError(f"out must have shape {expected}, got {out.shape}")
    if out.device != A.device:
        raise ValueError(f"out must be on the same device as A ({A.device}), got {out.device}")
    if out.dtype != A.dtype:
        raise ValueError(f"out dtype must match A dtype ({A.dtype}), got {out.dtype}")

    batch_size = A.size(0) if A.ndim == 3 else 1
    input_batch_stride = A.stride(0) if A.ndim == 3 else 0
    output_batch_stride = out.stride(0) if out.ndim == 3 else 0
    input_precision = "ieee" if (_IS_HIP or A.dtype == torch.float32) else "tf32"

    def grid(meta):
        return (
            batch_size
            * triton.cdiv(M, meta["BLOCK_SIZE_M"])
            * triton.cdiv(M, meta["BLOCK_SIZE_N"]),
        )
    ns_line_2_kernel[grid](
        A_ptr=A,
        C_ptr=out,
        M=M,
        a_stride_b=input_batch_stride,
        a_stride_r=A.stride(-2),
        a_stride_c=A.stride(-1),
        c_stride_b=output_batch_stride,
        c_stride_r=out.stride(-2),
        c_stride_c=out.stride(-1),
        alpha=alpha,
        beta=beta,
        INPUT_PRECISION=input_precision,
    )

    return out


def _get_gemm_configs():
    # Rectangular tiles are legitimate here (X is M x N); the space is kept
    # small so first-call autotune stays cheap.
    return [
        triton.Config(
            {
                "BLOCK_SIZE_M": bm,
                "BLOCK_SIZE_N": bn,
                "BLOCK_SIZE_K": bk,
                "GROUP_SIZE_M": 8,
            },
            num_stages=st,
            num_warps=wp,
        )
        for bm in (64, 128)
        for bn in (64, 128)
        for bk in (32, 64)
        for st, wp in ((3, 4), (4, 4))
    ]


@triton.jit
def _pid_to_block_ns3(
    pid,
    M,
    N,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    """Same helper as in your earlier kernels, extended with N."""
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)

    batch = pid // (num_pid_m * num_pid_n)
    pid = pid % (num_pid_m * num_pid_n)

    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n
    pid_m, pid_n = tl.swizzle2d(pid_m, pid_n, num_pid_m, num_pid_n, GROUP_SIZE_M)

    return batch, pid_m * BLOCK_SIZE_M, pid_n * BLOCK_SIZE_N


@triton.autotune(
    configs=_get_gemm_configs(),
    key=["M", "N", "b_stride_r", "b_stride_c", "x_stride_r", "x_stride_c"],
)
@triton.jit
def ns_line_3_kernel(
    B_ptr,  # [B, M, M]   symmetric
    X_ptr,  # [B, M, N]
    C_ptr,  # [B, M, N]
    M,
    N,  # rows(X)=M, cols(X)=N
    b_stride_b,
    b_stride_r,
    b_stride_c,
    x_stride_b,
    x_stride_r,
    x_stride_c,
    c_stride_b,
    c_stride_r,
    c_stride_c,
    alpha,  # scalar a  (scale of X)
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    INPUT_PRECISION: tl.constexpr = "tf32",
):
    pid = tl.program_id(axis=0)
    batch, m_start, n_start = _pid_to_block_ns3(
        pid, M, N, BLOCK_SIZE_M, BLOCK_SIZE_N, GROUP_SIZE_M
    )

    # Offset base pointers to this batch
    B_ptr += _batch_offset(batch, b_stride_b)
    X_ptr += _batch_offset(batch, x_stride_b)
    C_ptr += _batch_offset(batch, c_stride_b)

    # Create index ranges for the tile
    offs_m = m_start + tl.arange(0, BLOCK_SIZE_M)
    offs_n = n_start + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    # Pointers to B and X tiles
    b_ptrs = B_ptr + offs_m[:, None] * b_stride_r + offs_k[None, :] * b_stride_c
    x_ptrs = X_ptr + offs_k[:, None] * x_stride_r + offs_n[None, :] * x_stride_c

    # Accumulator, initialized with bias * alpha
    x_bias_ptrs = X_ptr + offs_m[:, None] * x_stride_r + offs_n[None, :] * x_stride_c
    acc = (
        tl.load(
            x_bias_ptrs, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N), other=0.0
        )
        * alpha
    ).to(tl.float32)

    # GEMM main loop
    for k in tl.range(0, tl.cdiv(M, BLOCK_SIZE_K)):
        k_mask = offs_k < M - k * BLOCK_SIZE_K
        b = tl.load(
            b_ptrs,
            mask=(offs_m[:, None] < M) & k_mask[None, :],
            other=0.0,
        )
        x = tl.load(
            x_ptrs,
            mask=k_mask[:, None] & (offs_n[None, :] < N),
            other=0.0,
        )
        acc = tl.dot(b, x, acc, input_precision=INPUT_PRECISION)
        b_ptrs += BLOCK_SIZE_K * b_stride_c
        x_ptrs += BLOCK_SIZE_K * x_stride_r

    out_dtype = C_ptr.dtype.element_ty
    acc = acc.to(out_dtype)

    # Store result
    c_ptrs = C_ptr + offs_m[:, None] * c_stride_r + offs_n[None, :] * c_stride_c
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=mask)


def ns_line_3(B: Tensor, X: Tensor, a: float, *, out: Tensor = None) -> Tensor:
    """
    Fused implementation of C = a * X + B @ X
    B must be square & symmetric, X has same leading dim, arbitrary trailing cols.
    """
    if not TRITON_AVAILABLE:
        raise RuntimeError("ns_line_3 requires the 'triton' package")
    if B.ndim not in (2, 3) or X.ndim not in (2, 3):
        raise ValueError(
            f"B and X must be 2D or 3D, but got B {B.ndim}D, X {X.ndim}D"
        )
    _reject_output_alias(out, B, X)
    if B.shape[-2] != B.shape[-1]:
        raise ValueError("B must be square")

    if B.shape[-2] != X.shape[-2]:
        raise ValueError("B and X must have the same number of rows")

    # Batch semantics: a 2D X pairs with a 3D B only when B's batch is 1
    # (broadcast); a 2D B broadcasts across any X batch; matching 3D
    # batches are elementwise.
    if X.ndim == 3:
        if B.ndim == 3 and X.shape[0] != B.shape[0]:
            raise ValueError(
                f"B and X must have the same batch size, got B {B.shape[0]}, X {X.shape[0]}"
            )
    elif B.ndim == 3 and B.shape[0] != 1:
        raise ValueError(
            "a 3D B with a 2D X is only supported when B's batch size is 1"
        )

    if B.device != X.device:
        raise ValueError(
            f"B and X must be on the same device, got {B.device} and {X.device}"
        )
    if B.dtype != X.dtype:
        raise ValueError(
            f"B and X must have the same dtype, got {B.dtype} and {X.dtype}"
        )

    if out is None:
        out = torch.empty_like(X)
    if out.shape != X.shape:
        raise ValueError(f"out must have shape {X.shape}, got {out.shape}")
    if out.device != X.device:
        raise ValueError(f"out must be on the same device as X ({X.device}), got {out.device}")
    if out.dtype != X.dtype:
        raise ValueError(f"out dtype must match X dtype ({X.dtype}), got {out.dtype}")

    # Broadcast & batch handling (supports 2- or 3-D inputs)
    M, N = X.shape[-2:]
    batch = X.shape[0] if X.ndim == 3 else 1
    input_precision = "ieee" if (_IS_HIP or X.dtype == torch.float32) else "tf32"

    def grid(meta):
        return (
            batch
            * triton.cdiv(M, meta["BLOCK_SIZE_M"])
            * triton.cdiv(N, meta["BLOCK_SIZE_N"]),
        )

    ns_line_3_kernel[grid](
        B_ptr=B,
        X_ptr=X,
        C_ptr=out,
        M=M,
        N=N,
        b_stride_b=B.stride(0) if B.ndim == 3 else 0,
        b_stride_r=B.stride(-2),
        b_stride_c=B.stride(-1),
        x_stride_b=X.stride(0) if X.ndim == 3 else 0,
        x_stride_r=X.stride(-2),
        x_stride_c=X.stride(-1),
        c_stride_b=out.stride(0) if out.ndim == 3 else 0,
        c_stride_r=out.stride(-2),
        c_stride_c=out.stride(-1),
        alpha=a,
        INPUT_PRECISION=input_precision,
    )
    return out


@torch.no_grad()
def zeropower_via_newtonschulz5_triton(G: Tensor, steps: int = 5) -> Tensor:
    """Keller-Jordan Newton-Schulz using Triton for all three matrix products.

    This kernel path is intentionally limited to the M=512 buckets for which
    correctness, stability, and performance have been validated. Callers must
    retain the Torch implementation for every other shape.
    """
    if not G.is_cuda:
        raise ValueError("the M=512 Triton path requires a CUDA tensor")
    if G.ndim not in (2, 3):
        raise ValueError("the M=512 Triton path requires a 2D or 3D tensor")
    if min(G.shape[-2:]) != 512:
        raise ValueError(
            f"the Triton path is validated only for min(rows, cols)=512, got {G.shape[-2:]}"
        )
    if steps != 5:
        raise ValueError(f"the M=512 Triton path requires steps=5, got {steps}")

    a, b, c = (3.4445, -4.7750, 2.0315)
    transposed = G.size(-2) > G.size(-1)
    X = G.to(dtype=torch.bfloat16)
    if transposed:
        X = X.mT
    X = (X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)).contiguous()

    A = torch.empty((*X.shape[:-1], X.size(-2)), device=X.device, dtype=X.dtype)
    B = torch.empty_like(A)
    C = torch.empty_like(X)
    for _ in range(steps):
        ns_line_1(X, out=A)
        ns_line_2(A, alpha=c, beta=b, out=B)
        ns_line_3(B, X, a, out=C)
        X, C = C, X

    return X.mT.contiguous() if transposed else X
