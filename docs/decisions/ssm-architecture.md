# SSM architecture: portable S4D first

- **Status:** Accepted for issue [#12](https://github.com/Ikkjo/neural-fx/issues/12)
- **Decision:** Implement a causal diagonal S4 (S4D) model with core PyTorch operations. Keep Mamba optional and defer it until the S4D execution and export contracts are complete.

## Context and evidence

Neural-fx needs one model that trains on sequences, processes audio with cached state sample by sample, runs in the core CPU installation, and has a credible deployment path.

| Candidate | Faithful implementation | Training and streaming | Portability and export implications |
|---|---|---|---|
| **S4** | The original S4 parameterizes the state matrix as diagonal-plus-low-rank (DPLR). Its official PyTorch repository says efficient Cauchy and Vandermonde kernels need either a compiled CUDA extension or PyKeOps; naive versions have suboptimal memory use. This is more machinery than the required portable baseline. | S4 supports convolutional sequence processing and recurrence, but the official repository recommends S4D for state forwarding because its S4 path is not optimized. | A faithful, efficient S4 brings non-core kernel dependencies or an inefficient fallback, increasing CPU and exporter risk. [Official S4 repository](https://github.com/state-spaces/s4#structured-kernels), [official model notes](https://github.com/state-spaces/s4/blob/main/models/s4/README.md#models)
| **S4D** | S4D replaces DPLR with a diagonal state matrix. The paper reports that its kernel computation takes two lines and performs comparably to S4 in almost all tested settings; the official repository includes a small standalone implementation using PyTorch tensor operations. | The same linear SSM has a convolution view for parallel full-sequence training and a recurrent view for causal stepping. The official implementation documents that forwarding an initial state is equivalent to repeated `step` calls and recommends S4D for this use. | No custom S4 kernel is required. For deployment, export the real-valued recurrent step and explicit cache rather than the complex/FFT training path; this is an engineering conclusion from the primitive diagonal recurrence, not a paper guarantee. [S4D paper](https://arxiv.org/abs/2206.11893), [minimal official S4D](https://github.com/state-spaces/s4/blob/main/models/s4/s4d.py), [state-forwarding notes](https://github.com/state-spaces/s4/blob/main/models/s4/README.md#state-forwarding)
| **Mamba** | Mamba makes SSM parameters input-dependent. The paper notes that selectivity prevents use of the convolution formulation and introduces a hardware-aware parallel scan instead. Reproducing the optimized architecture therefore entails more than a fixed diagonal recurrence. | It supports recurrent inference and parallel sequence processing, but not S4D's shared convolution kernel. | The official package lists Linux, NVIDIA GPU, and CUDA as requirements and ships custom selective-scan/causal-convolution code. A slow primitive fallback could be written, but it would not be the official optimized implementation and would enlarge validation and export scope. [Mamba paper](https://arxiv.org/abs/2312.00752), [official Mamba repository](https://github.com/state-spaces/mamba#installation)
| **S5** | S5 changes S4's bank of independent SISO systems into one MIMO system and uses parallel scans. This is attractive, but adds scan and MIMO parameterization work without solving a requirement that S4D misses. | The paper and repository support parallel offline processing; recurrent evaluation follows from the SSM formulation. | The official implementation is JAX-based, including separate JAX CPU/GPU requirements, so adopting it would require a PyTorch port and new equivalence/export validation. [S5 paper](https://arxiv.org/abs/2208.04933), [official S5 repository](https://github.com/lindermanlab/S5)

## Decision and execution contract

Implement **S4D** with a stable diagonal parameterization and zero-order-hold discretization, using only core PyTorch for the supported CPU path.

- Full-sequence training is causal convolution with no future leakage.
- Streaming inference exposes explicit reset/detach state and sample/block processing; repeated streaming must match full causal inference within tolerance.
- The portable export boundary is a real-valued recurrent cell with explicit state input/output. FFT-based kernel construction and complex tensors remain training/runtime implementation details and are not part of the portable exported graph.
- Any exporter that cannot represent that recurrent contract must fail with a specific, actionable error rather than silently changing the model.

This choice preserves the defining SSM duality needed by neural audio while minimizing faithful implementation and dependency surface. It also leaves room to optimize convolution kernel construction without changing the streaming interface.

## Consequences

S4D becomes the required CPU model and must not import optional compiled packages. Mamba remains a later, optional accelerator-backed model behind an explicit dependency boundary. Reconsider Mamba after S4D has CPU training, full/streaming equivalence, and export smoke tests; evaluate it on measured audio quality and latency rather than replacing the portable baseline by default.
