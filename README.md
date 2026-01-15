# Quantum-Inspired Machine Learning (QiML) Documentation

## Projected Quantum Kernels for Financial Fraud Detection

This documentation suite provides comprehensive technical guidance for the QiML pipeline, which leverages tensor network simulation of quantum circuits to perform machine learning classification tasks, with a primary focus on financial fraud detection.

---

## Documentation Index

### Theoretical Background

| Document | Description | Key Topics |
|----------|-------------|------------|
| [01_THEORETICAL_FOUNDATIONS.md](01_THEORETICAL_FOUNDATIONS.md) | Mathematical foundations of quantum machine learning | Quantum states, feature maps, kernel methods, quantum kernels |
| [02_PROJECTED_QUANTUM_KERNELS.md](02_PROJECTED_QUANTUM_KERNELS.md) | Detailed mathematics of Projected Quantum Kernels | Pauli measurements, RDM geometry, kernel derivation, complexity analysis |
| [03_TENSOR_NETWORK_SIMULATION.md](03_TENSOR_NETWORK_SIMULATION.md) | MPS-based quantum circuit simulation | Matrix Product States, ITensor, gate application, bond dimension |

### Implementation & Operations

| Document | Description | Key Topics |
|----------|-------------|------------|
| [04_QMLOPS_PIPELINE.md](04_QMLOPS_PIPELINE.md) | End-to-end pipeline workflow | Data flow, preprocessing, feature extraction, kernel construction, training, inference |
| [05_IMPLEMENTATION_REFERENCE.md](05_IMPLEMENTATION_REFERENCE.md) | Code reference and configuration | Class/function reference, parameters, gate encoding, file formats |
| [06_FRAUD_DETECTION_APPLICATION.md](06_FRAUD_DETECTION_APPLICATION.md) | Domain-specific fraud detection guide | Elliptic dataset, evaluation metrics, production deployment |

---

## Quick Start Guide

### Prerequisites

- Python 3.10+
- ITensor C++ library (compiled with `qubit.h` extension)
- MPI (OpenMPI recommended)
- Required Python packages: `mpi4py`, `pytket`, `pandas`, `scikit-learn`, `numpy`, `sympy`

### Basic Execution

```bash
# Clone and setup
cd "QML dataproc/ITensor_C"

# Run with MPI (4 processes)
mpirun -n 4 python main.py 12 10 1 0.5 100 100 456 bitstrings_12_preproc.csv
```

### Docker Execution

```bash
cd "QML dataproc/Installation-Script"
docker build -t qiml .
docker run --env NUM_NODES=4 --env NUM_FEATURES=12 qiml
```

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                         QiML ARCHITECTURE                           │
└─────────────────────────────────────────────────────────────────────┘

                    ┌─────────────────────────────┐
                    │      Classical Data         │
                    │    x ∈ ℝᵈ (d features)     │
                    └─────────────┬───────────────┘
                                  │
                    ┌─────────────▼───────────────┐
                    │    Quantum Feature Map      │
                    │  U(x) = H⊗ⁿ ∏[Rz·XXPhase]  │
                    │      (pytket circuits)      │
                    └─────────────┬───────────────┘
                                  │
                    ┌─────────────▼───────────────┐
                    │   MPS Tensor Network Sim    │
                    │    |ψ(x)⟩ = U(x)|0⟩ⁿ       │
                    │    (ITensor C++ backend)    │
                    └─────────────┬───────────────┘
                                  │
                    ┌─────────────▼───────────────┐
                    │   Projected Quantum Features│
                    │   Φ(x) = [⟨X⟩,⟨Y⟩,⟨Z⟩]ᵢ   │
                    │      (3n dimensions)        │
                    └─────────────┬───────────────┘
                                  │
                ┌─────────────────┴─────────────────┐
                │                                   │
    ┌───────────▼───────────┐         ┌───────────▼───────────┐
    │    Kernel Mode        │         │    Feature Mode       │
    │  k(x,x') = exp(-α·D)  │         │  Φ(x) → LightGBM     │
    │     → SVM             │         │                       │
    └───────────────────────┘         └───────────────────────┘
```

---

## Key Concepts

### Projected Quantum Kernel Formula

$$k_{\text{PQK}}(x, x') = \exp\left(-\alpha \sum_{i=1}^{n} 2\left[(\langle X_i \rangle^x - \langle X_i \rangle^{x'})^2 + (\langle Y_i \rangle^x - \langle Y_i \rangle^{x'})^2 + (\langle Z_i \rangle^x - \langle Z_i \rangle^{x'})^2\right]\right)$$

### Hamiltonian Ansatz Circuit

$$U(x) = H^{\otimes n} \prod_{\ell=1}^{r} \left[ \prod_{i=1}^{n} R_Z\left(\frac{\gamma x_i}{\pi}\right) \prod_{(i,j) \in E} R_{XX}\left(\gamma^2(1-x_i)(1-x_j)\right) \right]$$

### Parameter Glossary

| Symbol | Parameter | Typical Range | Description |
|--------|-----------|---------------|-------------|
| $n$ | `num_features` | 5-50 | Number of qubits/features |
| $r$ | `reps` | 2-10 | Circuit layer repetitions |
| $\gamma$ | `gamma` | 0.1-1.0 | Rotation scaling |
| $\alpha$ | `alpha` | 0.1-2.0 | Kernel bandwidth |
| $C$ | SVM regularization | 0.01-2.0 | Misclassification penalty |

---

## References

### Core Papers

1. **Huang, H.-Y. et al.** (2021). "Power of data in quantum machine learning." *Nature Communications*, 12, 2631. [Link](https://www.nature.com/articles/s41467-021-22539-9)

2. **Havlíček, V. et al.** (2019). "Supervised learning with quantum-enhanced feature spaces." *Nature*, 567, 209-212. [Link](https://www.nature.com/articles/s41586-019-0980-2)

3. **Fishman, M., White, S. R., & Stoudenmire, E. M.** (2022). "The ITensor Software Library for Tensor Network Calculations." *SciPost Physics Codebases*, 4. [Link](https://scipost.org/SciPostPhysCodeb.4)

### Fraud Detection Applications

4. **Heredge, J. et al.** (2023). "Quantum Multiple Kernel Learning in Financial Classification Tasks." [arXiv:2312.00260](https://arxiv.org/abs/2312.00260)

5. **Vasquez, A.C. et al.** (2023). "Financial Fraud Detection: A Comparative Study of Quantum Machine Learning Models." [arXiv:2308.05237](https://arxiv.org/abs/2308.05237)

---

## Document Revision History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | January 2026 | Initial comprehensive documentation |

---

## Navigation

- **Getting Started:** [04_QMLOPS_PIPELINE.md](04_QMLOPS_PIPELINE.md)
- **Theory Deep-Dive:** [01_THEORETICAL_FOUNDATIONS.md](01_THEORETICAL_FOUNDATIONS.md)
- **Implementation Details:** [05_IMPLEMENTATION_REFERENCE.md](05_IMPLEMENTATION_REFERENCE.md)
- **Fraud Detection:** [06_FRAUD_DETECTION_APPLICATION.md](06_FRAUD_DETECTION_APPLICATION.md)

---

*For additional guidance, see [CLAUDE.md](../CLAUDE.md) in the repository root.*
