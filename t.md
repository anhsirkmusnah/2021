# Quantum-Inspired Machine Learning (QiML) Platform
## Projected Quantum Kernels for Financial Fraud Detection
### Complete Technical Documentation

---

**Document Version:** 1.0
**Last Updated:** January 2026
**Authors:** Enterprise Quantum Engineering Team
**Classification:** Internal Technical Documentation

---

# Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Project Overview](#2-project-overview)
3. [Theoretical Foundations](#3-theoretical-foundations)
4. [Projected Quantum Kernels](#4-projected-quantum-kernels)
5. [Tensor Network Simulation](#5-tensor-network-simulation)
6. [QMLOps Pipeline](#6-qmlops-pipeline)
7. [Implementation Reference](#7-implementation-reference)
8. [Fraud Detection Application](#8-fraud-detection-application)
9. [Deployment Guide](#9-deployment-guide)
10. [References](#10-references)

---

# 1. Executive Summary

## 1.1 Purpose

This document provides comprehensive technical documentation for the Quantum-Inspired Machine Learning (QiML) platform, which implements **Projected Quantum Kernels (PQK)** for binary classification tasks, with primary application to **financial fraud detection**.

## 1.2 Key Capabilities

| Capability | Description |
|------------|-------------|
| **Quantum Feature Extraction** | Encodes classical data into quantum circuits and extracts Pauli expectation values |
| **Projected Quantum Kernels** | Computes kernel matrices using quantum-derived features for SVM classification |
| **Tensor Network Simulation** | Uses Matrix Product States (MPS) via ITensor for efficient classical simulation |
| **MPI Parallelization** | Distributed computation across multiple nodes for scalable kernel construction |
| **Production Pipeline** | End-to-end workflow from data ingestion to model deployment |

## 1.3 Technology Stack

| Layer | Technology |
|-------|------------|
| Quantum Simulation | ITensor C++ library with custom qubit site type |
| Circuit Construction | pytket (Quantinuum) |
| Parallelization | MPI (mpi4py, OpenMPI) |
| ML Framework | scikit-learn (SVM), LightGBM |
| Language Bridge | pybind11 (C++/Python) |
| Containerization | Docker |

## 1.4 Primary Use Case

**Financial Fraud Detection** using the Elliptic Bitcoin dataset, demonstrating:
- 85-92% accuracy on fraud classification
- 5-10% improvement over classical RBF kernels
- Sub-200ms inference latency per transaction

---

# 2. Project Overview

## 2.1 Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           QiML PLATFORM ARCHITECTURE                        │
└─────────────────────────────────────────────────────────────────────────────┘

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

## 2.2 Directory Structure

```
QML dataproc/
├── ITensor_C/                    # ITensor C++ backend
│   ├── helloitensor.cc          # C++ MPS simulation core
│   ├── qubit.h                  # Custom ITensor qubit site type
│   ├── main.py                  # Main execution script
│   ├── main_dlp.py              # Discrete log problem variant
│   ├── projected_kernel_ansatz.py  # Ansatz + kernel builder
│   └── datasets/                # Data directory
│
├── QuantumLibs/                  # Python quantum simulation
│   ├── main.py                  # Execution entry point
│   ├── projected_kernel_ansatz.py
│   └── projected_quantum_features.py
│
├── dataproc files/               # Production pipeline
│   ├── main.py                  # Orchestration script
│   ├── generate_pqf.py          # Feature generation
│   ├── train.py                 # Model training
│   └── test.py                  # Model evaluation
│
├── Installation-Script/          # Deployment utilities
│   ├── Dockerfile               # Container definition
│   ├── elliptic_preproc.py      # Data preprocessing
│   └── readme.md                # Installation guide
│
└── docs/                         # Documentation
```

## 2.3 Data Flow Summary

```
Raw Data (CSV)
     │
     ▼
┌────────────────────────────────────────────────────────────────────────────┐
│ PREPROCESSING: QuantileTransform → StandardScaler → MinMaxScaler[-π,π]    │
└────────────────────────────────────────────────────────────────────────────┘
     │
     ▼
┌────────────────────────────────────────────────────────────────────────────┐
│ QUANTUM ENCODING: Build circuit U(x) with Hadamard + Rz + XXPhase gates   │
└────────────────────────────────────────────────────────────────────────────┘
     │
     ▼
┌────────────────────────────────────────────────────────────────────────────┐
│ MPS SIMULATION: Simulate |ψ(x)⟩ = U(x)|0⟩ⁿ using ITensor                  │
└────────────────────────────────────────────────────────────────────────────┘
     │
     ▼
┌────────────────────────────────────────────────────────────────────────────┐
│ FEATURE EXTRACTION: Compute ⟨X⟩, ⟨Y⟩, ⟨Z⟩ for each qubit → Φ(x) ∈ ℝ³ⁿ   │
└────────────────────────────────────────────────────────────────────────────┘
     │
     ▼
┌────────────────────────────────────────────────────────────────────────────┐
│ KERNEL COMPUTATION: k(x,x') = exp(-α·||Φ(x)-Φ(x')||²)                     │
└────────────────────────────────────────────────────────────────────────────┘
     │
     ▼
┌────────────────────────────────────────────────────────────────────────────┐
│ CLASSIFICATION: SVM with precomputed kernel → Fraud/Legitimate prediction │
└────────────────────────────────────────────────────────────────────────────┘
```

---

# 3. Theoretical Foundations

## 3.1 Quantum State Representation

### 3.1.1 Single Qubit States

A qubit exists in superposition of computational basis states:

**Mathematical Definition:**
```
|ψ⟩ = α|0⟩ + β|1⟩
```

where α, β ∈ ℂ and |α|² + |β|² = 1.

**Bloch Sphere Representation:**
```
|ψ⟩ = cos(θ/2)|0⟩ + e^(iφ)sin(θ/2)|1⟩
```

where θ ∈ [0, π] and φ ∈ [0, 2π).

### 3.1.2 Multi-Qubit Systems

For n qubits, the state space is the tensor product:

```
ℋ = ℋ₁ ⊗ ℋ₂ ⊗ ... ⊗ ℋₙ = (ℂ²)^⊗n
```

A general n-qubit state requires 2ⁿ complex amplitudes:

```
|ψ⟩ = Σ c_{i₁i₂...iₙ} |i₁i₂...iₙ⟩
```

| Qubits | Amplitudes | Memory (complex128) |
|--------|------------|---------------------|
| 20 | ~1M | 16 MB |
| 30 | ~1B | 16 GB |
| 40 | ~1T | 16 TB |
| 50 | ~1P | 16 PB |

## 3.2 Quantum Feature Maps

### 3.2.1 Definition

A quantum feature map encodes classical data into quantum states:

```
φ: 𝒳 → ℋ
φ(x) = U(x)|0⟩^⊗n
```

where x ∈ 𝒳 ⊆ ℝᵈ is classical input data and U(x) is a parameterized unitary circuit.

### 3.2.2 The Hamiltonian Ansatz (Primary Implementation)

**Circuit Structure:**

```
Layer Structure (repeated r times):

1. INITIALIZATION (once): H^⊗n - Hadamard on all qubits

   |0⟩ ─[H]─ → |+⟩ = (|0⟩ + |1⟩)/√2

2. SINGLE-QUBIT ENCODING: Rz(γ·xᵢ/π) on qubit i

   Applies phase rotation based on feature value

3. TWO-QUBIT ENTANGLEMENT: R_XX(γ²(1-xᵢ)(1-xⱼ)) on pairs (i,j)

   Creates entanglement proportional to feature interaction
```

**Mathematical Formulation:**

```
U(x) = H^⊗n ∏_{ℓ=1}^{r} [ ∏_{i=1}^{n} Rz(γxᵢ/π) ∏_{(i,j)∈E} R_XX(γ²(1-xᵢ)(1-xⱼ)) ]
```

**Gate Definitions:**

```
         ┌                      ┐
Rz(θ) =  │ e^(-iθ/2)     0     │
         │     0      e^(iθ/2)  │
         └                      ┘

            ┌                                              ┐
R_XX(θ) =   │ cos(θ/2)    0         0      -i·sin(θ/2)   │
            │    0     cos(θ/2) -i·sin(θ/2)     0        │
            │    0    -i·sin(θ/2) cos(θ/2)      0        │
            │-i·sin(θ/2)   0         0       cos(θ/2)    │
            └                                              ┘
```

### 3.2.3 The Magic Ansatz (Alternative)

**Circuit Structure:**
```
Layer Structure (repeated r times):
1. Hadamard H on all qubits
2. T gate (π/8 rotation) on all qubits
3. CZ gates on connected pairs
4. Rz(xᵢ) encoding on qubit i
```

Creates "magic states" with quantum contextuality properties.

### 3.2.4 Entanglement Topology

**Linear Nearest-Neighbor Connectivity:**

```
E = {(i, i+1) : i ∈ [0, n-2]}

Example for n=5:
Qubits:  0 --- 1 --- 2 --- 3 --- 4
Pairs:   [(0,1), (2,3), (1,2), (3,4)]
```

**Rationale:**
- Matches typical quantum hardware constraints
- Sufficient for capturing local correlations
- Enables efficient MPS simulation

## 3.3 Kernel Methods in Machine Learning

### 3.3.1 The Kernel Trick

A function k: 𝒳 × 𝒳 → ℝ is a valid kernel if it corresponds to an inner product:

```
k(x, x') = ⟨φ(x), φ(x')⟩_ℱ
```

for some feature map φ: 𝒳 → ℱ.

### 3.3.2 Mercer's Theorem

A symmetric function k(x, x') is a valid kernel if and only if it is positive semi-definite:

```
Σᵢⱼ cᵢcⱼk(xᵢ, xⱼ) ≥ 0
```

for all finite sets {xᵢ} and coefficients {cᵢ} ⊂ ℝ.

### 3.3.3 Support Vector Machines

**Dual Form (Kernel Form):**

```
max_α Σᵢ αᵢ - (1/2)Σᵢⱼ αᵢαⱼyᵢyⱼk(xᵢ, xⱼ)

subject to: 0 ≤ αᵢ ≤ C, Σᵢ αᵢyᵢ = 0
```

**Decision Function:**

```
f(x) = sign(Σᵢ αᵢyᵢk(xᵢ, x) + b)
```

### 3.3.4 The RBF (Gaussian) Kernel

```
k_RBF(x, x') = exp(-||x - x'||² / 2σ²)
```

Corresponds to infinite-dimensional feature space.

## 3.4 Quantum Kernels

### 3.4.1 Fidelity Quantum Kernel (FQK)

Standard quantum kernel computes state overlap:

```
k_FQK(x, x') = |⟨φ(x')|φ(x)⟩|² = |⟨0ⁿ|U†(x')U(x)|0ⁿ⟩|²
```

**Limitations:**
- Exponentially small for distant points
- Requires full state access
- Numerical instability

### 3.4.2 Projected Quantum Kernel (PQK)

Computes kernel from reduced density matrices:

```
k_PQK(x, x') = exp(-α Σᵢ ||ρᵢ(x) - ρᵢ(x')||²_F)
```

where ρᵢ(x) is the single-qubit reduced density matrix.

**Advantages over FQK:**

| Property | FQK | PQK |
|----------|-----|-----|
| Dimensionality | 2ⁿ (full Hilbert space) | 3n (local observables) |
| Numerical stability | Exponentially small values | Well-conditioned |
| Classical simulability | Requires full state | Only local expectation values |

---

# 4. Projected Quantum Kernels

## 4.1 Mathematical Derivation

### 4.1.1 Pauli Matrices

```
     ┌     ┐         ┌      ┐         ┌      ┐
X =  │ 0 1 │    Y =  │ 0 -i │    Z =  │ 1  0 │
     │ 1 0 │         │ i  0 │         │ 0 -1 │
     └     ┘         └      ┘         └      ┘
```

**Properties:**
- Hermitian: P† = P
- Unitary: P² = I
- Traceless: Tr(P) = 0
- Eigenvalues: ±1

### 4.1.2 Bloch Vector Representation

Any single-qubit state can be written as:

```
ρ = (1/2)(I + r⃗ · σ⃗) = (1/2)(I + rₓX + rᵧY + rᵤZ)
```

where the Bloch vector r⃗ = (rₓ, rᵧ, rᵤ) has:

```
rₓ = ⟨X⟩,  rᵧ = ⟨Y⟩,  rᵤ = ⟨Z⟩
```

### 4.1.3 Frobenius Distance

For single-qubit RDMs with Bloch vectors r⃗ and s⃗:

```
||ρ - σ||²_F = (1/2)||r⃗ - s⃗||² = (1/2)[(rₓ-sₓ)² + (rᵧ-sᵧ)² + (rᵤ-sᵤ)²]
```

### 4.1.4 Complete PQK Formula

**Definition:**

```
k_PQK(x, x') = exp(-α · D(x, x'))
```

where the quantum distance is:

```
D(x, x') = Σᵢ₌₁ⁿ 2·[(⟨Xᵢ⟩ˣ - ⟨Xᵢ⟩ˣ')² + (⟨Yᵢ⟩ˣ - ⟨Yᵢ⟩ˣ')² + (⟨Zᵢ⟩ˣ - ⟨Zᵢ⟩ˣ')²]
```

This equals the squared Euclidean distance in the projected feature space:

```
D(x, x') = ||Φ(x) - Φ(x')||²₂
```

where:

```
Φ(x) = (⟨X₁⟩, ⟨Y₁⟩, ⟨Z₁⟩, ..., ⟨Xₙ⟩, ⟨Yₙ⟩, ⟨Zₙ⟩) ∈ ℝ³ⁿ
```

## 4.2 Kernel Properties

**Theorem:** k_PQK is a valid Mercer kernel.

**Properties:**
- k_PQK(x, x) = 1 (self-similarity)
- k_PQK(x, x') = k_PQK(x', x) (symmetry)
- 0 < k_PQK(x, x') ≤ 1 (bounded)

## 4.3 Connection to Classical Kernels

The PQK is an RBF kernel in a quantum-derived feature space:

```
k_PQK(x, x') = k_RBF(Φ(x), Φ(x'); γ = α)
```

**Key difference:** The quantum circuit acts as a nonlinear feature extractor that:
1. Encodes data into quantum states
2. Creates entanglement-mediated correlations
3. Projects back to classical observables

## 4.4 Computational Complexity

### 4.4.1 Per Data Point

| Operation | Complexity | Notes |
|-----------|------------|-------|
| Circuit simulation (MPS) | O(n · r · χ³) | χ = bond dimension |
| Expectation values | O(n · χ²) | Per qubit |

### 4.4.2 Kernel Matrix

| Operation | Complexity |
|-----------|------------|
| All MPS simulations | O((N_train + N_test) · n · r · χ³) |
| Kernel entries | O(N_train · N_test · n) |

### 4.4.3 MPI Parallelization

```
T(P) ≈ T(1)/P  for large N
```

where P is the number of MPI processes.

---

# 5. Tensor Network Simulation

## 5.1 Matrix Product States (MPS)

### 5.1.1 Definition

An MPS represents an n-qubit state as a chain of tensors:

```
|ψ⟩ = Σ_{i₁,...,iₙ} A^[1]_{i₁} A^[2]_{i₂} ... A^[n]_{iₙ} |i₁...iₙ⟩
```

where:
- A^[k]_{iₖ} is a matrix of dimensions χₖ₋₁ × χₖ
- iₖ ∈ {0, 1} is the physical index (qubit state)
- χₖ is the **bond dimension** at bond k

**Pictorial Representation:**

```
    i₁    i₂    i₃    i₄    i₅
    |     |     |     |     |
   [A¹]--[A²]--[A³]--[A⁴]--[A⁵]
       χ₁   χ₂   χ₃   χ₄
```

### 5.1.2 Bond Dimension and Entanglement

The bond dimension χ controls entanglement capacity:

```
S(ρ_left) ≤ log₂(χₖ)
```

The entanglement entropy across a cut is bounded by log of bond dimension.

### 5.1.3 Initial State

The all-zeros state |0⟩^⊗n has trivial MPS with χ = 1 (product state).

## 5.2 Gate Application on MPS

### 5.2.1 Single-Qubit Gates

```
Ã^[k]_j = Σᵢ U_ji A^[k]_i
```

Complexity: O(χ²) — does not increase bond dimension.

### 5.2.2 Two-Qubit Gates (Adjacent Sites)

**Algorithm:**
1. Contract neighboring tensors: Θ = A^[k] · A^[k+1]
2. Apply gate: Θ̃ = U · Θ
3. Decompose via SVD: Θ̃ = U Σ V†
4. Truncate to maximum bond dimension

Complexity: O(χ³) for SVD, may increase χ.

### 5.2.3 Non-Adjacent Gates

For gates between non-adjacent qubits:
1. SWAP qubits until adjacent
2. Apply the gate
3. SWAP back

The pytket compiler handles this via `DecomposeBRIDGE` pass.

## 5.3 ITensor Implementation

### 5.3.1 Overview

ITensor is a C++ library for tensor network computations developed at the Flatiron Institute.

**Reference:** Fishman, M., White, S. R., & Stoudenmire, E. M. (2022). "The ITensor Software Library for Tensor Network Calculations." SciPost Physics Codebases, 4.

### 5.3.2 Gate Application Pattern (C++)

**Single-Qubit Gate:**
```cpp
psi.position(i1+1);
auto G = op(site_inds, "Rz", i1+1, {"alpha=", a});
auto new_MPS = G * psi(i1+1);
new_MPS.noPrime();
psi.set(i1+1, new_MPS);
```

**Two-Qubit Gate:**
```cpp
psi.position(i1+1);
auto wf = psi(i1+1) * psi(i2+1);
wf *= G;
wf.noPrime();
auto [U, S, V] = svd(wf, inds(psi(i1+1)), {"Cutoff=", 1E-10});
psi.set(i1+1, U);
psi.set(i2+1, S*V);
```

### 5.3.3 Expectation Value Computation

```cpp
for (int i = 0; i < no_sites; i++) {
    psi.position(i+1);
    auto scalar_x = eltC(
        dag(prime(psi.A(i+1), "Site")) *
        site_inds.op("X_half", i+1) *
        psi.A(i+1)
    ).real();
    // Similarly for Y, Z
}
```

### 5.3.4 Python-C++ Bridge

```cpp
PYBIND11_MODULE(helloitensor, m) {
    m.def("circuit_xyz_exp",
          &circuit_xyz_exp<int,double>,
          "Extract X,Y,Z expectation values from circuit simulation");
}
```

**Python Usage:**
```python
from helloitensor import circuit_xyz_exp
exp_xyz = circuit_xyz_exp(circuit_gates, n_qubits)
# Returns: [[⟨X₁⟩, ⟨Y₁⟩, ⟨Z₁⟩], [⟨X₂⟩, ⟨Y₂⟩, ⟨Z₂⟩], ...]
```

## 5.4 Complexity and Scalability

### 5.4.1 MPS Operations

| Operation | Complexity |
|-----------|------------|
| Initialize |0⟩^⊗n | O(n) |
| Single-qubit gate | O(χ²) |
| Two-qubit gate (adjacent) | O(χ³) |
| Two-qubit gate (distance d) | O(d · χ³) |
| Expectation value | O(χ²) |

### 5.4.2 When MPS Works Well

MPS simulation is efficient when:
- Circuits have linear (1D) connectivity
- Entanglement remains bounded
- Gates are local (nearest-neighbor)

### 5.4.3 Scalability Guidelines

| Qubits | Recommended χ_max | Memory (per state) |
|--------|-------------------|-------------------|
| 10-20 | 50 | ~1 MB |
| 20-50 | 100 | ~10 MB |
| 50-100 | 200 | ~100 MB |
| 100+ | 500 | ~1 GB |

---

# 6. QMLOps Pipeline

## 6.1 Pipeline Stages Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           COMPLETE PIPELINE FLOW                            │
└─────────────────────────────────────────────────────────────────────────────┘

Stage 1: DATA INGESTION
├── Load CSV data
├── Handle missing values
├── Encode class labels
└── Filter unknown labels

Stage 2: PREPROCESSING
├── QuantileTransform (handle outliers)
├── StandardScaler (zero mean, unit variance)
├── MinMaxScaler (scale to [-π, π])
└── Feature selection (top k features)

Stage 3: CLASS BALANCING
├── Stratified sampling
├── n_illicit samples from fraud class
├── n_licit samples from legitimate class
└── Train/test split (80/20)

Stage 4: QUANTUM FEATURE EXTRACTION
├── Build symbolic circuit (pytket)
├── Substitute data values
├── Compile to linear architecture
├── Simulate with MPS (ITensor)
└── Extract ⟨X⟩, ⟨Y⟩, ⟨Z⟩

Stage 5: KERNEL CONSTRUCTION
├── Compute Φ(x) for all training points
├── Build K_train (N×N symmetric)
├── Build K_test (M×N)
└── MPI parallelization

Stage 6: MODEL TRAINING
├── SVM with precomputed kernel
├── Grid search over C values
└── Select best model

Stage 7: EVALUATION
├── Predict on test set
├── Compute metrics (Accuracy, Precision, Recall, F1, AUC)
└── Compare with RBF baseline

Stage 8: DEPLOYMENT
├── Save model (.pkl)
├── Save scaler (.pkl)
└── Production inference
```

## 6.2 Data Preprocessing Details

### 6.2.1 Preprocessing Pipeline

**Step 1: Quantile Transform**
```
x' = Φ⁻¹(F(x))
```
Maps data to Gaussian distribution, handles outliers.

**Step 2: Standardization**
```
x'' = (x' - μ) / σ
```
Zero mean, unit variance.

**Step 3: MinMax Scaling**
```
x''' = a + (x'' - min(x''))(b - a) / (max(x'') - min(x''))
```
Scale to [a, b] = [-π, π] or [-π/4, π/4].

### 6.2.2 Scaling Range Selection

| Qubits | Recommended Range |
|--------|-------------------|
| n < 10 | [-π, π] |
| 10 ≤ n < 20 | [-π/2, π/2] |
| n ≥ 20 | [-π/4, π/4] |

### 6.2.3 Class Balancing

```python
def draw_sample(df, ndmin, ndmaj, test_frac=0.2, seed=123):
    # Stratified sampling from each class
    data_reduced = pd.concat([
        df[df['Class']==0].sample(ndmin, random_state=seed*20+2),
        df[df['Class']==1].sample(ndmaj, random_state=seed*46+9)
    ])

    # Stratified train/test split
    train_df, test_df = train_test_split(
        data_reduced,
        stratify=data_reduced['Class'],
        test_size=test_frac,
        random_state=seed*26+19
    )
    return train_features, train_labels, test_features, test_labels
```

## 6.3 Circuit Construction Flow

```
Classical Feature Vector x = [x₁, x₂, ..., xₙ]
                           │
                           ▼
┌────────────────────────────────────────────────────────────────────────────┐
│              CREATE SYMBOLIC CIRCUIT (pytket)                              │
│                                                                            │
│    Symbols: f_0, f_1, ..., f_{n-1}                                        │
│                                                                            │
│    ┌───┐                                                                   │
│    │ H │ on all qubits (initialization)                                   │
│    └───┘                                                                   │
│         │  × r repetitions                                                 │
│    ┌──────────────┐                                                        │
│    │ Rz(γ·fᵢ/π)  │ on each qubit i                                       │
│    └──────────────┘                                                        │
│    ┌──────────────────────────────┐                                        │
│    │ XXPhase(γ²(1-fᵢ)(1-fⱼ))    │ on entangled pairs (i,j)               │
│    └──────────────────────────────┘                                        │
└────────────────────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────────────────────────┐
│              SYMBOL SUBSTITUTION                                           │
│    symbol_map = {f_0: x₁, f_1: x₂, ..., f_{n-1}: xₙ}                     │
│    circuit.symbol_substitution(symbol_map)                                │
└────────────────────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────────────────────────┐
│              CIRCUIT COMPILATION (pytket)                                  │
│    • Map to linear architecture                                           │
│    • Decompose BRIDGE gates (for non-adjacent qubits)                     │
│    • Insert SWAPs as needed                                               │
└────────────────────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────────────────────────┐
│              SERIALIZE TO GATE LIST                                        │
│    Output: [(0, 0, -1, 0), (0, 1, -1, 0), (2, 0, -1, θ₀), ...]           │
└────────────────────────────────────────────────────────────────────────────┘
```

## 6.4 MPI Parallelization Strategy

### 6.4.1 Data Distribution

```
Data X = [x₁, x₂, ..., xₙ] distributed across P processes:

Process 0: [x₁, ..., x_{N/P}]         → Chunk X₀
Process 1: [x_{N/P+1}, ..., x_{2N/P}] → Chunk X₁
...
Process P-1: [x_{(P-1)N/P+1}, ..., xₙ] → Chunk X_{P-1}
```

### 6.4.2 Round Robin Communication

```
Iteration 0:                    Iteration 1:
┌─────┐  ┌─────┐  ┌─────┐      ┌─────┐  ┌─────┐  ┌─────┐
│ P0  │  │ P1  │  │ P2  │      │ P0  │  │ P1  │  │ P2  │
│ Y0  │  │ Y1  │  │ Y2  │  →   │ Y2  │  │ Y0  │  │ Y1  │
└─────┘  └─────┘  └─────┘      └─────┘  └─────┘  └─────┘
   ↓        ↓        ↓            ↓        ↓        ↓
Compute  Compute  Compute      Compute  Compute  Compute
K[0,0]   K[1,1]   K[2,2]       K[0,2]   K[1,0]   K[2,1]

Final: MPI_Reduce to sum all partial kernel matrices
```

### 6.4.3 Symmetry Exploitation

For training kernel (X = Y), K_ij = K_ji:
- Only compute upper triangle
- Copy to lower triangle
- Reduces computation by ~50%

### 6.4.4 Checkpointing

```python
if minutes_per_checkpoint is not None:
    if last_checkpoint_time + 60*minutes_per_checkpoint < MPI.Wtime():
        np.save(checkpoint_file, kernel_mat)
        last_checkpoint_time = MPI.Wtime()
```

## 6.5 Output Artifacts

| Artifact | Location | Description |
|----------|----------|-------------|
| Training Kernel | `kernels/TrainKernel_*.npy` | N×N kernel matrix |
| Test Kernel | `kernels/TestKernel_*.npy` | M×N kernel matrix |
| Profiling Data | `*.json` | Performance metrics |
| Checkpoints | `tmp/checkpoint_*.npy` | Recovery files |
| Trained Model | `model/*.pkl` | Serialized classifier |
| Feature Arrays | `pqf_arr/*.npy` | Quantum features |
| Results | `Result_*.csv` | Predictions and metrics |

---

# 7. Implementation Reference

## 7.1 Core Classes

### 7.1.1 ProjectedKernelStateAnsatz

**Location:** `projected_kernel_ansatz.py`

```python
class ProjectedKernelStateAnsatz:
    """
    Creates parameterized quantum circuits for projected kernel computation.

    Attributes:
        ansatz_circ (Circuit): The pytket circuit with symbolic parameters
        feature_symbol_list (List[Symbol]): Symbols f_0, f_1, ..., f_{n-1}
        reps (int): Number of circuit layer repetitions
        gamma (float): Rotation scaling parameter
        num_features (int): Number of qubits/features
        hadamard_init (bool): Whether to apply initial Hadamard layer
        entanglement_map (List[Tuple[int, int]]): Qubit connectivity
    """

    def __init__(
        self,
        num_features: int,       # Number of qubits = features
        reps: int,               # Layer repetitions
        gamma: float,            # Rotation scaling [0.1, 1.0]
        entanglement_map: List[Tuple[int, int]],
        ansatz: str,             # "hamiltonian" or "magic"
        hadamard_init: bool = True
    )

    def circuit_for_data(self, feature_values: List[float]) -> Circuit
    def circuit_to_list(self, circuit: Circuit) -> List[List]
    def hamiltonian_ansatz(self) -> int
    def magic_ansatz(self) -> int
```

### 7.1.2 Key Functions

**build_kernel_matrix:**
```python
def build_kernel_matrix(
    mpi_comm,                          # MPI communicator
    ansatz: ProjectedKernelStateAnsatz,
    X: np.ndarray,                     # Data matrix (N, d)
    Y: Optional[np.ndarray] = None,    # Optional second dataset
    alpha: float = 1,                  # Kernel bandwidth
    info_file: Optional[str] = None,   # Profiling output file
    cpu_max_mem: int = 6,              # Memory limit (GB)
    minutes_per_checkpoint: Optional[int] = None
) -> np.ndarray
```

**entanglement_graph:**
```python
def entanglement_graph(nq: int, nn: int) -> List[Tuple[int, int]]
# Example: entanglement_graph(5, 1) -> [(0,1), (2,3), (1,2), (3,4)]
```

## 7.2 Parameter Reference

### 7.2.1 Quantum Circuit Parameters

| Parameter | Symbol | Type | Range | Default | Description |
|-----------|--------|------|-------|---------|-------------|
| `num_features` | n | int | [2, 100+] | - | Number of qubits/features |
| `reps` | r | int | [1, 20] | 2 | Circuit layer repetitions |
| `gamma` | γ | float | (0, 1] | 1.0 | Rotation scaling factor |
| `alpha` | α | float | (0, 10] | 1.0 | Kernel bandwidth |
| `ansatz` | - | str | {"hamiltonian", "magic"} | "hamiltonian" | Circuit type |
| `hadamard_init` | - | bool | {True, False} | True | Initial H gates |

### 7.2.2 Data Parameters

| Parameter | Type | Range | Description |
|-----------|------|-------|-------------|
| `n_illicit` | int | [1, N/2] | Fraud class sample size |
| `n_licit` | int | [1, N/2] | Legitimate class sample size |
| `data_seed` | int | Any | Random seed for reproducibility |
| `test_frac` | float | (0, 1) | Test set fraction (default 0.2) |

### 7.2.3 SVM Parameters

| Parameter | Type | Values | Description |
|-----------|------|--------|-------------|
| `C` | float | [0.01, 2.0] | Regularization parameter |
| `kernel` | str | "precomputed" | Must be precomputed for PQK |
| `tol` | float | 1e-5 | Optimization tolerance |

### 7.2.4 Parameter Guidelines

| Scenario | Recommended Settings |
|----------|---------------------|
| Few features (n < 10) | γ=1.0, reps=2-3, α=0.5-1.0 |
| Many features (n > 20) | γ=0.3-0.5, reps=5-10, α=0.1-0.5 |
| Small data (N < 500) | Higher C (1.0-2.0), α=0.5-1.0 |
| Large data (N > 5000) | Lower C (0.1-0.5), α=0.1-0.3 |

## 7.3 Gate Encoding Specification

| Code | Gate | Qubits | Parameter | ITensor Operator |
|------|------|--------|-----------|------------------|
| 0 | H | 1 | None | `"H"` |
| 1 | Rx | 1 | angle | `"Rx"` with `alpha=angle` |
| 2 | Rz | 1 | angle | `"Rz"` with `alpha=angle` |
| 3 | XXPhase | 2 | angle | `expHermitian(X⊗X, -i*θ)` |
| 4 | ZZPhase | 2 | angle | `expHermitian(Z⊗Z, -i*θ)` |
| 5 | SWAP | 2 | None | Manual matrix |
| 6 | T | 1 | None | `"T"` |
| 7 | CZ | 2 | None | Manual matrix |

**Gate List Format:**
```python
# [code, qubit1, qubit2, parameter]
# qubit2 = -1 for single-qubit gates
example_circuit = [
    [0, 0, -1, 0],      # H on qubit 0
    [0, 1, -1, 0],      # H on qubit 1
    [2, 0, -1, 0.5],    # Rz(0.5) on qubit 0
    [3, 0, 1, 0.25],    # XXPhase(0.25) on qubits 0,1
]
```

## 7.4 File Format Specifications

### 7.4.1 Input Data (CSV)

```csv
Class,Feature 1,Feature 2,...,Feature N
0,1.234,5.678,...,9.012
1,3.456,7.890,...,1.234
```

### 7.4.2 Kernel Matrix (NumPy)

```python
# Training kernel: Shape (N_train, N_train), dtype float64
kernel_train = np.load("kernels/TrainKernel_Nf-12_r-10_g-1_Ntr-100.npy")

# Test kernel: Shape (N_test, N_train)
kernel_test = np.load("kernels/TestKernel_Nf-12_r-10_g-1_Ntr-100.npy")
```

**Filename Convention:**
```
{Type}Kernel_Nf-{num_features}_r-{reps}_g-{gamma}_Ntr-{n_train}.npy
```

### 7.4.3 Profiling JSON

```json
{
    "lenX": [800, "entries"],
    "lenY": [200, "entries"],
    "r0_circ_gen": [1.23, "seconds"],
    "r0_circ_sim": [45.67, "seconds"],
    "avg_circ_sim": [0.0571, "seconds"],
    "kernel_mat_time": [120.5, "seconds"],
    "total_time": [180.3, "seconds"]
}
```

---

# 8. Fraud Detection Application

## 8.1 Problem Domain

### 8.1.1 Fraud Detection Challenges

| Challenge | Description | QiML Solution |
|-----------|-------------|---------------|
| Class Imbalance | Fraud < 5% of transactions | Balanced sampling |
| High Dimensionality | Many features | Multi-qubit encoding |
| Complex Patterns | Non-linear interactions | Quantum entanglement |
| Adversarial Evolution | Fraudsters adapt | Kernel method robustness |
| Real-time Requirements | Sub-second decisions | Inference optimization |

### 8.1.2 Binary Classification Formulation

```
y = { 0  (Fraud / Illicit)
    { 1  (Legitimate / Licit)
```

## 8.2 The Elliptic Bitcoin Dataset

### 8.2.1 Dataset Statistics

| Attribute | Value |
|-----------|-------|
| Total transactions | 203,769 |
| Labeled transactions | 46,564 |
| Unlabeled transactions | 157,205 |
| Illicit (fraud) | 4,545 (9.8% of labeled) |
| Licit (legitimate) | 42,019 (90.2% of labeled) |
| Features per transaction | 166 |
| Time steps | 49 |

### 8.2.2 Feature Structure

**Local Features (94):** Transaction characteristics, aggregated from one-hop neighborhood

**Aggregated Features (72):** Neighborhood statistics, graph structural features

### 8.2.3 Data Preprocessing

```python
# Load raw data
feature_data = pd.read_csv('elliptic_txs_features.csv')
node_class = pd.read_csv('elliptic_txs_classes.csv')

# Encode labels: "1" = illicit → 0, "2" = licit → 1
node_class.loc[node_class["Class"] == "1", "Class"] = 0
node_class.loc[node_class["Class"] == "2", "Class"] = 1

# Remove unlabeled data
clean_data = feature_data.drop(np.where(node_class['Class']=='unknown')[0])
```

## 8.3 Model Configuration

### 8.3.1 Recommended Hyperparameters

```python
# Feature selection
num_features = 12      # 12-20 most important features

# Circuit parameters
reps = 10              # Deep circuit for complex patterns
gamma = 1.0            # Full rotation scaling

# Kernel parameter
alpha = 0.5            # Moderate bandwidth

# Data parameters
n_illicit = 100        # Balanced sampling
n_licit = 100
data_seed = 456

# SVM regularization
C_values = [2, 1.5, 1, 0.5, 0.1, 0.05, 0.01]
```

### 8.3.2 Hyperparameter Tuning Strategy

**Phase 1: Coarse Grid**
- num_features ∈ {8, 12, 16, 20}
- reps ∈ {2, 5, 10}
- gamma ∈ {0.5, 1.0}
- alpha ∈ {0.1, 0.5, 1.0}

**Phase 2: Fine-Tune**
- gamma ∈ {0.8, 0.9, 1.0, 1.1}
- alpha ∈ {0.3, 0.5, 0.7}
- C ∈ {0.01, 0.05, 0.1, 0.5, 1.0, 1.5, 2.0}

**Phase 3: Validate**
- Multiple seeds: {123, 456, 789, 101, 202}
- Report mean ± std

## 8.4 Evaluation Metrics

### 8.4.1 Primary Metrics

| Metric | Formula | Target | Interpretation |
|--------|---------|--------|----------------|
| Recall | TP/(TP+FN) | > 0.80 | Fraud capture rate |
| Precision | TP/(TP+FP) | > 0.70 | Alert reliability |
| F1 Score | 2PR/(P+R) | > 0.75 | Balanced performance |
| AUC-ROC | Area under ROC | > 0.85 | Discrimination ability |

### 8.4.2 Confusion Matrix

```
                    Predicted
                  Fraud   Legit
              ┌─────────┬─────────┐
Actual Fraud  │   TP    │   FN    │  → Recall = TP/(TP+FN)
              │  (Hit)  │ (Miss)  │
              ├─────────┼─────────┤
Actual Legit  │   FP    │   TN    │  → Specificity = TN/(TN+FP)
              │ (False  │(Correct │
              │ Alarm)  │ Clear)  │
              └─────────┴─────────┘
                  ↓
              Precision = TP/(TP+FP)
```

### 8.4.3 Cost-Sensitive Analysis

| Error Type | Business Impact | Cost Ratio |
|------------|-----------------|------------|
| False Negative (Miss fraud) | Direct financial loss | 10-100x |
| False Positive (Flag legitimate) | Customer friction | 1x |

## 8.5 Expected Performance

### 8.5.1 Benchmark Results

| Model | Accuracy | Precision | Recall | F1 | AUC |
|-------|----------|-----------|--------|-----|-----|
| PQK-SVM (Hamiltonian) | 0.85-0.92 | 0.80-0.88 | 0.82-0.90 | 0.81-0.89 | 0.88-0.94 |
| RBF-SVM (Baseline) | 0.82-0.88 | 0.75-0.85 | 0.78-0.86 | 0.77-0.85 | 0.84-0.90 |
| LightGBM + QF | 0.88-0.94 | 0.85-0.92 | 0.85-0.91 | 0.85-0.91 | 0.90-0.96 |

### 8.5.2 Computational Benchmarks

| Configuration | Train Time | Inference | Memory |
|---------------|------------|-----------|--------|
| n=12, reps=10, N=200 | ~5 min | ~100 ms | ~500 MB |
| n=20, reps=10, N=200 | ~15 min | ~200 ms | ~1 GB |
| n=12, reps=10, N=1000 | ~2 hours | ~100 ms | ~2 GB |

---

# 9. Deployment Guide

## 9.1 Command-Line Usage

### 9.1.1 Kernel Mode (SVM)

```bash
mpirun -n <nodes> python main.py \
    <num_features> \
    <reps> \
    <gamma> \
    <alpha> \
    <n_illicit> \
    <n_licit> \
    <data_seed> \
    <data_file>
```

**Example:**
```bash
mpirun -n 4 python main_dlp.py 12 10 1 0.5 100 100 456 bitstrings_12_preproc.csv
```

### 9.1.2 Feature Mode (LightGBM)

```bash
mpirun -n <nodes> python main.py \
    <method: train|test|generate> \
    <train_data_info> \
    <test_data_info> \
    <target_label> \
    <train_flag: True|False>
```

## 9.2 Docker Deployment

### 9.2.1 Build

```bash
cd Installation-Script
docker build -t qiml .
```

### 9.2.2 Run

```bash
docker run \
   --env MPI_NODES=4 \
   --env NUM_FEATURES=12 \
   --env REPS=10 \
   --env GAMMA=1 \
   --env ALPHA=0.5 \
   --env N_ILLICIT=100 \
   --env N_LICIT=100 \
   --env DATA_SEED=456 \
   --env DATA_FILE=bitstrings_12_preproc.csv \
   qiml
```

## 9.3 ITensor Setup

### 9.3.1 Installation Steps

1. Install ITensor from https://itensor.org/docs.cgi?vers=cppv3&page=install
2. Copy `qubit.h` to `~/itensor/itensor/mps/sites/`
3. Add `#include "itensor/mps/sites/qubit.h"` to `~/itensor/itensor/all_mps.h`
4. Compile shared library

### 9.3.2 Compilation Commands

**Linux (GCC):**
```bash
g++ -m64 -std=c++17 -fconcepts -fPIC -c \
    -I. -I<pybind11_include> -I<itensor_path> \
    -O2 -DNDEBUG -Wall -Wno-unknown-pragmas \
    -o helloitensor.o helloitensor.cc \
    -I<python_include>

g++ -m64 -shared -std=c++17 -fconcepts -fPIC \
    helloitensor.o -o helloitensor.so \
    -L<itensor_lib> -litensor -lpthread -lblas -llapack
```

**macOS (Clang):**
```bash
clang++ -shared -undefined dynamic_lookup -std=c++17 -fPIC \
    -Wno-gcc-compat -I<pybind11_include> -I<itensor_path> \
    -O2 -DNDEBUG helloitensor.cc -o helloitensor.so \
    -L<itensor_lib> -litensor -framework Accelerate \
    -I<python_include>
```

## 9.4 Production Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    PRODUCTION DEPLOYMENT ARCHITECTURE                       │
└─────────────────────────────────────────────────────────────────────────────┘

Transaction Stream
        │
        ▼
┌─────────────────────┐
│   Pre-Filter        │  Fast classical rules (< 1ms)
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│   Classical Model   │  LightGBM screening (1-5ms)
└─────────────────────┘
        │ High-risk flagged
        ▼
┌─────────────────────┐
│   QiML Model        │  Quantum-enhanced analysis (50-200ms)
└─────────────────────┘
        │
        ▼
    Approve / Block / Review
```

## 9.5 Environment Variables

```bash
# MPI Configuration
NUM_NODES=4
OMPI_ALLOW_RUN_AS_ROOT=1

# OpenMP Configuration
OMP_NUM_THREADS=24
OMP_PROC_BIND=close
OMP_PLACES=cores

# ITensor Configuration
ITENSOR_USE_OMP=1
MKL_NUM_THREADS=4
OPENBLAS_NUM_THREADS=4

# Memory Optimization
OMP_STACKSIZE=2M
```

---

# 10. References

## 10.1 Core Papers

1. **Huang, H.-Y. et al.** (2021). "Power of data in quantum machine learning." *Nature Communications*, 12, 2631. https://www.nature.com/articles/s41467-021-22539-9

2. **Havlíček, V. et al.** (2019). "Supervised learning with quantum-enhanced feature spaces." *Nature*, 567, 209-212. https://www.nature.com/articles/s41586-019-0980-2

3. **Kübler, J. M., Buchholz, S., & Schölkopf, B.** (2021). "The Inductive Bias of Quantum Kernels." *NeurIPS 2021*.

4. **Fishman, M., White, S. R., & Stoudenmire, E. M.** (2022). "The ITensor Software Library for Tensor Network Calculations." *SciPost Physics Codebases*, 4. https://scipost.org/SciPostPhysCodeb.4

## 10.2 Fraud Detection Applications

5. **Heredge, J. et al.** (2023). "Quantum Multiple Kernel Learning in Financial Classification Tasks." arXiv:2312.00260. https://arxiv.org/abs/2312.00260

6. **Vasquez, A. C. et al.** (2023). "Financial Fraud Detection: A Comparative Study of Quantum Machine Learning Models." arXiv:2308.05237. https://arxiv.org/abs/2308.05237

7. **Weber, M. et al.** (2019). "Anti-Money Laundering in Bitcoin: Experimenting with Graph Convolutional Networks for Financial Forensics." *SIGKDD Workshop on Anomaly Detection in Finance*.

## 10.3 Additional Resources

8. **IBM Qiskit Documentation** - ZZFeatureMap. https://docs.quantum.ibm.com/api/qiskit/qiskit.circuit.library.ZZFeatureMap

9. **PennyLane** - Kernel-based training of quantum models. https://pennylane.ai/qml/demos/tutorial_kernel_based_training

10. **ITensor Documentation** - https://itensor.org/docs.cgi

---

# Appendix A: Quick Reference Card

## A.1 Command Line

```bash
# Kernel mode
mpirun -n 4 python main.py 12 10 1 0.5 100 100 456 data.csv

# Docker
docker build -t qiml . && docker run --env NUM_NODES=4 qiml
```

## A.2 Key Parameters

| Parameter | Typical Value | Description |
|-----------|---------------|-------------|
| num_features | 12 | Qubits/features |
| reps | 10 | Circuit depth |
| gamma | 1.0 | Rotation scaling |
| alpha | 0.5 | Kernel bandwidth |
| C | 0.1-1.0 | SVM regularization |

## A.3 Key Formulas

**PQK Kernel:**
```
k(x,x') = exp(-α Σᵢ 2[(⟨Xᵢ⟩ˣ-⟨Xᵢ⟩ˣ')² + (⟨Yᵢ⟩ˣ-⟨Yᵢ⟩ˣ')² + (⟨Zᵢ⟩ˣ-⟨Zᵢ⟩ˣ')²])
```

**Hamiltonian Ansatz:**
```
U(x) = H^⊗n ∏ᵣ [ ∏ᵢ Rz(γxᵢ/π) ∏_{(i,j)} R_XX(γ²(1-xᵢ)(1-xⱼ)) ]
```

## A.4 Expected Performance

| Metric | Target |
|--------|--------|
| Accuracy | 85-92% |
| Recall | 82-90% |
| F1 Score | 81-89% |
| AUC | 88-94% |
| Inference | <200ms |

---

**End of Document**

*Document Version 1.0 | January 2026 | Enterprise Quantum Engineering Team*
