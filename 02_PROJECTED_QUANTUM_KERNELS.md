# Projected Quantum Kernels: Mathematical Framework

This document provides a rigorous mathematical treatment of Projected Quantum Kernels (PQK) as implemented in this QiML pipeline.

---

## Table of Contents

1. [From Quantum States to Classical Features](#1-from-quantum-states-to-classical-features)
2. [Pauli Expectation Values](#2-pauli-expectation-values)
3. [Reduced Density Matrix Geometry](#3-reduced-density-matrix-geometry)
4. [The PQK Kernel Function](#4-the-pqk-kernel-function)
5. [Algorithmic Complexity Analysis](#5-algorithmic-complexity-analysis)
6. [Circuit Implementation Details](#6-circuit-implementation-details)
7. [Hyperparameter Analysis](#7-hyperparameter-analysis)

---

## 1. From Quantum States to Classical Features

### 1.1 The Projection Principle

The key innovation of Projected Quantum Kernels is **classical projection**: instead of computing global overlaps in exponentially large Hilbert space, we extract local information through single-qubit observables.

**Workflow:**

```
Classical Data x ∈ ℝᵈ
       ↓
   U(x)|0⟩^⊗n      (Quantum Encoding)
       ↓
   |ψ(x)⟩          (Quantum State in 2^n-dim Hilbert Space)
       ↓
   ⟨P_i⟩           (Local Measurements: 3n real numbers)
       ↓
   Φ(x) ∈ ℝ^{3n}   (Classical Feature Vector)
       ↓
   k(x,x') = exp(-α||Φ(x)-Φ(x')||²)  (RBF Kernel)
```

### 1.2 Mathematical Formulation

Given an encoded quantum state $|\psi(x)\rangle = U(x)|0\rangle^{\otimes n}$, we define the projected quantum feature map:

$$\Phi: \mathbb{R}^d \rightarrow \mathbb{R}^{3n}$$

$$\Phi(x) = \begin{pmatrix} \langle X_1 \rangle_x \\ \langle Y_1 \rangle_x \\ \langle Z_1 \rangle_x \\ \vdots \\ \langle X_n \rangle_x \\ \langle Y_n \rangle_x \\ \langle Z_n \rangle_x \end{pmatrix}$$

where $\langle P_i \rangle_x = \langle \psi(x) | P_i | \psi(x) \rangle$ for Pauli operators $P \in \{X, Y, Z\}$.

### 1.3 Information Preservation

**Question:** How much information is lost by projecting from $2^n$ to $3n$ dimensions?

**Answer:** For separable (product) states, no information is lost—the local observables fully characterize the state. For entangled states, global correlations are partially captured through entanglement-mediated local changes.

**Theorem (Information Content):** For a pure state $|\psi\rangle$, the set $\{\langle P_i \rangle : P \in \{X,Y,Z\}, i \in [n]\}$ uniquely determines all single-site reduced density matrices $\rho_i = \text{Tr}_{\bar{i}}(|\psi\rangle\langle\psi|)$.

---

## 2. Pauli Expectation Values

### 2.1 Pauli Matrices

The Pauli matrices form a basis for $2 \times 2$ Hermitian operators:

$$X = \sigma_x = \begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix}, \quad Y = \sigma_y = \begin{pmatrix} 0 & -i \\ i & 0 \end{pmatrix}, \quad Z = \sigma_z = \begin{pmatrix} 1 & 0 \\ 0 & -1 \end{pmatrix}$$

**Properties:**
- Hermitian: $P^\dagger = P$
- Unitary: $P^2 = I$
- Traceless: $\text{Tr}(P) = 0$
- Eigenvalues: $\pm 1$

### 2.2 Bloch Vector Representation

Any single-qubit state can be written as:

$$\rho = \frac{1}{2}(I + \vec{r} \cdot \vec{\sigma}) = \frac{1}{2}(I + r_x X + r_y Y + r_z Z)$$

where $\vec{r} = (r_x, r_y, r_z)$ is the **Bloch vector** with:

$$r_x = \langle X \rangle, \quad r_y = \langle Y \rangle, \quad r_z = \langle Z \rangle$$

For pure states: $|\vec{r}| = 1$ (on Bloch sphere surface)
For mixed states: $|\vec{r}| < 1$ (inside Bloch sphere)

### 2.3 Computing Expectation Values

For a state $|\psi\rangle = \alpha|0\rangle + \beta|1\rangle$:

$$\langle X \rangle = \alpha^*\beta + \alpha\beta^* = 2\text{Re}(\alpha^*\beta)$$
$$\langle Y \rangle = i(\alpha^*\beta - \alpha\beta^*) = 2\text{Im}(\alpha^*\beta)$$
$$\langle Z \rangle = |\alpha|^2 - |\beta|^2$$

### 2.4 Multi-Qubit Systems

For qubit $i$ in an n-qubit system:

$$\langle P_i \rangle = \langle \psi | (I^{\otimes (i-1)} \otimes P \otimes I^{\otimes (n-i)}) | \psi \rangle = \text{Tr}(\rho_i P)$$

This requires computing the reduced density matrix $\rho_i$ by tracing out all other qubits.

---

## 3. Reduced Density Matrix Geometry

### 3.1 Single-Qubit RDM

The reduced density matrix for qubit $i$ is:

$$\rho_i = \text{Tr}_{\bar{i}}(|\psi\rangle\langle\psi|) = \frac{1}{2}\begin{pmatrix} 1 + \langle Z_i \rangle & \langle X_i \rangle - i\langle Y_i \rangle \\ \langle X_i \rangle + i\langle Y_i \rangle & 1 - \langle Z_i \rangle \end{pmatrix}$$

### 3.2 Frobenius Distance

The Frobenius (Hilbert-Schmidt) distance between two density matrices is:

$$d_F(\rho, \sigma) = \|\rho - \sigma\|_F = \sqrt{\text{Tr}((\rho - \sigma)^2)}$$

**Explicit Calculation:**

For single-qubit RDMs with Bloch vectors $\vec{r}$ and $\vec{s}$:

$$\|\rho - \sigma\|_F^2 = \text{Tr}\left[\left(\frac{\vec{r} - \vec{s}}{2} \cdot \vec{\sigma}\right)^2\right]$$

Using $\text{Tr}(P_a P_b) = 2\delta_{ab}$:

$$\|\rho - \sigma\|_F^2 = \frac{1}{2}\|\vec{r} - \vec{s}\|^2 = \frac{1}{2}\left[(r_x - s_x)^2 + (r_y - s_y)^2 + (r_z - s_z)^2\right]$$

### 3.3 Geometric Interpretation

The Frobenius distance on the Bloch sphere:
- **Maximum distance:** $d_F^{\max} = \sqrt{2}$ (antipodal pure states)
- **Orthogonal states:** $d_F = 1$ (e.g., $|0\rangle$ and $|+\rangle$)
- **Same state:** $d_F = 0$

---

## 4. The PQK Kernel Function

### 4.1 Definition

The Projected Quantum Kernel implemented in this codebase is:

$$k_{\text{PQK}}(x, x') = \exp\left(-\alpha \cdot D(x, x')\right)$$

where the **quantum distance** is:

$$D(x, x') = \sum_{i=1}^{n} 2 \cdot \|\rho_i(x) - \rho_i(x')\|_F^2$$

### 4.2 Expanded Form

Substituting the Frobenius norm:

$$D(x, x') = \sum_{i=1}^{n} \left[(\langle X_i \rangle^x - \langle X_i \rangle^{x'})^2 + (\langle Y_i \rangle^x - \langle Y_i \rangle^{x'})^2 + (\langle Z_i \rangle^x - \langle Z_i \rangle^{x'})^2\right]$$

This is exactly the squared Euclidean distance in the projected feature space:

$$D(x, x') = \|\Phi(x) - \Phi(x')\|_2^2$$

### 4.3 Kernel Properties

**Theorem:** $k_{\text{PQK}}$ is a valid Mercer kernel.

*Proof:* The kernel is the composition of:
1. A feature map $\Phi: \mathbb{R}^d \rightarrow \mathbb{R}^{3n}$ (quantum encoding + measurement)
2. The RBF kernel on the feature space

Since RBF kernels are valid for any feature space, and $\Phi$ is well-defined, $k_{\text{PQK}}$ is positive semi-definite. ∎

**Properties:**
- $k_{\text{PQK}}(x, x) = 1$ (self-similarity)
- $k_{\text{PQK}}(x, x') = k_{\text{PQK}}(x', x)$ (symmetry)
- $0 < k_{\text{PQK}}(x, x') \leq 1$ (bounded)

### 4.4 Relationship to RBF Kernel

The PQK can be rewritten as:

$$k_{\text{PQK}}(x, x') = k_{\text{RBF}}(\Phi(x), \Phi(x'); \gamma = \alpha)$$

The key difference from classical RBF:
- **Classical RBF:** $k(x, x') = \exp(-\gamma\|x - x'\|^2)$ operates on raw features
- **PQK:** Operates on quantum-transformed features $\Phi(x)$

### 4.5 Kernel Matrix Construction

For training data $\{x_1, \ldots, x_N\}$, the kernel matrix is:

$$K_{ij} = k_{\text{PQK}}(x_i, x_j) = \exp\left(-\alpha \|\Phi(x_i) - \Phi(x_j)\|^2\right)$$

**Complexity:**
- Naive: $O(N^2)$ kernel evaluations
- With symmetry: $O(N(N+1)/2)$ evaluations
- Parallelizable with MPI round-robin

---

## 5. Algorithmic Complexity Analysis

### 5.1 Classical Simulation Complexity

**Per Data Point:**
- Circuit simulation (MPS): $O(n \cdot r \cdot \chi^3)$
  - $n$: number of qubits
  - $r$: circuit repetitions
  - $\chi$: MPS bond dimension
- Expectation value computation: $O(n \cdot \chi^2)$

**Kernel Matrix Construction:**
- Total MPS simulations: $O(N_{\text{train}} + N_{\text{test}})$
- Kernel entry computation: $O(N_{\text{train}} \cdot N_{\text{test}} \cdot n)$

### 5.2 MPI Parallelization

The implementation uses **round-robin distribution**:

1. **Data partitioning:** Divide $N$ samples across $P$ processes
2. **Local computation:** Each process simulates $\lceil N/P \rceil$ circuits
3. **Kernel tiling:** Compute $\lceil N/P \rceil \times \lceil N/P \rceil$ tiles per iteration
4. **Communication:** Pass Y-chunks in a ring topology

**Scaling:**
- Strong scaling: $T(P) \approx T(1)/P$ for large $N$
- Communication overhead: $O(N/P)$ per round-robin iteration

### 5.3 Memory Complexity

**Per Process:**
- MPS storage: $O(n \cdot \chi^2)$ per state
- Chunk storage: $O(N/P \cdot n \cdot \chi^2)$
- Kernel matrix tile: $O((N/P)^2)$

### 5.4 Comparison with Direct Quantum Kernel

| Aspect | FQK (Full Quantum) | PQK (Projected) |
|--------|-------------------|-----------------|
| State storage | $O(2^n)$ | $O(n \cdot \chi^2)$ |
| Kernel entry | $O(2^n)$ | $O(n)$ |
| Scalability | ~30 qubits max | 100+ qubits feasible |
| Hardware | Requires quantum device | Classical simulation |

---

## 6. Circuit Implementation Details

### 6.1 Gate-Level Decomposition

The Hamiltonian ansatz circuit decomposes as:

**Layer 1: Hadamard Initialization**
```
|0⟩ ─[H]─ → |+⟩ = (|0⟩ + |1⟩)/√2
```

**Layer 2: Rz Encoding (per qubit i)**
```
|ψ⟩ ─[Rz(γxᵢ/π)]─ → e^{-iγxᵢZ/(2π)} |ψ⟩
```

**Layer 3: XXPhase Entanglement (per pair (i,j))**
```
     ┌─────────────────────────┐
─────┤                         ├─────
     │  exp(-iθ X⊗X / 2)       │
─────┤  θ = γ²(1-xᵢ)(1-xⱼ)    ├─────
     └─────────────────────────┘
```

### 6.2 Gate Encoding for ITensor

The circuit is serialized as a list of gate descriptors:

| Code | Gate | Parameters |
|------|------|------------|
| 0 | H | qubit_index |
| 1 | Rx | qubit_index, angle |
| 2 | Rz | qubit_index, angle |
| 3 | XXPhase | qubit1, qubit2, angle |
| 4 | ZZPhase | qubit1, qubit2, angle |
| 5 | SWAP | qubit1, qubit2 |
| 6 | T | qubit_index |
| 7 | CZ | qubit1, qubit2 |

### 6.3 Circuit Compilation

The pytket compiler performs:
1. **Architecture mapping:** Map logical to physical qubits (line topology)
2. **BRIDGE decomposition:** Replace long-range gates with SWAP chains
3. **Gate optimization:** Merge consecutive single-qubit gates

---

## 7. Hyperparameter Analysis

### 7.1 Rotation Scaling Parameter (γ)

**Role:** Controls the magnitude of data-dependent rotations.

**Mathematical Effect:**
- Single-qubit rotation: $\theta_i = \gamma x_i / \pi$
- Two-qubit interaction: $\theta_{ij} = \gamma^2 (1 - x_i)(1 - x_j)$

**Guidelines:**
| Qubits | Recommended γ |
|--------|---------------|
| 5-10 | 0.8 - 1.0 |
| 10-20 | 0.5 - 0.8 |
| 20-50 | 0.3 - 0.5 |
| 50+ | 0.1 - 0.3 |

**Reasoning:** Larger systems require smaller γ to prevent over-rotation and maintain meaningful quantum correlations.

### 7.2 Kernel Bandwidth Parameter (α)

**Role:** Controls the kernel sensitivity to feature differences.

**Mathematical Effect:**
$$k(x, x') = \exp(-\alpha \cdot D(x, x'))$$

**Analysis:**
- **Small α (< 0.1):** Broad kernel, all points appear similar → underfitting
- **Large α (> 2.0):** Sharp kernel, only identical points are similar → overfitting
- **Optimal α:** Typically in range [0.1, 1.0], found via cross-validation

### 7.3 Circuit Depth (reps)

**Role:** Number of layer repetitions in the feature map.

**Trade-offs:**
| reps | Expressiveness | Entanglement | Simulation Cost |
|------|---------------|--------------|-----------------|
| 1 | Low | Weak | O(n) |
| 2-5 | Medium | Moderate | O(n·reps) |
| 5-10 | High | Strong | O(n·reps) |
| 10+ | Saturated | Maximal | Expensive |

**Recommendation:** Start with `reps=2`, increase if kernel values cluster near 1.

### 7.4 Entanglement Graph

**Current Implementation:** Linear nearest-neighbor

$$E_{\text{linear}} = \{(i, i+1) : i \in [0, n-2]\}$$

**Alternatives:**
- **Full connectivity:** $E_{\text{full}} = \{(i, j) : i < j\}$ — exponentially expensive
- **Circular:** $E_{\text{circ}} = E_{\text{linear}} \cup \{(0, n-1)\}$
- **Random sparse:** Sample $O(n)$ edges randomly

### 7.5 SVM Regularization (C)

**Role:** Trade-off between margin maximization and misclassification penalty.

**Interaction with Kernel:**
- Small α + large C → complex decision boundary
- Large α + small C → simple decision boundary

**Recommended Search Grid:**
```python
C_values = [0.01, 0.05, 0.1, 0.5, 1.0, 1.5, 2.0]
```

---

## Summary: Complete PQK Algorithm

```
Algorithm: Projected Quantum Kernel Classification

Input: Training data {(xᵢ, yᵢ)}ᵢ₌₁ᴺ, test data {x'ⱼ}ⱼ₌₁ᴹ
Parameters: γ (rotation), α (bandwidth), r (reps), C (SVM)

1. PREPROCESSING
   - Normalize features to [-π, π] or [-π/4, π/4]
   - Apply StandardScaler, then MinMaxScaler

2. QUANTUM FEATURE EXTRACTION
   For each data point x:
     a. Build circuit U(x) with Hamiltonian ansatz
     b. Simulate with MPS: |ψ(x)⟩ = U(x)|0⟩^⊗n
     c. Compute Φ(x) = (⟨X₁⟩, ⟨Y₁⟩, ⟨Z₁⟩, ..., ⟨Xₙ⟩, ⟨Yₙ⟩, ⟨Zₙ⟩)

3. KERNEL MATRIX CONSTRUCTION
   For i ∈ [N], j ∈ [N]:
     K_train[i,j] = exp(-α ||Φ(xᵢ) - Φ(xⱼ)||²)
   For i ∈ [M], j ∈ [N]:
     K_test[i,j] = exp(-α ||Φ(x'ᵢ) - Φ(xⱼ)||²)

4. SVM TRAINING & PREDICTION
   - Fit SVC(kernel="precomputed", C=C) on K_train, y
   - Predict labels for K_test

Output: Predicted labels ŷ, classification metrics
```

---

*Next: [03_TENSOR_NETWORK_SIMULATION.md](03_TENSOR_NETWORK_SIMULATION.md) - MPS-based quantum circuit simulation*
