# Tensor Network Simulation with Matrix Product States

This document explains how the QiML pipeline uses tensor network methods—specifically Matrix Product States (MPS)—to efficiently simulate quantum circuits on classical hardware via the ITensor library.

---

## Table of Contents

1. [Why Tensor Networks?](#1-why-tensor-networks)
2. [Matrix Product States (MPS)](#2-matrix-product-states-mps)
3. [Quantum Gates on MPS](#3-quantum-gates-on-mps)
4. [ITensor Implementation](#4-itensor-implementation)
5. [Expectation Value Computation](#5-expectation-value-computation)
6. [Computational Complexity](#6-computational-complexity)
7. [Limitations and Trade-offs](#7-limitations-and-trade-offs)

---

## 1. Why Tensor Networks?

### 1.1 The Exponential Wall

A naive representation of an n-qubit quantum state requires $2^n$ complex amplitudes:

$$|\psi\rangle = \sum_{i_1, \ldots, i_n \in \{0,1\}} c_{i_1 \ldots i_n} |i_1 \ldots i_n\rangle$$

| Qubits | Amplitudes | Memory (complex128) |
|--------|------------|---------------------|
| 20 | ~1M | 16 MB |
| 30 | ~1B | 16 GB |
| 40 | ~1T | 16 TB |
| 50 | ~1P | 16 PB |

Direct simulation becomes intractable beyond ~40-45 qubits on classical hardware.

### 1.2 The Tensor Network Solution

Tensor networks exploit the structure of **weakly entangled** states to achieve efficient representation:

**Key Insight:** Many physically relevant quantum states exhibit **area-law entanglement**—the entanglement entropy across a cut grows with the boundary size, not the volume. For 1D systems:

$$S(\rho_A) \leq c \cdot |\partial A|$$

Such states can be efficiently represented as **Matrix Product States**.

### 1.3 When MPS Works Well

MPS simulation is efficient when:
- Circuits have **linear (1D) connectivity**
- Entanglement remains **bounded** throughout computation
- Gates are applied **locally** (nearest-neighbor or few-body)

This matches our Hamiltonian ansatz with linear entanglement topology.

---

## 2. Matrix Product States (MPS)

### 2.1 Definition

An MPS represents an n-qubit state as a chain of tensors:

$$|\psi\rangle = \sum_{i_1, \ldots, i_n} A^{[1]}_{i_1} A^{[2]}_{i_2} \cdots A^{[n]}_{i_n} |i_1 \ldots i_n\rangle$$

where:
- $A^{[k]}_{i_k}$ is a matrix of dimensions $\chi_{k-1} \times \chi_k$
- $i_k \in \{0, 1\}$ is the physical index (qubit state)
- $\chi_k$ is the **bond dimension** at bond $k$

**Pictorial Representation:**

```
    i₁    i₂    i₃    i₄    i₅
    |     |     |     |     |
   [A¹]--[A²]--[A³]--[A⁴]--[A⁵]
       χ₁   χ₂   χ₃   χ₄
```

### 2.2 Bond Dimension and Entanglement

The bond dimension $\chi$ controls the **entanglement capacity**:

**Theorem (Schmidt Rank Bound):** For a bipartition at bond $k$, the Schmidt rank (number of non-zero singular values) is at most $\chi_k$.

**Consequence:** The entanglement entropy is bounded:
$$S(\rho_{\text{left}}) \leq \log_2(\chi_k)$$

### 2.3 Canonical Forms

MPS can be put into canonical forms to simplify computations:

**Left-Canonical Form:**
$$\sum_{i_k} A^{[k]\dagger}_{i_k} A^{[k]}_{i_k} = I$$

**Right-Canonical Form:**
$$\sum_{i_k} A^{[k]}_{i_k} A^{[k]\dagger}_{i_k} = I$$

**Mixed-Canonical Form (centered at site $k$):**
- Sites $1, \ldots, k-1$ are left-canonical
- Sites $k+1, \ldots, n$ are right-canonical
- Site $k$ contains the "orthogonality center"

### 2.4 Initial State: |0⟩^⊗n

The all-zeros state has a trivial MPS representation:

$$A^{[k]}_0 = 1, \quad A^{[k]}_1 = 0$$

with bond dimension $\chi = 1$ (a product state has no entanglement).

---

## 3. Quantum Gates on MPS

### 3.1 Single-Qubit Gates

A single-qubit gate $U$ acting on site $k$ is applied by:

$$\tilde{A}^{[k]}_{j} = \sum_{i} U_{ji} A^{[k]}_i$$

**Complexity:** $O(\chi^2)$ — does not increase bond dimension.

### 3.2 Hadamard Gate

$$H = \frac{1}{\sqrt{2}}\begin{pmatrix} 1 & 1 \\ 1 & -1 \end{pmatrix}$$

Applied to $|0\rangle$: $H|0\rangle = |+\rangle = \frac{1}{\sqrt{2}}(|0\rangle + |1\rangle)$

After applying H to all qubits on $|0\rangle^{\otimes n}$:
$$|+\rangle^{\otimes n} = \frac{1}{\sqrt{2^n}} \sum_{i_1, \ldots, i_n} |i_1 \ldots i_n\rangle$$

Still $\chi = 1$ (product state, no entanglement).

### 3.3 Rotation Gates

**Rz Gate:**
$$R_Z(\theta) = \begin{pmatrix} e^{-i\theta/2} & 0 \\ 0 & e^{i\theta/2} \end{pmatrix}$$

Applied element-wise to MPS tensors — no bond dimension change.

**Rx Gate:**
$$R_X(\theta) = \begin{pmatrix} \cos(\theta/2) & -i\sin(\theta/2) \\ -i\sin(\theta/2) & \cos(\theta/2) \end{pmatrix}$$

### 3.4 Two-Qubit Gates

Two-qubit gates between adjacent sites $(k, k+1)$ require:

1. **Contract** neighboring tensors:
$$\Theta_{i_k i_{k+1}} = \sum_\alpha A^{[k]}_{i_k, \alpha} A^{[k+1]}_{\alpha, i_{k+1}}$$

2. **Apply** the gate:
$$\tilde{\Theta}_{j_k j_{k+1}} = \sum_{i_k, i_{k+1}} U_{j_k j_{k+1}, i_k i_{k+1}} \Theta_{i_k i_{k+1}}$$

3. **Decompose** via SVD:
$$\tilde{\Theta} = U \Sigma V^\dagger$$

4. **Truncate** to maximum bond dimension $\chi_{\max}$

**Complexity:** $O(\chi^3)$ for SVD, potentially increases $\chi$.

### 3.5 XXPhase Gate Implementation

The $R_{XX}(\theta)$ gate:

$$R_{XX}(\theta) = e^{-i\theta X \otimes X / 2} = \begin{pmatrix} c & 0 & 0 & -is \\ 0 & c & -is & 0 \\ 0 & -is & c & 0 \\ -is & 0 & 0 & c \end{pmatrix}$$

where $c = \cos(\theta/2)$, $s = \sin(\theta/2)$.

**ITensor Implementation:**
```cpp
auto opx1 = op(site_inds, "X", i1+1);
auto opx2 = op(site_inds, "X", i2+1);
auto G = expHermitian(opx2 * opx1, -i*theta);
```

This uses the matrix exponential of the $X \otimes X$ operator.

### 3.6 Non-Adjacent Gates via SWAP

For gates between non-adjacent qubits $(i, j)$ with $|i - j| > 1$:

1. SWAP qubit $i$ toward qubit $j$ until adjacent
2. Apply the gate
3. SWAP back to original positions

**SWAP Gate:**
$$\text{SWAP} = \begin{pmatrix} 1 & 0 & 0 & 0 \\ 0 & 0 & 1 & 0 \\ 0 & 1 & 0 & 0 \\ 0 & 0 & 0 & 1 \end{pmatrix}$$

The pytket compiler's `DecomposeBRIDGE` pass handles this automatically.

---

## 4. ITensor Implementation

### 4.1 ITensor Library Overview

[ITensor](https://itensor.org/) is a C++ (and Julia) library for tensor network computations, developed at the Flatiron Institute.

**Key Features:**
- Automatic index contraction
- Efficient SVD and tensor decompositions
- MPS/MPO algorithms
- Quantum number conservation support

**Reference:** Fishman, M., White, S. R., & Stoudenmire, E. M. (2022). "The ITensor Software Library for Tensor Network Calculations." *SciPost Physics Codebases*, 4.

### 4.2 Custom Qubit Site Type

The implementation extends ITensor with a custom `Qubit` site type that provides:

**Standard Operators:**
- `"H"`: Hadamard gate
- `"X"`, `"Y"`, `"Z"`: Pauli operators
- `"Rx"`, `"Rz"`: Rotation gates with parameter $\alpha$
- `"T"`: T gate ($\pi/8$ phase)

**Measurement Operators:**
- `"X_half"`: Scaled X operator for expectation values
- `"Y_half"`: Scaled Y operator
- `"Z_half"`: Scaled Z operator

### 4.3 Circuit Simulation Flow

```cpp
MPS apply_gates3(vector<tuple<int,int,int,double>> circuits,
                 Qubit site_inds, int N, double cutoff) {

    // 1. Initialize MPS in |0⟩^⊗n state
    auto init = InitState(site_inds);
    for(auto n : range1(N)) {
        init.set(n, "Up");  // |0⟩ state
    }
    auto psi = MPS(init);

    // 2. Apply gates sequentially
    for (auto gate : circuits) {
        auto sym = get<0>(gate);    // Gate type
        auto i1 = get<1>(gate);     // First qubit
        auto i2 = get<2>(gate);     // Second qubit (-1 for single)
        auto a = get<3>(gate);      // Rotation angle

        // Apply gate based on type...
        // See Section 3 for gate implementations
    }

    return psi;
}
```

### 4.4 Gate Application Patterns

**Single-Qubit Gate Pattern:**
```cpp
// Position the orthogonality center at site i1+1
psi.position(i1+1);

// Create gate operator
auto G = op(site_inds, "Rz", i1+1, {"alpha=", a});

// Apply gate and remove prime indices
auto new_MPS = G * psi(i1+1);
new_MPS.noPrime();

// Update MPS
psi.set(i1+1, new_MPS);
```

**Two-Qubit Gate Pattern:**
```cpp
// Contract the two-site wavefunction
psi.position(i1+1);
auto wf = psi(i1+1) * psi(i2+1);

// Apply the gate
wf *= G;
wf.noPrime();

// SVD to restore MPS form
auto [U, S, V] = svd(wf, inds(psi(i1+1)), {"Cutoff=", 1E-10});

// Update MPS tensors
psi.set(i1+1, U);
psi.position(i2+1);
psi.set(i2+1, S*V);
```

### 4.5 Python-C++ Bridge

The ITensor C++ code is exposed to Python via **pybind11**:

```cpp
PYBIND11_MODULE(helloitensor, m) {
    m.def("circuit_xyz_exp",
          &circuit_xyz_exp<int,double>,
          "Extract X,Y,Z expectation values from circuit simulation");
}
```

**Usage in Python:**
```python
from helloitensor import circuit_xyz_exp

# circuit_gates: list of [gate_type, qubit1, qubit2, angle]
# n_qubits: number of qubits
exp_xyz = circuit_xyz_exp(circuit_gates, n_qubits)
# Returns: [[⟨X₁⟩, ⟨Y₁⟩, ⟨Z₁⟩], [⟨X₂⟩, ⟨Y₂⟩, ⟨Z₂⟩], ...]
```

---

## 5. Expectation Value Computation

### 5.1 Local Expectation Values in MPS

For an MPS in mixed-canonical form centered at site $k$:

$$\langle P_k \rangle = \text{Tr}(A^{[k]\dagger} P A^{[k]})$$

where the left and right environments cancel due to orthonormality.

### 5.2 ITensor Computation

```cpp
for (int i = 0; i < no_sites; i++) {
    psi.position(i+1);  // Move orthogonality center

    // Compute ⟨X⟩, ⟨Y⟩, ⟨Z⟩
    auto scalar_x = eltC(
        dag(prime(psi.A(i+1), "Site")) *
        site_inds.op("X_half", i+1) *
        psi.A(i+1)
    ).real();

    auto scalar_y = eltC(
        dag(prime(psi.A(i+1), "Site")) *
        site_inds.op("Y_half", i+1) *
        psi.A(i+1)
    ).real();

    auto scalar_z = eltC(
        dag(prime(psi.A(i+1), "Site")) *
        site_inds.op("Z_half", i+1) *
        psi.A(i+1)
    ).real();

    // Store [⟨X⟩, ⟨Y⟩, ⟨Z⟩] for qubit i
}
```

### 5.3 Diagrammatic Representation

```
        ⟨ψ|P|ψ⟩ at site k:

Left Environment    Site k      Right Environment
      (= I)           |              (= I)
                     [P]
                      |
     ───[A*]────────[A*]────────[A*]───
                      |
     ───[A ]────────[A ]────────[A ]───
                      |
                   Result
```

Due to canonical form, left/right contract to identity.

---

## 6. Computational Complexity

### 6.1 MPS Operations Complexity

| Operation | Complexity | Notes |
|-----------|------------|-------|
| Initialize $\|0\rangle^{\otimes n}$ | $O(n)$ | $\chi = 1$ |
| Single-qubit gate | $O(\chi^2)$ | No $\chi$ increase |
| Two-qubit gate (adjacent) | $O(\chi^3)$ | SVD step, may increase $\chi$ |
| Two-qubit gate (distance $d$) | $O(d \cdot \chi^3)$ | Requires $2d$ SWAPs |
| Expectation value | $O(\chi^2)$ | Per site |

### 6.2 Full Circuit Simulation

For a circuit with:
- $n$ qubits
- $r$ repetitions
- Linear entanglement map ($n-1$ two-qubit gates per layer)

**Total Complexity:**
$$T = O(n \cdot r \cdot \chi^3)$$

where $\chi$ is the maximum bond dimension encountered.

### 6.3 Bond Dimension Growth

**Theoretical Maximum:** After applying an entangling gate, bond dimension can double.

**Practical Behavior:** For typical feature map circuits:
- Initial: $\chi = 1$
- After Hadamards: $\chi = 1$ (still product state)
- After first entangling layer: $\chi \leq 2^{\min(k, n-k)}$
- With truncation (cutoff $\epsilon$): $\chi \leq \chi_{\max}$

### 6.4 Truncation and Approximation

The SVD truncation with cutoff $\epsilon$:
1. Compute full SVD: $\Theta = U \Sigma V^\dagger$
2. Keep singular values $\sigma_i > \epsilon \cdot \sigma_1$
3. Truncated state has error $\|\tilde{\psi} - \psi\| \leq \sqrt{\sum_{i > \chi'} \sigma_i^2}$

**Typical Cutoff:** `1E-10` to `1E-16` for high precision.

---

## 7. Limitations and Trade-offs

### 7.1 When MPS Fails

MPS becomes inefficient for:

1. **High Entanglement:** Volume-law entangled states require $\chi \sim 2^{n/2}$
2. **Non-Local Connectivity:** Random or all-to-all connectivity creates long-range entanglement
3. **Deep Circuits:** Entanglement accumulates with depth

### 7.2 Entanglement in Feature Maps

**Good News for QiML:** Feature map circuits typically have:
- Shallow depth (few repetitions)
- Local connectivity
- Bounded entanglement growth

**Empirical Observation:** For the Hamiltonian ansatz with `reps ≤ 10` and linear connectivity, $\chi$ rarely exceeds 20-50.

### 7.3 Comparison with Other Methods

| Method | Max Qubits | Entanglement | Best For |
|--------|------------|--------------|----------|
| State Vector | ~40 | Any | Short circuits, full fidelity |
| MPS | 100+ | Low-moderate | 1D circuits, local gates |
| PEPS | ~20 | 2D area-law | 2D systems |
| Clifford Simulation | Any | Stabilizer states | Clifford-only circuits |

### 7.4 Error Sources

1. **Truncation Error:** Controlled by SVD cutoff
2. **Floating Point:** IEEE-754 double precision
3. **Algorithmic:** None for exact MPS (no sampling)

### 7.5 Scalability Guidelines

| Qubits | Recommended $\chi_{\max}$ | Memory (per state) |
|--------|--------------------------|-------------------|
| 10-20 | 50 | ~1 MB |
| 20-50 | 100 | ~10 MB |
| 50-100 | 200 | ~100 MB |
| 100+ | 500 | ~1 GB |

---

## Summary: MPS Simulation Pipeline

```
Algorithm: MPS-Based Circuit Simulation

Input: Gate list [(type, q1, q2, angle), ...], n_qubits

1. INITIALIZATION
   Create MPS |ψ⟩ = |0⟩^⊗n with χ = 1

2. GATE APPLICATION
   For each gate (type, q1, q2, angle):
     - Position orthogonality center at q1
     - If single-qubit: Apply and update tensor
     - If two-qubit:
       a. Contract tensors at q1, q2
       b. Apply gate matrix
       c. SVD with truncation
       d. Update MPS tensors

3. MEASUREMENT
   For each qubit i:
     - Position at site i
     - Compute ⟨X_i⟩, ⟨Y_i⟩, ⟨Z_i⟩

Output: Expectation values [[⟨X₁⟩,⟨Y₁⟩,⟨Z₁⟩], ..., [⟨Xₙ⟩,⟨Yₙ⟩,⟨Zₙ⟩]]
```

---

*Next: [04_QMLOPS_PIPELINE.md](04_QMLOPS_PIPELINE.md) - End-to-end QMLOps pipeline documentation*
