# Theoretical Foundations of Quantum-Inspired Machine Learning

This document provides the mathematical and theoretical foundations underlying the Quantum-Inspired Machine Learning (QiML) pipeline for fraud detection.

---

## Table of Contents

1. [Introduction to Quantum Computing for ML](#1-introduction-to-quantum-computing-for-ml)
2. [Quantum State Representation](#2-quantum-state-representation)
3. [Quantum Feature Maps](#3-quantum-feature-maps)
4. [Kernel Methods in Machine Learning](#4-kernel-methods-in-machine-learning)
5. [Quantum Kernels](#5-quantum-kernels)
6. [References](#6-references)

---

## 1. Introduction to Quantum Computing for ML

### 1.1 The Quantum Advantage Hypothesis

Quantum machine learning (QML) seeks to leverage quantum mechanical phenomena—superposition, entanglement, and interference—to potentially achieve computational advantages over classical machine learning methods. The key insight is that quantum systems can represent and manipulate exponentially large state spaces efficiently.

**Definition (Quantum Advantage in ML):** A quantum algorithm provides an advantage for a learning task if it achieves comparable or better generalization performance with provably fewer resources (time, samples, or queries) than any classical algorithm.

### 1.2 Why Quantum Kernels?

Classical kernel methods map data into high-dimensional feature spaces where linear separation becomes possible. Quantum kernels extend this concept by:

1. **Exponential Feature Space**: An n-qubit system spans a 2^n-dimensional Hilbert space
2. **Efficient Inner Products**: Quantum circuits can compute inner products in this space efficiently
3. **Non-trivial Correlations**: Entanglement creates feature correlations inaccessible to classical kernels

### 1.3 The NISQ Constraint

Current quantum hardware operates in the Noisy Intermediate-Scale Quantum (NISQ) regime, characterized by:
- Limited qubit counts (tens to hundreds)
- High error rates requiring error mitigation
- Limited circuit depth before decoherence

This motivates **tensor network simulation** as a scalable alternative that captures quantum correlations while running on classical hardware.

---

## 2. Quantum State Representation

### 2.1 Qubit States

A single qubit exists in a superposition of computational basis states:

$$|\psi\rangle = \alpha|0\rangle + \beta|1\rangle$$

where $\alpha, \beta \in \mathbb{C}$ and $|\alpha|^2 + |\beta|^2 = 1$.

**Bloch Sphere Representation:**
Any pure single-qubit state can be written as:

$$|\psi\rangle = \cos\frac{\theta}{2}|0\rangle + e^{i\phi}\sin\frac{\theta}{2}|1\rangle$$

where $\theta \in [0, \pi]$ and $\phi \in [0, 2\pi)$ are angles on the Bloch sphere.

### 2.2 Multi-Qubit Systems

For n qubits, the state space is the tensor product:

$$\mathcal{H} = \mathcal{H}_1 \otimes \mathcal{H}_2 \otimes \cdots \otimes \mathcal{H}_n = (\mathbb{C}^2)^{\otimes n}$$

A general n-qubit state is:

$$|\psi\rangle = \sum_{i_1, i_2, \ldots, i_n \in \{0,1\}} c_{i_1 i_2 \ldots i_n} |i_1 i_2 \ldots i_n\rangle$$

with $\sum |c_{i_1 \ldots i_n}|^2 = 1$, requiring $2^n$ complex amplitudes.

### 2.3 Density Matrix Formalism

For mixed states (statistical ensembles), we use the density matrix:

$$\rho = \sum_i p_i |\psi_i\rangle\langle\psi_i|$$

where $p_i$ are classical probabilities. For pure states, $\rho = |\psi\rangle\langle\psi|$ with $\text{Tr}(\rho^2) = 1$.

### 2.4 Reduced Density Matrices

For a bipartite system $AB$, the reduced density matrix of subsystem $A$ is:

$$\rho_A = \text{Tr}_B(\rho_{AB})$$

This captures the local observable statistics on qubit $A$, crucial for projected quantum kernels.

---

## 3. Quantum Feature Maps

### 3.1 Definition

A quantum feature map is a parameterized unitary operation that encodes classical data into a quantum state:

**Definition (Quantum Feature Map):** A feature map $\phi: \mathcal{X} \rightarrow \mathcal{H}$ is defined by a unitary circuit $U(x)$ such that:

$$\phi(x) = U(x)|0\rangle^{\otimes n}$$

where $x \in \mathcal{X} \subseteq \mathbb{R}^d$ is the classical input data.

### 3.2 The ZZ Feature Map (Havlíček et al.)

The foundational quantum feature map proposed in [Havlíček et al., 2019] consists of:

**Circuit Structure:**
$$U_{\phi}(x) = \left[\prod_{k=1}^{d} H^{\otimes n} \cdot U_Z(x) \cdot U_{ZZ}(x)\right]^r$$

where:
- $H^{\otimes n}$ applies Hadamard gates to all qubits
- $U_Z(x) = \exp\left(i \sum_j x_j Z_j\right)$ encodes single features
- $U_{ZZ}(x) = \exp\left(i \sum_{j<k} (\pi - x_j)(\pi - x_k) Z_j Z_k\right)$ encodes feature interactions
- $r$ is the number of repetitions (circuit depth)

**Mathematical Form:**

$$U_Z(x) = \bigotimes_{j=1}^n R_Z(x_j) = \bigotimes_{j=1}^n \begin{pmatrix} e^{-i x_j/2} & 0 \\ 0 & e^{i x_j/2} \end{pmatrix}$$

### 3.3 The Hamiltonian Ansatz (This Implementation)

This project implements a **Hamiltonian-inspired feature map** with the structure:

**Layer Structure (repeated $r$ times):**

1. **Initialization (once):** $H^{\otimes n}$ - Hadamard on all qubits
2. **Single-qubit encoding:** $R_Z\left(\frac{\gamma \cdot x_i}{\pi}\right)$ on qubit $i$
3. **Two-qubit entanglement:** $R_{XX}\left(\gamma^2 (1-x_i)(1-x_j)\right)$ on connected pairs $(i,j)$

**Mathematical Formulation:**

$$U(x) = H^{\otimes n} \prod_{\ell=1}^{r} \left[ \prod_{i=1}^{n} R_Z\left(\frac{\gamma x_i}{\pi}\right) \prod_{(i,j) \in E} R_{XX}\left(\gamma^2(1-x_i)(1-x_j)\right) \right]$$

where:
- $\gamma \in (0, 1]$ is the rotation scaling parameter (bandwidth)
- $E$ is the entanglement graph (connectivity)
- $R_Z(\theta) = e^{-i\theta Z/2}$
- $R_{XX}(\theta) = e^{-i\theta X \otimes X/2}$

**Gate Definitions:**

$$R_Z(\theta) = \begin{pmatrix} e^{-i\theta/2} & 0 \\ 0 & e^{i\theta/2} \end{pmatrix}$$

$$R_{XX}(\theta) = \begin{pmatrix} \cos(\theta/2) & 0 & 0 & -i\sin(\theta/2) \\ 0 & \cos(\theta/2) & -i\sin(\theta/2) & 0 \\ 0 & -i\sin(\theta/2) & \cos(\theta/2) & 0 \\ -i\sin(\theta/2) & 0 & 0 & \cos(\theta/2) \end{pmatrix}$$

### 3.4 The Magic Ansatz (Alternative)

An alternative ansatz using Clifford+T gates for non-Gaussian features:

**Layer Structure (repeated $r$ times):**
1. Hadamard $H$ on all qubits
2. T gate ($\pi/8$ rotation) on all qubits
3. CZ gates on connected pairs
4. $R_Z(x_i)$ encoding on qubit $i$

This creates "magic states" that exhibit quantum contextuality, potentially useful for specific data distributions.

### 3.5 Entanglement Topology

The **entanglement map** $E$ defines which qubit pairs interact. This implementation uses a **linear nearest-neighbor** topology:

$$E = \{(i, i+d) : i \in [n-d], d \in [1, \text{nn}]\}$$

where $\text{nn}$ is the neighborhood depth. For $\text{nn}=1$:
$$E = \{(0,1), (1,2), (2,3), \ldots, (n-2, n-1)\}$$

**Why Linear Topology?**
- Matches typical quantum hardware constraints
- Sufficient for capturing local correlations
- Enables efficient MPS simulation (see [03_TENSOR_NETWORK_SIMULATION.md])

---

## 4. Kernel Methods in Machine Learning

### 4.1 The Kernel Trick

Kernel methods enable learning in high-dimensional feature spaces without explicitly computing the feature vectors.

**Definition (Kernel Function):** A function $k: \mathcal{X} \times \mathcal{X} \rightarrow \mathbb{R}$ is a valid kernel if it corresponds to an inner product in some Hilbert space $\mathcal{F}$:

$$k(x, x') = \langle \phi(x), \phi(x') \rangle_{\mathcal{F}}$$

for some feature map $\phi: \mathcal{X} \rightarrow \mathcal{F}$.

### 4.2 Mercer's Theorem

**Theorem (Mercer):** A symmetric function $k(x, x')$ is a valid kernel if and only if it is positive semi-definite:

$$\sum_{i,j} c_i c_j k(x_i, x_j) \geq 0$$

for all finite sets $\{x_i\}$ and coefficients $\{c_i\} \subset \mathbb{R}$.

### 4.3 Support Vector Machines

For binary classification with labels $y_i \in \{-1, +1\}$, the SVM optimization problem is:

**Primal Form:**
$$\min_{w, b} \frac{1}{2}\|w\|^2 + C \sum_i \xi_i$$
$$\text{s.t. } y_i(w^T \phi(x_i) + b) \geq 1 - \xi_i, \quad \xi_i \geq 0$$

**Dual Form (Kernel Form):**
$$\max_\alpha \sum_i \alpha_i - \frac{1}{2}\sum_{i,j} \alpha_i \alpha_j y_i y_j k(x_i, x_j)$$
$$\text{s.t. } 0 \leq \alpha_i \leq C, \quad \sum_i \alpha_i y_i = 0$$

The decision function uses only the kernel:
$$f(x) = \text{sign}\left(\sum_i \alpha_i y_i k(x_i, x) + b\right)$$

### 4.4 The RBF (Gaussian) Kernel

The Radial Basis Function kernel is defined as:

$$k_{\text{RBF}}(x, x') = \exp\left(-\frac{\|x - x'\|^2}{2\sigma^2}\right)$$

This corresponds to an **infinite-dimensional** feature space. The parameter $\sigma$ (or equivalently $\gamma = 1/(2\sigma^2)$) controls the kernel bandwidth.

---

## 5. Quantum Kernels

### 5.1 Fidelity Quantum Kernel (FQK)

The standard quantum kernel computes the overlap between encoded quantum states:

$$k_{\text{FQK}}(x, x') = |\langle \phi(x') | \phi(x) \rangle|^2 = |\langle 0^n | U^\dagger(x') U(x) | 0^n \rangle|^2$$

**Properties:**
- $k_{\text{FQK}}(x, x) = 1$ (normalization)
- $k_{\text{FQK}}(x, x') = k_{\text{FQK}}(x', x)$ (symmetry)
- Exponentially small for distant points (can cause numerical issues)

### 5.2 Projected Quantum Kernel (PQK)

To address FQK limitations, [Huang et al., 2021] introduced the **projected quantum kernel**:

**Definition:** The PQK is computed from reduced density matrices:

$$k_{\text{PQK}}(x, x') = \exp\left(-\alpha \sum_{i=1}^{n} \|\rho_i(x) - \rho_i(x')\|_F^2\right)$$

where:
- $\rho_i(x) = \text{Tr}_{\bar{i}}(|\phi(x)\rangle\langle\phi(x)|)$ is the single-qubit reduced density matrix
- $\|\cdot\|_F$ is the Frobenius norm
- $\alpha > 0$ is the bandwidth parameter

**Expanded Form:**

For a single-qubit density matrix:
$$\rho_i = \frac{1}{2}(I + \langle X \rangle_i X + \langle Y \rangle_i Y + \langle Z \rangle_i Z)$$

The Frobenius distance becomes:
$$\|\rho_i(x) - \rho_i(x')\|_F^2 = \frac{1}{2}\left[(\langle X \rangle_i^x - \langle X \rangle_i^{x'})^2 + (\langle Y \rangle_i^x - \langle Y \rangle_i^{x'})^2 + (\langle Z \rangle_i^x - \langle Z \rangle_i^{x'})^2\right]$$

### 5.3 This Implementation's Kernel Formula

The implementation computes:

$$k(x, x') = \exp\left(-\alpha \sum_{i=1}^{n} 2\left[(\langle X \rangle_i^x - \langle X \rangle_i^{x'})^2 + (\langle Y \rangle_i^x - \langle Y \rangle_i^{x'})^2 + (\langle Z \rangle_i^x - \langle Z \rangle_i^{x'})^2\right]\right)$$

where $\langle P \rangle_i^x = \langle \phi(x) | P_i | \phi(x) \rangle$ for $P \in \{X, Y, Z\}$.

**Key Insight:** This is equivalent to the RBF kernel in a $3n$-dimensional classical feature space spanned by the Pauli expectation values.

### 5.4 Advantages of PQK over FQK

| Property | FQK | PQK |
|----------|-----|-----|
| Dimensionality | $2^n$ (full Hilbert space) | $3n$ (local observables) |
| Numerical stability | Exponentially small values | Well-conditioned |
| Classical simulability | Requires full state | Only local expectation values |
| Expressiveness | Global correlations | Local + entanglement-mediated |

### 5.5 Connection to Classical Kernels

The PQK can be viewed as an RBF kernel in a **quantum-derived feature space**:

$$\Phi(x) = \left(\langle X \rangle_1, \langle Y \rangle_1, \langle Z \rangle_1, \ldots, \langle X \rangle_n, \langle Y \rangle_n, \langle Z \rangle_n\right) \in \mathbb{R}^{3n}$$

$$k_{\text{PQK}}(x, x') = \exp\left(-\alpha \|\Phi(x) - \Phi(x')\|^2\right)$$

The quantum circuit acts as a **nonlinear feature extractor** that:
1. Encodes data into quantum states
2. Creates entanglement-mediated correlations
3. Projects back to classical observables

---

## 6. References

### Foundational Papers

1. **Havlíček, V. et al.** (2019). "Supervised learning with quantum-enhanced feature spaces." *Nature*, 567, 209-212. [DOI: 10.1038/s41586-019-0980-2](https://www.nature.com/articles/s41586-019-0980-2)

2. **Huang, H.-Y. et al.** (2021). "Power of data in quantum machine learning." *Nature Communications*, 12, 2631. [DOI: 10.1038/s41467-021-22539-9](https://www.nature.com/articles/s41467-021-22539-9)

3. **Kübler, J. M., Buchholz, S., & Schölkopf, B.** (2021). "The Inductive Bias of Quantum Kernels." *NeurIPS 2021*. [Paper](https://proceedings.neurips.cc/paper/2021/file/69adc1e107f7f7d035d7baf04342e1ca-Paper.pdf)

### Quantum Kernels for Finance

4. **Heredge, J. et al.** (2023). "Quantum Multiple Kernel Learning in Financial Classification Tasks." *arXiv:2312.00260*. [arXiv](https://arxiv.org/abs/2312.00260)

5. **Vasquez, A. C. et al.** (2023). "Financial Fraud Detection: A Comparative Study of Quantum Machine Learning Models." *arXiv:2308.05237*. [arXiv](https://arxiv.org/abs/2308.05237)

### Tensor Networks

6. **Fishman, M., White, S. R., & Stoudenmire, E. M.** (2022). "The ITensor Software Library for Tensor Network Calculations." *SciPost Physics Codebases*, 4. [Paper](https://scipost.org/SciPostPhysCodeb.4)

### Additional Resources

7. **IBM Qiskit Documentation** - ZZFeatureMap. [Link](https://docs.quantum.ibm.com/api/qiskit/qiskit.circuit.library.ZZFeatureMap)

8. **PennyLane** - Kernel-based training of quantum models. [Tutorial](https://pennylane.ai/qml/demos/tutorial_kernel_based_training)

---

*Next: [02_PROJECTED_QUANTUM_KERNELS.md](02_PROJECTED_QUANTUM_KERNELS.md) - Mathematical framework for Projected Quantum Kernels*
