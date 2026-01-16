<div>
<h1>Quantum-Inspired Machine Learning (QiML) Platform</h1>
<h2>Projected Quantum Kernels for Financial Fraud Detection - Complete Technical Documentation</h2>
<ac:structured-macro ac:name="toc">
  <ac:parameter ac:name="printable">true</ac:parameter>
  <ac:parameter ac:name="style">square</ac:parameter>
  <ac:parameter ac:name="maxLevel">3</ac:parameter>
  <ac:parameter ac:name="indent">20px</ac:parameter>
  <ac:parameter ac:name="minLevel">1</ac:parameter>
  <ac:parameter ac:name="class">bigpink</ac:parameter>
  <ac:parameter ac:name="exclude">[1]</ac:parameter>
</ac:structured-macro>
<hr />
<h1>1. Executive Summary</h1>
<h2>1.1 What is This Project?</h2>
<p>This platform implements <strong>Quantum-Inspired Machine Learning (QiML)</strong> for binary classification, specifically designed for <strong>financial fraud detection</strong>.</p>
<p><strong>For ML practitioners:</strong> Think of this as a sophisticated feature engineering pipeline that:</p>
<ol>
  <li>Takes your classical tabular data (like transaction features)</li>
  <li>Passes it through a &quot;quantum circuit&quot; simulation</li>
  <li>Extracts new features that capture complex non-linear relationships</li>
  <li>Uses these features in a kernel-based SVM classifier</li>
</ol>
<p><strong>The key insight:</strong> We&#x27;re not running on actual quantum hardware. Instead, we simulate quantum circuits on classical computers using efficient tensor network methods. This gives us the mathematical benefits of quantum feature spaces without needing quantum hardware.</p>
<h2>1.2 Why Quantum-Inspired ML?</h2>
<ac:structured-macro ac:name="panel">
  <ac:parameter ac:name="title">The Core Problem</ac:parameter>
  <ac:parameter ac:name="borderStyle">solid</ac:parameter>
  <ac:parameter ac:name="borderColor">#ccc</ac:parameter>
  <ac:parameter ac:name="titleBGColor">#f0f0f0</ac:parameter>
  <ac:rich-text-body>
<p>In fraud detection, we often have:</p>
<ul>
  <li>High-dimensional data with complex feature interactions</li>
  <li>Non-linear decision boundaries that simple models miss</li>
  <li>Need for robust models that don&#x27;t overfit</li>
</ul>
<p>Traditional approaches like RBF kernels work well, but quantum-inspired kernels can capture <strong>different types of non-linear relationships</strong> through quantum mechanical principles.</p>
  </ac:rich-text-body>
</ac:structured-macro>
<h2>1.3 Key Results</h2>
<table>
  <tbody>
    <tr>
      <th>Metric</th>
      <th>PQK-SVM (Our Method)</th>
      <th>RBF-SVM (Baseline)</th>
      <th>Improvement</th>
    </tr>
    <tr>
      <td>Accuracy</td>
      <td>85-92%</td>
      <td>82-88%</td>
      <td>+3-4%</td>
    </tr>
    <tr>
      <td>Recall (Fraud Detection)</td>
      <td>82-90%</td>
      <td>78-86%</td>
      <td>+4-5%</td>
    </tr>
    <tr>
      <td>F1 Score</td>
      <td>81-89%</td>
      <td>77-85%</td>
      <td>+4-5%</td>
    </tr>
    <tr>
      <td>AUC-ROC</td>
      <td>88-94%</td>
      <td>84-90%</td>
      <td>+4%</td>
    </tr>
  </tbody>
</table>
<hr />
<h1>2. Understanding Quantum Computing Basics (For ML Practitioners)</h1>
<ac:structured-macro ac:name="note">
  <ac:parameter ac:name="title">Why This Section Matters</ac:parameter>
  <ac:rich-text-body>
<p>Before diving into the implementation, you need to understand the quantum concepts we&#x27;re borrowing. Don&#x27;t worry - we&#x27;ll explain everything using analogies to concepts you already know from ML.</p>
  </ac:rich-text-body>
</ac:structured-macro>
<h2>2.1 What is a Qubit? (The Quantum Version of a Bit)</h2>
<h3>2.1.1 Classical Bits vs Qubits</h3>
<p><strong>Classical bit:</strong> Can be either 0 or 1. That&#x27;s it.</p>
<p><strong>Qubit:</strong> Can be in a &quot;superposition&quot; - a combination of 0 and 1 simultaneously.</p>
<ac:structured-macro ac:name="code">
  <ac:parameter ac:name="title">Mathematical Representation</ac:parameter>
  <ac:plain-text-body><![CDATA[
Classical bit: b ∈ {0, 1}

Qubit state: |ψ⟩ = α|0⟩ + β|1⟩

where:
- |0⟩ and |1⟩ are the "basis states" (like unit vectors)
- α and β are complex numbers
- |α|² + |β|² = 1 (normalization constraint)
- |α|² = probability of measuring 0
- |β|² = probability of measuring 1
]]></ac:plain-text-body>
</ac:structured-macro>
<p><strong>ML Analogy:</strong> Think of a qubit like a probability distribution over {0, 1}, but with complex-valued &quot;amplitudes&quot; instead of just probabilities. The complex numbers allow for interference effects that create interesting mathematical properties.</p>
<h3>2.1.2 The Bloch Sphere (Visualizing a Qubit)</h3>
<p>Every single-qubit state can be represented as a point on a sphere called the &quot;Bloch sphere&quot;:</p>
<ac:structured-macro ac:name="code">
  <ac:parameter ac:name="title">Bloch Sphere Parameterization</ac:parameter>
  <ac:plain-text-body><![CDATA[
|ψ⟩ = cos(θ/2)|0⟩ + e^(iφ)·sin(θ/2)|1⟩

where:
- θ ∈ [0, π] is the polar angle (how much "1" vs "0")
- φ ∈ [0, 2π) is the azimuthal angle (the "phase")

Key points on the sphere:
- North pole (θ=0): |0⟩ state
- South pole (θ=π): |1⟩ state
- Equator (θ=π/2): Equal superposition of |0⟩ and |1⟩
]]></ac:plain-text-body>
</ac:structured-macro>
<ac:structured-macro ac:name="panel">
  <ac:parameter ac:name="title">Why This Matters for ML</ac:parameter>
  <ac:parameter ac:name="borderStyle">solid</ac:parameter>
  <ac:rich-text-body>
<p>The Bloch sphere gives us a 3D representation of a qubit. When we measure a qubit, we get three numbers: the X, Y, and Z coordinates on this sphere. These become our quantum features!</p>
<p><strong>This is the core of our approach:</strong> We encode data into qubit states, then extract the (X, Y, Z) coordinates as features.</p>
  </ac:rich-text-body>
</ac:structured-macro>
<h2>2.2 Multiple Qubits and the Exponential State Space</h2>
<h3>2.2.1 The Exponential Scaling</h3>
<p>When we have multiple qubits, the state space grows exponentially:</p>
<table>
  <tbody>
    <tr>
      <th>Number of Qubits</th>
      <th>State Space Dimension</th>
      <th>Equivalent to...</th>
    </tr>
    <tr>
      <td>1</td>
      <td>2</td>
      <td>2 probabilities</td>
    </tr>
    <tr>
      <td>2</td>
      <td>4</td>
      <td>4 probabilities</td>
    </tr>
    <tr>
      <td>10</td>
      <td>1,024</td>
      <td>Small neural network layer</td>
    </tr>
    <tr>
      <td>20</td>
      <td>1,048,576</td>
      <td>~1 million parameters</td>
    </tr>
    <tr>
      <td>30</td>
      <td>1,073,741,824</td>
      <td>~1 billion parameters</td>
    </tr>
    <tr>
      <td>50</td>
      <td>1,125,899,906,842,624</td>
      <td>More than all atoms on Earth!</td>
    </tr>
  </tbody>
</table>
<ac:structured-macro ac:name="code">
  <ac:parameter ac:name="title">Mathematical Representation of n-Qubit State</ac:parameter>
  <ac:plain-text-body><![CDATA[
For n qubits, a general state is:

|ψ⟩ = Σ c_{i₁i₂...iₙ} |i₁i₂...iₙ⟩

where:
- Each iₖ ∈ {0, 1}
- There are 2ⁿ coefficients c_{i₁i₂...iₙ}
- All |c|² must sum to 1
]]></ac:plain-text-body>
</ac:structured-macro>
<p><strong>ML Analogy:</strong> This is like having a probability distribution over 2ⁿ possible outcomes. The exponential scaling is both the promise and the challenge of quantum computing.</p>
<h3>2.2.2 Entanglement (Quantum Correlations)</h3>
<p><strong>Entanglement</strong> is a quantum phenomenon where qubits become correlated in ways that cannot be described by classical probability.</p>
<ac:structured-macro ac:name="code">
  <ac:parameter ac:name="title">Example: Bell State (Maximally Entangled)</ac:parameter>
  <ac:plain-text-body><![CDATA[
|Φ⁺⟩ = (1/√2)(|00⟩ + |11⟩)

This state means:
- If you measure qubit 1 and get 0, qubit 2 is DEFINITELY 0
- If you measure qubit 1 and get 1, qubit 2 is DEFINITELY 1
- But before measurement, both possibilities exist simultaneously

This correlation is STRONGER than any classical correlation can be.
]]></ac:plain-text-body>
</ac:structured-macro>
<p><strong>Why This Matters for ML:</strong> Entanglement allows our quantum circuits to create complex feature interactions. When we entangle qubits encoding different data features, we create correlations that capture feature interactions in ways classical methods cannot easily replicate.</p>
<h2>2.3 Quantum Gates (Transformations on Qubits)</h2>
<p>Just like neural networks apply transformations (weights, activations) to data, quantum circuits apply &quot;gates&quot; to qubits.</p>
<h3>2.3.1 Single-Qubit Gates</h3>
<ac:structured-macro ac:name="code">
  <ac:parameter ac:name="title">Common Single-Qubit Gates</ac:parameter>
  <ac:plain-text-body><![CDATA[
HADAMARD GATE (H):
- Creates superposition from basis state
- H|0⟩ = (|0⟩ + |1⟩)/√2 = |+⟩
- H|1⟩ = (|0⟩ - |1⟩)/√2 = |-⟩

Matrix form:
H = (1/√2) * | 1   1 |
             | 1  -1 |

ML Analogy: Like a fixed linear transformation that spreads information.

---

ROTATION GATES (Rx, Ry, Rz):
- Rotate the qubit state around X, Y, or Z axis of Bloch sphere
- Take a parameter θ (the rotation angle)

Rz(θ) = | e^(-iθ/2)    0      |
        |    0      e^(iθ/2)  |

ML Analogy: These are our PARAMETERIZED transformations - like weights in a neural network. We encode our data into these rotation angles!

---

T GATE:
- Fixed π/8 rotation around Z axis
- Creates "magic states" useful for certain computations

T = | 1    0        |
    | 0   e^(iπ/4)  |
]]></ac:plain-text-body>
</ac:structured-macro>
<h3>2.3.2 Two-Qubit Gates (Creating Entanglement)</h3>
<ac:structured-macro ac:name="code">
  <ac:parameter ac:name="title">Common Two-Qubit Gates</ac:parameter>
  <ac:plain-text-body><![CDATA[
CNOT (Controlled-NOT):
- Flips second qubit IF first qubit is |1⟩
- Creates entanglement when applied to superposition

CNOT|00⟩ = |00⟩
CNOT|01⟩ = |01⟩
CNOT|10⟩ = |11⟩  (flipped!)
CNOT|11⟩ = |10⟩  (flipped!)

---

XX-PHASE GATE (R_XX):
- Rotates jointly around X⊗X direction
- Used in our Hamiltonian ansatz
- Creates entanglement based on rotation angle

R_XX(θ) = | cos(θ/2)    0         0      -i·sin(θ/2) |
          |    0     cos(θ/2) -i·sin(θ/2)     0      |
          |    0    -i·sin(θ/2) cos(θ/2)      0      |
          |-i·sin(θ/2)   0         0       cos(θ/2)  |

ML Analogy: This is how we encode FEATURE INTERACTIONS. The rotation angle depends on TWO features, creating non-linear coupling.

---

CZ (Controlled-Z):
- Applies Z gate to second qubit IF first qubit is |1⟩
- Used in Magic ansatz
]]></ac:plain-text-body>
</ac:structured-macro>
<h2>2.4 Quantum Circuits (Putting It All Together)</h2>
<p>A quantum circuit is a sequence of gates applied to qubits - like a computational graph in deep learning.</p>
<ac:structured-macro ac:name="code">
  <ac:parameter ac:name="title">Circuit Structure</ac:parameter>
  <ac:plain-text-body><![CDATA[
QUANTUM CIRCUIT ANATOMY:

Initial State     Gates (Layers)           Measurement
    |                  |                       |
    v                  v                       v

|0⟩ ─────[H]─────[Rz(x₁)]─────[XX]─────────── ⟨X⟩, ⟨Y⟩, ⟨Z⟩
                               |
|0⟩ ─────[H]─────[Rz(x₂)]─────[XX]─────────── ⟨X⟩, ⟨Y⟩, ⟨Z⟩
                               |
|0⟩ ─────[H]─────[Rz(x₃)]─────[XX]─────────── ⟨X⟩, ⟨Y⟩, ⟨Z⟩

Where:
- |0⟩: Initial qubit states (all zeros)
- [H]: Hadamard gates (create superposition)
- [Rz(xᵢ)]: Rotation gates encoding feature xᵢ
- [XX]: Entangling gates between adjacent qubits
- ⟨X⟩, ⟨Y⟩, ⟨Z⟩: Measured expectation values (our output features)
]]></ac:plain-text-body>
</ac:structured-macro>
<p><strong>ML Analogy:</strong> Think of a quantum circuit like a neural network:</p>
<ul>
  <li>Initial state = input layer</li>
  <li>Gates = layers with transformations</li>
  <li>Measurements = output layer</li>
  <li>But unlike neural networks, we don&#x27;t train the circuit - we fix it and use it as a feature extractor</li>
</ul>
<hr />
<h1>3. The Projected Quantum Kernel Method</h1>
<p>Now that you understand the quantum basics, let&#x27;s see how we use them for machine learning.</p>
<h2>3.1 The Big Picture: Quantum Feature Maps</h2>
<h3>3.1.1 What is a Feature Map?</h3>
<p>In kernel methods, a <strong>feature map</strong> transforms input data into a (usually higher-dimensional) feature space:</p>
<ac:structured-macro ac:name="code">
  <ac:parameter ac:name="title">Classical vs Quantum Feature Maps</ac:parameter>
  <ac:plain-text-body><![CDATA[
CLASSICAL RBF FEATURE MAP:
- Maps x ∈ ℝᵈ to infinite-dimensional space
- φ(x) = exp(-||x||²/2) * [1, x₁, x₂, x₁², x₁x₂, ...]
- The RBF kernel computes inner products in this space

QUANTUM FEATURE MAP:
- Maps x ∈ ℝᵈ to 2ⁿ-dimensional Hilbert space
- φ(x) = U(x)|0⟩ⁿ (apply circuit to initial state)
- Result is a quantum state encoding the data
]]></ac:plain-text-body>
</ac:structured-macro>
<h3>3.1.2 The Problem with Full Quantum Kernels</h3>
<p>The standard quantum kernel (Fidelity Quantum Kernel) computes:</p>
<ac:structured-macro ac:name="code">
  <ac:parameter ac:name="title">Fidelity Quantum Kernel</ac:parameter>
  <ac:plain-text-body><![CDATA[
k_FQK(x, x') = |⟨φ(x')|φ(x)⟩|²

This measures the overlap between two quantum states.

PROBLEMS:
1. Values become exponentially small for different inputs
2. Requires access to full quantum state (2ⁿ complex numbers)
3. Numerically unstable for practical use
]]></ac:plain-text-body>
</ac:structured-macro>
<h3>3.1.3 The Solution: Projected Quantum Kernel</h3>
<p>Instead of computing full state overlap, we: 1. <strong>Project</strong> the quantum state to local observables (X, Y, Z on each qubit) 2. Compare these <strong>classical</strong> 3n-dimensional vectors</p>
<ac:structured-macro ac:name="code">
  <ac:parameter ac:name="title">Projected Quantum Kernel Concept</ac:parameter>
  <ac:plain-text-body><![CDATA[
WORKFLOW:

Classical Data x ∈ ℝᵈ
       │
       ▼
Quantum Circuit U(x)
       │
       ▼
Quantum State |ψ(x)⟩ (2ⁿ-dimensional, but we don't store it!)
       │
       ▼
Local Measurements: ⟨X₁⟩, ⟨Y₁⟩, ⟨Z₁⟩, ..., ⟨Xₙ⟩, ⟨Yₙ⟩, ⟨Zₙ⟩
       │
       ▼
Classical Feature Vector Φ(x) ∈ ℝ³ⁿ
       │
       ▼
RBF Kernel: k(x, x') = exp(-α||Φ(x) - Φ(x')||²)

RESULT: We get benefits of quantum feature extraction but work with classical 3n-dimensional vectors!
]]></ac:plain-text-body>
</ac:structured-macro>
<h2>3.2 Mathematical Derivation (Step by Step)</h2>
<h3>3.2.1 Step 1: Pauli Matrices and Measurements</h3>
<p>When we &quot;measure&quot; a qubit, we&#x27;re computing the expectation value of an observable. The three standard observables are the <strong>Pauli matrices</strong>:</p>
<ac:structured-macro ac:name="code">
  <ac:parameter ac:name="title">Pauli Matrices Explained</ac:parameter>
  <ac:plain-text-body><![CDATA[
PAULI-X (bit flip):
X = | 0  1 |
    | 1  0 |

- Eigenvalues: +1 (eigenvector |+⟩) and -1 (eigenvector |-⟩)
- Measures "which superposition": |+⟩ vs |-⟩
- On Bloch sphere: X-axis coordinate

PAULI-Y (bit+phase flip):
Y = | 0  -i |
    | i   0 |

- Eigenvalues: +1 and -1
- On Bloch sphere: Y-axis coordinate

PAULI-Z (phase flip):
Z = | 1   0 |
    | 0  -1 |

- Eigenvalues: +1 (eigenvector |0⟩) and -1 (eigenvector |1⟩)
- Measures "which basis state": |0⟩ vs |1⟩
- On Bloch sphere: Z-axis coordinate

EXPECTATION VALUES:
⟨X⟩ = ⟨ψ|X|ψ⟩ = probability(+1) - probability(-1)
⟨Y⟩ = ⟨ψ|Y|ψ⟩ = ...
⟨Z⟩ = ⟨ψ|Z|ψ⟩ = probability(|0⟩) - probability(|1⟩)

Range: All expectation values are in [-1, +1]
]]></ac:plain-text-body>
</ac:structured-macro>
<p><strong>ML Interpretation:</strong> These three numbers (⟨X⟩, ⟨Y⟩, ⟨Z⟩) completely describe the state of a single qubit. They&#x27;re the coordinates on the Bloch sphere!</p>
<h3>3.2.2 Step 2: Bloch Vector Representation</h3>
<p>Any single-qubit state can be written using its Bloch vector:</p>
<ac:structured-macro ac:name="code">
  <ac:parameter ac:name="title">Bloch Vector</ac:parameter>
  <ac:plain-text-body><![CDATA[
DENSITY MATRIX:
ρ = |ψ⟩⟨ψ| = (1/2)(I + rₓX + rᵧY + r_zZ)

where the Bloch vector r⃗ = (rₓ, rᵧ, r_z) has:
- rₓ = ⟨X⟩ (X-coordinate on Bloch sphere)
- rᵧ = ⟨Y⟩ (Y-coordinate)
- r_z = ⟨Z⟩ (Z-coordinate)

CONSTRAINTS:
- For pure states: |r⃗| = 1 (on sphere surface)
- For mixed states: |r⃗| < 1 (inside sphere)
]]></ac:plain-text-body>
</ac:structured-macro>
<h3>3.2.3 Step 3: Reduced Density Matrices</h3>
<p>For multi-qubit systems, each qubit has its own Bloch vector:</p>
<ac:structured-macro ac:name="code">
  <ac:parameter ac:name="title">Reduced Density Matrix</ac:parameter>
  <ac:plain-text-body><![CDATA[
For qubit i in an n-qubit system:

ρᵢ = Tr_{other qubits}(|ψ⟩⟨ψ|)

This "traces out" all other qubits, leaving just the local state of qubit i.

The Bloch vector for qubit i:
- ⟨Xᵢ⟩ = Tr(ρᵢ · X)
- ⟨Yᵢ⟩ = Tr(ρᵢ · Y)
- ⟨Zᵢ⟩ = Tr(ρᵢ · Z)
]]></ac:plain-text-body>
</ac:structured-macro>
<p><strong>ML Interpretation:</strong> Even though our full quantum state lives in 2ⁿ dimensions, each qubit&#x27;s local state is described by just 3 numbers. We extract these 3 numbers per qubit, giving us 3n features total.</p>
<h3>3.2.4 Step 4: Distance Between Quantum States</h3>
<p>How do we measure the &quot;distance&quot; between two quantum states? We use the <strong>Frobenius distance</strong> between their density matrices:</p>
<ac:structured-macro ac:name="code">
  <ac:parameter ac:name="title">Frobenius Distance</ac:parameter>
  <ac:plain-text-body><![CDATA[
FROBENIUS NORM:
||A||_F = √(Tr(A†A)) = √(Σᵢⱼ |Aᵢⱼ|²)

For single-qubit density matrices with Bloch vectors r⃗ and s⃗:

||ρ - σ||²_F = (1/2)||r⃗ - s⃗||²
             = (1/2)[(rₓ - sₓ)² + (rᵧ - sᵧ)² + (r_z - s_z)²]

This is just (half) the squared Euclidean distance between Bloch vectors!
]]></ac:plain-text-body>
</ac:structured-macro>
<h3>3.2.5 Step 5: The Complete PQK Formula</h3>
<p>Putting it all together, the <strong>Projected Quantum Kernel</strong> is:</p>
<ac:structured-macro ac:name="code">
  <ac:parameter ac:name="title">Projected Quantum Kernel - Complete Formula</ac:parameter>
  <ac:plain-text-body><![CDATA[
STEP 1: Define the quantum distance
D(x, x') = Σᵢ₌₁ⁿ 2·||ρᵢ(x) - ρᵢ(x')||²_F

Expanding using Bloch vectors:
D(x, x') = Σᵢ₌₁ⁿ [(⟨Xᵢ⟩ˣ - ⟨Xᵢ⟩ˣ')² + (⟨Yᵢ⟩ˣ - ⟨Yᵢ⟩ˣ')² + (⟨Zᵢ⟩ˣ - ⟨Zᵢ⟩ˣ')²]

STEP 2: Apply RBF kernel formula
k_PQK(x, x') = exp(-α · D(x, x'))

where α > 0 is the bandwidth parameter (like γ in classical RBF).

EQUIVALENT FORM:
If we define the projected feature vector:
Φ(x) = [⟨X₁⟩, ⟨Y₁⟩, ⟨Z₁⟩, ⟨X₂⟩, ⟨Y₂⟩, ⟨Z₂⟩, ..., ⟨Xₙ⟩, ⟨Yₙ⟩, ⟨Zₙ⟩]ᵀ ∈ ℝ³ⁿ

Then:
k_PQK(x, x') = exp(-α · ||Φ(x) - Φ(x')||²)

This is EXACTLY an RBF kernel in the 3n-dimensional projected feature space!
]]></ac:plain-text-body>
</ac:structured-macro>
<ac:structured-macro ac:name="panel">
  <ac:parameter ac:name="title">Key Insight for ML Practitioners</ac:parameter>
  <ac:parameter ac:name="borderStyle">solid</ac:parameter>
  <ac:parameter ac:name="borderColor">#090</ac:parameter>
  <ac:parameter ac:name="titleBGColor">#dfd</ac:parameter>
  <ac:rich-text-body>
<p><strong>The PQK is simply an RBF kernel applied to quantum-derived features.</strong> The &quot;quantum magic&quot; is in HOW we compute the feature vector Φ(x) - we use a quantum circuit as a non-linear feature extractor.</p>
<p><strong>Comparison to Deep Learning:</strong> This is similar to using a pre-trained CNN as a feature extractor, then applying a simple classifier on the extracted features. Here, the quantum circuit is our &quot;feature extractor.&quot;</p>
  </ac:rich-text-body>
</ac:structured-macro>
<h2>3.3 Why Does This Work? (Theoretical Justification)</h2>
<h3>3.3.1 Expressiveness of Quantum Feature Maps</h3>
<ac:structured-macro ac:name="code">
  <ac:parameter ac:name="title">Why Quantum Features Are Powerful</ac:parameter>
  <ac:plain-text-body><![CDATA[
CLASSICAL RBF:
- Creates features like: 1, x₁, x₂, x₁², x₁x₂, x₂², x₁³, ...
- All polynomial-like combinations

QUANTUM FEATURE MAP:
- Creates features involving trigonometric functions: sin(γx₁), cos(γx₁x₂), ...
- Complex interference patterns between features
- Entanglement creates correlations that polynomial features cannot capture

KEY PROPERTY:
Quantum feature maps can be "classically hard to simulate" - meaning classical computers cannot efficiently compute the same features. This is the source of potential quantum advantage.
]]></ac:plain-text-body>
</ac:structured-macro>
<h3>3.3.2 What Information is Captured?</h3>
<ac:structured-macro ac:name="code">
  <ac:parameter ac:name="title">Information Content Analysis</ac:parameter>
  <ac:plain-text-body><![CDATA[
WHAT WE KEEP:
- All single-qubit marginal distributions
- Local quantum correlations mediated by entanglement
- Non-linear functions of input features (via rotations)

WHAT WE LOSE:
- Full multi-qubit correlations (would need exponential storage)
- Global entanglement structure

WHY THIS IS OKAY:
For many ML tasks, local correlations + entanglement-mediated interactions are sufficient. We trade some quantum expressiveness for classical tractability.
]]></ac:plain-text-body>
</ac:structured-macro>
<hr />
<h1>4. Our Circuit Architecture (The Hamiltonian Ansatz)</h1>
<h2>4.1 Circuit Design Philosophy</h2>
<h3>4.1.1 Requirements for a Good Quantum Feature Map</h3>
<ac:structured-macro ac:name="code">
  <ac:parameter ac:name="title">Feature Map Design Criteria</ac:parameter>
  <ac:plain-text-body><![CDATA[
1. DATA ENCODING: Must encode all input features into quantum state
2. ENTANGLEMENT: Must create correlations between features
3. EXPRESSIVENESS: Must create non-trivial, non-linear features
4. SIMULABILITY: Must be efficiently simulable on classical computers
5. PARAMETERIZATION: Hyperparameters to tune for different datasets
]]></ac:plain-text-body>
</ac:structured-macro>
<h3>4.1.2 Why &quot;Hamiltonian&quot; Ansatz?</h3>
<p>The name comes from physics: our circuit structure resembles time evolution under a quantum Hamiltonian (energy operator). Specifically:</p>
<ac:structured-macro ac:name="code">
  <ac:parameter ac:name="title">Physics Inspiration</ac:parameter>
  <ac:plain-text-body><![CDATA[
QUANTUM TIME EVOLUTION:
|ψ(t)⟩ = exp(-iHt)|ψ(0)⟩

where H is the Hamiltonian operator.

OUR CIRCUIT:
|ψ(x)⟩ = exp(-iH(x)·γ)|+⟩ⁿ

where H(x) encodes our data features.

The Rz and R_XX gates implement this evolution approximately.
]]></ac:plain-text-body>
</ac:structured-macro>
<h2>4.2 Complete Circuit Structure</h2>
<ac:structured-macro ac:name="code">
  <ac:parameter ac:name="title">Hamiltonian Ansatz - Layer by Layer</ac:parameter>
  <ac:plain-text-body><![CDATA[
═══════════════════════════════════════════════════════════════════════════════
                        HAMILTONIAN ANSATZ CIRCUIT
═══════════════════════════════════════════════════════════════════════════════

INPUT: Classical feature vector x = [x₁, x₂, ..., xₙ] ∈ ℝⁿ
       (scaled to [-π, π] or [-π/4, π/4])

PARAMETERS:
- n: number of qubits (= number of features)
- r: number of layer repetitions (default: 2-10)
- γ: rotation scaling factor (default: 0.1-1.0)

───────────────────────────────────────────────────────────────────────────────
LAYER 0: INITIALIZATION (applied once)
───────────────────────────────────────────────────────────────────────────────

Apply Hadamard (H) gate to each qubit:

|0⟩ ──[H]── |+⟩ = (|0⟩ + |1⟩)/√2

PURPOSE: Create equal superposition as starting point.
         Without this, rotations would have no effect on |0⟩ state.

RESULT: |ψ_init⟩ = |+⟩⊗ⁿ = (1/√2ⁿ) Σ |i₁i₂...iₙ⟩

───────────────────────────────────────────────────────────────────────────────
LAYER 1-r: ENCODING + ENTANGLEMENT (repeated r times)
───────────────────────────────────────────────────────────────────────────────

STEP A: Single-qubit rotations (DATA ENCODING)

For each qubit i, apply:
  Rz(θᵢ) where θᵢ = γ · xᵢ / π

|ψ⟩ ──[Rz(γxᵢ/π)]── rotated |ψ⟩

WHAT THIS DOES:
- Encodes feature xᵢ as a rotation angle
- γ scales the rotation (smaller γ = gentler encoding)
- Division by π normalizes to reasonable rotation range

MATHEMATICAL EFFECT:
Rz(θ)|ψ⟩ adds a phase e^(±iθ/2) depending on |0⟩ vs |1⟩ component

---

STEP B: Two-qubit entangling gates (FEATURE INTERACTION)

For each pair (i, j) in entanglement map, apply:
  R_XX(θᵢⱼ) where θᵢⱼ = γ² · (1 - xᵢ) · (1 - xⱼ)

WHAT THIS DOES:
- Creates entanglement between qubits i and j
- Rotation angle depends on BOTH features xᵢ and xⱼ
- This is how we capture FEATURE INTERACTIONS

MATHEMATICAL EFFECT:
R_XX(θ) = exp(-i·θ·X⊗X/2)

This jointly rotates both qubits in a correlated way.

───────────────────────────────────────────────────────────────────────────────
OUTPUT: MEASUREMENT (expectation values)
───────────────────────────────────────────────────────────────────────────────

For each qubit i, compute:
  ⟨Xᵢ⟩ = ⟨ψ_final|Xᵢ|ψ_final⟩
  ⟨Yᵢ⟩ = ⟨ψ_final|Yᵢ|ψ_final⟩
  ⟨Zᵢ⟩ = ⟨ψ_final|Zᵢ|ψ_final⟩

OUTPUT: Φ(x) = [⟨X₁⟩, ⟨Y₁⟩, ⟨Z₁⟩, ..., ⟨Xₙ⟩, ⟨Yₙ⟩, ⟨Zₙ⟩] ∈ ℝ³ⁿ
]]></ac:plain-text-body>
</ac:structured-macro>
<h2>4.3 Visual Circuit Diagram</h2>
<ac:structured-macro ac:name="code">
  <ac:parameter ac:name="title">Circuit Diagram (5 qubits, 2 reps)</ac:parameter>
  <ac:plain-text-body><![CDATA[
q₀: |0⟩─[H]─[Rz]─[XX]──────────[Rz]─[XX]──────────⟨X₀,Y₀,Z₀⟩
              │                  │
q₁: |0⟩─[H]─[Rz]─[XX]─[XX]─────[Rz]─[XX]─[XX]────⟨X₁,Y₁,Z₁⟩
                   │                  │
q₂: |0⟩─[H]─[Rz]─────[XX]─[XX]─[Rz]─────[XX]─[XX]⟨X₂,Y₂,Z₂⟩
                       │                  │
q₃: |0⟩─[H]─[Rz]───────[XX]─[XX][Rz]─────[XX]─[XX]⟨X₃,Y₃,Z₃⟩
                             │                │
q₄: |0⟩─[H]─[Rz]─────────────[XX][Rz]─────────[XX]⟨X₄,Y₄,Z₄⟩
        │    │                │    │              │
        └────┴── Rep 1 ───────┘    └── Rep 2 ─────┘

Legend:
[H]  = Hadamard gate
[Rz] = Rz(γxᵢ/π) rotation (data encoding)
[XX] = R_XX(γ²(1-xᵢ)(1-xⱼ)) entangling gate
]]></ac:plain-text-body>
</ac:structured-macro>
<h2>4.4 Entanglement Topology</h2>
<h3>4.4.1 Linear Nearest-Neighbor Connectivity</h3>
<ac:structured-macro ac:name="code">
  <ac:parameter ac:name="title">Entanglement Map</ac:parameter>
  <ac:plain-text-body><![CDATA[
We use LINEAR nearest-neighbor connectivity:

Qubits:  0 ─── 1 ─── 2 ─── 3 ─── 4

Entanglement pairs: [(0,1), (1,2), (2,3), (3,4)]

WHY LINEAR?
1. Matches real quantum hardware constraints
2. Sufficient for capturing local correlations
3. Enables efficient MPS simulation (see next section)

ALTERNATIVE: Full connectivity (all pairs)
- Would create [(0,1), (0,2), (0,3), ..., (3,4)]
- More expressive but exponentially harder to simulate
]]></ac:plain-text-body>
</ac:structured-macro>
<h3>4.4.2 Gate Ordering</h3>
<ac:structured-macro ac:name="code">
  <ac:parameter ac:name="title">Entanglement Graph Algorithm</ac:parameter>
  <ac:plain-text-body><![CDATA[
def entanglement_graph(nq, nn=1):
    """
    Generate non-overlapping gate layers.

    nq: number of qubits
    nn: neighborhood depth (1 = nearest neighbors only)
    """
    pairs = []
    for distance in range(1, nn+1):
        busy = set()
        # Layer 1: Non-overlapping pairs
        for i in range(nq):
            if i not in busy and i+distance < nq:
                pairs.append((i, i+distance))
                busy.add(i+distance)
        # Layer 2: Remaining pairs
        for i in busy:
            if i+distance < nq:
                pairs.append((i, i+distance))
    return pairs

# Example: entanglement_graph(5, 1)
# Returns: [(0,1), (2,3), (1,2), (3,4)]
# This allows parallel execution of (0,1) and (2,3), then (1,2) and (3,4)
]]></ac:plain-text-body>
</ac:structured-macro>
<h2>4.5 Alternative: The Magic Ansatz</h2>
<ac:structured-macro ac:name="code">
  <ac:parameter ac:name="title">Magic Ansatz Structure</ac:parameter>
  <ac:plain-text-body><![CDATA[
For comparison, the MAGIC ANSATZ uses different gates:

LAYER STRUCTURE (repeated r times):
1. Hadamard (H) on all qubits
2. T gate (π/8 rotation) on all qubits
3. CZ gates on entangled pairs
4. Rz(xᵢ) encoding on each qubit

q₀: |0⟩─[H]─[T]─[CZ]────[Rz]─[H]─[T]─[CZ]────[Rz]─...
                 │                    │
q₁: |0⟩─[H]─[T]─[CZ]─[CZ][Rz]─[H]─[T]─[CZ]─[CZ][Rz]─...
                      │                    │
q₂: |0⟩─[H]─[T]──────[CZ][Rz]─[H]─[T]──────[CZ][Rz]─...

WHY "MAGIC"?
- The H+T combination creates "magic states"
- These states exhibit quantum contextuality
- May be useful for specific data distributions

COMPARISON:
- Hamiltonian: Smooth, continuous encoding (better for continuous features)
- Magic: Discrete, structured encoding (may work better for categorical-like data)
]]></ac:plain-text-body>
</ac:structured-macro>
<hr />
<h1>5. Tensor Network Simulation (How We Compute This Efficiently)</h1>
<ac:structured-macro ac:name="note">
  <ac:parameter ac:name="title">Why This Section is Important</ac:parameter>
  <ac:rich-text-body>
<p>The previous sections explained WHAT we compute. This section explains HOW we compute it efficiently on classical computers. This is the key engineering that makes the whole system practical.</p>
  </ac:rich-text-body>
</ac:structured-macro>
<h2>5.1 The Computational Challenge</h2>
<h3>5.1.1 The Exponential Wall</h3>
<ac:structured-macro ac:name="code">
  <ac:parameter ac:name="title">Why Naive Simulation Fails</ac:parameter>
  <ac:plain-text-body><![CDATA[
NAIVE APPROACH:
Store the full quantum state as a vector of 2ⁿ complex numbers.

MEMORY REQUIREMENTS:
| Qubits | State Vector Size | Memory (complex128) |
|--------|-------------------|---------------------|
| 20     | 2²⁰ = 1M          | 16 MB               |
| 30     | 2³⁰ = 1B          | 16 GB               |
| 40     | 2⁴⁰ = 1T          | 16 TB               |
| 50     | 2⁵⁰ = 1P          | 16 PB               |

PROBLEM: Even 50 qubits exceeds world's total storage!
]]></ac:plain-text-body>
</ac:structured-macro>
<h3>5.1.2 The Key Insight: Exploiting Structure</h3>
<ac:structured-macro ac:name="code">
  <ac:parameter ac:name="title">Why MPS Works</ac:parameter>
  <ac:plain-text-body><![CDATA[
NOT ALL QUANTUM STATES ARE EQUALLY COMPLEX

Key observation: Many useful quantum states have LIMITED ENTANGLEMENT.

AREA LAW of entanglement:
For 1D systems, entanglement across a cut grows with boundary size, not volume.
S(ρ_A) ≤ c · |∂A|

OUR CIRCUITS:
- Linear connectivity → 1D entanglement structure
- Shallow depth → bounded entanglement growth
- Result: Can be represented efficiently!
]]></ac:plain-text-body>
</ac:structured-macro>
<h2>5.2 Matrix Product States (MPS)</h2>
<h3>5.2.1 The MPS Representation</h3>
<ac:structured-macro ac:name="code">
  <ac:parameter ac:name="title">MPS Definition</ac:parameter>
  <ac:plain-text-body><![CDATA[
STANDARD STATE VECTOR:
|ψ⟩ = Σ c_{i₁i₂...iₙ} |i₁i₂...iₙ⟩

where c is a tensor with 2ⁿ elements.

MATRIX PRODUCT STATE:
|ψ⟩ = Σ A¹[i₁] · A²[i₂] · ... · Aⁿ[iₙ] |i₁i₂...iₙ⟩

where:
- Each Aᵏ[iₖ] is a MATRIX (not just a number)
- iₖ ∈ {0, 1} is the physical index
- Matrix dimensions are χₖ₋₁ × χₖ
- χₖ is called the BOND DIMENSION

PICTORIAL REPRESENTATION:
    i₁    i₂    i₃    i₄    i₅     (physical indices)
    │     │     │     │     │
   [A¹]──[A²]──[A³]──[A⁴]──[A⁵]    (MPS tensors)
      χ₁   χ₂   χ₃   χ₄           (bond dimensions)

Each line represents matrix multiplication.
The contraction of all matrices gives the coefficient c_{i₁i₂...iₙ}.
]]></ac:plain-text-body>
</ac:structured-macro>
<h3>5.2.2 Bond Dimension and Compression</h3>
<ac:structured-macro ac:name="code">
  <ac:parameter ac:name="title">Bond Dimension Explained</ac:parameter>
  <ac:plain-text-body><![CDATA[
BOND DIMENSION (χ):
- Controls the "capacity" of the MPS representation
- Larger χ = more entanglement can be captured
- Smaller χ = more compression

PARAMETER COUNT:
- Full state vector: 2ⁿ complex numbers
- MPS with bond dim χ: O(n · χ²) complex numbers

EXAMPLE (n=50 qubits):
- Full state: 2⁵⁰ ≈ 10¹⁵ numbers (impossible)
- MPS with χ=100: 50 × 100² × 2 = 10⁶ numbers (easy!)

ENTANGLEMENT BOUND:
Maximum entanglement entropy across any cut: S ≤ log₂(χ)

For χ=100: S_max = log₂(100) ≈ 6.6 bits
This is MUCH less than the maximum possible (n/2 bits for n qubits)
]]></ac:plain-text-body>
</ac:structured-macro>
<h3>5.2.3 Why This Works for Our Circuits</h3>
<ac:structured-macro ac:name="code">
  <ac:parameter ac:name="title">MPS Suitability Analysis</ac:parameter>
  <ac:plain-text-body><![CDATA[
OUR CIRCUIT PROPERTIES:

1. LINEAR CONNECTIVITY
   - Qubits only interact with neighbors
   - Entanglement has 1D structure
   - MPS is designed for exactly this!

2. SHALLOW DEPTH
   - Few layers (r = 2-10)
   - Entanglement grows slowly
   - Bond dimension stays manageable

3. LOCAL GATES
   - Single-qubit gates: Don't increase χ
   - Two-qubit gates on neighbors: Increase χ by at most 2x

EMPIRICAL OBSERVATION:
For our Hamiltonian ansatz with r ≤ 10, χ rarely exceeds 50.
This makes simulation very efficient.
]]></ac:plain-text-body>
</ac:structured-macro>
<h2>5.3 Gate Application on MPS</h2>
<h3>5.3.1 Single-Qubit Gates (Easy Case)</h3>
<ac:structured-macro ac:name="code">
  <ac:parameter ac:name="title">Single-Qubit Gate Application</ac:parameter>
  <ac:plain-text-body><![CDATA[
OPERATION: Apply gate U to qubit k

ALGORITHM:
1. Find tensor Aᵏ at position k
2. For each physical index value j:
   Ãᵏ[j] = Σᵢ U[j,i] · Aᵏ[i]
3. Replace Aᵏ with Ãᵏ

COMPLEXITY: O(χ²) - just matrix operations
BOND DIMENSION: Unchanged! Single-qubit gates don't create entanglement.

EXAMPLE (Rz gate):
Aᵏ[0] → e^(-iθ/2) · Aᵏ[0]
Aᵏ[1] → e^(+iθ/2) · Aᵏ[1]
]]></ac:plain-text-body>
</ac:structured-macro>
<h3>5.3.2 Two-Qubit Gates (The Tricky Part)</h3>
<ac:structured-macro ac:name="code">
  <ac:parameter ac:name="title">Two-Qubit Gate Application</ac:parameter>
  <ac:plain-text-body><![CDATA[
OPERATION: Apply gate U to adjacent qubits k and k+1

ALGORITHM:
1. CONTRACT neighboring tensors:
   Θ[i,j] = Σ_α Aᵏ[i]_α · Aᵏ⁺¹[j]_α
   (This creates a 4-index tensor)

2. APPLY the gate:
   Θ̃[i',j'] = Σ_{i,j} U[i'j', ij] · Θ[i,j]

3. DECOMPOSE via SVD (Singular Value Decomposition):
   Θ̃ = U · Σ · V†
   where Σ is diagonal with singular values σ₁ ≥ σ₂ ≥ ... ≥ σᵣ

4. TRUNCATE to maximum bond dimension χ_max:
   Keep only top χ_max singular values
   This introduces small approximation error

5. UPDATE MPS:
   Aᵏ ← U
   Aᵏ⁺¹ ← Σ · V†

COMPLEXITY: O(χ³) due to SVD
BOND DIMENSION: Can increase up to 2χ before truncation

TRUNCATION ERROR:
Error ≤ √(Σᵢ>χ_max σᵢ²)
With cutoff 10⁻¹⁰, errors are negligible.
]]></ac:plain-text-body>
</ac:structured-macro>
<ac:structured-macro ac:name="panel">
  <ac:parameter ac:name="title">ML Analogy: SVD Truncation</ac:parameter>
  <ac:parameter ac:name="borderStyle">solid</ac:parameter>
  <ac:rich-text-body>
<p>The SVD truncation in MPS is similar to <strong>PCA truncation</strong> or <strong>low-rank approximation</strong> in classical ML. We&#x27;re keeping the &quot;most important&quot; components and discarding the rest.</p>
<p>The difference: In MPS, this happens at EVERY two-qubit gate, not just once at the end. The errors can accumulate, but for our shallow circuits, they remain small.</p>
  </ac:rich-text-body>
</ac:structured-macro>
<h3>5.3.3 Non-Adjacent Gates (SWAP Strategy)</h3>
<ac:structured-macro ac:name="code">
  <ac:parameter ac:name="title">Non-Adjacent Gate Handling</ac:parameter>
  <ac:plain-text-body><![CDATA[
PROBLEM: What if we need a gate between non-adjacent qubits?

Example: Gate between qubits 0 and 3
   q₀ ─── q₁ ─── q₂ ─── q₃

SOLUTION: Use SWAP gates to move qubits next to each other

ALGORITHM:
1. SWAP q₀ with q₁: Now logical qubit 0 is at position 1
2. SWAP (logical 0) with q₂: Now at position 2
3. Apply the gate between positions 2 and 3
4. SWAP back to restore original ordering

COMPLEXITY: O(d × χ³) where d is the distance between qubits

OUR APPROACH:
The pytket compiler automatically handles this with its DecomposeBRIDGE pass.
It inserts the necessary SWAP gates during circuit compilation.
]]></ac:plain-text-body>
</ac:structured-macro>
<h2>5.4 Computing Expectation Values</h2>
<ac:structured-macro ac:name="code">
  <ac:parameter ac:name="title">Expectation Value Computation</ac:parameter>
  <ac:plain-text-body><![CDATA[
GOAL: Compute ⟨ψ|Xᵢ|ψ⟩ for qubit i (similarly for Y, Z)

ALGORITHM (using canonical form):
1. Put MPS in "mixed canonical form" centered at site i
   - Sites 1 to i-1: Left-canonical (L†L = I)
   - Sites i+1 to n: Right-canonical (RR† = I)
   - Site i: Contains all the "weight"

2. Compute local expectation value:
   ⟨Xᵢ⟩ = Tr(Aⁱ† · X · Aⁱ)

   Due to canonical form, all other sites cancel out!

COMPLEXITY: O(χ²) per expectation value
TOTAL: O(n × χ²) for all 3n expectation values

WHY CANONICAL FORM HELPS:

Before:
⟨ψ|Xᵢ|ψ⟩ = Σ (A¹...Aⁱ⁻¹)† · (Aⁱ†XAⁱ) · (Aⁱ⁺¹...Aⁿ)(A¹...Aⁿ)
         = Complicated contraction over entire network

After (canonical form):
⟨ψ|Xᵢ|ψ⟩ = Tr(Aⁱ† · X · Aⁱ)
         = Simple local computation!
]]></ac:plain-text-body>
</ac:structured-macro>
<h2>5.5 ITensor Library Implementation</h2>
<h3>5.5.1 What is ITensor?</h3>
<ac:structured-macro ac:name="code">
  <ac:parameter ac:name="title">ITensor Overview</ac:parameter>
  <ac:plain-text-body><![CDATA[
ITensor is a C++ library for tensor network calculations.

KEY FEATURES:
- Automatic index contraction (you don't manage dimensions manually)
- Efficient SVD and tensor decompositions
- Built-in MPS/MPO algorithms
- Used by quantum physics researchers worldwide

REFERENCE:
Fishman, M., White, S. R., & Stoudenmire, E. M. (2022).
"The ITensor Software Library for Tensor Network Calculations."
SciPost Physics Codebases, 4.
]]></ac:plain-text-body>
</ac:structured-macro>
<h3>5.5.2 Our C++ Implementation</h3>
<ac:structured-macro ac:name="code">
  <ac:parameter ac:name="title">Key C++ Code (helloitensor.cc)</ac:parameter>
  <ac:plain-text-body><![CDATA[
// Main simulation function
MPS apply_gates(vector<tuple<int,int,int,double>> circuits,
                Qubit site_inds, int N, double cutoff) {

    // 1. Initialize MPS in |0⟩^⊗n state
    auto init = InitState(site_inds);
    for(auto n : range1(N)) {
        init.set(n, "Up");  // |0⟩ state
    }
    auto psi = MPS(init);

    // 2. Apply each gate
    for (auto gate : circuits) {
        auto type = get<0>(gate);   // Gate type (0=H, 1=Rx, 2=Rz, etc.)
        auto q1 = get<1>(gate);     // First qubit
        auto q2 = get<2>(gate);     // Second qubit (-1 for single-qubit)
        auto angle = get<3>(gate);  // Rotation angle

        if (type == 0) {  // Hadamard
            psi.position(q1+1);
            auto G = op(site_inds, "H", q1+1);
            auto new_MPS = G * psi(q1+1);
            new_MPS.noPrime();
            psi.set(q1+1, new_MPS);
        }
        else if (type == 2) {  // Rz
            psi.position(q1+1);
            auto G = op(site_inds, "Rz", q1+1, {"alpha=", angle});
            auto new_MPS = G * psi(q1+1);
            new_MPS.noPrime();
            psi.set(q1+1, new_MPS);
        }
        else if (type == 3) {  // XXPhase (two-qubit)
            // Contract two sites
            psi.position(q1+1);
            auto wf = psi(q1+1) * psi(q2+1);

            // Apply gate
            auto opx1 = op(site_inds, "X", q1+1);
            auto opx2 = op(site_inds, "X", q2+1);
            auto G = expHermitian(opx2 * opx1, -i*theta);
            wf *= G;
            wf.noPrime();

            // SVD to restore MPS form
            auto [U, S, V] = svd(wf, inds(psi(q1+1)), {"Cutoff=", 1E-10});
            psi.set(q1+1, U);
            psi.set(q2+1, S*V);
        }
        // ... other gate types
    }
    return psi;
}

// Extract expectation values
vector<vector<double>> circuit_xyz_exp(gates, n_qubits) {
    MPS psi = apply_gates(gates, sites, n_qubits, 1E-16);

    vector<vector<double>> results;
    for (int i = 0; i < n_qubits; i++) {
        psi.position(i+1);

        double x = eltC(dag(prime(psi.A(i+1),"Site")) *
                        sites.op("X_half",i+1) * psi.A(i+1)).real();
        double y = eltC(dag(prime(psi.A(i+1),"Site")) *
                        sites.op("Y_half",i+1) * psi.A(i+1)).real();
        double z = eltC(dag(prime(psi.A(i+1),"Site")) *
                        sites.op("Z_half",i+1) * psi.A(i+1)).real();

        results.push_back({x, y, z});
    }
    return results;
}
]]></ac:plain-text-body>
</ac:structured-macro>
<h3>5.5.3 Python-C++ Bridge</h3>
<ac:structured-macro ac:name="code">
  <ac:parameter ac:name="title">pybind11 Interface</ac:parameter>
  <ac:plain-text-body><![CDATA[
// C++ side (in helloitensor.cc)
PYBIND11_MODULE(helloitensor, m) {
    m.def("circuit_xyz_exp", &circuit_xyz_exp<int,double>,
          "Extract X,Y,Z expectation values from circuit");
}

// Python side
from helloitensor import circuit_xyz_exp

# gates = [[type, q1, q2, angle], ...]
# Returns [[<X1>, <Y1>, <Z1>], [<X2>, <Y2>, <Z2>], ...]
expectation_values = circuit_xyz_exp(gate_list, n_qubits)
]]></ac:plain-text-body>
</ac:structured-macro>
<h2>5.6 Complexity Summary</h2>
<table>
  <tbody>
    <tr>
      <th>Operation</th>
      <th>Complexity</th>
      <th>Notes</th>
    </tr>
    <tr>
      <td>Initialize \</td>
      <td>0⟩^⊗n</td>
      <td>O(n)</td>
      <td>χ = 1</td>
    </tr>
    <tr>
      <td>Single-qubit gate</td>
      <td>O(χ²)</td>
      <td>No χ increase</td>
    </tr>
    <tr>
      <td>Two-qubit gate (adjacent)</td>
      <td>O(χ³)</td>
      <td>SVD step, may increase χ</td>
    </tr>
    <tr>
      <td>Two-qubit gate (distance d)</td>
      <td>O(d × χ³)</td>
      <td>Requires d SWAPs</td>
    </tr>
    <tr>
      <td>Expectation value (per qubit)</td>
      <td>O(χ²)</td>
      <td>Using canonical form</td>
    </tr>
    <tr>
      <td>Full circuit simulation</td>
      <td>O(n × r × χ³)</td>
      <td>n qubits, r reps</td>
    </tr>
  </tbody>
</table>
<ac:structured-macro ac:name="panel">
  <ac:parameter ac:name="title">Practical Performance</ac:parameter>
  <ac:parameter ac:name="borderStyle">solid</ac:parameter>
  <ac:rich-text-body>
<p>For typical parameters (n=12 qubits, r=10 reps, χ≈50):</p>
<ul>
  <li>Single circuit simulation: ~50-100 ms</li>
  <li>All expectation values: ~5 ms</li>
  <li>Total per data point: ~100 ms</li>
</ul>
  </ac:rich-text-body>
</ac:structured-macro>
<hr />
<h1>6. Complete Pipeline Implementation</h1>
<h2>6.1 Stage 1: Data Preprocessing</h2>
<ac:structured-macro ac:name="code">
  <ac:parameter ac:name="title">Complete Preprocessing Pipeline</ac:parameter>
  <ac:plain-text-body><![CDATA[
═══════════════════════════════════════════════════════════════════════════════
                         DATA PREPROCESSING PIPELINE
═══════════════════════════════════════════════════════════════════════════════

INPUT: Raw CSV data with features and labels

───────────────────────────────────────────────────────────────────────────────
STEP 1: Load and Clean Data
───────────────────────────────────────────────────────────────────────────────

import pandas as pd
import numpy as np

# Load data
data = pd.read_csv('elliptic_txs_features.csv')
labels = pd.read_csv('elliptic_txs_classes.csv')

# Encode labels: "1" = illicit → 0, "2" = licit → 1, "unknown" → remove
labels.loc[labels["Class"] == "1", "Class"] = 0   # Fraud
labels.loc[labels["Class"] == "2", "Class"] = 1   # Legitimate
labels.loc[labels["Class"] == "unknown", "Class"] = 99  # To be removed

# Remove unlabeled data
clean_data = data.drop(np.where(labels['Class']==99)[0])

───────────────────────────────────────────────────────────────────────────────
STEP 2: Class Balancing (Handle Imbalance)
───────────────────────────────────────────────────────────────────────────────

WHY NEEDED:
- Fraud is rare (~10% of labeled data)
- Imbalanced classes lead to biased models
- We undersample to create balanced training set

def draw_sample(df, n_fraud, n_legit, test_frac=0.2, seed=123):
    """
    Sample balanced data and create train/test split.
    """
    # Sample from each class
    fraud_samples = df[df['Class']==0].sample(n_fraud, random_state=seed*20+2)
    legit_samples = df[df['Class']==1].sample(n_legit, random_state=seed*46+9)
    balanced_data = pd.concat([fraud_samples, legit_samples])

    # Stratified split maintains class balance in both sets
    from sklearn.model_selection import train_test_split
    train, test = train_test_split(
        balanced_data,
        stratify=balanced_data['Class'],
        test_size=test_frac,
        random_state=seed*26+19
    )

    return train_features, train_labels, test_features, test_labels

# Example: 100 fraud + 100 legitimate samples
train_X, train_y, test_X, test_y = draw_sample(data, 100, 100, 0.2, seed=456)

───────────────────────────────────────────────────────────────────────────────
STEP 3: Feature Scaling (Critical for Quantum Encoding!)
───────────────────────────────────────────────────────────────────────────────

WHY EACH STEP IS NEEDED:

A) QUANTILE TRANSFORM:
   - Maps data to Gaussian distribution
   - Handles outliers (fraud amounts can be extreme)
   - Preserves rank ordering

   from sklearn.preprocessing import QuantileTransformer
   qt = QuantileTransformer(output_distribution='normal')
   X_qt = qt.fit_transform(X)

   # Before: [0.01, 0.5, 1000000]  (outlier!)
   # After:  [-2.3, 0.0, 2.3]       (normalized)

B) STANDARD SCALER:
   - Zero mean, unit variance
   - Makes features comparable in scale

   from sklearn.preprocessing import StandardScaler
   ss = StandardScaler()
   X_ss = ss.fit_transform(X_qt)

   # Result: mean=0, std=1 for each feature

C) MINMAX SCALER TO [-π, π] or [-π/4, π/4]:
   - Maps to quantum rotation range
   - CRITICAL: Quantum rotations are periodic with period 2π
   - Using [-π/4, π/4] for many qubits prevents over-rotation

   from sklearn.preprocessing import MinMaxScaler

   # For few qubits (n < 10): Full range
   mms = MinMaxScaler(feature_range=(-np.pi, np.pi))

   # For many qubits (n > 20): Restricted range
   mms = MinMaxScaler(feature_range=(-np.pi/4, np.pi/4))

   X_final = mms.fit_transform(X_ss)

───────────────────────────────────────────────────────────────────────────────
STEP 4: Feature Selection
───────────────────────────────────────────────────────────────────────────────

WHY NEEDED:
- Number of qubits = Number of features
- More qubits = slower simulation, more entanglement
- Select most informative features

# Simple approach: Take first k features (Elliptic features are pre-ordered)
num_features = 12  # = number of qubits
X_selected = X_final[:, :num_features]

# Alternative: Use feature importance
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=100)
rf.fit(X_final, y)
top_features = np.argsort(rf.feature_importances_)[::-1][:num_features]
X_selected = X_final[:, top_features]
]]></ac:plain-text-body>
</ac:structured-macro>
<h2>6.2 Stage 2: Quantum Circuit Construction</h2>
<ac:structured-macro ac:name="code">
  <ac:parameter ac:name="title">Circuit Construction Code</ac:parameter>
  <ac:plain-text-body><![CDATA[
═══════════════════════════════════════════════════════════════════════════════
                         CIRCUIT CONSTRUCTION
═══════════════════════════════════════════════════════════════════════════════

from pytket import Circuit, OpType
from sympy import Symbol

class ProjectedKernelStateAnsatz:
    def __init__(self, num_features, reps, gamma, entanglement_map,
                 ansatz="hamiltonian", hadamard_init=True):
        """
        Initialize the quantum feature map.

        Parameters:
        -----------
        num_features : int
            Number of features = number of qubits
        reps : int
            Number of layer repetitions (2-10 typical)
        gamma : float
            Rotation scaling (0.1-1.0)
        entanglement_map : list of tuples
            Pairs of qubits to entangle [(0,1), (1,2), ...]
        ansatz : str
            "hamiltonian" or "magic"
        hadamard_init : bool
            Whether to apply H gates initially
        """
        self.num_features = num_features
        self.reps = reps
        self.gamma = gamma

        # Create symbolic circuit (parameters will be substituted later)
        self.ansatz_circ = Circuit(num_features)

        # Create symbols for each feature
        self.feature_symbols = [Symbol(f'f_{i}') for i in range(num_features)]

        # Build the circuit
        if ansatz == "hamiltonian":
            self._build_hamiltonian_ansatz()
        elif ansatz == "magic":
            self._build_magic_ansatz()

    def _build_hamiltonian_ansatz(self):
        """Build Rz + XXPhase circuit."""

        # Initial Hadamard layer
        if self.hadamard_init:
            for i in range(self.num_features):
                self.ansatz_circ.H(i)

        # Repeated layers
        for _ in range(self.reps):
            # Single-qubit Rz rotations (encode features)
            for i in range(self.num_features):
                # Rotation angle = γ * x_i / π
                angle = (self.gamma / np.pi) * self.feature_symbols[i]
                self.ansatz_circ.Rz(angle, i)

            # Two-qubit XXPhase gates (create entanglement)
            for (q0, q1) in self.entanglement_map:
                # Rotation angle = γ² * (1 - x_i) * (1 - x_j)
                s0 = self.feature_symbols[q0]
                s1 = self.feature_symbols[q1]
                angle = self.gamma**2 * (1 - s0) * (1 - s1)
                self.ansatz_circ.XXPhase(angle, q0, q1)

        # Compile to linear architecture (insert SWAPs if needed)
        self._compile_circuit()

    def circuit_for_data(self, feature_values):
        """
        Create concrete circuit for specific data point.

        Parameters:
        -----------
        feature_values : array-like
            The scaled feature vector [x_1, x_2, ..., x_n]

        Returns:
        --------
        Circuit with all symbols replaced by values
        """
        # Create symbol → value mapping
        symbol_map = {
            sym: val
            for sym, val in zip(self.feature_symbols, feature_values)
        }

        # Copy circuit and substitute values
        concrete_circuit = self.ansatz_circ.copy()
        concrete_circuit.symbol_substitution(symbol_map)

        return concrete_circuit

    def circuit_to_list(self, circuit):
        """
        Convert circuit to gate list for ITensor.

        Returns:
        --------
        List of [gate_type, qubit1, qubit2, angle]
        """
        gates = []
        for cmd in circuit.get_commands():
            op_type = cmd.op.type
            q1 = cmd.qubits[0].index[0]
            q2 = cmd.qubits[1].index[0] if len(cmd.qubits) > 1 else -1
            angle = cmd.op.params[0] if cmd.op.params else 0

            # Map pytket gate types to our codes
            gate_codes = {
                OpType.H: 0,
                OpType.Rx: 1,
                OpType.Rz: 2,
                OpType.XXPhase: 3,
                OpType.ZZPhase: 4,
                OpType.SWAP: 5,
                OpType.T: 6,
                OpType.CZ: 7
            }

            gates.append([gate_codes[op_type], q1, q2, float(angle)])

        return gates
]]></ac:plain-text-body>
</ac:structured-macro>
<h2>6.3 Stage 3: Kernel Matrix Construction</h2>
<ac:structured-macro ac:name="code">
  <ac:parameter ac:name="title">Kernel Matrix Construction with MPI</ac:parameter>
  <ac:plain-text-body><![CDATA[
═══════════════════════════════════════════════════════════════════════════════
                    PARALLEL KERNEL MATRIX CONSTRUCTION
═══════════════════════════════════════════════════════════════════════════════

from mpi4py import MPI
import numpy as np
from helloitensor import circuit_xyz_exp

def build_kernel_matrix(mpi_comm, ansatz, X, Y=None, alpha=1.0):
    """
    Build the projected quantum kernel matrix using MPI parallelization.

    Parameters:
    -----------
    mpi_comm : MPI communicator
        MPI.COMM_WORLD typically
    ansatz : ProjectedKernelStateAnsatz
        The quantum circuit template
    X : ndarray of shape (N, d)
        Training data
    Y : ndarray of shape (M, d), optional
        Test data. If None, compute K(X, X)
    alpha : float
        Kernel bandwidth parameter

    Returns:
    --------
    Kernel matrix of shape (N, N) or (M, N)
    """

    # MPI setup
    rank = mpi_comm.Get_rank()      # This process's ID
    n_procs = mpi_comm.Get_size()   # Total number of processes
    root = 0                         # Master process

    n_qubits = ansatz.num_features

    # ═══════════════════════════════════════════════════════════════════════
    # PHASE 1: Distribute data and simulate circuits
    # ═══════════════════════════════════════════════════════════════════════

    # Each process handles a chunk of X
    entries_per_chunk = int(np.ceil(len(X) / n_procs))

    # Compute quantum features for this process's chunk
    my_features_X = []
    for i in range(entries_per_chunk):
        global_idx = rank * entries_per_chunk + i
        if global_idx < len(X):
            # Build circuit for this data point
            circuit = ansatz.circuit_for_data(X[global_idx])
            gate_list = ansatz.circuit_to_list(circuit)

            # Simulate and get expectation values
            exp_vals = circuit_xyz_exp(gate_list, n_qubits)
            # exp_vals = [[<X1>, <Y1>, <Z1>], [<X2>, <Y2>, <Z2>], ...]

            my_features_X.append(np.array(exp_vals).flatten())

    # Similarly for Y if provided
    if Y is not None:
        my_features_Y = []  # Compute for Y chunk
        # ... similar code ...

    # ═══════════════════════════════════════════════════════════════════════
    # PHASE 2: Round-robin kernel computation
    # ═══════════════════════════════════════════════════════════════════════

    # Initialize local kernel matrix contribution
    if Y is None:
        kernel_local = np.zeros((len(X), len(X)))
    else:
        kernel_local = np.zeros((len(Y), len(X)))

    # Round-robin: rotate Y features among processes
    for iteration in range(n_procs):
        # Current Y chunk on this process
        y_chunk_idx = (rank + iteration) % n_procs

        # Compute kernel entries for this tile
        for i, phi_x in enumerate(my_features_X):
            x_idx = rank * entries_per_chunk + i

            for j, phi_y in enumerate(current_y_chunk):
                y_idx = y_chunk_idx * entries_per_chunk + j

                # PQK kernel formula
                distance_sq = np.sum((phi_x - phi_y)**2)
                kernel_entry = np.exp(-alpha * distance_sq)

                kernel_local[y_idx, x_idx] = kernel_entry

        # Pass Y chunk to next process in ring
        current_y_chunk = mpi_comm.sendrecv(
            current_y_chunk,
            dest=(rank - 1) % n_procs
        )

    # ═══════════════════════════════════════════════════════════════════════
    # PHASE 3: Gather results at root
    # ═══════════════════════════════════════════════════════════════════════

    # Sum all local contributions
    kernel_matrix = mpi_comm.reduce(kernel_local, op=MPI.SUM, root=root)

    return kernel_matrix

# ═══════════════════════════════════════════════════════════════════════════
# USAGE EXAMPLE
# ═══════════════════════════════════════════════════════════════════════════

mpi_comm = MPI.COMM_WORLD

# Create ansatz
entanglement_map = entanglement_graph(num_features, 1)
ansatz = ProjectedKernelStateAnsatz(
    num_features=12,
    reps=10,
    gamma=1.0,
    entanglement_map=entanglement_map,
    ansatz="hamiltonian"
)

# Build kernel matrices
K_train = build_kernel_matrix(mpi_comm, ansatz, X_train, alpha=0.5)
K_test = build_kernel_matrix(mpi_comm, ansatz, X_train, Y=X_test, alpha=0.5)
]]></ac:plain-text-body>
</ac:structured-macro>
<h2>6.4 Stage 4: Model Training and Evaluation</h2>
<ac:structured-macro ac:name="code">
  <ac:parameter ac:name="title">Model Training Code</ac:parameter>
  <ac:plain-text-body><![CDATA[
═══════════════════════════════════════════════════════════════════════════════
                         MODEL TRAINING AND EVALUATION
═══════════════════════════════════════════════════════════════════════════════

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# ═══════════════════════════════════════════════════════════════════════════
# SVM WITH PRECOMPUTED KERNEL
# ═══════════════════════════════════════════════════════════════════════════

# Grid search over regularization parameter C
C_values = [2, 1.5, 1, 0.5, 0.1, 0.05, 0.01]
best_f1 = 0
best_model = None

for C in C_values:
    # Train SVM with our precomputed kernel
    svm = SVC(
        kernel="precomputed",  # Use our kernel matrix directly
        C=C,                   # Regularization parameter
        tol=1e-5,              # Convergence tolerance
        class_weight='balanced'  # Handle any remaining imbalance
    )

    # Fit on training kernel
    svm.fit(K_train, train_labels)

    # Predict on test kernel
    predictions = svm.predict(K_test)

    # Evaluate
    f1 = f1_score(test_labels, predictions)

    if f1 > best_f1:
        best_f1 = f1
        best_model = svm
        best_C = C

# ═══════════════════════════════════════════════════════════════════════════
# COMPUTE ALL METRICS
# ═══════════════════════════════════════════════════════════════════════════

predictions = best_model.predict(K_test)

results = {
    'Accuracy': accuracy_score(test_labels, predictions),
    'Precision': precision_score(test_labels, predictions, pos_label=0),
    'Recall': recall_score(test_labels, predictions, pos_label=0),
    'F1': f1_score(test_labels, predictions, pos_label=0),
    'AUC': roc_auc_score(test_labels, predictions)
}

print("═══════════════════════════════════════════════════")
print("              MODEL EVALUATION RESULTS             ")
print("═══════════════════════════════════════════════════")
for metric, value in results.items():
    print(f"{metric:15s}: {value:.4f}")

# ═══════════════════════════════════════════════════════════════════════════
# COMPARE WITH CLASSICAL RBF BASELINE
# ═══════════════════════════════════════════════════════════════════════════

# Classical RBF SVM for comparison
svm_rbf = SVC(kernel='rbf', C=best_C, gamma='scale')
svm_rbf.fit(X_train_scaled, train_labels)
rbf_predictions = svm_rbf.predict(X_test_scaled)

print("\n═══════════════════════════════════════════════════")
print("         COMPARISON: PQK vs RBF Kernel             ")
print("═══════════════════════════════════════════════════")
print(f"PQK Accuracy:  {results['Accuracy']:.4f}")
print(f"RBF Accuracy:  {accuracy_score(test_labels, rbf_predictions):.4f}")
print(f"Improvement:   {results['Accuracy'] - accuracy_score(test_labels, rbf_predictions):.4f}")
]]></ac:plain-text-body>
</ac:structured-macro>
<hr />
<h1>7. Parameter Reference Guide</h1>
<h2>7.1 Quantum Circuit Parameters</h2>
<table>
  <tbody>
    <tr>
      <th>Parameter</th>
      <th>Symbol</th>
      <th>Type</th>
      <th>Recommended Range</th>
      <th>Effect</th>
    </tr>
    <tr>
      <td>num_features</td>
      <td>n</td>
      <td>int</td>
      <td>5-50</td>
      <td>More qubits = more expressive but slower</td>
    </tr>
    <tr>
      <td>reps</td>
      <td>r</td>
      <td>int</td>
      <td>2-10</td>
      <td>More reps = deeper circuit, more entanglement</td>
    </tr>
    <tr>
      <td>gamma</td>
      <td>γ</td>
      <td>float</td>
      <td>0.1-1.0</td>
      <td>Smaller for more qubits; controls rotation magnitude</td>
    </tr>
    <tr>
      <td>alpha</td>
      <td>α</td>
      <td>float</td>
      <td>0.1-2.0</td>
      <td>Kernel bandwidth; like γ in classical RBF</td>
    </tr>
    <tr>
      <td>ansatz</td>
      <td>-</td>
      <td>str</td>
      <td>&quot;hamiltonian&quot;</td>
      <td>Circuit type; hamiltonian recommended for continuous data</td>
    </tr>
  </tbody>
</table>
<h2>7.2 Parameter Selection Guidelines</h2>
<ac:structured-macro ac:name="code">
  <ac:parameter ac:name="title">Parameter Selection Decision Tree</ac:parameter>
  <ac:plain-text-body><![CDATA[
HOW TO CHOOSE PARAMETERS:

1. NUMBER OF FEATURES (num_features):
   Start: 10-15
   Increase if: Model underfits, need more expressiveness
   Decrease if: Simulation too slow, overfitting

2. CIRCUIT DEPTH (reps):
   Start: 5
   Increase if: Kernel values all close to 1 (underfitting)
   Decrease if: Kernel values all close to 0 (overfitting to noise)

3. ROTATION SCALING (gamma):
   Start: 1.0 for n < 10, 0.5 for n < 20, 0.3 for n ≥ 20
   Increase if: Kernel values too uniform
   Decrease if: Kernel values too variable

4. KERNEL BANDWIDTH (alpha):
   Start: 0.5
   Increase if: Model too smooth (underfitting)
   Decrease if: Model too sharp (overfitting)

5. SVM REGULARIZATION (C):
   Grid search: [0.01, 0.05, 0.1, 0.5, 1.0, 1.5, 2.0]
   Higher C: More complex boundary, risk of overfitting
   Lower C: Simpler boundary, risk of underfitting
]]></ac:plain-text-body>
</ac:structured-macro>
<h2>7.3 Common Configurations</h2>
<table>
  <tbody>
    <tr>
      <th>Use Case</th>
      <th>num_features</th>
      <th>reps</th>
      <th>gamma</th>
      <th>alpha</th>
      <th>C</th>
    </tr>
    <tr>
      <td>Quick test</td>
      <td>8</td>
      <td>2</td>
      <td>1.0</td>
      <td>0.5</td>
      <td>1.0</td>
    </tr>
    <tr>
      <td>Balanced accuracy</td>
      <td>12</td>
      <td>5</td>
      <td>1.0</td>
      <td>0.5</td>
      <td>0.5</td>
    </tr>
    <tr>
      <td>High precision</td>
      <td>12</td>
      <td>10</td>
      <td>0.8</td>
      <td>0.3</td>
      <td>1.0</td>
    </tr>
    <tr>
      <td>Many features</td>
      <td>20</td>
      <td>10</td>
      <td>0.3</td>
      <td>0.1</td>
      <td>0.1</td>
    </tr>
  </tbody>
</table>
<hr />
<h1>8. Fraud Detection Specific Guidance</h1>
<h2>8.1 The Elliptic Bitcoin Dataset</h2>
<ac:structured-macro ac:name="code">
  <ac:parameter ac:name="title">Dataset Summary</ac:parameter>
  <ac:plain-text-body><![CDATA[
ELLIPTIC BITCOIN DATASET

Source: Anti-money laundering research
Type: Bitcoin transaction graph
Task: Binary classification (fraud vs legitimate)

STATISTICS:
- Total transactions: 203,769
- Labeled: 46,564 (23%)
- Unlabeled: 157,205 (77%)
- Fraud (Class 0): 4,545 (9.8% of labeled)
- Legitimate (Class 1): 42,019 (90.2% of labeled)
- Features: 166 per transaction
- Time steps: 49

FEATURES (166 total):
- Local features (94): Transaction characteristics
  - Input/output counts
  - Transaction fees
  - Value amounts

- Aggregated features (72): Neighborhood statistics
  - Mean/std of neighbor features
  - Graph structure features
]]></ac:plain-text-body>
</ac:structured-macro>
<h2>8.2 Recommended Configuration for Elliptic</h2>
<ac:structured-macro ac:name="code">
  <ac:parameter ac:name="title">Optimal Parameters for Elliptic Dataset</ac:parameter>
  <ac:plain-text-body><![CDATA[
# Tested configuration achieving ~90% F1 score

# Data parameters
n_fraud = 100       # Balanced sampling
n_legit = 100
test_frac = 0.2
data_seed = 456     # For reproducibility

# Feature selection
num_features = 12   # Top 12 features

# Circuit parameters
reps = 10           # Deep circuit
gamma = 1.0         # Full rotation range
ansatz = "hamiltonian"

# Kernel parameters
alpha = 0.5

# SVM parameters
C = 0.5             # Moderate regularization
]]></ac:plain-text-body>
</ac:structured-macro>
<h2>8.3 Evaluation Metrics for Fraud Detection</h2>
<ac:structured-macro ac:name="code">
  <ac:parameter ac:name="title">Metrics Interpretation</ac:parameter>
  <ac:plain-text-body><![CDATA[
FRAUD DETECTION METRICS GUIDE

For fraud detection, different metrics matter differently:

RECALL (Sensitivity, True Positive Rate):
- Definition: TP / (TP + FN)
- Meaning: What fraction of actual frauds did we catch?
- Target: > 80%
- Why important: Missing fraud means direct financial loss

PRECISION (Positive Predictive Value):
- Definition: TP / (TP + FP)
- Meaning: What fraction of our fraud alerts are real fraud?
- Target: > 70%
- Why important: False alerts waste investigation resources

F1 SCORE:
- Definition: 2 * (Precision * Recall) / (Precision + Recall)
- Meaning: Harmonic mean of precision and recall
- Target: > 75%
- Why important: Balances both concerns

AUC-ROC:
- Definition: Area under Receiver Operating Characteristic curve
- Meaning: Probability that random fraud ranks higher than random legitimate
- Target: > 85%
- Why important: Measures discrimination ability across all thresholds

CONFUSION MATRIX INTERPRETATION:
                    Predicted
                  Fraud   Legit
              ┌─────────┬─────────┐
Actual Fraud  │   TP    │   FN    │  ← We want to minimize FN (costly!)
              │ (Good!) │ (Bad!)  │
              ├─────────┼─────────┤
Actual Legit  │   FP    │   TN    │  ← FP is annoying but not as costly
              │(Annoying│ (Good!) │
              └─────────┴─────────┘
]]></ac:plain-text-body>
</ac:structured-macro>
<hr />
<h1>9. Deployment and Operations</h1>
<h2>9.1 Running the Pipeline</h2>
<ac:structured-macro ac:name="code">
  <ac:parameter ac:name="title">Command Line Usage</ac:parameter>
  <ac:plain-text-body><![CDATA[
═══════════════════════════════════════════════════════════════════════════════
                         RUNNING THE PIPELINE
═══════════════════════════════════════════════════════════════════════════════

# Basic execution with MPI
mpirun -n <num_processes> python main.py \
    <num_features> \
    <reps> \
    <gamma> \
    <alpha> \
    <n_illicit> \
    <n_licit> \
    <data_seed> \
    <data_file>

# Example: 4 processes, 12 features, 10 reps
mpirun -n 4 python main.py 12 10 1 0.5 100 100 456 elliptic_preproc.csv

# On a cluster with job scheduler (SLURM example)
srun --ntasks=16 --cpus-per-task=4 python main.py 12 10 1 0.5 100 100 456 data.csv
]]></ac:plain-text-body>
</ac:structured-macro>
<h2>9.2 Docker Deployment</h2>
<ac:structured-macro ac:name="code">
  <ac:parameter ac:name="title">Docker Commands</ac:parameter>
  <ac:plain-text-body><![CDATA[
# Build container
cd Installation-Script
docker build -t qiml .

# Run with environment variables
docker run \
   --env MPI_NODES=4 \
   --env NUM_FEATURES=12 \
   --env REPS=10 \
   --env GAMMA=1 \
   --env ALPHA=0.5 \
   --env N_ILLICIT=100 \
   --env N_LICIT=100 \
   --env DATA_SEED=456 \
   --env DATA_FILE=elliptic_preproc.csv \
   qiml
]]></ac:plain-text-body>
</ac:structured-macro>
<h2>9.3 Output Files</h2>
<table>
  <tbody>
    <tr>
      <th>File</th>
      <th>Location</th>
      <th>Description</th>
    </tr>
    <tr>
      <td>Training kernel</td>
      <td>kernels/TrainKernel_Nf-{n}_r-{r}_g-{γ}_Ntr-{N}.npy</td>
      <td>N×N kernel matrix</td>
    </tr>
    <tr>
      <td>Test kernel</td>
      <td>kernels/TestKernel_Nf-{n}_r-{r}_g-{γ}_Ntr-{N}.npy</td>
      <td>M×N kernel matrix</td>
    </tr>
    <tr>
      <td>Profiling</td>
      <td>{info_file}.json</td>
      <td>Timing information</td>
    </tr>
    <tr>
      <td>Model</td>
      <td>model/*.pkl</td>
      <td>Trained classifier</td>
    </tr>
    <tr>
      <td>Scaler</td>
      <td>model/scaler.pkl</td>
      <td>Fitted data scaler</td>
    </tr>
  </tbody>
</table>
<hr />
<h1>10. References and Further Reading</h1>
<h2>10.1 Core Papers</h2>
<ol>
  <li><strong>Huang, H.-Y. et al.</strong> (2021). &quot;Power of data in quantum machine learning.&quot; _Nature Communications_, 12, 2631. [Link|https://www.nature.com/articles/s41467-021-22539-9]</li>
</ol>
<ul>
  <li>Introduced projected quantum kernels</li>
  <li>Theoretical foundation for our approach</li>
</ul>
<ol>
  <li><strong>Havlíček, V. et al.</strong> (2019). &quot;Supervised learning with quantum-enhanced feature spaces.&quot; _Nature_, 567, 209-212. [Link|https://www.nature.com/articles/s41586-019-0980-2]</li>
</ol>
<ul>
  <li>Original quantum feature map paper</li>
  <li>ZZ feature map design</li>
</ul>
<ol>
  <li><strong>Fishman, M., White, S. R., &amp;amp; Stoudenmire, E. M.</strong> (2022). &quot;The ITensor Software Library.&quot; _SciPost Physics Codebases_, 4. [Link|https://scipost.org/SciPostPhysCodeb.4]</li>
</ol>
<ul>
  <li>ITensor library documentation</li>
  <li>MPS algorithms</li>
</ul>
<h2>10.2 Fraud Detection Applications</h2>
<ol>
  <li><strong>Heredge, J. et al.</strong> (2023). &quot;Quantum Multiple Kernel Learning in Financial Classification Tasks.&quot; [arXiv:2312.00260|https://arxiv.org/abs/2312.00260]</li>
</ol>
<ol>
  <li><strong>Vasquez, A.C. et al.</strong> (2023). &quot;Financial Fraud Detection: A Comparative Study of Quantum Machine Learning Models.&quot; [arXiv:2308.05237|https://arxiv.org/abs/2308.05237]</li>
</ol>
<h2>10.3 Additional Resources</h2>
<ul>
  <li>[ITensor Documentation|https://itensor.org/docs.cgi]</li>
  <li>[pytket Documentation|https://tket.quantinuum.com/api-docs/]</li>
  <li>[PennyLane Kernel Tutorial|https://pennylane.ai/qml/demos/tutorial_kernel_based_training]</li>
</ul>
<hr />
<h1>Appendix A: Quick Reference</h1>
<h2>A.1 Key Formulas</h2>
<ac:structured-macro ac:name="code">
  <ac:parameter ac:name="title">Essential Equations</ac:parameter>
  <ac:plain-text-body><![CDATA[
PROJECTED QUANTUM KERNEL:
k(x, x') = exp(-α × Σᵢ [(⟨Xᵢ⟩ˣ - ⟨Xᵢ⟩ˣ')² + (⟨Yᵢ⟩ˣ - ⟨Yᵢ⟩ˣ')² + (⟨Zᵢ⟩ˣ - ⟨Zᵢ⟩ˣ')²])

HAMILTONIAN ANSATZ:
U(x) = H⊗ⁿ × ∏ᵣ [∏ᵢ Rz(γxᵢ/π) × ∏_{(i,j)} R_XX(γ²(1-xᵢ)(1-xⱼ))]

FEATURE VECTOR:
Φ(x) = [⟨X₁⟩, ⟨Y₁⟩, ⟨Z₁⟩, ..., ⟨Xₙ⟩, ⟨Yₙ⟩, ⟨Zₙ⟩] ∈ ℝ³ⁿ
]]></ac:plain-text-body>
</ac:structured-macro>
<h2>A.2 Parameter Quick Guide</h2>
<table>
  <tbody>
    <tr>
      <th>Parameter</th>
      <th>Quick Setting</th>
    </tr>
    <tr>
      <td>num_features</td>
      <td>12</td>
    </tr>
    <tr>
      <td>reps</td>
      <td>10</td>
    </tr>
    <tr>
      <td>gamma</td>
      <td>1.0</td>
    </tr>
    <tr>
      <td>alpha</td>
      <td>0.5</td>
    </tr>
    <tr>
      <td>C</td>
      <td>0.5</td>
    </tr>
  </tbody>
</table>
<h2>A.3 Expected Performance</h2>
<table>
  <tbody>
    <tr>
      <th>Metric</th>
      <th>Target</th>
    </tr>
    <tr>
      <td>Accuracy</td>
      <td>85-92%</td>
    </tr>
    <tr>
      <td>Recall</td>
      <td>82-90%</td>
    </tr>
    <tr>
      <td>F1 Score</td>
      <td>81-89%</td>
    </tr>
    <tr>
      <td>Inference</td>
      <td>&lt;200ms</td>
    </tr>
  </tbody>
</table>
<hr />
<ac:structured-macro ac:name="panel">
  <ac:parameter ac:name="title">Document Information</ac:parameter>
  <ac:parameter ac:name="borderStyle">solid</ac:parameter>
  <ac:parameter ac:name="borderColor">#ccc</ac:parameter>
  <ac:parameter ac:name="titleBGColor">#f0f0f0</ac:parameter>
  <ac:rich-text-body>
<p><strong>Version:</strong> 1.0 <strong>Last Updated:</strong> January 2026 <strong>Authors:</strong> Enterprise Quantum Engineering Team <strong>Classification:</strong> Internal Technical Documentation</p>
  </ac:rich-text-body>
</ac:structured-macro>
</div>
