svd_cutoff — float (≥0)

Meaning: Threshold below which singular values are discarded in SVD truncations (used in MPS / MPO / tensor compressions).

Effect: Controls numerical compression: smaller svd_cutoff → keep more singular values → more accurate but heavier memory & compute; larger → more truncation (lossy) but faster. Important when contracting or compressing tensor networks. 
link.aps.org

Typical tuning: log-uniform sample in [1e-12, 1e-2]. Good starting default 1e-6.

Search strategy: log-uniform (because scale spans many orders).

max_dim (a.k.a. max bond dimension, χ) — int

Meaning: Upper limit on the bond dimension / Schmidt rank kept in tensor network (MPS) or limit on internal tensor sizes.

Effect: Governs representational capacity (how much entanglement you can capture). Larger = more expressive but cost ~ O(χ^3) or worse depending on ops. If too small, you underfit / miss correlations. 
link.aps.org

Typical tuning: choose from powers of two or small set: [8,16,32,64,128,256]. If GPU/memory limited, cap accordingly.

Search strategy: categorical over candidate values or log-uniform integer (but prefer categorical/powers-of-two).

reps — int (repetitions / layers)

Meaning: Number of repeated blocks / layers in the ansatz (either tensor network sweeps or repeated parameterized circuit layers).

Effect: More reps → higher capacity (but deeper circuits → harder to train and noisier on hardware). Tradeoff: performance vs trainability / overfitting. 
arXiv

Typical tuning: integer uniform 1..8 (start 1–4).

Search strategy: discrete uniform.

gamma — float (interpretation depends on context — two common meanings)

Meaning (A — circuit parameter): a rotation / entangling gate angle in parameterized quantum circuits; natural range [-π, π] or [0, π].

Meaning (B — penalty / regularizer / annealing rate): penalty weight (e.g., for constraints or loss term) or annealing rate scalar. Positive real — often spans many orders.

Effect:

If angle, influences expressivity and search landscape (periodic behavior). Sample from [0, π] or [-π, π].

If penalty/scale, it balances constraint enforcement (too high → dominates objective; too low → constraint violations).

Typical tuning:

angle: uniform [0, π] (or [-π,π]).

penalty/scale: log-uniform in [1e-4, 1e2]. Default often around 1e-2 .. 1e1 depending on problem scaling.

Search strategy: conditional on gamma_type (angle vs penalty vs lr).

entanglement_pattern / nearest_neighbor_for_non_skip_entanglement — categorical

Meaning: Which qubit/tensor connectivity to use when applying entangling gates / link tensors. E.g. nearest_neighbor vs skip vs all_to_all. (Your “nearest neighbour for non-skip entanglement” implies: when you are not using skip entanglement, use NN.) 
pennylane.ai
+1

Effect: Determines reachable correlations and trainability. NN is shallow and hardware-friendly; skip distances or long-range increase entanglement but may increase depth / swaps / noise. 
IBM

Typical options: ['nearest_neighbor', 'skip', 'all_to_all'].

Search strategy: categorical.

skip_level — boolean

Meaning: If true, add skip-distance entangling connections (i ↔ i+skip) in ansatz / entanglement pattern; if false, only the base pattern (usually NN) is used.

Effect: Enables long-range entanglement with minimal extra layers if used carefully. Must be paired with skip_distances. 
IBM

Search strategy: categorical [True, False]; conditional sampling for skip_distances when True.

skip_distances — array / categorical of ints (allowed values: [1,2,3,4])

Meaning: Which distances to use for skip entanglers (distance=1 is NN, 2 is next-nearest, etc.).

Effect: Multiple distances allow capturing correlations at multiple scales; combining them increases gates/depth.

Typical options: try single values (1,2,3,4) or small combos like [1,2], [1,3], or full [1,2,3,4].

Search strategy: if tuning combinatorially, prefer categorical choices over a curated set (e.g., ['[1]','[2]','[1,2]','[1,2,3]','[1,2,3,4]']) instead of exploring full powerset.

multi_distance — categorical (your provided semantics)

Meaning: Overall mode controlling entanglement composition:

just_nn — use only nearest-neighbour;

just_skip_level — use only skip entanglement distances (from skip_distances);

all_values_defined — combine NN and skip distances.

Effect: shorthand that controls how the code constructs entanglers.

Search strategy: categorical with the three options.

boev. (embedding/solver-related)* — grouped object with recommended tunables:

boev.use_boev_transform — bool — whether to use the Boev embedding (True/False). Default True. 
Frontiers

boev.lambda — float (Lagrange multiplier / penalty weight) — log-uniform [1e-4, 1e2]. Boev uses Lagrange multipliers to reduce hyperparameters; tune to control constraint strictness. 
Frontiers

boev.simcim_iters — int — SimCIM iterations / steps; e.g., [100, 5000]. 
arXiv

boev.simcim_noise — float — amplitude of stochastic noise in SimCIM dynamics — log-uniform [1e-6, 1e-1]. 
arXiv

boev.simcim_schedule — categorical/object — annealing/gain schedule (e.g., linear, exponential) — choose from ['linear', 'exponential', 'custom']. 
arXiv

boev.seed — int — RNG seed for reproducibility.

Feature Type	Suggested Gate	Why
Continuous numerical	RX (primary), RY (secondary)	Controls amplitude → predictive
Binary 0/1	RX(π·x) or RZ	Boolean-like rotation
Ordinal	RZ or RX	Ordered → phase or amplitude
Categorical (small cardinality)	RZ	Smooth mapping of class index
High-cardinality categorical	Amplitude encoding or embedding + RZ/RX	Dimensionality reduction
Periodic time features	RZ	Natural phase behavior
Velocity / deltas	RY	Good for directional behaviour
Normalised engineered score	RX	Probability-aligned
