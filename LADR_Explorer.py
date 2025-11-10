#!/usr/bin/env python3
"""
LADR_Explorer.py - Linear Algebra Done Right Interactive Study Companion

Dependencies:
    - numpy: pip install numpy
    - rich: pip install rich (optional, for prettier output)

This script provides an interactive menu-driven interface for exploring concepts
and exercises from "Linear Algebra Done Right" (3rd Edition) by Sheldon Axler.

Usage:
    python LADR_Explorer.py

Features:
    - Easy numbered menu navigation (no typing topic names!)
    - 16 core linear algebra concepts with detailed explanations
    - Python/NumPy code examples with syntax highlighting
    - Visualization suggestions for understanding
    - 11 exercises with helpful hints
    - Beautiful mathematical notation with rich formatting

Simply run the program and select options by number.
"""

import numpy as np
import sys

# Try to import rich for pretty output, fall back to plain text if not available
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.syntax import Syntax
    from rich.markdown import Markdown
    from rich.text import Text
    RICH_AVAILABLE = True
    console = Console()
except ImportError:
    RICH_AVAILABLE = False
    console = None


# ============================================================================
# KNOWLEDGE BASE: CONCEPTS
# ============================================================================

CONCEPTS = {
    "linear independence": {
        "explanation": """In Axler's "Linear Algebra Done Right", linear independence is a fundamental concept for describing a list of vectors with no redundancy. A list of vectors v₁, v₂, ..., vₘ in V is linearly independent if the only choice of scalars a₁, a₂, ..., aₘ that makes a₁v₁ + a₂v₂ + ... + aₘvₘ = 0 is a₁ = a₂ = ... = aₘ = 0.

This means that no vector in the list can be written as a linear combination of the others. It is the formal way of saying that the list is as efficient as possible, containing no unnecessary information. This is formally defined in Definition 2.17 on page 34. Linear independence is the conceptual counterpart to 'span', and a list that is both linearly independent and spans a space is called a 'basis'. Axler emphasizes understanding this property structurally rather than through determinants.""",

        "python_example": """import numpy as np

# A linearly INDEPENDENT list of vectors in R^3
independent_vectors = [
    np.array([1, 0, 0]),
    np.array([0, 1, 0]),
    np.array([0, 0, 1])
]

# A linearly DEPENDENT list of vectors in R^3 (v3 = v1 + v2)
dependent_vectors = [
    np.array([1, 2, 3]),
    np.array([4, 5, 6]),
    np.array([5, 7, 9])  # This is [1,2,3] + [4,5,6]
]

# A list of vectors is linearly independent if the rank of the matrix
# formed by them equals the number of vectors.
rank_independent = np.linalg.matrix_rank(np.column_stack(independent_vectors))
print(f"List 1: {len(independent_vectors)} vectors. Matrix rank: {rank_independent}.")
print(f"Result: The list is linearly independent.\\n")

rank_dependent = np.linalg.matrix_rank(np.column_stack(dependent_vectors))
print(f"List 2: {len(dependent_vectors)} vectors. Matrix rank: {rank_dependent}.")
print(f"Result: The list is linearly dependent.")""",

        "visualization": """In Desmos, define two linearly independent vectors in R^2, like u=(1,2) and v=(-1,1).
Plot the point P = (a*1 + b*(-1), a*2 + b*1). Create sliders for 'a' and 'b' ranging from -5 to 5.
As you move the sliders, you'll see that P can cover the entire 2D plane.

Now, try with two linearly dependent vectors, like u=(1,2) and w=(2,4).
The point P = (a*1 + b*2, a*2 + b*4) will only ever trace out a single line through the origin,
visually demonstrating the redundancy. This shows that dependent vectors don't add new "dimensions" to the span."""
    },

    "span": {
        "explanation": """In Axler's framework, the span of a list of vectors is the set of all possible linear combinations of those vectors. Given vectors v₁, v₂, ..., vₘ in V, their span is span(v₁, ..., vₘ) = {a₁v₁ + a₂v₂ + ... + aₘvₘ : a₁, ..., aₘ ∈ F}, where F is the underlying field.

The span is always a subspace of V (Definition 2.5 on page 20). A key insight from Axler is that span is about "reachability": it tells you which vectors you can construct from your given list. A list spans V if every vector in V can be written as a linear combination of the list. This concept is intimately connected to the range of a linear map and forms half of the definition of a basis (the other half being linear independence).""",

        "python_example": """import numpy as np

# Define two vectors in R^3
v1 = np.array([1, 0, 0])
v2 = np.array([0, 1, 0])

# These two vectors span a 2D plane (the xy-plane) in R^3
# We can create any vector in this plane as a linear combination

# Example: Create several vectors in the span
a1, b1 = 2, 3
vector_in_span_1 = a1 * v1 + b1 * v2
print(f"{a1}*v1 + {b1}*v2 = {vector_in_span_1}")

a2, b2 = -1, 4
vector_in_span_2 = a2 * v1 + b2 * v2
print(f"{a2}*v1 + {b2}*v2 = {vector_in_span_2}")

# The vector [0, 0, 1] is NOT in span(v1, v2)
v3 = np.array([0, 0, 1])
print(f"\\nVector {v3} is NOT in span(v1, v2) because it has a non-zero z-component.")

# Check dimension of span using rank
span_dimension = np.linalg.matrix_rank(np.column_stack([v1, v2]))
print(f"\\nThe dimension of span(v1, v2) is {span_dimension}.")""",

        "visualization": """In Desmos 3D (or using Matplotlib), visualize span(u, v) where u=(1,0,1) and v=(0,1,1) in R^3.
Set up parametric equations: x = a, y = b, z = a + b (where a and b are parameters).
This creates a plane through the origin. Any point on this plane is in the span.

For a 2D version, plot vectors u=(2,1) and v=(1,3) from the origin. The span is the entire R^2 plane
since these are linearly independent. You can show this by plotting au + bv with sliders and noting
that every point in the plane is reachable."""
    },

    "basis": {
        "explanation": """A basis of a vector space V is a list of vectors that is both linearly independent and spans V (Definition 2.27 on page 40). This is one of the most important concepts in linear algebra. Axler emphasizes that a basis provides a "coordinate system" for the vector space - every vector can be uniquely written as a linear combination of the basis vectors.

Remarkably, every basis of a finite-dimensional vector space has the same length (Theorem 2.35 on page 42), and this length is called the dimension of V. Axler's approach highlights that bases are not unique (there are infinitely many for most spaces), but they all share this common length property. Understanding bases is crucial for everything that follows in the book, from linear maps to eigenvalues.""",

        "python_example": """import numpy as np

# Standard basis for R^3
e1 = np.array([1, 0, 0])
e2 = np.array([0, 1, 0])
e3 = np.array([0, 0, 1])
standard_basis = [e1, e2, e3]

print("Standard basis for R^3:", standard_basis)

# Any vector can be written uniquely as a linear combination of basis vectors
v = np.array([5, -2, 7])
print(f"\\nVector v = {v}")
print(f"v = 5*e1 + (-2)*e2 + 7*e3")
print(f"Verification: {5*e1 + (-2)*e2 + 7*e3}")

# A different basis for R^3
b1 = np.array([1, 1, 0])
b2 = np.array([1, 0, 1])
b3 = np.array([0, 1, 1])
alternate_basis = [b1, b2, b3]

# Check that it's linearly independent (rank = 3)
rank = np.linalg.matrix_rank(np.column_stack(alternate_basis))
print(f"\\nAlternate basis: rank = {rank}, so it IS a basis for R^3.")

# Express v in the alternate basis (requires solving a system)
B = np.column_stack(alternate_basis)
coords = np.linalg.solve(B, v)
print(f"v in alternate basis coordinates: {coords}")
print(f"Verification: {coords[0]}*b1 + {coords[1]}*b2 + {coords[2]}*b3 = {coords[0]*b1 + coords[1]*b2 + coords[2]*b3}")""",

        "visualization": """In R^2, plot the standard basis {(1,0), (0,1)} as red arrows from the origin.
Then plot an alternate basis like {(1,1), (-1,1)} as blue arrows.
Choose a vector v=(3,2) and show it can be expressed in either basis:
- Standard: v = 3*(1,0) + 2*(0,1)
- Alternate: v = 2.5*(1,1) + (-0.5)*(-1,1)

Draw the parallelograms showing these linear combinations to visualize how the same vector
has different coordinate representations depending on the chosen basis."""
    },

    "dimension": {
        "explanation": """The dimension of a finite-dimensional vector space V is the length of any basis of V (Definition 2.36 on page 42). Axler proves that all bases of V have the same length (Theorem 2.35), making this definition well-defined and unambiguous.

Dimension is a fundamental invariant of a vector space - it tells you the "degrees of freedom" in the space. For example, dim(Rⁿ) = n, and dim(Pₘ(F)) = m+1 where Pₘ(F) is the space of polynomials with degree at most m. Axler's approach emphasizes that dimension is about the minimum number of vectors needed to span the space, or equivalently, the maximum number of linearly independent vectors. This concept is essential for understanding the rank-nullity theorem and many other results.""",

        "python_example": """import numpy as np

# Dimension of R^3 is 3 (any basis has exactly 3 vectors)
print("dim(R^3) = 3")

# For a subspace, the dimension is the rank of a matrix whose columns span it
# Example: The xy-plane in R^3
v1 = np.array([1, 0, 0])
v2 = np.array([0, 1, 0])
xy_plane_basis = np.column_stack([v1, v2])
dim_xy_plane = np.linalg.matrix_rank(xy_plane_basis)
print(f"dim(xy-plane in R^3) = {dim_xy_plane}")

# Example: A line through the origin in R^3
v = np.array([1, 2, 3])
line_basis = v.reshape(-1, 1)
dim_line = np.linalg.matrix_rank(line_basis)
print(f"dim(line through origin) = {dim_line}")

# Example: Dimension of null space
# For matrix A, dim(null(A)) = n - rank(A) (rank-nullity theorem)
A = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
])
n = A.shape[1]  # number of columns
rank_A = np.linalg.matrix_rank(A)
dim_null_A = n - rank_A
print(f"\\nFor a 3x3 matrix with rank {rank_A}:")
print(f"dim(null(A)) = {n} - {rank_A} = {dim_null_A}")""",

        "visualization": """Visualize dimension hierarchically:
- A point (0-dimensional): just the origin
- A line through the origin (1-dimensional): needs 1 basis vector
- A plane through the origin (2-dimensional): needs 2 basis vectors
- All of R^3 (3-dimensional): needs 3 basis vectors

In Desmos 3D, you can plot these nested subspaces to see how dimension increases.
Start with the origin, extend to a line (scalar multiples of one vector),
then to a plane (linear combinations of two vectors), then fill all of 3D space."""
    },

    "eigenvalue": {
        "explanation": """In Axler's determinant-free approach, an eigenvalue of a linear operator T ∈ L(V) is a scalar λ such that there exists a non-zero vector v ∈ V with Tv = λv (Definition 5.5 on page 140). The conceptual insight Axler emphasizes is that λ is an eigenvalue if and only if T - λI is not injective, or equivalently, null(T - λI) ≠ {0}.

Axler motivates eigenvalues through invariant subspaces - if λ is an eigenvalue, then null(T - λI) is a non-trivial invariant subspace consisting of all eigenvectors (plus the zero vector). This perspective, focusing on the structure rather than the characteristic polynomial, makes eigenvalues more conceptually transparent. The set of eigenvalues has profound implications for the structure of T and whether it can be diagonalized.""",

        "python_example": """import numpy as np

# Define a 2x2 matrix
A = np.array([
    [4, 1],
    [2, 3]
])

# Find eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(A)

print("Matrix A:")
print(A)
print(f"\\nEigenvalues: {eigenvalues}")
print(f"\\nEigenvectors (as columns):")
print(eigenvectors)

# Verify: For each eigenvalue λ and eigenvector v, check that Av = λv
for i in range(len(eigenvalues)):
    lambda_i = eigenvalues[i]
    v_i = eigenvectors[:, i]

    Av = A @ v_i
    lambda_v = lambda_i * v_i

    print(f"\\nEigenvalue {i+1}: λ = {lambda_i:.4f}")
    print(f"Eigenvector: v = {v_i}")
    print(f"Av = {Av}")
    print(f"λv = {lambda_v}")
    print(f"Av ≈ λv: {np.allclose(Av, lambda_v)}")""",

        "visualization": """In Desmos, visualize eigenvectors as special directions that don't change under transformation.
Set up a 2x2 matrix like A = [[2,1],[0,3]]. This has eigenvalues 2 and 3 with eigenvectors (1,0) and (1,1).

Plot several vectors and their images under A. Most vectors change direction, but:
- Vector (1,0) gets scaled by 2 (stays on x-axis)
- Vector (1,1) gets scaled by 3 (stays on the line y=x)

Use different colors for original vectors (blue) and transformed vectors (red).
The eigenvectors are the special blue arrows whose red images point in the same direction."""
    },

    "eigenvector": {
        "explanation": """An eigenvector of a linear operator T ∈ L(V) corresponding to eigenvalue λ is a non-zero vector v such that Tv = λv (Definition 5.5 on page 140). Axler emphasizes that eigenvectors are special directions that are merely scaled (not rotated or skewed) by the operator T.

The collection of all eigenvectors corresponding to a particular eigenvalue λ, together with the zero vector, forms a subspace called an eigenspace: E(λ, T) = null(T - λI). This is an invariant subspace of T. Eigenvectors corresponding to distinct eigenvalues are linearly independent (Theorem 5.10 on page 143), which is crucial for diagonalization. Finding a basis of eigenvectors (when possible) gives the simplest matrix representation of T.""",

        "python_example": """import numpy as np

# Matrix with clear geometric meaning
A = np.array([
    [3, 0],
    [0, 1]
])

print("Matrix A (scales x by 3, y by 1):")
print(A)

# Find eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(A)

print(f"\\nEigenvalues: {eigenvalues}")

for i in range(len(eigenvalues)):
    lambda_i = eigenvalues[i]
    v_i = eigenvectors[:, i]

    print(f"\\nEigenvector {i+1}: {v_i}")
    print(f"  Corresponding eigenvalue: {lambda_i}")
    print(f"  Geometric meaning: Points along this direction are scaled by {lambda_i}")

    # Show that Av = λv
    Av = A @ v_i
    lambda_v = lambda_i * v_i
    print(f"  A*v = {Av}")
    print(f"  λ*v = {lambda_v}")

# Eigenvectors corresponding to distinct eigenvalues are linearly independent
print(f"\\nThe two eigenvectors form a basis for R^2:")
print(f"Rank of [v1, v2] = {np.linalg.matrix_rank(eigenvectors)}")""",

        "visualization": """Plot a transformation grid in Desmos to see eigenvectors in action.
For matrix A = [[2,1],[1,2]], draw a grid of vectors in R^2.
After applying A, most grid points move to new positions.

The eigenvectors are (1,1) with eigenvalue 3 and (1,-1) with eigenvalue 1.
Draw these as thick arrows. Under transformation:
- Arrow (1,1) stretches to (3,3) - same direction, tripled length
- Arrow (1,-1) stays at (1,-1) - same direction and length

This shows eigenvectors as the "axes of transformation" along which the operator acts simply."""
    },

    "linear map": {
        "explanation": """A linear map (or linear transformation) from vector space V to vector space W is a function T: V → W that preserves addition and scalar multiplication. Specifically, T(u + v) = T(u) + T(v) and T(av) = aT(v) for all u, v ∈ V and all scalars a (Definition 3.3 on page 52).

Axler emphasizes that linear maps are the fundamental objects of study in linear algebra - they are functions that preserve vector space structure. The set of all linear maps from V to W is denoted L(V, W), which is itself a vector space (Theorem 3.7 on page 54). Key properties: T(0) = 0 for any linear map, and T is determined entirely by its action on a basis. Understanding linear maps is essential for everything from solving systems of equations to quantum mechanics.""",

        "python_example": """import numpy as np

# Define a linear map T: R^2 -> R^3 using a matrix
# T(x, y) = (x + 2y, 3x - y, y)
T_matrix = np.array([
    [1, 2],
    [3, -1],
    [0, 1]
])

print("Linear map T: R^2 -> R^3 represented by matrix:")
print(T_matrix)

# Test vectors
u = np.array([1, 2])
v = np.array([3, -1])
scalar = 5

# Verify linearity: T(u + v) = T(u) + T(v)
T_u = T_matrix @ u
T_v = T_matrix @ v
T_sum = T_matrix @ (u + v)

print(f"\\nu = {u}, v = {v}")
print(f"T(u) = {T_u}")
print(f"T(v) = {T_v}")
print(f"T(u + v) = {T_sum}")
print(f"T(u) + T(v) = {T_u + T_v}")
print(f"Additivity satisfied: {np.allclose(T_sum, T_u + T_v)}")

# Verify: T(scalar * u) = scalar * T(u)
T_scaled = T_matrix @ (scalar * u)
scaled_T = scalar * T_u
print(f"\\nT({scalar}*u) = {T_scaled}")
print(f"{scalar}*T(u) = {scaled_T}")
print(f"Homogeneity satisfied: {np.allclose(T_scaled, scaled_T)}")""",

        "visualization": """Visualize a linear map T: R^2 -> R^2 as a transformation of the plane.
Example: T(x,y) = (2x + y, x + 2y) - this is a stretching and rotation.

In Desmos, create a unit square with corners at (0,0), (1,0), (0,1), (1,1).
Apply T to each corner:
- T(0,0) = (0,0)
- T(1,0) = (2,1)
- T(0,1) = (1,2)
- T(1,1) = (3,3)

Draw both the original square (blue) and transformed parallelogram (red).
Linear maps preserve: lines stay lines, origin stays fixed, parallel lines stay parallel."""
    },

    "null space": {
        "explanation": """The null space (or kernel) of a linear map T ∈ L(V, W) is the set of all vectors in V that map to zero in W: null T = {v ∈ V : Tv = 0} (Definition 3.12 on page 57). Axler proves that null T is a subspace of V (Theorem 3.14 on page 58).

The null space captures the "information lost" by T - it consists of all vectors that become indistinguishable from zero after applying T. A linear map is injective if and only if null T = {0} (Theorem 3.16 on page 59). The dimension of the null space, called the nullity, is related to the dimension of the range by the fundamental rank-nullity theorem: dim V = dim null T + dim range T (Theorem 3.22 on page 62). Understanding null spaces is crucial for solving systems of linear equations and understanding invertibility.""",

        "python_example": """import numpy as np

# Define a linear map via matrix A: R^3 -> R^2
# This map "collapses" R^3 onto a plane in R^2
A = np.array([
    [1, 2, 3],
    [2, 4, 6]
])

print("Matrix A:")
print(A)
print("Note: Row 2 = 2 * Row 1, so rank is 1")

# Find the null space using SVD
# null(A) = {x : Ax = 0}
_, _, Vt = np.linalg.svd(A)
# The null space is spanned by the last (n - rank) columns of V
rank = np.linalg.matrix_rank(A)
n = A.shape[1]
nullity = n - rank

print(f"\\nRank of A: {rank}")
print(f"Nullity of A (dimension of null space): {nullity}")

# Basis for null space (last nullity columns of V)
null_basis = Vt[rank:].T
print(f"\\nBasis for null(A):")
print(null_basis)

# Verify: Each basis vector is in null space
for i in range(nullity):
    v = null_basis[:, i]
    Av = A @ v
    print(f"\\nA * (null basis vector {i+1}) = {Av}")
    print(f"Is approximately zero: {np.allclose(Av, 0)}")

# Rank-nullity theorem: dim(V) = dim(null T) + dim(range T)
print(f"\\nRank-Nullity Theorem verification:")
print(f"dim(R^3) = {n} = {nullity} + {rank} = nullity + rank ✓")""",

        "visualization": """For a map T: R^3 -> R^2 given by T(x,y,z) = (x+y, x+y), visualize the null space.
The null space is {(x,y,z) : x+y = 0}, which is a plane through the origin.

In 3D: Plot the plane x + y = 0 (equivalently, y = -x). This plane contains all vectors like:
- (1, -1, 0)
- (2, -2, 5)
- (0, 0, 1)

All these vectors map to (0, 0) under T. The null space shows the "kernel" of lost information.
Vectors not in this plane map to non-zero vectors in R^2."""
    },

    "range": {
        "explanation": """The range (or image) of a linear map T ∈ L(V, W) is the set of all possible outputs: range T = {Tv : v ∈ V} = {w ∈ W : w = Tv for some v ∈ V} (Definition 3.19 on page 60). Axler proves that range T is a subspace of W (Theorem 3.20 on page 60).

The range captures what T can "reach" - all vectors in W that are images of vectors in V. A linear map is surjective if and only if range T = W (Definition 3.52 on page 70). The dimension of the range is called the rank of T. The fundamental rank-nullity theorem states: dim V = dim null T + dim range T (Theorem 3.22 on page 62). This theorem is one of the cornerstones of linear algebra, connecting injectivity, surjectivity, and dimensionality.""",

        "python_example": """import numpy as np

# Define a linear map via matrix A: R^3 -> R^3
A = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
])

print("Matrix A:")
print(A)

# The range is the column space of A
rank = np.linalg.matrix_rank(A)
print(f"\\nRank of A (dimension of range): {rank}")

# Find a basis for the range using column pivoting
# In practice, we can use QR decomposition or just extract linearly independent columns
Q, R = np.linalg.qr(A)
range_basis = Q[:, :rank]

print(f"\\nBasis for range(A) (first {rank} columns of Q from QR decomposition):")
print(range_basis)

# Verify: Any vector Av can be written as a combination of range basis vectors
v = np.array([1, -1, 2])
Av = A @ v
print(f"\\nTest vector v = {v}")
print(f"A*v = {Av}")

# Express Av in terms of range basis
coords = np.linalg.lstsq(range_basis, Av, rcond=None)[0]
reconstructed = range_basis @ coords
print(f"Av expressed in range basis: {coords}")
print(f"Reconstructed: {reconstructed}")
print(f"Match: {np.allclose(Av, reconstructed)}")

# Rank-Nullity Theorem
n = A.shape[1]
nullity = n - rank
print(f"\\nRank-Nullity: dim(R^3) = {n} = {rank} + {nullity} = rank + nullity ✓")""",

        "visualization": """For T: R^3 -> R^3 given by a matrix of rank 2, the range is a 2D plane through the origin.

Example: A = [[1,0,0],[0,1,0],[0,0,0]] (projection onto xy-plane)
The range is the entire xy-plane: {(x,y,0) : x,y ∈ R}

Visualize in 3D:
- Plot the xy-plane (z=0) in blue - this is the range
- Pick any vector v in R^3 and compute Av
- The result always lands on the blue plane
- Vectors in the range are "reachable" by T"""
    },

    "inner product": {
        "explanation": """An inner product on a vector space V is a function that takes two vectors and produces a scalar, denoted ⟨u, v⟩, satisfying: positivity ⟨v, v⟩ ≥ 0 with equality iff v = 0; definiteness; additivity in first slot; homogeneity in first slot; and conjugate symmetry ⟨u, v⟩ = ⟨v, u⟩* (Definition 6.3 on page 164).

Axler emphasizes that inner products provide a notion of length (via ⟨v, v⟩ = ||v||²) and angle. The standard inner product on Rⁿ is the dot product. Inner products allow us to define orthogonality: vectors u and v are orthogonal if ⟨u, v⟩ = 0. This geometric structure is fundamental for understanding orthonormal bases, projections, and the spectral theorem. Inner product spaces have rich geometric properties not present in general vector spaces.""",

        "python_example": """import numpy as np

# Standard inner product on R^n is the dot product
u = np.array([1, 2, 3])
v = np.array([4, -1, 2])

# Compute inner product <u, v>
inner_product = np.dot(u, v)
print(f"u = {u}")
print(f"v = {v}")
print(f"⟨u, v⟩ = {inner_product}")

# Compute norms (lengths) using inner product
norm_u = np.sqrt(np.dot(u, u))
norm_v = np.sqrt(np.dot(v, v))
print(f"\\n||u|| = √⟨u, u⟩ = {norm_u:.4f}")
print(f"||v|| = √⟨v, v⟩ = {norm_v:.4f}")

# Check if vectors are orthogonal
print(f"\\nAre u and v orthogonal? {np.isclose(inner_product, 0)}")

# Example of orthogonal vectors
w1 = np.array([1, 0, 0])
w2 = np.array([0, 1, 0])
print(f"\\nw1 = {w1}, w2 = {w2}")
print(f"⟨w1, w2⟩ = {np.dot(w1, w2)} (orthogonal!)")

# Cauchy-Schwarz inequality: |⟨u, v⟩| ≤ ||u|| ||v||
cs_left = abs(inner_product)
cs_right = norm_u * norm_v
print(f"\\nCauchy-Schwarz: |⟨u, v⟩| = {cs_left:.4f} ≤ {cs_right:.4f} = ||u|| ||v|| ✓")""",

        "visualization": """In R^2, visualize the inner product geometrically: ⟨u, v⟩ = ||u|| ||v|| cos(θ)
where θ is the angle between u and v.

Plot vectors u = (3, 1) and v = (1, 2):
- Draw both from the origin
- The inner product ⟨u, v⟩ = 3*1 + 1*2 = 5
- Calculate: ||u|| ≈ 3.16, ||v|| ≈ 2.24, so cos(θ) = 5/(3.16*2.24) ≈ 0.71
- This gives θ ≈ 45°

Show orthogonal vectors like u = (2, 1) and v = (-1, 2):
⟨u, v⟩ = 2*(-1) + 1*2 = 0, so they meet at 90°."""
    },

    "orthogonality": {
        "explanation": """Two vectors u and v in an inner product space are orthogonal if their inner product is zero: ⟨u, v⟩ = 0 (Definition 6.7 on page 167). Orthogonality is the generalization of perpendicularity to abstract inner product spaces.

Axler develops orthogonality systematically: an orthonormal basis is a basis where all vectors have norm 1 and are pairwise orthogonal (Definition 6.23 on page 179). The Gram-Schmidt procedure (Theorem 6.31 on page 183) constructs orthonormal bases from arbitrary bases. Orthogonal projections onto subspaces (Definition 6.44 on page 194) minimize distance and have the property that u - Pu is orthogonal to the subspace. The orthogonal complement U^⊥ consists of all vectors orthogonal to every vector in U. These concepts are essential for least squares, the spectral theorem, and singular value decomposition.""",

        "python_example": """import numpy as np

# Two orthogonal vectors in R^3
u = np.array([1, 2, -1])
v = np.array([2, -1, 0])

print(f"u = {u}")
print(f"v = {v}")
print(f"⟨u, v⟩ = {np.dot(u, v)}")
print(f"Are u and v orthogonal? {np.isclose(np.dot(u, v), 0)}")

# Create an orthonormal basis using Gram-Schmidt
# Start with a linearly independent set
a1 = np.array([1.0, 1.0, 0.0])
a2 = np.array([1.0, 0.0, 1.0])
a3 = np.array([0.0, 1.0, 1.0])

# Gram-Schmidt process
e1 = a1 / np.linalg.norm(a1)

a2_perp = a2 - np.dot(a2, e1) * e1
e2 = a2_perp / np.linalg.norm(a2_perp)

a3_perp = a3 - np.dot(a3, e1) * e1 - np.dot(a3, e2) * e2
e3 = a3_perp / np.linalg.norm(a3_perp)

print(f"\\nOrthonormal basis after Gram-Schmidt:")
print(f"e1 = {e1} (norm = {np.linalg.norm(e1):.4f})")
print(f"e2 = {e2} (norm = {np.linalg.norm(e2):.4f})")
print(f"e3 = {e3} (norm = {np.linalg.norm(e3):.4f})")

print(f"\\nOrthogonality check:")
print(f"⟨e1, e2⟩ = {np.dot(e1, e2):.6f}")
print(f"⟨e1, e3⟩ = {np.dot(e1, e3):.6f}")
print(f"⟨e2, e3⟩ = {np.dot(e2, e3):.6f}")""",

        "visualization": """In R^2, visualize orthogonal vectors as perpendicular arrows.
Plot u = (3, 0) and v = (0, 2) - these are orthogonal since ⟨u, v⟩ = 0.

For Gram-Schmidt visualization:
Start with non-orthogonal vectors a = (2, 1) and b = (1, 2).
1. Normalize a to get e₁
2. Project b onto e₁: proj = ⟨b, e₁⟩e₁
3. Subtract: b_perp = b - proj
4. Normalize b_perp to get e₂

Animate this process showing how b_perp is constructed to be perpendicular to e₁."""
    },

    "diagonalization": {
        "explanation": """A linear operator T ∈ L(V) is diagonalizable if there exists a basis of V consisting entirely of eigenvectors of T (Definition 5.32 on page 155). When T is diagonalizable with respect to basis v₁, ..., vₙ with corresponding eigenvalues λ₁, ..., λₙ, the matrix of T with respect to this basis is diagonal with diagonal entries λ₁, ..., λₙ.

Axler proves that T is diagonalizable if and only if V = E(λ₁, T) ⊕ ... ⊕ E(λₘ, T) where λ₁, ..., λₘ are the distinct eigenvalues (Theorem 5.41 on page 158). Not all operators are diagonalizable. For operators on complex vector spaces, being diagonalizable is equivalent to being normal (T*T = TT*) for operators on real inner product spaces. Diagonalization simplifies many computations: powers of T, exponentials of T, and analyzing dynamical systems.""",

        "python_example": """import numpy as np

# A diagonalizable matrix
A = np.array([
    [4, -2],
    [1,  1]
])

print("Matrix A:")
print(A)

# Find eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(A)
print(f"\\nEigenvalues: {eigenvalues}")
print(f"Eigenvectors (as columns):\\n{eigenvectors}")

# Diagonalization: A = PDP^(-1) where D is diagonal
P = eigenvectors
D = np.diag(eigenvalues)
P_inv = np.linalg.inv(P)

print(f"\\nP (eigenvector matrix):\\n{P}")
print(f"\\nD (diagonal matrix):\\n{D}")

# Verify A = PDP^(-1)
A_reconstructed = P @ D @ P_inv
print(f"\\nA reconstructed from PDP^(-1):")
print(A_reconstructed)
print(f"\\nMatch: {np.allclose(A, A_reconstructed)}")

# Use diagonalization to compute A^10 efficiently
A_power_10 = P @ np.linalg.matrix_power(D, 10) @ P_inv
print(f"\\nA^10 computed via diagonalization:")
print(A_power_10.astype(int))

# Verify by direct computation
A_power_10_direct = np.linalg.matrix_power(A, 10)
print(f"\\nA^10 computed directly:")
print(A_power_10_direct.astype(int))
print(f"\\nMatch: {np.allclose(A_power_10, A_power_10_direct)}")""",

        "visualization": """Visualize diagonalization as a change of basis that makes T simple.

For T: R^2 -> R^2 with matrix [[3,1],[0,2]]:
- Standard basis: T acts as a shear + stretch (complicated)
- Eigenvector basis {(1,0), (1,1)}: T acts as pure scaling

Plot a unit circle in standard coordinates (blue).
After applying T, it becomes an ellipse (complicated shape).

Now plot the same in eigenvector coordinates (red).
In these coordinates, T just scales each axis independently (simple!).
This is the power of diagonalization."""
    },

    "invertibility": {
        "explanation": """A linear map T ∈ L(V, W) is invertible if there exists a linear map S ∈ L(W, V) such that ST = I_V and TS = I_W, where I denotes the identity map (Definition 3.52 on page 70). The map S is called the inverse of T and is denoted T^(-1).

Axler proves that T is invertible if and only if T is both injective and surjective (Theorem 3.56 on page 71). For finite-dimensional spaces with dim V = dim W, T is invertible iff T is injective iff T is surjective (Corollary 3.57 on page 72). A key result: if T is invertible, then T^(-1) is unique and also invertible. The determinant-free approach focuses on null space and range: T is invertible iff null T = {0} and range T = W.""",

        "python_example": """import numpy as np

# An invertible matrix
A = np.array([
    [2, 1],
    [1, 1]
])

print("Matrix A:")
print(A)

# Check if invertible (determinant ≠ 0, or rank = n)
rank = np.linalg.matrix_rank(A)
n = A.shape[0]
is_invertible = (rank == n)

print(f"\\nRank: {rank}, Size: {n}x{n}")
print(f"Is invertible? {is_invertible}")

if is_invertible:
    # Compute inverse
    A_inv = np.linalg.inv(A)
    print(f"\\nA^(-1):")
    print(A_inv)

    # Verify: A * A^(-1) = I
    product1 = A @ A_inv
    print(f"\\nA * A^(-1):")
    print(product1)
    print(f"Is identity? {np.allclose(product1, np.eye(n))}")

    # Verify: A^(-1) * A = I
    product2 = A_inv @ A
    print(f"\\nA^(-1) * A:")
    print(product2)
    print(f"Is identity? {np.allclose(product2, np.eye(n))}")

# A non-invertible matrix (rank deficient)
B = np.array([
    [1, 2],
    [2, 4]
])

print(f"\\n\\nMatrix B (non-invertible):")
print(B)
rank_B = np.linalg.matrix_rank(B)
print(f"Rank: {rank_B} < {n}, so NOT invertible")
print(f"null(B) ≠ {{0}}, so B is not injective")""",

        "visualization": """Visualize invertibility geometrically in R^2.

Invertible map T(x,y) = (2x+y, x+y):
- Maps the unit square to a parallelogram
- Every point in R^2 has exactly one preimage
- No "collapsing" or "folding" occurs

Non-invertible map S(x,y) = (x+y, x+y):
- Maps the entire plane onto the line y=x
- Many points map to the same output
- "Collapses" 2D onto 1D

Draw before/after grids to show that invertible maps preserve dimension and area."""
    },

    "orthogonal projection": {
        "explanation": """Let U be a finite-dimensional subspace of an inner product space V. The orthogonal projection of V onto U, denoted P_U, is the operator defined by P_U(v) = u where v = u + w with u ∈ U and w ∈ U^⊥ (Definition 6.44 on page 194). Axler proves that every v ∈ V can be uniquely written as v = u + w where u ∈ U and w ∈ U^⊥ (Theorem 6.46 on page 195).

The orthogonal projection P_U has special properties: P_U² = P_U (idempotent), range P_U = U, null P_U = U^⊥, and ||v - P_U(v)|| ≤ ||v - u|| for all u ∈ U with equality iff u = P_U(v) (minimization property). This means P_U(v) is the unique closest point in U to v. Orthogonal projections are fundamental for least squares approximation, Fourier series, and understanding self-adjoint operators.""",

        "python_example": """import numpy as np

# Define a subspace U of R^3 (the xy-plane)
# U is spanned by e1 = (1,0,0) and e2 = (0,1,0)
u_basis = np.array([
    [1, 0],
    [0, 1],
    [0, 0]
])  # Column vectors

# Orthogonal projection onto U: P_U(v) = (v · e1)e1 + (v · e2)e2
def project_onto_U(v, basis):
    '''Project v onto the subspace spanned by columns of basis.'''
    # P = basis @ (basis.T @ basis)^(-1) @ basis.T for general basis
    # For orthonormal basis: P = basis @ basis.T
    # First orthonormalize the basis
    Q, _ = np.linalg.qr(basis)
    return Q @ (Q.T @ v)

# Test vector
v = np.array([3, 4, 5])
print(f"Vector v = {v}")

# Project onto U (xy-plane)
projected = project_onto_U(v, u_basis)
print(f"\\nProjection P_U(v) = {projected}")
print(f"This is v with z-component removed")

# The orthogonal complement component
orthogonal_part = v - projected
print(f"\\nOrthogonal part (v - P_U(v)) = {orthogonal_part}")
print(f"This is perpendicular to U")

# Verify: projected is in U, orthogonal part is in U^⊥
print(f"\\nVerification:")
print(f"⟨projected, orthogonal_part⟩ = {np.dot(projected, orthogonal_part):.6f}")

# Minimization property: P_U(v) is closest point in U to v
# Distance from v to its projection
dist_to_projection = np.linalg.norm(orthogonal_part)
print(f"\\nDistance from v to P_U(v): {dist_to_projection:.4f}")

# Try another point in U
other_u = np.array([1, 1, 0])
dist_to_other = np.linalg.norm(v - other_u)
print(f"Distance from v to another point in U: {dist_to_other:.4f}")
print(f"P_U(v) is closer: {dist_to_projection < dist_to_other}")""",

        "visualization": """In R^3, visualize projection onto the xy-plane (U).

Plot:
1. The xy-plane (blue) - this is U
2. A vector v = (2, 3, 4) from origin (red arrow)
3. Its projection P_U(v) = (2, 3, 0) (green arrow on the plane)
4. The perpendicular component (0, 0, 4) (purple arrow pointing up)

Draw a right angle where the purple arrow meets the plane to show orthogonality.
The green arrow is the closest point on the plane to the tip of the red arrow."""
    }
}


# ============================================================================
# KNOWLEDGE BASE: EXERCISES
# ============================================================================

EXERCISES = {
    "2.A.11": {
        "text": """Prove that if a list of vectors is linearly independent, then no vector in the list is in the span of the other vectors.""",
        "hint": """Assume for contradiction that some vector v_k in the linearly independent list is in the span of the other vectors. This means you can write v_k as a linear combination of v_1, ..., v_(k-1), v_(k+1), ..., v_m. Rearrange this equation to express 0 as a non-trivial linear combination of all the vectors. What does this contradict?"""
    },

    "2.B.3": {
        "text": """Suppose v₁, v₂, v₃, v₄ spans V. Prove that v₁ - v₂, v₂ - v₃, v₃ - v₄, v₄ also spans V.""",
        "hint": """To show that the new list spans V, you need to show that any vector in V (which can be written as a linear combination of v₁, v₂, v₃, v₄) can also be written as a linear combination of v₁ - v₂, v₂ - v₃, v₃ - v₄, v₄. Try working backwards: express each v_i in terms of the new list by adding/subtracting the differences appropriately. For instance, v₄ is already in the new list. What is v₃ in terms of the new list?"""
    },

    "2.C.7": {
        "text": """Prove that every subspace of a finite-dimensional vector space is finite-dimensional.""",
        "hint": """Start with a finite-dimensional vector space V and a subspace U of V. You know dim V = n for some non-negative integer n. Try constructing a basis for U using the following algorithm: pick any non-zero vector in U (if U ≠ {0}), then add vectors one at a time ensuring they remain linearly independent. Why must this process terminate? Use the fact that any linearly independent list in V has length ≤ dim V."""
    },

    "3.A.10": {
        "text": """Suppose T ∈ L(V,W) is invertible. Prove that T⁻¹ is also invertible and (T⁻¹)⁻¹ = T.""",
        "hint": """You know that T has an inverse T⁻¹ satisfying T⁻¹T = I_V and TT⁻¹ = I_W. To show T⁻¹ is invertible, you need to find a linear map S such that T⁻¹S = I_W and ST⁻¹ = I_V. What is the natural candidate for S? Simply verify that T satisfies these properties with respect to T⁻¹."""
    },

    "3.B.9": {
        "text": """Prove that if V is finite-dimensional and T ∈ L(V,W), then null T = {0} if and only if dim range T = dim V.""",
        "hint": """Use the fundamental rank-nullity theorem: dim V = dim null T + dim range T. The forward direction: if null T = {0}, then dim null T = 0, so what must dim range T equal? The reverse direction: if dim range T = dim V, then dim null T = 0 (by rank-nullity), which means null T = {0} (the only subspace with dimension 0)."""
    },

    "5.A.17": {
        "text": """Suppose T ∈ L(V) and there exist non-zero vectors v and w in V such that Tv = 3v and Tw = 4w. Prove that v and w are linearly independent.""",
        "hint": """You know v and w are eigenvectors corresponding to distinct eigenvalues (3 and 4 respectively). Axler proves that eigenvectors corresponding to distinct eigenvalues are linearly independent (Theorem 5.10 on page 143). To prove it directly: suppose av + bw = 0 for some scalars a, b. Apply T to both sides and use the fact that Tv = 3v and Tw = 4w. Then compare with the original equation to show a = b = 0."""
    },

    "5.B.4": {
        "text": """Define T ∈ L(F³) by T(z₁, z₂, z₃) = (z₂, z₃, 0). Find all eigenvalues and eigenvectors of T.""",
        "hint": """An eigenvalue λ satisfies T(z₁, z₂, z₃) = λ(z₁, z₂, z₃) for some non-zero vector (z₁, z₂, z₃). This means (z₂, z₃, 0) = (λz₁, λz₂, λz₃). From the third component: 0 = λz₃. Consider two cases: (1) λ = 0, or (2) z₃ = 0. Work out what constraints this places on z₁ and z₂ in each case. You should find that λ = 0 is the only eigenvalue."""
    },

    "6.A.12": {
        "text": """Prove that if V is a real inner product space and ⟨u, v⟩ = 0 for all u ∈ V, then v = 0.""",
        "hint": """This is a direct application of the definiteness property of inner products. If ⟨u, v⟩ = 0 for ALL u ∈ V, then this must hold for the specific choice u = v. What does ⟨v, v⟩ = 0 imply about v using the definiteness property (which states that ⟨v, v⟩ = 0 if and only if v = 0)?"""
    },

    "6.B.5": {
        "text": """Prove that if e₁, ..., eₘ is an orthonormal list of vectors in V, then ||v||² = |⟨v, e₁⟩|² + ... + |⟨v, eₘ⟩|² for all v ∈ span(e₁, ..., eₘ).""",
        "hint": """Since v ∈ span(e₁, ..., eₘ), you can write v = ⟨v, e₁⟩e₁ + ... + ⟨v, eₘ⟩eₘ (this follows from the orthonormal basis representation). Now compute ||v||² = ⟨v, v⟩ by substituting this expression for v. Use the properties of the inner product (additivity and homogeneity) and the fact that ⟨eᵢ, eⱼ⟩ = 0 when i ≠ j and ⟨eᵢ, eᵢ⟩ = 1. Most terms will vanish due to orthogonality."""
    },

    "7.A.8": {
        "text": """Suppose T ∈ L(V) and U is a subspace of V. Prove that U is invariant under T if and only if P_U TP_U = TP_U, where P_U denotes the orthogonal projection onto U.""",
        "hint": """For the forward direction: assume U is invariant under T (meaning Tu ∈ U for all u ∈ U). For any v ∈ V, note that P_U(v) ∈ U, so T(P_U(v)) ∈ U by invariance. Therefore T(P_U(v)) = P_U(T(P_U(v))). For the reverse direction: assume P_U TP_U = TP_U. To show U is invariant, take any u ∈ U. Then u = P_U(u), so Tu = TP_U(u) = P_U TP_U(u). What does this tell you about where Tu lies?"""
    },

    "1.B.4": {
        "text": """Explain why there does not exist a vector space V such that V consists of exactly 8 vectors.""",
        "hint": """Think about the structure of finite vector spaces. A vector space must contain the zero vector and be closed under scalar multiplication. If V has n elements and contains a non-zero vector v, then all scalar multiples av must also be in V. How many scalar multiples can a non-zero vector have? This problem reveals that finite vector spaces over infinite fields cannot exist, but finite vector spaces over finite fields can - and their cardinality follows a specific pattern."""
    },

    "3.D.11": {
        "text": """Suppose V is finite-dimensional and S, T ∈ L(V, W). Prove that null S ⊆ null T if and only if there exists E ∈ L(W, W) such that T = ES.""",
        "hint": """For the forward direction, define E on range S by setting E(Sv) = Tv. You need to verify this is well-defined: if Sv₁ = Sv₂, does it follow that Tv₁ = Tv₂? Use the hypothesis that null S ⊆ null T. Extend E to all of W by choosing a basis. For the reverse direction, if T = ES, show that any vector in null S is also in null T by noting that if Sv = 0, then Tv = ESv = E(0) = 0."""
    }
}


# ============================================================================
# FORMATTING HELPERS
# ============================================================================

def enhance_math_notation(text):
    """Enhance mathematical notation with better Unicode characters."""
    # Superscripts
    superscripts = {'0': '⁰', '1': '¹', '2': '²', '3': '³', '4': '⁴',
                   '5': '⁵', '6': '⁶', '7': '⁷', '8': '⁸', '9': '⁹',
                   'n': 'ⁿ', '-': '⁻', '+': '⁺'}

    # Common replacements for prettier math
    replacements = {
        'R^2': 'ℝ²',
        'R^3': 'ℝ³',
        'R^n': 'ℝⁿ',
        'C^2': 'ℂ²',
        'C^3': 'ℂ³',
        'F^2': '𝔽²',
        'F^3': '𝔽³',
        '->': '→',
        '>=': '≥',
        '<=': '≤',
        '!=': '≠',
        '~=': '≈',
        'sqrt': '√',
        'infinity': '∞',
        'alpha': 'α',
        'beta': 'β',
        'gamma': 'γ',
        'delta': 'δ',
        'theta': 'θ',
        'pi': 'π',
        'sigma': 'σ',
        'tau': 'τ',
        'forall': '∀',
        'exists': '∃',
    }

    result = text
    for old, new in replacements.items():
        result = result.replace(old, new)

    return result


def print_section(title, content, style="info"):
    """Print a section with rich formatting if available, otherwise plain text."""
    if RICH_AVAILABLE:
        # Enhance the content with better math notation
        enhanced = enhance_math_notation(content)

        # Choose color based on style
        colors = {
            "info": "cyan",
            "code": "green",
            "visual": "magenta",
            "hint": "yellow",
            "problem": "blue"
        }
        color = colors.get(style, "white")

        panel = Panel(
            enhanced,
            title=f"[bold {color}]{title}[/bold {color}]",
            border_style=color,
            padding=(1, 2)
        )
        console.print(panel)
    else:
        # Fallback to plain text
        print("\n" + "─"*70)
        print(title)
        print("─"*70)
        print(enhance_math_notation(content))


def print_code(code, language="python"):
    """Print code with syntax highlighting if rich is available."""
    if RICH_AVAILABLE:
        syntax = Syntax(code, language, theme="monokai", line_numbers=True)
        console.print(Panel(syntax, title="[bold green]Python Example[/bold green]",
                          border_style="green", padding=(1, 2)))
    else:
        print("\n" + "─"*70)
        print("🐍 PYTHON EXAMPLE (NUMPY)")
        print("─"*70)
        print(code)


def print_header(text, style="bold cyan"):
    """Print a formatted header."""
    if RICH_AVAILABLE:
        console.print(f"\n[{style}]{'='*70}[/{style}]")
        console.print(f"[{style}]{text.center(70)}[/{style}]")
        console.print(f"[{style}]{'='*70}[/{style}]")
    else:
        print("\n" + "="*70)
        print(text.center(70))
        print("="*70)


# ============================================================================
# COMMAND HANDLERS
# ============================================================================

def get_sorted_concepts():
    """Return sorted list of concept names."""
    return sorted(CONCEPTS.keys())


def get_sorted_exercises():
    """Return sorted list of exercise numbers."""
    return sorted(EXERCISES.keys())


def display_concept_menu():
    """Display numbered list of concepts and return the list."""
    concepts = get_sorted_concepts()
    print("\n" + "="*70)
    print("*** SELECT A CONCEPT ***")
    print("="*70)
    for i, concept in enumerate(concepts, 1):
        print(f"  {i:2d}. {concept.title()}")
    print(f"\n  0. Back to main menu")
    print("="*70)
    return concepts


def display_exercise_menu():
    """Display numbered list of exercises and return the list."""
    exercises = get_sorted_exercises()
    print("\n" + "="*70)
    print("*** SELECT AN EXERCISE ***")
    print("="*70)
    for i, exercise in enumerate(exercises, 1):
        print(f"  {i:2d}. Exercise {exercise}")
    print(f"\n  0. Back to main menu")
    print("="*70)
    return exercises


def show_concept(concept_name):
    """Display a concept's full information."""
    concept = CONCEPTS[concept_name]

    # Header
    print_header(f"CONCEPT: {concept_name.upper()}")

    # Conceptual Explanation
    print_section(
        "📖 CONCEPTUAL EXPLANATION (from Axler's LADR)",
        concept["explanation"],
        style="info"
    )

    # Python Example with syntax highlighting
    print_code(concept["python_example"])

    # Visualization
    print_section(
        "📊 VISUALIZATION IDEA",
        concept["visualization"],
        style="visual"
    )
    print()


def show_exercise(exercise_num):
    """Display an exercise's problem and hint."""
    exercise = EXERCISES[exercise_num]

    # Header
    print_header(f"EXERCISE {exercise_num}")

    # Problem Statement
    print_section(
        "📝 PROBLEM STATEMENT",
        exercise["text"],
        style="problem"
    )

    # Hint
    print_section(
        "💡 HINT",
        exercise["hint"],
        style="hint"
    )
    print()


def concepts_menu():
    """Interactive menu for browsing concepts."""
    while True:
        concepts = display_concept_menu()
        try:
            choice = input("\nEnter number (or 0 to go back): ").strip()

            if choice == '0':
                return

            if choice.isdigit():
                idx = int(choice) - 1
                if 0 <= idx < len(concepts):
                    show_concept(concepts[idx])
                    input("\n[Press Enter to continue...]")
                else:
                    print("❌ Invalid number. Please try again.")
            else:
                print("❌ Please enter a number.")
        except KeyboardInterrupt:
            return


def exercises_menu():
    """Interactive menu for browsing exercises."""
    while True:
        exercises = display_exercise_menu()
        try:
            choice = input("\nEnter number (or 0 to go back): ").strip()

            if choice == '0':
                return

            if choice.isdigit():
                idx = int(choice) - 1
                if 0 <= idx < len(exercises):
                    show_exercise(exercises[idx])
                    input("\n[Press Enter to continue...]")
                else:
                    print("❌ Invalid number. Please try again.")
            else:
                print("❌ Please enter a number.")
        except KeyboardInterrupt:
            return


# ============================================================================
# MAIN LOOP
# ============================================================================

def display_main_menu():
    """Display the main menu."""
    print("\n" + "="*70)
    print(" "*20 + "🎓 LADR EXPLORER 🎓")
    print("="*70)
    print("\nYour interactive study companion for")
    print("'Linear Algebra Done Right' (3rd Edition) by Sheldon Axler")
    print("\n" + "─"*70)
    print("MAIN MENU")
    print("─"*70)
    print("  1. Browse Concepts (16 topics)")
    print("  2. Browse Exercises (11 problems)")
    print("  3. About this program")
    print("  4. Quit")
    print("="*70)


def show_about():
    """Display information about the program."""
    print("\n" + "="*70)
    print("ABOUT LADR EXPLORER")
    print("="*70)
    print("""
This program provides an interactive way to explore concepts and exercises
from "Linear Algebra Done Right" (3rd Edition) by Sheldon Axler.

Features:
  • 16 core concepts with detailed explanations
  • Python/NumPy code examples for each concept
  • Visualization suggestions for understanding
  • 11 exercises with helpful hints
  • Easy numbered navigation

Each concept includes:
  📖 Conceptual explanation from Axler's approach
  🐍 Python/NumPy code examples
  📊 Visualization ideas for Desmos or Matplotlib

Mathematical Notation:
  • v₁, v₂, ... = vectors (subscripts)
  • λ = lambda (eigenvalue)
  • ⟨u, v⟩ = inner product
  • ∈ = element of
  • ⊕ = direct sum
  • ⊥ = orthogonal complement

Dependencies: numpy (install with: pip install numpy)

Created to help students understand linear algebra through
Axler's determinant-free, conceptual approach.
    """)
    print("="*70)


def main():
    """Main interactive loop for the LADR Explorer."""
    # Show a message if rich is not available
    if not RICH_AVAILABLE:
        print("\n" + "="*70)
        print("NOTE: For enhanced visuals, install rich: pip install rich")
        print("="*70)

    while True:
        try:
            display_main_menu()
            choice = input("\nEnter your choice (1-4): ").strip()

            if choice == '1':
                concepts_menu()
            elif choice == '2':
                exercises_menu()
            elif choice == '3':
                show_about()
                input("\n[Press Enter to continue...]")
            elif choice == '4':
                if RICH_AVAILABLE:
                    console.print("\n[bold green]👋 Goodbye! Keep exploring linear algebra![/bold green]\n")
                else:
                    print("\n👋 Goodbye! Keep exploring linear algebra!\n")
                break
            else:
                print("\n❌ Invalid choice. Please enter a number between 1 and 4.")

        except KeyboardInterrupt:
            if RICH_AVAILABLE:
                console.print("\n\n[bold green]👋 Goodbye! Keep exploring linear algebra![/bold green]\n")
            else:
                print("\n\n👋 Goodbye! Keep exploring linear algebra!\n")
            break

        except Exception as e:
            print(f"\n❌ An error occurred: {e}")
            print("Please try again.\n")


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    main()
