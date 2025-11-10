# LADR Explorer

**Interactive study companion for "Linear Algebra Done Right" (3rd Edition) by Sheldon Axler**

## Features

- 🎓 **16 core linear algebra concepts** with detailed explanations
- 📝 **11 exercises** with helpful hints
- 🐍 **Python/NumPy code examples** with syntax highlighting
- 📊 **Visualization suggestions** for Desmos or Matplotlib
- 🎨 **Beautiful formatting** with enhanced mathematical notation
- 🔢 **Easy numbered menu navigation** - no typing concept names!

## Installation

### Required
```bash
pip install numpy
```

### Recommended (for enhanced visuals)
```bash
pip install rich
```

The `rich` library provides:
- ✨ Color-coded panels with borders
- 🎨 Syntax-highlighted Python code
- 📐 Enhanced Unicode mathematical symbols (ℝ², ℂ³, etc.)
- 🌈 Color-coded sections (cyan for explanations, green for code, magenta for visualizations)

The program works fine without `rich`, but the output is much prettier with it!

## Usage

```bash
python LADR_Explorer.py
```

Then navigate using numbered menus:
```
Main Menu
  ├─ 1. Browse Concepts (16 topics)
  ├─ 2. Browse Exercises (11 problems)
  ├─ 3. About this program
  └─ 4. Quit
```

### Example Flow
```
Main Menu → 1 (Concepts) → 4 (Eigenvalue) → View → Press Enter → Back to menu
```

## Concepts Covered

1. Basis
2. Diagonalization
3. Dimension
4. Eigenvalue
5. Eigenvector
6. Inner Product
7. Invertibility
8. Linear Independence
9. Linear Map
10. Null Space
11. Orthogonality
12. Orthogonal Projection
13. Range
14. Span

## Mathematical Notation

The program displays proper Unicode mathematical symbols:
- **ℝⁿ, ℂⁿ, 𝔽ⁿ** - Vector spaces (instead of R^n, C^n, F^n)
- **v₁, v₂, ...** - Subscripts for vectors
- **λ** - Lambda (eigenvalues)
- **⟨u, v⟩** - Inner product
- **∈, ⊕, ⊥** - Set membership, direct sum, orthogonal
- **→, ≥, ≤, ≠, ≈** - Arrows and relations
- **√, ∞, π** - Mathematical constants

## About Axler's Approach

This tool follows Sheldon Axler's **determinant-free** approach to linear algebra:
- Focus on **conceptual understanding** over computational tricks
- Emphasis on **vector spaces and linear maps** as fundamental objects
- Understanding through **null spaces, ranges, and bases**
- Clean treatment of **eigenvalues via invariant subspaces**

Each concept includes:
- 📖 **Conceptual Explanation** - Understanding from Axler's perspective
- 🐍 **Python Example** - Concrete implementation with NumPy
- 📊 **Visualization Idea** - How to visualize the concept geometrically

## License

Educational resource for students studying linear algebra.

## Credits

Based on "Linear Algebra Done Right" (3rd Edition) by Sheldon Axler.
