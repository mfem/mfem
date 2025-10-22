# Test problem: 2D periodic checkerboard potential

## Domain & units
- **Domain:** \([0, L_x] \times [0, L_y]\), periodic in both \(x, y\).
- **WLOG:** set normalized units \(m = 1\), \(q = 1\) unless you want physical units.
- **Choose wavenumbers:**  
  \(k_x = 2\pi N_x / L_x,\quad k_y = 2\pi N_y / L_y\)  
  (e.g., \(L_x = L_y = 2\pi\), \(k_x = k_y = 1\)).

---

## Electric potential (given)
\[
\phi(x, y) = \Phi_0 \cos(k_x x)\cos(k_y y)
\]

---

## Electric field (analytic for pusher)
\[
E_x(x, y) = \Phi_0 k_x \sin(k_x x)\cos(k_y y), \quad
E_y(x, y) = \Phi_0 k_y \cos(k_x x)\sin(k_y y)
\]

---

## Equations of motion (electrostatic)
\[
\ddot{x} = E_x(x, y), \quad \ddot{y} = E_y(x, y)
\]

---

## Hamiltonian (constant in time ⇒ energy conserved)
\[
H(x, y, \dot{x}, \dot{y}) = \frac{1}{2}(\dot{x}^2 + \dot{y}^2) + \phi(x, y)
\]

---

## Critical points & trapping (for \(\Phi_0 > 0\))
- **Potential extrema:** \(\phi \in [-\Phi_0, +\Phi_0]\)
- **Wells (stable):** where \(\phi = -\Phi_0\), e.g. \((x, y) = (\pi/k_x, 0)\) mod periods
- **Peaks (unstable):** where \(\phi = +\Phi_0\), e.g. \((0, 0)\) mod periods
- **Saddles (separatrices):** \(\phi = 0\) at \((\pi / 2k_x, \pi / 2k_y)\) mod periods

### Trapped vs passing
With particle kinetic energy \(K = \frac{1}{2}(\dot{x}^2 + \dot{y}^2)\):
- **Trapped** if \(H < 0\) (i.e., \(K < \Phi_0\))
- **Passing** if \(H > 0\)

---

## Small-amplitude frequencies (near a well)
Linearize near a minimum (e.g., \(x = \pi / k_x\), \(y = 0\)):

\[
\omega_x = \sqrt{\Phi_0 k_x^2}, \quad \omega_y = \sqrt{\Phi_0 k_y^2}
\]

---

## Time step guidance
- For **symplectic/leapfrog**: pick \(\Delta t \lesssim 0.1 / \max(\omega_x, \omega_y)\)
- If you’ll add a **B-field**, ensure \(\Delta t \ll 1 / \Omega_c\)