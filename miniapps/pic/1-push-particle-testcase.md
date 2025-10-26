# Test problem: 2D periodic cosine-sum potential

## Domain & units
- **Domain:** \([0, L_x] \times [0, L_y]\), periodic in both \(x, y\).
- **WLOG:** set normalized units \(m = 1\), \(q = 1\) unless you want physical units.
- **Choose wavenumbers:**  
  \(k_x = \frac{2\pi N_x}{L_x},\quad k_y = \frac{2\pi N_y}{L_y}\)  
  (e.g., \(L_x = L_y = 2\pi\), \(k_x = k_y = 1\)).

---

## Electric potential (given)
The potential used in the code is
\[
\phi(x, y) = \Phi_0 \left[\cos(k_x x) + \cos(k_y y)\right]
\]
where  
\[
k_x = \frac{2\pi N_x}{L_x}, \quad k_y = \frac{2\pi N_y}{L_y}.
\]

---

## Electric field (analytic for pusher)
We take \(\mathbf{E} = -\nabla \phi\).  
Then
\[
E_x(x, y) = \Phi_0 k_x \sin(k_x x), \quad
E_y(x, y) = \Phi_0 k_y \sin(k_y y)
\]

Note:
- Each component now depends only on its own coordinate (no checkerboard cross-coupling: no \(\cos k_y y\) factor in \(E_x\), etc.).
- The field is still periodic in both \(x\) and \(y\).

---

## Equations of motion (electrostatic)
\[
\ddot{x} = E_x(x, y) = \Phi_0 k_x \sin(k_x x), \quad
\ddot{y} = E_y(x, y) = \Phi_0 k_y \sin(k_y y)
\]

These equations are now separable in \(x\) and \(y\).

---

## Hamiltonian (constant in time ⇒ energy conserved)
\[
H(x, y, \dot{x}, \dot{y}) = \frac{1}{2}(\dot{x}^2 + \dot{y}^2) + \phi(x, y)
= \frac{1}{2}(\dot{x}^2 + \dot{y}^2) + \Phi_0 \left[\cos(k_x x) + \cos(k_y y)\right]
\]

Total energy \(H\) is conserved for a given particle in this static field.

---

## Critical points & trapping (for \(\Phi_0 > 0\))
- **Potential range:**  
  Since \(\cos \in [-1, +1]\),  
  \[
  \phi \in [-2\Phi_0,\; +2\Phi_0].
  \]

- **Wells (stable minima):** where \(\cos(k_x x) = -1\) and \(\cos(k_y y) = -1\).  
  That happens at  
  \[
  x = \frac{\pi}{k_x} \ (\text{mod } 2\pi/k_x), \quad
  y = \frac{\pi}{k_y} \ (\text{mod } 2\pi/k_y),
  \]
  giving \(\phi = -2\Phi_0\).

- **Peaks (unstable maxima):** where \(\cos(k_x x) = +1\) and \(\cos(k_y y) = +1\).  
  That happens at  
  \[
  x = 0 \ (\text{mod } 2\pi/k_x), \quad
  y = 0 \ (\text{mod } 2\pi/k_y),
  \]
  giving \(\phi = +2\Phi_0\).

- **Saddles / separatrices:** mixed points such as  
  \[
  \cos(k_x x)=+1,\ \cos(k_y y)=-1 \quad\text{or}\quad
  \cos(k_x x)=-1,\ \cos(k_y y)=+1,
  \]
  which yield \(\phi = 0\). At these points the particle can transition between being locally trapped in one well vs passing over the barrier in one direction.

### Trapped vs passing
Let the particle kinetic energy be \(K = \frac{1}{2}(\dot{x}^2 + \dot{y}^2)\).

- The absolute minimum of the potential is \(-2\Phi_0\).  
  So the minimum possible Hamiltonian is
  \[
  H_{\min} = -2\Phi_0.
  \]

- The saddle energy is \(\phi = 0\). Crossing that means you can escape a single well in at least one direction.

- A particle starting in a well (near a minimum) is **trapped in that well** if its total energy satisfies
  \[
  H < 0,
  \]
  i.e. it cannot climb to the saddle at \(\phi = 0\).

- If \(H > 0\), it has enough energy to pass over the saddle in both \(x\) and \(y\) directions somewhere in phase space, so it is effectively **passing**.

(Compared to the checkerboard case, the numerical thresholds changed because \(\phi\) now spans \([-2\Phi_0, +2\Phi_0]\) instead of \([-\Phi_0, +\Phi_0]\).)

---

## Small-amplitude frequencies (near a well)
Linearize around one potential minimum.  
Pick a stable well, e.g.
\[
x_0 = \frac{\pi}{k_x}, \quad y_0 = \frac{\pi}{k_y}.
\]

Expand \(\cos(k_x x)\) about \(x_0\):
\[
\cos(k_x x) \approx -1 + \tfrac{1}{2} (k_x^2)(x - x_0)^2.
\]

Similarly for \(y\):
\[
\cos(k_y y) \approx -1 + \tfrac{1}{2} (k_y^2)(y - y_0)^2.
\]

So near the minimum,
\[
\phi(x,y) \approx -2\Phi_0 + \frac{1}{2}\Phi_0 k_x^2 (x - x_0)^2 + \frac{1}{2}\Phi_0 k_y^2 (y - y_0)^2.
\]

This is just two uncoupled harmonic wells. The small oscillation frequencies are
\[
\omega_x = \sqrt{\Phi_0}\, k_x, \quad
\omega_y = \sqrt{\Phi_0}\, k_y.
\]

(Same functional form as before, but note the potential curvature now comes from each cosine separately rather than the product \(\cos k_x x \cos k_y y\).)

---

## Time step guidance
- For **symplectic/leapfrog**: pick
  \[
  \Delta t \lesssim \frac{0.1}{\max(\omega_x, \omega_y)}
  \]
  using the updated \(\omega_x, \omega_y\) above.

- If you’ll add a **B-field**, ensure \(\Delta t \ll 1 / \Omega_c\) as usual.