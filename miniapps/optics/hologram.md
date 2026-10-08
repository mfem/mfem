# Computer-Generated Hologram via Whittaker–Sinc Fresnel Propagation

This miniapp reads a target PNG and retrieves a **phase mask**. It writes the
mask and the reconstructed **monochromatic hologram** as PNG files. The mask
has unit amplitude. A thin lens multiplies it, and Fresnel propagation carries
it to the observation plane. `hologram` is the serial program. When MFEM is
built with MPI, `phologram` runs the same solver.

PNG I/O needs MFEM built with **zlib** (`general/zpng.hpp`). The default
device is `cpu` (`-d`). The default propagator is the Toeplitz–FFT product
(`-fft`) when the library for that device is linked: **FFTW** on a host
device, **cuFFT** or **hipFFT** on CUDA or HIP. Otherwise the default is the
dense product (`-gem`). `-fft` does not run FFTW on a CUDA or HIP device. It
aborts when the library for the active device was not linked.

The default target is MFEM’s logo (`doc/logo-small.png`). The physical
defaults are a wavelength of 532 nm (`-w`), a mask width of 19 mm with the
height set by the image aspect ratio, and a propagation distance and focal
length of 0.80 m (`-z`, `-f`). With those defaults the observation plane is
the focal plane. The output files are `mask.png` and `hologram.png`.

The grid is a Cartesian quadrilateral mesh. The samples are the nodes of an
L2 space of order `-p`, one element per block of $(p+1)^2$ pixels. The source
amplitude is constant per element. The solve uses MFEM’s `Operator`,
`NewtonSolver`, `CGSolver`, and device kernels.

## Mathematical model

Scalar Fresnel propagation from the source field $u(x,y)$ to an observation
plane a distance $z$ away is

$$
U(X,Y) = \frac{e^{ikz}}{i\lambda z}
\iint_{\mathbb{R}^2}
\exp\negthinspace\Bigl(\frac{ik}{2z}\bigl[(X-x)^2+(Y-y)^2\bigr]\Bigr)\thinspace
u(x,y)\thinspace dx\thinspace dy,\quad k = 2\pi/\lambda.
$$

The prefactor equals $-ik\thinspace e^{ikz}/(2\pi z)$. Equivalently $U = h_F * u$ with

$$
h_F(x,y) = -\frac{ik\thinspace e^{ikz}}{2\pi z}
\exp\bigl(ik(x^2+y^2)/(2z)\bigr).
$$

### Whittaker–sinc quadrature

The code replaces $u$ by the Shannon–Whittaker interpolant of its pixel
samples. On a uniform grid of spacing $\delta$ that interpolant is bandlimited
to $1/(2\delta)$:

$$
u(x,y) = \sum_{j,\ell} u_{j\ell}\thinspace
\operatorname{sinc}\negthinspace\Bigl(\frac{x-x_j}{\delta}\Bigr)
\operatorname{sinc}\negthinspace\Bigl(\frac{y-y_\ell}{\delta}\Bigr),
\qquad
\operatorname{sinc}(t)=\frac{\sin(\pi t)}{\pi t}.
$$

The source and observation grids are the same $n_x\times n_y$ pixels, with
$\delta_x = L_x/n_x$ and $\delta_y = L_y/n_y$. Substituting the interpolant
into the Fresnel integral separates into one-dimensional Fresnel integrals of
a single sinc (`fresnel_phi` / `fresnel_CS` in `sinc.hpp`). The kernel depends
on squared lags, so the weights are even, and the matrices below are complex
symmetric and **Toeplitz**. Stored with rows along $y$ and columns along $x$,

$$
U = e^{ikz}\thinspace\omega^y\thinspace u\thinspace\omega^x.
$$

`GemFresnel` builds the dense matrices and applies $\omega^y$ on the left and
$\omega^x$ on the right. `FftFresnel` applies the same product by embedding each
Toeplitz factor in a circulant matrix. `FresnelOp::Propagate` multiplies by
$e^{ikz}$ and then applies this product.

### Thin lens

In the mask plane the code multiplies by the thin-lens transmission, sampled
at pixel centers measured from the aperture center:

$$
T(x,y) = \exp\negthinspace\Bigl(-\frac{i\pi}{\lambda f}(x^2+y^2)\Bigr)
$$

(`Lens` in `hologram.hpp`, with `scale = -π/(λ f)`).

### Phase retrieval algorithm

The grid is a Cartesian quadrilateral mesh. The samples are the nodes of an
L2 space of order `-p` (default $0$), one element per block of $(p+1)^2$
pixels. The source amplitude $A_{\mathrm{src}}=1$ is constant per element.
Phases stay independent, one per sample, initialized at $0$, so the mask
field is

$$
u = A_{\mathrm{src}}\thinspace e^{i\varphi}.
$$

`ObjectiveOp` applies the thin lens and the product
$\omega^y\thinspace(\cdot)\thinspace\omega^x$. Call that field $V$. The residual omits
$e^{ikz}$; the factor has modulus one, so $J$ is the same either way.
`FresnelOp::Propagate` multiplies by it on the hologram replay. With
$m_i = |V_i|$, `RasterField::MatchTarget` rescales
$V$ by the least-squares factor

$$
s = \frac{\sum_i A_i m_i}{\sum_i m_i^2}
$$

($s = 0$ when the denominator is at most $10^{-30}$) and sets $U = sV$. Each
target sample $A_i\in[0,1]$ is the Rec. 601 luma of the PNG,

$$
A = 0.299\thinspace R + 0.587\thinspace G + 0.114\thinspace B,
$$

with the 8-bit channels divided by $255$. The objective in
`PhaseMask::MeanAmplitudeError` is the mean-square amplitude error on $N$ pixels,

$$
J(\varphi) = \frac{1}{N}\sum_{i=1}^{N}
\bigl(|U_i|_ {\varepsilon} - A_i\bigr)^2,
\qquad
|U|_ {\varepsilon} = \sqrt{|U|^2 + \varepsilon},
\quad \varepsilon = 10^{-24}.
$$

Forward evaluations recompute $s$. Derivatives hold that value fixed, which
contributes the real factor $s$ in the chain rule.

`ObjectiveOp::Mult` returns $\nabla_\varphi J$. The partial derivatives of $J$
with respect to the real and imaginary parts of $U$ are multiplied by $s$,
pulled back through the Fresnel adjoint and the conjugate lens, and mapped to
the phase by the adjoint of $\partial u/\partial\varphi = iu$. `NewtonSolver`
solves $\nabla_\varphi J = 0$ with right-hand side zero, starting from the
current phase (`iterative_mode`).

The Jacobian supplied to Newton is the damped Gauss–Newton operator `GaussNewtonOp`
for the unsmoothed residual $r_i = s m_i - A_i$:

$$
H = \frac{2}{N} J_r^{\mathsf T} J_r + \frac{\lambda}{N} I,
\qquad J_r = \frac{\partial r}{\partial\varphi},
$$

with $\lambda$ initialized to $10^{-2}$. Pixels with $|U| \le 10^{-14}$ add
nothing to $J_r$. The smoothed modulus $\varepsilon$ is used in $J$ and in
$\nabla_\varphi J$. CG solves $H\thinspace\delta\varphi = \nabla_\varphi J$ to a
relative tolerance of $10^{-3}$ and an absolute tolerance of $10^{-12}$, for at
most `-cg` iterations (default 20).

`PhaseNewton` wraps trial and accepted phases into $[-\pi,\pi)$. The update
$\varphi - \alpha\thinspace\delta\varphi$ tries at most eight step lengths,
$\alpha = 1, \tfrac12, \ldots, \tfrac{1}{128}$. The first trial with a strictly
smaller $J$ is accepted, and $\lambda$ becomes $\max(\lambda/2,\thinspace10^{-8})$. If
no trial decreases $J$, $\lambda$ becomes $\min(10\lambda,\thinspace10^{4})$, CG runs
again, and the backtracking search is repeated once. A second failure stops
the solve.

Stopping conditions are $\Vert\nabla_\varphi J\Vert \le 10^{-8}$, the iteration cap
`-mi` (default 50), a failed line search, or `StallController`: once four
iterations have been reached, three consecutive iterations whose relative
decrease in $J$ is under `-stall` (default 1%).

The mask PNG encodes this wrapped phase as an HSV hue $(\varphi+\pi)/(2\pi)$
at full saturation and value. The hologram PNG replays unit-amplitude
$e^{i\varphi}$ through the lens and `FresnelOp::Propagate`, then encodes the
intensity $|V|^2$ of that unscaled field as a CIE 1931 2° monochromatic sRGB
color at the chosen wavelength. The white point is the 99.5th percentile of
the intensity.
