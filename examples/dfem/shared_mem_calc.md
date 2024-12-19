* Calculate shared memory requirements
* Interpolation and integration
---
* If grad involved, need B and G
* Fit largest field, depends on polynomial order (#dofs)
  -> vdim is irrelevant
* Temporaries for each sum
  - DDQ (d1d x d1d x q1d) x 2 -> DDQ0, DDQ1
  - DQQ (d1d x q1d x q1d) x 3 -> DQQ0, DQQ1, DQQ2
  - QQQ (q1d x q1d x q1d) x 3 -> QQQ0, QQQ1, QQQ2

We need the following combinations at the same time
(1) DDQ0 + DDQ1 + DQQ0 + DQQ1 + DQQ2
(2) DQQ0 + DQQ1 + DQQ2 + QQQ0 + QQQ1 + QQQ2
(3) QQQ0 + QQQ1 + QQQ2 + QQD0 + QQD1 + QQD2
(4) QQD0 + QQD1 + QQD2 + QDD0 + QDD1 + QDD2

Allocate largest memory footprint from 2, 3 or 4 and
add memory footprint of fields and B/G.

Annotations with NR and R mean "not reusable" and
"reusable", respectively. This means the memory location is
reused for _all_ e.g. interpolation of a value etc.

----
For the action of nonlinear diffusion in 2D we have
(rho * |u|^2 \nabla u, \nabla v)

* Load
RHO (D x D) | R (after interpolation)
U (D x D x VDIM) | R (after interpolation)
B (Q x D) | NR
G (Q x D) | NR

* Interpolate Value
Temporary (Q x D) | R
R (Q x Q) | NR
U (Q x Q x VDIM) | NR

* Interpolate Grad
Temporaries (Q x D) + (Q x D) | R
U (Q x Q x DIM x VDIM) | NR

Quadrature point function
-> purely thread local

* Integrate Grad
R | temp from Interpolation
R | U from Load
