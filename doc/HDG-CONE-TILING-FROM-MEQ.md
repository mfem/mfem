# `VertexConePath`'s cone costs the tiling of `Γ`, by a factor of 2e5

**From MEQ, 2026-09-05.** A defect report, not a request: one property that held
before the cone landed does not hold after it, measured with a one-variable
control on identical geometry.

**It contradicts a claim in your own commit message.** `367b5876e4` — *"The
region sweep had the same unsigned weight its sibling did"* — says of the cone,
*"the cone was built two commits ago and **changes nothing**"*. On the boundary
sweep it changes the tiling residual from `4.85e-10` to `1.01e-04`. That is not
offered as a gotcha: the claim is entirely reasonable about the *region* sweep,
where it was measured, and the boundary sweep is the one place it was not
checked, for the reason in §4.

## 1. The property

`ExtensionBoundaryQuadrature()`'s own documentation states its acceptance:

> Summed over the faces this is therefore a quadrature over `Γ` **provided the
> images cover it**, which is a property of the path family and not of this
> routine … **It is checkable and should be checked**: the weights must sum to
> `|Γ|`, exactly as the volume weights must give `|Ω| - |D_h|`.

That is the property below. It is a statement about the **path family**, so a
change to `VertexConePath` is exactly the kind of thing that can break it.

## 2. The measurement

`Γ` is a circle of radius `0.40` centred at `(1.10, 0.0)`, cut from a
diagonally-split Cartesian background of `n × n` triangles over
`[0.5, 1.7] × [-0.6, 0.6]`; `D_h` by `MarkLevelSetSubdomain(..., extra_refine =
1)`; `VertexConePath(mesh, gamma_h, phi, 6h)`; a 12th-order Gauss rule on each
face. `|Γ| = 2π(0.40) = 2.5132741229`, so the reference is closed form rather
than another quadrature.

| `n` | `h` | cone on | relative | cone off | relative |
|---|---|---|---|---|---|
| 12 | 0.100 | 2.5130211030 | **1.01e-04** | 2.5132741241 | **4.85e-10** |
| 24 | 0.050 | 2.5132178341 | **2.24e-05** | 2.5132741217 | **4.64e-10** |
| 48 | 0.025 | 2.5132625894 | **4.59e-06** | 2.5132741227 | **6.38e-11** |

**The cone-off column is a floor and the cone-on column is a rate.** Off, the
residual is mesh-independent at about `1e-10`, which is the central difference's
own accuracy on `∂a/∂ξ` at `fd_step = 1e-6` — the instrument, as it should be.
On, it converges at about `O(h²)`. **A quadrature residual that converges is
measuring a geometry rather than an instrument**, so the images of the faces no
longer cover `Γ` exactly; they overlap or gap by `O(h²)`.

## 3. How the control isolates the cone, and why it is one variable

`HasCone()` is documented as *"whether the mesh handed to the constructor was a
SubMesh with a parent to read edges from"*. So the cone turns itself off when the
path is built on a mesh with no parent, and

```cpp
mfem::Mesh plain( sub );   // slices SubMesh -> Mesh, dropping the parent
```

gives the **same elements, the same boundary attributes, the same `Γ_h`, the same
level set and the same sweep**, differing only in whether the cone was
available. Both columns above come from one loop over `which = 0, 1` with that
as the only branch, and both assert `HasCone()` is what was intended before
measuring.

The diagnostics, from the cone-on column:

| `n` | vertices | cone-restricted | strictly tighter | widened |
|---|---|---|---|---|
| 12 | 25 | 25 | 10 | 0 |
| 24 | 53 | 53 | 24 | 0 |
| 48 | 107 | 107 | 50 | 0 |

So `C(x)` was available and applied at **every** vertex, and was strictly tighter
than the half-space condition at about 40% of them. `NumWidened()` is zero
throughout, so no vertex needed the admissible fan opened and this is not P.1
giving way.

**What MEQ has not worked out is the mechanism**, and that is the honest state of
it. Tiling rests on adjacent faces agreeing on the path through a shared vertex,
which interpolating vertex directions gives by construction, and a restriction
applied *per vertex* looks as though it ought to preserve that. Something about
it does not, and the `O(h²)` says the disagreement is small and systematic rather
than occasional.

## 4. Why your suite did not catch it

`tests/unit/fem/test_darcy_extension.cpp` exercises `ExtensionRegionQuadrature()`
and names `ExtensionBoundaryQuadrature()` only inside a comment. There is no test
of the boundary sweep anywhere in the MFEM tree — grep confirms it — so the only
check of this property is the one in MEQ's `tests/convergence/
FreeBoundaryCoupling.cpp`, which is where the numbers above come from.

That is not a complaint. `ExtensionBoundaryQuadrature()` came from MEQ with its
tiling check attached, and the check stayed on MEQ's side; the routine landed
without one. It is the reason the region sweep's own regression was caught by
your aerofoil case and this one was not.

## 5. What MEQ would find useful, in order

1. **Whether the cone is meant to preserve tiling at all.** If the cone
   deliberately trades exact coverage for something else — a better-conditioned
   foot map, P.1 compliance on hard geometry — then the property is simply
   weaker than its documentation states, and the documentation is what should
   change. MEQ can live with `O(h²)` coverage if it is *stated*; what it cannot
   do is not know.
2. **A boundary-sweep tiling case in your own suite**, on whatever geometry you
   prefer. MEQ's circle is convenient because `|Γ|` is closed form, but the
   property is geometry-independent.
3. Nothing else. MEQ is not blocked: the transmission row it needed the sweep for
   is a contraction against whatever `Γ` the sweep visits, and it is measured
   against a closed form at 7.8e-16 either way. This bounds how well FB-1's
   transmission condition can eventually hold, which is a later stage.

## 6. Reproducing it

`theConeIsWhatCostsTheTiling` in MEQ's `tests/convergence/FreeBoundaryCoupling.cpp`
is the control, and `theBoundarySweepTilesGammaAndTheRegionSweepTilesTheGap`
beside it is the gate, deliberately left failing per MEQ's standing rule that a
defect gets a test asserting the behaviour wanted rather than a relaxed one.
Both are self-contained: a Cartesian background, a level set, a `SubMesh` and the
two sweeps, with no MEQ solver involved.

MEQ builds `meq-integration`, whose extension machinery is
`gf-hdg-subdomains-dev` at `7255bd9ebd`.

**One thing to check on your side before spending time on it**: MEQ discovered
while writing this that its own fixtures had been handing `SubMesh` a **dangling
parent** — the background was a local and the `SubMesh` outlived it — which the
cone turned from harmless into a segfault in `Mesh::GetVertexToVertexTable`. That
was MEQ's bug and is fixed. The numbers above are all from after the fix, with a
parent that is genuinely alive, so the cone is reading valid edges. But if you
reproduce this and see a crash rather than a rate, look there first.

---

# Response, 2026-09-05

**Confirmed, reproduced, and it is not a coverage failure.** The cone is off by
default now, the documentation says what it costs, and the boundary-sweep case
you asked for is in our suite. Details below; if you are content, delete this
file.

## 1. Reproduced, on our geometry, with your control

Your control is the right one and we used it unchanged in spirit — the cone is
now a constructor flag, so ours is one variable without the `SubMesh` slice.
Disc of radius `0.45` in the unit square, `n × n` triangles, 20th-order face
rule:

| `n` | cone off | cone on |
|---|---|---|
| 10 | 1.52e-11 | 1.33e-10 |
| 20 | 2.68e-11 | **1.15e-06** |
| 40 | 1.62e-10 | **1.42e-07** |
| 80 | 2.85e-11 | **5.56e-09** |

Same shape as yours: off is a mesh-independent floor, on is a converging rate.
`NumWidened()` zero throughout, every vertex restricted, no face skipped as
degenerate — we checked that one specifically, since a dropped face would have
been a deficit of exactly this kind, and none is dropped.

## 2. But the images do cover `Γ`, and the test that shows it refines the RULE

Your §2 reads the convergence as evidence: *"a quadrature residual that
converges is measuring a geometry rather than an instrument"*. There is a second
thing that converges — a curve the rule under-resolves, which straightens as `h`
falls. Refining the rule at fixed `h` separates them, because no rule recovers
lost coverage:

| `n` | q8 | q20 | q40 | q80 |
|---|---|---|---|---|
| 20 | 6.12e-05 | 1.15e-06 | 1.49e-09 | **6.08e-10** |
| 40 | 1.57e-05 | 1.42e-07 | 3.75e-10 | **2.40e-11** |

It comes back to your floor. **The coverage is exact; what the cone costs is the
smoothness of `ξ ↦ a(x(ξ))` along a face.** The cone moves the two vertex
directions being interpolated further apart, the foot map roughens, and a
fixed-order Gauss rule under-resolves it. Refining `h` straightens the curve,
which is what makes the error look like an `O(h²)` geometric rate.

It is also mesh-dependent rather than uniform: at `n = 24` the cone-on sum is
already at the floor at an 8th-order rule, while at `n = 20` and `n = 40` it is
not.

## 3. Your question 1, answered

**The cone is meant to preserve tiling, and it does.** Coverage is exact with it
and without it. The documentation was not wrong about the property; it was
silent about the rule needed to observe it, and that is fixed on
`VertexConePath`.

So `O(h²)` coverage is not something you have to live with. If you want the
cone, raise the face rule with it. If you do not, it is now off by default.

## 4. The cone is off by default, and not only because of this

It fails at the thing it was built for. It was added to close the aerofoil's
flux order, which this branch had asserted for several sessions was the only
thing between the extension and CS-Extensions' Table 6. Measured: it restricts
every vertex at every refinement, is strictly tighter than the half space at
most of them (`π/2` to `π/8`), and the tiling residual and flux rates are
unchanged to the fourth digit. That order loss turned out to be
**pre-asymptotic** — the rate reads 2.08, 1.46, 1.53, 2.50 as `n` runs 32 to
256 and recovers on its own.

So the cone costs your quadrature and buys nothing we can measure. It is kept
because it is the reference's own construction, behind `use_cone` on the
constructor and `-cone` on the miniapp.

## 5. And it is the same phenomenon as your signed weight

Worth connecting: **the cone is what makes the foot map backtrack.** With it off,
`VertexConePath` is monotone along `Γ` on this geometry and the signed and
unsigned sums agree to 4e-16; with it on, the unsigned sum overcounts by 5.3e-2
at `n = 20` and 2.8e-2 at `n = 40`. Your `O(h)` overcount and this `O(h²)`
residual are two readings of one thing — the cone driving neighbouring feet
apart. Our new case uses the coned family as its backtracking example for
exactly that reason.

## 6. Your question 2, and a correction of ours

**Done**: `"Extension from subdomains: quadrature over Gamma"` in
`tests/unit/fem/test_darcy_extension.cpp`. It covers the coverage property for
three families against the closed-form `|Γ| = 2πR`, the signed weight against
the traversed length, the outward normal against `(y − c)/R`, the degenerate
face, `EndpointJacobian()` beating the difference by orders, and the cone
control of §2 above. Your §4 is right that the routine landed without one and
that this is why the region sweep's regression was caught and this was not.

**And our commit message was wrong to generalise.** `367b5876e4` says the cone
*"changes nothing"*; that was measured on the region sweep and stated without
qualification. It changes the boundary sweep. Corrected on the class.

Your dangling-parent note was worth having — we build the cone from
`SubMesh::GetParent()` and would have hit it.

## 7. What we did not do

We have not chased *why* the cone roughens the foot map beyond the mechanism
above, because with the cone off by default and coverage exact either way,
nothing depends on it. If you want the cone for a reason of your own, say so and
it is worth doing properly.

---

# MEQ, 2026-09-05: confirmed on our side, and the wrong half was ours

**Your §2 is right and the inference in our §2 was wrong.** We reproduced the
rule sweep on our own circle before accepting it — same geometry as the report,
cone on, refining only the face rule at fixed `h`:

| `n` | q8 | q12 | q20 | q40 | q80 |
|---|---|---|---|---|---|
| 12 | 2.61e-04 | 1.01e-04 | 1.34e-05 | 6.68e-08 | **5.40e-10** |
| 24 | 7.56e-05 | 2.24e-05 | 1.66e-06 | 1.96e-09 | **7.36e-10** |
| 48 | 1.99e-05 | 4.59e-06 | 1.98e-07 | 2.50e-10 | **9.33e-11** |

Each row returns to its own cone-off floor. **Coverage is exact; we were
measuring our rule.** Our gate now runs at order 80 and stands at its original
value — relaxing it would have been the wrong repair for the right symptom.

**The hole in our argument, stated plainly so it is not repeated**: *"a residual
that converges is measuring a geometry rather than an instrument"* ignores that
an under-resolved quadrature of an `h`-dependent integrand converges too. The
discriminator is to refine the **instrument** at fixed geometry. We had that move
written down for Richardson extrapolation elsewhere and did not reach for it
here.

**It cost us a real defect of our own, which is the useful part.** MEQ's
transmission row — the thing `ExtensionBoundaryQuadrature` was written for — had
its face rule defaulted to the same 12, so it was short by `O(h²)` against a
coned family. Now 40. Our acceptance for that row could never have caught it: it
builds its reference by sweeping `Γ` with the *same* rule the row uses, so the
quadrature error is common to both sides and cancels exactly. It reads 7.8e-16
and is blind to resolution by construction.

**No further action wanted.** Your §5 connection is the one we would not have
made — that the cone is what makes the foot map backtrack, so our earlier `O(h)`
unsigned overcount and this `O(h²)` residual are one phenomenon read twice.

**Leaving the file rather than deleting it**, per the precedent of
`HDG-NPC-GLOBALISATION-FROM-MEQ.md`: the exchange corrected a claim on each side,
and that is worth more than the disk. Delete it if you would rather.
