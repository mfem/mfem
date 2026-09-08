// Copyright (c) 2010-2025, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#ifndef MFEM_BILININTEG_HDG
#define MFEM_BILININTEG_HDG

#include "../bilininteg.hpp"

#include <vector>

namespace mfem
{

/** Integrator for the DG form:
    $$
      \alpha \langle (u \cdot n) \{v\},[w] \rangle,
    $$
    where $v$ and $w$ are the trial and test variables, respectively, and $\rho$/$u$ are
    given scalar/vector coefficients. $\{v\}$ represents the average value of $v$ on
    the face and $[v]$ is the jump such that $\{v\}=(v_++v_-)/2$ and $[v]=(v_+-v_-)$ for the
    face with $+$ and $-$ sides. For boundary elements, $v_-=0$. The vector coefficient,
    $u$, is assumed to be continuous across the faces and when given the scalar coefficient.

    The corresponding HDG stabilization is then
    $$\begin{align}
        \langle \{\tau\} v_\pm, w_\pm \rangle, & -\langle \tau_\mp \lambda,          w_\pm \rangle,\\
        \langle \{\tau\} v_\pm, \mu   \rangle, & -\langle (\tau_+ + \tau_-) \lambda, \mu   \rangle,
    \end{align}$$
    where $\tau_\pm = |\alpha (u \cdot n)| \pm \alpha (u \cdot n)$
    and $\lambda$, $\mu$ are the trial and test trace functions, respectively.
    */
class HDGConvectionCenteredIntegrator : public DGTraceIntegrator
{
#ifndef MFEM_THREAD_SAFE
   Vector tr_shape, shape1, shape2;
#endif

public:
   HDGConvectionCenteredIntegrator(VectorCoefficient &u_, real_t a = 1.)
      : DGTraceIntegrator(u_, a, 0.) { }

   /** @brief The rule AssembleHDGFaceMatrix() uses on this interior face.

       ONE SOURCE OF TRUTH, and the reason it exists: the batched face
       assembly has to integrate at exactly the rule the per-face loop does or
       the two build different operators, and a rule copied into the batch
       driver would diverge silently the day this one changed.
       AssembleHDGFaceMatrix() calls this, so it cannot. */
   const IntegrationRule &GetHDGFaceIntRule(
      const FiniteElement &trace_el, const FiniteElement &el1,
      const FiniteElement &el2, FaceElementTransformations &Trans) const;

   void AssembleHDGFaceMatrix(const FiniteElement &trace_el,
                              const FiniteElement &el1,
                              const FiniteElement &el2,
                              FaceElementTransformations &Trans,
                              DenseMatrix &elmat) override;

   void AssembleHDGFaceMatrix(int side, const FiniteElement &trace_el,
                              const FiniteElement &el,
                              FaceElementTransformations &Trans,
                              DenseMatrix &elmat) override;

   void AssembleHDGFaceVector(int type,
                              const FiniteElement &trace_face_fe,
                              const FiniteElement &fe,
                              FaceElementTransformations &Tr,
                              const Vector &trfun, const Vector &elfun,
                              Vector &elvect) override;
};

/** Integrator for the DG form:
    $$
      \alpha \langle (u \cdot n) \{v\},[w] \rangle + \beta \langle |u \cdot n| [v],[w] \rangle,
    $$
    where $v$ and $w$ are the trial and test variables, respectively, and $\rho$/$u$ are
    given scalar/vector coefficients. $\{v\}$ represents the average value of $v$ on
    the face and $[v]$ is the jump such that $\{v\}=(v_++v_-)/2$ and $[v]=(v_+-v_-)$ for the
    face with $+$ and $-$ sides. For boundary elements, $v_-=0$. The vector coefficient,
    $u$, is assumed to be continuous across the faces and when given the scalar coefficient.

    The corresponding HDG stabilization is then
    $$\begin{align}
        \langle \tau_\pm v_\pm, w_\pm \rangle, & -\langle \tau_\mp \lambda,          w_\pm \rangle,\\
        \langle \tau_\pm v_\pm, \mu   \rangle, & -\langle (\tau_+ + \tau_-) \lambda, \mu   \rangle,
    \end{align}$$
    where $\tau_\pm = (\beta |u \cdot n| \pm 1/2 \alpha (u \cdot n))$
    and $\lambda$, $\mu$ are the trial and test trace functions, respectively.
    */
class HDGConvectionUpwindedIntegrator : public DGTraceIntegrator
{
#ifndef MFEM_THREAD_SAFE
   Vector tr_shape, shape1, shape2;
#endif

public:
   /// Construct integrator with $\beta = \alpha/2$.
   HDGConvectionUpwindedIntegrator(VectorCoefficient &u_, real_t a = 1.)
      : DGTraceIntegrator(u_, a) { }

   HDGConvectionUpwindedIntegrator(VectorCoefficient &u_, real_t a, real_t b)
      : DGTraceIntegrator(u_, a, b) { }

   /** @brief The rule AssembleHDGFaceMatrix() uses on this interior face.

       ONE SOURCE OF TRUTH, and the reason it exists: the batched face
       assembly has to integrate at exactly the rule the per-face loop does or
       the two build different operators, and a rule copied into the batch
       driver would diverge silently the day this one changed.
       AssembleHDGFaceMatrix() calls this, so it cannot. */
   const IntegrationRule &GetHDGFaceIntRule(
      const FiniteElement &trace_el, const FiniteElement &el1,
      const FiniteElement &el2, FaceElementTransformations &Trans) const;

   void AssembleHDGFaceMatrix(const FiniteElement &trace_el,
                              const FiniteElement &el1,
                              const FiniteElement &el2,
                              FaceElementTransformations &Trans,
                              DenseMatrix &elmat) override;

   void AssembleHDGFaceMatrix(int side, const FiniteElement &trace_el,
                              const FiniteElement &el,
                              FaceElementTransformations &Trans,
                              DenseMatrix &elmat) override;

   void AssembleHDGFaceVector(int type,
                              const FiniteElement &trace_face_fe,
                              const FiniteElement &fe,
                              FaceElementTransformations &Tr,
                              const Vector &trfun, const Vector &elfun,
                              Vector &elvect) override;
};

/** @brief Stabilization function of the HDG numerical flux,
    $$
       \hat q_h + \hat F_h = q_h + F(\hat u_h)
                            + s(u_h, \hat u_h)(u_h - \hat u_h) n .
    $$

    This is Eq. (5) of Nguyen, Peraire and Cockburn, J. Comput. Phys. 228
    (2009) 8841-8855, in which $s$ is a function of the potential and of its
    own trace rather than a coefficient. Their section 2.4 splits it as
    $s = s_{diff} + s_{conv}(u_h, \hat u_h)$ with $s_{diff} = \kappa/\ell$
    constant, and for a linear flux the positivity bound of their Eq. (7)
    reduces $s_{conv}$ to a constant as well -- which is exactly the
    stabilization HDGDiffusionIntegrator and HDGConvectionUpwindedIntegrator
    already apply. The constant case is therefore not a special case bolted on
    here; it is the specialization those integrators implement, and it keeps
    its own assembly path.

    A derived class that leaves IsConstant() true costs nothing at run time:
    the integrators query it once per face, never per quadrature point.

    A class that returns false makes the face term nonlinear in the unknowns
    even for a linear equation, so it is only meaningful on the residual and
    gradient assembly, which are the only paths that see the state. The
    bilinear-form path refuses it rather than silently dropping the
    dependence. */
class HDGStabilization
{
public:
   virtual ~HDGStabilization() { }

   /** @brief True when $s$ depends on neither the potential nor its trace.
       The default is the constant case. */
   virtual bool IsConstant() const { return true; }

   /** @brief The stabilization at one quadrature point.
       @param s_diff the value the integrator forms on its own, that is the
                     constant part built from the diffusion coefficient and the
                     local element size, with any quadrature weight removed
       @param un     the normal component of the convective velocity, unscaled
       @param u      the potential at the point
       @param uhat   the trace of the potential at the point
       @param Tr     element transformation, with the integration point set */
   virtual real_t Eval(real_t s_diff, real_t un, real_t u, real_t uhat,
                       ElementTransformation &Tr) const
   { return s_diff; }

   /** @brief The derivatives of $s$ with respect to the potential and to its
       trace, written $\partial_1 s$ and $\partial_2 s$ in Eq. (15) of the
       reference. Called only when IsConstant() is false.

       These are not optional refinements. In a hybridized method the Jacobian
       is never assembled globally, so omitting them gives no wrong answer,
       only slow Newton convergence -- a failure that survives a passing
       regression suite. */
   virtual void EvalGrad(real_t s_diff, real_t un, real_t u, real_t uhat,
                         ElementTransformation &Tr,
                         real_t &d1s, real_t &d2s) const
   { d1s = 0.; d2s = 0.; }
};

/** @brief Batched, device-capable assembly of HDGDiffusionIntegrator's
    INTERIOR-face matrices: every face's matrix in one mfem::forall instead of
    one host call per face.

    Roadmap: step 2 of doc/HDG-DEVICE-OFFLOAD.md, and the first kernel of the
    five. The face term reduces to two scalars per quadrature point -- the
    stabilization weight on each side, which absorbs the coefficient, the
    HDGStabilization hook and the geometry -- and a set of outer products in
    the trace and element shapes. The outer products are O(NQ*SZ^2) against
    O(NQ*ND) for the shapes, so they are the part worth moving.

    @param tr_fes   the trace (constraint) space, a DG_Interface space.
    @param el_fes   the potential space.
    @param Q        the diffusion coefficient, or NULL for 1.
    @param beta     the integrator's beta.
    @param stab     an HDGStabilization hook, or NULL for the built-in.
    @param elmats   filled with one (SZ x SZ) matrix per INTERIOR face, in mesh
                    face order, SZ = 2*ND + TRD.

    @note Uniform order and one element geometry, which CanBatch() checks.
          Interior faces only; boundary faces stay on the per-face path.
    @note The per-quadrature-point weights and shapes are precomputed on the
          HOST, so this is element-assembly rather than partial assembly: it
          moves the outer products to the device and leaves the geometry where
          it was. Removing that precompute needs the ELEMENT Jacobian
          determinant at FACE quadrature points, which FaceGeometricFactors
          does not carry, and is the next piece. */
void HDGDiffusionFaceMatricesBatched(const FiniteElementSpace &tr_fes,
                                     const FiniteElementSpace &el_fes,
                                     Coefficient *Q, real_t beta,
                                     const HDGStabilization *stab,
                                     DenseTensor &elmats);

/** @brief The same face term, SCATTERED STRAIGHT INTO the hybridization's
    storage instead of into dense per-face matrices.

    This is the version that serves a full-device path. The dense form above
    has to be read back so host code can split it into E, G, H and D -- 193 MB
    and 183 ms against an 85 ms kernel at n=96 order 3 -- and that transfer is
    larger than the arithmetic it was meant to accelerate. Writing the blocks
    where they belong removes it: nothing comes back.

    @param face_list   the INTERIOR faces, in mesh face order.
    @param E_offsets   per-face offsets into @a E_data and @a G_data; the side
                       1 block is (ND x c_dof) at E_offsets[f] and side 2
                       follows it at + c_dof*ND.
    @param H_offsets   per-face offsets into @a H_data, (c_dof x c_dof).
    @param Df_offsets  per-ELEMENT offsets into @a Df_data, (ND x ND).

    @note E, G and H are written per face and race with nothing. D ACCUMULATES
          per element, and two faces of the same element collide, so it goes
          through AtomicAdd -- which is why this needs no colouring where the
          host loop does.
    @note @a Df_data is added to and must be zeroed by the caller;
          @a E_data, @a G_data and @a H_data are overwritten.

    @note MEASURED, and the atomics are why this is a DEVICE path and not a
          host one. Correct either way -- E, G and H come out bit-exact
          against the per-face integrator and D to 8.7e-17, which is the
          accumulation order. But on the host AtomicAdd is a real atomic where
          the per-face loop does a plain +=, and it costs: 2-D quads, n=64
          order 3, per-face 38.0 ms against 60.5 ms scattered. On the device
          the same call is 43.5 ms and, unlike the dense form, NOTHING COMES
          BACK. A host build wanting this shape should take the colouring
          DarcyHybridization already builds rather than the atomics. */
/** @brief Whether HDGFaceScatterBatched() can take @a integs on @a face_list.

    Every integrator has to be one the batched kernel implements -- an
    HDGDiffusionIntegrator or either HDGConvection*Integrator, in any
    combination -- and each has to want ONE integration rule across the whole
    face list, since the kernel samples every face at the same points. A
    non-uniform ElementTransformation::OrderW() defeats the second, which is
    why it is asked of the faces rather than of the mesh.

    False means the caller keeps its per-face loop, not that anything is
    wrong. */
bool HDGFaceScatterCanBatch(const FiniteElementSpace &tr_fes,
                            const FiniteElementSpace &el_fes,
                            const Array<BilinearFormIntegrator*> &integs,
                            const Array<int> &face_list);

/** @brief The interior-face HDG constraint of every integrator in @a integs,
    for every face in @a face_list, scattered straight into E, G, H and D.

    The generalisation of HDGDiffusionFaceScatterBatched(), which covered a
    lone pure-diffusion term. What every supported integrator has in common is
    its SHAPE: at each quadrature point it contributes

        D1 += wd1 s1 s1^T,   E1 -= we1 s1 tr^T,   G1 += wg1 tr s1^T,
        D2 += wd2 s2 s2^T,   E2 -= we2 s2 tr^T,   G2 += wg2 tr s2^T,
        H  -= wh  tr tr^T,

    and they differ only in the seven weights. Diffusion is the degenerate
    case wd1 = we1 = wg1 and wd2 = we2 = wg2; the convection forms are not --
    the centred one has E and G on different weights, and the upwinded one
    crosses them, side 1's E carrying side 2's weight. In all three
    wh = wd1 + wd2, which is checked rather than assumed.

    Each integrator is applied in its own pass at its own rule, accumulating,
    after one pass that zeroes E, G and H. D is not zeroed here: it accumulates
    across the faces of an element and its zeroing belongs to the caller. */
void HDGFaceScatterBatched(const FiniteElementSpace &tr_fes,
                           const FiniteElementSpace &el_fes,
                           const Array<BilinearFormIntegrator*> &integs,
                           const Array<int> &face_list,
                           const Array<int> &E_offsets,
                           const Array<int> &H_offsets,
                           const Array<int> &Df_offsets,
                           Vector &E_data, Vector &G_data,
                           Vector &H_data, Vector &Df_data);

/** @brief Whether HDGBdrFaceScatterBatched() can take @a integs.

    The same question HDGFaceScatterCanBatch() asks, on boundary faces. The
    face lists are per integrator because a boundary integrator carries an
    attribute marker and two of them need not apply to the same faces. */
/** @brief Whether HDGElementMassBatched() can take @a integs on @a fes.

    Every integrator has to be a MassIntegrator or a VectorMassIntegrator --
    the two the Darcy mass forms carry -- and they all have to want one
    integration rule across the mesh, since the kernel samples every element
    at the same points. A VectorMassIntegrator's vdim has to match the space's;
    it is a settable field and DarcyOperator does set it. */
bool HDGElementMassCanBatch(const FiniteElementSpace &fes,
                            const Array<BilinearFormIntegrator*> &integs);

/** @brief Every element's mass matrix, for every integrator in @a integs, as
    one device-resident Vector of NE blocks of (ndof*vdim)^2.

    Element assembly rather than partial assembly, which is what the caller
    wants: DarcyHybridization stores an element matrix per element and
    condenses it, so a PA operator would have to be un-applied to get back
    what it already has. The blocks come out in NATIVE dof order and
    field-outermost, which is VectorMassIntegrator::AssembleElementMatrix()'s
    own layout and Ordering::byNODES's -- so a caller's element vdofs index
    them directly, with no lexicographic reordering of the kind
    EABilinearFormExtension::GetElementMatrices() has to do.

    @a emat is overwritten, not accumulated.

    NOT the same route as MFEM's AssemblyLevel::ELEMENT, deliberately. That
    one has no notion of vdim -- EABilinearFormExtension sizes ea_data as
    ne*ndof*ndof with a scalar ndof -- and for a DG space it folds the form's
    FACE terms into the element matrices, which is exactly the work
    DarcyForm::Assemble() routes itself. Either would be wrong here. */
void HDGElementMassBatched(const FiniteElementSpace &fes,
                           const Array<BilinearFormIntegrator*> &integs,
                           Vector &emat);

/** @brief Whether HDGElementDivBatched() can take @a integs.

    Every integrator has to be a VectorDivergenceIntegrator -- the one the
    Darcy divergence form carries on a discontinuous flux -- and they all have
    to want one integration rule across the mesh. The flux space's vdim has to
    be the mesh's space dimension, which is what makes its element matrix
    (test dofs) x (sdim * trial dofs) and lets the hat-dof mask index it. */
bool HDGElementDivCanBatch(const FiniteElementSpace &trial_fes,
                           const FiniteElementSpace &test_fes,
                           const Array<BilinearFormIntegrator*> &integs);

/** @brief Every element's divergence block, as one device-resident Vector of
    NE blocks of (test dofs) x (sdim * trial dofs), column-major.

    The mixed counterpart of HDGElementMassBatched(), and it exists for a
    sharper reason than that one: MFEM has NO mixed element assembly at all.
    MixedBilinearForm::SetAssemblyLevel aborts on AssemblyLevel::ELEMENT
    ("stay tuned"), the two-space AssembleEA virtual is commented out in
    fem/bilininteg.hpp, and VectorDivergenceIntegrator has no EA. So there is
    nothing upstream to route through even in principle, on any element shape.

    Columns are field-outermost -- column `k*trial_dofs + a` is component k of
    trial dof a -- which is DenseMatrix::GradToDiv()'s own layout and
    Ordering::byNODES's, so the caller's element vdofs index it directly. */
void HDGElementDivBatched(const FiniteElementSpace &trial_fes,
                          const FiniteElementSpace &test_fes,
                          const Array<BilinearFormIntegrator*> &integs,
                          Vector &emat);

/** @brief Whether HDGMixedConductionResidualBatched() can take @a integ on
    @a fes.

    @a integ has to be a MixedConductionNLFIntegrator whose constitutive law
    is a LinearDiffusionFlux. That is the whole of the state-independence
    requirement and it is a guarantee rather than a guess: the law's
    ComputeDualFlux() takes its state argument UNNAMED, so the dual flux is a
    fixed linear map of the flux at each point, and ComputeDualFluxJacobian()
    hands that map over directly. A FunctionDiffusionFlux is refused -- its
    conductivity is a caller-supplied `std::function<real_t(const Vector &x,
    real_t u)>` evaluated at the CURRENT potential, so there is neither a
    device-callable form of it nor a per-point weight that can be computed
    ahead of the kernel. Lifting that needs a different caller contract, not a
    guard.

    One equation only. MixedConductionNLFIntegrator::AssembleElementGrad()
    asserts a (neq*dim)-square dual-flux Jacobian and LinearDiffusionFlux
    returns a dim-square one, so neq > 1 does not work through this law in the
    per-element route either.

    A scalar-range flux basis, so one shape value serves every spatial
    component and the reference shape table is the same for every element. An
    H(div) element needs CalcVShape(), which is per element, and lays its dofs
    out per equation rather than per (equation, direction).

    And one element geometry, one dof count and one integration rule across
    the mesh, which is what lets the kernel sample every element at the same
    points -- the same three conditions HDGElementMassCanBatch() asks.

    Those three are checked over EVERY element, so this is O(NE) with an
    ElementTransformation built per element, and the kernel's own
    MFEM_VERIFY runs it a second time. That is the house style here and the
    cost is inside the timing on
    DarcyHybridization::CanBatchLocalResidual() rather than hidden from it;
    but note it is paid per residual EVALUATION, not once per assembly the
    way HDGElementMassCanBatch()'s is. */
bool HDGMixedConductionResidualCanBatch(const FiniteElementSpace &fes,
                                        BlockNonlinearFormIntegrator *integ);

/** @brief The flux row of a MixedConductionNLFIntegrator's ELEMENT term,
    every element at once, matrix free.

    @a u_all is the local flux, element-blocked: element e at offset
    (ndof*vdim)*e, component c of dof i at c*ndof + i, which is
    FiniteElementSpace::GetElementVDofs()'s own Ordering::byNODES layout and
    therefore DarcyHybridization::el_u_dofs's. @a ru_all is overwritten with
    the same layout.

    Matrix free rather than assemble-then-apply, and that is the cheaper way
    round here as well as the one that needs no storage: the residual costs
    O(nq * ndof * dim) where forming the block first costs O(nq * ndof^2 *
    dim). The per-point conductivity is read out of the law once per point per
    element -- exactly as many coefficient evaluations as the per-element
    route makes -- and the apply is one mfem::forall over ELEMENTS with the
    dof and quadrature loops inside it, which is the shape that keeps the
    index arithmetic out of the kernel.

    Round-off against MixedConductionNLFIntegrator::AssembleElementVector(),
    and the association is deliberately its association: the accumulation is
    written `w * shape(i) * mF[c]` in that order, and the dual flux is formed
    by the same k-order contraction the law's own MultABt() uses, so on the
    cases measured it comes out BITWISE. That is a measurement, not a
    guarantee -- Vector::operator*() reduces through mfem::reduce and a
    device backend need not associate a dot product the way the host does. */
void HDGMixedConductionResidualBatched(const FiniteElementSpace &fes,
                                       BlockNonlinearFormIntegrator *integ,
                                       const Vector &u_all, Vector &ru_all);

bool HDGBdrFaceScatterCanBatch(const FiniteElementSpace &tr_fes,
                               const FiniteElementSpace &el_fes,
                               const Array<BilinearFormIntegrator*> &integs,
                               const std::vector<Array<int>> &face_lists);

/** @brief The BOUNDARY-face HDG constraint of every integrator in @a integs,
    scattered straight into E, G, H and D.

    One-sided throughout, and that is the whole difference from the interior
    form: a boundary face's E slot is (element dofs) x (trace dofs) rather
    than twice that, H takes one side's weight, and there is no side 2. The
    weight identity wh = wd1 + wd2 that holds on an interior face does NOT
    hold here for the upwinded form -- its H term keeps 2*beta*|u.n| at a
    boundary deliberately, "for stability reasons", where its D takes
    beta|u.n| + alpha(u.n)/2. That is why the weights are carried separately
    rather than derived from one another.

    @a face_lists gives, per integrator, the faces its attribute marker
    admits; @a all_faces is their union and is what gets zeroed first. E, G
    and H are assigned by the per-face route (CopyMN), so zero-then-accumulate
    reproduces it; D accumulates there and here. */
void HDGBdrFaceScatterBatched(const FiniteElementSpace &tr_fes,
                              const FiniteElementSpace &el_fes,
                              const Array<BilinearFormIntegrator*> &integs,
                              const std::vector<Array<int>> &face_lists,
                              const Array<int> &all_faces,
                              const Array<int> &E_offsets,
                              const Array<int> &H_offsets,
                              const Array<int> &Df_offsets,
                              Vector &E_data, Vector &G_data,
                              Vector &H_data, Vector &Df_data);

void HDGDiffusionFaceScatterBatched(const FiniteElementSpace &tr_fes,
                                    const FiniteElementSpace &el_fes,
                                    Coefficient *Q, real_t beta,
                                    const HDGStabilization *stab,
                                    const Array<int> &face_list,
                                    const Array<int> &E_offsets,
                                    const Array<int> &H_offsets,
                                    const Array<int> &Df_offsets,
                                    Vector &E_data, Vector &G_data,
                                    Vector &H_data, Vector &Df_data);

/// Whether HDGDiffusionFaceMatricesBatched() can run on these spaces.
bool HDGDiffusionFaceMatricesCanBatch(const FiniteElementSpace &tr_fes,
                                      const FiniteElementSpace &el_fes);

/** Integrator for the H/LDG diffusion stabilization term
    The LDG stabilization takes the form
    $$
        1/2 \beta \langle \{h^{-1} Q\} [v], [w] \rangle
    $$
    where $Q$ is a scalar or matrix diffusion coefficient and $v$, $w$ are the trial
    and test functions, respectively.

    The corresponding HDG stabilization is then
    $$\begin{align}
        \langle \tau_\pm v_\pm, w_\pm \rangle, & -\langle \tau_\pm \lambda,          w_\pm \rangle,\\
        \langle \tau_\pm v_\pm, \mu   \rangle, & -\langle (\tau_+ + \tau_-) \lambda, \mu   \rangle,
    \end{align}$$
    where $\tau_\pm = (\beta \pm 1/2 \alpha (u \cdot n) / |u \cdot n|) \{h^{-1} Q\}$
    and $\lambda$, $\mu$ are the trial and test trace functions, respectively. The vector
    coefficient $u$ is assumed continuous across the faces. */
class HDGDiffusionIntegrator : public BilinearFormIntegrator
{
protected:
   VectorCoefficient *v;
   Coefficient *Q;
   MatrixCoefficient *MQ;
   real_t alpha, beta;
   const HDGStabilization *stab{};

   /** @brief The weighted stabilization at one quadrature point.

       With no user object this is the built-in expression, unchanged and with
       no call. With one, the quadrature weight is divided out so that the
       object sees s itself, and put back afterwards. @a u and @a uhat are only
       meaningful where the state is available; the bilinear paths pass zero,
       which is why they insist the object be constant. */
public:
   /** @brief Whether the face term is the pure diffusion one: no velocity and
       a scalar (or absent) coefficient.

       The batched assembly covers that case and nothing else, so it asks
       before taking over. */
   bool IsPureDiffusion() const { return v == nullptr && MQ == nullptr; }

   /// The scalar coefficient, or null.
   Coefficient *GetCoefficient() const { return Q; }
   /// The matrix coefficient, or null.
   MatrixCoefficient *GetMatrixCoefficient() const { return MQ; }
   /// The velocity coefficient, or null.
   VectorCoefficient *GetVelocity() const { return v; }
   /// The alpha the constructor took.
   real_t GetAlpha() const { return alpha; }
   /// The beta the constructor took.
   real_t GetBeta() const { return beta; }

   /** @brief StabValue() made public, so a batched assembly weighs a point
       the way the per-face loop does instead of copying the expression.

       It was copied once already, into HDGFaceScatterBatched()'s ancestor,
       and a copy of a two-line expression is exactly the kind of thing that
       survives a change to the original. */
   real_t EvalStabilization(real_t wq, real_t ba, real_t un, real_t face_w,
                            real_t u, real_t uhat,
                            ElementTransformation &Tr) const
   { return StabValue(wq, ba, un, face_w, u, uhat, Tr); }

   /** @brief The rule AssembleHDGFaceMatrix() uses on this interior face.

       ONE SOURCE OF TRUTH, and the reason it exists: the batched face
       assembly has to integrate at exactly the rule the per-face loop does or
       the two build different operators, and a rule copied into the batch
       driver would diverge silently the day this one changed.
       AssembleHDGFaceMatrix() calls this, so it cannot. */
   const IntegrationRule &GetHDGFaceIntRule(
      const FiniteElement &trace_el, const FiniteElement &el1,
      const FiniteElement &el2, FaceElementTransformations &Trans) const;


protected:
   inline real_t StabValue(real_t wq, real_t ba, real_t un, real_t face_w,
                           real_t u, real_t uhat,
                           ElementTransformation &Tr) const
   {
      if (!stab) { return wq * ba; }
      const real_t s_diff = (face_w != 0.) ? (wq * ba / face_w) : 0.;
      return face_w * stab->Eval(s_diff, un, u, uhat, Tr);
   }

   /** @brief Per-point scratch, and a MEMBER only in a build that cannot
       thread.

       This used to read "these are not thread-safe!", which was true and is
       now handled the way MFEM handles it everywhere else: members when
       MFEM_THREAD_SAFE is off, method-local declarations when it is on -- see
       FluxFunction::ComputeFluxDotN() for the same pattern. Each method sizes
       what it uses, so the local declarations are bare.

       **And this IS on the element-local hot path, which is not obvious.**
       A first version of this comment said the opposite -- that a
       BilinearFormIntegrator is evaluated when the forms are assembled and not
       once per residual. It is wrong twice over: BilinearFormIntegrator
       derives from NonlinearFormIntegrator, so one can be registered on the
       nonlinear potential form, and DarcyForm::Assemble() collects that form's
       interior face integrators into a SumNLFIntegrator and hands it to
       SetConstraintIntegrators() as c_nlfi_p. DarcyHybridization's
       ConstructGrad() and LocalResidual() then call AssembleHDGFaceGrad() and
       AssembleHDGFaceVector() on it once per element per evaluation. That is
       exactly how every nonlinear HDG diffusion problem in this tree is posed
       -- the unit tests' pedestal included. */
#ifndef MFEM_THREAD_SAFE
   Vector tr_shape, shape1, shape2, vu, nor, nh, ni;
   Vector nor_Jt, nor_Ji, ni_Jt, ni_Ji;
   DenseMatrix mq;
#endif

public:
   /// Construct integrator with $\alpha = 0$ and $\beta = a$.
   HDGDiffusionIntegrator(const real_t a = 0.5)
      : v(NULL), Q(NULL), MQ(NULL), alpha(0.), beta(a) { }

   /// Construct integrator with $\alpha = 0$ and $\beta = a$.
   HDGDiffusionIntegrator(Coefficient &q, const real_t a = 0.5)
      : v(NULL), Q(&q), MQ(NULL), alpha(0.), beta(a) { }

   /// Construct integrator with $\alpha = 0$ and $\beta = a$.
   HDGDiffusionIntegrator(MatrixCoefficient &q, const real_t a = 0.5)
      : v(NULL), Q(NULL), MQ(&q), alpha(0.), beta(a) { }

   /// Construct integrator with $\alpha = a$ and $\beta = a/2$.
   HDGDiffusionIntegrator(VectorCoefficient &v_, const real_t a = 0.5)
      : v(&v_), Q(NULL), MQ(NULL), alpha(a), beta(0.5*a) { }

   /// Construct integrator with $\alpha = a$ and $\beta = a/2$.
   HDGDiffusionIntegrator(VectorCoefficient &v_, Coefficient &q,
                          const real_t a = 0.5)
      : v(&v_), Q(&q), MQ(NULL), alpha(a), beta(0.5*a) { }

   /// Construct integrator with $\alpha = a$ and $\beta = a/2$.
   HDGDiffusionIntegrator(VectorCoefficient &v_, MatrixCoefficient &q,
                          const real_t a = 0.5)
      : v(&v_), Q(NULL), MQ(&q), alpha(a), beta(0.5*a) { }

   /** @brief Replace the stabilization with a user supplied one.

       The object is referenced, not owned, and must outlive the integrator.
       Leaving it unset keeps the built-in constant stabilization and the
       assembly path that goes with it. */
   void SetStabilization(const HDGStabilization &s) { stab = &s; }

   const HDGStabilization *GetStabilization() const { return stab; }

   using BilinearFormIntegrator::AssembleFaceMatrix;
   void AssembleFaceMatrix(const FiniteElement &el1,
                           const FiniteElement &el2,
                           FaceElementTransformations &Trans,
                           DenseMatrix &elmat) override;

   void AssembleHDGFaceMatrix(const FiniteElement &trace_el,
                              const FiniteElement &el1,
                              const FiniteElement &el2,
                              FaceElementTransformations &Trans,
                              DenseMatrix &elmat) override;

   void AssembleHDGFaceMatrix(int side, const FiniteElement &trace_el,
                              const FiniteElement &el,
                              FaceElementTransformations &Trans,
                              DenseMatrix &elmat) override;

   void AssembleHDGFaceVector(int type,
                              const FiniteElement &trace_face_fe,
                              const FiniteElement &fe,
                              FaceElementTransformations &Tr,
                              const Vector &trfun, const Vector &elfun,
                              Vector &elvect) override;

   /** @brief The gradient of the face residual.

       Overridden rather than inherited because the base class builds it from
       AssembleHDGFaceMatrix(), which cannot see the state and so cannot carry
       the derivatives of a solution dependent stabilization. */
   void AssembleHDGFaceGrad(int type,
                            const FiniteElement &trace_face_fe,
                            const FiniteElement &fe,
                            FaceElementTransformations &Tr,
                            const Vector &trfun, const Vector &elfun,
                            DenseMatrix &elmat) override;

   real_t ComputeHDGFaceEnergy(int side,
                               const FiniteElement &trace_face_fe,
                               const FiniteElement &fe,
                               FaceElementTransformations &Tr,
                               const Vector &trfun, const Vector &elfun,
                               Vector *d_energy = NULL) override;
};

}

#endif //MFEM_BILININTEG_HDG
