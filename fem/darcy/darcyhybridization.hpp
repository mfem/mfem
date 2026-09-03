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

#ifndef MFEM_DARCYHYBRIDIZATION
#define MFEM_DARCYHYBRIDIZATION

#include "../../config/config.hpp"
#include "../bilinearform.hpp"
#include "../nonlinearform.hpp"
#ifdef MFEM_USE_MPI
#include "../pbilinearform.hpp"
#include "../pnonlinearform.hpp"
#endif //MFEM_USE_MPI

#include <functional>
#include <map>
#include <utility>

#define MFEM_DARCY_HYBRIDIZATION_ELIM_BCS
#define MFEM_DARCY_HYBRIDIZATION_GRAD_MAT

namespace mfem
{

/// Class for total flux hybridization of Darcy-like mixed systems
/** Class DarcyHybridization performs total flux hybridization of mixed systems
    with (anti)symmetric weak form common for parabolic and elliptic problems.
    They can be written as:
    \verbatim
        ┌        ┐┌   ┐   ┌    ┐
        | Mu ±Bᵀ || u | _ | bu |
        | B  Mp  || p | ̅  | bp |
        └        ┘└   ┘   └    ┘
    \endverbatim
    where @a u is the flux (continuous or discontinuous) and @a p is the
    potential (assumed always discontinuous). The bilinear forms @a Mu
    and @a Mp are the mass terms of the flux and potential respectively. The
    mixed bilinear form @a B is the divergence of flux (in a generalized sense)
    and @a bu and @a bp are the right-hand-side terms of the flux and potential
    respectively.

    The sign convention of the system is chosen in the constructor
    DarcyHybridization(). Given the set of the forms (Mu, B, Mp), either a
    symmetric system without a sign change (#bsym == false) or with a flipped
    sign (#bsym == true) is formed respectively:
    \verbatim
        ┌       ┐        ┌        ┐
        | Mu Bᵀ |        | Mu -Bᵀ |
        | B  Mp |   or   | -B -Mp |
        └       ┘        └        ┘
    \endverbatim

    The process of hybridization introduces an additional constraint equation
    mediating coupling between elements, meaning discontinuous, "broken" spaces
    can be used instead. The constraint enforces continuity of the total flux,
    which can have contributions from both, flux and potential parts. The full
    system then takes the form:
    \verbatim
        ┌           ┐┌   ┐   ┌    ┐
        | Mu ±Bᵀ Cᵀ || u |   | bu |
        | B   D  E  || p | = | bp |
        | C   G  H  || λ |   | br |
        └           ┘└   ┘   └    ┘
    \endverbatim
    where @a C is the constraint term with optional stabilization contributions
    in @a E, @a G, @a H and @a D. The new variable @a λ is the Lagrange
    multiplier approximating the trace of @a p. Note the best conditioning of
    the problem is achieved with @a λ taken from the trace space of @a u,
    but the generality of the construction allows different choices with
    sufficient stabilization. Also note the hybridized system is not
    necessarily equivalent to the original mixed formulation.

    An important advantage of the hybridized formulation is discontinuity of
    the spaces for the primary quantities, which enables to eliminate (often
    referred to as static condensation) the flux and potential equations
    by means of local inversion (i.e., the inverted matrix is block diagonal):
    \verbatim
                         ┌        ┐-1 ┌    ┐
                         | Mu ±Bᵀ |   | Cᵀ |
        H  ← H  - [ C G ]| B   D  |   | E  |
                         └        ┘   └    ┘
                         ┌        ┐-1 ┌    ┐
                         | Mu ±Bᵀ |   | bu |
        br ← br - [ C G ]| B   D  |   | bp |
                         └        ┘   └    ┘
        H λ = br
    \endverbatim
    This reduced linear system is equivalent to the full formulation and
    enables an economical solution procedure, where the original quantities
    can be recovered afterwards as follows:
    \verbatim
        ┌   ┐   ┌        ┐-1 /┌    ┐   ┌    ┐   \
        | u | _ | Mu ±Bᵀ |  | | bu |   | Cᵀ |    |
        | p | ̅  | B   D  |  | | bp | ̅  | E  | λ  |
        └   ┘   └        ┘   \└    ┘   └    ┘   /
    \endverbatim

    The first step of the hybridization process is assembly at the element/face
    level. It is initiated by a call of Init() followed by subsequent calls of
    Assemble*Matrix() methods and ComputeAndAssemblePot*FaceMatrix() for face
    integration of the potential constraint. The assembly process is finished
    by Finalize(), enabling to use Mult() or access the hybridized system
    matrix through GetMatrix() (or GetParallelMatrix() in parallel). The right
    hand side of the mixed system can be reduced through ReduceRHS(). After
    solution of the hybridized system, the original quantities of the system
    can be recovered through ComputeSolution().

    Some common configurations of finite element spaces are Raviart-Thomas
    elements for the fluxes and Lagrange elements for the potentials. This RTDG
    scheme does not require stabilization due to compatibility of the spaces
    and the constraint space can be naturally chosen as
    DG_Interface_FECollection, which coincides with the trace space of RT
    elements (up to the sign convention). The hybridized scheme is then
    equivalent to the original mixed formulation.

    However, continuity of the flux space does not allow stabilization of
    the trace for advection in the potential equation. Therefore, the flux
    space can be chosen as broken Raviart-Thomas (BrokenRT_FECollection), which
    is discontinuous and enables upwinding of the trace. For more details see:
    Egger, H., & Schoberl, J. (2009). A hybrid mixed discontinuous Galerkin
    finite-element method for convection-diffusion problems. IMA Journal of
    Numerical Analysis, 30(4), 1206–1234. https://doi.org/10.1093/imanum/drn083

    Generalizing the fluxes as discontinuous, a logical choice is using
    Lagrange elements for both quantities, which yields the well-known
    Hybridizable Discontinuous Galerkin (HDG) method. As the spaces are not
    mutually compatible (in inf-sup sense), the scheme requires stabilization
    of the trace unknown to converge to the actual trace of the potential and
    vice versa. This can be achieved through redefinition of the total flux
    with a forcing term like τ(p̂-λ), which naturally stabilizes the scheme
    ( @a τ is a coefficient and @a p̂ is trace of the potential @a p ). These
    contributions populate the terms @a E, @a G, @a H and @a D and require the
    potential constraint integrator to compute all these face matrices, which
    are collectively denoted as the HDG face matrix. Some common integrators of
    this type can be found in bilininteg_hdg.hpp. For more details about
    construction of HDG for convection-diffusion problems see:
    Nguyen, N. C., Peraire, J., & Cockburn, B. (2009). An implicit high-order
    hybridizable discontinuous Galerkin method for linear convection–diffusion
    equations. Journal of Computational Physics, 228(9), 3232–3254.
    https://doi.org/10.1016/j.jcp.2009.01.030

    A notable feature of HDG schemes is reconstruction of the total flux and
    superconvergent quantities in turn. The constraint equation, which enforces
    continuity of the total flux, is used to project the total flux on the
    face restriction of the total flux finite element (typically from
    Raviart-Thomas space). The interior DOFs are determined by integral
    projection of the flux function passed to ReconstructTotalFlux(). Apart
    from being useful on its own, the total flux can be used for reconstruction
    of the original quantites (flux and potential) with polynomial order higher
    by one, where it is used as a source term for every element in the mixed
    formulation. For more details, refer to section 4 of the cited paper.

    In so far, the trace space was considered only as DG_Interface_FECollection
    matching restriction of Raviart-Thomas to skeleton of the mesh. However,
    trace space of H1 elements can be used as well (H1_Trace_FECollection).
    This choice corresponds to the Embedded Discontinuous Galerkin (EDG) method
    known in the literature: Nguyen, N. C., Peraire, J., & Cockburn, B. (2015).
    A class of embedded discontinuous Galerkin methods for computational fluid
    dynamics. Journal of Computational Physics, 302, 674–692.
    https://doi.org/10.1016/j.jcp.2015.09.024. Such construction is more
    economical, sharing the nodal DOFs between adjacent faces, at the expense
    of local conservation properties and conditioning of the local problem.
 */
class DarcyHybridization : public Hybridization
{
public:
   enum class LSsolveType
   {
      LBFGS,
      LBB,
      Newton,
   };

   enum class LPrecType
   {
      GMRES,
      LU,
   };

protected:
   FiniteElementSpace &fes_p;       ///< potential FE space
#ifdef MFEM_USE_MPI
   ParFiniteElementSpace *pfes;     ///< parallel flux FE space
   ParFiniteElementSpace *pfes_p;   ///< parallel potential FE space
   ParFiniteElementSpace *c_pfes;   ///< parallel constraint FE space
#endif
   std::unique_ptr<BilinearFormIntegrator> c_bfi_p;      ///< constraint integrator
   std::unique_ptr<NonlinearFormIntegrator> c_nlfi_p;
   std::unique_ptr<BlockNonlinearFormIntegrator> c_nlfi;
   NonlinearFormIntegrator *m_nlfi_u{};
   NonlinearFormIntegrator *m_nlfi_p{};
   bool own_m_nlfi_u{};
   bool own_m_nlfi_p{};
   BlockNonlinearFormIntegrator *m_nlfi{};
   bool own_m_nlfi{};

   /// The potential constraint boundary face integrators
   std::vector<BilinearFormIntegrator*> boundary_constraint_pot_integs;
   /// Boundary markers for potential constraint face integrators
   std::vector<Array<int>*> boundary_constraint_pot_integs_marker;
   std::vector<NonlinearFormIntegrator*> boundary_constraint_pot_nonlin_integs;
   std::vector<Array<int>*> boundary_constraint_pot_nonlin_integs_marker;
   std::vector<BlockNonlinearFormIntegrator*> boundary_constraint_nonlin_integs;
   std::vector<Array<int>*> boundary_constraint_nonlin_integs_marker;
   /// Indicates if the boundary_constraint_pot_integs integrators are owned externally
   bool extern_bdr_constr_pot_integs{false};

   bool bsym{};      ///< sign convention, see DarcyReduction()
   bool bfin{};      ///< indicates finalized hybridization
   DiagonalPolicy diag_policy{DIAG_ONE};  ///< diagonal policy
   Array<int> ess_tdof_list;              ///< essential flux true DOFs

   /** @brief Per-face polynomial degree of the trace, or empty for uniform.

       Set by SetTraceOrders(). Empty means a uniform trace and every accessor
       then falls through to the constraint space itself, returning the same
       pointers and the same nulls, so a caller that never sets a degree pays
       nothing and gets byte-identical results. See SetTraceOrders() for what
       the entries mean. */
   Array<int> tr_order;

   /** @brief The essential trace true DOFs the *caller* asked for, in the
       CONSTRAINT SPACE's numbering.

       That is the numbering a caller can name, being the space's own;
       #ess_tdof_list is the same set in the constrained numbering, which
       depends on the degrees. Finalize() rebuilds one from the other through
       MapEssentialTraceDofs(), so a second Finalize() after Reset() with
       different degrees does not inherit the first one's map. The two
       coincide while the trace is uniform. */
   Array<int> ess_tdof_user;

   /** @brief The constrained trace: the numbering, the two face maps, and the
       prolongation composed out of them.

       Built lazily by BuildTraceConstraint() and empty while the trace is
       uniform, so a caller that never sets a per-face degree allocates
       nothing and every accessor falls through to the constraint space's
       own. Cleared by SetTraceOrders() and by Init(); nothing else can
       change what they depend on, which is #tr_order and the space. */
   mutable Array<int> ctr_offsets;
   /// ctdof -> constraint-space true DOF, block diagonal over faces.
   mutable std::unique_ptr<SparseMatrix> ctr_PE;
   /// ctdof -> constraint-space VDOF, i.e. cP composed with #ctr_PE. Serial.
   mutable std::unique_ptr<SparseMatrix> ctr_Pv;
   /// E and R, keyed by (face geometry, degree) rather than by face.
   mutable std::map<std::pair<int,int>,DenseMatrix> ctr_E, ctr_R;
   /// Constraint-space VDOF -> its true DOF, or -1 when it has none here.
   mutable Array<int> ctr_vdof2tdof;
   mutable int ctr_ntdof{0};
   mutable bool ctr_built{false};
#ifdef MFEM_USE_MPI
   /// ctdof -> constraint-space LDOF, i.e. Dof_TrueDof composed with #ctr_PE.
   mutable std::unique_ptr<HypreParMatrix> ctr_pP;
   mutable Array<HYPRE_BigInt> ctr_col_starts;
#endif

private:
   struct
   {
      LSsolveType type;
      int iters;
      real_t rtol;
      real_t atol;
      int print_lvl;
      struct
      {
         LPrecType type;
         int iters;
         real_t rtol;
         real_t atol;
      } prec;
   } lsolve;

   Array<int> Ae_offsets;
   Array<real_t> Af_lin_data, Ae_data;
   bool A_empty{true};

   Array<int> Bf_offsets, Be_offsets;
   Array<real_t> Bf_data, Be_data;

   /** @brief The solution-dependent part of the local gradient's (0,1) block,
       d(flux residual)/dp.

       For a flux law q = D(p) u the flux equation depends on the potential, so
       the (0,1) block of the local Jacobian is not simply the transpose of the
       linear divergence form: it is that plus the derivative the flux function
       supplies as J_u. @a Bf_data holds the linear part and is assembled once;
       this holds the part that changes with the solution and is rebuilt at
       every Newton step.

       Indexed by @a Bf_offsets, which gives each element the same number of
       entries, but each block is read as a_dofs by d_dofs -- the transpose
       orientation of @a Bf_data -- because that is the shape of a (0,1) block.

       Left empty whenever no integrator contributes such a term, which covers
       every linear problem and every nonlinear one whose coefficients do not
       depend on the potential; @a Bnl_empty then short-circuits the extra
       work. */
   mutable Array<real_t> Bnl_data;
   mutable bool Bnl_empty{true};

   /** @brief Load the (0,1) gradient block of element @a el into @a Bnl.

       Returns false, leaving @a Bnl untouched, when there is no such block. */
   bool GetBnlMatrix(int el, DenseMatrix &Bnl) const;

   Array<int> Df_offsets, Df_f_offsets;
   mutable Array<real_t> Df_data, Df_lin_data;
   mutable Array<int> Df_ipiv;
   bool D_empty{true};

   Array<int> Ct_offsets;
   Array<real_t> Ct_data;

   mutable Array<int> E_offsets;
   mutable Array<real_t> E_data;

   Array<int> &G_offsets{E_offsets};
   mutable Array<real_t> G_data;

   mutable Array<int> H_offsets;
   mutable Array<real_t> H_data;

   mutable Array<int> darcy_offsets, darcy_toffsets;
   mutable BlockVector darcy_rhs;
   Vector darcy_u, darcy_p;
   mutable Array<int> f_2_b;

   std::unique_ptr<SparseMatrix> He;
   OperatorHandle pHe;
   mutable std::unique_ptr<SparseMatrix> Grad;
   mutable OperatorHandle pGrad;

#ifndef MFEM_DARCY_HYBRIDIZATION_GRAD_MAT
   friend class Gradient;
   class Gradient : public Operator
   {
      const DarcyHybridization &dh;
   public:
      Gradient(const DarcyHybridization &dh)
         : Operator(dh.Width()), dh(dh) { }

      void Mult(const Vector &x, Vector &y) const override;
   };
#endif //MFEM_DARCY_HYBRIDIZATION_GRAD_MAT

#ifdef MFEM_USE_MPI
   friend class ParOperator;
   class ParOperator : public Operator
   {
      const DarcyHybridization &dh;
      mutable OperatorHandle pGrad;
   public:
      ParOperator(const DarcyHybridization &dh)
         : Operator(dh.c_fes.GetTrueVSize()), dh(dh) { }

      void Mult(const Vector &x, Vector &y) const override;
      Operator& GetGradient(const Vector &x) const override;
   };
   mutable OperatorHandle pOp;

#ifndef MFEM_DARCY_HYBRIDIZATION_GRAD_MAT
   class ParGradient : public Operator
   {
      const DarcyHybridization &dh;
   public:
      ParGradient(const DarcyHybridization &dh)
         : Operator(dh.c_fes.GetTrueVSize()), dh(dh) { }

      void Mult(const Vector &x, Vector &y) const override;
   };
#endif //MFEM_DARCY_HYBRIDIZATION_GRAD_MAT
#endif //MFEM_USE_MPI

   enum class LocalOpType { FluxNL, PotNL, FullNL };
   LocalOpType lop_type{LocalOpType::FullNL};

   friend class LocalNLOperator;
   class LocalNLOperator : public Operator
   {
   protected:
      const DarcyHybridization &dh;
      int el;
      const BlockVector &trps;
      const Array<int> &faces;

      const int a_dofs_size, d_dofs_size;
      DenseMatrix B;
      TransposeOperator Bt;
      const FiniteElement *fe_u, *fe_p;
      IsoparametricTransformation *Tr;
      std::vector<FaceElementTransformations*> FTrs;
      std::vector<IsoparametricTransformation*> NbrTrs;
      const Array<int> offsets;
      mutable Vector Au, Dp, DpEx;
      mutable DenseMatrix grad_A, grad_D;
      /** The (0,1) block and, when it is nonzero, the dense sum of it with
          the linear +/-B^T that would otherwise stand there alone. */
      mutable DenseMatrix grad_Aup, grad_Bt;
      mutable BlockOperator grad;

      void AddMultBlock(const Vector &u_l, const Vector &p_l, Vector &bu,
                        Vector &bp) const;
      void AddMultA(const Vector &u_l, Vector &bu) const;
      void AddMultDE(const Vector &p_l, Vector &bp) const;
      void AddGradBlock(const Vector &u_l, const Vector &p_l, DenseMatrix &gA,
                        DenseMatrix &gD) const;
      void AddGradA(const Vector &u_l, DenseMatrix &gA) const;
      void AddGradDE(const Vector &p_l, DenseMatrix &gD) const;

   public:
      LocalNLOperator(const DarcyHybridization &dh, int el, const BlockVector &trps,
                      const Array<int> &faces);
      virtual ~LocalNLOperator();

      inline const Array<int>& GetOffsets() const { return offsets; }

      void Mult(const Vector &x, Vector &y) const override;
      Operator &GetGradient(const Vector &x) const override;
   };

   class LocalFluxNLOperator : public LocalNLOperator
   {
      const Vector &bp;
      LUFactors LU_D;

      mutable Vector p_l;

   public:
      LocalFluxNLOperator(const DarcyHybridization &dh, int el, const Vector &bp,
                          const BlockVector &trps, const Array<int> &faces);

      void SolveP(const Vector &u_l, Vector &p_l) const;
      void Mult(const Vector &x, Vector &y) const override;
      Operator &GetGradient(const Vector &x) const override;
   };

   class LocalPotNLOperator : public LocalNLOperator
   {
      const Vector &bu;
      LUFactors LU_A;

      mutable Vector u_l;

   public:
      LocalPotNLOperator(const DarcyHybridization &dh, int el, const Vector &bu,
                         const BlockVector &trps, const Array<int> &faces);

      void SolveU(const Vector &p_l, Vector &u_l) const;
      void Mult(const Vector &x, Vector &y) const override;
      Operator &GetGradient(const Vector &x) const override;
   };

   class DenseMatrixLUSolver : public Solver
   {
      const DenseMatrix *mat;
      DenseMatrixInverse inv;
   public:
      DenseMatrixLUSolver() { }

      void SetOperator(const Operator &op) override
      {
         mat = dynamic_cast<const DenseMatrix*>(&op);
         MFEM_VERIFY(mat, "Not a DenseMatrix operator!");
         height = mat->Height();
         width = mat->Width();
         MFEM_ASSERT(height == width, "Not a square matrix!");
         inv.Factor(*mat);
      }

      void Mult(const Vector &x, Vector &y) const override
      {
         inv.Mult(x, y);
      }
   };

   bool IsNonlinear() const { return c_nlfi || c_nlfi_p || m_nlfi || m_nlfi_u || m_nlfi_p; }
#ifdef MFEM_USE_MPI
   bool ParallelU() const { return pfes != NULL; }
   bool ParallelP() const { return pfes_p != NULL; }
   bool ParallelC() const { return c_pfes != NULL; }
#else
   bool ParallelU() const { return false; }
   bool ParallelP() const { return false; }
   bool ParallelC() const { return false; }
#endif

   void GetFDofs(int el, Array<int> &fdofs) const;
   void GetEDofs(int el, Array<int> &edofs) const;
   FaceElementTransformations *GetFaceTransformation(int f) const;

   /// Allocate everything whose size comes from the trace element.
   void AllocTraceBlocks();

   /// Convert a list of constraint-space VDOFs to true DOFs.
   void TraceVDofsToTDofs(const Array<int> &vdofs, Array<int> &tdofs) const;

   /** @brief Express the caller's essential trace DOFs in the constrained
       numbering.

       #ess_tdof_user holds them as the constraint space's own true DOFs,
       which is what the caller can name; the reduced system is in the
       constrained ones. A trace face's DOFs are face-interior, so a face is
       essential in a given field or it is not, and the map is per face rather
       than per DOF. Called from Finalize(), which is where the degrees stop
       changing, and idempotent because it rebuilds #ess_tdof_list from
       #ess_tdof_user each time. */
   void MapEssentialTraceDofs();

   /** @brief Build #ctr_offsets, #ctr_PE and #ctr_Pv. Idempotent, lazy. */
   void BuildTraceConstraint() const;

   /** @brief The face maps of face @a f: E embeds its degree in the ceiling,
       R reads a ceiling function back at its nodes.

       E(j,i) = phi_i^lo(node_j^hi) and R(i,j) = phi_j^hi(node_i^lo), so
       R E = I exactly and a degree-p_f function has exactly nt(p_f) degrees
       of freedom in the ceiling's storage. Cached by (geometry, degree),
       because the rows of E are the ceiling element's nodes in the face's own
       reference ordering -- which is the ordering GetFaceVDofs() returns --
       so orientation never enters them. It enters the nonconforming transfer
       and the parallel true-DOF map, and both act at the ceiling where the
       space handles it already.

       R E = I is CHECKED here, once per distinct key, rather than assumed:
       it is what says the ceiling collection really is nodal and really does
       contain the coarse space. A modal or non-nested collection fails it
       loudly instead of quietly discretising something else. */
   const DenseMatrix &TraceEmbedding(int f) const;
   const DenseMatrix &TraceRestrictionMat(int f) const;

   /** @brief The trace element of face @a f's own DEGREE, as against
       TraceFE(), which is the ceiling's and is where its DOFs live.

       Private, and needed only to build the constraint: nothing outside this
       class has a use for a basis the stored coefficients are not in. */
   const FiniteElement *FaceDegreeFE(int f) const;

   void AssembleCtFaceMatrix(int face, const DenseMatrix &elmat);
   void AssembleCtSubMatrix(int el, const DenseMatrix &elmat,
                            DenseMatrix &Ct, int ioff=0);
   using face_getter = std::function<void(int, DenseMatrix &)>;
   void AssembleNCSlaveFaceMatrix(int f,
                                  face_getter fx_Ct = face_getter(), const DenseMatrix *Ct = NULL,
                                  face_getter fx_C = face_getter(), const DenseMatrix *C = NULL,
                                  face_getter fx_H = face_getter(), const DenseMatrix *H = NULL);
   void AssembleNCSlaveCtFaceMatrix(int f, const DenseMatrix &Ct);
   void AssembleNCSlaveEGFaceMatrix(int f, const DenseMatrix &E,
                                    const DenseMatrix &G);
   void AssembleNCSlaveHFaceMatrix(int f, const DenseMatrix &H);
   void ConstructC();
   void AllocD() const;
   void AllocEG() const;
   void AllocH() const;
   enum class MultNlMode { Mult, Sol, Grad, GradMult };
   void MultNL(MultNlMode mode, const Vector &bu, const Vector &bp,
               const Vector &x, Vector &y) const;
   void MultNL(MultNlMode mode, const BlockVector &b, const Vector &x,
               Vector &y) const
   { MultNL(mode, b.GetBlock(0), b.GetBlock(1), x, y); }
   void ParMultNL(MultNlMode mode, const BlockVector &b, const Vector &x,
                  Vector &y) const;
   void InvertA();
   void InvertD();
   void GetElementFaces(int el, Array<int> &faces) const;
   enum class ComputeHMode { Linear, Gradient };
   void ComputeH(ComputeHMode mode, std::unique_ptr<SparseMatrix> &H) const;
#ifdef MFEM_USE_MPI
   void ComputeParH(ComputeHMode mode, std::unique_ptr<SparseMatrix> &H,
                    OperatorHandle &pH) const;
#endif
   void GetCtFaceMatrix(int f, int side, DenseMatrix & Ct) const;
   void GetEFaceMatrix(int f, int side, DenseMatrix &E) const;
   void GetGFaceMatrix(int f, int side, DenseMatrix &G) const;
   void GetHFaceMatrix(int f, DenseMatrix &H) const;
   void GetCtSubMatrix(int el, const Array<int> &c_dofs, DenseMatrix &Ct) const;
   void MultInvNL(int el, const Vector &bu_l, const Vector &bp_l,
                  const BlockVector &x_l, Vector &u_l, Vector &p_l) const;
   void MultInv(int el, const Vector &bu, const Vector &bp, Vector &u,
                Vector &p) const;
   void ConstructGrad(int el, const Array<int> &faces, const BlockVector &x_l,
                      const Vector &u_l,
                      const Vector &p_l) const;
   void AssembleHDGGrad(int el, FaceElementTransformations *FTr,
                        NonlinearFormIntegrator &nlfi,
                        const Vector &x_f, const Vector &p_l) const;
   void AssembleHDGGrad(int el, FaceElementTransformations *FTr,
                        BlockNonlinearFormIntegrator &nlfi,
                        const Vector &x_f, const Vector &u_l, const Vector &p_l) const;

public:
   /// Constructor
   /** @param fes_u     flux space
       @param fes_p     potential space
       @param fes_c     constraint space
       @param bsymmetrize   sign convention of the mixed formulation, where
                            false keeps all terms without a change, while true
                            flips the sign of B and Mp to obtain a symmetric
                            system with -Bᵀ in the flux equation
    */
   DarcyHybridization(FiniteElementSpace *fes_u, FiniteElementSpace *fes_p,
                      FiniteElementSpace *fes_c, bool bsymmetrize = true);

   /// Destructor
   ~DarcyHybridization();

   /** @brief Sets Operator::DiagonalPolicy used upon construction of the
       linear system.
       Policies include:

       - DIAG_ZERO (Set the diagonal values to zero)
       - DIAG_ONE  (Set the diagonal values to one)
       - DIAG_KEEP (Keep the diagonal values)
   */
   void SetDiagonalPolicy(const DiagonalPolicy diag_policy_)
   { diag_policy = diag_policy_; }

   /** @brief Gets Operator::DiagonalPolicy used upon construction of the
       linear system. */
   DiagonalPolicy GetDiagonalPolicy() const { return diag_policy; }

   void SetLocalNLSolver(LSsolveType type, int iters = 1000, real_t rtol = 1e-6,
                         real_t atol = 0., int print_lvl = -1)
   {
      lsolve.type = type;
      lsolve.iters = iters;
      lsolve.rtol = rtol;
      lsolve.atol = atol;
      lsolve.print_lvl = print_lvl;
   }

   void SetLocalNLPreconditioner(LPrecType type, int iters = 1000,
                                 real_t rtol = -1., real_t atol = -1.)
   {
      lsolve.prec.type = type;
      lsolve.prec.iters = iters;
      lsolve.prec.rtol = rtol;
      lsolve.prec.atol = atol;
   }

   /// N/A, use SetConstraintIntegrators()
   void SetConstraintIntegrator(BilinearFormIntegrator *c_integ) = delete;

   /// Sets the constraint integrators
   /** Set the integrators that will be used to construct the constraint
       matrices for fluxes @a C and (if provided) stabilization contributions
       to @a E, @a G, @a D and @a H for potentials. Note the potential
       integrator is required to implement the method
       BilinearFormIntegrator::AssembleHDGFaceMatrix(). The DarcyHybridization
       object assumes ownership of the integrators, i.e. it will delete the
       integrators when destroyed. */
   void SetConstraintIntegrators(BilinearFormIntegrator *c_flux_integ,
                                 BilinearFormIntegrator *c_pot_integ);

   void SetConstraintIntegrators(BilinearFormIntegrator *c_flux_integ,
                                 NonlinearFormIntegrator *c_pot_integ);

   void SetConstraintIntegrators(BilinearFormIntegrator *c_flux_integ,
                                 BlockNonlinearFormIntegrator *c_integ);

   void SetFluxMassNonlinearIntegrator(NonlinearFormIntegrator *flux_integ,
                                       bool own = true);

   void SetPotMassNonlinearIntegrator(NonlinearFormIntegrator *pot_integ,
                                      bool own = true);

   void SetBlockNonlinearIntegrator(BlockNonlinearFormIntegrator *block_integ,
                                    bool own = true);

   /// Returns the flux constraint integrator
   BilinearFormIntegrator* GetFluxConstraintIntegrator() const { return c_bfi.get(); }

   /// Returns the potential constraint integrator
   BilinearFormIntegrator* GetPotConstraintIntegrator() const { return c_bfi_p.get(); }
   NonlinearFormIntegrator* GetPotConstraintNonlinearIntegrator() const { return c_nlfi_p.get(); }

   NonlinearFormIntegrator* GetFluxMassNonlinearIntegrator() const { return m_nlfi_p; }
   NonlinearFormIntegrator* GetPotMassNonlinearIntegrator() const { return m_nlfi_p; }

   /** @brief Not available, use AddBdrFluxConstraintIntegrator()
       or AddBdrPotConstraintIntegrator(). */
   void AddBdrConstraintIntegrator(BilinearFormIntegrator *c_integ) = delete;

   /** @brief Not available, use AddBdrFluxConstraintIntegrator()
       or AddBdrPotConstraintIntegrator(). */
   void AddBdrConstraintIntegrator(BilinearFormIntegrator *c_integ,
                                   Array<int> &bdr_marker) = delete;

   /** @brief Not available, use GetBdrFluxConstraintIntegrator()
       or GetBdrPotConstraintIntegrator(). */
   Array<BilinearFormIntegrator*> *GetBCBFI() = delete;

   /** @brief Not available, use GetBdrFluxConstraintIntegratorMarker()
       or GetBdrPotConstraintIntegratorMarker(). */
   Array<Array<int>*> *GetBCBFI_Marker() = delete;

   /// Adds flux boundary constraint integrator
   /** Add the boundary face integrator that will be used to construct the
       constraint matrix @a C. The DarcyHybridization object assumes ownership
       of the integrator, i.e. it will delete the integrator when destroyed. */
   void AddBdrFluxConstraintIntegrator(BilinearFormIntegrator *c_integ)
   { Hybridization::AddBdrConstraintIntegrator(c_integ); }

   /// Adds flux boundary constraint integrator (with a boundary marker)
   /** Add the boundary face integrator that will be used to construct the
       constraint matrix @a C. The DarcyHybridization object assumes ownership
       of the integrator, i.e. it will delete the integrator when destroyed.
       The boundary attribute marker array is referenced and must remain valid
       over the lifetime. */
   void AddBdrFluxConstraintIntegrator(BilinearFormIntegrator *c_integ,
                                       Array<int> &bdr_marker)
   { Hybridization::AddBdrConstraintIntegrator(c_integ, bdr_marker); }

   /// Get number of all integrators added with AddBdrFluxConstraintIntegrator().
   inline int NumBdrFluxConstraintIntegrators() const { return Hybridization::NumBdrConstraintIntegrators(); }

   /// Access all integrators added with AddBdrFluxConstraintIntegrator().
   BilinearFormIntegrator& GetBdrFluxConstraintIntegrator(int i) { return Hybridization::GetBdrConstraintIntegrator(i); }

   /// Access all boundary markers added with AddBdrFluxConstraintIntegrator().
   /** If no marker was specified when the integrator was added, the
       corresponding pointer (to Array<int>) will be NULL. */
   Array<int>* GetBdrFluxConstraintIntegratorMarker(int i) { return Hybridization::GetBdrConstraintIntegratorMarker(i); }

   /// Adds potential boundary constraint integrator
   /** Add the boundary face integrator that will be used to construct the
       constraint stabilization matrices @a E, @a G, @a D and @a H. Note the
       integrator is required to implement the method
       BilinearFormIntegrator::AssembleHDGFaceMatrix(). The DarcyHybridization
       object assumes ownership of the integrator, i.e. it will delete the
       integrator when destroyed. */
   void AddBdrPotConstraintIntegrator(BilinearFormIntegrator *c_integ)
   {
      boundary_constraint_pot_integs.push_back(c_integ);
      boundary_constraint_pot_integs_marker.push_back(
         NULL); // NULL marker means apply everywhere
   }

   /// Adds potential boundary constraint integrator (with a boundary marker)
   /** Add the boundary face integrator that will be used to construct the
       constraint stabilization matrices @a E, @a G, @a D and @a H. Note the
       integrator is required to implement the method
       BilinearFormIntegrator::AssembleHDGFaceMatrix(). The DarcyHybridization
       object assumes ownership of the integrator, i.e. it will delete the
       integrator when destroyed. The boundary attribute marker array is
       referenced and must remain valid over the lifetime. */
   void AddBdrPotConstraintIntegrator(BilinearFormIntegrator *c_integ,
                                      Array<int> &bdr_marker)
   {
      boundary_constraint_pot_integs.push_back(c_integ);
      boundary_constraint_pot_integs_marker.push_back(&bdr_marker);
   }

   /// Get number of all integrators added with AddBdrPotConstraintIntegrator().
   inline int NumBdrPotConstraintIntegrators() const { return boundary_constraint_pot_integs.size(); }

   /// Access all integrators added with AddBdrPotConstraintIntegrator().
   BilinearFormIntegrator& GetBdrPotConstraintIntegrator(int i) { return *boundary_constraint_pot_integs[i]; }

   /// Access all boundary markers added with AddBdrPotConstraintIntegrator().
   /** If no marker was specified when the integrator was added, the
       corresponding pointer (to Array<int>) will be NULL. */
   Array<int>* GetBdrPotConstraintIntegratorMarker(int i) { return boundary_constraint_pot_integs_marker[i]; }

   void AddBdrPotConstraintIntegrator(NonlinearFormIntegrator *c_integ)
   {
      boundary_constraint_pot_nonlin_integs.push_back(c_integ);
      boundary_constraint_pot_nonlin_integs_marker.push_back(
         NULL); // NULL marker means apply everywhere
   }
   void AddBdrPotConstraintIntegrator(NonlinearFormIntegrator *c_integ,
                                      Array<int> &bdr_marker)
   {
      boundary_constraint_pot_nonlin_integs.push_back(c_integ);
      boundary_constraint_pot_nonlin_integs_marker.push_back(&bdr_marker);
   }

   /// Get number of all non-linear integrators added with AddBdrPotConstraintIntegrator().
   inline int NumBdrPotConstraintNLIntegrators() const { return boundary_constraint_pot_nonlin_integs.size(); }

   /// Access all non-linear integrators added with AddBdrPotConstraintIntegrator().
   NonlinearFormIntegrator& GetBdrPotConstraintNLIntegrator(int i) { return *boundary_constraint_pot_nonlin_integs[i]; }

   /// Access all boundary markers added with AddBdrPotConstraintIntegrator().
   /** If no marker was specified when the non-linear integrator was added, the
       corresponding pointer (to Array<int>) will be NULL. */
   Array<int>* GetBdrPotConstraintNLIntegratorMarker(int i) { return boundary_constraint_pot_nonlin_integs_marker[i]; }

   void AddBdrConstraintIntegrator(BlockNonlinearFormIntegrator *c_integ)
   {
      boundary_constraint_nonlin_integs.push_back(c_integ);
      boundary_constraint_nonlin_integs_marker.push_back(
         NULL); // NULL marker means apply everywhere
   }
   void AddBdrConstraintIntegrator(BlockNonlinearFormIntegrator *c_integ,
                                   Array<int> &bdr_marker)
   {
      boundary_constraint_nonlin_integs.push_back(c_integ);
      boundary_constraint_nonlin_integs_marker.push_back(&bdr_marker);
   }

   /// Get number of all non-linear integrators added with AddBdrConstraintIntegrator().
   inline int NumBdrConstraintNLIntegrators() const { return boundary_constraint_pot_integs.size(); }

   /// Access all non-linear integrators added with AddBdrConstraintIntegrator().
   BlockNonlinearFormIntegrator& GetBdrConstraintNLIntegrator(int i) { return *boundary_constraint_nonlin_integs[i]; }

   /// Access all boundary markers added with AddBdrConstraintIntegrator().
   /** If no marker was specified when the non-linear integrator was added, the
       corresponding pointer (to Array<int>) will be NULL. */
   Array<int>* GetBdrConstraintNLIntegratorMarker(int i) { return boundary_constraint_nonlin_integs_marker[i]; }

   void UseExternalBdrConstraintIntegrators() = delete;

   /// Indicate that boundary flux constraint integrators are not owned
   void UseExternalBdrFluxConstraintIntegrators() { Hybridization::UseExternalBdrConstraintIntegrators(); }

   /// Indicate that boundary potential constraint integrators are not owned
   void UseExternalBdrPotConstraintIntegrators() { extern_bdr_constr_pot_integs = 1; }

   /// Prepare the DarcyHybridization object for assembly.
   /** @param ess_flux_tdof_list    essential true DOFs of the flux */
   void Init(const Array<int> &ess_flux_tdof_list) override;

   /// Specify essential boundary conditions.
   void SetEssentialBC(const Array<int> &bdr_attr_is_ess);

   /// Specify essential VDOFs.
   /** Use either SetEssentialBC() or SetEssentialTrueDofs() if possible. */
   void SetEssentialVDofs(const Array<int> &ess_vdofs_list);

   /// Specify essential true DOFs.
   void SetEssentialTrueDofs(const Array<int> &ess_tdof_list_)
   { ess_tdof_list_.Copy(ess_tdof_list); ess_tdof_list_.Copy(ess_tdof_user); }

   /// Return a (read-only) list of all essential true DOFs.
   const Array<int> &GetEssentialTrueDofs() const { return ess_tdof_list; }

   /// How a face's trace degree follows the degrees of its two elements.
   enum class TraceOrderRule
   {
      /** @brief The lower of the two. It is the configuration every existing
          case uses, and it is **not** the safe choice it looks like: it gets
          WORSE as the degree jump across an interface grows. Measured on
          `convdiff -o 1 -nx 8 -pref n`, which raises half the domain by `n`
          degrees, the relative flux error goes

              n = 1      n = 2      n = 3
              1.066e-2   1.620e-2   1.519e-2

          -- adding degrees to the refined half makes the answer worse, because
          the interface trace is held at the coarse degree and becomes a
          tighter bottleneck the richer its other side gets. Max holds
          1.051e-2, 1.051e-2, 1.054e-2 over the same three. */
      Min,
      /** @brief The higher of the two, which is the usual choice in the
          literature. It needs the HDG face quadrature to take the trace
          element's order into account, or the trace-trace block comes back
          rank-deficient -- that is fixed, see the note at the top of
          bilininteg_hdg.cpp. Measured to be *exactly redundant* on a face
          whose two neighbours agree, so it can only pay at a genuine
          `p`-interface -- and it does.

          **Where the interface is put decides whether the rule matters, and
          the two studies that look contradictory are not.** On a PRESCRIBED
          interface (`convdiff -pref`, half the domain by geometry) Max costs
          0.2 to 2.6 per cent more active dofs and returns, at a one-degree
          jump, nothing outside the last digits -- net dof efficiency 0.998 to
          1.004, exactly the wash the redundancy argument predicts. At a jump
          of two or more it reaches Min's *flux* error at 12 to 35 per cent
          fewer active dofs, and the *potential* still never sees it: ratios
          0.997 to 1.0005 at every jump and mesh. Neither rule changes the
          convergence rate; Min's penalty is an `O(h^(r+1))` term on an
          `O(h^r)` error, so the gap closes -- 0.598, 0.640, 0.758, 0.857 over
          four meshes at `-o 2 -pref 2`.

          In an ADAPTIVE loop the potential does see it, by 21 to 27 per cent
          of the dofs at fixed error over four decades
          (`anisodiff -p 5 -ks 1e2 --hp-adaptivity`, table in that file). The
          difference is not the metric but the placement: a geometric interface
          lands wherever the domain is cut, which on that problem is away from
          the layer, while an adaptive one is put exactly on the feature. A
          rule can only matter where its interface does. */
      Max,
   };

   /** @brief Derive one trace degree per face from one degree per element.

       @a cap is the ceiling, normally the degree the constraint space was
       built at. Boundary faces take their single neighbour's degree.

       **On a nonconforming mesh the unit is the hanging-node family, not the
       face.** A family carries one trace unknown, on its master face, so it
       has one degree: the rule taken over the slaves, each of which has seen
       a fine element and the coarse one. The master's own entry says nothing,
       because a master face is never integrated directly.

       This used to force the family to the ceiling instead, and that was a
       limitation of the retired route rather than a choice -- see
       SetTraceOrders() for why, and for the 0.284 / 1.06 / 3.67 it cost to
       ignore. Constraining the surplus removes it.

       **In parallel a shared face takes both ranks' degrees**, obtained by one
       exchange over face neighbours. The mesh is taken by non-const reference
       for that reason: the exchange builds face-neighbour data on it. The rule
       is applied to the same pair of degrees on both sides and min and max are
       both symmetric, so the two ranks agree by construction rather than by
       convention. */
   static void FaceOrdersFromElementOrders(Mesh &mesh,
                                           const Array<int> &elem_order,
                                           TraceOrderRule rule, int cap,
                                           Array<int> &face_order);

   /** @brief Give each face its own trace polynomial degree, for `p`-adaptivity.

       @a face_order holds one degree per mesh face, and every entry must lie
       between 0 and the degree of the constraint space, which stays a uniform
       space at the maximum degree. A face of degree `p_f` then carries
       `nt(p_f)` unknowns instead of the `nt(p_max)` slots it owns, and the
       difference is **constrained away, not retired**: the slots hold the
       ceiling-basis coefficients of a function that happens to be of degree
       `p_f`, and

           E(j, i) = phi_i^{p_f}( node_j^{p_max} )

       is the per-face matrix that says so. The reduced system is in the
       constrained unknowns, so its size really is the sum of `nt(p_f)`; there
       are no unit rows standing in for the surplus.

       **Why it is constrained rather than truncated, which is what this used
       to do.** Storing a coarse function as coarse coefficients in the first
       `nt(p_f)` slots is well defined inside DarcyHybridization, which knows
       the degrees, and wrong for every reader that does not -- and the
       space's own machinery is exactly such a reader. Three things followed
       from that and all three are gone:

       | was refused | because | cost measured then | and now |
       |---|---|---|---|
       | a hanging-node family below the ceiling | the conforming prolongation interpolates master onto slave in the ceiling basis | the error went 0.284, 1.06, 3.67 as the mesh refined | works; the demonstrator runs families below the ceiling every cycle |
       | an essential datum on a coarsened boundary face | the caller projects the datum in the ceiling basis | 21x, 0.0124 to 0.259 as the ceiling went 2, 3, 5, 8 | 0.0124172 at every one of those ceilings, to every printed digit |
       | a face shared between ranks | the two ranks order its slots by their own view of the orientation | 144 retired true DOFs at one rank, 152 at two, 162 at three | 432 constrained true DOFs at every rank count, and `pconvdiff --p-refine 1 --p-refine-x 0` gives 5.867e-4 at 1, 2, 3 and 4 ranks where it gave 5.9e-4 and 0.56 |

       All three are one cause and the constraint is one repair. The stored
       coefficients are the ceiling's own, so the prolongation, the dof
       ordering and the true-dof numbering all apply unchanged, and
       orientation stops being this route's problem: only the OWNER of a face
       builds an `E` for it, in its own ordering, and every other rank
       receives values through the space's map as it always did.

       **That it cannot change the discretisation is measured, not argued.**
       `phi_i^lo = sum_j E(j,i) phi_j^hi` POINTWISE, so a face matrix
       assembled against the coarse trace equals the ceiling one restricted by
       `E`, under any quadrature rule applied to both. The unit tests
       `"A coarse trace basis is an exact combination of the ceiling's"` and
       `"The constrained ceiling system IS the coarse system"` check the two
       halves -- the identity on one face, and `E^T H(ceiling) E == H(coarse)`
       on the assembled reduced system, with a plain slot selection alongside
       as the control that says the comparison can fail.

       The low-order face is a genuine subspace of the high-order one here,
       and BuildTraceConstraint() verifies that once per (geometry, degree)
       rather than assuming it: `R E == I`, where `R` reads a ceiling function
       back at the coarse nodes. A collection that is modal, or whose lower
       degrees are not nested, fails loudly instead of quietly discretising
       something else.

       The degrees are stored, not derived. A `p`-adaptive driver picks them
       from the neighbouring element degrees by whatever rule it wants; note
       that a face richer than *both* its neighbours is exactly redundant
       rather than wrong (measured -- see the test "A trace richer than both
       its elements is exactly redundant"), so `min` costs nothing that `max`
       would have bought there. A hanging-node family is the exception to
       "one degree per face": it carries one unknown, on its master, so
       FaceOrdersFromElementOrders() gives the whole family one degree and
       only the master's is ever read.

       **The constraint space's degree is a ceiling, not a starting point.**
       This route reuses that space's storage, so a face can only go *down*
       from the degree it was constructed with; there is no way to enrich past
       it. A driver that means to raise degrees must therefore build the
       constraint space at the highest degree the run will ever reach and start
       its faces below that, not at the degree it happens to start from. Every
       caller in the tree today builds it at the element degree, which is
       coarsening-only.

       **What that ceiling costs, measured rather than predicted.** The stored
       blocks and a constraint-space vector go as `nt(p_max)` per face
       whatever the degrees are, and so do the local blocks: they are sized
       from TraceFE(), which is the ceiling's element. The trace-dependent
       element-local work is `n_el^2 n_c + n_el n_c^2`, so it should grow as
       `r` and `r^2` in `r = nt(p_max)/nt(p_f)` -- a predicted 4.0x at the
       demonstrator's configuration, 2D quads at element degree 2 under a
       ceiling of 7, where `r = 8/3`.

       The controlled experiment is a fixed mesh with every face at degree 2,
       sweeping only the ceiling, so the answer cannot move and does not
       (t_err 1.6883e-05 at every one). Seconds, 64x64:

       | ceiling | hybridization | assembly | trace solve |
       |---|---|---|---|
       | 2 | 0.056 | 0.204 | 0.0626 |
       | 3 | 0.071 | 0.268 | 0.0627 |
       | 5 | 0.079 | 0.392 | 0.0623 |
       | 7 | 0.100 | 0.659 | 0.0634 |

       So 3.2x on assembly at the extreme ceiling -- the prediction's shape,
       a little below its size, since assembly also carries trace-independent
       element work -- and **the solve is flat to the third digit**, which is
       what constraining bought: the reduced system is the same size at every
       ceiling. Under the retired route it was not, and the ceiling cost 1.03
       to 1.19x there instead.

       Keeping the local blocks at `p_f` is possible -- it means applying `E`
       at each of the twelve gather and scatter sites instead of once in the
       prolongation -- and this measurement does not justify it. End to end
       the demonstrator's hp path is 0.67 s to 8.8e-5 and 1.32 s to 5.3e-6,
       against 0.81 s recorded before the change; but an UNCHANGED path moved
       too on the same machine (h-adaptivity 0.51 s to 0.38 s), so the
       end-to-end comparison is inside the drift and only the controlled table
       above should be quoted. At the modest ceiling a coarsening-only driver
       needs, `p_max = order + 1` and `r = 4/3`, none of this is measurable.

       **A face richer than its element is a real thing and stays one.** The
       measured claim that a trace above both its elements is exactly
       redundant was taken on a conforming mesh and does not carry over to a
       hanging node: a master face sees several fine elements, which between
       them do reach the higher modes, so the extra degrees are determined
       rather than annihilated. On the same mesh that is BETTER -- 0.0818
       against 0.0982 on an identical hanging-node mesh whose only difference
       is four family faces at degree 3 instead of 2. An earlier note said the
       opposite, comparing two runs that had refined differently by then; it
       is withdrawn.

       **Call it straight after DarcyForm::EnableHybridization() and before
       Assemble().** C, E, G and H are sized from the trace element and this
       rebuilds them, calling Reset() and discarding anything assembled. Under
       the present sizing that rebuild is a no-op, since those blocks follow
       the ceiling either way; the contract is kept as it is because keeping
       the local blocks at `p_f` would need it back, and a contract that
       tightens again later is worse than one that never loosened. Passing an
       empty array returns to a uniform trace on the same terms. */
   void SetTraceOrders(const Array<int> &face_order);

   /// Return the per-face trace degrees, empty when the trace is uniform.
   const Array<int> &GetTraceOrders() const { return tr_order; }

   /** @brief The trace finite element the DOFs of face @a f live in.

       Always the constraint space's own, which is the CEILING's element, and
       that is the point: a face's coefficients are ceiling-basis
       coefficients, of a function constrained to its own degree. Anything
       reading a trace vector -- an estimator, a reconstruction, a
       GridFunction -- therefore reads the right function without knowing the
       degrees, which is what constraining buys over truncating.

       The face's own DEGREE, which is a different question, is
       GetTraceOrders()[f]; FaceDegreeFE() is its element and is private,
       being needed only to build the constraint. The local blocks are sized
       from here, so they follow the ceiling; see SetTraceOrders() for what
       that costs and what the alternative would be. */
   const FiniteElement *TraceFE(int f) const;

   /** @brief The constraint-space VDOFs of face @a f.

       All of them: a face's storage is the ceiling's whatever its degree.
       Kept as a method rather than folded back into
       FiniteElementSpace::GetFaceVDofs() so that the twelve places which
       gather and scatter a face's block have one door to go through, which is
       where the per-face degree would re-enter if the local blocks were ever
       taken back down to it. */
   void TraceVDofs(int f, Array<int> &vdofs) const;

   /** @brief The number of trace unknowns the reduced system solves for.

       The sum over the faces this process owns of nt(p_f) * vdim -- one per
       degree of freedom a face actually has, the ceiling's surplus
       constrained away rather than retired into a unit row. With no per-face
       degrees set it is the constraint space's own true size. */
   int GetTraceTrueVSize() const;

   /** @brief The trace prolongation, from the constrained true DOFs to the
       constraint space's VDOFs, or NULL when it would be the identity.

       This is what every reader of a trace prolongation goes through once a
       per-face degree is in play. With no degrees set it returns **exactly**
       what FiniteElementSpace::GetConformingProlongation() returns, the same
       pointer and the same null, so the uniform path is unchanged by
       construction rather than by test.

       Serial only; see GetParTraceProlongation() for the parallel one. */
   const SparseMatrix *GetTraceProlongationMatrix() const;

#ifdef MFEM_USE_MPI
   /** @brief The trace prolongation in parallel: constrained true DOFs to
       constraint-space LDOFs, i.e. Dof_TrueDof composed with the per-face
       constraint. Never null.

       The composition is in that order, and that ordering is the whole of the
       shared-face repair: the constraint acts on TRUE DOFs, so only the owner
       of a face builds an E for it, in its own ordering, and every other rank
       receives LDOF values through the space's map exactly as it did for a
       uniform trace. Orientation stops being this route's problem, which is
       what the retired route could never manage -- there the meaning of a
       slot was per-rank, and no exchange repairs a disagreement about what a
       number means. */
   HypreParMatrix *GetParTraceProlongation() const;
#endif

   /** @brief GetTraceProlongationMatrix(), or its parallel counterpart, as an
       Operator. NULL only in the serial case where it would be the
       identity. */
   const Operator *GetTraceProlongation() const;

   /** @brief Prolong a constrained trace vector to constraint-space VDOFs.

       The result is a genuine ceiling-basis representation of a function of
       each face's own degree, so anything reading the trace space generically
       -- a GridFunction, the error estimator, the reconstruction, GLVis --
       reads the right function. That is what constraining buys over
       truncating, where the same slots would have held a coarser basis's
       coefficients and every such reader would have been silently wrong. */
   void ProlongTrace(const Vector &X_c, Vector &x) const;

   /** @brief Read a constraint-space VDOF vector back at each face's own
       degree.

       A left inverse of ProlongTrace(), and **not** its transpose: it
       interpolates at the coarse nodes where the transpose would integrate
       against the ceiling's. Use it for DATA -- a boundary datum, an initial
       guess -- and the transpose for residuals.

       The difference is why the essential-datum refusal existed. This map is
       a function of the FUNCTION, so restricting one function represented at
       two different ceilings gives the same answer twice; a least-squares
       pseudo-inverse is a function of the ceiling's NODE SET and does not.
       Both are left inverses, so only the second property separates them, and
       it is measured in "A coarse trace basis is an exact combination of the
       ceiling's". */
   void RestrictTrace(const Vector &x, Vector &X_c) const;

   /// Not available, use a specific Assemble*MassMatrix() instead.
   void AssembleMatrix(int el, const DenseMatrix &A) override
   { MFEM_ABORT("Not supported, system part must be specified"); }

   /// Assemble element matrix of @a Mu
   void AssembleFluxMassMatrix(int el, const DenseMatrix &A);

   /// Assemble element matrix of @a Mp
   void AssemblePotMassMatrix(int el, const DenseMatrix &D);

   /// Assemble element matrix of @a B
   void AssembleDivMatrix(int el, const DenseMatrix &B);

   /// Computes and assembles potential face matrix
   /** The provided provided potential constraint integrator (see
       SetConstraintIntegrators()) is used to compute the HDG face matrix,
       which contributes to @a D, @a E, @a G and @a H. The element
       contributions to @a D are returned in @p elmat1 and @p elmat2 together
       with the VDOFs lists @p vdofs1 and @p vdofs2. The flag for skipping
       zeros for contributions of @a H to the hybridized matrix can be set in
       @p skip_zeros. */
   void ComputeAndAssemblePotFaceMatrix(int face,
                                        DenseMatrix & elmat1, DenseMatrix & elmat2,
                                        Array<int>& vdofs1, Array<int>& vdofs2, int skip_zeros = 1);

   /// Computes and assembles potential boundary face matrix
   /** The provided provided potential constraint integrator (see
       SetConstraintIntegrators()) is used to compute the HDG boundary face
       matrix, which contributes to @a D, @a E, @a G and @a H. The element
       contributions to @a D are returned in @p elmat together with the VDOFs
       list @p vdofs. The flag for skipping zeros for contributions of @a H to
       the hybridized matrix can be set in @p skip_zeros. */
   void ComputeAndAssemblePotBdrFaceMatrix(int bface, DenseMatrix & elmat,
                                           Array<int>& vdofs, int skip_zeros = 1);

   /// Assemble the boundary element matrix A into the hybridized system matrix.
   //void AssembleBdrMatrix(int bdr_el, const DenseMatrix &A);

   /// Finalize the construction of the hybridized matrix.
   void Finalize() override;

   /// Use the stored eliminated part of the system to modify the r.h.s.
   /** @param vdofs_flux   list of VDOFs of flux @a u
       @param x            solution vector providing the VDOF values
       @param b            right hand side vector
   */
   void EliminateVDofsInRHS(const Array<int> &vdofs_flux,
                            const BlockVector &x, BlockVector &b);

   /// Use the stored eliminated part of the system to modify the r.h.s.
   /** @param tdofs_flux   list of true DOFs of flux @a u
       @param X            solution vector providing the true DOF values
       @param B            (true) right hand side vector
   */
   virtual void EliminateTrueDofsInRHS(const Array<int> &tdofs_flux,
                                       const BlockVector &X, BlockVector &B);

   /// Eliminate the given true DOFs, storing the eliminated part internally.
   /** This method works in conjunction with EliminateTraceTrueDofsInRHS() and
       allows elimination of boundary conditions in multiple right-hand sides.
       In this method, @p tdofs is a list of true DOFs. */
   void EliminateTraceTrueDofs(const Array<int> &tdofs,
                               DiagonalPolicy dpolicy = DIAG_ONE);

   /// Eliminate the essential true DOFs.
   /** This method eliminates the essential true DOFs set previously through
       SetEssentialTrueDofs() (or derived methods). See EliminateTraceTrueDofs(
       const Array<int> &, DiagonalPolicy) for details. */
   void EliminateTraceTrueDofs(DiagonalPolicy dpolicy = DIAG_ONE);

   /// Use the stored eliminated part of the hybridized matrix to modify r.h.s.
   /** This method works in conjunction with EliminateTraceTrueDofs(
       const Array<int>&, DiagonalPolicy) to modify the r.h.s.
       @param vdofs     list of true DOFs (non-directional, i.e. >= 0)
       @param x         solution vector providing the true DOF values
       @param b         right hand side vector
    */
   void EliminateTraceTrueDofsInRHS(const Array<int> &vdofs, const Vector &x,
                                    Vector &b);

   /// Use the stored eliminated part of the hybridized matrix to modify r.h.s.
   /** This method works in conjunction with EliminateTraceTrueDofs(
       DiagonalPolicy) to modify the r.h.s.
       @param x         solution vector providing the  true DOF values
       @param b         right hand side vector
    */
   void EliminateTraceTrueDofsInRHS(const Vector &x, Vector &b);

   /// Return the eliminated part of the hybridized matrix.
   /** See EliminateTraceTrueDofs() for generation of this matrix. */
   SparseMatrix& GetMatrixElim() const { return *He; };

#ifdef MFEM_USE_MPI
   /// Return the parallel hybridized operator.
   void GetParallelOperator(OperatorHandle &H_h) const { H_h = pOp; }
#endif //MFEM_USE_MPI

   /// Not available, use ReduceRHS(const BlockVector &, Vector &) instead.
   void ReduceRHS(const Vector &b, Vector &b_r) const override
   { MFEM_ABORT("Use BlockVector version instead"); }

   /// Hybridize r.h.s. of the mixed system.
   /** @param b      r.h.s. of the mixed system (VDOFs)
       @param b_r    r.h.s of the hybridized system (TDOFs)
    */
   void ReduceRHS(const BlockVector &b, Vector &b_r) const;

   /// Projects trace of the solution onto the trace variable
   /** @note The trace projection performs simple averaging of the face values,
       which may not be consistent with the implicit definition in the
       hybridized system. Therefore, the values should serve only as an
       approximation or an initial guess.
       @param sol    solution of the mixed system (VDOFs)
       @param sol_r  solution of the hybridized system (VDOFs)
    */
   void ProjectSolution(const BlockVector &sol, Vector &sol_r) const;

   /// Apply the hybridized operator.
   /** @note The DarcyHybridization object must be finalized by Finalize(). */
   void Mult(const Vector &x, Vector &y) const override;

   /// Evaluate the gradient operator at the point @a x.
   Operator &GetGradient(const Vector &x) const override;

   /** @brief Not available, use ComputeSolution(const BlockVector &,
       const Vector &, BlockVector &) instead. */
   void ComputeSolution(const Vector &b, const Vector &sol_r,
                        Vector &sol) const override
   { MFEM_ABORT("Use BlockVector version instead"); }

   /// Compute solution of the mixed system.
   /** @param b      r.h.s. of the mixed system (VDOFs)
       @param sol_r  solution of the hybridized system (TDOFs)
       @param sol    solution of the mixed system (TDOFs)
    */
   void ComputeSolution(const BlockVector &b, const Vector &sol_r,
                        BlockVector &sol) const;

   /// Total flux function
   /** @param Tr  element transformation (with set integration point)
       @param u   flux at the integration point
       @param p   potential at the integration point
       @param ut  total flux at the integration point
   */
   /** @brief The flux law, evaluated at a quadrature point, that turns the
       computed flux and potential into the total flux.

       @a u is the flux, @a p the potential and @a ut the total flux, all of
       them per equation: for a system of `neq` equations in `dim` dimensions
       @a p has `neq` entries and @a u and @a ut have `neq*dim`, with the block
       of equation `e` occupying `[e*dim, (e+1)*dim)`. That is the layout the
       block integrators build; see VectorBlockDiagonalIntegrator. */
   using total_flux_fun =
      std::function<void(ElementTransformation &Tr, const Vector &u,
                         const Vector &p, Vector &ut)>;

   /// Reconstruct the total flux from the provided solution.
   /** The total flux function is normally continuous and its finite element
       space is assumed to have equal number of DOFs at faces as the trace
       variable. For the interiors of elements, the quadrature function must
       be provided to calculate the total flux from the provided flux and
       potential values. Currrently, vector dimension of the system is not
       supported.
       @param sol    solution of the mixed system
       @param sol_r  solution of the hybridized system
       @param ut_fx  total flux function
       @param ut     total flux
   */
   void ReconstructTotalFlux(const BlockVector &sol, const Vector &sol_r,
                             total_flux_fun ut_fx, GridFunction &ut) const;

   /// Resets the assembled data
   /** @note Assumes topology of the mesh does not change, otherwise recreate
       the object. */
   void Reset() override;
};

}

#endif
