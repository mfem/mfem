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

#define MFEM_DARCY_HYBRIDIZATION_ELIM_BCS

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

   /** @brief The order in which hybridization and linearisation are applied to
       a nonlinear problem.

       The two orderings solve the same discrete problem and agree at
       convergence; they differ in what an element has to do. */
   enum class NLOrdering
   {
      /** @brief Condense first, linearise second. Eliminating the flux and the
          potential on an element is itself a nonlinear solve, run once per
          element per residual evaluation of the outer iteration, and the outer
          unknown is the trace alone. */
      CondenseThenLinearise,
      /** @brief Linearise first, condense second: Newton on the full
          (q, u, u_hat) system, with the resulting linear system hybridized.
          Every local operation is then a linear solve, which is how the method
          is defined -- Nguyen, Peraire & Cockburn, JCP 228 (2009) 8841-8855,
          eqs (14)-(18). See SetNonlinearOrdering(). */
      LineariseThenCondense,
   };

   /** @brief How the loop over elements that builds the reduced system is
       executed.

       Static condensation is defined by each element's flux and potential
       being eliminable independently of every other, so the loop is parallel
       by construction rather than by accident. What is not independent is the
       result: a trace dof lives on a face and a face has two elements, so the
       trace matrix receives two contributions per entry.

       Both modes assemble the *same matrix, entry for entry and bit for bit*.
       That is a property worth having rather than a coincidence -- see
       SetAssemblyMode() for what it costs and why it was paid. */
   enum class AssemblyMode
   {
      /** @brief One thread, elements in order. The historical behaviour and
          the default: no existing caller pays anything for the other mode. */
      Serial,
      /** @brief The element-local work -- factoring A, forming and factoring
          the Schur complement, and evaluating the element's blocks of H -- is
          run in parallel; the scatter into the trace matrix stays serial and
          in element order. Requires MFEM_USE_OPENMP and MFEM_THREAD_SAFE. */
      Threaded,
   };

   /** @brief How the element-local blocks are factored.

       The flux mass A and the potential block D are factored one element at a
       time, and each element's factorisation is independent of every other's
       -- that independence is what static condensation is. These modes differ
       only in how that loop is written, not in what it computes. */
   enum class LocalFactorMode
   {
      /** @brief One LUFactors per element, in element order. The historical
          behaviour and the default: no existing caller pays anything for the
          other mode. */
      Serial,
      /** @brief The whole array in one BatchedLinAlg::LUFactor() call, which
          makes the device path a backend selection rather than new kernels --
          see BatchedLinAlg::SetActiveBackend(). Requires every element's
          block to be the same size; CanBatchLocalFactor() answers that in
          advance, and the loop above is taken when it is false. */
      Batched,
   };

   /** @brief Whether the reduced (trace) gradient is assembled as a sparse
       matrix or only applied.

       Both build the same operator and both need the local blocks factored
       once per linearisation -- that is what condensation is, and no setting
       avoids it. They differ in whether the global trace matrix is formed.

       Assembling costs one local back-substitution per trace dof of the
       element, so it is worth roughly that many matrix-free applications: six
       for k = 1 triangles, near a hundred for k = 3 hexes. Against that it
       buys a matrix a direct solver or an algebraic preconditioner can use, so
       it wins outright at low order. MatrixFree carries no global matrix at
       all, does no sparse gather or scatter, and does identical work on every
       element; the case for it is memory and device fitness, and the cost of
       it is that GS, AMG and a direct factorisation are no longer available to
       precondition the trace solve. */
   enum class GradientMode
   {
      /// Assemble the Schur complement; GetGradient() returns a SparseMatrix.
      Assembled,
      /// Apply it; GetGradient() returns an Operator with no stored matrix.
      MatrixFree,
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
   /** @brief Essential *trace* true DOFs, in the constraint space @a c_fes.

       Not flux dofs, which is what this said for long enough to mislead an
       outside user into writing a substitute for the accessor below. The flux
       ones are Init()'s @a ess_flux_tdof_list and are not retained here.
       Every setter -- SetEssentialBC(), SetEssentialVDofs(),
       SetEssentialTrueDofs() -- fills this from @a c_fes, and these are the
       rows the diagonal policy pins in the reduced operator: Mult() zeroes the
       residual on them and GetGradient() leaves a unit row. */
   Array<int> ess_tdof_list;

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

   NLOrdering nl_ordering{NLOrdering::CondenseThenLinearise};
   GradientMode grad_mode{GradientMode::Assembled};
   AssemblyMode asm_mode{AssemblyMode::Serial};
   LocalFactorMode lfac_mode{LocalFactorMode::Serial};

   mutable long num_local_nl_iters{0};

   /** @brief The point the local Jacobian in @a Af_data, @a Df_data and
       @a Bnl_data was assembled at.

       Only used by NLOrdering::LineariseThenCondense, and refreshed only by
       GetGradient(); see SetNonlinearOrdering(). @a lin_trace is a trace
       L-vector, @a lin_u and @a lin_p are flux and potential L-vectors.

       The local residual here is deliberately not retained. Every evaluation
       recomputes it at the fields it is actually using, which is what keeps
       the reduced gradient the derivative of the reduced residual; see
       MultInvLin(), which also records how completely these fields have to
       solve the local problem for that to hold. */
   mutable Vector lin_trace, lin_u, lin_p;
   /// Scratch for the next linearisation point, swapped in when it is complete.
   mutable Vector lin_u_next, lin_p_next;
   mutable bool lin_valid{false};

   std::unique_ptr<SparseMatrix> He;
   OperatorHandle pHe;
   mutable std::unique_ptr<SparseMatrix> Grad;
   mutable OperatorHandle pGrad;

   /** @brief Trace dofs whose row in the assembled reduced gradient would be
       empty, and which the diagonal policy therefore regularises.

       Rebuilt by GetGradient() in GradientMode::MatrixFree, where there is no
       matrix for SetDiagIdentity() to act on. A row is empty when every face
       carrying the dof contributes nothing through C, G or H, which is what
       happens to a boundary trace dof of a problem whose constraint has no
       boundary face term. Without this the two modes are different operators
       -- the matrix-free one singular exactly where the assembled one was
       regularised to be nonsingular. */
   mutable Array<int> mf_diag_marker;
   /// Fill @a mf_diag_marker; see it.
   void MarkEmptyTraceRows() const;

   friend class Gradient;
   /// The reduced gradient applied rather than assembled; see GradientMode.
   class Gradient : public Operator
   {
      const DarcyHybridization &dh;
   public:
      Gradient(const DarcyHybridization &dh)
         : Operator(dh.Width()), dh(dh) { }

      void Mult(const Vector &x, Vector &y) const override;
   };

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

   class ParGradient : public Operator
   {
      const DarcyHybridization &dh;
   public:
      ParGradient(const DarcyHybridization &dh)
         : Operator(dh.c_fes.GetTrueVSize()), dh(dh) { }

      void Mult(const Vector &x, Vector &y) const override;
   };
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
   /** @a force_relin makes a MultNlMode::Grad pass re-substitute and
       relinearise even at the trace it is already linearised about. Only the
       initialisation below wants that; a gradient asked for twice at one trace
       must be idempotent, which is why it is not the default. */
   void MultNL(MultNlMode mode, const Vector &bu, const Vector &bp,
               const Vector &x, Vector &y, bool force_relin = false) const;
   void MultNL(MultNlMode mode, const BlockVector &b, const Vector &x,
               Vector &y) const
   { MultNL(mode, b.GetBlock(0), b.GetBlock(1), x, y); }
   void ParMultNL(MultNlMode mode, const BlockVector &b, const Vector &x,
                  Vector &y) const;
   void InvertA();
   void InvertD();
   /** @brief The size every element's block has in @a f_offsets, or -1
       when they are not all equal. */
   static int UniformBlockSize(const Array<int> &f_offsets, int NE);
   void GetElementFaces(int el, Array<int> &faces) const;
   /** @brief What ComputeH() is being asked for.

       GradientFactorOnly does the first half of Gradient -- factoring A and
       the Schur complement of every element, with the Jacobian's (0,1) block
       -- and stops before assembling the global matrix. It is what
       GradientMode::MatrixFree needs, and it is also all the initialisation
       pass in MultNL() ever needed: that pass discarded the matrix it built. */
   enum class ComputeHMode { Linear, Gradient, GradientFactorOnly };
   /// Total trace dofs on @a faces, which is the side of an element's H block.
   int GetElementTraceSize(const Array<int> &faces) const;
   /** @brief The element-local half of ComputeH() for one element: factor A,
       form and factor the Schur complement, and -- unless the mode is
       GradientFactorOnly -- evaluate this element's face-pair blocks of H
       into @a Hel.

       Everything it reads is either shared and const or indexed by @a el, and
       everything it writes is indexed by @a el, so two elements may run
       concurrently. Its scratch is local for the same reason: the scratch the
       single loop hoisted and reused across elements is exactly what threads
       cannot share.

       @a Hel receives the (f2,f1) blocks contiguously, f1 outer and f2 inner,
       in the order ScatterElementH() replays them; it may be NULL when the
       mode is GradientFactorOnly. */
   void ComputeElementH(int el, ComputeHMode mode, real_t *Hel) const;
   /** @brief Add the blocks ComputeElementH() left in @a Hel to @a H.
       Serial by contract -- see SetAssemblyMode(). */
   void ScatterElementH(int el, const real_t *Hel, SparseMatrix &H) const;
   /// Elements per chunk of the element loop; see ComputeH().
   int AssemblyChunkSize(int NE) const;
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
   /** @brief The flux and potential the linearisation implies for the trace
       @a x_l, that is (q, u)(L) of SetNonlinearOrdering(). */
   /** @brief Whether the retained linearisation belongs to the trace @a x,
       compared bit for bit: anything else is a different iterate. */
   bool LinearisedAt(const Vector &x) const;
   /** @a corrections is how many frozen-Jacobian local Newton steps follow the
       affine prediction, or a negative value to iterate to the tolerance
       SetLocalNLSolver() carries, keeping the best iterate seen.

       Evaluating the reduced operator uses one. Forming a linearisation point
       iterates, because the retained fields' own local residual is what
       limits the accuracy of the gradient: it used to take a fixed two steps
       and the gradient was then wrong by 3e-04 on a stiff source that still
       converges. MultInvLin() carries that measurement. */
   void MultInvLin(int el, const Array<int> &faces, const BlockVector &x_l,
                   const Vector &bu_l, const Vector &bp_l, Vector &u_l,
                   Vector &p_l, int corrections) const;
   /** @brief Record @a el's contribution to the linearisation point: the
       fields and the local Jacobian there. */
   void Relinearise(int el, const Array<int> &faces, const BlockVector &x_l,
                    const Vector &u_l, const Vector &p_l) const;
   /// The local nonlinear residual of @a el at (@a u_l, @a p_l).
   void LocalResidual(int el, const Array<int> &faces, const BlockVector &x_l,
                      const Vector &bu_l, const Vector &bp_l,
                      const Vector &u_l, const Vector &p_l,
                      Vector &ru_l, Vector &rp_l) const;
   /** @brief Apply the inverse of the local block system to (@a bu, @a bp).

       With @a with_bnl, the (0,1) block is taken to be the Jacobian's, that
       is -/+B^T plus the solution-dependent d(flux residual)/dp of
       @a Bnl_data, rather than the linear -/+B^T alone. The Schur complement
       held in @a Df_data must have been built the same way, which is what
       ComputeH(ComputeHMode::Gradient) does. */
   void MultInv(int el, const Vector &bu, const Vector &bp, Vector &u,
                Vector &p, bool with_bnl = false) const;
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

   /** @brief Choose whether a nonlinear problem is condensed and then
       linearised, or linearised and then condensed. The default is
       NLOrdering::CondenseThenLinearise, which is what every caller written
       before this had.

       Under NLOrdering::LineariseThenCondense the object handed to the outer
       solver is the condensed Jacobian rather than the derivative of a
       condensed residual, and no element assembles a local Jacobian per local
       iteration: GetNumLocalNLIterations() stays at zero, because every local
       step is a solve with the ONE factorisation M already holds. What the
       reduced operator computes is

           (q, u)(L)  =  (q, u)_lin - M^-1 [C; E] (L - L_lin)
                         followed by frozen-Jacobian local corrections
           F(L)       =  the trace residual at (L, q(L), u(L))

       where the linearisation point (L_lin, (q, u)_lin) and the factored
       local Jacobian M are established at L by whichever of Mult() or
       GetGradient() reaches a trace they are not already at.

       Evaluating takes one correction. ESTABLISHING the linearisation point
       iterates to the tolerance SetLocalNLSolver() carries, and that is not a
       detail: GetGradient() is the Schur complement of the Jacobian at the
       retained fields, so it is the derivative of Mult() only as far as those
       fields solve the local problem. A fixed budget of two steps shipped for
       a while and put the gradient 3e-04 out on a stiff source that still
       converged; see MultInvLin() for the sweep, and the unit test "The
       reduced gradient survives a stiff local problem" for the pin.

       The local residual is deliberately NOT retained and does not appear
       above. An earlier version of this comment carried a "- r_lin" in the
       prediction and listed r_lin among the things GetGradient() keeps.
       Applying a retained residual there is precisely the defect that cost
       the gradient its exactness, and it is fixed; the comment described the
       bug rather than the code for some time after.

       dF/dL is the condensed Jacobian, because (q, u)(L) solves the
       linearised local equations exactly and its sensitivity is the Schur
       complement itself -- to the extent the retained fields solve the local
       problem, which is what the paragraph above is about. Where the
       frozen-Jacobian correction cannot converge, no number of steps recovers
       it: the guard in MultInvLin() then keeps the best iterate and stops,
       and the gradient is as good as that point is. Measured on a pedestal
       source at n = 8, k = 1, that boundary sits between widths 0.05 and 0.02;
       inside it the gradient matches a central difference to round-off, and
       outside it neither ordering does.

       SetLocalNLSolver()'s iteration cap and tolerances govern the
       correction loop that forms the linearisation point, and they matter for
       the same reason they matter in the other mode: an inexact local solve is
       itself an error, and here it is the gradient's. The solver TYPE and
       SetLocalNLPreconditioner() are inert -- the correction is a Newton step
       on the factors M already holds, so there is nothing to choose.

       **Mult() linearises at its own argument, and that is what makes an
       ordinary NewtonSolver work.** It did not always. The condition guarding
       the establishing pass asked whether there was a linearisation anywhere
       rather than whether there was one *here*, so NewtonSolver -- which
       evaluates the residual before it asks for the gradient, on every step
       and not only the first -- would take r at x_k about the linearisation
       retained at x_{k-1}, then J at x_k, and solve a step from a residual
       and a Jacobian belonging to different operators. On a stiff semilinear
       source that failed outright: reported from a caller, three of seven
       benchmark configurations converged under CondenseThenLinearise and did
       not converge in sixty iterations under this ordering. Two were the
       mismatched residual and gradient above. The third was the fixed
       correction budget described earlier, and with both repaired all seven
       converge -- 7, 6, 8, 7, 8, 6 and 9 iterations against the exact
       ordering's 7, 6, 8, 7, 10, 7 and 10.

       That is the caller's reproducer and not a general claim, and the wider
       sweep is worth having beside it. Over 144 configurations of the same
       source (n = 8..24, k = 1..3, six widths from 0.02 to 0.001), the cases
       where CondenseThenLinearise converges and this ordering does not went
       from six to three, with none added, and where both converge this
       ordering took fewer iterations in 15 and more in 10. Six further cases
       stopped converging, every one of them a case CondenseThenLinearise also
       fails -- so no parity was lost, though converging on a problem the
       exact ordering cannot solve was never evidence of much.

       It costs nothing in a plain Newton loop: the advance happens in Mult()
       instead of in GetGradient(), which then finds the linearisation already
       at x and reuses it -- one advance per iterate either way. A line search
       does pay one advance per trial point, which is the price of the trial
       residual being the residual.

       A property this mode does NOT have, recorded so that it is not mistaken
       for a defect and "fixed": across a linearisation that *advances* onto a
       trace, Mult() is not a function of that trace alone -- the fields it
       starts the advance from are the previous point's. The gap was measured
       at 5.0e-10, 4.8e-06 and 1.1e-02 as the nonlinearity grew, and it cannot
       be closed within this ordering: exactness there needs the local problem
       solved exactly, which is CondenseThenLinearise. Only the two smaller
       values are pinned by a test.

       Making the linearisation point iterate did not remove that and was not
       expected to; it moved where it bites. Measured on a pedestal source at
       n = 16, k = 1 by evaluating at a trace, wandering, and returning: where
       the correction converges the gap is round-off, 1.5e-16 at widths 0.05
       and 0.02 as it was before. Where it does not converge, the guard
       truncates at a step count that depends on the data, and that is a
       discontinuity the fixed budget did not have -- 4.3e-08 at width 0.01
       against 1.5e-16 before, and between 1e-03 and 4e-01 below that. Better
       where the ordering works and worse where it does not, which is the
       trade this mode is.

       @note This mode places no requirement on the SOLVER, and the API that
       used to exist for one is gone. There was a @warning here that the
       linearisation advanced only in GetGradient(), so an outer iteration had
       to ask for a gradient once per accepted iterate: KINSolver::SetJFNK(true)
       needed SetMaxSetupCalls(1) against KINSOL's default of ten, and a
       gradient-free solver had to call AdvanceLinearisation() by hand, with
       SetMaxEvalsWithoutAdvance() guarding the requirement. None of it holds
       now, and all three methods have been removed rather than left as
       no-ops. A Jacobian-free Newton-Krylov solve that never asks for a
       gradient reaches the reference answer to 2.5e-15, where the same solve
       previously converged to round-off on a frozen operator and was wrong in
       the fourth digit. */
   void SetNonlinearOrdering(NLOrdering ordering);

   /** @brief Choose how the element loop that builds the reduced system runs.
       AssemblyMode::Serial by default, so nothing existing changes.

       AssemblyMode::Threaded parallelises the element-local work and leaves
       the scatter serial and in element order. The scatter is not threaded
       because it cannot safely be: SparseMatrix::AddSubMatrix() reaches the
       matrix through SetColPtr(), and an unfinalized SparseMatrix carries one
       @a current_row, one column-pointer scratch array and one RowNode
       allocator for the whole matrix. Two threads adding to *disjoint rows*
       still collide on those, and the observed failure is a hang, not a wrong
       answer -- so element colouring, which buys disjoint rows, does not make
       this loop safe on its own.

       The two modes agree bit for bit, so a test may assert equality rather
       than a tolerance -- but note what does and does not buy that. It is not
       the ordering: a trace dof lives on a face and a face has at most two
       elements, so each entry is a sum of at most two terms and IEEE addition
       of two terms is order-independent. Scattering a chunk back-to-front was
       measured and changed nothing. What exactness rests on is that the
       element-local arithmetic is per-element and so reassociates nothing.
       Element order is kept because it is free and deterministic.

       Aborts if the build cannot honour it: MFEM_USE_OPENMP is what makes it
       parallel, and MFEM_THREAD_SAFE is what stops GetElementFaces() keeping
       its scratch in a function-local static. Falling back quietly would
       report a speedup nobody got. */
   void SetAssemblyMode(AssemblyMode mode);

   /// The mode set by SetAssemblyMode().
   AssemblyMode GetAssemblyMode() const { return asm_mode; }

   /** @brief Choose how the element-local blocks A and D are factored.
       LocalFactorMode::Serial by default, so nothing existing changes.

       LocalFactorMode::Batched sends the whole array through
       BatchedLinAlg::LUFactor(). Note what that does and does not buy on a
       host: the native backend is an mfem::forall, so it threads only in a
       build with MFEM_USE_OPENMP *and* a device configured to use it
       (Device("omp")), and is otherwise the same serial traversal reached by
       a different route. Its real payoff is the gpu_blas and magma backends.

       The two modes agree **bit for bit** in a build without LAPACK, and that
       is a fact about the code rather than a hope: BatchedLinAlg's native
       backend calls kernels::LUFactor(), and LUFactors::Factor() *is* that
       same routine when MFEM_USE_LAPACK is undefined -- the same partial
       pivoting, the same 1-based ipiv. With LAPACK, LUFactors::Factor() calls
       getrf_ instead, whose blocked update reassociates the arithmetic, and
       the two then agree only to round-off. A test asserting equality must
       know which build it is in.

       One behaviour differs deliberately. A block whose factorisation meets
       an exact zero pivot aborts here, where the serial loop discards
       LUFactors::Factor()'s return value and carries on into whatever the
       division by it produces.

       **What it is worth, measured, so that nobody has to guess.** In an
       MFEM_USE_OPENMP build with Device("omp"), the factorisation alone
       (NE blocks of n*n, best of five) speeds up like this against the
       serial loop:

           threads      n=8    n=16    n=32    n=64
                 1     0.85    1.08    0.96    1.00
                 2     1.43    1.70    1.83    1.56
                 4     2.45    3.10    3.17    2.76
                 8     3.82    5.56    4.52    4.94

       and the factors agree to the last bit at every thread count, pivots
       included. But the *in situ* difference -- the wall time of
       DarcyForm::Assemble(), whose only difference between the two modes is
       this call -- stayed inside run-to-run scatter at every size tried, from
       nx=24 at order 5 to nx=128 at order 2, with deltas of both signs.

       The reason is worth knowing before anyone spends time here.
       InvertA() and InvertD() run **once**, from Finalize(), and only for
       LocalOpType::PotNL and FluxNL. The factorisation that runs once per
       *linearisation* is the one in ComputeElementH(), which factors A itself
       unless the local operator is PotNL -- and that one is already inside
       the loop AssemblyMode::Threaded parallelises. So this setting batches
       the cold path. Its value is that it makes the device backends reachable
       for that work at all, not that it moves a host solve. Reaching the hot
       path means factoring all of A in one batched pre-pass before ComputeH()'s
       element loop and having ComputeElementH() skip it, which is a larger
       change than this one and is not made here. */
   void SetLocalFactorMode(LocalFactorMode mode);

   /** @brief Whether LocalFactorMode::Batched would actually be taken, which
       needs every element's A block, and every element's D block, to be the
       size of every other's.

       A uniform mesh at a uniform order is **not** enough, and assuming it is
       is the trap here. Af_f_offsets sizes each element's block by counting
       that element's *free* hat dofs, and a hat dof is essential when it
       depends only on ess_flux_tdof_list -- so any problem with essential
       flux dofs gives its boundary elements a smaller block than its interior
       ones on a perfectly uniform mesh. The question is therefore asked of
       the offsets themselves, never of the mesh and the order.

       Valid once Init() has built the offsets. */
   bool CanBatchLocalFactor() const;

   /** @brief Choose whether GetGradient() assembles the reduced system or only
       applies it. See GradientMode; the default is Assembled, which is what
       every caller written before this existed gets.

       GradientMode::MatrixFree returns an Operator with no stored matrix, so a
       caller must solve with something that needs only the action -- an
       unpreconditioned Krylov method, or one preconditioned by something not
       built from the matrix. GSSmoother, UMFPackSolver and the algebraic
       preconditioners all require a SparseMatrix and will abort.

       Not supported when only the flux mass is nonlinear
       (LocalOpType::FluxNL): the Schur complement has nowhere to live there,
       @a Df_data being occupied by the factored linear potential mass.
       GetGradient() aborts rather than returning something wrong. */
   void SetGradientMode(GradientMode mode);

   /** @brief The number of local nonlinear iterations performed, summed over
       elements and over every residual and gradient evaluation.

       Zero for a linear problem, and zero under
       NLOrdering::LineariseThenCondense, which is what says the ordering
       really changed rather than merely working. */
   long GetNumLocalNLIterations() const { return num_local_nl_iters; }

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

   /// Specify essential boundary conditions on the trace.
   /** Takes a *boundary attribute* marker and produces essential true DOFs of
       the constraint (trace) space, not of the flux -- the flux ones are
       Init()'s argument. See @a ess_tdof_list. */
   void SetEssentialBC(const Array<int> &bdr_attr_is_ess);

   /// Specify essential VDOFs of the constraint (trace) space.
   /** Use either SetEssentialBC() or SetEssentialTrueDofs() if possible. */
   void SetEssentialVDofs(const Array<int> &ess_vdofs_list);

   /// Specify essential true DOFs of the constraint (trace) space.
   void SetEssentialTrueDofs(const Array<int> &ess_tdof_list_)
   { ess_tdof_list_.Copy(ess_tdof_list); }

   /// Return a (read-only) list of the essential *trace* true DOFs.
   /** These index the reduced system, so this is the list a caller needs to
       compare GetGradient() against a finite difference of Mult(): the
       residual is masked on them and the Jacobian carries a unit row, so they
       have to be left out or the comparison is meaningless. See
       @a ess_tdof_list. */
   const Array<int> &GetEssentialTrueDofs() const { return ess_tdof_list; }

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
