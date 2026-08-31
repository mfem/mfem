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

   GradientMode grad_mode{GradientMode::Assembled};
   AssemblyMode asm_mode{AssemblyMode::Serial};
   LocalFactorMode lfac_mode{LocalFactorMode::Serial};

   mutable long num_local_nl_iters{0};


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
   /** @a AtFields and @a GradAtFields are the NPC modes: the flux and
       potential are Newton STATE, supplied in @a darcy_u and @a darcy_p, and
       no local solve, substitution or linearisation happens at all. Every
       other mode produces the fields from the trace one way or another, which
       is what a reduced operator on the trace alone has to do and what NPC
       does not do. See NPCResidual(). */
   enum class MultNlMode { Mult, Sol, Grad, GradMult, AtFields, GradAtFields };
   /** @a r_local is where MultNlMode::AtFields writes the local rows of the
       full residual, and is required by that mode and ignored by every other.
       It is a parameter rather than a member because a member would smuggle
       state between calls, and because adding one to this class changes its
       layout -- which, with no header dependency tracking in this build,
       silently corrupts every translation unit that was not recompiled.

       NPC's reduction and recovery have their own element loops rather than
       sharing ReduceRHS()/ComputeSolution(): those two apply the LINEAR (0,1)
       block and negate the potential block on the way in, both right for a
       linear system and wrong for a Jacobian. */
   void MultNL(MultNlMode mode, const Vector &bu, const Vector &bp,
               const Vector &x, Vector &y,
               BlockVector *r_local = nullptr) const;
   void MultNL(MultNlMode mode, const BlockVector &b, const Vector &x,
               Vector &y, BlockVector *r_local = nullptr) const
   { MultNL(mode, b.GetBlock(0), b.GetBlock(1), x, y, r_local); }
   void ParMultNL(MultNlMode mode, const BlockVector &b, const Vector &x,
                  Vector &y) const;
   /** @brief The half of a gradient that follows the fields existing: factor
       the local blocks, form the Schur complement, and hand back the reduced
       trace operator -- assembled or matrix-free according to
       SetGradientMode().

       @a mode selects how the element loop obtains the fields, and is the only
       difference between GetGradient() (MultNlMode::Grad, fields produced from
       the trace) and NPCGradient() (MultNlMode::GradAtFields, fields supplied
       as Newton state). */
   Operator &ReducedGradient(MultNlMode mode, const Vector &x_tr) const;
#ifdef MFEM_USE_MPI
   /// The same, assembling a HypreParMatrix. @a x_tr is in TRUE dofs.
   Operator &ParReducedGradient(MultNlMode mode, const Vector &x_tr) const;
#endif
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
       @a x_l, by a local nonlinear solve. */
   /** @brief The trace space's prolongation from true dofs to L-dofs, or
       NULL when the two coincide and no mapping is needed.

       NPC's element loops work in L-dofs, as every loop in this class does;
       its public interface is in true dofs, as every MFEM Operator's is. This
       is the one place the two meet. The flux and potential need no such
       mapping under NPC because it refuses anything but a discontinuous flux
       space, so their L-dofs are their true dofs. */
   const Operator *TraceProlongation() const;
   /** @brief NPC's shared precondition: finalized, a discontinuous flux
       space, and not LocalOpType::FluxNL. The last is the guard the reduced
       operator never had; the reason is at the definition. */
   void NPCCheck() const;
   /// A correctly sized zero load for a gradient pass; see the definition.
   void ZeroLoad(BlockVector &b, bool true_dofs) const;

   /// The local nonlinear residual of @a el at (@a u_l, @a p_l).
   void LocalResidual(int el, const Array<int> &faces, const BlockVector &x_l,
                      const Vector &bu_l, const Vector &bp_l,
                      const Vector &u_l, const Vector &p_l,
                      Vector &ru_l, Vector &rp_l) const;
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

   /** @brief A consequence of a correct Jacobian that a caller can mistake
       for a regression, kept here because it cost one a day's debugging.

       **A better Jacobian can converge to a DIFFERENT solution.** Where a
       coarse discretisation carries more than one, an iteration driven by an
       inaccurate gradient wanders and can settle on the branch a Picard
       iteration finds; with the gradient right it converges faster and stays
       on its own. A caller had a test pinning Newton against Anderson-Picard
       on one mesh at 1e-6, and after a gradient fix it read 9.1e-05 --
       bit identical when the tolerance was tightened by four orders, so both
       iterations were fully converged and their fixed points genuinely
       differed, at 1e-13 on two other meshes and 3e-06 on a third with no
       trend. That is not a defect and not a discretisation regression: it is
       a gate that was green for the wrong reason. Pinning "two solvers agree"
       on a single coarse mesh should be a sweep. */

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

   /** @name The NPC method: Newton on the full (q, u, lambda) system

       Nguyen, Peraire & Cockburn, JCP 228 (2009) 8841-8855, eqs (14)-(18).
       These four calls are one Newton step of it, exposed raw so that the
       shape of the method is visible. **For ordinary use take
       DarcyNPCOperator and DarcyNPCSolver instead**, which wrap them as an
       MFEM Operator over the full (q, u, lambda) vector that NewtonSolver
       drives with no special support.

       What cannot be wrapped is an Operator on the TRACE ALONE: the flux and
       the potential are Newton state here, and a trace-only operator has
       nowhere to put them. A mode that tried -- NLOrdering, and a
       LineariseThenCondense that claimed to be this method -- is deleted; it
       was a condensation in disguise, measurably slower than
       CondenseThenLinearise and unable to solve problems it solved.

       One step, given a state (@a x, @a x_tr) and the load @a b:

           NPCResidual (b, x, x_tr, r, r_tr)   F(q, u, lambda)
           S = NPCGradient (x, x_tr)           factor J; S is the reduced
                                               H - C' M^-1 [C; E], assembled
                                               or matrix-free
           NPCReduce   (r, r_tr, b_tr)         -(F_lambda - C' M^-1 F_local)
           solve S dtr = b_tr                  the caller's linear solver
           NPCRecover  (r, dtr, dx)            -M^-1 (F_local + [C; E] dtr)
           x += dx;  x_tr += dtr

       That is **one local factorisation and one local linear solve per outer
       step**, no local nonlinear iteration anywhere, and therefore nothing to
       globalise locally. The convergence test belongs on the full residual
       (@a r together with @a r_tr), which is the other half of what makes it
       NPC: a test on the trace residual alone is judging half the system.

       Call NPCGradient() before NPCReduce() and NPCRecover(): both need the
       factored local blocks and the Schur complement it leaves behind, and
       both apply the Jacobian's (0,1) block rather than the linear one.

       **What it delivers, measured.** On a problem whose full system is linear
       one step is exact -- the residual goes 6.96e-01 to 6.22e-15, from any
       starting point -- which is the check that falsifies the elimination
       algebra if anything in it is wrong. On the pedestal source it converges
       quadratically in the full residual: 6.7e-01, 1.5e-02, 2.8e-04, 1.2e-07,
       2.3e-14, with GetNumLocalNLIterations() identically zero. The two
       gradient modes agree at every iterate above round-off.

       And it solves stiff problems the deleted trace-only mode could not. Of
       the four configurations where CondenseThenLinearise converges and that
       mode did not, three fall to NPC with a backtracking line search on the
       full residual -- 13, 10 and 17 steps -- and the fourth stalls at
       2.9e-03, which is ordinary Newton stagnation. Undamped, NPC wanders on all four
       exactly as any cold Newton does: **the globalisation this method wants
       is on the OUTER step and there is none to do locally**, which is the
       whole point of the ordering. A line search here is well defined for a
       reason worth keeping in view -- the fields and the trace scale together
       because both are state, where a line search on a trace-only operator
       scales the trace and leaves the field update to whatever the
       substitution makes of it.

       @note The flux space must be discontinuous, which is the HDG case.
       An H(div) flux makes the local rows of @a r a conforming scatter with
       sign conventions this has not been checked against, and the RT paths are
       deliberately left alone. */
   ///@{
   /** @brief The residual of the full system at the given state: no local
       solve, no substitution, no linearisation.

       @a r's potential block carries the sign convention of the symmetrized
       system when that is in force, which is what NPCReduce() and
       NPCRecover() consume; its norm is unaffected and nothing else should
       read it. */
   void NPCResidual(const BlockVector &b, const BlockVector &x,
                    const Vector &x_tr, BlockVector &r, Vector &r_tr);
   /** @brief Assemble and factor the Jacobian at the same state, and return
       the reduced trace operator S = H - C' M^-1 [C; E].

       **Whether S is assembled at all is SetGradientMode()'s choice**, exactly
       as it is for GetGradient(). GradientMode::Assembled returns a
       SparseMatrix, so a direct solve or an algebraic preconditioner works;
       GradientMode::MatrixFree returns an Operator that applies S one element
       at a time with nothing stored, for a Krylov method that needs only the
       action. NPCReduce() and NPCRecover() work identically either way --
       both modes factor the local blocks and form the Schur complement, and
       it is only the global trace matrix that the matrix-free mode declines to
       build.

       Not available for LocalOpType::FluxNL in matrix-free mode, for the
       reason SetGradientMode() gives.

       The returned reference does not outlive the next call. */
   Operator &NPCGradient(const BlockVector &x, const Vector &x_tr);
   /// @brief The right-hand side of eq (18) for the trace increment.
   void NPCReduce(const BlockVector &r, const Vector &r_tr,
                  Vector &b_tr) const;
   /// @brief The local increments implied by a trace increment @a dtr.
   void NPCRecover(const BlockVector &r, const Vector &dtr,
                   BlockVector &dx) const;
   ///@}

   /** @brief The number of local nonlinear iterations performed, summed over
       elements and over every residual and gradient evaluation.

       Zero for a linear problem, and zero under NPC -- see NPCResidual() --
       which is the acceptance signal that NPC really is running a single
       local linear solve per outer step rather than a condensation. */
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

/** @brief The NPC method as an ordinary MFEM Operator on the full
    (q, u, lambda) system, so that NewtonSolver or KINSolver can drive it.

    Pair it with DarcyNPCSolver, which inverts the Jacobian by hybridized
    elimination:

        DarcyNPCOperator npc(*darcy.GetHybridization(), offsets, load);
        DarcyNPCSolver   lin(trace_solver);
        NewtonSolver newton;
        newton.SetOperator(npc);
        newton.SetSolver(lin);
        newton.Mult(zero, x);        // x = (q, u, lambda)

    @a offsets is size 4 and partial-summed over the flux, potential and trace
    vector sizes; @a load is the (flux, potential) right-hand side.

    **The unknown is the whole system, and that is the point.** A reduced
    operator on the trace alone cannot be NPC, because the fields would have to
    be a function of the trace and here they are Newton state -- which is
    exactly what an Operator over all three blocks gives them. So the outer
    solver needs no special support, its convergence test is on the full
    residual by construction, and its line search scales the fields and the
    trace together. An earlier version of this comment said NewtonSolver could
    not drive NPC "because it has nowhere to keep the fields"; it keeps them in
    @a x, and the claim was about a trace-only operator rather than about
    NewtonSolver. */
class DarcyNPCOperator : public Operator
{
public:
   DarcyNPCOperator(DarcyHybridization &dh, const Array<int> &offsets,
                    const BlockVector &load);

   /// The full residual F(q, u, lambda). No local solve of any kind.
   void Mult(const Vector &x, Vector &y) const override;

   /** @brief Assemble and factor the Jacobian at @a x.

       The returned handle is **solve-only** and its Mult() aborts: after
       ComputeH() the local arrays hold the FACTORED blocks and the Schur
       complement, so the Jacobian can no longer be applied out of them. Only
       DarcyNPCSolver understands it. That is a real constraint of hybridized
       elimination and not an oversight -- applying J would need unfactored
       copies of every local block. */
   Operator &GetGradient(const Vector &x) const override;

   /// The (q, u) sub-blocks of the full vector, for a caller splitting it.
   const Array<int> &LocalOffsets() const { return loc_offsets; }

   /// The handle GetGradient() returns; DarcyNPCSolver takes it.
   class Jacobian : public Operator
   {
   public:
      Jacobian(DarcyHybridization &dh_, Operator &S_, const Array<int> &offs,
               const Array<int> &loc_offs)
         : Operator(offs.Last()), dh(dh_), S(S_), offsets(offs),
           loc_offsets(loc_offs) { }
      void Mult(const Vector &, Vector &) const override
      {
         MFEM_ABORT("The NPC Jacobian is solve-only: the local blocks are "
                    "factored in place, so it cannot be applied. Use "
                    "DarcyNPCSolver.");
      }
      DarcyHybridization &dh;
      Operator &S;                  ///< reduced trace operator, assembled or not
      const Array<int> &offsets;    ///< {0, flux, potential, trace}
      const Array<int> &loc_offsets;///< {0, flux, potential}
   };

private:
   DarcyHybridization *dh;
   const BlockVector &load;
   Array<int> offsets, loc_offsets;
   mutable BlockVector r_loc, x_loc;
   mutable Vector r_tr;
   mutable std::unique_ptr<Jacobian> jac;
};

/** @brief Solves the Jacobian of DarcyNPCOperator by hybridized elimination:
    reduce to the trace, solve there, recover the local increments.

    The trace solve is the caller's: pass any Solver. With
    DarcyHybridization::GradientMode::Assembled it receives a SparseMatrix, so
    a direct solver or an algebraic preconditioner works; with MatrixFree it
    receives an operator that only applies S, so it must be a Krylov method
    that needs no matrix. */
class DarcyNPCSolver : public Solver
{
public:
   /// @a trace_solver is used for the reduced trace system, once per step.
   explicit DarcyNPCSolver(Solver &trace_solver);

   /// Expects the handle from DarcyNPCOperator::GetGradient().
   void SetOperator(const Operator &op) override;

   /// @a b is the outer residual; @a x comes back as the Newton CORRECTION,
   /// in that solver's convention of x_new = x - correction.
   void Mult(const Vector &b, Vector &x) const override;

private:
   Solver &trace_solver;
   const DarcyNPCOperator::Jacobian *jac{nullptr};
   mutable BlockVector r_loc, dx_loc;
   mutable Vector r_tr, b_tr, dtr;
};

}

#endif
