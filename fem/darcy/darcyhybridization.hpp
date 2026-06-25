// Copyright (c) 2010-2024, Lawrence Livermore National Security, LLC. Produced
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
    the spaces for the primiary quantities, which enables to eliminate (often
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
class DarcyHybridization : public Hybridization, public Operator
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
   ParFiniteElementSpace *pfes_p;   ///< parallel potetial FE space
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
   { ess_tdof_list_.Copy(ess_tdof_list); }

   /// Return a (read-only) list of all essential true DOFs.
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

   /// Apply the hybridized operator.
   /** @note The DarcyHybridization object must be finalized by Finalize(). */
   void Mult(const Vector &x, Vector &y) const override;

   /// Evaluate the gradient operator at the point @a x.
   Operator &GetGradient(const Vector &x) const override;

   /// Finalize the construction of the hybridized matrix.
   void Finalize() override;

   /// Use the stored eliminated part of the sytem to modify the r.h.s.
   /** @param vdofs_flux   list of VDOFs of flux @a u
       @param x            solution vector providing the VDOF values
       @param b            right hand side vector
   */
   void EliminateVDofsInRHS(const Array<int> &vdofs_flux,
                            const BlockVector &x, BlockVector &b);

   /// Use the stored eliminated part of the sytem to modify the r.h.s.
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
   /** @param b      r.h.s. of the mixed system
       @param b_r    r.h.s of the hybridized system
    */
   void ReduceRHS(const BlockVector &b, Vector &b_r) const;

   /// Projects trace of the solution onto the trace variable
   /** @note The trace projection performs simple averaging of the face values,
       which may not be consistent with the implicit definition in the
       hybridized system. Therefore, the values should serve only as an
       approximation or an initial guess.
       @param sol    solution of the mixed system
       @param sol_r  solution of the hybridized system
    */
   void ProjectSolution(const BlockVector &sol, Vector &sol_r) const;

   /** @brief Not available, use ComputeSolution(const BlockVector &,
       const Vector &, BlockVector &) instead. */
   void ComputeSolution(const Vector &b, const Vector &sol_r,
                        Vector &sol) const override
   { MFEM_ABORT("Use BlockVector version instead"); }

   /// Compute solution of the mixed system.
   /** @param b      r.h.s. of the mixed system
       @param sol_r  solution of the hybridized system
       @param sol    solution of the mixed system
    */
   void ComputeSolution(const BlockVector &b, const Vector &sol_r,
                        BlockVector &sol) const;

   /// Total flux function
   /** @param Tr  element transformation (with set integration point)
       @param u   flux at the integration point
       @param p   potential at the integration point
       @param ut  total flux at the integration point
   */
   using total_flux_fun =
      std::function<void(ElementTransformation &Tr, const Vector &u, real_t p,
                         Vector &ut)>;

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
