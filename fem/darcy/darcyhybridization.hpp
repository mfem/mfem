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

#include <vector>

namespace mfem
{

class HDGDiffusionIntegrator;

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
      /** @brief The element-local loops run in parallel: ComputeH()'s, whose
          scatter into the trace matrix stays serial and in element order, and
          MultNL()'s -- the residual and the Jacobian assembly, and so
          NPCResidual() and NPCGradient() -- which is walked in colour order
          instead. Both agree with Serial bit for bit. Requires
          MFEM_USE_OPENMP and MFEM_THREAD_SAFE, and obliges the caller's own
          integrators to be thread-safe; see SetAssemblyMode(). */
      Threaded,
      /** @brief The interior-face potential term is assembled by one batched
          kernel that scatters straight into E, G, H and D, instead of one
          host call per face.

          A DEVICE mode: it needs the storage to be device-resident to be
          worth anything, and its D accumulation goes through AtomicAdd, which
          costs on a host where the per-face loop's plain += does not.

          **It falls back silently, and for most problems it does.** The
          kernel covers HDGDiffusionIntegrator with any of its coefficients,
          both HDGConvection*Integrators, and a SumIntegrator of them, on an
          interior face of a serial conforming mesh under NPC.
          CanBatchPotFaceAssembly() answers whether it was taken -- ask it,
          because this mode was unreachable for EVERY caller in the tree until
          DarcyForm's SumIntegrator wrapper was looked through, and no answer
          and no timing showed that. */
      Batched,
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
   bool bnpc{};      ///< NPC requested on a form that may be linear
   /** @brief A load assembled on the SKELETON, in L-dofs of @a c_fes, or null.

       DarcyForm owns it -- see DarcyForm::GetTraceRHS() -- and registers it
       here through SetTraceRHS() so that BOTH routes carry it without the
       caller wiring anything: ReduceRHS() adds @f$P^T b_\lambda@f$ to the
       reduced right-hand side, and NPCResidual() subtracts it from the trace
       block, which is the same convention read off
       @f$r = A x - b@f$. Borrowed, not owned. */
   const Vector *trace_rhs{};
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
   /** @brief Still an Array, and deliberately: it is copy-assigned from and
       Swap()ped with Hybridization::Af_data, which is the UPSTREAM base
       class's. The two have to change type together, so both are left for a
       separate pass over fem/hybridization.hpp. */
   Array<real_t> Af_lin_data;
   Vector Ae_data;
   bool A_empty{true};

   Array<int> Bf_offsets, Be_offsets;
   Vector Bf_data, Be_data;

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
   mutable Vector Bnl_data;
   mutable bool Bnl_empty{true};

   /** @brief Load the (0,1) gradient block of element @a el into @a Bnl.

       Returns false, leaving @a Bnl untouched, when there is no such block. */
   bool GetBnlMatrix(int el, DenseMatrix &Bnl) const;

   Array<int> Df_offsets, Df_f_offsets;
   mutable Vector Df_data, Df_lin_data;
   mutable Array<int> Df_ipiv;
   /** @brief The factored Schur complement, for LocalOpType::FluxNL only.

       Everywhere else the Schur complement is built into @a Df_data, over the
       potential block, because nothing else needs that storage. FluxNL is the
       exception: the potential mass is LINEAR there, so @a Df_data holds its
       factorisation and LocalFluxNLOperator::SolveP() needs it on every local
       nonlinear iteration. The Schur complement used to be built into a
       function-local temporary and thrown away, which is why the matrix-free
       gradient was refused in that mode and why anything reading a Schur
       complement back out of @a Df_data got the potential mass instead.

       Allocated only when the mode calls for it. MultInv() reads these
       whenever it is applying the JACOBIAN's blocks, which is exactly its
       @a with_bnl argument. */
   mutable Vector Sf_data;
   mutable Array<int> Sf_ipiv;
   bool D_empty{true};

   Array<int> Ct_offsets;
   Vector Ct_data;

   mutable Array<int> E_offsets;
   mutable Vector E_data;

   Array<int> &G_offsets{E_offsets};
   mutable Vector G_data;

   mutable Array<int> H_offsets;
   mutable Vector H_data;

   mutable Array<int> darcy_offsets, darcy_toffsets;
   mutable BlockVector darcy_rhs;
   Vector darcy_u, darcy_p;
   mutable Array<int> f_2_b;
   /** @brief An element colouring, and the elements ordered by it. Built once,
       lazily, and only when AssemblyMode::Threaded is asked for.

       Two elements of one colour share no face -- Mesh::GetElementColoring()
       colours the element-to-element graph, whose edges ARE the faces -- so
       within a colour the trace dofs written by different elements are
       disjoint. That is what makes MultNL()'s two shared writes safe without
       atomics: the scatter of the trace row into @a y, and H_f's accumulation
       in AssembleHDGGrad(), which both sides of a face add into.

       E and G need no such protection and are left alone: they are stored per
       (face, SIDE), so the two elements of a face write different halves. */
   mutable Array<int> colour_order, colour_offsets;
   /** @brief Every element's flux and potential dofs, concatenated in element
       order, so that the whole gather or scatter between an L-vector and the
       element-blocked layout is ONE Vector::GetSubVector()/SetSubVector().

       Those overloads are mfem::forall kernels (linalg/vector.cpp:676, :740),
       so with the maps and the blocked vector marked UseDevice() the gather
       runs where the local blocks already live. The per-element loop they
       replace could not: it goes through the `real_t *elem_data` overloads,
       which begin with HostRead()/HostReadWrite() and are host loops by
       construction.

       Laid out to match @a Af_f_offsets and @a Df_f_offsets exactly -- the
       flux half is GetFDofs(el) for each element in turn, the potential half
       fes_p.GetElementVDofs(el) -- so element @a el's slice is at
       Af_f_offsets[el] / Df_f_offsets[el], which is where the blocked vectors
       put it. Entries carry MFEM's signed-dof convention and the kernels
       honour it, so an H(div) flux gathers correctly.

       Built once, lazily, by BuildElementDofMaps(). Both are pure functions of
       the spaces and of @a hat_dofs_marker, which Init() fixes and nothing
       afterwards changes: ConstructC() only re-marks free dofs from 0 to -1,
       and GetFDofs() selects on != 1. */
   mutable Array<int> el_u_dofs, el_p_dofs;

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

   /// Defined below, after the members it is scratch for; see there.
   struct TransWorkspace;

   /** @brief The scratch ComputeElementsHBatched() works in, hoisted out of
       the chunk loop by the caller.

       It is a struct and an argument rather than six locals because the
       allocation is not free and was not small: at one chunk of 256 elements
       and order 2 the buffers come to about 1.5 MB, every one of them above
       glibc's mmap threshold, so making them locals asked the kernel for
       fresh pages and faulted them in on first touch once per chunk. Measured
       -- that alone was 0.36 s of a 0.99 s face-pair loop at n=128.

       Vector::SetSize() does not shrink the allocation, so sizing for the
       first chunk sizes for all of them and the last, short chunk reuses what
       is there. */
   struct ElementHWorkspace
   {
      /// The element's trace blocks side by side, (na, T).
      Vector Ct;
      /// A^-1 Ct, same shape.
      Vector AiCt;
      /// E and then G; the two are the same size and their lives do not
      /// overlap.
      Vector EG;
      /// S^-1 (B A^-1 Ct - E), (nd, T).
      Vector BAiCt;
      /// C A^-1 B^T + G, (T, nd).
      Vector CAiBt;
      /// The element matrix before it is packed into blocks, (T, T).
      Vector Hfull;
   };

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
      /** The caller's per-thread scratch. Every transformation this operator
          uses lives in it, so the operator allocates nothing per element --
          it used to allocate nine objects and two pointer vectors. The face
          transformations are read straight out of @a ws.lop_faces; there is no
          indirection vector because every entry would be `&ws.lop_faces[f]`. */
      TransWorkspace &ws;
      /** @brief This element's slice of the block integrator's flux row,
          precomputed for the whole mesh, or NULL to call the integrator.

          See DarcyHybridization::CanBatchLocalResidual(). Not owned. A
          member here costs nothing outside darcyhybridization.cpp -- this is
          a nested class and no translation unit constructs or contains one --
          which is exactly why the same value is a PARAMETER on
          DarcyHybridization's own methods. */
      const Vector *elem_flux_row;
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
      /** @a ws supplies every transformation this operator uses; see
          TransWorkspace::lop_elem. Nothing here is owned, so there is no
          destructor. */
      LocalNLOperator(const DarcyHybridization &dh, int el, const BlockVector &trps,
                      const Array<int> &faces, TransWorkspace &ws,
                      const Vector *elem_flux_row = NULL);
      virtual ~LocalNLOperator() = default;

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
                          const BlockVector &trps, const Array<int> &faces,
                          TransWorkspace &ws);

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
                         const BlockVector &trps, const Array<int> &faces,
                         TransWorkspace &ws);

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
   /** @brief True when the element-local blocks and the element-wise H
       must be kept, which is what NPC reads and what a nonlinear form
       needs anyway. IsNonlinear() alone was the test, and it is the
       wrong one for NPC: a linear form needs exactly the same data. */
   bool NPCEnabled() const { return bnpc || IsNonlinear(); }
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
   /** @brief Caller-allocated transformation storage for one thread of an
       element loop.

       Mesh keeps exactly ONE FaceElemTr, one Transformation and one
       Transformation2, and GetFaceElementTransformations(f) hands back a
       pointer into them -- so two threads in the same element loop overwrite
       each other's geometry, silently and with no wrong answer that looks like
       a race. Holding the objects here instead is the whole fix, and it needs
       no change to Mesh: every caller-allocated overload exists and is const
       (mesh.hpp, pmesh.hpp). LocalNLOperator already owns its transformations
       for this reason, which is why the local nonlinear solve was never the
       obstacle.

       This is a TYPE, not a member, so it adds nothing to the class layout --
       the trap that has cost this branch four rebuild cycles. It is passed as
       a parameter, which is what the standing note recommends. */
   struct TransWorkspace
   {
      IsoparametricTransformation elem;    ///< the element transformation
      FaceElementTransformations face;     ///< one face at a time
      IsoparametricTransformation f1, f2;  ///< that face's two side transforms

      /** @brief LocalNLOperator's storage: ALL of an element's faces at once.

          The four above serve a loop that visits one face at a time, which is
          what ConstructGrad() does. LocalNLOperator cannot use them: its
          AddMultDE() and AddMultBlock() iterate an element's faces with every
          transformation live, so it needs one per face simultaneously.

          It used to heap-allocate them -- one IsoparametricTransformation, and
          per face a FaceElementTransformations plus another
          IsoparametricTransformation for the neighbour -- in its constructor,
          which runs ONCE PER ELEMENT PER RESIDUAL EVALUATION. Measured on
          `convdiff -p 1 -o 2 -dg -hb -nl -npc -nls 3` at 128x128 with a direct
          trace solve, that constructor and destructor were 0.098 s of
          NPCResidual's 0.744 s -- 13%, and twice what the integrators inside
          the same routine cost.

          Held here they are allocated once for the whole element loop and
          reused: the vectors grow to the largest face count seen and never
          shrink. @a lop_elem is deliberately separate from @a elem so that a
          LocalNLOperator and a ConstructGrad() pass over the same element can
          never alias each other's geometry.

          **End to end it is worth about 1.5%, not the 3% the 13% above
          predicts, and the case for it is not the 1.5%.** Interleaved A/B over
          eight pairs on the case above, two binaries built from the same tree
          so no rebuild sits between the halves: minimum 3.804 s -> 3.750 s,
          median 3.928 s -> 3.865 s, faster in seven pairs of eight. What it
          buys that the number does not show is that the residual's element
          loop now allocates NOTHING per element, which is a precondition for
          threading or offloading it -- a malloc per element per evaluation is
          exactly what stops such a loop scaling.

          **A blocked measurement said the opposite and was wrong.** Building
          one version, timing five runs, rebuilding the other and timing five
          more reported 3.33 s -> 3.63 s, i.e. a 9% REGRESSION, because the two
          halves ran minutes apart on a machine whose load was still settling.
          Best-of-five does not defend against drift between the halves; only
          interleaving does. Where two variants differ by a couple of percent,
          build both binaries first and alternate the runs. */
      IsoparametricTransformation lop_elem;
      std::vector<FaceElementTransformations> lop_faces;
      std::vector<IsoparametricTransformation> lop_nbrs;
   };

   /// The shared Mesh cache. Valid only on a single-threaded path.
   /** @brief Build @a colour_order / @a colour_offsets if they are not built,
       and warm the lazily-built tables an element loop would otherwise race to
       fill. Cheap, once, and only on the threaded path. */
   void BuildElementColouring() const;

   /** @brief Build @a el_u_dofs / @a el_p_dofs if they are not built. */
   void BuildElementDofMaps() const;

   /** @brief Make the element-local block arrays readable on the host.

       The batched routes hand these arrays to BatchedLinAlg through
       Read()/ReadWrite(), whose default is on_dev = true, so with a Device
       configured they come back valid there -- and the debug backend
       mprotects the host page while that is so. Every other consumer is host
       code reaching them through raw pointers (Array::operator[],
       Vector::GetData()), which does not sync, so it has to be told.

       **Af_ipiv is the sharp one, and it is why this exists.** LUFactors uses
       its entries as ARRAY INDICES, so a host reader indexes on whatever is
       there; the fault lands inside LUFactors::Solve() naming no array; and
       Af_ipiv.GetMemory().HostIsValid() reports **1** the whole time, because
       the flag and the page protection are not the same thing. Guarding on
       the flag would therefore pass and crash anyway. Measured on the NPC
       route under Device("debug").

       That route is also what withdraws the note on InvertA() saying nothing
       reaches this. It reasoned that the only host reader is a nonlinear
       local solve and that one does not run with a device configured. NPC has
       no local nonlinear solve at all -- it reaches the same host MultInv()
       directly -- so the reader is reached, and by the ordering that matters:
       InvertA() or FactorElementsBatched() leaves the pivots on the device,
       NPCReduce()'s MultInvBatched() puts them back there, and the
       matrix-free Gradient::Mult() then indexes them on the host.

       It covers the face blocks Ct, E, G and H as well as the element ones,
       because AssemblyMode::Batched's face kernel writes E, G, H and D on the
       device and the BOUNDARY face pass reads D on the host in the very next
       loop.

       Costs a flag check per array when nothing has moved. */
   // (declared in the public section; see SyncLocalBlocksToHost() there)

   /** @brief Whether each element's field dofs are its own.

       True for two discontinuous spaces, and then an element loop that WRITES
       field dofs writes somewhere no other element does. Two things need it,
       for the same reason and at opposite ends of the machine: a threaded loop
       needs it to run without a colouring (see CanThreadFieldLoop()), and a
       device scatter needs it because Vector::SetSubVector() is an unordered
       mfem::forall -- two entries of the map naming one dof would race, where
       the serial loop's last-writer-wins is element order.

       An H(div) flux shares dofs across faces, so both fall back to the serial
       element loop there. That is the RT pathway, which this branch leaves
       alone. */
   bool FieldDofsAreElementLocal() const
   {
      return fes.FEColl()->GetContType() ==
             FiniteElementCollection::DISCONTINUOUS
             && fes_p.FEColl()->GetContType() ==
             FiniteElementCollection::DISCONTINUOUS;
   }

   /** @brief Whether an element loop that writes FIELD dofs may be threaded.

       It may when both field spaces are discontinuous, and then each element's
       flux and potential dofs are its own -- no colouring, no atomics. An
       H(div) flux shares dofs across faces, and there these loops either
       accumulate into a shared entry or, in ComputeSolution(), overwrite it,
       where serial's last-writer-wins is element order and a colouring would
       change which element wins. That is a question about the RT pathway, and
       the standing instruction on this branch is to leave it alone: with a
       non-discontinuous space these loops keep the serial loop they have
       always had.

       ReduceRHS() does not use this. It writes only TRACE dofs, so the
       colouring covers it whatever the flux space is. */
   bool CanThreadFieldLoop() const
   {
      return asm_mode == AssemblyMode::Threaded && FieldDofsAreElementLocal();
   }

   FaceElementTransformations *GetFaceTransformation(int f) const;
   /// @brief The same face transformation, built into @a ws instead of into
   /// the Mesh's shared cache, so it is safe on a threaded element loop.
   FaceElementTransformations *GetFaceTransformation(int f,
                                                     TransWorkspace &ws) const;
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
   void ComputeElementH(int el, ComputeHMode mode, real_t *Hel,
                        const Vector *AiBt_all = NULL) const;
   /** @brief The element-local FACTORISATION half of ComputeElementH() -- the
       LU of A, the Schur complement and its LU -- for every element in one
       batch of BatchedLinAlg calls, and A^-1 times the negated (0,1) block
       into @a AiBt_all.

       Returns false, having done nothing, unless LocalFactorMode::Batched is
       asked for and the three arrays it views as DenseTensors are present and
       one block size; the element loop then factors as it always has. When it
       returns true the caller passes @a AiBt_all to ComputeElementH(), which
       skips the arithmetic done here and reads its element's slice.

       Every operation is a BatchedLinAlg call or one mfem::forall, so with a
       device configured the whole factorisation runs there and nothing comes
       back -- and in ComputeHMode::GradientFactorOnly nothing needs to, that
       mode having no face loop after it. The assembling modes do: their face
       loop is host dense work over Ct/E/G, and it pulls A, the Schur
       complement and @a AiBt_all back. That transfer is step 2's to remove,
       not this one's.

       **Bit-for-bit the element loop in a build without LAPACK, and this
       used to say it was not.** The claim was that the Schur complement goes
       through BatchedLinAlg::AddMult() where the loop uses mfem::AddMult()
       and "the two accumulate a product in different orders". They do not:
       both run the same j-k-i loop over the same products, and
       kernels::AddMult()'s `alpha` is 1.0 here, which is an exact multiply.
       The test that says so was already in the tree and passing -- "The
       batched element factorisation assembles the same trace operator"
       compares the assembled H entrywise with RequireSame(), not with a
       tolerance. An asserted claim sat next to a measurement that refuted it
       for as long as nobody read them together.
       With LAPACK the element side is dgemm_ and only round-off agreement is
       claimed; likewise on the GPU_BLAS and MAGMA backends. Measured
       agreement is on SetLocalFactorMode(). */
   bool FactorElementsBatched(ComputeHMode mode, Vector &AiBt_all) const;
   /** @brief The (element, local face) index map ComputeElementsHBatched()
       reads: three offsets per entry -- into @a Ct_data, into @a E_data and
       @a G_data, and into @a H_data -- for the whole mesh, laid out
       interleaved at 3*(el*@a nf + lf).

       Returns false, and @a nf and @a nc are then meaningless, unless every
       element has the same number of faces and every face the same number of
       trace dofs. That is the batched face loop's whole precondition beyond
       the uniform A and D blocks CanBatchLocalFactor() asks for, and it is
       asked of the mesh and the trace space rather than assumed from
       "uniform mesh at uniform order" -- a per-face trace degree
       (SetTraceOrders()) breaks it while leaving the local blocks uniform.

       The H offset is -1 where this element is not the face's FIRST, which is
       how the element loop's "integrate the face contribution only on one
       side" is carried into a kernel that cannot branch on the mesh. */
   bool BuildElementHFaceMap(int na, int nd, bool with_h, int &nf, int &nc,
                             Array<int> &face_map) const;
   /** @brief The FACE-PAIR half of ComputeElementH() -- everything after the
       factorisation -- for a chunk of @a nel elements at once, writing the
       same (f2, f1) block buffer the element loop writes.

       **This is the largest offloadable item in an NPC step**: 24% of one at
       order 2 and 45% at order 3, and the only share that GROWS with order.
       See ComputeH() for the profile it comes from.

       The double loop it replaces is one matrix identity per element, which
       is what makes the whole thing five BatchedLinAlg calls:

           H_el = -C^T A^-1 C + (C^T A^-1 B^T + G) S^-1 (B A^-1 C - E)

       with C the element's trace blocks side by side, (na, T), and G and E
       the matching stacks, T = @a nf * @a nc. The element loop's (f2, f1)
       block is exactly that expression's (f2, f1) block, so nothing is
       reassociated across faces -- what changes is that the inner products
       run over all of C at once instead of one face pair at a time.

       Bit-for-bit the element loop in a build without LAPACK, for the reason
       FactorElementsBatched() gives, and everything that is a SUM of two
       matrices is additionally done in the loop's own order: E is subtracted
       after the product rather than folded into an AddMult beta, which would
       accumulate on top of -E instead.

       **What it is worth, measured, interleaved A/B/C over five or six runs
       each, one thread, `convdiff -dg -hb -npc -nls 3 -gm 0 -rtol 1e-12`.**
       A is the element loop, B is LocalFactorMode::Batched before this
       existed, C is it now. Seconds, summed over the run's ComputeH() calls:

           case                    A      B      C     pairs B->C
           o2 n=128 -p 2 -nld    0.571  0.645  0.491     1.45x
           o3 n=128 -p 2 -nld    1.705  1.842  1.530     1.39x
           o2 n=128 -p 1 -nl     0.494  0.537  0.439     1.36x
           o5  n=64 -p 2 -nld    2.840  3.018  2.313     1.59x

       So the face-pair loop itself is 1.36-1.59x faster batched, and it is
       what makes LocalFactorMode::Batched a net gain at all -- **B is 5-9%
       SLOWER than A at every size**, which SetLocalFactorMode() used to claim
       the opposite of.

       **End to end it is inside run-to-run scatter**, and that is not a
       disappointment but the plan's own gate: at o3 n=128 the whole solve is
       8 s, of which the trace solve is 54-59%, so a 0.18 s saving cannot be
       resolved against a 5% spread (median 8.27 s against 8.57 s, min 7.91
       against 7.77, answers identical to every digit). Measure this where it
       happens, not at the end of a solve.

       @a Hel must be sized @a nel * T * T; the caller's chunking is what
       bounds it, and the alias Memory views mean the chunk needs no copy of
       the local blocks. The chunk length itself is nearly irrelevant on a
       host -- swept 4, 8, 16, 32, 64, 128, 256, 1024 elements at o2 n=128 and
       the face-pair time stayed in 0.73-0.86 s with no trend, which is what
       ruled out cache streaming as the explanation of the first, slow
       version. */
   void ComputeElementsHBatched(ComputeHMode mode, int el_0, int nel,
                                int na, int nd, int nf, int nc,
                                const Vector &AiBt_all,
                                const Array<int> &face_map,
                                ElementHWorkspace &ws, Vector &Hel) const;
   /** @brief Add the blocks ComputeElementH() left in @a Hel to @a H.
       Serial by contract -- see SetAssemblyMode(). */
   void ScatterElementH(int el, const real_t *Hel, SparseMatrix &H) const;
   /// Elements per chunk of the element loop; see ComputeH().
   int AssemblyChunkSize(int NE) const;
   /** @brief Build the trace-trace block H, and factor the local blocks on
       the way.

       **This is the largest single cost of an NPC step, and it was not where
       the device-offload plan expected to find it.** Measured on
       `convdiff -dg -hb -npc -nls 3 -gm 0 -rtol 1e-12`, 128x128 quads, one
       thread, with a DIRECT trace solve so the Krylov question is excluded --
       shares of the step's own work, that trace solve being a further 54-59%
       of the run:

       | | `-p 1 -nl`, k=2 | `-p 2 -nld`, k=2 | `-p 2 -nld`, k=3 |
       |---|---|---|---|
       | this routine | 39% | 54% | 61% |
       | -- ComputeElementH(), dense | 24% | 31% | 45% |
       | -- ScatterElementH(), sparse | 10% | 13% | 9% |
       | -- Finalize() + RAP | 7% | 11% | 7% |
       | the integrators in the residual | 5% | 7% | 6% |

       Two things follow. The dense half GROWS with order (dofs cubed) while
       the integrator evaluation shrinks (quadrature points times dofs), so a
       plan ranked on the integrators gets further from the truth exactly where
       the method is expensive. And the whole of that dense half is batched
       now: LocalFactorMode::Batched takes the factorisation through
       FactorElementsBatched() and the `C A^-1 C^T` face-PAIR loop through
       ComputeElementsHBatched(), which is where the timings live.

       The sparse third of it, ScatterElementH() plus Finalize(), is host-only
       work by construction; GradientMode::MatrixFree is what deletes it, at
       the cost of an unpreconditioned trace solve. **It is the next item**,
       and it is now comparable to the dense half rather than a third of it:
       with the face pairs batched, the scatter is 0.21-0.35 s against the
       pairs' 0.27-1.15 s at the four sizes on ComputeElementsHBatched(). */
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
                  const BlockVector &x_l, Vector &u_l, Vector &p_l,
                  TransWorkspace &ws) const;
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
   /** @brief NPC's shared precondition: finalized and a discontinuous flux
       space. LocalOpType::FluxNL was refused here too until its Schur
       complement got somewhere to live (@a Sf_data); see the definition. */
   void NPCCheck() const;
   /// A correctly sized zero load for a gradient pass; see the definition.
   void ZeroLoad(BlockVector &b, bool true_dofs) const;

   /** @brief The local nonlinear residual of @a el at (@a u_l, @a p_l).

       @a elem_flux_row, when given, is this element's slice of the block
       integrator's flux row as HDGMixedConductionResidualBatched() computed
       it for the whole mesh, and the element term is read from there instead
       of from the integrator. A PARAMETER and not a member, per the standing
       note: a member on DarcyHybridization moves the class layout and every
       translation unit that includes mfem.hpp with it. */
   void LocalResidual(int el, const Array<int> &faces, const BlockVector &x_l,
                      const Vector &bu_l, const Vector &bp_l,
                      const Vector &u_l, const Vector &p_l,
                      Vector &ru_l, Vector &rp_l, TransWorkspace &ws,
                      const Vector *elem_flux_row = NULL) const;
   void MultInv(int el, const Vector &bu, const Vector &bp, Vector &u,
                Vector &p, bool with_bnl = false) const;
   /** @brief MultInv() for every element at once, on element-blocked vectors.

       @a bu and @a bp carry the elements' right-hand sides end to end in
       element order -- sizes Af_f_offsets.Last() and Df_f_offsets.Last() --
       and @a u and @a p come back the same way. Every step is a
       BatchedLinAlg call, so on a device nothing is read back.

       The caller's gather into @a bu / @a bp and scatter out of @a u / @a p
       are kernels too now, one Vector::GetSubVector()/SetSubVector() over
       el_u_dofs / el_p_dofs each; this doxygen used to say they "remain on
       the host", and they were the larger cost. What remains on the host is
       the FACE loop at each call site, and where one follows the answers it
       still has to be read back.

       Requires CanBatchLocalFactor(), since a DenseTensor is one block size.
       Bit-for-bit the per-element route in a build without LAPACK, where
       LUFactors::Solve() is the kernels::LSolve/USolve pair that
       NativeBatchedLinAlg::LUSolve() calls; with LAPACK the per-element side
       is dgetrs_ and only round-off agreement is claimed. See MultInv(). */
   void MultInvBatched(const Vector &bu, const Vector &bp, Vector &u,
                       Vector &p, bool with_bnl = false) const;
   void ConstructGrad(int el, const Array<int> &faces, TransWorkspace &ws,
                      const BlockVector &x_l,
                      const Vector &u_l,
                      const Vector &p_l) const;
   /** @brief One face integrator's contribution to D, E, G and H.

       @a eg_written says whether E and G for this face and side have already
       been written during this gradient pass: false overwrites them, true adds
       to them, and it is set on the way out. It exists because E and G hold
       one block per face and side and are *rewritten* rather than reset, so
       the first writer has to clear whatever the previous pass left and every
       writer after it has to accumulate. See ConstructGrad(). */
   void AssembleHDGGrad(int el, FaceElementTransformations *FTr,
                        NonlinearFormIntegrator &nlfi,
                        const Vector &x_f, const Vector &p_l,
                        bool &eg_written) const;
   void AssembleHDGGrad(int el, FaceElementTransformations *FTr,
                        BlockNonlinearFormIntegrator &nlfi,
                        const Vector &x_f, const Vector &u_l, const Vector &p_l,
                        bool &eg_written) const;

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

       **Two loops are threaded, and they are threaded differently.**
       ComputeH()'s is the one described above: element-local work in parallel,
       scatter serial and in element order. MultNL()'s -- which is the residual
       and the Jacobian assembly, and so NPCResidual() and NPCGradient() too --
       is walked in COLOUR order instead, Mesh::GetElementColoring() colouring
       the element-to-element graph whose edges are the faces, so no two
       elements of a colour share one. That is what makes its two shared writes
       safe: the accumulation of the trace row into @a y, and H_f's in
       AssembleHDGGrad(), which both sides of a face add into. E and G need
       nothing, being stored per (face, SIDE).

       **Why colouring works there and not in ComputeH().** The difference is
       the target, not the loop. MultNL() scatters into a Vector, where
       disjoint indices are genuinely independent; ComputeH() scatters into an
       unfinalized SparseMatrix, where two threads on disjoint ROWS still
       collide on @a current_row, the column-pointer scratch and the RowNode
       allocator.

       **Still bit for bit**, by the same two-term argument: colouring changes
       the ORDER in which a face's two elements accumulate, and a + b == b + a
       exactly. It would not survive a trace space whose dofs are shared
       between faces -- an H1_Trace (EDG) one -- where a dof sees more than two
       contributions and associativity would start to matter.

       **What MultNL() needed beyond the colouring**, none of it visible here:
       the integrators it calls had to become thread-safe (MFEM's
       #ifndef MFEM_THREAD_SAFE convention, applied to
       MixedConductionNLFIntegrator and the HDG face integrators), the Mesh
       transformation cache had to stop being shared (TransWorkspace, since
       Mesh keeps one FaceElemTr and one Transformation for the whole mesh),
       and num_local_nl_iters had to become an atomic update.

       **A caller obligation follows and there is no way to check it here.**
       Any integrator the caller installs -- a source term in the potential
       mass, a constraint integrator -- sits on this loop and must be
       thread-safe too. An integrator holding per-point scratch as a plain
       member will race, silently.

       **Measured, on the pedestal problem at (n, k) = (32,1), (48,2), (64,2)
       and (32,3), speedup at 8 threads against the serial mode:**

           NPCResidual      5.6  5.8  5.8  6.1 x
           NPCGradient      2.6  2.8  2.7  3.3 x
           whole NPC step   1.9  1.9  1.9  2.1 x

       NPCGradient lags NPCResidual because it carries the serial scatter --
       40-47% of that call, measured separately; see the NPCResidual group.
       The step number is bounded by that scatter and by the trace solve, which
       is not element-local and does not move here. 1.9-2.1x is against a
       predicted ceiling of about 2.3x, and the answer is identical to every
       digit at every thread count.

       Aborts if the build cannot honour it: MFEM_USE_OPENMP is what makes it
       parallel, and MFEM_THREAD_SAFE is what stops GetElementFaces() keeping
       its scratch in a function-local static. Falling back quietly would
       report a speedup nobody got. */
   void SetAssemblyMode(AssemblyMode mode);

   /// The mode set by SetAssemblyMode().
   AssemblyMode GetAssemblyMode() const { return asm_mode; }

   /** @brief Whether AssemblyMode::Batched's face kernel would actually be
       taken, which is a much narrower question than whether it was asked for.

       It needs the mode, an NPC problem (the kernel writes H into H_data,
       where the reduced route's assembled H is what a non-NPC solve reads), a
       serial constraint space (a shared face is not Mesh::FaceIsInterior()),
       every integrator on the potential-mass constraint to be one
       HDGFaceScatterBatched() implements, one integration rule across the
       face list, and the geometry HDGDiffusionFaceMatricesCanBatch() asks for.

       Measured against the 88 hybridized regression references: taken on 15,
       against 6 when the kernel covered pure diffusion alone. Of the 73 it
       still refuses, 52 carry a NONLINEAR constraint -- c_nlfi_p rather than
       c_bfi_p -- which is not an assembly-time term at all and would need a
       different kernel, over element-face pairs and per Newton step.

       **Ask this rather than inferring it from a timing.** The mode was
       unreachable for every caller in the tree until the SumIntegrator
       DarcyForm wraps around the constraint was looked through, and an
       assembly timing did not reveal that -- the run-to-run scatter is wider
       than what the mode costs. A silent fallback is the normal case here,
       so a caller that cares has to be able to ask. */
   bool CanBatchPotFaceAssembly() const;

   /** @brief How many integrators the potential-mass face constraint carries,
       with DarcyForm's SumIntegrator wrapper unwrapped.

       Reported rather than inferred for the same reason
       CanBatchPotFaceAssembly() is: a caller cannot see through the wrapper,
       and one of these numbers being 1 rather than 2 is the difference
       between exercising the batched accumulation and not. */
   int NumPotFaceConstraintIntegrators() const;

   /** @brief Whether the ELEMENT integrator of the NPC local residual is
       evaluated by one batched kernel for the whole mesh instead of one
       integrator call per element.

       False, having done nothing, unless AssemblyMode::Batched is asked for
       and the block nonlinear integrator is one
       HDGMixedConductionResidualBatched() implements -- see there for what
       that means and why a FunctionDiffusionFlux cannot be one.

       **Ask this rather than inferring it**, for the reason
       CanBatchPotFaceAssembly() gives at length: the fallback is silent, it
       is the normal case, and a timing does not separate the two.

       What it does NOT cover, each measured against the 88 hybridized
       references in miniapps/hdg/regress_test/ rather than guessed at:

       * @a m_nlfi_u / @a m_nlfi_p, which reach the residual as
         SumNLFIntegrators of pure BilinearFormIntegrators on 50 of those 88
         -- VectorMassIntegrator, VectorFEMassIntegrator and
         ConservativeConvectionIntegrator. Those element matrices are
         CONSTANT (a BilinearFormIntegrator's AssembleElementMatrix() does not
         see the state), so the residual re-assembles a fixed matrix once per
         element per evaluation in order to multiply by it. Batching that
         wants a kernel per integrator -- the mass one exists
         (HDGElementMassBatched()), the convection one does not -- and is the
         largest remaining piece.
       * @a c_nlfi / @a c_nlfi_p, a state-carrying FACE constraint, on 17 of
         the 88 (HyperbolicFormIntegrator, the -nlc cases). Those are per
         face, not per element, and the flux hierarchy is a separate piece of
         work.
       * A MixedConductionNLFIntegrator over a FunctionDiffusionFlux, on 5 of
         the 88 (`-p 8`). Its conductivity is a host std::function of the
         current potential; there is no device form of it and no per-point
         weight that can be computed ahead of the kernel.

       **What it costs. It is the first kernel of the offload plan measured
       to pay on the HOST -- and only inside NPCResidual; end to end it is
       inside run-to-run scatter, the trace solve being 54-59% of a run.**
       `convdiff -p 2 -nld -dg -hb -npc -nls 3 -gm 0
       -rtol 1e-12 -bam` at 128x128, one thread, direct trace solve, ON and
       OFF interleaved back to back in one binary through an environment gate
       so no rebuild sits between the halves -- time in NPCResidual over its
       three calls:

       | | order 2 | order 3 |
       |---|---|---|
       | per-element integrator | 0.321-0.355 s | 0.529-0.659 s |
       | batched kernel | 0.258-0.297 s | 0.409-0.491 s |
       | ratio, median of 4 pairs | 0.86 | 0.77 |

       Faster in 8 pairs of 8, and the reason is not the batching: the kernel
       is matrix free per point against ONE reference shape table for the
       mesh, where AssembleElementVector() calls CalcShape() per element per
       point and forms each contracted flux through Vector::operator*(). The
       kernel's own share is 0.019 s and 0.038 s per call; the integrator it
       replaces was 0.037 s and 0.075 s per call in a separately instrumented
       build, which is indicative only -- the ratios above are the
       interleaved measurement and the only one that survives this machine's
       drift.

       The HostRead() that follows it is FREE here -- 0.0000 s to the timer's
       resolution -- because with no Device configured it is a no-op. On a
       device it is the transfer the offload plan's gate is about, and it is
       the reason this is a link in a chain rather than a speedup: the
       consumer is still the host element loop. */
   bool CanBatchLocalResidual() const;

   /** @brief Assemble the flux mass block for every element in one batched
       kernel, from @a M_u's DOMAIN integrators, instead of one
       ComputeElementMatrix() and one AssembleFluxMassMatrix() per element.

       False, having done nothing, unless AssemblyMode::Batched is asked for
       and every domain integrator is one HDGElementMassBatched() implements;
       the caller then keeps its element loop.

       It does NOT cover the form's face or boundary integrators. That is
       deliberate and it is the difference from MFEM's AssemblyLevel::ELEMENT,
       which folds them in for a DG space -- DarcyForm routes those itself,
       into the constraint blocks rather than into the element matrix, so
       folding them here would double-count them. */
   bool AssembleFluxMassMatricesBatched(BilinearForm *M_u);

   /// The same for the potential mass block; see the flux one.
   bool AssemblePotMassMatricesBatched(BilinearForm *M_p);

   /** @brief The same for the DIVERGENCE block, from @a B's domain
       integrators.

       Its scatter is the flux mass's mask with the rows unmasked: an
       element's block is (potential dofs) x (hat dofs), a free COLUMN goes to
       Bf and an essential one to Be, and every potential row goes with it. */
   bool AssembleDivMatricesBatched(MixedBilinearForm *B);

   /** @brief Whether the batched flux / potential mass assembly would
       actually be taken. Ask rather than infer: both fall back silently, on
       an integrator the kernel does not implement or on a mesh whose elements
       do not all want one integration rule. */
   bool CanBatchFluxMass(BilinearForm *M_u) const
   { return CanBatchElementMass(M_u, fes); }
   bool CanBatchPotMass(BilinearForm *M_p) const
   { return CanBatchElementMass(M_p, fes_p); }
   /// Whether the batched divergence assembly would actually be taken.
   bool CanBatchDiv(MixedBilinearForm *B) const;

   /** @brief Whether AssemblyMode::Batched's BOUNDARY face kernel would
       actually be taken. The same conditions CanBatchPotFaceAssembly() asks,
       of the boundary integrators and the faces their markers admit. */
   bool CanBatchPotBdrFaceAssembly() const;

   /** @brief Make the element-local and face block arrays readable on the
       host; see the private note.

       Public because DarcyForm::AssemblePotHDGFaces() owns the sequencing.
       Its interior and boundary passes may each be a device kernel, and one
       sync after both is right where one after each would push D back to the
       device only to pull it down again. */
   void SyncLocalBlocksToHost() const;

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

       **The hot path is batched too now**, and the paragraph that used to sit
       here said it was not. InvertA() and InvertD() run once, from
       Finalize(), and only for LocalOpType::PotNL and FluxNL -- so on their
       own they batch a cold path. What runs once per *linearisation* is
       ComputeH(), and both of its element-local halves are batched:
       FactorElementsBatched() factors every A, forms every Schur complement
       and factors those too, and ComputeElementsHBatched() then does every
       element's `C A^-1 C^T` face-PAIR loop as five BatchedLinAlg calls.

       **Only the two together are worth taking, and the factorisation alone
       is a LOSS.** This doxygen used to quote "0.553 s to 0.416 s at order 2,
       n=128" for the factorisation half, and that number is real but it is
       only what leaves ComputeElementH(); it does not count what
       FactorElementsBatched() spends. Counting both, and interleaved rather
       than in blocked halves, the factorisation on its own is 5-9% slower
       than the element loop at every size tried. The face-pair loop is
       1.36-1.59x faster and pays for it. The table is on
       ComputeElementsHBatched(); the lesson is the branch's own -- a
       measurement of one half of a change is not a measurement of the
       change.

       **The setting does more than its name says**, and the name is kept for
       compatibility. Batched also sends the local SOLVES through
       MultInvBatched() -- every element's `M^-1 (bu, bp)` in one batch of
       BatchedLinAlg calls rather than one LUFactors triple per element -- in
       ReduceRHS(), ComputeSolution(), NPCReduce() and NPCRecover(); and it
       gathers the element-blocked right-hand side, and scatters the recovered
       fields, with one Vector::GetSubVector()/SetSubVector() kernel each over
       el_u_dofs / el_p_dofs instead of a per-element host loop.

       **What that last part was worth, measured, because the claim it
       replaces was the opposite.** This doxygen used to record the batched
       route as 5 to 15% *slower* in situ at every size tried, and blamed the
       transfer around it. The transfer was not the cost. Timing the four
       phases of ReduceRHS() separately, order 2 on 64x64 quads under
       `-d cuda`, per call:

           host gather loop   1.11 ms
           MultInvBatched     6.25 ms
           HostRead() back    0.26 ms
           face loop          5.90 ms

       -- the gather was **four times** the copy-back it was supposed to be
       hiding behind. With both loops replaced by kernels the host gather
       falls from 0.75 ms to 0.10 ms and the scatter from 0.80 ms to 0.08 ms,
       and the end-to-end sign flips. Steady state, `-d cpu`, batched against
       serial:

           order 2,  64x64    FormLinearSystem  9.54 -> 8.35 ms
                              RecoverFEMSolution 9.28 -> 8.33 ms
           order 2, 160x160   FormLinearSystem 61.6 -> 54.2 ms
                              RecoverFEMSolution 60.3 -> 54.3 ms
           order 6,  48x48    FormLinearSystem 58.3 -> 71.8 ms
                              RecoverFEMSolution 58.0 -> 72.5 ms

       So: 10-12% faster at order 2 and 24% slower at order 6. The remaining
       loss is the batched dense kernels themselves on large blocks, not the
       plumbing, and it is a different problem from the one that was fixed.

       **A contract that changes with a Device configured.** In this mode the
       recovered fields come back DEVICE-valid: ComputeSolution() and
       NPCRecover() scatter with a kernel and deliberately do not read the
       answer back, which is the point. Reading them on the host is then the
       caller's business, through HostRead() or any Vector operation that
       syncs -- Vector::operator() and GetData() do not. The library propagates
       the write through the BlockVector's aliases so the flags are right; see
       ComputeSolution(). An H(div) flux keeps the host loop, since two
       elements share a dof and an unordered scatter would race.

       CanBatchLocalSolve() answers whether any of it is taken. */
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

   /** @brief Whether the local SOLVES are batched too, not only the
       factorisation.

       LocalFactorMode::Batched asked for, uniform blocks, and a stored Schur
       complement -- which the FullNL local operator has not got, its local
       problem going through MultInvNL() and a local Newton instead. When this
       is true, ReduceRHS() and ComputeSolution() do every element's local
       solve in one batch of BatchedLinAlg calls rather than one LUFactors
       triple per element.

       **What it is worth, and the honest answer is "nothing yet".** Two
       measurements, and the second is the one that matters.

       The local solve ON ITS OWN, batched against the LUFactors loop, on
       synthetic blocks of a hybridization's shape (n_a = dim*ND flux dofs,
       n_d = ND potential):

           n_d   NE      host (g++)   CUDA
             9   2304        1.04x    0.67x
            25   2304        0.71x    1.35x
            49   2304        0.72x    2.64x
            49   9216        0.69x    3.53x

       So on a host it is level at small blocks and about 30% slower at large
       ones -- same scalar work through the same kernels, but streaming the
       whole blocked arrays six times where the per-element route keeps one
       element's vectors in cache. On a device it crosses over around a
       thousand elements and reaches 3.5x.

       IN SITU it **used to be** 5 to 15% slower at every size tried, and that
       was blamed on the transfer around it. It was not the transfer: the
       gather and the scatter were per-element host loops through
       Vector::GetSubVector(real_t*), which begins with HostRead(), and they
       cost four times what the one copy-back did. With both replaced by
       kernels the in-situ figure is 10-12% FASTER at order 2 and 24% slower
       at order 6; the tables and the phase split are on SetLocalFactorMode().

       What is left of the gate (doc/HDG-DEVICE-OFFLOAD.md) still stands and
       still bites: the face loops around these solves are host dense work, so
       the chain is not device-resident and one HostRead() per call remains
       where a face loop follows. Removing it is step 2's business, not this
       one's. What this setting buys is that the local factorisation and solve
       are EXPRESSIBLE on a device at all, which the whole-chain target
       requires and which they were not before -- see the two upstream defects
       the attempt turned up, recorded on GPUBlasBatchedLinAlg::AddMult and
       NativeBatchedLinAlg::LUSolve. Hence Serial by default. */
   bool CanBatchLocalSolve() const;

   /** @brief Choose whether GetGradient() assembles the reduced system or only
       applies it. See GradientMode; the default is Assembled, which is what
       every caller written before this existed gets.

       GradientMode::MatrixFree returns an Operator with no stored matrix, so a
       caller must solve with something that needs only the action -- an
       unpreconditioned Krylov method, or one preconditioned by something not
       built from the matrix. GSSmoother, UMFPackSolver and the algebraic
       preconditioners all require a SparseMatrix and will abort.

       Supported in every LocalOpType. It was not: with only the flux mass
       nonlinear (LocalOpType::FluxNL) the Schur complement had nowhere to
       live, @a Df_data being occupied by the factored linear potential mass,
       and GetGradient() aborted rather than return something wrong. It now
       goes to @a Sf_data.

       **"Matrix free" is about the GLOBAL trace matrix, not the Jacobian.**
       The local blocks are still assembled and factored in every mode -- the
       action of S is defined in terms of them -- so no mode here is
       Jacobian-free in the stronger sense of forming no Jacobian at all.

       **What the mode costs, measured rather than asserted.** convdiff
       -p 2 -o 2 -dg -hb -up -nlu -nls 3, wall seconds, all three modes
       agreeing to six digits in both error norms:

       | nx | Assembled + UMFPack | Assembled + GS | MatrixFree |
       | -- | ------------------- | -------------- | ---------- |
       | 10 | 0.10 | 0.10 | 0.10 |
       | 20 | 0.20 | 0.10 | 0.40 |
       | 40 | 0.50 | 0.50 | 4.00 |

       So it runs and it is 8x at nx = 40, growing with the problem, because
       nothing preconditions it: with no assembled S there is no Gauss-Seidel,
       no AMG and no factorisation, and the obvious block-Jacobi replacement
       costs the same local solves that assembly costs. MatrixFree is for a
       caller who cannot afford the memory, or who brings a preconditioner not
       built from S. See doc/HDG-JACOBIAN-FREE-TRACE.md, which is where that
       open question lives. */
   void SetGradientMode(GradientMode mode);

   /** @brief Keep what NPC reads, on a form that carries no nonlinear
       integrator at all.

       Without this a linear DarcyForm cannot use the NPC pathway, and fails
       by segfault rather than by refusal. Finalize() takes a route for the
       linear case that factors each element's A and D in place and keeps no
       copy, and routes the face H into the global sparse H instead of the
       element-wise H_data -- both correct when the only thing that will ever
       be asked is one reduced solve, and both fatal to NPC, which evaluates
       the residual and the gradient at ARBITRARY states and so needs the
       blocks rather than their factorisations.

       A linear form wants NPC for the reason any DAE integrator does: the
       condensation route iterates on the trace alone, so the vector it
       iterates on is not the vector the integrator integrates and there is no
       residual over the full (q, u, lambda) state to hand it. That is as true
       of a linear problem as of a nonlinear one, and an integrator evaluating
       at predictor states or finite-difference probes needs it just as much.

       Opt-in rather than unconditional because the retained blocks are the
       largest thing the hybridization owns and no reduced-route caller should
       pay for them.

       Call it any time before Assemble(); it allocates H itself if Init() has
       already run, which it normally has -- the hybridization does not exist
       until DarcyForm::EnableHybridization() has made it, and that is what
       calls Init().

       @note This forecloses the reduced route on the same assembly:
       DarcyForm::FormLinearSystem() has no reduced H to hand back and aborts.
       They are different methods; comparing them needs two assemblies. */
   /** @brief Register a load assembled on the skeleton, in L-dofs of the
       constraint space, or null to clear it.

       Callers do not normally reach for this: DarcyForm::Assemble() registers
       DarcyForm::GetTraceRHS() here. It is public because a caller driving
       this class without a DarcyForm has no other way in.

       The vector is BORROWED and must outlive the next ReduceRHS() or
       NPCResidual(). */
   void SetTraceRHS(const Vector *b_tr_load) { trace_rhs = b_tr_load; }

   /// The registered skeleton load, or null.
   const Vector *GetTraceRHS() const { return trace_rhs; }

   /** @brief Add @f$P^T b_\lambda@f$ of the registered skeleton load into a
       reduced (true-dof) trace vector. Does nothing if none is registered. */
   void AddTraceRHS(Vector &b_tr, real_t a = 1.0) const;

   void EnableNPC();

   /** @name The NPC method: Newton on the full (q, u, lambda) system

       Nguyen, Peraire & Cockburn, JCP 228 (2009) 8841-8855, eqs (14)-(18).
       These four calls are one Newton step of it, exposed raw so that the
       shape of the method is visible. **For ordinary use take
       DarcyNPCOperator and DarcyNPCSolver instead**, which wrap them as an
       MFEM Operator over the full (q, u, lambda) vector that NewtonSolver
       drives with no special support.

       What cannot be wrapped is an Operator on the TRACE ALONE: the flux and
       the potential are Newton state here, and a trace-only operator has
       nowhere to put them. A mode that tried -- an NLOrdering enum whose
       LineariseThenCondense claimed to be this method -- is deleted; it was a
       condensation in disguise, measurably slower than the condensation it
       was meant to beat and unable to solve problems that one solves.

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

       **Where the step's time goes, and what that means for threading.** The
       four calls split into loops that reach integrators and the Mesh
       transformation cache (NPCResidual, through LocalResidual(); NPCGradient's
       first pass, through ConstructGrad()) and loops that are dense linear
       algebra alone (NPCReduce and NPCRecover, both only MultInv()). That
       separation is real -- the deleted trace-only mode fused the two inside a
       local Newton iteration and could not offer it -- but it is worth less
       than it looks. Measured over six steps on the pedestal problem at
       (n, k) = (32,1), (48,2), (64,2) and (32,3), one thread:

           integrator-bound loops     60.8  59.7  59.4  63.2 %
           integrator-free loops       6.2   5.4   5.5   5.7 %
           trace solve                33.0  34.8  35.1  31.1 %

       So the two loops that could be threaded with no integrator work at all
       are **under 6% of the step**, flat in mesh size and order, and Amdahl
       caps any gain there. NPCRecover is nonetheless the easiest loop in the
       class to thread -- it writes only the calling element's L2 flux and
       potential dofs, so it needs neither colouring nor atomics.

       **NPCGradient's column is three parts, not the two this used to name.**
       It said the column was ConstructGrad (integrators, serial) plus
       ComputeElementH (dense, already threaded by AssemblyMode::Threaded) and
       that separating them was the next measurement. Taken, in a build with
       MFEM_USE_OPENMP and MFEM_THREAD_SAFE, on the same four cases:
       GradientMode::MatrixFree runs the same threaded ComputeElementH and
       skips the scatter, so the mode difference *is* the scatter, and a thread
       fit on MatrixFree separates the other two. Shares of NPCGradient:

           ConstructGrad (integrators, serial)   58  50  50  45 %
           ComputeElementH (already threaded)     2   6   4  12 %
           scatter into the SparseMatrix         40  43  47  43 %

       **The part nobody had counted is the scatter, and it is serial by
       design** -- see SetAssemblyMode(), which explains why a SparseMatrix
       target cannot be threaded even with element colouring. The
       already-threaded half is 2-12% of the column, so the answer to the
       question this used to pose is that ComputeElementH does *not* dominate
       and the integrator-bound share is essentially as the table above says.

       **End to end, AssemblyMode::Threaded buys 1.4 to 7.7% of an NPC step**
       (1/2/4/8 threads, best of; the answer is bit-identical at every thread
       count and in both assembly modes, to every digit of the solution norm
       and the final residual). Recomposing a step with the scatter taken out
       of NPCGradient's column: integrator-bound 46-53%, threadable with no
       new infrastructure 7-10%, serial assembly 12-17%, trace solve 26-31%.
       So the ceiling on threading an NPC step -- with perfect integrator
       thread-safety and a perfectly threaded ComputeElementH -- is about 2.3x,
       and the two items outside it are the scatter and the trace solve.

       One consequence for GradientMode::MatrixFree, which is not only a memory
       play: it deletes 40-47% of NPCGradient. What it pays for that is an
       unpreconditioned trace solve, so it still loses overall -- but the prize
       behind "what preconditions a never-assembled trace system" is larger
       than a memory argument makes it look. doc/HDG-JACOBIAN-FREE-TRACE.md and
       doc/HDG-ELEMENT-LOCAL-PARALLELISM.md carry those two open questions.

       **What it delivers, measured.** On a problem whose full system is linear
       one step is exact -- the residual goes 6.96e-01 to 6.22e-15, from any
       starting point -- which is the check that falsifies the elimination
       algebra if anything in it is wrong. On the pedestal source it converges
       quadratically in the full residual: 6.7e-01, 1.5e-02, 2.8e-04, 1.2e-07,
       2.3e-14, with GetNumLocalNLIterations() identically zero. The two
       gradient modes agree at every iterate above round-off.

       And it solves stiff problems the deleted trace-only mode could not. Of
       the four configurations where the reduced trace operator converges and
       that mode did not, three fall to NPC with a backtracking line search on
       the full residual -- 13, 10 and 17 steps -- and the fourth stalls at
       2.9e-03, which is ordinary Newton stagnation. Undamped, NPC converges
       one of the four (k = 3, n = 12, in 12 steps) and wanders on the other
       three as any cold Newton does -- an earlier version of this paragraph
       said all four, which a sweep of the four configurations disproved: **the globalisation this method wants
       is on the OUTER step and there is none to do locally**, which is the
       whole point of the ordering.

       **Do not read that as a general recommendation of a line search.** The
       backtracking above is an l2 merit over all three blocks. Where the
       nonlinearity sits in the potential block and the flux and trace rows are
       linear -- the shape of every DarcyForm problem whose nonlinearity is in
       Mnl_p, and measured true here: a full step takes the flux row to 4e-17
       and the trace row to 3e-14 -- a full step is exactly optimal for two
       blocks and any damping restores part of them. Whether backtracking then
       helps or hurts turns on how badly the full step treats the potential
       block: here it improves it and the search helps, on meq's
       Grad-Shafranov discretisation it multiplies it by 77 and the search made
       every case worse.

       **Do not write a line search to fix that.** KINSolver(KIN_LINESEARCH) is
       KINSOL's Dennis & Schnabel search, with the sufficient-decrease and
       curvature conditions and a minimum-step test, and meq report it failing
       on exactly the same cases -- so a correct backtracking implementation
       does not rescue this, and a block-weighted merit does not either:
       accepting a full step on meq's published residuals needs the LINEAR flux
       row weighted 79x above the nonlinear potential row, which is a merit
       that a Newton step annihilates by construction and hence no line search
       at all. What the evidence points at is non-monotonicity -- undamped
       Newton converges cases every monotone search kills -- for which KINSOL
       offers Anderson-accelerated KIN_PICARD / KIN_FP. See
       doc/HDG-NPC-GLOBALISATION-FROM-MEQ.md and section 6 of
       doc/HDG-ORDERING-API.md.

       A line search here is well defined for a
       reason worth keeping in view -- the fields and the trace scale together
       because both are state, where a line search on a trace-only operator
       scales the trace and leaves the field update to whatever the
       substitution makes of it.

       @note **The flux space must be discontinuous, and the reason is
       representational rather than a matter of sign conventions -- an earlier
       version of this note said it was, and had not been measured.** The state
       NPC iterates on is the BROKEN one: each element owns its own copy of the
       flux dofs on a shared face, and the trace row is what makes the two
       copies agree. A conforming H(div) space has no room for that -- summing
       the element vdof counts of an RT space gives 192 against a space size of
       144 on a 4x4 quad mesh at order 1, one dof short per interior face -- so
       both elements read the SAME value, their Ct blocks carry opposite signs,
       and their contributions cancel identically. Measured on RT at five
       random states of (q, u, lambda): |C' q| = 3.2e-16 against a flux row of
       9.8, where the same problem gives 7.9-9.0 with an L2 flux and 7.1-8.0
       with a broken-RT one. **So the trace row is annihilated for every
       conforming state, not merely for the ones an iteration visits**, lambda
       is never driven, and NPC stalls: on a 6x6 triangle mesh it stops at
       |F| = 0.0804 and lands 12% off in the flux and 44% off in the trace,
       while the reduced route on the same problem converges in 6 steps.

       The route to an H(div)-shaped discretisation under NPC is therefore
       BrokenRT_FECollection, which is already admitted -- its GetContType()
       is DISCONTINUOUS, its element vdof counts sum to its space size, and it
       carries the same RT element on a space with room for the broken state.
       Measured on the problem above: quadratic convergence in 7 steps to the
       same potential and the same trace as conforming RT's reduced route,
       3.08934 and 3.31561 both ways. Lifting the guard would need the flux
       unknown carried in the hat space instead, which is a different operator
       size and a different caller contract, not a bug fix.

       @note **A load assembled on the TRACE has no slot**, here or in the
       reduced route: DarcyForm offers GetFluxRHS() and GetPotentialRHS() and
       nothing for the skeleton, and @a b is (flux, potential). The caller
       carries it instead, and where it goes is measured rather than inferred:
       subtract it from @a r_tr between NPCResidual() and NPCReduce(), which is
       the same r = F(x) - b convention NewtonSolver::Mult(b, x) applies on the
       reduced route. Checked against that route at three load scales spanning
       20x, |lambda| running 4.69 to 64.2 against 3.31 unloaded: the two agree
       to 1.2e-13 - 2.9e-12 relative.

       @note **A parity test against the reduced trace operator must run on a
       resolved mesh.** The two routes reach the same discrete solution, and
       the tests here assert it -- but on an UNDER-resolved mesh a semilinear
       source can give the discrete system more than one solution, and which
       one a solve lands on is then a property of the route rather than of the
       discretisation. Reported by meq on a Grad-Shafranov benchmark: three
       fully converged solves of the same system to rel_tol 1e-12 gave
       max psi_h of 3.1831e-01 (NPC), 3.4779e-01 (reduced operator) and
       3.1514e-01 (NPC after Anderson-accelerated Picard) -- a spread of 9.4%,
       collapsing under one refinement. So a disagreement between the two
       orderings on a coarse mesh is not evidence that either is wrong, and a
       test that asserts agreement to a tight tolerance has to be posed where
       both converge quickly or it measures this instead. */
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

       Every LocalOpType works in both modes. LocalOpType::FluxNL did not
       until @a Sf_data was added; see SetGradientMode().

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

   /** @brief Assemble the INTERIOR-face potential term with one batched
       kernel, scattering straight into E, G, H and D.

       @returns false, having done nothing, whenever it does not apply --
       AssemblyMode is not Batched, the face term is not a single
       pure-diffusion HDGDiffusionIntegrator, or the spaces do not admit the
       batched form. The caller then takes the per-face loop, so this is an
       optimisation and never a restriction.

       Boundary faces are NOT covered and still go through
       ComputeAndAssemblePotBdrFaceMatrix(). */
   bool AssemblePotFaceMatricesBatched();
   /// The integrators the batched face assembly would apply, sum unwrapped.
   void PotFaceConstraintIntegrators(
      Array<BilinearFormIntegrator*> &integs) const;
   /// The boundary constraint integrators, and the faces each marker admits.
   void PotBdrFaceLists(std::vector<Array<int>> &lists, Array<int> &all,
                        Array<BilinearFormIntegrator*> &integs) const;
   bool AssemblePotBdrFaceMatricesBatched();
   /// Shared by CanBatchFluxMass() and CanBatchPotMass().
   bool CanBatchElementMass(BilinearForm *M,
                            const FiniteElementSpace &f) const;
   /// The local index of each free / essential hat dof, and where each
   /// element's essential run starts; see AssembleFluxMassMatricesBatched().
   void HatDofMaps(Array<int> &free_map, Array<int> &ess_map,
                   Array<int> &ess_offsets) const;
   /// The interior faces, which is what the batched face assembly covers.
   void InteriorFaceList(Array<int> &flist) const;
   NonlinearFormIntegrator* GetPotConstraintNonlinearIntegrator() const { return c_nlfi_p.get(); }

   /** @brief The nonlinear flux mass integrator, or NULL.

       @note **This returned the *potential* mass integrator until this branch
       fixed it**, i.e. exactly what GetPotMassNonlinearIntegrator() below
       returns, so the two accessors were indistinguishable and this one never
       gave the flux. The signature is unchanged, so a caller that depended on
       the old value changes behaviour silently on upgrading; it wants
       GetPotMassNonlinearIntegrator(). A deprecated alias is not offered
       because it could only preserve the old value under this name, which is
       the name being corrected -- and the old value is already reachable, and
       correctly named, one line down. */
   NonlinearFormIntegrator* GetFluxMassNonlinearIntegrator() const { return m_nlfi_u; }
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

   /// Total flux function, for one field.
   /** @param Tr  element transformation (with set integration point)
       @param u   flux at the integration point
       @param p   potential at the integration point
       @param ut  total flux at the integration point
   */
   using total_flux_fun =
      std::function<void(ElementTransformation &Tr, const Vector &u, real_t p,
                         Vector &ut)>;

   /** @brief The flux law of a *system*, evaluated at a quadrature point.

       As #total_flux_fun, but the potential is a vector with one entry per
       equation. @a u is the flux, @a p the potential and @a ut the total
       flux, all of them per equation: for `neq` equations in `dim` dimensions
       @a p has `neq` entries and @a u and @a ut have `neq*dim`, with the block
       of equation `e` occupying `[e*dim, (e+1)*dim)`. That is the layout the
       block integrators build; see VectorBlockDiagonalIntegrator.

       This is a second type rather than a widening of #total_flux_fun so that
       a single-field caller written against the scalar signature goes on
       compiling. The two overloads of ReconstructTotalFlux() below cannot be
       ambiguous: `Vector(int)` is `explicit`, so a `real_t` argument never
       converts to a `const Vector &`, and a callable taking one is viable for
       exactly one of them. */
   using total_flux_sys_fun =
      std::function<void(ElementTransformation &Tr, const Vector &u,
                         const Vector &p, Vector &ut)>;

   /// Reconstruct the total flux from the provided solution.
   /** The total flux function is normally continuous and its finite element
       space is assumed to have equal number of DOFs at faces as the trace
       variable. For the interiors of elements, the quadrature function must
       be provided to calculate the total flux from the provided flux and
       potential values.

       @note **Systems are supported**, and this note used to say they were
       not. @a ut, the constraint space and the potential must all carry the
       same number of fields; every block below is that many copies of a
       scalar one, field outermost. Two things carried the generalisation and
       both are worth knowing about, because neither shows up as an abort.
       The face solve inverts the *scalar* face mass, which MassIntegrator
       builds regardless of vdim, so it is factored once and applied once per
       field rather than once against an neq-times-too-long right-hand side.
       And the element interior's boundary/interior split is a scalar count:
       GetNumElementInteriorDofs() counts one field's, and the interior dofs
       are the tail of each field's block of the vdof list, not the tail of
       the list -- which is the same thing only when there is one field.
       @param sol    solution of the mixed system
       @param sol_r  solution of the hybridized system
       @param ut_fx  total flux function
       @param ut     total flux
   */
   void ReconstructTotalFlux(const BlockVector &sol, const Vector &sol_r,
                             total_flux_sys_fun ut_fx, GridFunction &ut) const;

   /// Reconstruct the total flux of a single field.
   /** Equivalent to the overload above with a one-entry potential; provided so
       that a scalar flux law needs no reshaping. Aborts if the problem carries
       more than one field, rather than silently using the first.
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
