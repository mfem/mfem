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

#include "../config/config.hpp"
#include "../fem/bilinearform.hpp"
#include "../fem/nonlinearform.hpp"

#define MFEM_DARCY_HYBRIDIZATION_ELIM_BCS

namespace mfem
{

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
   };

private:
   FiniteElementSpace *fes_p;
   BilinearFormIntegrator *c_bfi_p{};
   NonlinearFormIntegrator *c_nlfi_p{};
   NonlinearFormIntegrator *m_nlfi_u{}, *m_nlfi_p{};
   bool own_m_nlfi_u{}, own_m_nlfi_p{};
   BlockNonlinearFormIntegrator *m_nlfi{};
   bool own_m_nlfi{};

   /// Set of constraint boundary face integrators to be applied.
   Array<BilinearFormIntegrator*>  boundary_constraint_pot_integs;
   Array<Array<int>*>              boundary_constraint_pot_integs_marker;
   Array<NonlinearFormIntegrator*> boundary_constraint_pot_nonlin_integs;
   Array<Array<int>*>              boundary_constraint_pot_nonlin_integs_marker;
   /// Indicates if the boundary_constraint_pot_integs integrators are owned externally
   int extern_bdr_constr_pot_integs{};

   bool bsym{}, bnl{}, bfin{};
   DiagonalPolicy diag_policy{DIAG_ONE};

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
   real_t *Af_lin_data{}, *Ae_data{};

   Array<int> Bf_offsets, Be_offsets;
   real_t *Bf_data{}, *Be_data{};

   Array<int> Df_offsets, Df_f_offsets;
   mutable real_t *Df_data{}, *Df_lin_data{};
   mutable int *Df_ipiv{};
   bool D_empty{true};

   Array<int> Ct_offsets;
   real_t *Ct_data{};

   mutable Array<int> E_offsets;
   mutable real_t *E_data{};

   Array<int> &G_offsets{E_offsets};
   mutable real_t *G_data{};

   mutable Array<int> H_offsets;
   mutable real_t *H_data{};

   mutable Array<int> darcy_offsets;
   mutable BlockVector darcy_rhs;
   Vector darcy_u, darcy_p;
   mutable Array<int> f_2_b;
   mutable OperatorHandle pGrad;

   friend class Gradient;
   class Gradient : public Operator
   {
      const DarcyHybridization &dh;
   public:
      Gradient(const DarcyHybridization &dh)
         : Operator(dh.Width()), dh(dh) { }

      void Mult(const Vector &x, Vector &y) const override;
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
      Array<FaceElementTransformations*> FTrs;
      Array<IsoparametricTransformation*> NbrTrs;
      const Array<int> offsets;
      mutable Vector Au, Dp, DpEx;
      mutable DenseMatrix grad_A, grad_D;
      mutable BlockOperator grad;

      void AddMultA(const Vector &u_l, Vector &bu) const;
      void AddMultDE(const Vector &p_l, Vector &bp) const;
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

   void GetFDofs(int el, Array<int> &fdofs) const;
   void GetEDofs(int el, Array<int> &edofs) const;
   void AssembleCtFaceMatrix(int face, int el1, int el2, const DenseMatrix &elmat);
   void AssembleCtSubMatrix(int el, const DenseMatrix &elmat,
                            DenseMatrix &Ct, int ioff=0);
   void ConstructC();
   void AllocD() const;
   void AllocEG() const;
   void AllocH() const;
   enum class MultNlMode { Mult, Sol, Grad, GradMult };
   void MultNL(MultNlMode mode, const BlockVector &b, const Vector &x,
               Vector &y) const;
   void InvertA();
   void InvertD();
   void ComputeH();
   FaceElementTransformations * GetCtFaceMatrix(int f, DenseMatrix & Ct_1,
                                                DenseMatrix & Ct_2) const;
   FaceElementTransformations *GetEFaceMatrix(int f, DenseMatrix &E_1,
                                              DenseMatrix &E_2) const;
   FaceElementTransformations *GetGFaceMatrix(int f, DenseMatrix &Gt_1,
                                              DenseMatrix &Gt_2) const;
   void GetHFaceMatrix(int f, DenseMatrix &H) const;
   void GetCtSubMatrix(int el, const Array<int> &c_dofs, DenseMatrix &Ct) const;
   void MultInvNL(int el, const Vector &bu_l, const Vector &bp_l,
                  const BlockVector &x_l, Vector &u_l, Vector &p_l) const;
   void MultInv(int el, const Vector &bu, const Vector &bp, Vector &u,
                Vector &p) const;
   void ConstructGrad(int el, const Array<int> &faces, const BlockVector &x_l,
                      const Vector &u_l,
                      const Vector &p_l) const;
   void AssembleHDGGrad(int el, int f, NonlinearFormIntegrator &nlfi,
                        const Vector &x_f, const Vector &p_l) const;

public:
   /// Constructor
   DarcyHybridization(FiniteElementSpace *fes_u, FiniteElementSpace *fes_p,
                      FiniteElementSpace *fes_c, bool bsymmetrize = true);
   /// Destructor
   ~DarcyHybridization();

   void SetDiagonalPolicy(const DiagonalPolicy diag_policy_)
   { diag_policy = diag_policy_; }

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

   void SetConstraintIntegrator(BilinearFormIntegrator *c_integ) = delete;

   /** Set the integrator that will be used to construct the constraint matrix
       C. The Hybridization object assumes ownership of the integrator, i.e. it
       will delete the integrator when destroyed. */
   void SetConstraintIntegrators(BilinearFormIntegrator *c_flux_integ,
                                 BilinearFormIntegrator *c_pot_integ);

   /** Set the integrator that will be used to construct the constraint matrix
       C. The Hybridization object assumes ownership of the integrator, i.e. it
       will delete the integrator when destroyed. */
   void SetConstraintIntegrators(BilinearFormIntegrator *c_flux_integ,
                                 NonlinearFormIntegrator *c_pot_integ);

   void SetFluxMassNonlinearIntegrator(NonlinearFormIntegrator *flux_integ,
                                       bool own = true);

   void SetPotMassNonlinearIntegrator(NonlinearFormIntegrator *pot_integ,
                                      bool own = true);

   void SetBlockNonlinearIntegrator(BlockNonlinearFormIntegrator *block_integ,
                                    bool own = true);

   BilinearFormIntegrator* GetFluxConstraintIntegrator() const { return c_bfi; }

   BilinearFormIntegrator* GetPotConstraintIntegrator() const { return c_bfi_p; }
   NonlinearFormIntegrator* GetPotConstraintNonlinearIntegrator() const { return c_nlfi_p; }

   NonlinearFormIntegrator* GetFluxMassNonlinearIntegrator() const { return m_nlfi_p; }
   NonlinearFormIntegrator* GetPotMassNonlinearIntegrator() const { return m_nlfi_p; }

   void AddBdrConstraintIntegrator(BilinearFormIntegrator *c_integ) = delete;
   void AddBdrConstraintIntegrator(BilinearFormIntegrator *c_integ,
                                   Array<int> &bdr_marker) = delete;

   Array<BilinearFormIntegrator*> *GetBCBFI() = delete;
   Array<Array<int>*> *GetBCBFI_Marker() = delete;

   void AddBdrFluxConstraintIntegrator(BilinearFormIntegrator *c_integ)
   { Hybridization::AddBdrConstraintIntegrator(c_integ); }

   void AddBdrFluxConstraintIntegrator(BilinearFormIntegrator *c_integ,
                                       Array<int> &bdr_marker)
   { Hybridization::AddBdrConstraintIntegrator(c_integ, bdr_marker); }

   /// Access all integrators added with AddBdrFluxConstraintIntegrator().
   Array<BilinearFormIntegrator*> *GetFluxBCBFI() { return Hybridization::GetBCBFI(); }

   /// Access all boundary markers added with AddBdrFluxConstraintIntegrator().
   /** If no marker was specified when the integrator was added, the
       corresponding pointer (to Array<int>) will be NULL. */
   Array<Array<int>*> *GetFluxBCBFI_Marker() { return Hybridization::GetBCBFI_Marker(); }

   void AddBdrPotConstraintIntegrator(BilinearFormIntegrator *c_integ)
   {
      boundary_constraint_pot_integs.Append(c_integ);
      boundary_constraint_pot_integs_marker.Append(
         NULL); // NULL marker means apply everywhere
   }
   void AddBdrPotConstraintIntegrator(BilinearFormIntegrator *c_integ,
                                      Array<int> &bdr_marker)
   {
      boundary_constraint_pot_integs.Append(c_integ);
      boundary_constraint_pot_integs_marker.Append(&bdr_marker);
   }

   /// Access all integrators added with AddBdrPotConstraintIntegrator().
   Array<BilinearFormIntegrator*> *GetPotBCBFI() { return &boundary_constraint_pot_integs; }

   /// Access all boundary markers added with AddBdrPotConstraintIntegrator().
   /** If no marker was specified when the integrator was added, the
       corresponding pointer (to Array<int>) will be NULL. */
   Array<Array<int>*> *GetPotBCBFI_Marker() { return &boundary_constraint_pot_integs_marker; }

   void AddBdrPotConstraintIntegrator(NonlinearFormIntegrator *c_integ)
   {
      boundary_constraint_pot_nonlin_integs.Append(c_integ);
      boundary_constraint_pot_nonlin_integs_marker.Append(
         NULL); // NULL marker means apply everywhere
   }
   void AddBdrPotConstraintIntegrator(NonlinearFormIntegrator *c_integ,
                                      Array<int> &bdr_marker)
   {
      boundary_constraint_pot_nonlin_integs.Append(c_integ);
      boundary_constraint_pot_nonlin_integs_marker.Append(&bdr_marker);
   }

   /// Access all integrators added with AddBdrPotConstraintIntegrator().
   Array<NonlinearFormIntegrator*> *GetPotBCNLFI() { return &boundary_constraint_pot_nonlin_integs; }

   /// Access all boundary markers added with AddBdrPotConstraintIntegrator().
   /** If no marker was specified when the integrator was added, the
       corresponding pointer (to Array<int>) will be NULL. */
   Array<Array<int>*> *GetPotBCNLFI_Marker() { return &boundary_constraint_pot_nonlin_integs_marker; }

   void UseExternalBdrConstraintIntegrators() = delete;

   /// Indicate that boundary flux constraint integrators are not owned
   void UseExternalBdrFluxConstraintIntegrators() { Hybridization::UseExternalBdrConstraintIntegrators(); }

   /// Indicate that boundary potential constraint integrators are not owned
   void UseExternalBdrPotConstraintIntegrators() { extern_bdr_constr_pot_integs = 1; }

   /// Prepare the Hybridization object for assembly.
   void Init(const Array<int> &ess_flux_tdof_list) override;

   /// Assemble the element matrix A into the hybridized system matrix.
   void AssembleMatrix(int el, const DenseMatrix &A) override
   { MFEM_ABORT("Not supported, system part must be specified"); }

   void AssembleFluxMassMatrix(int el, const DenseMatrix &A);

   void AssemblePotMassMatrix(int el, const DenseMatrix &D);

   void AssembleDivMatrix(int el, const DenseMatrix &B);

   void ComputeAndAssemblePotFaceMatrix(int face,
                                        DenseMatrix & elmat1, DenseMatrix & elmat2,
                                        Array<int>& vdofs1, Array<int>& vdofs2);

   void ComputeAndAssemblePotBdrFaceMatrix(int bface, DenseMatrix & elmat,
                                           Array<int>& vdofs);

   /// Assemble the boundary element matrix A into the hybridized system matrix.
   //void AssembleBdrMatrix(int bdr_el, const DenseMatrix &A);

   /// Operator application: `y=A(x)`.
   void Mult(const Vector &x, Vector &y) const override;

   /// Evaluate the gradient operator at the point @a x.
   Operator &GetGradient(const Vector &x) const override;

   /// Finalize the construction of the hybridized matrix.
   void Finalize() override;

   /** @brief Use the stored eliminated part of the matrix to modify the r.h.s.
       @a b; @a vdofs_flux is a list of DOFs (non-directional, i.e. >= 0). */
   void EliminateVDofsInRHS(const Array<int> &vdofs_flux,
                            const BlockVector &x, BlockVector &b);

   void ReduceRHS(const Vector &b, Vector &b_r) const override
   { MFEM_ABORT("Use BlockVector version instead"); }

   /** Perform the reduction of the given r.h.s. vector, b, to a r.h.s vector,
       b_r, for the hybridized system. */
   void ReduceRHS(const BlockVector &b, Vector &b_r) const;

   void ComputeSolution(const Vector &b, const Vector &sol_r,
                        Vector &sol) const override
   { MFEM_ABORT("Use BlockVector version instead"); }

   /** Reconstruct the solution of the original system, sol, from solution of
       the hybridized system, sol_r, and the original r.h.s. vector, b.
       It is assumed that the vector sol has the right essential b.c. */
   void ComputeSolution(const BlockVector &b, const Vector &sol_r,
                        BlockVector &sol) const;

   void Reset() override;
};

}

#endif
