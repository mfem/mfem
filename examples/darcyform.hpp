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

#ifndef MFEM_DARCYFORM
#define MFEM_DARCYFORM

#include "../config/config.hpp"
#include "../fem/bilinearform.hpp"
#include "../fem/nonlinearform.hpp"

namespace mfem
{

class DarcyHybridization;

class DarcyForm : public Operator
{
   Array<int> offsets;

   FiniteElementSpace *fes_u, *fes_p;

   bool bsym;

   BilinearForm *M_u, *M_p;
   NonlinearForm *Mnl_p;
   MixedBilinearForm *B;

   OperatorHandle pM_u, pM_p, pB, pBt;

   /// The assembly level of the form (full, partial, etc.)
   AssemblyLevel assembly;

   BlockOperator *block_op;

   DarcyHybridization *hybridization; ///< Owned.

   void AssemblePotHDGFaces(int skip_zeros);

   const Operator* ConstructBT(const MixedBilinearForm *B);
   const Operator* ConstructBT(const Operator *opB);

public:
   DarcyForm(FiniteElementSpace *fes_u, FiniteElementSpace *fes_p,
             bool bsymmetrize = true);

   inline const Array<int>& GetOffsets() const { return offsets; }

   BilinearForm *GetFluxMassForm();
   const BilinearForm *GetFluxMassForm() const;

   BilinearForm *GetPotentialMassForm();
   const BilinearForm *GetPotentialMassForm() const;

   NonlinearForm *GetPotentialMassNonlinearForm();
   const NonlinearForm *GetPotentialMassNonlinearForm() const;

   MixedBilinearForm *GetFluxDivForm();
   const MixedBilinearForm *GetFluxDivForm() const;

   /// Set the desired assembly level.
   /** Valid choices are:

       - AssemblyLevel::LEGACY (default)
       - AssemblyLevel::FULL
       - AssemblyLevel::PARTIAL
       - AssemblyLevel::ELEMENT
       - AssemblyLevel::NONE

       If used, this method must be called before assembly. */
   void SetAssemblyLevel(AssemblyLevel assembly_level);

   /// Returns the assembly level
   AssemblyLevel GetAssemblyLevel() const { return assembly; }

   /// Enable hybridization.
   /** For details see the description for class
       Hybridization in fem/hybridization.hpp. This method should be called
       before assembly. */
   void EnableHybridization(FiniteElementSpace *constr_space,
                            BilinearFormIntegrator *constr_flux_integ,
                            const Array<int> &ess_flux_tdof_list);

   DarcyHybridization *GetHybridization() const { return hybridization; }

   /// Assembles the form i.e. sums over all domain/bdr integrators.
   void Assemble(int skip_zeros = 1);

   /// Finalizes the matrix initialization.
   void Finalize(int skip_zeros = 1);

   /** @brief Form the linear system A X = B, corresponding to this bilinear
       form and the linear form @a b(.). */
   /** This method applies any necessary transformations to the linear system
       such as: eliminating boundary conditions; applying conforming constraints
       for non-conforming AMR; parallel assembly; static condensation;
       hybridization.

       The GridFunction-size vector @a x must contain the essential b.c. The
       BilinearForm and the LinearForm-size vector @a b must be assembled.

       The vector @a X is initialized with a suitable initial guess: when using
       hybridization, the vector @a X is set to zero; otherwise, the essential
       entries of @a X are set to the corresponding b.c. and all other entries
       are set to zero (@a copy_interior == 0) or copied from @a x
       (@a copy_interior != 0).

       This method can be called multiple times (with the same @a ess_tdof_list
       array) to initialize different right-hand sides and boundary condition
       values.

       After solving the linear system, the finite element solution @a x can be
       recovered by calling RecoverFEMSolution() (with the same vectors @a X,
       @a b, and @a x).

       NOTE: If there are no transformations, @a X simply reuses the data of
             @a x. */
   void FormLinearSystem(const Array<int> &ess_flux_tdof_list,
                         BlockVector &x, BlockVector &b, OperatorHandle &A, Vector &X,
                         Vector &B, int copy_interior = 0);

   /** @brief Form the linear system A X = B, corresponding to this bilinear
       form and the linear form @a b(.). */
   /** Version of the method FormLinearSystem() where the system matrix is
       returned in the variable @a A, of type OpType, holding a *reference* to
       the system matrix (created with the method OpType::MakeRef()). The
       reference will be invalidated when SetOperatorType(), Update(), or the
       destructor is called. */
   template <typename OpType>
   void FormLinearSystem(const Array<int> &ess_flux_tdof_list,
                         Vector &x, Vector &b,
                         OpType &A, Vector &X, Vector &B,
                         int copy_interior = 0)
   {
      OperatorHandle Ah;
      FormLinearSystem(ess_flux_tdof_list, x, b, Ah, X, B, copy_interior);
      OpType *A_ptr = Ah.Is<OpType>();
      MFEM_VERIFY(A_ptr, "invalid OpType used");
      A.MakeRef(*A_ptr);
   }

   /// Form the linear system matrix @a A, see FormLinearSystem() for details.
   virtual void FormSystemMatrix(const Array<int> &ess_flux_tdof_list,
                                 OperatorHandle &A);

   /// Form the linear system matrix A, see FormLinearSystem() for details.
   /** Version of the method FormSystemMatrix() where the system matrix is
       returned in the variable @a A, of type OpType, holding a *reference* to
       the system matrix (created with the method OpType::MakeRef()). The
       reference will be invalidated when SetOperatorType(), Update(), or the
       destructor is called. */
   template <typename OpType>
   void FormSystemMatrix(const Array<int> &ess_flux_tdof_list, OpType &A)
   {
      OperatorHandle Ah;
      FormSystemMatrix(ess_flux_tdof_list, Ah);
      OpType *A_ptr = Ah.Is<OpType>();
      MFEM_VERIFY(A_ptr, "invalid OpType used");
      A.MakeRef(*A_ptr);
   }

   /// Recover the solution of a linear system formed with FormLinearSystem().
   /** Call this method after solving a linear system constructed using the
       FormLinearSystem() method to recover the solution as a GridFunction-size
       vector in @a x. Use the same arguments as in the FormLinearSystem() call.
   */
   void RecoverFEMSolution(const Vector &X, const Vector &b, Vector &x) override
   { MFEM_ABORT("This class uses BlockVectors instead of Vectors."); }

   void RecoverFEMSolution(const Vector &X, const BlockVector &b, BlockVector &x);

   /** @brief Use the stored eliminated part of the matrix (see
       EliminateVDofs(const Array<int> &, DiagonalPolicy)) to modify the r.h.s.
       @a b; @a vdofs_flux is a list of DOFs (non-directional, i.e. >= 0). */
   void EliminateVDofsInRHS(const Array<int> &vdofs_flux,
                            const BlockVector &x, BlockVector &b);

   /// Operator application
   void Mult (const Vector & x, Vector & y) const override { block_op->Mult(x, y); }

   /// Action of the transpose operator
   void MultTranspose (const Vector & x, Vector & y) const override { block_op->MultTranspose(x, y); }

   /// Return the flux FE space associated with the DarcyForm.
   FiniteElementSpace *FluxFESpace() { return fes_u; }
   /// Read-only access to the associated flux FiniteElementSpace.
   const FiniteElementSpace *FluxFESpace() const { return fes_u; }

   /// Return the flux FE space associated with the DarcyForm.
   FiniteElementSpace *PotentialFESpace() { return fes_p; }
   /// Read-only access to the associated flux FiniteElementSpace.
   const FiniteElementSpace *PotentialFESpace() const { return fes_p; }

   virtual void Update();

   /// Destroys Darcy form.
   virtual ~DarcyForm();

   /// Return the type ID of the Operator class.
   Type GetType() const { return MFEM_Block_Operator; }
};

class DarcyHybridization : public Hybridization, public Operator
{
public:
   enum class LSsolveType
   {
      LBFGS,
      Newton,
   };

   enum class LPrecType
   {
      GMRES,
   };

private:
   FiniteElementSpace *fes_p;
   BilinearFormIntegrator *c_bfi_p;
   NonlinearFormIntegrator *c_nlfi_p;
   NonlinearFormIntegrator *m_nlfi_p;
   bool own_m_nlfi_p;

   /// Set of constraint boundary face integrators to be applied.
   Array<BilinearFormIntegrator*>  boundary_constraint_pot_integs;
   Array<Array<int>*>              boundary_constraint_pot_integs_marker;
   Array<NonlinearFormIntegrator*> boundary_constraint_pot_nonlin_integs;
   Array<Array<int>*>              boundary_constraint_pot_nonlin_integs_marker;
   /// Indicates if the boundary_constraint_pot_integs integrators are owned externally
   int extern_bdr_constr_pot_integs;

   bool bsym, bnl, bfin;

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
   real_t *Ae_data;

   Array<int> Bf_offsets, Be_offsets;
   real_t *Bf_data, *Be_data;

   Array<int> Df_offsets, Df_f_offsets;
   real_t *Df_data;
   int *Df_ipiv;

   Array<int> Ct_offsets;
   real_t *Ct_data;

   Array<int> E_offsets;
   real_t *E_data;

   Array<int> &G_offsets{E_offsets};
   real_t *G_data;

   mutable Array<int> darcy_offsets;
   mutable BlockVector darcy_rhs;

   friend class LocalNLOperator;
   class LocalNLOperator : public Operator
   {
      const DarcyHybridization &dh;
      int el;
      const Vector &bu;
      const BlockVector &trps;
      const Array<int> &faces;

      const int a_dofs_size, d_dofs_size;
      LUFactors LU_A;
      DenseMatrix B;
      const FiniteElement *fe;
      ElementTransformation *Tr;
      mutable Vector u_l, Dp, DpEx;
      mutable DenseMatrix grad;

   public:
      LocalNLOperator(const DarcyHybridization &dh, int el, const Vector &bu,
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
   void AllocEG();
   void MultNL(int mode, const BlockVector &b, const Vector &x, Vector &y) const;
   void InvertA();
   void ComputeH();
   FaceElementTransformations * GetCtFaceMatrix(int f, DenseMatrix & Ct_1,
                                                DenseMatrix & Ct_2, Array<int>& c_dofs) const;
   FaceElementTransformations *GetEFaceMatrix(int f, DenseMatrix &E_1,
                                              DenseMatrix &E_2, Array<int> &c_dofs) const;
   FaceElementTransformations *GetGFaceMatrix(int f, DenseMatrix &Gt_1,
                                              DenseMatrix &Gt_2, Array<int> &c_dofs) const;
   void GetCtSubMatrix(int el, const Array<int> &c_dofs, DenseMatrix &Ct) const;
   void MultInvNL(int el, const Vector &bu_l, const Vector &bp_l, const Vector &x,
                  Vector &u_l, Vector &p_l) const;
   void MultInv(int el, const Vector &bu, const Vector &bp, Vector &u,
                Vector &p) const;

public:
   /// Constructor
   DarcyHybridization(FiniteElementSpace *fes_u, FiniteElementSpace *fes_p,
                      FiniteElementSpace *fes_c, bool bsymmetrize = true);
   /// Destructor
   ~DarcyHybridization();

   void SetLocalNLSolver(LSsolveType type, int iters = 1000, real_t rtol = 1e-6,
                         real_t atol = 0., int print_lvl = 0)
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

   void SetPotMassNonlinearIntegrator(NonlinearFormIntegrator *pot_integ,
                                      bool own = true);

   BilinearFormIntegrator* GetFluxConstraintIntegrator() const { return c_bfi; }

   BilinearFormIntegrator* GetPotConstraintIntegrator() const { return c_bfi_p; }
   NonlinearFormIntegrator* GetPotConstraintNonlinearIntegrator() const { return c_nlfi_p; }

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
