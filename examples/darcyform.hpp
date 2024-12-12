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
#include "darcyreduction.hpp"
#include "darcyhybridization.hpp"

namespace mfem
{

class DarcyForm : public Operator
{
   Array<int> offsets;

   FiniteElementSpace *fes_u, *fes_p;

   bool bsym;

   BilinearForm *M_u{}, *M_p{};
   NonlinearForm *Mnl_u{}, *Mnl_p{};
   MixedBilinearForm *B{};
   BlockNonlinearForm *Mnl{};

   OperatorHandle pM_u, pM_p, pB, pBt, pM;
   mutable OperatorHandle pG;

   /// The assembly level of the form (full, partial, etc.)
   AssemblyLevel assembly{AssemblyLevel::LEGACY};

   BlockOperator *block_op{};
   mutable BlockOperator *block_grad{};

   DarcyReduction *reduction{}; ///< Owned.
   DarcyHybridization *hybridization{}; ///< Owned.

   void EnableReduction(const Array<int> &ess_flux_tdof_list,
                        DarcyReduction *reduction);

   void AssembleDivLDGFaces(int skip_zeros);
   void AssemblePotLDGFaces(int skip_zeros);
   void AssemblePotHDGFaces(int skip_zeros);

   void AllocBlockOp();
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

   NonlinearForm *GetFluxMassNonlinearForm();
   const NonlinearForm *GetFluxMassNonlinearForm() const;

   NonlinearForm *GetPotentialMassNonlinearForm();
   const NonlinearForm *GetPotentialMassNonlinearForm() const;

   MixedBilinearForm *GetFluxDivForm();
   const MixedBilinearForm *GetFluxDivForm() const;

   BlockNonlinearForm *GetBlockNonlinearForm();
   const BlockNonlinearForm *GetBlockNonlinearForm() const;

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

   /// Enable flux reduction.
   /** For details see the description for class
       DarcyFluxReduction in darcyreduction.hpp. This method should be
       called before assembly. */
   void EnableFluxReduction()
   {
      Array<int> ess_flux_tdof_list; //empty
      EnableReduction(ess_flux_tdof_list, new DarcyFluxReduction(fes_u, fes_p));
   }

   /// Enable potential reduction.
   /** For details see the description for class
       DarcyPotentialReduction in darcyreduction.hpp. This method should be
       called before assembly. */
   void EnablePotentialReduction(const Array<int> &ess_flux_tdof_list)
   { EnableReduction(ess_flux_tdof_list, new DarcyPotentialReduction(fes_u, fes_p)); }

   DarcyReduction *GetReduction() const { return reduction; }

   /// Enable hybridization.
   /** For details see the description for class
       DarcyHybridization in darcyhybridization.hpp. This method should be called
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
   void Mult (const Vector & x, Vector & y) const override;

   /// Action of the transpose operator
   void MultTranspose (const Vector & x, Vector & y) const override;

   /// Evaluate the gradient operator at the point @a x.
   Operator &GetGradient(const Vector &x) const override;

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

}

#endif
