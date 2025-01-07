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

#ifndef MFEM_PDARCYFORM
#define MFEM_PDARCYFORM

#include "../../config/config.hpp"

#ifdef MFEM_USE_MPI

#include "darcyform.hpp"
#include "../pbilinearform.hpp"

namespace mfem
{

class ParDarcyForm : public DarcyForm
{
protected:
   Array<int> toffsets;

   ParFiniteElementSpace *pfes_u, *pfes_p;

   ParBilinearForm *pM_u{}, *pM_p{};
   //NonlinearForm *Mnl_u{}, *Mnl_p{};
   ParMixedBilinearForm *pB{};
   //BlockNonlinearForm *Mnl{};

   void AllocBlockOp();

   using DarcyForm::ConstructBT;
   const Operator* ConstructBT(const HypreParMatrix *opB);

public:
   ParDarcyForm(ParFiniteElementSpace *fes_u, ParFiniteElementSpace *fes_p,
                bool bsymmetrize = true);

   inline const Array<int>& GetTrueOffsets() const { return toffsets; }

   using DarcyForm::GetFluxMassForm;
   BilinearForm *GetFluxMassForm();

   ParBilinearForm *GetParFluxMassForm()
   { return static_cast<ParBilinearForm*>(GetFluxMassForm()); }

   const ParBilinearForm *GetParFluxMassForm() const { return pM_u; }

   using DarcyForm::GetPotentialMassForm;
   BilinearForm *GetPotentialMassForm();

   ParBilinearForm *GetParPotentialMassForm()
   { return static_cast<ParBilinearForm*>(GetPotentialMassForm()); }

   const ParBilinearForm *GetParPotentialMassForm() const { return pM_p; }

   //NonlinearForm *GetFluxMassNonlinearForm();
   //const NonlinearForm *GetFluxMassNonlinearForm() const;

   //NonlinearForm *GetPotentialMassNonlinearForm();
   //const NonlinearForm *GetPotentialMassNonlinearForm() const;

   using DarcyForm::GetFluxDivForm;
   MixedBilinearForm *GetFluxDivForm();

   ParMixedBilinearForm *GetParFluxDivForm()
   { return static_cast<ParMixedBilinearForm*>(GetFluxDivForm()); }

   const ParMixedBilinearForm *GetParFluxDivForm() const { return pB; }

   //BlockNonlinearForm *GetBlockNonlinearForm();
   //const BlockNonlinearForm *GetBlockNonlinearForm() const;

   /// Assembles the form i.e. sums over all domain/bdr integrators.
   //void Assemble(int skip_zeros = 1);

   /// Finalizes the matrix initialization.
   void Finalize(int skip_zeros = 1);

   /// Assembles the form on the true dofs, i.e. P^t A P.
   void ParallelAssembleInternal();

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
                         Vector &B, int copy_interior = 0) override;

   /// Form the linear system matrix @a A, see FormLinearSystem() for details.
   virtual void FormSystemMatrix(const Array<int> &ess_flux_tdof_list,
                                 OperatorHandle &A);

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
   void ParallelEliminateTDofsInRHS(const Array<int> &tdofs_flux,
                                    const BlockVector &x, BlockVector &b);

   /// Operator application
   void Mult (const Vector & x, Vector & y) const override;

   /// Return the flux FE space associated with the DarcyForm.
   ParFiniteElementSpace *ParFluxFESpace() { return pfes_u; }
   /// Read-only access to the associated flux FiniteElementSpace.
   const ParFiniteElementSpace *ParFluxFESpace() const { return pfes_u; }

   /// Return the flux FE space associated with the DarcyForm.
   ParFiniteElementSpace *ParPotentialFESpace() { return pfes_p; }
   /// Read-only access to the associated flux FiniteElementSpace.
   const ParFiniteElementSpace *ParPotentialFESpace() const { return pfes_p; }

   //virtual void Update();

   /// Destroys Darcy form.
   virtual ~ParDarcyForm();

   /// Return the type ID of the Operator class.
   Type GetType() const { return MFEM_Block_Operator; }
};

}

#endif // MFEM_USE_MPI

#endif
