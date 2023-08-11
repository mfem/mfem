// Copyright (c) 2010-2023, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#ifndef MFEM_PCOMPLEX_DPGWEAKFORM
#define MFEM_PCOMPLEX_DPGWEAKFORM

#include "mfem.hpp"
#include "complexweakform.hpp"

#ifdef MFEM_USE_MPI

namespace mfem
{

/** @brief Class representing the parallel weak formulation.
   (Convenient for DPG Equations) */
class ParComplexDPGWeakForm : public ComplexDPGWeakForm
{

protected:
   /// Trial FE spaces
   Array<ParFiniteElementSpace * > trial_pfes;

   /// ess_tdof list for each space
   Array<Array<int> *> ess_tdofs;

   /** split ess_tdof_list given in global tdof (for all spaces)
    to individual lists for each space */
   void FillEssTdofLists(const Array<int> & ess_tdof_list);

   // Block Prolongation
   BlockOperator * P = nullptr;
   // Block Restriction
   BlockMatrix * R = nullptr;

   ComplexOperator * p_mat = nullptr;
   BlockOperator * p_mat_r = nullptr;
   BlockOperator * p_mat_i = nullptr;
   BlockOperator * p_mat_e_r = nullptr;
   BlockOperator * p_mat_e_i = nullptr;

   void BuildProlongation();

public:

   ParComplexDPGWeakForm() {}

   /// Creates bilinear form associated with FE spaces @a trial_pfes_.
   ParComplexDPGWeakForm(Array<ParFiniteElementSpace* > & trial_pfes_,
                         Array<FiniteElementCollection* > & fecol_)
      : ComplexDPGWeakForm()
   {
      SetParSpaces(trial_pfes_,fecol_);
   }

   void SetParSpaces(Array<ParFiniteElementSpace* > & trial_pfes_,
                     Array<FiniteElementCollection* > & fecol_)
   {
      trial_pfes = trial_pfes_;
      ess_tdofs.SetSize(trial_pfes.Size());

      Array<FiniteElementSpace * > trial_sfes(trial_pfes.Size());
      for (int i = 0; i<trial_sfes.Size(); i++)
      {
         trial_sfes[i] = (FiniteElementSpace *)trial_pfes[i];
         ess_tdofs[i] = new Array<int>();
      }
      SetSpaces(trial_sfes,fecol_);
   }


   /// Assembles the form i.e. sums over all domain integrators.
   void Assemble(int skip_zeros = 1);

   /// Returns the matrix assembled on the true dofs, i.e. P^t A P.
   /** The returned matrix has to be deleted by the caller. */
   void ParallelAssemble(BlockMatrix *mat_r, BlockMatrix *mat_i);

   void FormLinearSystem(const Array<int> &ess_tdof_list, Vector &x,
                         OperatorHandle &A,
                         Vector &X, Vector &B,
                         int copy_interior = 0);

   void FormSystemMatrix(const Array<int> &ess_tdof_list, OperatorHandle &A);

   /** Call this method after solving a linear system constructed using the
       FormLinearSystem method to recover the solution as a ParGridFunction-size
       vector in x. Use the same arguments as in the FormLinearSystem call. */
   virtual void RecoverFEMSolution(const Vector &X, Vector &x);

   virtual void Update();

   /// Destroys bilinear form.
   virtual ~ParComplexDPGWeakForm();


};

} // namespace mfem


#endif // MFEM_USE_MPI

#endif
