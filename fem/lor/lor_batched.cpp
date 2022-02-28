// Copyright (c) 2010-2021, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#include "lor_batched.hpp"
#include "lor_restriction.hpp"
#include "../../general/forall.hpp"

// Specializations
#include "lor_diffusion.hpp"

namespace mfem
{

template <typename T>
bool HasIntegrator(BilinearForm &a)
{
   Array<BilinearFormIntegrator*> *integs = a.GetDBFI();
   if (integs != NULL && integs->Size() == 1)
   {
      BilinearFormIntegrator *i = (*integs)[0];
      if (dynamic_cast<T*>(i))
      {
         return true;
      }
   }
   return false;
}

template <typename T1, typename T2>
bool HasIntegrators(BilinearForm &a)
{
   Array<BilinearFormIntegrator*> *integs = a.GetDBFI();
   if (integs != NULL && integs->Size() == 2)
   {
      BilinearFormIntegrator *i0 = (*integs)[0];
      BilinearFormIntegrator *i1 = (*integs)[1];

      if ((dynamic_cast<T1*>(i0) && dynamic_cast<T2*>(i1)) ||
          (dynamic_cast<T2*>(i0) && dynamic_cast<T1*>(i1)))
      {
         return true;
      }
   }
   return false;
}

bool BatchedLORAssembly::FormIsSupported(BilinearForm &a)
{
   // We want to support the following configurations:
   // H1, ND, and RT spaces: M, A, M + K
   if (HasIntegrator<DiffusionIntegrator>(a))
   {
      return true;
   }
   return false;
}

SparseMatrix *BatchedLORAssembly::AssembleWithoutBC()
{
   MFEM_VERIFY(UsesTensorBasis(fes_ho),
               "Batched LOR assembly requires tensor basis");

   const int vsize = fes_ho.GetVSize();
   SparseMatrix *A = new SparseMatrix(vsize, vsize, 0);
   A->GetMemoryI().New(A->Height()+1, A->GetMemoryI().GetMemoryType());
   const int nnz = R.FillI(*A);
   A->GetMemoryJ().New(nnz, A->GetMemoryJ().GetMemoryType());
   A->GetMemoryData().New(nnz, A->GetMemoryData().GetMemoryType());
   R.FillJAndZeroData(*A); // J, A = 0.0

   AssemblyKernel(*A);
   A->Finalize();
   return A;
}

#ifdef MFEM_USE_MPI
void BatchedLORAssembly::ParAssemble(OperatorHandle &A)
{
   // SparseMatrix *A_diag = AssembleWithoutBC();
}
#endif

void BatchedLORAssembly::Assemble(OperatorHandle &A)
{
#ifdef MFEM_USE_MPI
   if (dynamic_cast<ParFiniteElementSpace*>(&fes_ho))
   {
      return ParAssemble(A);
   }
#endif

   SparseMatrix *A_mat = AssembleWithoutBC();

   // Eliminate essential DOFs (BCs) from the matrix (what we do here is
   // equivalent to  DiagonalPolicy::DIAG_KEEP).
   const int n_ess_dofs = ess_dofs.Size();
   const auto ess_dofs_d = ess_dofs.Read();
   const auto I = A_mat->ReadI();
   const auto J = A_mat->ReadJ();
   auto dA = A_mat->ReadWriteData();

   MFEM_FORALL(i, n_ess_dofs,
   {
      const int idof = ess_dofs_d[i];
      for (int j=I[idof]; j<I[idof+1]; ++j)
      {
         const int jdof = J[j];
         if (jdof != idof)
         {
            dA[j] = 0.0;
            for (int k=I[jdof]; k<I[jdof+1]; ++k)
            {
               if (J[k] == idof)
               {
                  dA[k] = 0.0;
                  break;
               }
            }
         }
      }
   });

   A.Reset(A_mat);
}

BatchedLORAssembly::BatchedLORAssembly(LORBase &lor_disc_,
                                       BilinearForm &a_,
                                       FiniteElementSpace &fes_ho_,
                                       const Array<int> &ess_dofs_)
   : lor_disc(lor_disc_), R(fes_ho_), fes_ho(fes_ho_), ess_dofs(ess_dofs_)
{ }

void BatchedLORAssembly::Assemble(LORBase &lor_disc,
                                  BilinearForm &a,
                                  FiniteElementSpace &fes_ho,
                                  const Array<int> &ess_dofs,
                                  OperatorHandle &A)
{
   if (HasIntegrator<DiffusionIntegrator>(a))
   {
      BatchedLORDiffusion(lor_disc, a, fes_ho, ess_dofs).Assemble(A);
   }
}

} // namespace mfem
