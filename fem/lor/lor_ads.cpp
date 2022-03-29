// Copyright (c) 2010-2022, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#include "lor_ads.hpp"
#include "../../general/forall.hpp"
#include "../../fem/pbilinearform.hpp"

namespace mfem
{

#ifdef MFEM_USE_MPI

BatchedLOR_ADS::BatchedLOR_ADS(BilinearForm &a,
                               ParFiniteElementSpace &pfes_ho_,
                               const Array<int> &ess_dofs)
   : face_fes(pfes_ho_),
     dim(face_fes.GetParMesh()->Dimension()),
     order(face_fes.GetMaxElementOrder()),
     edge_fec(order, dim),
     edge_fes(face_fes.GetParMesh(), &edge_fec),
     ams(a, edge_fes, ess_dofs)
{
   FormCurlMatrix();
}

void BatchedLOR_ADS::FormCurlMatrix()
{

}

BatchedLOR_ADS::~BatchedLOR_ADS()
{
   delete C;
}

LORSolver<HypreADS>::LORSolver(
   ParBilinearForm &a_ho, const Array<int> &ess_tdof_list, int ref_type)
{
   if (BatchedLORAssembly::FormIsSupported(a_ho))
   {
      // ParFiniteElementSpace &pfes = *a_ho.ParFESpace();
      // BatchedLOR_ADS lor_ads(a_ho, pfes, ess_tdof_list);
      // BatchedLOR_RT lor_rt(a_ho, pfes, ess_tdof_list);
      // lor_rt.Assemble(A);
      // xyz = batched_lor.StealCoordinateVector();
      // solver = new HypreADS(*A.As<HypreParMatrix>(),
      //                       batched_lor.StealGradientMatrix(),
      //                       batched_lor.StealXCoordinate(),
      //                       batched_lor.StealYCoordinate(),
      //                       batched_lor.StealZCoordinate());
   }
   else
   {
      ParLORDiscretization lor(a_ho, ess_tdof_list, ref_type);
      // Assume ownership of the system matrix so that `lor` can be safely
      // deleted
      A.Reset(lor.GetAssembledSystem().Ptr());
      lor.GetAssembledSystem().SetOperatorOwner(false);
      solver = new HypreADS(lor.GetAssembledMatrix(), &lor.GetParFESpace());
   }
   width = solver->Width();
   height = solver->Height();
}

void LORSolver<HypreADS>::SetOperator(const Operator &op)
{
   solver->SetOperator(op);
}

void LORSolver<HypreADS>::Mult(const Vector &x, Vector &y) const
{
   solver->Mult(x, y);
}

HypreADS &LORSolver<HypreADS>::GetSolver() { return *solver; }

const HypreADS &LORSolver<HypreADS>::GetSolver() const { return *solver; }

LORSolver<HypreADS>::~LORSolver()
{
   delete solver;
   delete xyz;
}

#endif

} // namespace mfem
