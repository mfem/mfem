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

#include "blockstaticcond.hpp"

namespace mfem
{


BlockStaticCondensation::BlockStaticCondensation(Array<FiniteElementSpace *> &
                                                 fes_)
{
   SetSpaces(fes_);
}

void BlockStaticCondensation::SetSpaces(Array<FiniteElementSpace*> & fes_)
{
   fes = fes_;
   nblocks = fes.Size();
   tr_fes.SetSize(nblocks);
   IsTraceSpace.SetSize(nblocks);
   mesh = fes[0]->GetMesh();
   const FiniteElementCollection * fec;
   for (int i = 0; i < nblocks; i++)
   {
      fec = fes[i]->FEColl();
      IsTraceSpace[i] =
         (dynamic_cast<const H1_Trace_FECollection*>(fec) ||
          dynamic_cast<const ND_Trace_FECollection*>(fec) ||
          dynamic_cast<const RT_Trace_FECollection*>(fec));

      // skip if it's an L2 space (no trace space to construct)
      tr_fes[i] = (fec->GetContType() == FiniteElementCollection::DISCONTINUOUS) ?
                  nullptr : (IsTraceSpace[i]) ? fes[i] :
                  new FiniteElementSpace(mesh, fec->GetTraceCollection(), fes[i]->GetVDim(),
                                         fes[i]->GetOrdering());
   }
}

void BlockStaticCondensation::Init()
{

}

void BlockStaticCondensation::GetReduceElementIndices(int el,
                                                      Array<int> & trace_ldofs,
                                                      Array<int> & interior_ldofs)
{
   int dim = mesh->Dimension();
   Array<int> dofs;
   Array<int> faces, ori;
   if (dim == 1)
   {
      mesh->GetElementVertices(el, faces);
   }
   if (dim == 2)
   {
      mesh->GetElementEdges(el, faces, ori);
   }
   else //dim = 3
   {
      mesh->GetElementFaces(el,faces,ori);
   }
   int numfaces = faces.Size();

   trace_ldofs.SetSize(0);
   interior_ldofs.SetSize(0);
   // construct Array of bubble dofs to be extracted
   int skip=0;
   Array<int> tr_dofs;
   Array<int> int_dofs;
   for (int i = 0; i<tr_fes.Size(); i++)
   {
      int td = 0;
      int ndof;
      // if it's an L2 space (bubbles)
      if (!tr_fes[i])
      {
         ndof = fes[i]->GetVDim()*fes[i]->GetFE(el)->GetDof();
         td = 0;
      }
      else if (IsTraceSpace[i])
      {
         for (int iface = 0; iface < numfaces; iface++)
         {
            td += fes[i]->GetVDim()*fes[i]->GetFaceElement(faces[iface])->GetDof();
         }
         ndof = td;
      }
      else
      {
         Array<int> trace_dofs;
         ndof = fes[i]->GetVDim()*fes[i]->GetFE(el)->GetDof();
         tr_fes[i]->GetElementVDofs(el, trace_dofs);
         td = trace_dofs.Size(); // number of trace dofs
         mfem::out << ndof << std::endl;
         mfem::out << td << std::endl;
      }

      tr_dofs.SetSize(td);
      int_dofs.SetSize(ndof - td);
      for (int j = 0; j<td; j++)
      {
         tr_dofs[j] = skip + j;
      }
      for (int j = 0; j<ndof-td; j++)
      {
         int_dofs[j] = skip + td + j;
      }
      skip+=ndof;

      trace_ldofs.Append(tr_dofs);
      interior_ldofs.Append(int_dofs);
   }
}


void BlockStaticCondensation::AssembleReducedSystem(int el,
                                                    const DenseMatrix &elmat,
                                                    const Vector & elvect)
{
   mfem::out << "Assemble matrix and vector for element no: " << el << std::endl;
   // Get Shur Complement
   Array<int> tr_idx, int_idx;
   GetReduceElementIndices(el, tr_idx,int_idx);

   // mfem::out << "elmat  size = " << elmat.Height() << " x " << elmat.Width() <<
   //           std::endl;
   // mfem::out << "elvect size = " << elvect.Size() << std::endl;
   // mfem::out << "tr_idx   = " ; tr_idx.Print() ;
   // mfem::out << "int_idx  = " ; int_idx.Print() ;
   // std::cin.get();

   // TODO
   // Assemble matrix and rhs
   // 1. Extract the submatrices based on tr_idx and int_idx
   // 2. Construct the appopriate offsets (same as in NormalEquations::Assembly())
   // 3. ...
}

void BlockStaticCondensation::Finalize()
{

}

void BlockStaticCondensation::SetEssentialTrueDofs(const Array<int>
                                                   &ess_tdof_list)
{

}

void BlockStaticCondensation::EliminateReducedTrueDofs(const Array<int>
                                                       &ess_rtdof_list,
                                                       Matrix::DiagonalPolicy dpolicy)
{

}

void BlockStaticCondensation::EliminateReducedTrueDofs(Matrix::DiagonalPolicy
                                                       dpolicy)
{

}

void BlockStaticCondensation::ReduceRHS(const Vector &b, Vector &sc_b) const
{

}

void BlockStaticCondensation::ReduceSolution(const Vector &sol,
                                             Vector &sc_sol) const
{

}

void BlockStaticCondensation::ReduceSystem(Vector &x, Vector &b, Vector &X,
                                           Vector &B,
                                           int copy_interior) const
{

}

void BlockStaticCondensation::ConvertMarkerToReducedTrueDofs(
   const Array<int> &ess_tdof_marker,
   Array<int> &ess_rtdof_marker) const
{

}

void BlockStaticCondensation::ConvertListToReducedTrueDofs(
   const Array<int> &ess_tdof_list,
   Array<int> &ess_rtdof_list) const
{

}

void BlockStaticCondensation::ComputeSolution(const Vector &b,
                                              const Vector &sc_sol,
                                              Vector &sol) const
{

}

BlockStaticCondensation::~BlockStaticCondensation()
{

}

}

