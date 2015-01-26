// Copyright (c) 2010, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-443211. All Rights
// reserved. See file COPYRIGHT for details.
//
// This file is part of the MFEM library. For more information and source code
// availability see http://mfem.googlecode.com.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the GNU Lesser General Public License (as published by the Free
// Software Foundation) version 2.1 dated February 1999.

#include "../config/config.hpp"

#ifdef MFEM_USE_MPI

#include "fem.hpp"
#include <iostream>
#include <limits>
using namespace std;

namespace mfem
{

ParGridFunction::ParGridFunction(ParFiniteElementSpace *pf, GridFunction *gf)
{
   fes = pfes = pf;
   SetDataAndSize(gf->GetData(), gf->Size());
}

ParGridFunction::ParGridFunction(ParFiniteElementSpace *pf, HypreParVector *tv)
   : GridFunction(pf), pfes(pf)
{
   Distribute(tv);
}

ParGridFunction::ParGridFunction(ParMesh *pmesh, GridFunction *gf, int * partitioning)
{
   // duplicate the FiniteElementCollection from 'gf'
   fec = FiniteElementCollection::New(gf->FESpace()->FEColl()->Name());
   fes = pfes = new ParFiniteElementSpace(pmesh, fec, gf->FESpace()->GetVDim(),
                                          gf->FESpace()->GetOrdering());
   SetSize(pfes->GetVSize());

   if(partitioning)
   {
      Array<int> gvdofs, lvdofs;
      Vector lnodes;
      int element_counter = 0;
      Mesh & mesh(*gf->FESpace()->GetMesh());
      int MyRank;
      MPI_Comm_rank(pfes->GetComm(), &MyRank);
      for (int i = 0; i < mesh.GetNE(); i++)
         if (partitioning[i] == MyRank)
         {
            pfes->GetElementVDofs(element_counter, lvdofs);
            gf->FESpace()->GetElementVDofs(i, gvdofs);
            gf->GetSubVector(gvdofs, lnodes);
            SetSubVector(lvdofs, lnodes);
            element_counter++;
         }
   }
}

void ParGridFunction::Update(ParFiniteElementSpace *f)
{
   face_nbr_data.Destroy();
   GridFunction::Update(f);
   pfes = f;
}

void ParGridFunction::Update(ParFiniteElementSpace *f, Vector &v, int v_offset)
{
   face_nbr_data.Destroy();
   GridFunction::Update(f, v, v_offset);
   pfes = f;
}

void ParGridFunction::Distribute(const Vector *tv)
{
   pfes->Dof_TrueDof_Matrix()->Mult(*tv, *this);
}

void ParGridFunction::GetTrueDofs(Vector &tv) const
{
#if 0
   for (int i = 0; i < size; i++)
   {
      int tdof = pfes->GetLocalTDofNumber(i);
      if (tdof >= 0)
         tv(tdof) = (*this)(i);
   }
#else
   hypre_ParCSRMatrix *P = *pfes->Dof_TrueDof_Matrix();
   hypre_CSRMatrix *diag = hypre_ParCSRMatrixDiag(P);
   int *I = hypre_CSRMatrixI(diag) + 1;
   int *J = hypre_CSRMatrixJ(diag);
   for (int i = 0, j = 0; i < size; i++)
      if (j < I[i])
         tv(J[j++]) = (*this)(i);
#endif
}

HypreParVector *ParGridFunction::GetTrueDofs() const
{
   HypreParVector *tv = pfes->NewTrueDofVector();
   GetTrueDofs(*tv);
   return tv;
}

void ParGridFunction::ParallelAverage(Vector &tv) const
{
   pfes->Dof_TrueDof_Matrix()->MultTranspose(*this, tv);
   pfes->DivideByGroupSize(tv);
}

void ParGridFunction::ParallelAverage(HypreParVector &tv) const
{
   pfes->Dof_TrueDof_Matrix()->MultTranspose(*this, tv);
   pfes->DivideByGroupSize(tv);
}

HypreParVector *ParGridFunction::ParallelAverage() const
{
   HypreParVector *tv = pfes->NewTrueDofVector();
   ParallelAverage(*tv);
   return tv;
}

void ParGridFunction::ParallelAssemble(Vector &tv) const
{
   pfes->Dof_TrueDof_Matrix()->MultTranspose(*this, tv);
}

void ParGridFunction::ParallelAssemble(HypreParVector &tv) const
{
   pfes->Dof_TrueDof_Matrix()->MultTranspose(*this, tv);
}

HypreParVector *ParGridFunction::ParallelAssemble() const
{
   HypreParVector *tv = pfes->NewTrueDofVector();
   ParallelAssemble(*tv);
   return tv;
}

void ParGridFunction::ExchangeFaceNbrData()
{
   pfes->ExchangeFaceNbrData();

   if (pfes->GetFaceNbrVSize() <= 0)
      return;

   ParMesh *pmesh = pfes->GetParMesh();

   face_nbr_data.SetSize(pfes->GetFaceNbrVSize());
   Vector send_data(pfes->send_face_nbr_ldof.Size_of_connections());

   int *send_offset = pfes->send_face_nbr_ldof.GetI();
   int *send_ldof = pfes->send_face_nbr_ldof.GetJ();
   int *recv_offset = pfes->face_nbr_gdof.GetI();
   MPI_Comm MyComm = pfes->GetComm();

   int num_face_nbrs = pmesh->GetNFaceNeighbors();
   MPI_Request *requests = new MPI_Request[2*num_face_nbrs];
   MPI_Request *send_requests = requests;
   MPI_Request *recv_requests = requests + num_face_nbrs;
   MPI_Status  *statuses = new MPI_Status[num_face_nbrs];

   for (int i = 0; i < send_data.Size(); i++)
      send_data[i] = data[send_ldof[i]];

   for (int fn = 0; fn < num_face_nbrs; fn++)
   {
      int nbr_rank = pmesh->GetFaceNbrRank(fn);
      int tag = 0;

      MPI_Isend(&send_data(send_offset[fn]),
                send_offset[fn+1] - send_offset[fn],
                MPI_DOUBLE, nbr_rank, tag, MyComm, &send_requests[fn]);

      MPI_Irecv(&face_nbr_data(recv_offset[fn]),
                recv_offset[fn+1] - recv_offset[fn],
                MPI_DOUBLE, nbr_rank, tag, MyComm, &recv_requests[fn]);
   }

   MPI_Waitall(num_face_nbrs, send_requests, statuses);
   MPI_Waitall(num_face_nbrs, recv_requests, statuses);

   delete [] statuses;
   delete [] requests;
}

double ParGridFunction::GetValue(int i, const IntegrationPoint &ip, int vdim)
   const
{
   Array<int> dofs;
   Vector DofVal, LocVec;
   int nbr_el_no = i - pfes->GetParMesh()->GetNE();
   if (nbr_el_no >= 0)
   {
      int fes_vdim = pfes->GetVDim();
      pfes->GetFaceNbrElementVDofs(nbr_el_no, dofs);
      if (fes_vdim > 1)
      {
         int s = dofs.Size()/fes_vdim;
         Array<int> _dofs(&dofs[(vdim-1)*s], s);
         face_nbr_data.GetSubVector(_dofs, LocVec);
         DofVal.SetSize(s);
      }
      else
      {
         face_nbr_data.GetSubVector(dofs, LocVec);
         DofVal.SetSize(dofs.Size());
      }
      pfes->GetFaceNbrFE(nbr_el_no)->CalcShape(ip, DofVal);
   }
   else
   {
      fes->GetElementDofs(i, dofs);
      fes->DofsToVDofs(vdim-1, dofs);
      DofVal.SetSize(dofs.Size());
      fes->GetFE(i)->CalcShape(ip, DofVal);
      GetSubVector(dofs, LocVec);
   }

   return (DofVal * LocVec);
}

void ParGridFunction::ProjectCoefficient(Coefficient &coeff)
{
   DeltaCoefficient *delta_c = dynamic_cast<DeltaCoefficient *>(&coeff);

   if (delta_c == NULL)
   {
      GridFunction::ProjectCoefficient(coeff);
   }
   else
   {
      double loc_integral, glob_integral;

      ProjectDeltaCoefficient(*delta_c, loc_integral);

      MPI_Allreduce(&loc_integral, &glob_integral, 1, MPI_DOUBLE, MPI_SUM,
                    pfes->GetComm());

      (*this) *= (delta_c->Scale() / glob_integral);
   }
}

void ParGridFunction::Save(std::ostream &out) const
{
   for (int i = 0; i < size; i++)
      if (pfes->GetDofSign(i) < 0)
         data[i] = -data[i];

   GridFunction::Save(out);

   for (int i = 0; i < size; i++)
      if (pfes->GetDofSign(i) < 0)
         data[i] = -data[i];
}

void ParGridFunction::SaveAsOne(std::ostream &out)
{
   int i, p;

   MPI_Comm MyComm;
   MPI_Status status;
   int MyRank, NRanks;

   MyComm = pfes -> GetComm();

   MPI_Comm_size(MyComm, &NRanks);
   MPI_Comm_rank(MyComm, &MyRank);

   double **values = new double*[NRanks];
   int *nv = new int[NRanks];
   int *nvdofs = new int[NRanks];
   int *nedofs = new int[NRanks];
   int *nfdofs = new int[NRanks];
   int *nrdofs = new int[NRanks];

   values[0] = data;
   nv[0]     = pfes -> GetVSize();
   nvdofs[0] = pfes -> GetNVDofs();
   nedofs[0] = pfes -> GetNEDofs();
   nfdofs[0] = pfes -> GetNFDofs();

   if (MyRank == 0)
   {
      pfes -> Save(out);
      out << '\n';

      for (p = 1; p < NRanks; p++)
      {
         MPI_Recv(&nv[p], 1, MPI_INT, p, 455, MyComm, &status);
         MPI_Recv(&nvdofs[p], 1, MPI_INT, p, 456, MyComm, &status);
         MPI_Recv(&nedofs[p], 1, MPI_INT, p, 457, MyComm, &status);
         MPI_Recv(&nfdofs[p], 1, MPI_INT, p, 458, MyComm, &status);
         values[p] = new double[nv[p]];
         MPI_Recv(values[p], nv[p], MPI_DOUBLE, p, 460, MyComm, &status);
      }

      int vdim = pfes -> GetVDim();

      for (p = 0; p < NRanks; p++)
         nrdofs[p] = nv[p]/vdim - nvdofs[p] - nedofs[p] - nfdofs[p];

      if (pfes->GetOrdering() == Ordering::byNODES)
      {
         for (int d = 0; d < vdim; d++)
         {
            for (p = 0; p < NRanks; p++)
               for (i = 0; i < nvdofs[p]; i++)
                  out << *values[p]++ << '\n';

            for (p = 0; p < NRanks; p++)
               for (i = 0; i < nedofs[p]; i++)
                  out << *values[p]++ << '\n';

            for (p = 0; p < NRanks; p++)
               for (i = 0; i < nfdofs[p]; i++)
                  out << *values[p]++ << '\n';

            for (p = 0; p < NRanks; p++)
               for (i = 0; i < nrdofs[p]; i++)
                  out << *values[p]++ << '\n';
         }
      }
      else
      {
         for (p = 0; p < NRanks; p++)
            for (i = 0; i < nvdofs[p]; i++)
               for (int d = 0; d < vdim; d++)
                  out << *values[p]++ << '\n';

         for (p = 0; p < NRanks; p++)
            for (i = 0; i < nedofs[p]; i++)
               for (int d = 0; d < vdim; d++)
                  out << *values[p]++ << '\n';

         for (p = 0; p < NRanks; p++)
            for (i = 0; i < nfdofs[p]; i++)
               for (int d = 0; d < vdim; d++)
                  out << *values[p]++ << '\n';

         for (p = 0; p < NRanks; p++)
            for (i = 0; i < nrdofs[p]; i++)
               for (int d = 0; d < vdim; d++)
                  out << *values[p]++ << '\n';
      }

      for (p = 1; p < NRanks; p++)
      {
         values[p] -= nv[p];
         delete [] values[p];
      }
   }
   else
   {
      MPI_Send(&nv[0], 1, MPI_INT, 0, 455, MyComm);
      MPI_Send(&nvdofs[0], 1, MPI_INT, 0, 456, MyComm);
      MPI_Send(&nedofs[0], 1, MPI_INT, 0, 457, MyComm);
      MPI_Send(&nfdofs[0], 1, MPI_INT, 0, 458, MyComm);
      MPI_Send(data, nv[0], MPI_DOUBLE, 0, 460, MyComm);
   }

   delete [] values;
   delete [] nv;
   delete [] nvdofs;
   delete [] nedofs;
   delete [] nfdofs;
   delete [] nrdofs;
}

double GlobalLpNorm(const double p, double loc_norm, MPI_Comm comm)
{
   double glob_norm;

   if (p < numeric_limits<double>::infinity())
   {
      // negative quadrature weights may cause the error to be negative
      if (loc_norm < 0.0)
         loc_norm = -pow(-loc_norm, p);
      else
         loc_norm = pow(loc_norm, p);

      MPI_Allreduce(&loc_norm, &glob_norm, 1, MPI_DOUBLE, MPI_SUM, comm);

      if (glob_norm < 0.0)
         glob_norm = -pow(-glob_norm, 1.0/p);
      else
         glob_norm = pow(glob_norm, 1.0/p);
   }
   else
   {
      MPI_Allreduce(&loc_norm, &glob_norm, 1, MPI_DOUBLE, MPI_MAX, comm);
   }

   return glob_norm;
}

}

#endif
