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

#ifdef MFEM_USE_MPI

#include "fem.hpp"

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

ParGridFunction::ParGridFunction(ParMesh *pmesh, GridFunction *gf)
{
   // duplicate the FiniteElementCollection from 'gf'
   fec = FiniteElementCollection::New(gf->FESpace()->FEColl()->Name());
   fes = pfes = new ParFiniteElementSpace(pmesh, fec, gf->FESpace()->GetVDim(),
                                          gf->FESpace()->GetOrdering());
   SetSize(pfes->GetVSize());
}

void ParGridFunction::Update(ParFiniteElementSpace *f)
{
   GridFunction::Update(f);
   pfes = f;
}

void ParGridFunction::Update(ParFiniteElementSpace *f, Vector &v, int v_offset)
{
   GridFunction::Update(f, v, v_offset);
   pfes = f;
}

void ParGridFunction::Distribute(HypreParVector *tv)
{
   pfes->Dof_TrueDof_Matrix()->Mult(*tv, *this);
}

void ParGridFunction::ParallelAverage(HypreParVector &tv)
{
   pfes->Dof_TrueDof_Matrix()->MultTranspose(*this, tv);
   pfes->DivideByGroupSize(tv);
}

HypreParVector *ParGridFunction::ParallelAverage()
{
   HypreParVector *tv = new HypreParVector(pfes->GlobalTrueVSize(),
                                           pfes->GetTrueDofOffsets());
   ParallelAverage(*tv);
   return tv;
}

double ParGridFunction::ComputeL1Error(Coefficient *exsol[],
                                       const IntegrationRule *irs[]) const
{
   double lerr, gerr;

   lerr = GridFunction::ComputeW11Error(*exsol, NULL, 1, NULL, irs);

   MPI_Allreduce(&lerr, &gerr, 1, MPI_DOUBLE, MPI_SUM, pfes->GetComm());

   return gerr;
}

double ParGridFunction::ComputeL1Error(VectorCoefficient &exsol,
                                       const IntegrationRule *irs[]) const
{
   double lerr, gerr;

   lerr = GridFunction::ComputeL1Error(exsol, irs);

   MPI_Allreduce(&lerr, &gerr, 1, MPI_DOUBLE, MPI_SUM, pfes->GetComm());

   return gerr;
}

double ParGridFunction::ComputeL2Error(Coefficient *exsol[],
                                       const IntegrationRule *irs[]) const
{
   double lerr, gerr;

   lerr = GridFunction::ComputeL2Error(exsol, irs);
   lerr *= lerr;

   MPI_Allreduce(&lerr, &gerr, 1, MPI_DOUBLE, MPI_SUM, pfes->GetComm());

   return sqrt(gerr);
}

double ParGridFunction::ComputeL2Error(VectorCoefficient &exsol,
                                       const IntegrationRule *irs[],
                                       Array<int> *elems) const
{
   double lerr, gerr;

   lerr = GridFunction::ComputeL2Error(exsol, irs, elems);
   lerr *= lerr;

   MPI_Allreduce(&lerr, &gerr, 1, MPI_DOUBLE, MPI_SUM, pfes->GetComm());

   return sqrt(gerr);
}

double ParGridFunction::ComputeMaxError(Coefficient *exsol[],
                                        const IntegrationRule *irs[]) const
{
   double lerr, gerr;

   lerr = GridFunction::ComputeMaxError(exsol, irs);

   MPI_Allreduce(&lerr, &gerr, 1, MPI_DOUBLE, MPI_MAX, pfes->GetComm());

   return gerr;
}

double ParGridFunction::ComputeMaxError(VectorCoefficient &exsol,
                                        const IntegrationRule *irs[]) const
{
   double lerr, gerr;

   lerr = GridFunction::ComputeMaxError(exsol, irs);

   MPI_Allreduce(&lerr, &gerr, 1, MPI_DOUBLE, MPI_MAX, pfes->GetComm());

   return gerr;
}

void ParGridFunction::Save(ostream &out)
{
   for (int i = 0; i < size; i++)
      if (pfes->GetDofSign(i) < 0)
         data[i] = -data[i];

   GridFunction::Save(out);

   for (int i = 0; i < size; i++)
      if (pfes->GetDofSign(i) < 0)
         data[i] = -data[i];
}

void ParGridFunction::SaveAsOne(ostream &out)
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

#endif
