// Copyright (c) 2010-2025, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.


#include "mfem.hpp"
#include "general/forall.hpp"
#include "../common/mfem-common.hpp"

namespace gslib
{
   #include "gslib.h"
}

using namespace mfem;
using namespace std;

template<int T_FIELDS=0>
void transferParticleData(Vector &pdata,
                          Array<int> &indices,
                          const Array<unsigned int> &proc,
                          int nfields=0)
{

   struct gslib::comm *gsl_comm = new gslib::comm;
   struct gslib::crystal *cr      = new gslib::crystal;
   comm_init(gsl_comm, MPI_COMM_WORLD);
   crystal_init(cr, gsl_comm);

   const int MF = T_FIELDS ? T_FIELDS : 100;
   const int NF = T_FIELDS ? T_FIELDS : nfields;
   std::cout << gsl_comm->id << " Transfering " <<
             NF << " fields of size " << MF
             << " for " << indices.Size() << " particles." << endl;
   MFEM_VERIFY(NF > 0, "Number of fields must be greater than 0");

   struct send_pt
   {
      unsigned int index, proc;
      double data[MF];
   };

   const int nsend = indices.Size();

   struct gslib::array send_pt_array;
   array_init(struct send_pt, &send_pt_array, indices.Size());
   send_pt_array.n = nsend;

   struct send_pt *pt;
   pt = (struct send_pt *)send_pt_array.ptr;
   for (int i = 0; i < nsend; i++)
   {
      pt->index = indices[i];
      pt->proc = proc[i];
      for (int j = 0; j < NF; j++)
      {
         pt->data[j] = pdata[i * NF + j];
      }
      ++pt;
   }
   pt = (struct send_pt *)send_pt_array.ptr;

   // sarray_transfer_ext(send_pt, &send_pt_array, proc.GetData(), 1, cr);
   sarray_transfer(send_pt, &send_pt_array, proc, 1, cr);

   int nrecv = send_pt_array.n;
   pdata.SetSize(nrecv * NF);
   indices.SetSize(nrecv);
   pdata = 0.0;
   std::cout << gsl_comm->id << " Received " << NF << " fields of size " << MF
             << " for " << nrecv << " particles." << endl;

   pt = (struct send_pt *)send_pt_array.ptr;
   for (int i = 0; i < nrecv; i++)
   {
      indices[i] = pt->index;
      for (int j = 0; j < NF; j++)
      {
         pdata[i * NF + j] = pt->data[j];
      }
      ++pt;
   }

   crystal_free(cr);
   comm_free(gsl_comm);
}

void transferDataWrapper(Vector &pdata,
                           Array<int> &indices, Array<unsigned int> &proc,
                           int nfields)
{
   switch (nfields)
   {
      case 2:
         return transferParticleData<2>(pdata, indices, proc);
      case 3:
         return transferParticleData<3>(pdata, indices, proc);
      case 4:
         return transferParticleData<4>(pdata, indices, proc);
      case 5:
         return transferParticleData<5>(pdata, indices, proc);
      case 6:
         return transferParticleData<6>(pdata, indices, proc);
      default:
         return transferParticleData(pdata, indices, proc, nfields);
   }
}

int main (int argc, char *argv[])
{
   // Initialize MPI and HYPRE.
   Mpi::Init(argc, argv);
   int num_procs = Mpi::WorldSize();
   int myid = Mpi::WorldRank();
   Hypre::Init();

   int npart = 100;
   int nfields = 4;
   OptionsParser args(argc, argv);
   args.AddOption(&npart, "-np", "--npart",
                  "# of particles");
   args.AddOption(&nfields, "-nf", "--nfields",
                  "Number of fields");
   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(cout);
      return 1;
   }
   if (myid == 0) { args.PrintOptions(cout); }

   Vector pdata(npart*nfields);
   pdata.Randomize();
   if (myid != 0)
   {
      pdata.SetSize(0);
      npart = 0;
   }

   Array<unsigned int> proc(myid == 0 ? npart : 0);
   Array<int> indices(myid == 0 ? npart : 0);
   for (int i = 0; i < npart; i++)
   {
      proc[i] = (int)(rand_real()*num_procs) % num_procs;
      indices[i] = myid + i*num_procs;
   }

   // Do the transfer
   transferDataWrapper(pdata, indices, proc, nfields);

   return 0;
}
