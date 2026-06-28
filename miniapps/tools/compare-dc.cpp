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
//
//    -------------------------------------------------------------------
//    Compare DC Miniapp:  Compare fields saved via DataCollection classes
//    -------------------------------------------------------------------
//
// This miniapp loads previously saved data and computes the l2 norm of the
// difference. Currently, only the VisItDataCollection class is supported.
//
// Compile with: make compare-dc
//
// Serial sample runs:
//   > compare-dc -r0 ../../examples/Example5 -r1 ../../examples/alt/Example5
//   > compare-dc -r0 Example5 -r1 alt/Example5 -tol 1e-6
//
// Parallel sample runs:
//   > mpirun -np 4 compare-dc -r0 ../../examples/Example5-Parallel
//                             -r1 ../../examples/alt/Example5-Parallel
//
//  NB: when no tolerance is provided the difference is simple reported.
//  If a tolerance is provided this is compared with the symmetric
//  relative difference. An error is given if difference exceeds the tolerance.

#include "mfem.hpp"

using namespace std;
using namespace mfem;

int main(int argc, char *argv[])
{
#ifdef MFEM_USE_MPI
   Mpi::Init();
   if (!Mpi::Root()) { mfem::out.Disable(); mfem::err.Disable(); }
   Hypre::Init();
#endif

   // Parse command-line options.
   const char *coll_name0 = NULL;
   const char *coll_name1 = NULL;
   int cycle = 0;
   int pad_digits_cycle = 6;
   int pad_digits_rank = 6;
   real_t tol = -1;

   OptionsParser args(argc, argv);
   args.AddOption(&coll_name0, "-r0", "--root-file_0",
                  "Set the VisIt data collection root file prefix.", true);
   args.AddOption(&coll_name1, "-r1", "--root-file_1",
                  "Set the VisIt data collection root file prefix.", true);
   args.AddOption(&cycle, "-c", "--cycle", "Set the cycle index to read.");
   args.AddOption(&pad_digits_cycle, "-pdc", "--pad-digits-cycle",
                  "Number of digits in cycle.");
   args.AddOption(&pad_digits_rank, "-pdr", "--pad-digits-rank",
                  "Number of digits in MPI rank.");
   args.AddOption(&tol, "-tol", "--tolerance",
                  "Tolerance for checking the results.");
   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(mfem::out);
      return 1;
   }
   args.PrintOptions(mfem::out);

#ifdef MFEM_USE_MPI
   VisItDataCollection dc0(MPI_COMM_WORLD, coll_name0);
#else
   VisItDataCollection dc0(coll_name0);
#endif
   dc0.SetPadDigitsCycle(pad_digits_cycle);
   dc0.SetPadDigitsRank(pad_digits_rank);
   dc0.Load(cycle);

   if (dc0.Error() != DataCollection::No_Error)
   {
      mfem::out << "Error loading VisIt data collection: " << coll_name0 << endl;
      return 1;
   }

#ifdef MFEM_USE_MPI
   VisItDataCollection dc1(MPI_COMM_WORLD, coll_name1);
#else
   VisItDataCollection dc1(coll_name1);
#endif
   dc1.SetPadDigitsCycle(pad_digits_cycle);
   dc1.SetPadDigitsRank(pad_digits_rank);
   dc1.Load(cycle);

   if (dc1.Error() != DataCollection::No_Error)
   {
      mfem::out << "Error loading VisIt data collection: " << coll_name1 << endl;
      return 1;
   }

   typedef DataCollection::FieldMapType fields_t;
   const fields_t &fields0 = dc0.GetFieldMap();
   // Print the names of all fields.
   bool error = false;
   for (fields_t::const_iterator it0 = fields0.begin();
        it0 != fields0.end() ; ++it0)
   {
      GridFunction *gf0 = dc0.GetField(it0->first);
      if (!gf0)
      {
         mfem::out << "Error loading:"<<it0->first<< endl;
         mfem::out << "From data collection: " << coll_name0 << endl;
         return 1;
      }

      GridFunction *gf1 = dc1.GetField(it0->first);
      if (!gf1)
      {
         mfem::out << "Error loading:"<<it0->first<< endl;
         mfem::out << "From data collection: " << coll_name1 << endl;
         return 1;
      }
      if (gf0->Size() != gf1->Size())
      {
         mfem::out << "Size error for:"<<it0->first<< endl;
         mfem::out << "In data collection: " << coll_name0
                   <<" size is "<<gf0->Size()<< endl;
         mfem::out << "In data collection: " << coll_name1
                   <<" size is "<<gf1->Size()<< endl;
         return 1;
      }

      // Norm of vectors
      real_t nrm0 = gf0->Norml2();
      real_t nrm1 = gf1->Norml2();

      // Difference
      (*gf0) -= (*gf1);
      real_t nrmd = gf0->Norml2();
      real_t rel_sym = 2*nrmd/(nrm0 + nrm1);
      if (gf0->Norml2() > rel_sym) { error = true; }

      // Report
      mfem::out <<"==========================================="<<std::endl;
      mfem::out <<"|"<<it0->first<<"_0|  = "<<nrm0<<std::endl;
      mfem::out <<"|"<<it0->first<<"_1|  = "<<nrm1<<std::endl;
      mfem::out <<"\n|"<<it0->first<<"_0 - "<<it0->first<<"_1| = "<<nrmd <<std::endl;

      mfem::out <<"\n2|"<<it0->first<<"_0 - "<<it0->first<<"_1|"<<std::endl;
      mfem::out << std::setfill('-') << std::setw(15 + 2*it0->first.length())
                <<" = "<<rel_sym<<std::endl;
      mfem::out <<"(|"<<it0->first<<"_0| + |"<<it0->first<<"_1|)\n"<<std::endl;

   }

   if (error && tol > 0.0)
   {
      mfem::out << "Data collections: " << coll_name0
                << " & " << coll_name1 << " are outside of the tolerance!\n";
      return -1;
   }
   return 0;
}
