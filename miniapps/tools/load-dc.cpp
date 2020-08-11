// Copyright (c) 2010-2020, Lawrence Livermore National Security, LLC. Produced
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
//    Load DC Miniapp:  Visualize fields saved via DataCollection classes
//    -------------------------------------------------------------------
//
// This miniapp loads and visualizes (in GLVis) previously saved data using
// DataCollection sub-classes, see e.g. Example 5/5p. Currently, only the
// VisItDataCollection class is supported.
//
// Compile with: make load-dc
//
// Serial sample runs:
//   > load-dc -r ../../examples/Example5
//
// Parallel sample runs:
//   > mpirun -np 4 load-dc -r ../../examples/Example5-Parallel

#include "mfem.hpp"

using namespace std;
using namespace mfem;

int main(int argc, char *argv[])
{
#ifdef MFEM_USE_MPI
   MPI_Session mpi;
   if (!mpi.Root()) { mfem::out.Disable(); mfem::err.Disable(); }
#endif

   // Parse command-line options.
   const char *coll_name = NULL;
   int cycle = 0;
   bool visualization = true;

   OptionsParser args(argc, argv);
   args.AddOption(&coll_name, "-r", "--root-file",
                  "Set the VisIt data collection root file prefix.", true);
   args.AddOption(&cycle, "-c", "--cycle", "Set the cycle index to read.");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(mfem::out);
      return 1;
   }
   args.PrintOptions(mfem::out);

#ifdef MFEM_USE_MPI
   VisItDataCollection dc(MPI_COMM_WORLD, coll_name);
#else
   VisItDataCollection dc(coll_name);
#endif
   dc.Load(cycle);

   if (dc.Error() != DataCollection::NO_ERROR)
   {
      mfem::out << "Error loading VisIt data collection: " << coll_name << endl;
      return 1;
   }

   typedef DataCollection::FieldMapType fields_t;
   const fields_t &fields = dc.GetFieldMap();
   // Print the names of all fields.
   mfem::out << "fields: [ ";
   for (fields_t::const_iterator it = fields.begin(); it != fields.end(); ++it)
   {
      if (it != fields.begin()) { mfem::out << ", "; }
      mfem::out << it->first;
   }
   mfem::out << " ]" << endl;

   if (!visualization) { return 0; }

   char vishost[] = "localhost";
   int  visport   = 19916;

   // Visualize all fields. If there are no fields, visualize the mesh.
   for (fields_t::const_iterator it = fields.begin();
        it != fields.end() || fields.begin() == fields.end(); ++it)
   {
      socketstream sol_sock(vishost, visport);
      bool succeeded = sol_sock.good();
#ifdef MFEM_USE_MPI
      bool all_succeeded;
      MPI_Allreduce(&succeeded, &all_succeeded, 1,
                    MPI_C_BOOL, MPI_LAND, MPI_COMM_WORLD);
      succeeded = all_succeeded;
#endif
      if (!succeeded)
      {
         mfem::out << "Connection to " << vishost << ':' << visport
                   << " failed." << endl;
         return 1;
      }
#ifdef MFEM_USE_MPI
      sol_sock << "parallel " << mpi.WorldSize() << " " << mpi.WorldRank()
               << "\n";
#endif
      if (fields.begin() == fields.end())
      {
         // no fields, just mesh:
         sol_sock << "mesh\n" << *dc.GetMesh() << flush;
         break;
      }
      sol_sock.precision(8);
      sol_sock << "solution\n" << *dc.GetMesh() << *it->second
               << "window_title '" << it->first << "'\n" << flush;
   }

   return 0;
}
