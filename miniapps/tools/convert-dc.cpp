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
//      ---------------------------------------------------------------
//      Convert DC: Convert between different types of data collections
//      ---------------------------------------------------------------
//
// This tool demonstrates how to convert between MFEM's different concrete
// DataCollection options.
//
// Currently supported data collection type options:
//    visit:                VisItDataCollection (default)
//    sidre or sidre_hdf5:  SidreDataCollection
//    json:                 ConduitDataCollection w/ protocol json
//    conduit_json:         ConduitDataCollection w/ protocol conduit_json
//    conduit_bin:          ConduitDataCollection w/ protocol conduit_bin
//    hdf5:                 ConduitDataCollection w/ protocol hdf5
//
// Compile with: make convert-dc
//
// Serial sample runs:
//    convert-dc -s ../../examples/Example5 -st visit -o Example5_Conduit -ot json
//
// Parallel sample runs:
//    mpirun -np 4 convert-dc -s ../../examples/Example5-Parallel -st visit
//                            -o Example5-Parallel_Conduit -ot json

#include "mfem.hpp"

using namespace std;
using namespace mfem;

DataCollection *create_data_collection(const std::string &dc_name,
                                       const std::string &dc_type)
{
   DataCollection *dc = NULL;

   if (dc_type == "visit")
   {
#ifdef MFEM_USE_MPI
      dc = new VisItDataCollection(MPI_COMM_WORLD, dc_name);
#else
      dc = new VisItDataCollection(dc_name);
#endif
   }
   else if ( dc_type == "sidre" || dc_type == "sidre_hdf5")
   {
#ifdef MFEM_USE_SIDRE
      dc = new SidreDataCollection(dc_name);
#else
      MFEM_ABORT("Must build with MFEM_USE_SIDRE=YES for sidre support.");
#endif
   }
   else if ( dc_type == "json" ||
             dc_type == "conduit_json" ||
             dc_type == "conduit_bin"  ||
             dc_type == "hdf5")
   {
#ifdef MFEM_USE_CONDUIT
#ifdef MFEM_USE_MPI
      ConduitDataCollection *conduit_dc = new ConduitDataCollection(MPI_COMM_WORLD,
                                                                    dc_name);
#else
      ConduitDataCollection *conduit_dc = new ConduitDataCollection(dc_name);
#endif
      conduit_dc->SetProtocol(dc_type);
      dc = conduit_dc;
#else
      MFEM_ABORT("Must build with MFEM_USE_CONDUIT=YES for conduit support.");
#endif
   }
   else
   {
      MFEM_ABORT("Unsupported Data Collection type:" << dc_type);
   }

   return dc;
}

int main(int argc, char *argv[])
{
#ifdef MFEM_USE_MPI
   MPI_Session mpi;
   if (!mpi.Root()) { mfem::out.Disable(); mfem::err.Disable(); }
#endif

   // Parse command-line options.
   const char *src_coll_name = NULL;
   const char *src_coll_type = "visit";
   int src_cycle = 0;
   const char *out_coll_name = NULL;
   const char *out_coll_type = "visit";

   OptionsParser args(argc, argv);
   args.AddOption(&src_coll_name, "-s", "--source-root-prefix",
                  "Set the source data collection root file prefix.", true);
   args.AddOption(&out_coll_name, "-o", "--output-root-prefix",
                  "Set the source data collection root file prefix.", true);
   args.AddOption(&src_cycle, "-c", "--cycle",
                  "Set the source cycle index to read.");
   args.AddOption(&src_coll_type, "-st", "--source-type",
                  "Set the source data collection type. Options:\n"
                  "\t   visit:                VisItDataCollection (default)\n"
                  "\t   sidre or sidre_hdf5:  SidreDataCollection\n"
                  "\t   json:                 ConduitDataCollection w/ protocol json\n"
                  "\t   conduit_json:         ConduitDataCollection w/ protocol conduit_json\n"
                  "\t   conduit_bin:          ConduitDataCollection w/ protocol conduit_bin\n"
                  "\t   hdf5:                 ConduitDataCollection w/ protocol hdf5");
   args.AddOption(&out_coll_type, "-ot", "--output-type",
                  "Set the output data collection type. Options:\n"
                  "\t   visit:                VisItDataCollection (default)\n"
                  "\t   sidre or sidre_hdf5:  SidreDataCollection\n"
                  "\t   json:                 ConduitDataCollection w/ protocol json\n"
                  "\t   conduit_json:         ConduitDataCollection w/ protocol conduit_json\n"
                  "\t   conduit_bin:          ConduitDataCollection w/ protocol conduit_bin\n"
                  "\t   hdf5:                 ConduitDataCollection w/ protocol hdf5");
   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(mfem::out);
      return 1;
   }
   args.PrintOptions(mfem::out);

   DataCollection *src = create_data_collection(std::string(src_coll_name),
                                                std::string(src_coll_type));

   DataCollection *out = create_data_collection(std::string(out_coll_name),
                                                std::string(out_coll_type));

   src->Load(src_cycle);

   if (src->Error() != DataCollection::NO_ERROR)
   {
      mfem::out << "Error loading data collection: "
                << src_coll_name
                << " (type = "
                << src_coll_type
                << ")"
                << endl;
      return 1;
   }

   out->SetOwnData(false);

   // add mesh from source dc to output dc
#ifdef MFEM_USE_MPI
   out->SetMesh(MPI_COMM_WORLD,src->GetMesh());
#else
   out->SetMesh(src->GetMesh());
#endif

   // propagate the basics
   out->SetCycle(src->GetCycle());
   out->SetTime(src->GetTime());
   out->SetTimeStep(src->GetTimeStep());

   // loop over all fields in the source dc, and add them to the output dc
   const DataCollection::FieldMapType &src_fields = src->GetFieldMap();

   for (DataCollection::FieldMapType::const_iterator it = src_fields.begin();
        it != src_fields.end();
        ++it)
   {
      out->RegisterField(it->first,it->second);
   }

   out->Save();

   if (out->Error() != DataCollection::NO_ERROR)
   {
      mfem::out << "Error saving data collection: "
                << out_coll_name
                << " (type = "
                << out_coll_type
                << ")"
                << endl;
      return 1;
   }

   // cleanup
   delete src;
   delete out;

   return 0;
}
