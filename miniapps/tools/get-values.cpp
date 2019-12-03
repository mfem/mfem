// Copyright (c) 2010, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-443211. All Rights
// reserved. See file COPYRIGHT for details.
//
// This file is part of the MFEM library. For more information and source code
// availability see http://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the GNU Lesser General Public License (as published by the Free
// Software Foundation) version 2.1 dated February 1999.
//
//    --------------------------------------------------------------------
//    Get Values Miniapp:  Extract field values via DataCollection classes
//    --------------------------------------------------------------------
//
// This miniapp loads and visualizes (in GLVis) previously saved data using
// DataCollection sub-classes, see e.g. Example 5/5p. Currently, only the
// VisItDataCollection class is supported.
//
// Compile with: make get-values
//
// Serial sample runs:
//   > load-dc -r ../../examples/Example5
//
// Parallel sample runs:
//   > mpirun -np 4 load-dc -r ../../examples/Example5-Parallel

#include "mfem.hpp"

#include <set>
#include <string>

using namespace std;
using namespace mfem;

bool isScalarField(GridFunction & gf);

int main(int argc, char *argv[])
{
#ifdef MFEM_USE_MPI
   MPI_Session mpi;
   if (!mpi.Root()) { mfem::out.Disable(); mfem::err.Disable(); }
#endif

   // Parse command-line options.
   const char *coll_name = NULL;
   int cycle = 0;

   const char *field_name_c_str = "ALL";
   Vector pts;

   OptionsParser args(argc, argv);
   args.AddOption(&coll_name, "-r", "--root-file",
                  "Set the VisIt data collection root file prefix.", true);
   args.AddOption(&cycle, "-c", "--cycle", "Set the cycle index to read.");
   args.AddOption(&pts, "-p", "--points", "List of points.");
   args.AddOption(&field_name_c_str, "-fn", "--field-names",
                  "List of field names to get values from.");
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

   mfem::out << endl;
   mfem::out << "Collection Name: " << dc.GetCollectionName() << endl;
   mfem::out << "Cycle:           " << dc.GetCycle() << endl;
   mfem::out << "Time:            " << dc.GetTime() << endl;
   mfem::out << "Time Step:       " << dc.GetTimeStep() << endl;
   mfem::out << endl;

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

   // Parsing desired field names
   set<string> field_names;
   {
      string field_name_str(field_name_c_str);
      string field_name;

      for (string::iterator it=field_name_str.begin();
           it!=field_name_str.end(); it++)
      {
         if (*it == '\\')
         {
            it++;
            field_name.push_back(*it);
         }
         else if (*it == ' ')
         {
            if (!field_name.empty())
            {
               field_names.insert(field_name);
            }
            field_name.clear();
         }
         else if (it == field_name_str.end() - 1)
         {
            field_name.push_back(*it);
            field_names.insert(field_name);
         }
         else
         {
            field_name.push_back(*it);
         }
      }
      if (field_names.size() == 0)
      {
         field_names.insert("ALL");
      }
   }

   // Print field names to be extracted
   mfem::out << "Extracting fields: ";
   for (set<string>::iterator it=field_names.begin();
        it!=field_names.end(); it++)
   {
      mfem::out << " \"" << *it << "\"";
   }
   mfem::out << endl;

   int spaceDim = dc.GetMesh()->SpaceDimension();
   int npts = pts.Size() / spaceDim;

   DenseMatrix pt_mat(pts.GetData(), spaceDim, npts);

   Array<int> elem_ids;
   Array<IntegrationPoint> ip;
   int nfound = dc.GetMesh()->FindPoints(pt_mat, elem_ids, ip);
   mfem::out << "Found " << nfound << " points." << endl;

   for (int e=0; e<elem_ids.Size(); e++)
   {
      if (elem_ids[e] >= 0)
      {
         mfem::out << e;
         for (int d=0; d<spaceDim; d++)
         {
            mfem::out << ' ' << pt_mat(d, e);
         }

         // Loop over all fields.
         for (fields_t::const_iterator it = fields.begin();
              it != fields.end(); ++it)
         {
            if (field_names.find("ALL") != field_names.end() ||
                field_names.find(it->first) != field_names.end())
            {
               if (isScalarField(*it->second))
               {
                  mfem::out << ' ' << it->second->GetValue(elem_ids[e], ip[e]);
               }
               else
               {
                  Vector val;
                  it->second->GetVectorValue(elem_ids[e], ip[e], val);
                  for (int d=0; d<spaceDim; d++)
                  {
                     mfem::out << ' ' << val[d];
                  }
               }
            }
         }
         mfem::out << endl;
      }
   }

   return 0;
}

bool isScalarField(GridFunction & gf)
{
   return (gf.VectorDim() == 1);
}
