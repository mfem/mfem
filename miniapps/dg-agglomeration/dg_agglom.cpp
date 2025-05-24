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
//                     -----------------------
//                     DG Agglomeration Solver
//                     -----------------------
//

#include "mfem.hpp"
#include <iostream>
#include <memory>

#include "partition.hpp"

using namespace std;
using namespace mfem;

int main(int argc, char *argv[])
{
   //Mpi::Init(argc, argv);
   //Hypre::Init();

   const char *mesh_file = "../../data/star.mesh";
   int ncoarse = 3;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh", "Mesh file to use.");
   args.AddOption(&ncoarse, "-n", "--n-coarse", "Number of coarse elements.");
   args.ParseCheck();

   Mesh mesh(mesh_file);
   //const Mesh mesh_copy(mesh);
   mesh.UniformRefinement();
   //mesh.UniformRefinement();
   DG_FECollection fec(0, mesh.Dimension());
   FiniteElementSpace fes(&mesh, &fec);
   GridFunction p(&fes);
   const int ne = mesh.GetNE();
   const int num_partitions = std::ceil(std::log(ne)/std::log(ncoarse));

   ParaViewDataCollection pv("Agglomeration", &mesh);
   pv.SetPrefixPath("ParaView");
   pv.RegisterField("p", &p);

   std::vector<std::vector<int>> E(num_partitions); 

   p = 0;
   pv.SetCycle(0);
   pv.SetTime(0.0);
   pv.Save();

   Array<int> partitioning = PartitionMesh(mesh, ncoarse);
   for (int i = 0; i < p.Size(); ++i)
   {
      p[i] = partitioning[i];
   }
   for (int i = 0; i < ncoarse; ++i)
   {
      E[0].push_back(0);
   }
   std::vector<std::vector<int>> macro_elements(ncoarse);
   for (int k = 0; k < ne; ++k)
   {
      const int i = partitioning[k];
      macro_elements[i].push_back(k);
   }
   pv.SetCycle(1);
   pv.SetTime(1.0);
   pv.Save();
   for(int j = 1; j < num_partitions; ++j)
   {
      std::vector<std::vector<int>> macro_elements(E[j-1].size());
      for (int i = 0; i < p.Size(); ++i)
      {
         const int k = p[i];
         macro_elements[k].push_back(i);
      }
      int num_total_parts = 0;
      for(int e = 0; e < E[j-1].size(); ++e)
      { 
         const int num_el_part = macro_elements[e].size();
         Array<int> subset(num_el_part);
         for(int i=0; i<num_el_part; i++){subset[i] = macro_elements[e][i];}
         Array<int> partitioning = PartitionMesh(mesh, ncoarse, subset);
         int num_actual_parts = 0;
         for (int ip = 0; ip < partitioning.Size(); ++ip)
         {
            const int i = partitioning[ip];
            num_actual_parts = (i > num_actual_parts) ? i : num_actual_parts;
            p[subset[ip]] = i + num_total_parts;
         }
         for (int k = 0; k <= num_actual_parts; ++k){E[j].push_back(e);}
         num_total_parts = num_total_parts + num_actual_parts + 1;
      }
      pv.SetCycle(j+1);
      pv.SetTime(j+1);
      pv.Save();
   }


   // // **************TESTS***********
   // //Test to make sure final mesh is fully refined 
   // Array<int> p_arr(p.Size()); 
   // for (int ip = 0; ip < p.Size(); ip++){
   //    p_arr[ip] = p[ip];
   // }
   // p_arr.Sort();
   // bool no_duplicates = true;
   // for (int ip = 0; ip < p_arr.Size(); ip++){
   //    if (p_arr[ip] != ip){
   //       no_duplicates = false;
   //       break;
   //    }
   // }
   // if (no_duplicates){std::cout << "YES - Refined mesh IS fully refined" << std::endl;}
   // else{std::cout << "NO - Refined mesh IS NOT fully refined" << std::endl;}

   // //Test to make sure that E's have correct sizes 
   // bool correct_size = true;
   // for (int ie = 1; ie < num_partitions; ie++){
   //    int max_value = *std::max_element(E[ie].begin(), E[ie].end());
   //    if(max_value+1 != E[ie-1].size()){std::cout << "ie with wrong size: " << ie << std::endl; correct_size = false; break;}
   //    if (ie == num_partitions - 1)
   //    {
   //       if(E[ie].size() == ne){std::cout << "CORRECT FINAL SIZE FOR E" << std::endl;}
   //       else{std::cout << "WRONG FINAL SIZE FOR E" << std::endl;}
   //    }
   // }
   // if (correct_size){std::cout << "YES - E's have correct sizes" << std::endl;}
   // else{std::cout << "NO - an E has an incorrect size" << std::endl;}
   return 0;
}
