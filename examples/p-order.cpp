// Copyright (c) 2010-2023, Lawrence Livermore National Security, LLC. Produced
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
//                      -------------------------------
//                      Convergence Rates Test (Serial)
//                      -------------------------------
//
// Compile with: make p-order
//
// Sample runs:  p-order -m ../data/inline-quad.mesh -rs 2


#include "../mfem.hpp"
#include <fstream>
#include <iostream>
#include <set>
using namespace std;
using namespace mfem;

int dim;

int main(int argc, char *argv[])
{
   // 1. Parse command-line options.
   const char *mesh_file = "../data/inline-quad.mesh";
   bool visualization = true;
   int rs = 2;
   double prob = 0.5;
   int max_order = 5;
   int init_order = 3;
   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&rs, "-rs", "--serial_ref",
                  "Number of serial refinements.");
   args.AddOption(&prob, "-prob", "--prob",
                  "probablity of refinement.");
   args.AddOption(&init_order, "-io", "--init-order",
                  "initial order.");
   args.AddOption(&max_order, "-mo", "--max-order",
                  "max order for random refinement.");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(cout);
      return 1;
   }
   args.PrintOptions(cout);

   Mesh mesh(mesh_file, 1, 1);
   dim = mesh.Dimension();

   for (int i = 0; i < rs; i++)
   {
      mesh.UniformRefinement();
   }
   int nelem = mesh.GetNE();

   L2_FECollection l2fec(0, dim);
   FiniteElementSpace l2fes(&mesh, &l2fec);
   GridFunction ordergf(&l2fes);

   std::set<int> int_el_set;

   ordergf = init_order*1.0;
   Array<int> els_to_refine(0);
   for (int e = 0; e < nelem; e++)
   {
      if ((double) rand() / RAND_MAX < prob)
      {
         els_to_refine.Append(e);
         ordergf(e) = (rand() % (max_order-1) + 2)*1.0;
         int_el_set.insert(e);
      }
   }

   std::cout << "Number of elements: " << nelem << std::endl;
   els_to_refine.Print();

   // make table of element and its neighbors
   const Table &eln = mesh.ElementToElementTable();
   //   MFEM_ABORT(" ");

   if (visualization)
   {
      osockstream sock(19916, "localhost");
      sock << "solution\n";
      mesh.Print(sock);
      ordergf.Save(sock);
      sock.send();
      sock << "window_title 'Order'\n"
           << "window_geometry "
           << 0 << " " << 0 << " " << 600 << " " << 600 << "\n"
           << "keys jRmcA" << endl;
   }

   Array<bool> order_propogated(nelem);
   Array<int> propogate_list(0);
   order_propogated = false;
   propogate_list.Append(els_to_refine);
   Array<int> propogate_list_new = propogate_list;
   Array<int> propogate_list_full = propogate_list;

   bool donee = false;
   int iter = 0;
   while (!donee)
   {
      //       std::cout << iter <<  "k10here\n";
      Array<int> new_orders(nelem);
      for (int e = 0; e < nelem; e++)
      {
         new_orders[e] = ordergf(e);
      }
      propogate_list = propogate_list_new;
      propogate_list_new.SetSize(0);
      //       std::cout << " Propogate list" << std::endl;
      //       propogate_list.Print();

      for (int i = 0; i < propogate_list.Size(); i++)
      {
         propogate_list_full.Append(propogate_list[i]);
         order_propogated[propogate_list[i]] = true;
      }


      for (int i = 0; i < propogate_list.Size(); i++)
      {
         int elem = propogate_list[i];
         Array<int> elem_neighbor_indices;
         eln.GetRow(elem, elem_neighbor_indices);
         int elem_order = new_orders[elem];
         for (int n = 0; n < elem_neighbor_indices.Size(); n++)
         {
            int elem_neighbor_index = elem_neighbor_indices[n];
            if (order_propogated[elem_neighbor_index]) { continue; }
            int current_order = new_orders[elem_neighbor_index];
            if (current_order < elem_order-1)
            {
               propogate_list_new.Append(elem_neighbor_index);
               new_orders[elem_neighbor_index] = elem_order-1;
            }
         }
      }

      if (propogate_list_new.Size() > 0)
      {
         propogate_list_new.Sort();
         propogate_list_new.Unique();
         propogate_list_full.Append(propogate_list_new);
         //           std::cout << " Propogate list new" << std::endl;
         //           propogate_list_new.Print();
      }
      else
      {
         donee = true;
      }

      for (int e = 0; e < nelem; e++)
      {
         ordergf[e] = new_orders[e];
      }
      //       std::cout << iter << " " << donee <<  " k10iter\n";
      //       if (iter == 2) { donee = true; }
      iter++;
   }

   if (visualization)
   {
      osockstream sock(19916, "localhost");
      sock << "solution\n";
      mesh.Print(sock);
      ordergf.Save(sock);
      sock.send();
      sock << "window_title 'Order'\n"
           << "window_geometry "
           << 600 << " " << 0 << " " << 600 << " " << 600 << "\n"
           << "keys jRmcA" << endl;
   }




   return 0;
}
