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
// Compile with: make subdomainmap
//
// Sample runs:
//   subdomainmap -m1 global.mesh -m2 local.mesh

#include "mfem.hpp"
#include <fstream>

using namespace mfem;
using namespace std;

double funccoeff(const Vector & x);


int main (int argc, char *argv[])
{
   // Set the method's default parameters.
   // const char *src_mesh_file = "torus.mesh";
   // const char *src_mesh_file = "torus_ovlp_partition/ovlp_torus.mesh.000000";
   const char *mesh_file1 = "torus_ovlp_partition/ovlp_torus1_5.mesh";
   const char *mesh_file2 = "torus_ovlp_partition/ovlp_torus5_9.mesh";
   int order          = 1; // unused

   // Parse command-line options.
   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file1, "-m1", "--mesh1",
                  "Mesh file for the starting solution.");
   args.AddOption(&mesh_file2, "-m2", "--mesh2",
                  "Mesh file for interpolation.");
   args.AddOption(&order, "-o", "--order",
                  "Order of the interpolated solution.");
   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(cout);
      return 1;
   }
   args.PrintOptions(cout);

   // Input meshes.
   Array< Mesh * > meshes(2);
   meshes[0] = new Mesh(mesh_file1, 1, 1, false);
   meshes[1] = new Mesh(mesh_file2, 1, 1, false);

   int ref = 2;
   for (int i = 0; i<ref; i++)
   {
      meshes[0]->UniformRefinement();
      meshes[1]->UniformRefinement();
   }
   // meshes[0]->UniformRefinement();
   // meshes[1]->UniformRefinement();
   const int dim = meshes[0]->Dimension();
   MFEM_ASSERT(dim == meshes[1]->Dimension(), "Source and target meshes "
               "must be in the same dimension.");
   MFEM_VERIFY(dim > 1, "GSLIB requires a 2D or a 3D mesh" );

   if (meshes[0]->GetNodes() == NULL) { meshes[0]->SetCurvature(1); }
   if (meshes[1]->GetNodes() == NULL) { meshes[1]->SetCurvature(1); }
   const int mesh_poly_deg1 = meshes[0]->GetNodes()->FESpace()->GetOrder(0);
   const int mesh_poly_deg2 = meshes[1]->GetNodes()->FESpace()->GetOrder(0);
   cout << "mesh 1 curvature: "
        << meshes[0]->GetNodes()->OwnFEC()->Name() << endl
        << "mesh 2 curvature: "
        << meshes[1]->GetNodes()->OwnFEC()->Name() << endl;


   MFEM_VERIFY(meshes[0]->GetNumGeometries(meshes[1]->Dimension()) == 1, "Mixed meshes"
               "are not currently supported.");

   // Ensure the source grid function can be transferred using GSLIB-FindPoints.
   const int ne1 = meshes[0]->GetNE();
   const int ne2 = meshes[1]->GetNE();

   Vector centers(ne1*dim);
   for (int i = 0; i < ne1; i++)
   {
      Vector center(dim);
      meshes[0]->GetElementCenter(i,center);
      for (int d=0; d<dim; d++)
      {
         centers[ne1*d + i] = center[d];
      }
   }

   // Evaluate mesh 1 grid function.
   FindPointsGSLIB finder;
   finder.Setup(*meshes[1]);
   finder.FindPoints(centers);

   Array<unsigned int> elem_map = finder.GetElem();
   Array<unsigned int> code = finder.GetCode();

   cout << "mesh1 elems   = " << ne1 << endl;
   cout << "mesh2 elems   = " << ne2 << endl;
   cout << "elem map size = " << elem_map.Size() << endl;
   // cout << "elem map = " << endl;
   // elem_map.Print(cout,10);

   // cout << "code = " << endl;
   // code.Print(cout,10);
   // Free the internal gslib data.
   finder.FreeData();

   // testing dof maps 
   // H1 test;
   FiniteElementCollection * H1fec = new H1_FECollection(order,dim);
   FiniteElementSpace * fes1 = new FiniteElementSpace(meshes[0],H1fec);
   FiniteElementSpace * fes2 = new FiniteElementSpace(meshes[1],H1fec);

   // dof map from fes_1 to fes_2
   Array<int> dof_map(fes1->GetTrueVSize());
   dof_map = -1;
   for (int iel1 = 0; iel1<ne1; iel1++)
   {
      // skip the elements that are not found 
      if (code[iel1] == 2) continue;
      int iel2 = elem_map[iel1];
      Array<int> ElemDofs1;
      Array<int> ElemDofs2;
      fes1->GetElementDofs(iel1,ElemDofs1);
      fes2->GetElementDofs(iel2,ElemDofs2);
         // the sizes have to match
      MFEM_VERIFY(ElemDofs1.Size() == ElemDofs2.Size(),
                  "Size inconsistency");
      // loop through the dofs and take into account the signs;
      int ndof = ElemDofs1.Size();
      for (int i = 0; i<ndof; ++i)
      {
         int pdof_ = ElemDofs1[i];
         int gdof_ = ElemDofs2[i];
         int pdof = (pdof_ >= 0) ? pdof_ : abs(pdof_) - 1;
         int gdof = (gdof_ >= 0) ? gdof_ : abs(gdof_) - 1;
         dof_map[pdof] = gdof;
      }
   }

   Array<GridFunction *> gfs(2);

   gfs[0] = new GridFunction(fes1);
   *gfs[0] = 0.0;
   FunctionCoefficient cf1(funccoeff);
   // ConstantCoefficient one(1.0);
   gfs[0]->ProjectCoefficient(cf1);

   gfs[1] = new GridFunction(fes2);
   *gfs[1] = 0.0;
   for (int i = 0; i< dof_map.Size(); i++)
   {
      int j = dof_map[i];   
      if (j < 0) continue;
      (*gfs[1])[j] = (*gfs[0])[i];
   }

   

   // char vishost[] = "localhost";
   // int  visport   = 19916;
   // string keys;
   // if (dim ==2 )
   // {
   //    keys = "keys mrRljc\n";
   // }
   // else
   // {
   //    keys = "keys mc\n";
   // }
   // socketstream sol_sock1(vishost, visport);
   // sol_sock1.precision(8);
   // sol_sock1 << "solution\n" << mesh_1 << gf1 << keys 
   //           << "window_title ' ' " << flush;                     

   // socketstream sol_sock2(vishost, visport);
   // sol_sock2.precision(8);
   // sol_sock2 << "solution\n" << mesh_2 << gf2 << keys 
   //           << "window_title ' ' " << flush;  
      
   for (int ip = 0; ip<meshes.Size(); ++ip)
   {
      // char vishost[] = "localhost";
      // int  visport   = 19916;
      // socketstream sol_sock(vishost, visport);
      // sol_sock << "parallel " << meshes.Size() << " " << ip << "\n";
      // sol_sock.precision(8);
      // sol_sock << "mesh\n" << *meshes[ip] << flush;
      ostringstream mesh_name;
      mesh_name << "output/mesh." << setfill('0') << setw(6) << ip;
      ofstream mesh_ofs(mesh_name.str().c_str());
      mesh_ofs.precision(8);
      meshes[ip]->Print(mesh_ofs);
      ostringstream gf_name;
      gf_name << "output/gf." << setfill('0') << setw(6) << ip;
      ofstream gf_ofs(gf_name.str().c_str());
      gf_ofs.precision(8);
      gfs[ip]->Save(gf_ofs);
   }
   
   
   
   // {
   //    socketstream sol_sock(vishost, visport);
   //    sol_sock << "parallel " << 2 << " " << 0 << "\n";
   //    sol_sock.precision(8);
   //    sol_sock << "solution\n" << mesh_1 << gf1 
   //             << keys << flush;  
   // }
   // {
   //    socketstream sol_sock(vishost, visport);
   //    sol_sock << "parallel " << 2 << " " << 1 << "\n";
   //    sol_sock.precision(8);
   //    sol_sock << "solution\n" << mesh_2 << gf2 
   //             << keys << flush;  
   // }



   return 0;
}


double funccoeff(const Vector & x)
{
   return sin(3*M_PI*(x.Sum()));
}