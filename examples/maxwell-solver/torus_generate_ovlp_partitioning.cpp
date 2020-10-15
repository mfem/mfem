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
int get_angle_range(double angle, Array<double> angles);


int main (int argc, char *argv[])
{
   // Set the method's default parameters.
   const char *src_mesh_file = "torus.mesh";
   // const char *tar_mesh_file = "torus1_4.mesh";
   const char *tar_mesh_file = "torus2_4.mesh";
   int order          = 3; // unused

   // Parse command-line options.
   OptionsParser args(argc, argv);
   args.AddOption(&src_mesh_file, "-m1", "--mesh1",
                  "Mesh file for the starting solution.");
   args.AddOption(&tar_mesh_file, "-m2", "--mesh2",
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
   Mesh mesh_1(src_mesh_file, 1, 1, false);
   // Mesh mesh_2(tar_mesh_file, 1, 1, false);
   mesh_1.UniformRefinement();
   mesh_1.UniformRefinement();
   // mesh_2.UniformRefinement();
   const int dim = mesh_1.Dimension();
   // MFEM_ASSERT(dim == mesh_2.Dimension(), "Source and target meshes "
   //             "must be in the same dimension.");
   // MFEM_VERIFY(dim > 1, "GSLIB requires a 2D or a 3D mesh" );

   // if (mesh_1.GetNodes() == NULL) { mesh_1.SetCurvature(order); }
   // if (mesh_2.GetNodes() == NULL) { mesh_2.SetCurvature(order); }
   // const int mesh_poly_deg1 = mesh_1.GetNodes()->FESpace()->GetOrder(0);
   // const int mesh_poly_deg2 = mesh_2.GetNodes()->FESpace()->GetOrder(0);
   // cout << "Source mesh curvature: "
   //      << mesh_1.GetNodes()->OwnFEC()->Name() << endl
   //      << "Target mesh curvature: "
   //      << mesh_2.GetNodes()->OwnFEC()->Name() << endl;

   // L2_FECollection src_fec(0, dim);
   // FiniteElementSpace src_fes(&mesh_2, &src_fec);

   // MFEM_VERIFY(mesh_2.GetNumGeometries(mesh_2.Dimension()) == 1, "Mixed meshes"
   //             "are not currently supported.");

   // // Ensure the source grid function can be transferred using GSLIB-FindPoints.
   // const int NE = mesh_2.GetNE(),
   //           nsp = src_fes.GetFE(0)->GetNodes().GetNPoints();

   // // Generate list of points where the grid function will be evaluated.
   // Vector vxyz(nsp*NE*dim);
   // for (int i = 0; i < NE; i++)
   // {
   //    const FiniteElement *fe = src_fes.GetFE(i);
   //    const IntegrationRule ir = fe->GetNodes();
   //    ElementTransformation *et = src_fes.GetElementTransformation(i);

   //    DenseMatrix pos;
   //    et->Transform(ir, pos);
   //    Vector rowx(vxyz.GetData() + i*nsp, nsp),
   //           rowy(vxyz.GetData() + i*nsp + NE*nsp, nsp),
   //           rowz;
   //    if (dim == 3)
   //    {
   //       rowz.SetDataAndSize(vxyz.GetData() + i*nsp + 2*NE*nsp, nsp);
   //    }
   //    pos.GetRow(0, rowx);
   //    pos.GetRow(1, rowy);
   //    if (dim == 3) { pos.GetRow(2, rowz); }
   // }

   // const int nodes_cnt = vxyz.Size() / dim;

   // // Evaluate source grid function.
   // Vector interp_vals(nodes_cnt);
   // FindPointsGSLIB finder;
   // finder.Setup(mesh_1);
   // finder.FindPoints(vxyz);

   // Array<unsigned int> elem_map = finder.GetElem();
   // Array<unsigned int> code = finder.GetCode();
   // // for (int i = 0; i < elem_map.Size(); i++) {
   //    //  std::cout << i << " " << elem_map[i] << " Subdomain and Global element number \n";
   // // }

   // // Free the internal gslib data.
   // finder.FreeData();

   // // testing dof maps 
   // // H1 test;
   // FiniteElementCollection * fec = new H1_FECollection(order,dim);
   // FiniteElementSpace * fes1 = new FiniteElementSpace(&mesh_1,fec);
   // FiniteElementSpace * fes2 = new FiniteElementSpace(&mesh_2,fec);

   int ne1 = mesh_1.GetNE();
   // int ne2 = mesh_2.GetNE();

   // // dof map from fes_2 to fes_1
   // Array<int> dof_map(fes2->GetTrueVSize());
   // for (int iel2 = 0; iel2<ne2; iel2++)
   // {
   //    int iel1 = elem_map[iel2];
   //    Array<int> ElemDofs1;
   //    Array<int> ElemDofs2;
   //    fes1->GetElementDofs(iel1,ElemDofs1);
   //    fes2->GetElementDofs(iel2,ElemDofs2);
   //       // the sizes have to match
   //    MFEM_VERIFY(ElemDofs1.Size() == ElemDofs2.Size(),
   //                "Size inconsistency");
   //    // loop through the dofs and take into account the signs;
   //    int ndof = ElemDofs1.Size();
   //    for (int i = 0; i<ndof; ++i)
   //    {
   //       int pdof_ = ElemDofs2[i];
   //       int gdof_ = ElemDofs1[i];
   //       int pdof = (pdof_ >= 0) ? pdof_ : abs(pdof_) - 1;
   //       int gdof = (gdof_ >= 0) ? gdof_ : abs(gdof_) - 1;
   //       dof_map[pdof] = gdof;
   //    }
   // }

   // GridFunction gf2(fes2);
   // gf2 = 0;
   // FunctionCoefficient cf2(funccoeff);
   // gf2.ProjectCoefficient(cf2);

   // GridFunction gf1(fes1);
   // gf1 = 0;
   // gf1.SetSubVector(dof_map,gf2);
   int subdivisions = 40;
   Array<double> angles(subdivisions+1);
   angles[0] = 0.0;
   double length = 360/subdivisions;
   double range;
   for (int i = 1; i<=subdivisions; i++)
   {
      range = i*length;
      angles[i] = range;
   }
   
   // set element attributes
   for (int i = 0; i < ne1; ++i)
   {
      Element *el = mesh_1.GetElement(i);
      // roughly the element center
      Vector center(dim);
      int geom = mesh_1.GetElementBaseGeometry(i);
      ElementTransformation * tr = mesh_1.GetElementTransformation(i);
      tr->Transform(Geometries.GetCenter(geom),center);
      // center.Print();
      double x = center[0];
      double y = center[1];
      double theta = atan(y/x);
      int k = 0;
      
      if (x<0)
      {
         k = 1;
      }
      else if (y<0)
      {
         k = 2;
      }
      theta += k*M_PI;

      double thetad = theta * 180.0/M_PI;

      // Find the angle relative to (0,0,z)
      int attr = get_angle_range(thetad, angles) + 1;
      el->SetAttribute(attr);
   }
   mesh_1.SetAttributes();
   ofstream mesh_ofs("mesh1.mesh");
   mesh_ofs.precision(8);
   mesh_1.Print(mesh_ofs);

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

   return 0;
}


double funccoeff(const Vector & x)
{
   return sin(3*M_PI*(x.Sum()));
}

int get_angle_range(double angle, Array<double> angles)
{
   auto it = std::upper_bound(angles.begin(), angles.end(), angle);
   return std::distance(angles.begin(),it)-1;
   
}