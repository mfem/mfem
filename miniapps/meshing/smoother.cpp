// Copyright (c) 2010-2021, Lawrence Livermore National Security, LLC. Produced
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
#include "../common/mfem-common.hpp"

using namespace mfem;
using namespace std;

char vishost[] = "localhost";
int  visport   = 19916;
int  wsize     = 350;

int problem = 0;

double level_set_func(const Vector &coord)
{
   double x = coord(0), y = coord(1);

   if (problem == 0)
   {
      // Smooth square
      x -= 0.5; y -= 0.5;
      return 0.25 - sqrt(x*x + y*y + 1e-12);
   }
   if (problem == 1)
   {
      // Vertical line.
      return (x < 0.5) ? 0.5 - x : 5.0 - 10.0 * x;
   }
   return 1.0;
}

class GradientDot11Coefficient : public Coefficient
{
private:
   const ParGridFunction &u;
public:
   GradientDot11Coefficient(const ParGridFunction &g) : u(g) { }

   virtual double Eval(ElementTransformation &T, const IntegrationPoint &ip);
};

int main (int argc, char *argv[])
{
   // 0. Initialize MPI.
   int num_procs, myid;
   MPI_Init(&argc, &argv);
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
   MPI_Comm_rank(MPI_COMM_WORLD, &myid);

   const char *mesh_file = "../../data/inline-quad.mesh";
   int rs_levels = 2;
   int order = 2;

   // 2. Parse command-line options.
   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&problem, "-p", "--problem",
                  "Problem setup to use.");
   args.AddOption(&rs_levels, "-rs", "--refine-serial",
                  "Number of times to refine the mesh uniformly in serial.");
   args.AddOption(&order, "-o", "--order",
                  "Polynomial degree of mesh finite element space.");
   args.Parse();
   if (!args.Good())
   {
      if (myid == 0) { args.PrintUsage(cout); }
      return 1;
   }
   if (myid == 0) { args.PrintOptions(cout); }

   Mesh *mesh = new Mesh(mesh_file, 1, 1, false);
   for (int lev = 0; lev < rs_levels; lev++) { mesh->UniformRefinement(); }
   ParMesh pmesh(MPI_COMM_WORLD, *mesh);
   delete mesh;
   const int dim = pmesh.Dimension();

   H1_FECollection fec_H1(order, dim);
   ParFiniteElementSpace pfes_H1(&pmesh, &fec_H1);
   ParGridFunction u(&pfes_H1);

   FunctionCoefficient ls_coeff(level_set_func);
   u.ProjectCoefficient(ls_coeff);
   socketstream sock_u;
   common::VisualizeField(sock_u, vishost, visport, u, "u",
                          0, 0, wsize, wsize, "Rjm***");

   L2_FECollection fec_L2(order, dim);
   ParFiniteElementSpace pfes_L2(&pmesh, &fec_L2);
   ParGridFunction grad_u_dot_1(&pfes_L2);
   GradientDot11Coefficient grad_u_dot_1_coeff(u);
   grad_u_dot_1.ProjectCoefficient(grad_u_dot_1_coeff);

   socketstream sock_grad;
   common::VisualizeField(sock_grad, vishost, visport, grad_u_dot_1,
                          "grad_u . 1", 0, wsize, wsize, wsize, "Rjm***");

   MPI_Finalize();
   return 0;
}

double GradientDot11Coefficient::Eval(ElementTransformation &T,
                                      const IntegrationPoint &ip)
{
   const int dim = T.GetDimension();
   Vector grad_u(dim), one(dim);
   u.GetGradient(T, grad_u);
   one = 1.0;
   return grad_u * one;
}
