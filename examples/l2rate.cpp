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
// Compile with: make l2rate
//
// Sample runs:  rates -m ../data/inline-segment.mesh -rs 1 -o 1


#include "../mfem.hpp"
#include <fstream>
#include <iostream>
using namespace std;
using namespace mfem;

// H1
double scalar_u_exact(const Vector &x);

int dim;
int prob=0;

int main(int argc, char *argv[])
{
   // 1. Parse command-line options.
   const char *mesh_file = "../data/inline-quad.mesh";
   int order = 1;
   bool visualization = 1;
   int rs = 1;
   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree)");
   args.AddOption(&rs, "-rs", "--serial_ref",
                  "Number of serial refinements.");
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

   Vector el_size(rs);
   Vector l2_error(rs);
   for (int i = 0; i < rs; i++) {
       // Read the (serial) mesh from the given mesh file.
       Mesh mesh(mesh_file, 1, 1);
       dim = mesh.Dimension();

       // Refine the mesh upto desired level
       for (int ii = 0; ii < i; ii++) {
           mesh.UniformRefinement();
       }

       // Define a finite element space on the parallel mesh.
       L2_FECollection fec(order, dim);
       FiniteElementSpace fespace(&mesh, &fec);

       // Define the solution vector x
       GridFunction x(&fespace);
       x = 0.0;

       // Define the function coefficient
       FunctionCoefficient scoeff(scalar_u_exact);

       // do nodal projection - matches only nodal values
       x.ProjectCoefficient(scoeff);

       // do L2 projection instead - matches L2 integral over the element
       // (i.e. values at quadrature points matter as well)
//       x = 0.0;
//       ConstantCoefficient one(1.0);
//       BilinearForm a(&fespace);
//       LinearForm b(&fespace);
//       b.AddDomainIntegrator(new DomainLFIntegrator(scoeff));
//       a.AddDomainIntegrator(new MassIntegrator(one));
//       b.Assemble();
//       a.Assemble();
//       a.Finalize();
//       const SparseMatrix &A = a.SpMat();
//       GSSmoother M(A);
//       PCG(A, M, b, x, 0, 500, 1e-12, 0.0);
       //L2 projection complete

       el_size(i) = mesh.GetElementSize(0); //store size of first element
       l2_error(i) = x.ComputeL2Error(scoeff); // store global l2 error
       double conv_rate = 0.0;
       double h_reduction = 1.0;
       double e_reduction = 1.0;
       if (i == 0) {
           std::cout << std::right<< std::setw(11)<< "Order (p) "
                     << std::setw(13) << " Element Size (h) "
                     << std::setw(13) << " Error (e)   "
                     << std::setw(13) << " h-reduction "
                     << std::setw(13) << " e-reduction ";
           std::cout <<  std::setw(15) << "Rate (p+1) " << "\n";
       }
       else {
           h_reduction = el_size(i-1)/el_size(i);
           e_reduction = l2_error(i-1)/l2_error(i);
           conv_rate = std::log(e_reduction)/log(h_reduction);
       }

       std::cout << std::setw(11)<< order
                 << std::setw(13) << el_size(i)
                 << std::setw(13) << l2_error(i)
                 << std::setw(13) << h_reduction
                 << std::setw(13) << e_reduction;
       std::cout <<  std::setw(15) << conv_rate << "\n";

       // 8. Send the solution by socket to a GLVis server.
       if (visualization)
       {
          char vishost[] = "localhost";
          int  visport   = 19916;
          socketstream sol_sock(vishost, visport);
          sol_sock.precision(8);
          sol_sock << "solution\n" << mesh << x <<
                   "window_title 'Numerical Solution' "
                   << flush;
       }
   }


   return 0;
}

double scalar_u_exact(const Vector &x)
{
   double val = 1.0;
   val = sin(2.0*M_PI*(x(0))); //sin(2*pi*x)
//   val = std::pow(x(0), 3);
   return val;
}
