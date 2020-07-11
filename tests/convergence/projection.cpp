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

// Compile with: make projection
//
// Sample runs:  projection -m ../../data/inline-segment.mesh -sr 4 -prob 0 -o 1
//               projection -m ../../data/inline-quad.mesh -sr 3 -prob 0 -o 2
//               projection -m ../../data/inline-quad.mesh -sr 3 -prob 1 -o 2
//               projection -m ../../data/inline-quad.mesh -sr 3 -prob 2 -o 2
//               projection -m ../../data/inline-tri.mesh -sr 2 -prob 2 -o 3
//               projection -m ../../data/star.mesh -sr 2 -prob 1 -o 4
//               projection -m ../../data/fichera.mesh -sr 3 -prob 2 -o 1
//               projection -m ../../data/inline-wedge.mesh -sr 1 -prob 0 -o 2
//               projection -m ../../data/inline-hex.mesh -sr 1 -prob 1 -o 2
//               projection -m ../../data/square-disc.mesh -sr 2 -prob 1 -o 1
//
// Description:  This example code is used for testing the LF-integrators
//               (Q,grad v), (Q,curl V), (Q, div v)
//               by solving the appropriate energy projection problems
//
//               prob 0: (grad u, grad v) + (u,v) = (grad u_exact, grad v) + (u_exact, v)
//               prob 1: (curl u, curl v) + (u,v) = (curl u_exact, curl v) + (u_exact, v)
//               prob 2: (div  u, div  v) + (u,v) = (div  u_exact, div  v) + (u_exact, v)

// #include "mfem.hpp"
#include "conv_rates.hpp"
#include <fstream>
#include <iostream>
using namespace std;
using namespace mfem;

// H1
double u_exact(const Vector &x);
void gradu_exact(const Vector &x, Vector &gradu);
// Vector FE
void U_exact(const Vector &x, Vector & U);
// H(curl)
void curlU_exact(const Vector &x, Vector &curlU);
// H(div)
double divU_exact(const Vector &x);

int dim;
int prob=0;
Vector alpha;

int main(int argc, char *argv[])
{
   // 1. Parse command-line options.
   const char *mesh_file = "../../data/inline-quad.mesh";
   int order = 1;
   bool visualization = 1;
   int sr = 1;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree)");
   args.AddOption(&prob, "-prob", "--problem",
                  "Problem kind: 0: H1, 1: H(curl), 2: H(div)");
   args.AddOption(&sr, "-sr", "--serial_ref",
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

   // 2. Read the (serial) mesh from the given mesh file. We can 
   //    handle triangular, quadrilateral, tetrahedral, hexahedral, surface
   //    and volume meshes with the same code.
   Mesh *mesh = new Mesh(mesh_file, 1, 1);
   dim = mesh->Dimension();  if (dim == 1 ) prob = 0;

   if (prob >2 || prob <0) prob = 0; //default problem = H1

   // 3. Set up parameters for exact solution
   alpha.SetSize(dim); // x,y,z coefficients of the solution
   for (int i=0; i<dim; i++) { alpha(i) = M_PI*(double)(i+1);}

   // 4. Refine the serial mesh on all processors to increase the resolution.
   mesh->UniformRefinement();

   // 7. Define a finite element space on the parallel mesh.
   FiniteElementCollection *fec=nullptr;
   switch (prob)
   {
      case 0: fec = new H1_FECollection(order,dim);   break;
      case 1: fec = new ND_FECollection(order,dim);   break;
      case 2: fec = new RT_FECollection(order-1,dim); break;
      default: break;
   }
   FiniteElementSpace *fespace = new FiniteElementSpace(mesh, fec);

   // 8. Define the solution vector u_gf as a parallel finite element grid function
   //     corresponding to fespace.
   GridFunction u_gf(fespace);
   
   // 9. Set up the linear form b(.) and the bilinear form a(.,.).
   FunctionCoefficient *u=nullptr;
   FunctionCoefficient *divU=nullptr;
   VectorFunctionCoefficient *U=nullptr;
   VectorFunctionCoefficient *gradu=nullptr;
   VectorFunctionCoefficient *curlU=nullptr;

   ConstantCoefficient one(1.0);
   LinearForm b(fespace);
   BilinearForm a(fespace);

   switch (prob)
   {
      case 0:
         //(grad u_ex, grad v) + (u_ex,v)
         u = new FunctionCoefficient(u_exact);
         gradu = new VectorFunctionCoefficient(dim,gradu_exact);
         b.AddDomainIntegrator(new DomainLFGradIntegrator(*gradu));
         b.AddDomainIntegrator(new DomainLFIntegrator(*u));

         // (grad u, grad v) + (u,v)
         a.AddDomainIntegrator(new DiffusionIntegrator(one));
         a.AddDomainIntegrator(new MassIntegrator(one));

         break;
      case 1:
         //(curl u_ex, curl v) + (u_ex,v)
         U = new VectorFunctionCoefficient(dim,U_exact);
         curlU = new VectorFunctionCoefficient((dim ==3)?dim:1,curlU_exact);
         b.AddDomainIntegrator(new VectorFEDomainLFCurlIntegrator(*curlU));
         b.AddDomainIntegrator(new VectorFEDomainLFIntegrator(*U));

         // (curl u, curl v) + (u,v)
         a.AddDomainIntegrator(new CurlCurlIntegrator(one));
         a.AddDomainIntegrator(new VectorFEMassIntegrator(one));
         break;

      case 2:
         //(div u_ex, div v) + (u_ex,v)
         U = new VectorFunctionCoefficient(dim,U_exact);
         divU = new FunctionCoefficient(divU_exact);
         b.AddDomainIntegrator(new VectorFEDomainLFDivIntegrator(*divU));
         b.AddDomainIntegrator(new VectorFEDomainLFIntegrator(*U));

         // (div u, div v) + (u,v)
         a.AddDomainIntegrator(new DivDivIntegrator(one));
         a.AddDomainIntegrator(new VectorFEMassIntegrator(one));
         break;

      default:
         break;
   }
   // 10. Perform successive parallel refinements, compute the L2 error and the
   //     corresponding rate of convergence
   Convergence rates;
   // ConvergenceRates rates;
   rates.Clear();
   for (int l = 0; l <= sr; l++)
   {
      b.Assemble();
      a.Assemble();
      Array<int> ess_tdof_list;
      if (mesh->bdr_attributes.Size())
      {
         Array<int> ess_bdr(mesh->bdr_attributes.Max());
         ess_bdr = 0;
         fespace->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
      }
      OperatorPtr A;
      Vector X, B;
      a.FormLinearSystem(ess_tdof_list, u_gf, b, A, X,B);

      GSSmoother M(*A.As<SparseMatrix>());
      PCG(*A, M, B, X, 0, 500, 1e-12, 0.0);

      a.RecoverFEMSolution(X,B,u_gf);
      switch (prob)
      {
         case 0:
         {
            rates.AddGridFunction(&u_gf,u,gradu);
            break;
         }
         case 1:
         {
            rates.AddGridFunction(&u_gf,U,curlU);
            break;
         }
         case 2:
         {
            rates.AddGridFunction(&u_gf,U,divU);
            break;
         }
         default:
            break;
      }

      if (l==sr) break;

      mesh->UniformRefinement();
      fespace->Update();
      a.Update();
      b.Update();
      u_gf.Update();
   }

   rates.Print(true);

   // 11. Send the solution by socket to a GLVis server.
   if (visualization)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      string keys;
      if (dim ==2 )
      {
         keys = "keys UUmrRljc\n";
      }
      else
      {
         keys = "keys mc\n";
      }
      socketstream sol_sock(vishost, visport);
      sol_sock.precision(8);
      sol_sock << "solution\n" << *mesh << u_gf <<
               "window_title 'Numerical Pressure (real part)' "
               << keys << flush;
   }

   // 12. Free the used memory.
   delete u;
   delete divU;
   delete U;
   delete gradu;
   delete curlU;
   delete fespace;
   delete fec;
   delete mesh;
   return 0;
}

double u_exact(const Vector &x)
{
   double u;
   double y=0;
   for (int i=0; i<dim; i++)
   {
      y+= alpha(i) * x(i);
   }
   u = cos(y);
   return u;
}

void gradu_exact(const Vector &x, Vector &du)
{
   double s=0.0;
   for (int i=0; i<dim; i++)
   {
      s+= alpha(i) * x(i);
   }
   for (int i=0; i<dim; i++)
   {
      du[i] = -alpha(i) * sin(s);
   }
}

void U_exact(const Vector &x, Vector & U)
{
   double s = x.Sum();
   for (int i=0; i<dim; i++)
   {
      U[i] = cos(alpha(i) * s);
   }
}
// H(curl)
void curlU_exact(const Vector &x, Vector &curlU)
{
   if (dim==3)
   {
      double s = x.Sum();
      curlU[0] = -alpha(2)*sin(alpha(2) * s) + alpha(1)*sin(alpha(1) * s);
      curlU[1] = -alpha(0)*sin(alpha(0) * s) + alpha(2)*sin(alpha(2) * s);
      curlU[2] = -alpha(1)*sin(alpha(1) * s) + alpha(0)*sin(alpha(0) * s);
   }
   else
   {
      double s = x(0) + x(1);
      curlU[0] = -alpha(1)*sin(alpha(1) * s) + alpha(0)*sin(alpha(0) * s);
   }
   
   
}

// H(div)
double divU_exact(const Vector &x)
{
   double divu = 0.0;
   double s = x.Sum();

   for (int i = 0; i<dim; i++)
   {
      divu += -alpha(i) * sin(alpha(i) * s);
   }
   return divu;
}
