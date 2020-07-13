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

// Compile with: make rates
//
// Sample runs:  mpirun -np 4 rates -m ../../data/inline-segment.mesh -sr 1 -pr 4 -prob 0 -o 1
//               mpirun -np 4 rates -m ../../data/inline-quad.mesh -sr 1 -pr 3 -prob 0 -o 2
//               mpirun -np 4 rates -m ../../data/inline-quad.mesh -sr 1 -pr 3 -prob 1 -o 2
//               mpirun -np 4 rates -m ../../data/inline-quad.mesh -sr 1 -pr 3 -prob 2 -o 2
//               mpirun -np 4 rates -m ../../data/inline-tri.mesh -sr 1 -pr 3 -prob 2 -o 3
//               mpirun -np 4 rates -m ../../data/star.mesh -sr 1 -pr 2 -prob 1 -o 4
//               mpirun -np 4 rates -m ../../data/fichera.mesh -sr 1 -pr 2 -prob 2 -o 2
//               mpirun -np 4 rates -m ../../data/inline-wedge.mesh -sr 0 -pr 2 -prob 0 -o 2
//               mpirun -np 4 rates -m ../../data/inline-hex.mesh -sr 0 -pr 1 -prob 1 -o 3
//               mpirun -np 4 rates -m ../../data/square-disc.mesh -sr 1 -pr 2 -prob 1 -o 2
//               mpirun -np 4 rates -m ../../data/square-disc.mesh -sr 1 -pr 2 -prob 3 -o 2
//
// Description:  This example code demonstrates the use of MFEM to define
//               and solve finite element problem for various discretizations
//               and provide convergence rates
//
//               prob 0: H1 projection:
//                       (grad u, grad v) + (u,v) = (grad u_exact, grad v) + (u_exact, v)
//               prob 1: H(curl) projection
//                       (curl u, curl v) + (u,v) = (curl u_exact, curl v) + (u_exact, v)
//               prob 2: H(div) projection
//                       (div  u, div  v) + (u,v) = (div  u_exact, div  v) + (u_exact, v)
//               prob 3: DG discretization for the Poisson problem 
//                       -Delta u = f
#include "mfem.hpp"
#include "conv_rates.hpp"
#include <fstream>
#include <iostream>
using namespace std;
using namespace mfem;

// Exact solution parameters: 
double sol_s[3] = { -0.32, 0.15, 0.24 };
double sol_k[3] = { 1.21, 1.45, 1.37 };

// H1
double u_exact(const Vector &x);
double rhs_func(const Vector &x);
void gradu_exact(const Vector &x, Vector &gradu);

// Vector FE
void U_exact(const Vector &x, Vector & U);
// H(curl)
void curlU_exact(const Vector &x, Vector &curlU);
// H(div)
double divU_exact(const Vector &x);

int dim;
int prob=0;

int main(int argc, char *argv[])
{
   // 1. Initialize MPI.
   int num_procs, myid;
   MPI_Init(&argc, &argv);
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
   MPI_Comm_rank(MPI_COMM_WORLD, &myid);

   // 2. Parse command-line options.
   const char *mesh_file = "../../data/inline-quad.mesh";
   int order = 1;
   bool visualization = 1;
   int sr = 1;
   int pr = 1;
   double sigma = -1.0;
   double kappa = -1.0;
   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree)");
   args.AddOption(&prob, "-prob", "--problem",
                  "Problem kind: 0: H1, 1: H(curl), 2: H(div), 3: DG ");
   args.AddOption(&sigma, "-s", "--sigma",
                  "One of the two DG penalty parameters, typically +1/-1."
                  " See the documentation of class DGDiffusionIntegrator.");
   args.AddOption(&kappa, "-k", "--kappa",
                  "One of the two DG penalty parameters, should be positive."
                  " Negative values are replaced with (order+1)^2.");                     
   args.AddOption(&sr, "-sr", "--serial_ref",
                  "Number of serial refinements.");
   args.AddOption(&pr, "-pr", "--parallel_ref",
                  "Number of parallel refinements.");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.Parse();
   if (!args.Good())
   {
      if (myid == 0)
      {
         args.PrintUsage(cout);
      }
      MPI_Finalize();
      return 1;
   }
   if (prob >3 || prob <0) prob = 0; //default problem = H1
   if (prob == 3)
   {
      if (kappa < 0)
      {
         kappa = (order+1)*(order+1);
      }
   }
   if (myid == 0)
   {
      args.PrintOptions(cout);
   }

   // 3. Read the (serial) mesh from the given mesh file on all processors.  We
   //    can handle triangular, quadrilateral, tetrahedral, hexahedral, surface
   //    and volume meshes with the same code.
   Mesh *mesh = new Mesh(mesh_file, 1, 1);
   dim = mesh->Dimension();
   // 5. Refine the serial mesh on all processors to increase the resolution.
   for (int i = 0; i < sr; i++ )
   {
      mesh->UniformRefinement();
   }

   // 6. Define a parallel mesh by a partitioning of the serial mesh. Once the
   //    parallel mesh is defined, the serial mesh can be deleted.
   ParMesh *pmesh = new ParMesh(MPI_COMM_WORLD, *mesh);
   delete mesh;

   // 7. Define a parallel finite element space on the parallel mesh.
   FiniteElementCollection *fec=nullptr;
   switch (prob)
   {
      case 0: fec = new H1_FECollection(order,dim);   break;
      case 1: fec = new ND_FECollection(order,dim);   break;
      case 2: fec = new RT_FECollection(order-1,dim); break;
      case 3: fec = new DG_FECollection(order,dim); break;
      default: break;
   }
   ParFiniteElementSpace *fespace = new ParFiniteElementSpace(pmesh, fec);

   // 8. Define the solution vector x as a parallel finite element grid function
   //     corresponding to fespace.
   ParGridFunction x(fespace);
   x = 0.0;
   // 9. Set up the parallel linear form b(.) and the parallel bilinear form
   //    a(.,.).
   FunctionCoefficient *f=nullptr;
   FunctionCoefficient *u=nullptr;
   FunctionCoefficient *divU=nullptr;
   VectorFunctionCoefficient *U=nullptr;
   VectorFunctionCoefficient *gradu=nullptr;
   VectorFunctionCoefficient *curlU=nullptr;

   ConstantCoefficient one(1.0);
   ParLinearForm b(fespace);
   ParBilinearForm a(fespace);

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
         curlU = new VectorFunctionCoefficient((dim==3)?dim:1,curlU_exact);
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

      case 3:
         u = new FunctionCoefficient(u_exact);
         f = new FunctionCoefficient(rhs_func);
         gradu = new VectorFunctionCoefficient(dim,gradu_exact);
         b.AddDomainIntegrator(new DomainLFIntegrator(*f));
         b.AddBdrFaceIntegrator(
      new DGDirichletLFIntegrator(*u, one, sigma, kappa));
         a.AddDomainIntegrator(new DiffusionIntegrator(one));
         a.AddInteriorFaceIntegrator(new DGDiffusionIntegrator(one, sigma, kappa));
         a.AddBdrFaceIntegrator(new DGDiffusionIntegrator(one, sigma, kappa));
         break;   

      default:
         break;
   }

   // 10. Perform successive parallel refinements, compute the L2 error and the
   //     corresponding rate of convergence
   Convergence rates;
   for (int l = 0; l <= pr; l++)
   {
      b.Assemble();
      a.Assemble();
      a.Finalize();

      HypreParMatrix *A = a.ParallelAssemble();
      HypreParVector *B = b.ParallelAssemble();
      HypreParVector *X = x.ParallelProject();

      Solver *prec = NULL;
      IterativeSolver *solver = NULL;
      switch (prob)
      {
         case 0:
         case 3:
            prec = new HypreBoomerAMG(*A);
            dynamic_cast<HypreBoomerAMG *>(prec)->SetPrintLevel(0);
            break;
         case 1:
            prec = new HypreAMS(*A, fespace);
            dynamic_cast<HypreAMS *>(prec)->SetPrintLevel(0);
            break;
         case 2:
            if (dim == 2)
            {
               prec = new HypreAMS(*A, fespace);
               dynamic_cast<HypreAMS *>(prec)->SetPrintLevel(0);
            }
            else
            {
               prec = new HypreADS(*A, fespace);
               dynamic_cast<HypreADS *>(prec)->SetPrintLevel(0);
            }
            break;
         default:
            break;
      }
      if (prob==3 && sigma !=-1.0)
      {
         solver = new GMRESSolver(MPI_COMM_WORLD);
      }
      else
      {
         solver = new CGSolver(MPI_COMM_WORLD);
      }
      solver->SetRelTol(1e-12);
      solver->SetMaxIter(2000);
      solver->SetPrintLevel(0);
      solver->SetPreconditioner(*prec); 
      solver->SetOperator(*A);
      solver->Mult(*B, *X);
      delete prec;
      delete solver;

      x = *X;
      switch (prob)
      {
         case 0: rates.AddGridFunction(&x,u,gradu); break;
         case 1: rates.AddGridFunction(&x,U,curlU); break;
         case 2: rates.AddGridFunction(&x,U,divU);  break;
         case 3: rates.AddGridFunction(&x,u,gradu,&one); break;
      }

      if (l==pr) break;

      pmesh->UniformRefinement();
      fespace->Update();
      a.Update();
      b.Update();
      x.Update();
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
      sol_sock << "parallel " << num_procs << " " << myid << "\n";
      sol_sock.precision(8);
      sol_sock << "solution\n" << *pmesh << x <<
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
   delete pmesh;

   MPI_Finalize();

   return 0;
}

double rhs_func(const Vector &x)
{
   double val = 1.0, lap = 0.0;
   for (int d = 0; d < x.Size(); d++)
   {
      const double f = sin(M_PI*(sol_s[d]+sol_k[d]*x(d)));
      val *= f;
      lap = lap*f + val*M_PI*M_PI*sol_k[d]*sol_k[d];
   }
   return lap;
}

double u_exact(const Vector &x)
{
   double val = 1.0;
   for (int d = 0; d < x.Size(); d++)
   {
      val *= sin(M_PI*(sol_s[d]+sol_k[d]*x(d)));
   }
   return val;
}

void gradu_exact(const Vector &x, Vector &grad)
{
   grad.SetSize(x.Size());
   double *g = grad.GetData();
   double val = 1.0;
   for (int d = 0; d < x.Size(); d++)
   {
      const double y = M_PI*(sol_s[d]+sol_k[d]*x(d));
      const double f = sin(y);
      for (int j = 0; j < d; j++) { g[j] *= f; }
      g[d] = val*M_PI*sol_k[d]*cos(y);
      val *= f;
   }
}

void U_exact(const Vector &x, Vector & U)
{
   double s = x.Sum();
   for (int d=0; d<dim; d++)
   {
      U[d] = cos(M_PI*sol_s[d] * s);
   }
}
// H(curl)
void curlU_exact(const Vector &x, Vector &curlU)
{
   if (dim==3)
   {
      double s = x.Sum();
      curlU[0] = - M_PI*sol_s[2]*sin(M_PI*sol_s[2] * s) 
                 + M_PI*sol_s[1]*sin(M_PI*sol_s[1] * s);
      curlU[1] = - M_PI*sol_s[0]*sin(M_PI*sol_s[0] * s) 
                 + M_PI*sol_s[2]*sin(M_PI*sol_s[2] * s);
      curlU[2] = - M_PI*sol_s[1]*sin(M_PI*sol_s[1] * s)  
                 + M_PI*sol_s[0]*sin(M_PI*sol_s[0] * s);
   }
   else
   {
      double s = x(0) + x(1);
      curlU[0] = - M_PI*sol_s[1]*sin(M_PI*sol_s[1] * s) 
                 + M_PI*sol_s[0]*sin(M_PI*sol_s[0] * s);
   }
}

// H(div)
double divU_exact(const Vector &x)
{
   double divu = 0.0;
   double s = x.Sum();

   for (int d = 0; d<dim; d++)
   {
      divu += -M_PI*sol_s[d] * sin(M_PI*sol_s[d] * s);
   }
   return divu;
}
