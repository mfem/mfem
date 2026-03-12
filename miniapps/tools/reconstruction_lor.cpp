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
// --------------------------------------------------------------
// Reconstruction using LOR Transfer
// --------------------------------------------------------------


#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

int problem = 1; // problem type

int Wx = 0, Wy = 0; // window position
int Ww = 350, Wh = 350; // window size
int offx = Ww+5, offy = Wh+25; // window offsets

string space;
string direction;

// Exact functions to project
real_t RHO_exact(const Vector &x);

// Helper functions
void visualize(VisItDataCollection &, string, int, int, int visport = 19916);
real_t compute_mass(FiniteElementSpace *, real_t, VisItDataCollection &,
                    string);

int main(int argc, char *argv[])
{
   // Parse command-line options.
   const char *mesh_file = "../../data/star.mesh";
   int order_ho = 3;
   int order_im = 3;
   int lref = order_im+1;
   int order_lo = 0;
   bool vis = true;
   bool useH1 = false;
   int visport = 19916;
   bool use_pointwise_transfer = false;
   const char *device_config = "cpu";
   bool use_ea       = false;

   int refinement_levels = 0;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&problem, "-p", "--problem",
                  "Problem type (see the RHO_exact function).");
   args.AddOption(&order_im, "-io", "--order_im",
                  "Finite element order (polynomial degree) for intermediate space.");
   args.AddOption(&order_ho, "-ho", "--order_ho",
                  "Finite element order (polynomial degree) for high-order space.");                  
   args.AddOption(&refinement_levels, "-r", "--refine",
                  "Number of times to refine the mesh uniformly.");                  
   args.AddOption(&lref, "-lref", "--lor-ref-level", "LOR refinement level.");
   args.AddOption(&order_lo, "-lo", "--lor-order",
                  "LOR space order (polynomial degree, zero by default).");
   args.AddOption(&vis, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&useH1, "-h1", "--use-h1", "-l2", "--use-l2",
                  "Use H1 spaces instead of L2.");
   args.AddOption(&use_pointwise_transfer, "-t", "--use-pointwise-transfer",
                  "-no-t", "--dont-use-pointwise-transfer",
                  "Use pointwise transfer operators instead of L2 projection.");
   args.AddOption(&device_config, "-d", "--device",
                  "Device configuration string, see Device::Configure().");
   args.AddOption(&use_ea, "-ea", "--ea-version", "-no-ea",
                  "--no-ea-version", "Use element assembly version.");
   args.ParseCheck();

   // Configure device
   Device device(device_config);

   // ======================================================
   // Create meshes
   // ======================================================

   // intermediate mesh
   const int num_x = 2;
   const int num_y = 2;
   real_t element_size = 1.0 / std::max(num_x, num_y);
   Mesh mesh_im = Mesh::MakeCartesian2D(num_x, num_y, Element::QUADRILATERAL, false, 1.0, 1.0 ); // sx = sy =1
   for (int i = 0; i < refinement_levels; i++) {
      mesh_im.UniformRefinement();
      element_size /= 2.0;
   }
   mesh_im.EnsureNCMesh();

   int dim = mesh_im.Dimension();

   // low-order refined mesh
   Mesh mesh_lo = Mesh::MakeRefined(mesh_im, lref, BasisType::ClosedUniform); // GaussLobatto, ClosedUniform

   // Other Refinement methods? 

   // Mesh Mesh::MakeRefined(Mesh &orig_mesh, const Array<int> &ref_factors,
   //                     int ref_type)

   // ======================================================
   // Create spaces
   // ======================================================

   // _lo: low-order given
   // _im: intermediate
   // _hi: high-order reconstructed
   // _ex: high-order exact

   FiniteElementCollection *fec_lo;
   FiniteElementCollection *fec_im;
   FiniteElementCollection *fec_hi;

   std::cout<<"SK: order_im = "<<order_im<<", order_ho = "<<order_ho<<std::endl;

   fec_lo = new L2_FECollection(order_lo, dim);
   // fec_im = new L2_FECollection(order_im, dim);
   fec_im = new H1_FECollection(order_im, dim);
   fec_hi = new L2_FECollection(order_ho, dim);

   FiniteElementSpace fespace_lo(&mesh_lo, fec_lo);
   FiniteElementSpace fespace_im(&mesh_im, fec_im);
   FiniteElementSpace fespace_hi(&mesh_lo, fec_hi);

   GridFunction rho_lo(&fespace_lo);
   GridFunction rho_im(&fespace_im);
   GridFunction rho_hi(&fespace_hi);

   // Data collections for vis/analysis
   VisItDataCollection dc_lo("LO", &mesh_lo);
   dc_lo.RegisterField("density", &rho_lo);   
   VisItDataCollection dc_im("IM", &mesh_im);
   dc_im.RegisterField("density", &rho_im);
   VisItDataCollection dc_hi("HO", &mesh_lo);
   dc_hi.RegisterField("density", &rho_hi);

   // ======================================================
   // Create BilinearForms
   // ======================================================

   BilinearForm M_lo(&fespace_lo);
   M_lo.AddDomainIntegrator(new MassIntegrator);
   M_lo.Assemble();
   M_lo.Finalize();

   BilinearForm M_im(&fespace_im);
   M_im.AddDomainIntegrator(new MassIntegrator);
   M_im.Assemble();
   M_im.Finalize();

   BilinearForm M_hi(&fespace_hi);
   M_hi.AddDomainIntegrator(new MassIntegrator);
   M_hi.Assemble();
   M_hi.Finalize();

   // Set up the right-hand side vector for the exact solution
   FunctionCoefficient RHO(RHO_exact);
   LinearForm b_lo(&fespace_lo);
   // b_lo.AddDomainIntegrator(new DomainLFIntegrator(RHO));
   DomainLFIntegrator *lf_integ = new DomainLFIntegrator(RHO);
   const IntegrationRule &ir_rhs = IntRules.Get(fespace_lo.GetFE(0)->GetGeomType(), 
                                                order_hi+1);
   lf_integ->SetIntRule(&ir_rhs);
   b_lo.AddDomainIntegrator(lf_integ);
   b_lo.Assemble();

   // ======================================================
   // Defind GridTransfers
   // ======================================================

   std::cout<<"Defind GridTransfers"<<std::endl;

   GridTransfer *gt1 = nullptr;
   GridTransfer *gt2 = nullptr;
   if (use_pointwise_transfer)
   {
      gt1 = new InterpolationGridTransfer(fespace_im, fespace_lo);
      gt2 = new InterpolationGridTransfer(fespace_hi, fespace_im);
   }
   else
   {
      gt1 = new L2ProjectionGridTransfer(fespace_im, fespace_lo); // (dom_fes_, ran_fes_)
      gt2 = new L2ProjectionGridTransfer(fespace_im, fespace_hi);
   }

   // Configure element assembly for device acceleration
   gt1->UseEA(use_ea);
   gt2->UseEA(use_ea);

   const Operator &P1 = gt1->BackwardOperator();   // Prolongation 1 (LO->IM)
   const Operator &P2 = gt2->ForwardOperator();    // Prolongation 2 (IM->HO)

   const Operator &R1 = gt1->ForwardOperator();    // Restriction 1 (IM->LO)
   const Operator &R2 = gt2->BackwardOperator();   // Restriction 2 (HO->IM)

   // ======================================================
   // Compute GridFunctions
   // ======================================================

   std::cout<<"Compute GridFunctions"<<std::endl;

   // STEP1: LO & EX projections
   
   // L2 projection of RHO onto rho_lo

   // rho_lo.ProjectCoefficient(RHO); // This is not an L2 projection.   
   SparseMatrix &M_mat_lo = M_lo.SpMat();
   CGSolver cg;
   cg.SetOperator(M_mat_lo);
   cg.SetRelTol(1e-16);
   cg.SetMaxIter(1000);
   cg.SetPrintLevel(0);
   rho_lo = 0.0;
   cg.Mult(b_lo, rho_lo); // Solve: M * rho_lo = b_lo
   rho_lo.SetTrueVector();
   rho_lo.SetFromTrueVector();

   real_t mass_ex = compute_mass(&fespace_hi, -1.0, dc_ex, "EX");
   if (vis) { visualize(dc_ex, "EX", Wx, Wy, visport); Wx += offx; }

   real_t mass_lo = compute_mass(&fespace_lo, -1.0, dc_lo, "LO");
   if (vis) { visualize(dc_lo, "LO", Wx, Wy, visport); Wx += offx; }

   // STEP2: Prolongation 1 (LO->IM)

   P1.Mult(rho_lo, rho_im); // rho_im = P1 * rho

   real_t mass_im = compute_mass(&fespace_im, -1.0, dc_im, "IM=P1(LO) ");
   if (vis) { visualize(dc_im, "IM=P1(LO)", Wx, Wy, visport); Wx += offx; }

   // STEP3: Prolongation 2 (IM->HO)

   P2.Mult(rho_im, rho_hi); // rho_hi = P2 * rho_im
   real_t mass_hi = compute_mass(&fespace_hi, -1.0, dc_hi, "HO=P2(IM) ");
   if (vis) { visualize(dc_hi, "HO=P2(IM)", Wx, Wy, visport); Wx += offx; }

   // ======================================================
   // Compute Errors
   // ======================================================   
   // Compute error with respect to exact solution

   cout.precision(16);
   cout << "h ="<<element_size<<", |IM - EX|_{L^2} = " << rho_im.ComputeL2Error(RHO) << endl;
   cout << "h ="<<element_size<<", |HO - EX|_{L^2} = " << rho_hi.ComputeL2Error(RHO) << endl;

   // ======================================================
   // Free memory
   // ======================================================
   delete gt1;
   delete gt2;
   delete fec_lo;
   delete fec_im;
   delete fec_hi;

   return 0;
}


real_t RHO_exact(const Vector &x)
{
   switch (problem)
   {
      case 1: // smooth field
         return x(1)+0.25*cos(2*M_PI*x.Norml2());
      case 2: // cubic function
         return x(1)*x(1)*x(1) + 2*x(0)*x(1) + x(0);
      case 3: // sharp gradient
         return M_PI/2-atan(5*(2*x.Norml2()-1));
      case 4: // basis function
         return (x.Norml2() < 0.1) ? 1 : 0;
      default:
         return 1.0;
   }
}


void visualize(VisItDataCollection &dc, string prefix, int x, int y,
               int visport)
{
   int w = Ww, h = Wh;

   char vishost[] = "localhost";

   socketstream sol_sockL2(vishost, visport);
   sol_sockL2.precision(8);
   sol_sockL2 << "solution\n" << *dc.GetMesh() << *dc.GetField("density")
              << "window_geometry " << x << " " << y << " " << w << " " << h
              << "plot_caption '" << space << " " << prefix << " Density'"
              << "window_title '" << direction << "'" << flush;
}


real_t compute_mass(FiniteElementSpace *L2, real_t massL2,
                    VisItDataCollection &dc, string prefix)
{
   ConstantCoefficient one(1.0);
   LinearForm lf(L2);
   lf.AddDomainIntegrator(new DomainLFIntegrator(one));
   lf.Assemble();

   real_t newmass = lf(*dc.GetField("density"));
   cout.precision(18);
   cout << space << " " << prefix << " mass   = " << newmass;
   if (massL2 >= 0)
   {
      cout.precision(4);
      cout << " ("  << fabs(newmass-massL2)*100/massL2 << "%)";
   }
   cout << endl;
   return newmass;
}
