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
//       --------------------------------------------------------------
//       LOR Transfer Miniapp:  Map functions between HO and LOR spaces
//       --------------------------------------------------------------
//
// This miniapp visualizes the maps between a high-order (HO) finite element
// space, typically using high-order functions on a high-order mesh, and a
// low-order refined (LOR) finite element space, typically defined by 0th or 1st
// order functions on a low-order refinement of the HO mesh.
//
// The grid transfer operators are represented using either
// InterpolationGridTransfer or L2ProjectionGridTransfer (depending on the
// options requested by the user). The two transfer operators are then:
//
//  1. R: HO -> LOR, defined by GridTransfer::ForwardOperator
//  2. P: LOR -> HO, defined by GridTransfer::BackwardOperator
//
// While defined generally, these operators have some nice properties for
// particular finite element spaces. For example they satisfy PR=I, plus mass
// conservation in both directions for L2 fields.
//
// Compile with: make lor-transfer
//
// Sample runs:  lor-transfer
//               lor-transfer -h1
//               lor-transfer -ea -w
//               lor-transfer -t
//               lor-transfer -m ../../data/star-q2.mesh -lref 5 -p 4
//               lor-transfer -m ../../data/star-mixed.mesh -lref 3 -p 2
//               lor-transfer -lref 4 -o 4 -lo 0 -p 1
//               lor-transfer -lref 5 -o 4 -lo 0 -p 1
//               lor-transfer -lref 5 -o 4 -lo 3 -p 2
//               lor-transfer -lref 5 -o 4 -lo 0 -p 3

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
real_t W_exact(const Vector &x);
real_t weight(const Vector &x);

// Helper functions
void visualize(VisItDataCollection &, string, int, int, int visport = 19916);
real_t compute_mass(GridFunction &, real_t, string, CoefficientWithOrder);

int main(int argc, char *argv[])
{
   // Parse command-line options.
   const char *mesh_file = "../../data/star.mesh";
   int order = 3;
   int lref = order+1;
   int lorder = 0;
   bool vis = true;
   bool useH1 = false;
   int visport = 19916;
   bool use_pointwise_transfer = false;
   bool use_weighted_transfer = false;
   const char *device_config = "cpu";
   bool use_ea       = false;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&problem, "-p", "--problem",
                  "Problem type (see the RHO_exact function).");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree) or -1 for"
                  " isoparametric space.");
   args.AddOption(&lref, "-lref", "--lor-ref-level", "LOR refinement level.");
   args.AddOption(&lorder, "-lo", "--lor-order",
                  "LOR space order (polynomial degree, zero by default).");
   args.AddOption(&vis, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&useH1, "-h1", "--use-h1", "-l2", "--use-l2",
                  "Use H1 spaces instead of L2.");
   args.AddOption(&use_pointwise_transfer, "-t", "--use-pointwise-transfer",
                  "-no-t", "--dont-use-pointwise-transfer",
                  "Use pointwise transfer operators instead of L2 projection.");
   args.AddOption(&use_weighted_transfer, "-w", "--use-weighted-transfer",
                  "-no-w", "--dont-use-weighted-transfer",
                  "Use coefficient-weighted L2 projection.");
   args.AddOption(&device_config, "-d", "--device",
                  "Device configuration string, see Device::Configure().");
   args.AddOption(&use_ea, "-ea", "--ea-version", "-no-ea",
                  "--no-ea-version", "Use element assembly version.");
   args.ParseCheck();

   // Configure device
   Device device(device_config);

   if (use_weighted_transfer && !use_pointwise_transfer)
   {
      if (problem != 5)
      {
         cout << "Switching to positive problem = 5 for weighted transfer.\n";
      }
      problem = 5;
   }

   // Read the mesh from the given mesh file.
   Mesh mesh(mesh_file, 1, 1);
   int dim = mesh.Dimension();

   // Create the low-order refined mesh
   int basis_lor = BasisType::GaussLobatto; // BasisType::ClosedUniform;
   Mesh mesh_lor = Mesh::MakeRefined(mesh, lref, basis_lor);

   // Create spaces
   FiniteElementCollection *fec, *fec_lor;
   if (useH1)
   {
      space = "H1";
      if (lorder == 0)
      {
         lorder = 1;
         cerr << "Switching the H1 LOR space order from 0 to 1\n";
      }
      fec = new H1_FECollection(order, dim);
      fec_lor = new H1_FECollection(lorder, dim);
   }
   else
   {
      space = "L2";
      fec = new L2_FECollection(order, dim);
      fec_lor = new L2_FECollection(lorder, dim);
   }

   FiniteElementSpace fespace(&mesh, fec);
   FiniteElementSpace fespace_lor(&mesh_lor, fec_lor);

   FunctionCoefficient weight_fn_coeff(weight);
   CoefficientWithOrder weight_coeff;
   if (use_weighted_transfer)
   {
      weight_coeff.coeff = &weight_fn_coeff;
      weight_coeff.order = 2;
   }

   GridFunction rho(&fespace);
   GridFunction rho_lor(&fespace_lor);

   // Data collections for vis/analysis
   VisItDataCollection HO_dc("HO", &mesh);
   HO_dc.RegisterField("density", &rho);
   VisItDataCollection LOR_dc("LOR", &mesh_lor);
   LOR_dc.RegisterField("density", &rho_lor);

   BilinearForm M_ho(&fespace);
   M_ho.AddDomainIntegrator(new MassIntegrator);
   M_ho.Assemble();
   M_ho.Finalize();

   BilinearForm M_lor(&fespace_lor);
   M_lor.AddDomainIntegrator(new MassIntegrator);
   M_lor.Assemble();
   M_lor.Finalize();

   // HO projections
   direction = "HO -> LOR @ HO";
   FunctionCoefficient RHO(RHO_exact);
   rho.ProjectCoefficient(RHO);
   // Make sure AMR constraints are satisfied
   rho.SetTrueVector();
   rho.SetFromTrueVector();

   real_t ho_mass = compute_mass(rho, -1.0, "HO       ", weight_coeff);
   if (vis) { visualize(HO_dc, "HO", Wx, Wy, visport); Wx += offx; }

   GridTransfer *gt;
   if (use_pointwise_transfer)
   {
      gt = new InterpolationGridTransfer(fespace, fespace_lor);
   }
   else
   {
      gt = new L2ProjectionGridTransfer(fespace, fespace_lor, weight_coeff,
                                        weight_coeff);
   }

   // Configure element assembly for device acceleration
   gt->UseEA(use_ea);

   const Operator &R = gt->ForwardOperator();

   // HO->LOR restriction
   direction = "HO -> LOR @ LOR";
   R.Mult(rho, rho_lor);
   compute_mass(rho_lor, ho_mass, "R(HO)    ", weight_coeff);
   if (vis) { visualize(LOR_dc, "R(HO)", Wx, Wy, visport); Wx += offx; }

   if (use_weighted_transfer && !use_pointwise_transfer)
   {
      // Transfer velocity while conserving rho-weighted momentum.
      GridFunctionCoefficient rho_coeff(&rho);
      GridFunctionCoefficient rho_lor_coeff(&rho_lor);
      ProductCoefficient prod_coeff(weight_fn_coeff, rho_coeff);
      ProductCoefficient prod_lor_coeff(weight_fn_coeff, rho_lor_coeff);
      CoefficientWithOrder prod_weight(prod_coeff, order + 2);
      CoefficientWithOrder prod_lor_weight(prod_lor_coeff, lorder + 2);

      GridFunction w(&fespace), w_lor(&fespace_lor);
      FunctionCoefficient W(W_exact);
      w.ProjectCoefficient(W);

      cout << '\n';
      const real_t ho_momentum = compute_mass(w, -1.0, "rho w HO ", prod_weight);

      L2ProjectionGridTransfer vel_gt(fespace, fespace_lor, prod_weight,
                                      prod_lor_weight);
      vel_gt.UseEA(use_ea);
      vel_gt.ForwardOperator().Mult(w, w_lor);
      compute_mass(w_lor, ho_momentum, "rho w LOR", prod_lor_weight);

      if (vel_gt.SupportsBackwardsOperator())
      {
         GridFunction w_prev = w;
         vel_gt.BackwardOperator().Mult(w_lor, w);
         compute_mass(w, ho_momentum, "P(rho w) ", prod_weight);

         w_prev -= w;
         cout.precision(12);
         cout << "|w - P(R(w))|_∞     = " << w_prev.Normlinf() << "\n\n";
      }
   }

   if (gt->SupportsBackwardsOperator())
   {
      const Operator &P = gt->BackwardOperator();
      // LOR->HO prolongation
      direction = "HO -> LOR @ HO";
      GridFunction rho_prev = rho;
      P.Mult(rho_lor, rho);
      compute_mass(rho, ho_mass, "P(R(HO)) ", weight_coeff);
      if (vis) { visualize(HO_dc, "P(R(HO))", Wx, Wy, visport); Wx = 0; Wy += offy; }

      rho_prev -= rho;
      cout.precision(12);
      cout << "|HO - P(R(HO))|_∞   = " << rho_prev.Normlinf() << endl;
   }

   // HO* to LOR* dual fields
   LinearForm M_rho(&fespace), M_rho_lor(&fespace_lor);
   if (!use_pointwise_transfer && gt->SupportsBackwardsOperator())
   {
      const Operator &P = gt->BackwardOperator();
      M_ho.Mult(rho, M_rho);
      P.MultTranspose(M_rho, M_rho_lor);
      cout << "HO -> LOR dual field: " << abs(M_rho.Sum()-M_rho_lor.Sum()) << "\n\n";
   }

   // LOR projections
   direction = "LOR -> HO @ LOR";
   rho_lor.ProjectCoefficient(RHO);
   GridFunction rho_lor_prev = rho_lor;
   real_t lor_mass = compute_mass(rho_lor, -1.0, "LOR      ", weight_coeff);
   if (vis) { visualize(LOR_dc, "LOR", Wx, Wy, visport); Wx += offx; }

   if (gt->SupportsBackwardsOperator())
   {
      const Operator &P = gt->BackwardOperator();
      // Prolongate to HO space
      direction = "LOR -> HO @ HO";
      P.Mult(rho_lor, rho);
      compute_mass(rho, lor_mass, "P(LOR)   ", weight_coeff);
      if (vis) { visualize(HO_dc, "P(LOR)", Wx, Wy, visport); Wx += offx; }

      // Restrict back to LOR space. This won't give the original function because
      // the rho_lor doesn't necessarily live in the range of R.
      direction = "LOR -> HO @ LOR";
      R.Mult(rho, rho_lor);
      compute_mass(rho_lor, lor_mass, "R(P(LOR))", weight_coeff);
      if (vis) { visualize(LOR_dc, "R(P(LOR))", Wx, Wy, visport); }

      rho_lor_prev -= rho_lor;
      cout.precision(12);
      cout << "|LOR - R(P(LOR))|_∞ = " << rho_lor_prev.Normlinf() << endl;
   }

   // LOR* to HO* dual fields
   if (!use_pointwise_transfer)
   {
      M_lor.Mult(rho_lor, M_rho_lor);
      R.MultTranspose(M_rho_lor, M_rho);
      cout << "LOR -> HO dual field: " << abs(M_rho.Sum() - M_rho_lor.Sum()) << '\n';
   }

   delete gt;
   delete fec;
   delete fec_lor;

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
      case 5: // positive function
         return 2.0 + 2*x(0)*x(0) + 3*x(1)*x(1) - x(0)*x(1) + 0.1*sin(x.Norml2());
      default:
         return 1.0;
   }
}


real_t W_exact(const Vector &x)
{
   return x(1) + 0.25*cos(2*M_PI*x.Norml2());
}


real_t weight(const Vector &x)
{
   return x(0)*x(0) + x(1)*x(1) + 1.0;
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


real_t compute_mass(GridFunction &gf, real_t oldmass, string prefix,
                    CoefficientWithOrder mass_coeff)
{
   FiniteElementSpace &fes = *gf.FESpace();
   Mesh &mesh = *fes.GetMesh();
   const int order = 2*fes.GetMaxElementOrder()
                     + mesh.GetTypicalElementTransformation()->OrderW()
                     + mass_coeff.order;

   ConstantCoefficient one(1.0);
   DomainLFIntegrator *integ = mass_coeff
                               ? new DomainLFIntegrator(*mass_coeff.coeff)
                               : new DomainLFIntegrator(one);
   integ->SetIntegrationRule(
      IntRules.Get(mesh.GetTypicalElementGeometry(), order));

   LinearForm lf(&fes);
   lf.AddDomainIntegrator(integ);
   lf.Assemble();

   const real_t newmass = lf(gf);
   cout.precision(18);
   cout << space << " " << prefix << " mass   = " << newmass;
   if (oldmass >= 0)
   {
      cout.precision(4);
      cout << " ("  << fabs(newmass-oldmass)*100/oldmass << "%)";
   }
   cout << endl;
   return newmass;
}
