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
//       --------------------------------------------------------------
//       LOR Transfer Miniapp:  Map functions between HO and LOR spaces
//       --------------------------------------------------------------
//
// This miniapp visualizes the maps between a high-order (HO) finite element
// space, typically using high-order functions on a high-order mesh, and a
// low-order refined (LOR) finite element space, typically defined by 0th or 1st
// order functions on a low-order refinement of the HO mesh.
//
// Two main operators are illustrated:
//
//  1. R: HO -> LOR, defined by FiniteElementSpace::GetTransferOperator
//  2. P: LOR -> HO, defined by FiniteElementSpace::GetReverseTransferOperator
//
// While defined generally, these operators have some nice properties for
// particular finite element spaces. For example they satisfy PR=I, plus mass
// conservation in both directions for L2 fields.
//
// Compile with: make lor-transfer
//
// Sample runs:  lor-transfer
//               lor-transfer -h1
//               lor-transfer -t
//               lor-transfer -m ../../data/star-q2.mesh -lref 5 -p 4
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
double RHO_exact(const Vector &x);

// Helper functions
void visualize(VisItDataCollection &, string, int, int);
double compute_mass(FiniteElementSpace *, double, VisItDataCollection &,
                    string);

int main(int argc, char *argv[])
{
   // Parse command-line options.
   const char *mesh_file = "../../data/star.mesh";
   int order = 4;
   int lref = order;
   int lorder = 0;
   bool vis = true;
   bool useH1 = false;
   bool use_transfer = false;

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
   args.AddOption(&use_transfer, "-t", "--use-pointwise-transfer", "-no-t",
                  "--dont-use-pointwise-transfer",
                  "Use pointwise transfer operators instead of L2 projection.");
   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(cout);
      return 1;
   }
   args.PrintOptions(cout);

   // Read the mesh from the given mesh file.
   Mesh mesh(mesh_file, 1, 1);
   int dim = mesh.Dimension();

   // Create the low-order refined mesh
   int basis_lor = BasisType::GaussLobatto; // BasisType::ClosedUniform;
   Mesh mesh_lor(&mesh, lref, basis_lor);

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
      fec = new H1_FECollection(order-1, dim);
      fec_lor = new H1_FECollection(lorder, dim);
   }
   else
   {
      space = "L2";
      fec = new L2_FECollection(order-1, dim);
      fec_lor = new L2_FECollection(lorder, dim);
   }

   FiniteElementSpace fespace(&mesh, fec);
   FiniteElementSpace fespace_lor(&mesh_lor, fec_lor);

   GridFunction rho(&fespace);
   GridFunction rho_lor(&fespace_lor);

   // Data collections for vis/analysis
   VisItDataCollection HO_dc("HO", &mesh);
   HO_dc.RegisterField("density", &rho);
   VisItDataCollection LOR_dc("LOR", &mesh_lor);
   LOR_dc.RegisterField("density", &rho_lor);


   // HO projections
   direction = "HO -> LOR @ HO";
   FunctionCoefficient RHO(RHO_exact);
   rho.ProjectCoefficient(RHO);
   double ho_mass = compute_mass(&fespace, -1.0, HO_dc, "HO       ");
   if (vis) { visualize(HO_dc, "HO", Wx, Wy); Wx += offx; }

   GridTransfer *gt;
   if (use_transfer)
   {
      gt = new InterpolationGridTransfer(fespace, fespace_lor);
   }
   else
   {
      gt = new L2ProjectionGridTransfer(fespace, fespace_lor);
   }
   const Operator &R = gt->ForwardOperator();
   const Operator &P = gt->BackwardOperator();

   // HO->LOR restriction
   direction = "HO -> LOR @ LOR";
   R.Mult(rho, rho_lor);
   compute_mass(&fespace_lor, ho_mass, LOR_dc, "R(HO)    ");
   if (vis) { visualize(LOR_dc, "R(HO)", Wx, Wy); Wx += offx; }

   // LOR->HO prolongation
   direction = "HO -> LOR @ HO";
   GridFunction rho_prev = rho;
   P.Mult(rho_lor, rho);
   compute_mass(&fespace, ho_mass, HO_dc, "P(R(HO)) ");
   if (vis) { visualize(HO_dc, "P(R(HO))", Wx, Wy); Wx = 0; Wy += offy; }

   rho_prev -= rho;
   cout.precision(12);
   cout << "|HO - P(R(HO))|_∞   = " << rho_prev.Normlinf() << endl << endl;

   // LOR projections
   direction = "LOR -> HO @ LOR";
   rho_lor.ProjectCoefficient(RHO);
   GridFunction rho_lor_prev = rho_lor;
   double lor_mass = compute_mass(&fespace_lor, -1.0, LOR_dc, "LOR      ");
   if (vis) { visualize(LOR_dc, "LOR", Wx, Wy); Wx += offx; }

   // Prolongate to HO space
   direction = "LOR -> HO @ HO";
   P.Mult(rho_lor, rho);
   compute_mass(&fespace, lor_mass, HO_dc, "P(LOR)   ");
   if (vis) { visualize(HO_dc, "P(LOR)", Wx, Wy); Wx += offx; }

   // Restrict back to LOR space. This won't give the original function because
   // the rho_lor doesn't necessarily live in the range of R.
   direction = "LOR -> HO @ LOR";
   R.Mult(rho, rho_lor);
   compute_mass(&fespace_lor, lor_mass, LOR_dc, "R(P(LOR))");
   if (vis) { visualize(LOR_dc, "R(P(LOR))", Wx, Wy); }

   rho_lor_prev -= rho_lor;
   cout.precision(12);
   cout << "|LOR - R(P(LOR))|_∞ = " << rho_lor_prev.Normlinf() << endl;

   delete fec;
   delete fec_lor;

   return 0;
}


double RHO_exact(const Vector &x)
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


void visualize(VisItDataCollection &dc, string prefix, int x, int y)
{
   int w = Ww, h = Wh;

   char vishost[] = "localhost";
   int  visport   = 19916;

   socketstream sol_sockL2(vishost, visport);
   sol_sockL2.precision(8);
   sol_sockL2 << "solution\n" << *dc.GetMesh() << *dc.GetField("density")
              << "window_geometry " << x << " " << y << " " << w << " " << h
              << "plot_caption '" << space << " " << prefix << " Density'"
              << "window_title '" << direction << "'" << flush;
}


double compute_mass(FiniteElementSpace *L2, double massL2,
                    VisItDataCollection &dc, string prefix)
{
   ConstantCoefficient one(1.0);
   BilinearForm ML2(L2);
   ML2.AddDomainIntegrator(new MassIntegrator(one));
   ML2.Assemble();

   GridFunction rhoone(L2);
   rhoone = 1.0;

   double newmass = ML2.InnerProduct(*dc.GetField("density"),rhoone);
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
