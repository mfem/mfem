// Copyright (c) 2010-2024, Lawrence Livermore National Security, LLC. Produced
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
// Stabilized Convection-Diffusion

#include "stab_condif.hpp"
#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include <list>

using namespace std;
using namespace mfem;

real_t att_param = 1.0;
real_t kappa_param = 1.0;
real_t pi  = (real_t)(M_PI);

using VectorFun = std::function<void(const Vector & x, Vector & a)>;
using ScalarFun = std::function<real_t(const Vector & x)>;

#include "skew.hpp"
#include "manu.hpp"

void evaluate1D(Vector &x, Vector &f, GridFunction *gf, int lod)
{
   // Get Mesh and Nodes gridfunction
   Mesh *mesh = gf->FESpace()->GetMesh();
   GridFunction *nodes = mesh->GetNodes();
   if (!nodes)
   {
      nodes = new GridFunction(gf->FESpace());
      mesh->GetNodes(*nodes);
   }

   // Evaluate
   std::list<pair<real_t,real_t>> sol;
   Vector vals,coords;
   for (int i = 0; i <  mesh->GetNE(); i++)
   {
      int geom       = mesh->GetElementBaseGeometry(i);
      RefinedGeometry *refined_geo = GlobGeometryRefiner.Refine(( Geometry::Type)geom, 1, lod);

      gf->GetValues(i, refined_geo->RefPts, vals);
      nodes->GetValues(i, refined_geo->RefPts, coords);

      for (int j = 0; j < vals.Size(); j++)
      {
         sol.push_back(std::make_pair(coords[j],vals[j]));
      }
   }

   // Sort and make unique
   sol.sort();
   sol.unique();

   // Convert to Vectors
   x.SetSize(sol.size());
   f.SetSize(sol.size());
   int i = 0;
   for (std::list<pair<real_t,real_t>>::iterator d = sol.begin() ; d != sol.end(); ++d, i++)
   {
      x[i] = d->first;
      f[i] = d->second;
   }
}

int main(int argc, char *argv[])
{
   // 1. Parse command-line options.
   const char *mesh_file = "../../data/inline-quad.mesh";
   const char *ref_file  = "";
   int problem = 0;
   int sstype = 0;
   bool static_cond = false;
   bool visualization = false;
   int lod = 0;
   real_t penalty = -1;
   Array<int> order(1);
   order[0] = 2;
   int ref_levels = 0;

   bool mono = true;

   OptionsParser args(argc, argv);

   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&ref_file, "-rf", "--ref-file",
                  "File with refinement data");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree) or -1 for"
                  " isoparametric space.");
   args.AddOption(&ref_levels, "-r", "--refine",
                  "Number of times to refine the mesh.");
   args.AddOption(&kappa_param , "-k", "--kappa",
                  "Sets the diffusion parameters, should be positive.");
   args.AddOption(&att_param , "-a", "--att",
                  "Sets the velocity direction");
   args.AddOption(&problem, "-p", "--problem",
                  "Select the problem to solve:\n"
                  " 0 = convection skew-to-the mesh\n"
                   "1 = manufactured solution\n");
   args.AddOption(&sstype, "-s", "--stab", " Stabilization type:\n\t"
                  "  Galerkin 0 \n\t"
                  "  SUPG 1\n\t"
                  "  GLS 2\n\t"
                  "  VMS 3\n\t");
   args.AddOption(&mono, "-mo", "--mono", "-co",
                  "--comp",
                  "Use a monolithic integrator or a composed one.");
   args.AddOption(&static_cond, "-sc", "--static-condensation", "-no-sc",
                  "--no-static-condensation", "Enable static condensation.");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&lod, "-lod", "--level-of-detail",
                  "Refinement level for 1D solution output (0 means no output).");
   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(cout);

   }
   args.PrintOptions(mfem::out);

   StabType  stype = (StabType)sstype;
   switch (stype)
   {
      case GALERKIN:
         mfem::out<<"Galerkin formulation"<<endl;
         break;
      case SUPG:
         mfem::out<<"SUPG formulation"<<endl;
         break;
      case GLS:
         mfem::out<<"GLS formulation"<<endl;
         break;
      case VMS:
         mfem::out<<"VMS formulation"<<endl;
         break;
      default:
         mfem::out<<"GAL"<<"\t"<<"SUPG"<<"\t"<<"GLS"<<"\t"<<"VMS"<<endl;
         mfem::out<<GALERKIN<<"\t"<<SUPG<<"\t"<<GLS<<"\t"<<VMS<<endl;
         mfem_error("Wrong formulation");
   }


   // 2. Read the mesh from the given mesh file. We can handle triangular,
   //    quadrilateral, tetrahedral, hexahedral, surface and volume meshes with
   //    the same code.
   Mesh *mesh = new Mesh(mesh_file, 1, 1);
   int dim = mesh->Dimension();

   // 3. Refine the mesh to increase the resolution. In this example we do
   //    'ref_levels' of uniform refinement and knot insertion of knots defined
   //    in a refinement file. We choose 'ref_levels' to be the largest number
   //    that gives a final mesh with no more than 50,000 elements.
   {
      // Mesh refinement as defined in refinement file
      if (mesh->NURBSext && (strlen(ref_file) != 0))
      {
         mesh->RefineNURBSFromFile(ref_file);
      }

      for (int l = 0; l < ref_levels; l++)
      {
         mesh->UniformRefinement();
      }
      mesh->PrintInfo();
   }

   // 4. Define a finite element space on the mesh. Here we use continuous
   //    Lagrange finite elements of the specified order. If order < 1, we
   //    instead use an isoparametric/isogeometric space.
   FiniteElementCollection *fec;
   NURBSExtension *NURBSext = nullptr;
   int own_fec = 1;

   if (mesh->NURBSext)
   {
      fec = new NURBSFECollection(order[0]);

      int nkv = mesh->NURBSext->GetNKV();
      if (order.Size() == 1)
      {
         int tmp = order[0];
         order.SetSize(nkv);
         order = tmp;
      }

      if (order.Size() != nkv ) { mfem_error("Wrong number of orders set."); }
      NURBSext = new NURBSExtension(mesh->NURBSext, order);
   }
   else if (order[0] == -1) // Isoparametric
   {
      if (mesh->GetNodes())
      {
         fec = mesh->GetNodes()->OwnFEC();
         own_fec = 0;
         mfem::out << "Using isoparametric FEs: " << fec->Name() << endl;
      }
      else
      {
         mfem::out <<"Mesh does not have FEs --> Assume order 1.\n";
         fec = new H1_FECollection(1, dim);
      }
   }
   else
   {
      if (order.Size() > 1) { cout <<"Wrong number of orders set, needs one.\n"; }
      fec = new H1_FECollection(abs(order[0]), dim);
   }

   FiniteElementSpace *fespace = new FiniteElementSpace(mesh, NURBSext, fec);
   mfem::out << "Number of finite element unknowns: "
             << fespace->GetTrueVSize() << endl;

   // 5. Determine the list of true (i.e. conforming) essential boundary dofs.
   //    In this example, the boundary conditions are defined by marking all
   //    the boundary attributes from the mesh as essential (Dirichlet) and
   //    converting them to a list of true dofs.
   Array<int> ess_tdof_list;
   if (mesh->bdr_attributes.Size())
   {
      Array<int> ess_bdr(mesh->bdr_attributes.Max());
      ess_bdr = 1;
      fespace->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
   }

   // 6. Set up the linear form b(.) which corresponds to the right-hand side of
   //    the FEM linear system, which in this case is (1,phi_i) where phi_i are
   //    the basis functions in the finite element fespace.
   VectorFunctionCoefficient *adv, *grad;
   FunctionCoefficient *kappa,*force, *sol, *lap;

   if (problem == 0)
   {
      if (mesh->Dimension() != 2) mfem_error("Advection skew to the mesh needs a 2D mesh!");
      adv = new VectorFunctionCoefficient(mesh->Dimension(), skew::adv);
      kappa= new FunctionCoefficient(skew::kappa);

      force = new FunctionCoefficient(skew::force);
      sol = new FunctionCoefficient(skew::sol);
      grad = new VectorFunctionCoefficient(mesh->Dimension(), skew::grad);
      lap = new FunctionCoefficient(skew::laplace);

   }
   else if (problem == 1)
   {
      adv = new VectorFunctionCoefficient(mesh->Dimension(), manufactured::adv);
      kappa= new FunctionCoefficient(manufactured::kappa);

      force = new FunctionCoefficient(manufactured::force);
      sol = new FunctionCoefficient(manufactured::sol);
      grad = new VectorFunctionCoefficient(mesh->Dimension(), manufactured::grad);
      lap = new FunctionCoefficient(manufactured::laplace);
   }
   else 
   {
      mfem_error("Incorrect problem!");
   }

   // 7. Define the solution vector x as a finite element grid function
   //    corresponding to fespace. Initialize x with initial guess of zero,
   //    which satisfies the boundary conditions.
   GridFunction x(fespace);
   x.ProjectCoefficient(*sol);

   if (problem == 1)
   {
      real_t err  = x.ComputeL2Error(*sol);
      real_t gerr = x.ComputeGradError(grad);
      real_t lerr = x.ComputeLaplaceError(lap);

      mfem::out << "|| x_h - x_ex || = " << err  << "\n";
      mfem::out << "|| grad x_h - grad x_ex ||    = " << gerr << "\n";
      mfem::out << "|| lap x_h - lap x_ex ||      = " << lerr << "\n";
   }

   // 9. Assemble the bilinear form and the corresponding linear system,
   //    applying any necessary transformations such as: eliminating boundary
   //    conditions, applying conforming constraints for non-conforming AMR,
   //    static condensation, etc.
   FFH92Tau tau (adv, kappa, fespace);
   StabConDifComposition stab_condif_comp(adv, kappa, force, &tau);

   BilinearForm a(fespace);
   LinearForm b(fespace);

   if (mono)
   {
      a.AddDomainIntegrator(new StabConDifIntegrator(adv, kappa, force, &tau, stype));
      b.AddDomainIntegrator(new StabConDifIntegrator(adv, kappa, force, &tau, stype));
   }
   else
   {
      stab_condif_comp.SetBilinearIntegrators(&a, stype);
      stab_condif_comp.SetLinearIntegrators(&b, stype);
   }

   a.Assemble();
   b.Assemble();

   if (static_cond) { a.EnableStaticCondensation(); }
   SparseMatrix A;
   Vector B, X;

   a.FormLinearSystem(ess_tdof_list, x, b, A, X, B);

   mfem::out << "Size of linear system: " << A.Height() << endl;

#ifndef MFEM_USE_SUITESPARSE
   // 10. Define a simple Jacobi preconditioner and use it to
   //     solve the system A X = B with PCG.
   GSSmoother M(A);
   GMRES(A, M, B, X, 1, 2000, 2000, 1e-16, 0.0);
#else
   // 10. If MFEM was compiled with SuiteSparse, use UMFPACK to solve the system.
   UMFPackSolver umf_solver;
   umf_solver.Control[UMFPACK_ORDERING] = UMFPACK_ORDERING_METIS;
   umf_solver.SetOperator(A);
   umf_solver.Mult(B, X);
#endif

   // 11. Recover the solution as a finite element grid function.
   a.RecoverFEMSolution(X, b, x);

   // 12. Save the refined mesh and the solution. This output can be viewed later
   //     using GLVis: "glvis -m refined.mesh -g sol.gf".
   {
      ofstream mesh_ofs("refined.mesh");
      mesh_ofs.precision(8);
      mesh->Print(mesh_ofs);
      ofstream sol_ofs("sol.gf");
      sol_ofs.precision(8);
      x.Save(sol_ofs);
      sol_ofs.close();
   }

   // 13. Send the solution by socket to a GLVis server.
   if (visualization)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      socketstream sol_sock(vishost, visport);
      sol_sock.precision(8);
      sol_sock << "solution\n" << *mesh << x << flush;
   }

   if (mesh->Dimension() == 1 && lod > 0)
   {
      Vector coord, val;
      evaluate1D(coord, val, &x, lod);

      ofstream sol_ofs("solution.dat");
      for (int i = 0; i < x.Size();i++)
      {
         sol_ofs<<coord[i] <<"\t"<<val[i]<<endl;
      }
      sol_ofs.close();
   }

   // 14. Error computation
   Vector norm(3);
   if (problem == 0)
   {
      norm[0] = x.ComputeL2Error(*sol);
      norm[1] = x.ComputeGradError(grad);
      norm[2] = x.ComputeLaplaceError(lap);

      mfem::out << "|| x_h - x_ex || = " << norm[0]  << "\n";
      mfem::out << "|| grad x_h - grad x_ex ||    = " << norm[1] << "\n";
      mfem::out << "|| lap x_h - lap x_ex ||      = " << norm[2] << "\n";
   }
   else
   {
      norm = -1.0;
   }

   // 15. Save data in the VisIt format
   VisItDataCollection visit_dc("condif", mesh);
   visit_dc.RegisterField("solution", &x);
   visit_dc.Save();

   // 16. Free the used memory.
   delete fespace;
   if (own_fec) { delete fec; }
   delete mesh;
   delete adv, grad;
   delete kappa, force, sol, lap;

   return 0;
}

