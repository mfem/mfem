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
// Stabilized  Navier-Stokes

#include "stab_condif.hpp"
#include "stab_navsto.hpp"
#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include <list>

using namespace std;
using namespace mfem;

real_t kappa_param = 1.0;
real_t pi  = (real_t)(M_PI);

using VectorFun = std::function<void(const Vector & x, Vector & a)>;
using ScalarFun = std::function<real_t(const Vector & x)>;


void sol_fun(const Vector & x, Vector &sol)
{
   sol = 0.0;
   if ((x[1] - 0.99 > 0.0) &&
       (fabs(x[0] - 0.5) < 0.49) )
   {
      sol[0] = 1.0;
   }
}

real_t kappa_fun(const Vector & x)
{
   return kappa_param;
}

void force_fun(const Vector & x, Vector &f)
{
   f = 0.0;
}

StabType GetStabilisationType(int stype)
{
   switch (stype)
   {
      case GALERKIN:
         mfem::out<<"Galerkin formulation"<<std::endl;
         break;
      case SUPG:
         mfem::out<<"SUPG formulation"<<std::endl;
         break;
      case GLS:
         mfem::out<<"GLS formulation"<<std::endl;
         break;
      case VMS:
         mfem::out<<"VMS formulation"<<std::endl;
         break;
      default:
         mfem::out<<"GAL"<<"\t"<<"SUPG"<<"\t"<<"GLS"<<"\t"<<"VMS"<<std::endl;
         mfem::out<<GALERKIN<<"\t"<<SUPG<<"\t"<<GLS<<"\t"<<VMS<<std::endl;
         mfem_error("Wrong formulation");
   }
   return (StabType) stype;
}

int main(int argc, char *argv[])
{
   // 1. Parse command-line options.
   const char *mesh_file = "../../data/inline-quad.mesh";
   const char *ref_file  = "";
   int problem = 0;
   int sstype = -2;
   bool static_cond = false;
   bool visualization = false;

   real_t penalty = -1;
   int order = 1;
   int ref_levels = 0;

   bool mono = true;

   OptionsParser args(argc, argv);

   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&ref_file, "-rf", "--ref-file",
                  "File with refinement data");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order isoparametric space.");
   args.AddOption(&ref_levels, "-r", "--refine",
                  "Number of times to refine the mesh.");
   args.AddOption(&kappa_param , "-k", "--kappa",
                  "Sets the diffusion parameters, should be positive.");

   args.AddOption(&problem, "-p", "--problem",
                  "Select the problem to solve:\n\t"
                  "  0 = convection skew-to-the mesh\n\t"
                  "  1 = manufactured solution\n");
   args.AddOption(&sstype, "-s", "--stab", " Stabilization type:\n\t"
                  " -2 = Galerkin\n\t"
                  " -1 = GLS\n\t"
                  "  0 = SUPG\n\t"
                  "  1 = VMS\n");
   args.AddOption(&mono, "-mo", "--mono", "-co",
                  "--comp",
                  "Use a monolithic integrator or a composed one.");
   args.AddOption(&static_cond, "-sc", "--static-condensation", "-no-sc",
                  "--no-static-condensation", "Enable static condensation.");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(cout);

   }
   args.PrintOptions(mfem::out);

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
   FiniteElementCollection *fec_u = new H1_FECollection(order+1, dim);
   FiniteElementCollection *fec_p = new H1_FECollection(order, dim);

   FiniteElementSpace *fespace_u = new FiniteElementSpace(mesh, fec_u, dim);
   FiniteElementSpace *fespace_p = new FiniteElementSpace(mesh, fec_p);
   
   Array<FiniteElementSpace *> spaces(2);
   spaces[0] = fespace_u;
   spaces[1] = fespace_p;

   mfem::out << "Number of finite element unknowns:\n"
             << "\tVelocity = "<<fespace_u->GetTrueVSize() << endl
             << "\tPressure = "<<fespace_p->GetTrueVSize() << endl;
   // 5. Determine the list of true (i.e. conforming) essential boundary dofs.
   //    In this example, the boundary conditions are defined by marking all
   //    the boundary attributes from the mesh as essential (Dirichlet) and
   //    converting them to a list of true dofs.
   Array<Array<int> *> ess_bdr(2);
   Array<int> ess_tdof_list;

   Array<int> ess_bdr_u(spaces[0]->GetMesh()->bdr_attributes.Max());
   Array<int> ess_bdr_p(spaces[1]->GetMesh()->bdr_attributes.Max());

   ess_bdr_p = 0;
   ess_bdr_u = 1;
  // ess_bdr_u[0] = 1;
  // ess_bdr_u[1] = 1;

   ess_bdr[0] = &ess_bdr_u;
   ess_bdr[1] = &ess_bdr_p;

   // 6. Set up the linear form b(.) which corresponds to the right-hand side of
   //    the FEM linear system, which in this case is (1,phi_i) where phi_i are
   //    the basis functions in the finite element fespace.
   VectorFunctionCoefficient *force, *sol;
   FunctionCoefficient *kappa;

   if (mesh->Dimension() != 2) mfem_error("Advection skew to the mesh needs a 2D mesh!");

   kappa= new FunctionCoefficient(kappa_fun);

   force = new VectorFunctionCoefficient(dim, force_fun);
   sol = new VectorFunctionCoefficient(dim, sol_fun);

   // 7. Define the solution vector x as a finite element grid function
   //    corresponding to fespace. Initialize x with initial guess of zero,
   //    which satisfies the boundary conditions.
   Array<int> block_trueOffsets(3);
   block_trueOffsets[0] = 0;
   block_trueOffsets[1] = fespace_u->GetTrueVSize();
   block_trueOffsets[2] = fespace_p->GetTrueVSize();
   block_trueOffsets.PartialSum();

   BlockVector xp(block_trueOffsets);

   GridFunction x_u(fespace_u);
   GridFunction x_p(fespace_p);

   x_u.MakeTRef(fespace_u, xp.GetBlock(0), 0);
   x_p.MakeTRef(fespace_p, xp.GetBlock(1), 0);

   x_u.ProjectCoefficient(*sol);
   x_p = 0.0;

   x_u.SetTrueVector();
   x_p.SetTrueVector();
   
   {
      VisItDataCollection visit_dc("navsto", mesh);
      visit_dc.RegisterField("u", &x_u);
      visit_dc.RegisterField("p", &x_p);
      visit_dc.Save();
   }

   // 10. Initialize the incompressible neo-Hookean operator
   real_t newton_rel_tol = 1e-8;
   real_t newton_abs_tol = 1e-8;
   int newton_iter = 10;

   StabInNavStoOperator oper(spaces, ess_bdr, block_trueOffsets,
                             newton_rel_tol, newton_abs_tol, newton_iter, *kappa);

   // 11. Solve the Newton system
   oper.Solve(xp);

   // 14. Save data in the VisIt format
   {
      VisItDataCollection visit_dc("navsto", mesh);
      visit_dc.RegisterField("u", &x_u);
      visit_dc.RegisterField("p", &x_p);
      visit_dc.Save();
   }

   // 16. Free the used memory.
   delete fespace_u, fespace_p;
   //delete fec_u, fec_p;
   delete mesh;
   delete kappa, force, sol;

   return 0;
}

