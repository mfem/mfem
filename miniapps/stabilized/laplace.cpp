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

bool biharm = true;
real_t kappa_param = 1.0;

real_t dif_fun(const Vector & x)
{
   return kappa_param;
}

//----------------------------------------------------------
real_t force_fun(const Vector & x)
{
   int d = x.Size();

   real_t kappa = dif_fun(x);

   real_t pi  = (real_t)(M_PI);

   real_t sx = sin(pi*x[0]);
   real_t sy = 1.0;
   real_t sz = 1.0;

   if (d >= 2)
   {
      sy = sin(pi*x[1]);
   }
   if (d >= 3)
   {
      sz = sin(pi*x[2]);
   }
  if (biharm) return  d*d*kappa*pi*pi*pi*pi*sx*sy*sz;
  return  d*kappa*pi*pi*sx*sy*sz;
}

//----------------------------------------------------------
real_t exact_fun(const Vector & x)
{
   real_t pi  = (real_t)(M_PI);

   real_t sx = sin(pi*x[0]);
   real_t sy = 1.0;
   real_t sz = 1.0;

   int d = x.Size();
   if (d >= 2)
   {
      sy = sin(pi*x[1]);
   }
   if (d >= 3)
   {
      sz = sin(pi*x[2]);
   }

   return sx*sy*sz;
}

//----------------------------------------------------------
void gexact_fun(const Vector & x, Vector & a)
{
   real_t pi  = (real_t)(M_PI);

   real_t sx = sin(pi*x[0]);
   real_t cx = cos(pi*x[0]);
   real_t sy = 1.0;
   real_t cy = 1.0;
   real_t sz = 1.0;
   real_t cz = 1.0;

   int d = x.Size();
   if (d >= 2)
   {
      sy = sin(pi*x[1]);
      cy = cos(pi*x[1]);
   }
   if (d >= 3)
   {
      sz = sin(pi*x[2]);
      cz = cos(pi*x[2]);
   }

   a[0] = pi*cx*sy;
   a[1] = pi*sx*cy;
}

//----------------------------------------------------------
real_t lexact_fun(const Vector & x)
{
   real_t pi  = (real_t)(M_PI);

   real_t sx = sin(pi*x[0]);
   real_t cx = cos(pi*x[0]);
   real_t sy = 1.0;
   real_t cy = 1.0;
   real_t sz = 1.0;
   real_t cz = 1.0;

   int d = x.Size();
   if (d >= 2)
   {
      sy = sin(pi*x[1]);
      cy = cos(pi*x[1]);
   }
   if (d >= 3)
   {
      sz = sin(pi*x[2]);
      cz = cos(pi*x[2]);
   }

   return -d*pi*pi*sx*sy*sz;
}


//----------------------------------------------------------
void evaluate1D(Vector &x, Vector &f, GridFunction *gf, int lod);

Vector compute(int argc, char *argv[], int er = 0);

//----------------------------------------------------------
int main(int argc, char *argv[])
{
   int stages = 6;
   Vector normL2(stages);
   Vector normH1(stages);
   Vector normH2(stages); // -- Laplacian


   for (int s = 0; s < stages; s++)
   {
      Vector norm = compute(argc, argv, s);
      normL2[s] = norm[0];
      normH1[s] = norm[1];
      normH2[s] = norm[2];
      if (s == 0)
      {
         cout<<"\n\n";
         cout<<"---------------------------------------------------------------------------------------------\n";
         cout<<" norm L2        | order         || norm H1      | order         || norm H2      | order      \n";
         cout<<"---------------------------------------------------------------------------------------------\n";
         cout<<normL2[0]<<"\t| "<<"   --   "<<"\t|| ";
         cout<<normH1[0]<<"\t| "<<"   --   "<<"\t|| ";
         cout<<normH2[0]<<"\t| "<<"   --   "<<endl;
         mfem::out.Disable();
      }
      else
      {
         real_t orderL2 = log(normL2[s-1]/normL2[s])/log(2.0);
         real_t orderH1 = log(normH1[s-1]/normH1[s])/log(2.0);
         real_t orderH2 = log(normH2[s-1]/normH2[s])/log(2.0);
         cout<<normL2[s]<<"\t| "<<orderL2<<"\t|| ";
         cout<<normH1[s]<<"\t| "<<orderH1<<"\t|| ";
         cout<<normH2[s]<<"\t| "<<orderH2<<endl;
      }
   }
   return 0;
}

//----------------------------------------------------------
Vector compute(int argc, char *argv[], int er)
{
   // 1. Parse command-line options.
   const char *mesh_file = "../../data/square-nurbs.mesh";
   const char *per_file  = "none";
   const char *ref_file  = "";
   int ref_levels = 0;
   Array<int> master(0);
   Array<int> slave(0);
   bool static_cond = false;
   bool visualization = false;
   int lod = 0;
   real_t penalty = -1;
   Array<int> order(1);
   order[0] = 2;
   //real_t kappa_param = -1;
   int sstype = 0;

   OptionsParser args(argc, argv);
   args.AddOption(&biharm, "-b", "--biharm", "-l",
                  "--laplace",
                  "Biharm or Laplace.");
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&ref_levels, "-r", "--refine",
                  "Number of times to refine the mesh uniformly, -1 for auto.");
   args.AddOption(&per_file, "-p", "--per",
                  "Periodic BCS file.");
   args.AddOption(&ref_file, "-rf", "--ref-file",
                  "File with refinement data");
   args.AddOption(&master, "-pm", "--master",
                  "Master boundaries for periodic BCs");
   args.AddOption(&slave, "-ps", "--slave",
                  "Slave boundaries for periodic BCs");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree) or -1 for"
                  " isoparametric space.");
   args.AddOption(&kappa_param , "-k", "--kappa",
                  "Sets the diffusion parameters, should be positive."
                  " Negative values are replaced with function defined in source.");
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
     // return 1;
   }

   args.PrintOptions(mfem::out);

ref_levels += er;


  // if (order.Min()< 2)
  // {
  //    mfem_error("Wrong order."); 
  // }//
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

      /*if (ref_levels < 0)
      {
         ref_levels =
            (int)floor(log(5000./mesh->GetNE())/log(2.)/dim);
      }*/
      if (ref_levels < 0)ref_levels = 0;
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
   NURBSExtension *NURBSext = NULL;
   int own_fec = 0;

   if (mesh->NURBSext)
   {
      fec = new NURBSFECollection(order[0]);
      own_fec = 1;

      int nkv = mesh->NURBSext->GetNKV();
      if (order.Size() == 1)
      {
         int tmp = order[0];
         order.SetSize(nkv);
         order = tmp;
      }

      if (order.Size() != nkv ) { mfem_error("Wrong number of orders set."); }
      NURBSext = new NURBSExtension(mesh->NURBSext, order);

      // Read periodic BCs from file
      std::ifstream in;
      in.open(per_file, std::ifstream::in);
      if (in.is_open())
      {
         int psize;
         in >> psize;
         master.SetSize(psize);
         slave.SetSize(psize);
         master.Load(in, psize);
         slave.Load(in, psize);
         in.close();
      }
      master.Print();
      slave.Print();
      NURBSext->ConnectBoundaries(master,slave);
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
         own_fec = 1;
      }
   }
   else
   {
      if (order.Size() > 1) { cout <<"Wrong number of orders set, needs one.\n"; }
      fec = new H1_FECollection(abs(order[0]), dim);
      own_fec = 1;
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

      // Remove periodic BCs
      for (int i = 0; i < master.Size(); i++)
      {
         ess_bdr[master[i]-1] = 0;
         ess_bdr[slave[i]-1] = 0;
      }
      fespace->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
   }

   // 6. Set up the linear form b(.) which corresponds to the right-hand side of
   //    the FEM linear system, which in this case is (1,phi_i) where phi_i are
   //    the basis functions in the finite element fespace.
   ConstantCoefficient u_dir(0.0);

   Coefficient *kappa_tmp;
   if (kappa_param < 0.0)
   {
      kappa_tmp = new FunctionCoefficient(dif_fun);
   }
   else
   {
      kappa_tmp = new ConstantCoefficient(kappa_param);
   }

   Coefficient& kappa = *kappa_tmp;
   FunctionCoefficient force(force_fun);

   LinearForm *b = new LinearForm(fespace);
   b->AddDomainIntegrator(new DomainLFIntegrator(force));
   b->Assemble();

   // 7. Define the solution vector x as a finite element grid function
   //    corresponding to fespace. Initialize x with initial guess of zero,
   //    which satisfies the boundary conditions.
   GridFunction x(fespace);
   x = 0.0;

   // 8. Set up the bilinear form a(.,.) on the finite element space
   //    corresponding to the Laplacian operator -Delta, by adding the Diffusion
   //    domain integrator.
   BilinearForm *a = new BilinearForm(fespace);
   if (biharm)
   {
      a->AddDomainIntegrator(new LaplaceLaplaceIntegrator(kappa));
   }
   else
   {
      a->AddDomainIntegrator(new DiffusionIntegrator(kappa));
   }

   // 9. Assemble the bilinear form and the corresponding linear system,
   //    applying any necessary transformations such as: eliminating boundary
   //    conditions, applying conforming constraints for non-conforming AMR,
   //    static condensation, etc.
   if (static_cond) { a->EnableStaticCondensation(); }
   a->Assemble();

   SparseMatrix A;
   Vector B, X;
   a->FormLinearSystem(ess_tdof_list, x, *b, A, X, B);

   mfem::out << "Size of linear system: " << A.Height() << endl;

//#ifndef MFEM_USE_SUITESPARSE
   // 10. Define a simple Jacobi preconditioner and use it to
   //     solve the system A X = B with PCG.
   GSSmoother M(A);
   GMRES(A, M, B, X, 1, 2000, 2000, 1e-16, 0.0);
/*#else
   // 10. If MFEM was compiled with SuiteSparse, use UMFPACK to solve the system.
   UMFPackSolver umf_solver;
   umf_solver.Control[UMFPACK_ORDERING] = UMFPACK_ORDERING_METIS;
   umf_solver.SetOperator(A);
   umf_solver.Mult(B, X);
#endif*/

   // 11. Recover the solution as a finite element grid function.
   a->RecoverFEMSolution(X, *b, x);

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
   int order_quad = 3*order.Max() + 4;
   const IntegrationRule *irs[Geometry::NumGeom];
   for (int i=0; i < Geometry::NumGeom; ++i)
   {
      irs[i] = &(IntRules.Get(i, order_quad));
   }
   FunctionCoefficient xExact(exact_fun);
   real_t err  = x.ComputeL2Error(xExact, irs);
   real_t norm = ComputeLpNorm(2., xExact, *mesh, irs);

   mfem::out << "|| x_h - x_ex || / || x_ex || = " << err / norm << "\n";

   VectorFunctionCoefficient gExact(mesh->Dimension(), gexact_fun);
   real_t gerr = x.ComputeGradError(&gExact, irs);
   mfem::out << "|| grad x_h - grad x_ex ||    = " << gerr << "\n";

   FunctionCoefficient lExact(lexact_fun);
   real_t lerr = x.ComputeLaplaceError(&lExact, irs);
   mfem::out << "|| lap x_h - lap x_ex ||      = " << lerr << "\n";


   // 15. Save data in the VisIt format
   VisItDataCollection visit_dc("condif", mesh);
   visit_dc.RegisterField("solution", &x);
   visit_dc.Save();

   // 16. Free the used memory.
   delete fespace;
   if (own_fec) { delete fec; }
   delete mesh;

   Vector rtn(3);
   rtn[0] = err;
   rtn[1] = gerr;
   rtn[2] = lerr;

   return rtn;
}

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
