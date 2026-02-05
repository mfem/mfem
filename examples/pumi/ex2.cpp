//                                MFEM Example 2
//                              PUMI Modification
//
// Compile with: make ex2
//
// Sample runs:
//    ex2 -m ../../data/pumi/serial/pillbox.smb -p ../../data/pumi/geom/pillbox.dmg
//        -bf ../../data/pumi/serial/boundary.mesh
//
// Note:         Example models + meshes for the PUMI examples can be downloaded
//               from github.com/mfem/data/pumi. After downloading we recommend
//               creating a symbolic link to the above directory in ../../data.
//
// Description:  This example code solves a simple linear elasticity problem
//               describing a multi-material cantilever beam.
//
//               Specifically, we approximate the weak form of -div(sigma(u))=0
//               where sigma(u)=lambda*div(u)*I+mu*(grad*u+u*grad) is the stress
//               tensor corresponding to displacement field u, and lambda and mu
//               are the material Lame constants. The boundary conditions are
//               u=0 on the fixed part of the boundary with attribute 1, and
//               sigma(u).n=f on the remainder with f being a constant pull down
//               vector on boundary elements with attribute 2, and zero
//               otherwise. The geometry of the domain is assumed to be as
//               follows:
//                                        boundary
//                                       attribute 2
//                                       (push down)
//                                            ||
//                                            \/
//                                       +----------+
//                                       |          |
//                                       |          |
//                             +---------| material |----------+
//                boundary --->| material|    2     | material |<--- boundary
//                attribute 1  |    1    |          |    3     |     attribute 1
//                 (fixed)     +---------+----------+----------+     (fixed)
//
//               The example demonstrates the use of high-order and NURBS vector
//               finite element spaces with the linear elasticity bilinear form,
//               meshes with curved elements, and the definition of piece-wise
//               constant and vector coefficient objects. Static condensation is
//               also illustrated.
//
//               We recommend viewing Example 1 before viewing this example.
//
// NOTE:         Model/Mesh files for this example are in the (large) data file
//               repository of MFEM here https://github.com/mfem/data under the
//               folder named "pumi", which consists of the following sub-folders:
//               a) geom -->  model files
//               b) parallel --> parallel pumi mesh files
//               c) serial --> serial pumi mesh files


#include "mfem.hpp"
#include <fstream>
#include <iostream>

#include "../../general/text.hpp"

#ifdef MFEM_USE_SIMMETRIX
#include <SimUtil.h>
#include <gmi_sim.h>
#endif
#include <apfMDS.h>
#include <gmi_null.h>
#include <PCU.h>
#include <apfConvert.h>
#include <gmi_mesh.h>
#include <crv.h>

#ifndef MFEM_USE_PUMI
#error This example requires that MFEM is built with MFEM_USE_PUMI=YES
#endif

using namespace std;
using namespace mfem;

int main(int argc, char *argv[])
{
   // 1. Initialize MPI (required by PUMI) and HYPRE.
   Mpi::Init(argc, argv);
   Hypre::Init();

   // 2. Parse command-line options.
   const char *mesh_file = "../../data/pumi/serial/pillbox.smb";
   const char *boundary_file = "../../data/pumi/serial/boundary.mesh";
#ifdef MFEM_USE_SIMMETRIX
   const char *model_file = "../../data/pumi/geom/pillbox.smd";
#else
   const char *model_file = "../../data/pumi/geom/pillbox.dmg";
#endif
   int order = 1;
   bool static_cond = false;
   bool visualization = 1;
   int geom_order = 1;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree).");
   args.AddOption(&static_cond, "-sc", "--static-condensation", "-no-sc",
                  "--no-static-condensation", "Enable static condensation.");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&model_file, "-p", "--parasolid",
                  "Parasolid model to use.");
   args.AddOption(&geom_order, "-go", "--geometry_order",
                  "Geometric order of the model");
   args.AddOption(&boundary_file, "-bf", "--txt",
                  "txt file containing boundary tags");
   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(cout);
      return 1;
   }
   args.PrintOptions(cout);

   // 3. Read the SCOREC Mesh.
   PCU_Comm_Init();
#ifdef MFEM_USE_SIMMETRIX
   Sim_readLicenseFile(0);
   gmi_sim_start();
   gmi_register_sim();
#endif
   gmi_register_mesh();

   apf::Mesh2* pumi_mesh;
   pumi_mesh = apf::loadMdsMesh(model_file, mesh_file);

   // 4. Increase the geometry order if necessary.
   if (geom_order > 1)
   {
      crv::BezierCurver bc(pumi_mesh, geom_order, 0);
      bc.run();
   }
   pumi_mesh->verify();

   // Read boundary
   string bdr_tags;
   named_ifgzstream input_bdr(boundary_file);
   input_bdr >> ws;
   getline(input_bdr, bdr_tags);
   filter_dos(bdr_tags);
   cout << " the boundary tag is : " << bdr_tags << endl;
   Array<int> Dirichlet;
   int numOfent;
   if (bdr_tags == "Dirichlet")
   {
      input_bdr >> numOfent;
      cout << " num of Dirichlet bdr conditions : " << numOfent << endl;
      Dirichlet.SetSize(numOfent);
      for (int kk = 0; kk < numOfent; kk++)
      {
         input_bdr >> Dirichlet[kk];
      }
   }
   Dirichlet.Print();

   Array<int> load_bdr;
   skip_comment_lines(input_bdr, '#');
   input_bdr >> bdr_tags;
   filter_dos(bdr_tags);
   cout << " the boundary tag is : " << bdr_tags << endl;
   if (bdr_tags == "Load")
   {
      input_bdr >> numOfent;
      load_bdr.SetSize(numOfent);
      cout << " num of load bdr conditions : " << numOfent << endl;
      for (int kk = 0; kk < numOfent; kk++)
      {
         input_bdr >> load_bdr[kk];
      }
   }
   load_bdr.Print();

   // 5. Create the MFEM mesh object from the PUMI mesh. We can handle triangular
   //    and tetrahedral meshes. Other inputs are the same as MFEM default
   //    constructor.
   Mesh *mesh = new PumiMesh(pumi_mesh, 1, 1);
   int dim = mesh->Dimension();

   //  Boundary conditions hack.
   apf::MeshIterator* itr = pumi_mesh->begin(dim-1);
   apf::MeshEntity* ent ;
   int bdr_cnt = 0;
   while ((ent = pumi_mesh->iterate(itr)))
   {
      apf::ModelEntity *me = pumi_mesh->toModel(ent);
      if (pumi_mesh->getModelType(me) == (dim-1))
      {
         // Everywhere 3 as initial
         (mesh->GetBdrElement(bdr_cnt))->SetAttribute(3);
         int tag = pumi_mesh->getModelTag(me);
         if (Dirichlet.Find(tag) != -1)
         {
            // Dirichlet attr -> 1
            (mesh->GetBdrElement(bdr_cnt))->SetAttribute(1);
         }
         else if (load_bdr.Find(tag) != -1)
         {
            // Load attr -> 2
            (mesh->GetBdrElement(bdr_cnt))->SetAttribute(2);
         }
         bdr_cnt++;
      }
   }
   pumi_mesh->end(itr);

   // Assign attributes for elements.
   double ppt[3];
   Vector cent(ppt, dim);
   for (int el = 0; el < mesh->GetNE(); el++)
   {
      (mesh->GetElementTransformation(el))->
      Transform(Geometries.GetCenter(mesh->GetElementBaseGeometry(el)),cent);
      if (cent(0) <= -0.05)
      {
         mesh->SetAttribute(el, 1);
      }
      else if (cent(0) >= 0.05)
      {
         mesh->SetAttribute(el, 2);
      }
      else
      {
         mesh->SetAttribute(el, 3);
      }
   }
   mesh->SetAttributes();
   if (mesh->attributes.Max() < 2 || mesh->bdr_attributes.Max() < 2)
   {
      cerr << "\nInput mesh should have at least two materials and "
           << "two boundary attributes! (See schematic in ex2.cpp)\n"
           << endl;
      return 3;
   }

   // 6. Refine the mesh to increase the resolution. In this example we do
   //    'ref_levels' of uniform refinement. We choose 'ref_levels' to be the
   //    largest number that gives a final mesh with no more than 5,000
   //    elements.
   {
      int ref_levels =
         (int)floor(log(5000./mesh->GetNE())/log(2.)/dim);
      for (int l = 0; l < ref_levels; l++)
      {
         mesh->UniformRefinement();
      }
   }

   // 7. Define a finite element space on the mesh. Here we use vector finite
   //    elements, i.e. dim copies of a scalar finite element space. The vector
   //    dimension is specified by the last argument of the FiniteElementSpace
   //    constructor. For NURBS meshes, we use the (degree elevated) NURBS space
   //    associated with the mesh nodes.
   FiniteElementCollection *fec;
   FiniteElementSpace *fespace;
   if (mesh->NURBSext)
   {
      fec = NULL;
      fespace = mesh->GetNodes()->FESpace();
   }
   else
   {
      fec = new H1_FECollection(order, dim);
      fespace = new FiniteElementSpace(mesh, fec, dim);
   }
   cout << "Number of finite element unknowns: " << fespace->GetTrueVSize()
        << endl << "Assembling: " << flush;

   // 8. Determine the list of true (i.e. conforming) essential boundary dofs.
   //    In this example, the boundary conditions are defined by marking only
   //    boundary attribute 1 from the mesh as essential and converting it to a
   //    list of true dofs.
   Array<int> ess_tdof_list, ess_bdr(mesh->bdr_attributes.Max());
   ess_bdr = 0;
   ess_bdr[0] = 1;
   fespace->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);

   // 9. Set up the linear form b(.) which corresponds to the right-hand side of
   //    the FEM linear system. In this case, b_i equals the boundary integral
   //    of f*phi_i where f represents a "pull down" force on the Neumann part
   //    of the boundary and phi_i are the basis functions in the finite element
   //    fespace. The force is defined by the VectorArrayCoefficient object f,
   //    which is a vector of Coefficient objects. The fact that f is non-zero
   //    on boundary attribute 2 is indicated by the use of piece-wise constants
   //    coefficient for its last component.
   VectorArrayCoefficient f(dim);
   for (int i = 0; i < dim-1; i++)
   {
      f.Set(i, new ConstantCoefficient(0.0));
   }
   {
      Vector pull_force(mesh->bdr_attributes.Max());
      pull_force = 0.0;
      pull_force(1) = -3.0e-2;
      f.Set(dim-1, new PWConstCoefficient(pull_force));
      f.Set(dim-2, new PWConstCoefficient(pull_force));
   }

   LinearForm *b = new LinearForm(fespace);
   b->AddBoundaryIntegrator(new VectorBoundaryLFIntegrator(f));
   cout << "r.h.s. ... " << flush;
   b->Assemble();

   // 10. Define the solution vector x as a finite element grid function
   //     corresponding to fespace. Initialize x with initial guess of zero,
   //     which satisfies the boundary conditions.
   GridFunction x(fespace);
   x = 0.0;

   // 11. Set up the bilinear form a(.,.) on the finite element space
   //     corresponding to the linear elasticity integrator with piece-wise
   //     constants coefficient lambda and mu.
   Vector lambda(mesh->attributes.Max());
   lambda = 1.0;
   lambda(0) = lambda(1)*10;
   lambda(1) = lambda(1)*100;
   PWConstCoefficient lambda_func(lambda);
   Vector mu(mesh->attributes.Max());
   mu = 1.0;
   mu(0) = mu(1)*10;
   mu(1) = mu(1)*100;
   PWConstCoefficient mu_func(mu);

   BilinearForm *a = new BilinearForm(fespace);
   a->AddDomainIntegrator(new ElasticityIntegrator(lambda_func,mu_func));

   // 12. Assemble the bilinear form and the corresponding linear system,
   //     applying any necessary transformations such as: eliminating boundary
   //     conditions, applying conforming constraints for non-conforming AMR,
   //     static condensation, etc.
   cout << "matrix ... " << flush;
   if (static_cond) { a->EnableStaticCondensation(); }
   a->Assemble();

   SparseMatrix A;
   Vector B, X;
   a->FormLinearSystem(ess_tdof_list, x, *b, A, X, B);
   cout << "done." << endl;

   cout << "Size of linear system: " << A.Height() << endl;

#ifndef MFEM_USE_SUITESPARSE
   // 13. Define a simple symmetric Gauss-Seidel preconditioner and use it to
   //     solve the system Ax=b with PCG.
   GSSmoother M(A);
   PCG(A, M, B, X, 1, 500, 1e-8, 0.0);
#else
   // 13. If MFEM was compiled with SuiteSparse, use UMFPACK to solve the system.
   UMFPackSolver umf_solver;
   umf_solver.Control[UMFPACK_ORDERING] = UMFPACK_ORDERING_METIS;
   umf_solver.SetOperator(A);
   umf_solver.Mult(B, X);
#endif

   // 14. Recover the solution as a finite element grid function.
   a->RecoverFEMSolution(X, *b, x);

   // 15. For non-NURBS meshes, make the mesh curved based on the finite element
   //     space. This means that we define the mesh elements through a fespace
   //     based transformation of the reference element. This allows us to save
   //     the displaced mesh as a curved mesh when using high-order finite
   //     element displacement field. We assume that the initial mesh (read from
   //     the file) is not higher order curved mesh compared to the chosen FE
   //     space.
   if (!mesh->NURBSext)
   {
      mesh->SetNodalFESpace(fespace);
   }

   // 16. Save the displaced mesh and the inverted solution (which gives the
   //     backward displacements to the original grid). This output can be
   //     viewed later using GLVis: "glvis -m displaced.mesh -g sol.gf".
   {
      GridFunction *nodes = mesh->GetNodes();
      *nodes += x;
      x *= -1;
      ofstream mesh_ofs("displaced.mesh");
      mesh_ofs.precision(8);
      mesh->Print(mesh_ofs);
      ofstream sol_ofs("sol.gf");
      sol_ofs.precision(8);
      x.Save(sol_ofs);
   }

   // 17. Send the above data by socket to a GLVis server. Use the "n" and "b"
   //     keys in GLVis to visualize the displacements.
   if (visualization)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      socketstream sol_sock(vishost, visport);
      sol_sock.precision(8);
      sol_sock << "solution\n" << *mesh << x << flush;
   }

   // 18. Free the used memory.
   delete a;
   delete b;
   if (fec)
   {
      delete fespace;
      delete fec;
   }
   delete mesh;

   pumi_mesh->destroyNative();
   apf::destroyMesh(pumi_mesh);
   PCU_Comm_Free();
#ifdef MFEM_USE_SIMMETRIX
   gmi_sim_stop();
   Sim_unregisterAllKeys();
#endif

   return 0;
}
