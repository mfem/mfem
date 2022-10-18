//                       MFEM Example 2 - Parallel Version
//

// mpirun -np 6 ./amg_tests -lambda 100.0 -mu 100.0 -pr 4 -sx 50
// hypre iterations: 126, 148, 168, 179, 203

// mpirun -np 6 ./amg_tests -lambda 100.0 -mu 100.0 -pr 4 -sx 50 -elast
// hypre iterations: 100, 97, 120, 118, 452

#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

int main(int argc, char *argv[])
{
   // 1. Initialize MPI and HYPRE.
   Mpi::Init(argc, argv);
   int num_procs = Mpi::WorldSize();
   int myid = Mpi::WorldRank();
   Hypre::Init();

   // 2. Parse command-line options.
   const char *mesh_file = "../data/beam-hex.mesh";
   int order = 1;
   bool static_cond = false;
   bool visualization = 1;
   double lambda = 1.0;
   double mu = 1.0;
   bool amg_elast = 0;
   int sref = 0;
   int pref = 0;
   double sx = 1.0;
   double sy = 1.0;
   double sz = 1.0;
   bool reorder_space = false;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree).");
   args.AddOption(&sref, "-sr", "--sref",
                  "Number of serial refinements");
   args.AddOption(&pref, "-pr", "--pref",
                  "Number of parallel refinements");
   args.AddOption(&lambda, "-lambda", "--lambda",
                  "Lame constant λ");
   args.AddOption(&mu, "-mu", "--mu",
                  "Lame constant μ");
   args.AddOption(&sx, "-sx", "--sx",
                  "Length in the x direction");
   args.AddOption(&sy, "-sy", "--sy",
                  "Length in the y direction");
   args.AddOption(&sz, "-sz", "--sz",
                  "Length in the z direction");
   args.AddOption(&amg_elast, "-elast", "--amg-for-elasticity", "-sys",
                  "--amg-for-systems",
                  "Use the special AMG elasticity solver (GM/LN approaches), "
                  "or standard AMG for systems (unknown approach).");
   args.AddOption(&reorder_space, "-nodes", "--by-nodes", "-vdim", "--by-vdim",
                  "Use byNODES ordering of vector space instead of byVDIM");
   args.AddOption(&static_cond, "-sc", "--static-condensation", "-no-sc",
                  "--no-static-condensation", "Enable static condensation.");
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
      return 1;
   }
   if (myid == 0)
   {
      args.PrintOptions(cout);
   }


   Mesh mesh = Mesh::MakeCartesian3D((int)sx, (int)sy, int(sz),
                                     mfem::Element::HEXAHEDRON,
                                     sx,sy,sz);

   int dim = mesh.Dimension();

   //set attributes
   for (int i = 0; i<mesh.GetNBE(); i++)
   {
      Element * be = mesh.GetBdrElement(i);
      Array<int> vertices;
      be->GetVertices(vertices);

      double * coords0 = mesh.GetVertex(vertices[0]);
      double * coords1 = mesh.GetVertex(vertices[1]);
      double * coords2 = mesh.GetVertex(vertices[2]);
      double * coords3 = mesh.GetVertex(vertices[3]);



      Vector center(3);
      center(0) = 0.25*(coords0[0] + coords1[0] + coords2[0] + coords3[0]);

      if (abs(center(0) - 0.0) < 1e-10)
      {
         // the left face
         be->SetAttribute(1);
      }
      else if (abs(center(0) - sx) < 1e-10)
      {
         // the right face
         be->SetAttribute(2);
      }
      else
      {
         // all other boundaries
         be->SetAttribute(3);
      }
   }
   mesh.SetAttributes();


   for (int l = 0; l < sref; l++)
   {
      mesh.UniformRefinement();
   }

   ParMesh *pmesh = new ParMesh(MPI_COMM_WORLD, mesh);
   mesh.Clear();

   FiniteElementCollection *fec = new H1_FECollection(order, dim);
   ParFiniteElementSpace *fespace;
   if (reorder_space)
   {
      fespace = new ParFiniteElementSpace(pmesh, fec, dim, Ordering::byNODES);
   }
   else
   {
      fespace = new ParFiniteElementSpace(pmesh, fec, dim, Ordering::byVDIM);
   }

   HYPRE_BigInt size = fespace->GlobalTrueVSize();
   if (myid == 0)
   {
      cout << "Number of finite element unknowns: " << size << endl
           << "Assembling: " << flush;
   }

   Array<int> ess_tdof_list, ess_bdr(pmesh->bdr_attributes.Max());
   ess_bdr = 0;
   ess_bdr[0] = 1;

   VectorArrayCoefficient f(dim);
   for (int i = 0; i < dim-1; i++)
   {
      f.Set(i, new ConstantCoefficient(0.0));
   }
   {
      Vector pull_force(pmesh->bdr_attributes.Max());
      pull_force = 0.0;
      pull_force(1) = -1.0e-2;
      f.Set(dim-1, new PWConstCoefficient(pull_force));
   }

   ParLinearForm *b = new ParLinearForm(fespace);
   b->AddBoundaryIntegrator(new VectorBoundaryLFIntegrator(f));

   ParGridFunction x(fespace);
   x = 0.0;

   ConstantCoefficient lambda_cf(lambda);
   ConstantCoefficient mu_cf(mu);

   ParBilinearForm *a = new ParBilinearForm(fespace);
   a->AddDomainIntegrator(new ElasticityIntegrator(lambda_cf, mu_cf));

   if (static_cond) { a->EnableStaticCondensation(); }
   for (int i = 0; i<=pref; i++)
   {
      a->Assemble();
      b->Assemble();
      fespace->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);

      HypreParMatrix A;
      Vector B, X;
      a->FormLinearSystem(ess_tdof_list, x, *b, A, X, B);
      if (myid == 0)
      {
         cout << "done." << endl;
         cout << "Size of linear system: " << A.GetGlobalNumRows() << endl;
      }

      HypreBoomerAMG *amg = new HypreBoomerAMG(A);
      if (amg_elast && !a->StaticCondensationIsEnabled())
      {
         amg->SetElasticityOptions(fespace);
      }
      else
      {
         amg->SetSystemsOptions(dim, reorder_space);
      }

      amg->SetPrintLevel(0);
      CGSolver *pcg = new CGSolver(MPI_COMM_WORLD);
      pcg->SetRelTol(1e-8);
      pcg->SetMaxIter(500);
      pcg->SetPrintLevel(3);
      pcg->SetPreconditioner(*amg);
      pcg->SetOperator(A);
      pcg->Mult(B, X);

      // 15. Recover the parallel grid function corresponding to X. This is the
      //     local finite element solution on each processor.
      a->RecoverFEMSolution(X, *b, x);

      pmesh->SetNodalFESpace(fespace);

      GridFunction *nodes = pmesh->GetNodes();
      *nodes += x;
      x *= -1;

      if (visualization)
      {
         char vishost[] = "localhost";
         int  visport   = 19916;
         socketstream sol_sock(vishost, visport);
         sol_sock << "parallel " << num_procs << " " << myid << "\n";
         sol_sock.precision(8);
         sol_sock << "solution\n" << *pmesh << x << flush;
      }
      *nodes += x;

      // 19. Free the used memory.
      delete pcg;
      delete amg;

      if (i == pref)
      {
         break;
      }

      pmesh->UniformRefinement();
      fespace->Update();
      a->Update();
      b->Update();
      x.Update();
   }

   delete a;
   delete b;
   delete fespace;
   delete fec;
   delete pmesh;

   return 0;
}
