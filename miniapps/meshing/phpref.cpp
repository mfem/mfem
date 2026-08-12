//                       Parallel hp-refinement example
//
// Compile with: make phpref
//
// Sample runs:  mpirun -np 4 phpref -dim 2 -n 1000
//               mpirun -np 8 phpref -dim 3 -n 200
//               mpirun -np 8 phpref -dim 3 -n 20 --anisotropic --fixed-order
//
// Description:  This example demonstrates h- and p-refinement in a parallel
//               finite element discretization of the Poisson problem (cf. ex1p)
//               -Delta u = 1 with homogeneous Dirichlet boundary conditions.
//               Refinements are performed iteratively, each iteration having h-
//               or p-refinements on all MPI processes. For simplicity, we
//               randomly choose the elements and the type of refinement, for
//               each iteration. In practice, these choices may be made in a
//               problem-dependent way, but this example serves only to
//               illustrate the capabilities of hp-refinement in parallel.
//
//               We recommend viewing Example 1 before viewing this example.

#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

real_t CheckH1Continuity(ParGridFunction & x);

// Deterministic function for "random" integers.
int DetRand(int & seed)
{
   seed++;
   return int(std::abs(1.0e5 * sin(seed * 1.1234 * M_PI)));
}

void f_exact(const Vector &x, Vector &f);

int main(int argc, char *argv[])
{
   // 1. Initialize MPI and HYPRE.
   Mpi::Init();
   const int num_procs = Mpi::WorldSize();
   const int myid = Mpi::WorldRank();
   Hypre::Init();

   // 2. Parse command-line options.
   int order = 1;
   const char *device_config = "cpu";
   bool visualization = true;
   int numIter = 0;
   int dim = 2;
   bool anisotropic = false;
   bool fixedOrder = false;
   bool deterministic = true;
   bool projectSolution = false;

   OptionsParser args(argc, argv);
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree) or -1 for"
                  " isoparametric space.");
   args.AddOption(&device_config, "-d", "--device",
                  "Device configuration string, see Device::Configure().");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&numIter, "-n", "--num-iter", "Number of hp-ref iterations");
   args.AddOption(&dim, "-dim", "--dim", "Mesh dimension (2 or 3)");
   args.AddOption(&anisotropic, "-aniso", "--anisotropic", "-iso",
                  "--isotropic",
                  "Whether to use anisotropic refinements");
   args.AddOption(&fixedOrder, "-fo", "--fixed-order", "-vo",
                  "--variable-order",
                  "Whether to fix the finite element order on all elements");
   args.AddOption(&deterministic, "-det", "--deterministic", "-not-det",
                  "--not-deterministic",
                  "Use deterministic random refinements");
   args.AddOption(&projectSolution, "-proj", "--project-solution", "-no-proj",
                  "--no-project",
                  "Project a coefficient to solution");

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

   MFEM_VERIFY(!anisotropic || fixedOrder,
               "Variable-order is not supported with anisotropic refinement");

   // 3. Enable hardware devices such as GPUs, and programming models such as
   //    CUDA, OCCA, RAJA and OpenMP based on command line options.
   Device device(device_config);
   if (myid == 0) { device.Print(); }

   // 4. Construct a uniform coarse mesh on all processors.
   Mesh mesh;
   if (dim == 3)
   {
      mesh = Mesh::MakeCartesian3D(2, 2, 2, Element::HEXAHEDRON);
   }
   else
   {
      mesh = Mesh::MakeCartesian2D(2, 2, Element::QUADRILATERAL, true);
   }

   mesh.EnsureNCMesh();

   // 5. Define a parallel mesh by a partitioning of the serial mesh. Refine
   //    this mesh further in parallel to increase the resolution. Once the
   //    parallel mesh is defined, the serial mesh can be deleted.
   ParMesh pmesh(MPI_COMM_WORLD, mesh);
   mesh.Clear();
   {
      int par_ref_levels = 0;
      for (int l = 0; l < par_ref_levels; l++)
      {
         pmesh.UniformRefinement();
      }
   }

   // 6. Define a parallel finite element space on the parallel mesh. Here we
   //    use continuous Lagrange finite elements of the specified order. If
   //    order < 1, we instead use an isoparametric/isogeometric space.
   FiniteElementCollection *fec;
   bool delete_fec;
   if (order > 0)
   {
      fec = new H1_FECollection(order, dim);
      delete_fec = true;
   }
   else if (pmesh.GetNodes())
   {
      fec = pmesh.GetNodes()->OwnFEC();
      delete_fec = false;
      if (myid == 0)
      {
         cout << "Using isoparametric FEs: " << fec->Name() << endl;
      }
   }
   else
   {
      fec = new H1_FECollection(order = 1, dim);
      delete_fec = true;
   }

   const int fespaceDim = projectSolution ? dim : 1;
   ParFiniteElementSpace fespace(&pmesh, fec, fespaceDim);

   // 7. Iteratively perform h- and p-refinements.

   int numH = 0;
   int numP = 0;
   int seed = myid;

   const std::vector<char> hp_char = {'h', 'p'};

   for (int iter=0; iter<numIter; ++iter)
   {
      const int r1 = deterministic ? DetRand(seed) : rand();
      const int r2 = deterministic ? DetRand(seed) : rand();
      const int elem = r1 % pmesh.GetNE();
      int hp = r2 % 2;
      char htype = 7;
      MPI_Bcast(&hp, 1, MPI_INT, 0, MPI_COMM_WORLD);

      if (fixedOrder) { hp = 0; } // Only perform h-refinement
      if (anisotropic)
      {
         const int r3 = deterministic ? DetRand(seed) : rand();
         htype = (r3 % 7) + 1;
      }

      if (myid == 0)
         cout << "hp-refinement iteration " << iter << ": "
              << hp_char[hp] << "-refinement\n";

      if (hp == 1)
      {
         // p-refinement
         Array<pRefinement> refs;
         refs.Append(pRefinement(elem, 1));  // Increase the element order by 1
         fespace.PRefineAndUpdate(refs);
         numP++;
      }
      else
      {
         // h-refinement
         Array<Refinement> refs;
         refs.Append(Refinement(elem, htype));
         if (anisotropic)
         {
            std::set<int> conflicts; // Indices in refs of conflicting elements
            const bool conflict = pmesh.AnisotropicConflict(refs, conflicts);
            if (conflict)
            {
               if (myid == 0)
                  cout << "Anisotropic conflict on iteration " << iter
                       << ", retrying\n";
               iter--;
               continue;
            }
         }

         pmesh.GeneralRefinement(refs);
         fespace.Update(false);
         numH++;
      }
   }

   const HYPRE_BigInt size = fespace.GlobalTrueVSize();
   const int maxP = fespace.GetMaxElementOrder();
   if (myid == 0)
   {
      cout << "Number of finite element unknowns: " << size << endl;
      cout << "Total number of h-refinements: " << numH
           << "\nTotal number of p-refinements: " << numP
           << "\nMaximum order " << maxP << "\n";
   }

   ParGridFunction x(&fespace);
   Vector X;

   if (projectSolution)
   {
      VectorFunctionCoefficient vec_coef(dim, f_exact);
      x.ProjectCoefficient(vec_coef);

      X.SetSize(fespace.GetTrueVSize());

      fespace.GetRestrictionMatrix()->Mult(x, X);
      fespace.GetProlongationMatrix()->Mult(X, x);

      // Compute and print the L^2 norm of the error.
      const real_t error = x.ComputeL2Error(vec_coef);
      if (myid == 0)
      {
         cout << "\n|| E_h - E ||_{L^2} = " << error << '\n' << endl;
      }
   }
   else
   {
      // 8. Determine the list of true (i.e. parallel conforming) essential
      //    boundary dofs. In this example, the boundary conditions are defined
      //    by marking all the boundary attributes from the mesh as essential
      //    (Dirichlet) and converting them to a list of true dofs.
      Array<int> ess_tdof_list;
      if (pmesh.bdr_attributes.Size())
      {
         Array<int> ess_bdr(pmesh.bdr_attributes.Max());
         ess_bdr = 1;
         fespace.GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
      }

      // 9. Set up the parallel linear form b(.) which corresponds to the
      //    right-hand side of the FEM linear system, which in this case is
      //    (1,phi_i) where phi_i are the basis functions in fespace.
      ParLinearForm b(&fespace);
      ConstantCoefficient one(1.0);
      b.AddDomainIntegrator(new DomainLFIntegrator(one));
      b.Assemble();

      // 10. Define the solution vector x as a parallel finite element grid
      //     function corresponding to fespace. Initialize x with initial guess of
      //     zero, which satisfies the boundary conditions.
      x = 0.0;

      // 11. Set up the parallel bilinear form a(.,.) on the finite element space
      //     corresponding to the Laplacian operator -Delta, by adding the
      //     diffusion domain integrator.
      ParBilinearForm a(&fespace);
      a.AddDomainIntegrator(new DiffusionIntegrator(one));

      // 12. Assemble the parallel bilinear form and the corresponding linear
      //     system, applying any necessary transformations such as: parallel
      //     assembly, eliminating boundary conditions, applying conforming
      //     constraints for non-conforming AMR, static condensation, etc.
      a.Assemble();

      OperatorPtr A;
      Vector B;
      a.FormLinearSystem(ess_tdof_list, x, b, A, X, B);

      // 13. Solve the linear system A X = B.
      //     * With full assembly, use the BoomerAMG preconditioner from hypre.
      //     * With partial assembly, use Jacobi smoothing, for now.
      Solver *prec = new HypreBoomerAMG;
      CGSolver cg(MPI_COMM_WORLD);
      cg.SetRelTol(1e-12);
      cg.SetMaxIter(2000);
      cg.SetPrintLevel(1);
      if (prec) { cg.SetPreconditioner(*prec); }
      cg.SetOperator(*A);
      cg.Mult(B, X);
      delete prec;

      // 14. Recover the parallel grid function corresponding to X. This is the
      //     local finite element solution on each processor.
      a.RecoverFEMSolution(X, b, x);
   }

   if (fespaceDim == 1)
   {
      const real_t h1error = CheckH1Continuity(x);
      if (myid == 0) { cout << "H1 continuity error " << h1error << endl; }
      MFEM_VERIFY(h1error < 1.0e-12, "H1 continuity is not satisfied");
   }

   L2_FECollection fecL2(0, dim);
   ParFiniteElementSpace l2fespace(&pmesh, &fecL2);
   ParGridFunction xo(&l2fespace);
   xo = 0.0;

   for (int e=0; e<pmesh.GetNE(); ++e)
   {
      const int p_elem = fespace.GetElementOrder(e);
      Array<int> dofs;
      l2fespace.GetElementDofs(e, dofs);
      MFEM_VERIFY(dofs.Size() == 1, "");
      xo[dofs[0]] = p_elem;
   }

   // 15. Save the refined mesh and the solution in parallel. This output can
   //     be viewed later using GLVis: "glvis -np <np> -m mesh -g sol".
   std::unique_ptr<GridFunction> vis_x = x.ProlongateToMaxOrder();
   {
      ostringstream mesh_name, sol_name, order_name;
      mesh_name << "mesh." << setfill('0') << setw(6) << myid;
      sol_name << "sol." << setfill('0') << setw(6) << myid;
      order_name << "order." << setfill('0') << setw(6) << myid;

      ofstream mesh_ofs(mesh_name.str().c_str());
      mesh_ofs.precision(8);
      pmesh.ParPrint(mesh_ofs);

      ofstream sol_ofs(sol_name.str().c_str());
      sol_ofs.precision(8);

      vis_x->Save(sol_ofs);

      ofstream order_ofs(order_name.str().c_str());
      order_ofs.precision(8);
      xo.Save(order_ofs);
   }

   // 16. Send the solution by socket to a GLVis server.
   if (visualization)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      socketstream sol_sock(vishost, visport);
      sol_sock << "parallel " << num_procs << " " << myid << "\n";
      sol_sock.precision(8);
      sol_sock << "solution\n" << pmesh << *vis_x << flush;
   }

   // 17. Free the used memory.
   if (delete_fec)
   {
      delete fec;
   }

   return 0;
}

real_t CheckH1Continuity(ParGridFunction & x)
{
   x.ExchangeFaceNbrData();

   const ParFiniteElementSpace *fes = x.ParFESpace();
   ParMesh *mesh = fes->GetParMesh();

   const int dim = mesh->Dimension();

   // Following the example of KellyErrorEstimator::ComputeEstimates(),
   // we loop over interior faces and then shared faces.

   // Compute error contribution from local interior faces
   real_t errorMax = 0.0;
   for (int f = 0; f < mesh->GetNumFaces(); f++)
   {
      if (mesh->FaceIsInterior(f))
      {
         int Inf1, Inf2, NCFace;
         mesh->GetFaceInfos(f, &Inf1, &Inf2, &NCFace);

         auto FT = mesh->GetFaceElementTransformations(f);

         const int faceOrder = dim == 3 ? fes->GetFaceOrder(f) :
                               fes->GetEdgeOrder(f);
         auto &int_rule = IntRules.Get(FT->FaceGeom, 2 * faceOrder);
         const auto nip = int_rule.GetNPoints();

         // Convention
         // * Conforming face: Face side with smaller element id handles
         // the integration
         // * Non-conforming face: The slave handles the integration.
         // See FaceInfo documentation for details.
         bool isNCSlave    = FT->Elem2No >= 0 && NCFace >= 0;
         bool isConforming = FT->Elem2No >= 0 && NCFace == -1;
         if ((FT->Elem1No < FT->Elem2No && isConforming) || isNCSlave)
         {
            for (int i = 0; i < nip; i++)
            {
               const auto &fip = int_rule.IntPoint(i);
               IntegrationPoint ip;

               FT->Loc1.Transform(fip, ip);
               const real_t v1 = x.GetValue(FT->Elem1No, ip);

               FT->Loc2.Transform(fip, ip);
               const real_t v2 = x.GetValue(FT->Elem2No, ip);

               const real_t err_i = std::abs(v1 - v2);
               errorMax = std::max(errorMax, err_i);
            }
         }
      }
   }

   // Compute error contribution from shared interior faces
   for (int sf = 0; sf < mesh->GetNSharedFaces(); sf++)
   {
      const int f = mesh->GetSharedFace(sf);
      const bool trueInterior = mesh->FaceIsTrueInterior(f);
      if (!trueInterior) { continue; }

      auto FT = mesh->GetSharedFaceTransformations(sf, true);
      const int faceOrder = dim == 3 ? fes->GetFaceOrder(f) : fes->GetEdgeOrder(f);
      const auto &int_rule = IntRules.Get(FT->FaceGeom, 2 * faceOrder);
      const auto nip = int_rule.GetNPoints();

      for (int i = 0; i < nip; i++)
      {
         const auto &fip = int_rule.IntPoint(i);
         IntegrationPoint ip;

         FT->Loc1.Transform(fip, ip);
         const real_t v1 = x.GetValue(FT->Elem1No, ip);

         FT->Loc2.Transform(fip, ip);
         const real_t v2 = x.GetValue(FT->Elem2No, ip);

         const real_t err_i = std::abs(v1 - v2);
         errorMax = std::max(errorMax, err_i);
      }
   }

   real_t errorMaxGlobal = 0.0;
   MPI_Allreduce(&errorMax, &errorMaxGlobal, 1, MFEM_MPI_REAL_T, MPI_MAX,
                 fes->GetComm());
   return errorMaxGlobal;
}

void f_exact(const Vector &x, Vector &f)
{
   constexpr real_t freq = 1.0;
   constexpr real_t kappa = freq * M_PI;

   if (x.Size() == 3)
   {
      f(0) = (1. + kappa * kappa) * sin(kappa * x(1));
      f(1) = (1. + kappa * kappa) * sin(kappa * x(2));
      f(2) = (1. + kappa * kappa) * sin(kappa * x(0));
   }
   else
   {
      f(0) = (1. + kappa * kappa) * sin(kappa * x(1));
      f(1) = (1. + kappa * kappa) * sin(kappa * x(0));
      if (x.Size() == 3) { f(2) = 0.0; }
   }
}
