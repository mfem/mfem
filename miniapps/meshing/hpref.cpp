//                       Serial hp-refinement example
//
// Compile with: make hpref
//
// Sample runs:  hpref -dim 2 -n 1000
//               hpref -dim 3 -n 500
//               hpref -m ../../data/star-mixed.mesh -pref -n 100
//               hpref -m ../../data/fichera-mixed.mesh -pref -n 30
//
// Description:  This example demonstrates h- and p-refinement in a serial
//               finite element discretization of the Poisson problem (cf. ex1)
//               -Delta u = 1 with homogeneous Dirichlet boundary conditions.
//               Refinements are performed iteratively, each iteration having h-
//               or p-refinements. For simplicity, we randomly choose the
//               elements and the type of refinement, for each iteration. In
//               practice, these choices may be made in a problem-dependent way,
//               but this example serves only to illustrate the capabilities of
//               hp-refinement.
//
//               We recommend viewing Example 1 before viewing this example.

#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

real_t CheckH1Continuity(GridFunction & x);

// Deterministic function for "random" integers.
int DetRand(int & seed)
{
   seed++;
   return int(std::abs(1.0e5 * sin(seed * 1.1234 * M_PI)));
}

void f_exact(const Vector &x, Vector &f);

int main(int argc, char *argv[])
{
   // 1. Parse command-line options.
   const char *mesh_file = "";
   int order = 1;
   const char *device_config = "cpu";
   bool visualization = true;
   int numIter = 0;
   int dim = 2;
   bool deterministic = true;
   bool projectSolution = false;
   bool onlyPref = false;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
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
   args.AddOption(&deterministic, "-det", "--deterministic", "-not-det",
                  "--not-deterministic",
                  "Use deterministic random refinements");
   args.AddOption(&projectSolution, "-proj", "--project-solution", "-no-proj",
                  "--no-project",
                  "Project a coefficient to solution");
   args.AddOption(&onlyPref, "-pref", "--only-p-refinement", "-no-pref",
                  "--hp-refinement",
                  "Use only p-refinement");
   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(cout);
      return 1;
   }
   args.PrintOptions(cout);

   // 2. Enable hardware devices such as GPUs, and programming models such as
   //    CUDA, OCCA, RAJA and OpenMP based on command line options.
   Device device(device_config);
   device.Print();

   // 3. Construct or load a coarse mesh.
   std::string mesh_filename(mesh_file);
   Mesh mesh;
   if (!mesh_filename.empty())
   {
      mesh = Mesh::LoadFromFile(mesh_filename, 1, 1);
      dim = mesh.Dimension();
   }
   else if (dim == 3)
   {
      mesh = Mesh::MakeCartesian3D(2, 2, 2, Element::HEXAHEDRON);
   }
   else
   {
      mesh = Mesh::MakeCartesian2D(2, 2, Element::QUADRILATERAL, true);
   }

   mesh.EnsureNCMesh();

   // 4. Define a finite element space on the mesh. Here we use continuous
   //    Lagrange finite elements of the specified order. If order < 1, we
   //    instead use an isoparametric/isogeometric space.
   FiniteElementCollection *fec;
   bool delete_fec;
   if (order > 0)
   {
      fec = new H1_FECollection(order, dim);
      delete_fec = true;
   }
   else if (mesh.GetNodes())
   {
      fec = mesh.GetNodes()->OwnFEC();
      delete_fec = false;
      cout << "Using isoparametric FEs: " << fec->Name() << endl;
   }
   else
   {
      fec = new H1_FECollection(order = 1, dim);
      delete_fec = true;
   }

   const int fespaceDim = projectSolution ? dim : 1;
   FiniteElementSpace fespace(&mesh, fec, fespaceDim);

   // 5. Iteratively perform h- and p-refinements.
   int numH = 0;
   int numP = 0;
   int seed = 0;

   const std::vector<char> hp_char = {'h', 'p'};

   for (int iter=0; iter<numIter; ++iter)
   {
      const int r1 = deterministic ? DetRand(seed) : rand();
      const int r2 = deterministic ? DetRand(seed) : rand();
      const int elem = r1 % mesh.GetNE();
      const int hp = onlyPref ? 1 : r2 % 2;

      cout << "hp-refinement iteration " << iter << ": "
           << hp_char[hp] << "-refinement" << endl;

      if (hp == 1)
      {
         // p-ref
         Array<pRefinement> refs;
         refs.Append(pRefinement(elem, 1));  // Increase the element order by 1
         fespace.PRefineAndUpdate(refs);
         numP++;
      }
      else
      {
         // h-ref
         Array<Refinement> refs;
         refs.Append(Refinement(elem));
         mesh.GeneralRefinement(refs);
         fespace.Update(false);
         numH++;
      }
   }

   const int size = fespace.GetTrueVSize();
   cout << "Number of finite element unknowns: " << size << endl;

   const int maxP = fespace.GetMaxElementOrder();
   cout << "Total number of h-refinements: " << numH
        << "\nTotal number of p-refinements: " << numP
        << "\nMaximum order " << maxP << "\n";

   GridFunction x(&fespace);
   Vector X;

   if (projectSolution)
   {
      VectorFunctionCoefficient vec_coef(dim, f_exact);
      x.ProjectCoefficient(vec_coef);

      X.SetSize(fespace.GetTrueVSize());

      fespace.GetHpRestrictionMatrix()->Mult(x, X);
      fespace.GetProlongationMatrix()->Mult(X, x);

      // Compute and print the L^2 norm of the error.
      const real_t error = x.ComputeL2Error(vec_coef);
      cout << "\n|| E_h - E ||_{L^2} = " << error << '\n' << endl;
   }
   else
   {
      // 6. Determine the list of essential boundary dofs. In this example, the
      //    boundary conditions are defined by marking all the boundary attributes
      //    from the mesh as essential (Dirichlet) and converting them to a list of
      //    true dofs.
      Array<int> ess_tdof_list;
      if (mesh.bdr_attributes.Size())
      {
         Array<int> ess_bdr(mesh.bdr_attributes.Max());
         ess_bdr = 1;
         fespace.GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
      }

      // 7. Set up the linear form b(.) which corresponds to the right-hand side of
      //    the FEM linear system, which in this case is (1,phi_i) where phi_i are
      //    the basis functions in fespace.
      LinearForm b(&fespace);
      ConstantCoefficient one(1.0);
      b.AddDomainIntegrator(new DomainLFIntegrator(one));
      b.Assemble();

      // 8. Define the solution vector x as a finite element grid function
      //    corresponding to fespace. Initialize x with initial guess of zero,
      //    which satisfies the boundary conditions.
      x = 0.0;

      // 9. Set up the bilinear form a(.,.) on the finite element space
      //    corresponding to the Laplacian operator -Delta, by adding the diffusion
      //    domain integrator.
      BilinearForm a(&fespace);
      a.AddDomainIntegrator(new DiffusionIntegrator(one));

      // 10. Assemble the bilinear form and the corresponding linear system,
      //     applying any necessary transformations such as: assembly, eliminating
      //     boundary conditions, applying conforming constraints for non-conforming
      //     AMR, static condensation, etc.
      a.Assemble();

      OperatorPtr A;
      Vector B;
      a.FormLinearSystem(ess_tdof_list, x, b, A, X, B);

      // 11. Solve the linear system A X = B.
      {
#ifndef MFEM_USE_SUITESPARSE
         // Use a simple symmetric Gauss-Seidel preconditioner with PCG.
         GSSmoother M((SparseMatrix&)(*A));
         PCG(*A, M, B, X, 1, 200, 1e-12, 0.0);
#else
         // If MFEM was compiled with SuiteSparse, use UMFPACK to solve the system.
         UMFPackSolver umf_solver;
         umf_solver.Control[UMFPACK_ORDERING] = UMFPACK_ORDERING_METIS;
         umf_solver.SetOperator(*A);
         umf_solver.Mult(B, X);
#endif
      }

      // 12. Recover the grid function corresponding to X.
      a.RecoverFEMSolution(X, b, x);
   }

   if (fespaceDim == 1)
   {
      const real_t h1error = CheckH1Continuity(x);
      cout << "H1 continuity error " << h1error << endl;
      MFEM_VERIFY(h1error < 1.0e-12, "H1 continuity is not satisfied");
   }

   L2_FECollection fecL2(0, dim);
   FiniteElementSpace l2fespace(&mesh, &fecL2);
   GridFunction xo(&l2fespace);
   xo = 0.0;

   for (int e=0; e<mesh.GetNE(); ++e)
   {
      const int p_elem = fespace.GetElementOrder(e);
      Array<int> dofs;
      l2fespace.GetElementDofs(e, dofs);
      MFEM_VERIFY(dofs.Size() == 1, "");
      xo[dofs[0]] = p_elem;
   }

   // 13. Save the refined mesh and the solution. This output can be viewed later
   //     using GLVis: "glvis -m refined.mesh -g sol.gf".
   std::unique_ptr<GridFunction> vis_x = x.ProlongateToMaxOrder();
   ofstream mesh_ofs("refined.mesh");
   mesh_ofs.precision(8);
   mesh.Print(mesh_ofs);
   ofstream sol_ofs("sol.gf");
   sol_ofs.precision(8);
   vis_x->Save(sol_ofs);
   ofstream order_ofs("order.gf");
   order_ofs.precision(8);
   xo.Save(order_ofs);

   // 14. Send the solution by socket to a GLVis server.
   if (visualization)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      socketstream sol_sock(vishost, visport);
      sol_sock.precision(8);
      sol_sock << "solution\n" << mesh << *vis_x << flush;
   }

   // 15. Free the used memory.
   if (delete_fec)
   {
      delete fec;
   }

   return 0;
}

real_t CheckH1Continuity(GridFunction & x)
{
   const FiniteElementSpace *fes = x.FESpace();
   Mesh *mesh = fes->GetMesh();

   const int dim = mesh->Dimension();

   // Following the example of KellyErrorEstimator::ComputeEstimates(),
   // we loop over interior faces and compute their error contributions.
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

   return errorMax;
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
