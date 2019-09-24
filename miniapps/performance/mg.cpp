#include "mfem.hpp"
#include <fstream>
#include <iostream>

#include "eigenvalue.hpp"
#include "multigrid.hpp"

using namespace std;
using namespace mfem;

class PoissonMultigridOperator : public MultigridOperator
{
 private:
   Array<ParBilinearForm *> forms;
   bool partialAssembly;
   ConstantCoefficient one;

 public:
   PoissonMultigridOperator(bool partialAssembly_)
       : MultigridOperator(), partialAssembly(partialAssembly_), one(1.0)
   {
   }

   PoissonMultigridOperator(ParMesh *mesh, ParFiniteElementSpace *fespace,
                            const Array<int> &essentialDofs, int coarseOrder,
                            bool partialAssembly_)
       : MultigridOperator(), partialAssembly(partialAssembly_), one(1.0)
   {
      Operator *coarseOpr = ConstructOperator(fespace, essentialDofs);
      Solver *coarseSolver =
          ConstructCoarseSolver(mesh, coarseOpr, essentialDofs, coarseOrder);

      AddCoarseLevel(coarseOpr, coarseSolver, false, true);
   }

   ~PoissonMultigridOperator()
   {
      MFEM_FORALL(i, forms.Size(), delete forms[i];);
   }

   Operator *ConstructOperator(ParFiniteElementSpace *fespace,
                               const Array<int> &essentialDofs)
   {
      ParBilinearForm *form = new ParBilinearForm(fespace);
      if (partialAssembly)
      {
         form->SetAssemblyLevel(AssemblyLevel::PARTIAL);
      }
      form->AddDomainIntegrator(new DiffusionIntegrator(one));
      if (!partialAssembly)
      {
         form->UsePrecomputedSparsity();
      }
      form->Assemble();

      OperatorPtr opr;

      if (partialAssembly)
      {
         opr.SetType(Operator::ANY_TYPE);
      }
      else
      {
         opr.SetType(Operator::Hypre_ParCSR);
      }

      form->FormSystemMatrix(essentialDofs, opr);
      opr.SetOperatorOwner(false);

      forms.Append(form);

      return opr.Ptr();
   }

   Solver *ConstructCoarseSolver(ParMesh *mesh, Operator *opr,
                                 const Array<int> &essentialDofs,
                                 int coarseOrder)
   {
      HypreParMatrix *hypreCoarseMat = nullptr;

      // Reuse matrix for AMG
      if (!partialAssembly && coarseOrder == 1)
      {
         hypreCoarseMat = dynamic_cast<HypreParMatrix *>(opr);
      }
      else
      {
         ParBilinearForm *a_pc = nullptr;

         if (coarseOrder > 1)
         {
            ParMesh *pmesh_lor =
                new ParMesh(mesh, coarseOrder, BasisType::GaussLobatto);
            H1_FECollection *fec_lor = new H1_FECollection(
                1, mesh->Dimension(), BasisType::GaussLobatto);
            ParFiniteElementSpace *fespace_lor =
                new ParFiniteElementSpace(pmesh_lor, fec_lor);
            a_pc = new ParBilinearForm(fespace_lor);
         }
         else
         {
            H1_FECollection *fec_lor = new H1_FECollection(
                1, mesh->Dimension(), BasisType::GaussLobatto);
            ParFiniteElementSpace *fespace_lor =
                new ParFiniteElementSpace(mesh, fec_lor);
            a_pc = new ParBilinearForm(fespace_lor);
         }

         a_pc->AddDomainIntegrator(new DiffusionIntegrator(one));
         a_pc->UsePrecomputedSparsity();
         a_pc->Assemble();

         hypreCoarseMat = new HypreParMatrix();
         a_pc->FormSystemMatrix(essentialDofs, *hypreCoarseMat);
      }

      HypreBoomerAMG *amg = new HypreBoomerAMG(*hypreCoarseMat);
      amg->SetPrintLevel(-1);
      amg->SetMaxIter(2);
      return amg;
   }

   Solver *ConstructSmoother(ParFiniteElementSpace *fespace,
                             Operator *solveOperator,
                             const Array<int> &essentialDofs)
   {
      Solver *smoother = nullptr;

      if (partialAssembly)
      {
         Vector diag(fespace->GetTrueVSize());
         forms.Last()->AssembleDiagonal(diag);

         Vector ev(solveOperator->Width());
         OperatorJacobiSmoother invDiagOperator(diag, essentialDofs, 1.0);
         ProductOperator diagPrecond(&invDiagOperator, solveOperator, false,
                                     false);
         double estLargestEigenvalue = PowerMethod::EstimateLargestEigenvalue(
             MPI_COMM_WORLD, diagPrecond, ev, 10, 1e-8);
         smoother = new OperatorChebyshevSmoother(
             solveOperator, diag, essentialDofs, 3, estLargestEigenvalue);
      }
      else
      {
         smoother =
             new HypreSmoother(static_cast<HypreParMatrix &>(*solveOperator));
      }

      return smoother;
   }

   void AddLevel(ParFiniteElementSpace *lFEspace,
                 ParFiniteElementSpace *hFEspace,
                 const Array<int> &essentialDofs)
   {
      Operator *opr = ConstructOperator(hFEspace, essentialDofs);
      Solver *smoother = ConstructSmoother(hFEspace, opr, essentialDofs);
      Operator *P = new TrueTransferOperator(*lFEspace, *hFEspace);
      MultigridOperator::AddLevel(opr, smoother, P, partialAssembly, true,
                                  true);
   }

   void FormLinearSystem(const Array<int> &ess_tdof_list, Vector &x, Vector &b,
                         Vector &X, Vector &B, int copy_interior = 0)
   {
      OperatorPtr dummy;
      forms.Last()->FormLinearSystem(ess_tdof_list, x, b, dummy, X, B,
                                     copy_interior);
   }

   void RecoverFEMSolution(const Vector &X, const Vector &b, Vector &x)
   {
      forms.Last()->RecoverFEMSolution(X, b, x);
   }
};

int main(int argc, char *argv[])
{
   int num_procs, myid;
   MPI_Init(&argc, &argv);
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
   MPI_Comm_rank(MPI_COMM_WORLD, &myid);

   // 1. Parse command-line options.
   const char *mesh_file = "../../data/inline-hex.mesh";
   int ref_levels = 0;
   int order = 1;
   int h_levels = 1;
   int o_levels = 1;
   bool visualization = 1;
   bool partialAssembly = true;
   const char *precondInput = "LOR";

   enum class Method
   {
      MG = 0,
      LOR = 1,
      LORS = 2
   } method;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh", "Mesh file to use.");
   args.AddOption(
       &ref_levels, "-r", "--refine",
       "Number of times to refine the initial mesh uniformly;"
       "This mesh will be the coarse mesh in the multigrid hierarchy");
   args.AddOption(&order, "-o", "--order",
                  "Order of the finite element spaces");
   args.AddOption(&h_levels, "-hl", "--hlevels",
                  "Number of geometric levels in the multigrid hierarchy");
   args.AddOption(&o_levels, "-ol", "--orderlevels",
                  "Number of order levels in the multigrid hierarchy");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&partialAssembly, "-pa", "--partialassembly", "-no-pa",
                  "--no-partialassembly",
                  "Enable or disable partial assembly.");
   args.AddOption(&precondInput, "-p", "--precond",
                  "Preconditioner: MG - Multigrid, LOR = Low-order refined, "
                  "LORS = Low-order refined with smoothing");

   args.Parse();
   if (!args.Good())
   {
      if (myid == 0)
      {
         args.PrintUsage(cout);
      }
      MPI_Finalize();
      return 1;
   }

   if (myid == 0)
   {
      args.PrintOptions(cout);
   }

   if (o_levels > 1 && order > 1)
   {
      MFEM_ABORT("Order refinements are not supported with order > 1");
   }

   std::map<std::string, Method> mapInputToPrecond = {
       {"MG", Method::MG},
       {"LOR", Method::LOR},
       {"LORS", Method::LORS},
   };

   auto it = mapInputToPrecond.find(std::string(precondInput));
   if (it == mapInputToPrecond.end())
   {
      MFEM_ABORT("Method " << precondInput << " not found");
   }
   method = it->second;

   // See class BasisType in fem/fe_coll.hpp for available basis types
   int basis = BasisType::GaussLobatto;
   if (myid == 0)
   {
      cout << "Using " << BasisType::Name(basis) << " basis ..." << endl;
   }

   // 2. Read the mesh from the given mesh file. We can handle triangular,
   //    quadrilateral, tetrahedral, hexahedral, surface and volume meshes with
   //    the same code.
   Mesh *mesh = new Mesh(mesh_file, 1, 1);
   int dim = mesh->Dimension();

   Array<int> ess_bdr(mesh->bdr_attributes.Max());
   ess_bdr = 1;

   // Initial refinements of the input grid
   for (int i = 0; i < ref_levels; ++i)
   {
      mesh->UniformRefinement();
   }

   ParMesh *pmesh = new ParMesh(MPI_COMM_WORLD, *mesh);
   delete mesh;
   mesh = nullptr;

   Array<int> orders;
   Array<FiniteElementCollection *> feCollectons;
   orders.Append(order);
   feCollectons.Append(new H1_FECollection(order, dim, basis));

   // Set up coarse grid finite element space
   ParFiniteElementSpace *fespace =
       new ParFiniteElementSpace(pmesh, feCollectons.Last());
   HYPRE_Int size = fespace->GlobalTrueVSize();

   if (myid == 0)
   {
      cout << "Number of finite element unknowns on level 0: " << size << endl;
   }

   Array<Array<int> *> essentialTrueDoFs;
   essentialTrueDoFs.Append(new Array<int>());
   fespace->GetEssentialTrueDofs(ess_bdr, *essentialTrueDoFs.Last());

   // Build hierarchy of meshes and spaces
   // Geometric refinements
   ParSpaceHierarchy *spaceHierarchy =
       new ParSpaceHierarchy(pmesh, fespace, true, true);
   for (int level = 1; level < h_levels; ++level)
   {
      spaceHierarchy->AddUniformlyRefinedLevel();
      orders.Append(order);
      if (myid == 0)
      {
         cout << "h refinement" << endl;
      }
   }

   // Order refinements
   for (int level = 1; level < o_levels; ++level)
   {
      int newOrder = std::pow(2, level);
      feCollectons.Append(new H1_FECollection(newOrder, dim, basis));
      spaceHierarchy->AddOrderRefinedLevel(feCollectons.Last());
      orders.Append(newOrder);
      if (myid == 0)
      {
         cout << "p refinement from order " << std::pow(2, level - 1) << " to "
              << newOrder << endl;
      }
   }

   // Collect essential dofs
   for (int level = 1; level < spaceHierarchy->GetNumLevels(); ++level)
   {
      essentialTrueDoFs.Append(new Array<int>());
      spaceHierarchy->GetFESpaceAtLevel(level).GetEssentialTrueDofs(
          ess_bdr, *essentialTrueDoFs[level]);

      size = spaceHierarchy->GetFESpaceAtLevel(level).GlobalTrueVSize();
      if (myid == 0)
      {
         cout << "Number of finite element unknowns on level " << level << ": "
              << size << endl;
      }
   }

   PoissonMultigridOperator *solveOperator = nullptr;

   if (myid == 0)
   {
      cout << "Setting up operators..." << flush;
   }

   tic_toc.Clear();
   tic_toc.Start();

   // Construct multigrid operator depending on method
   if (method == Method::LOR || method == Method::LORS)
   {
      solveOperator = new PoissonMultigridOperator(
          &spaceHierarchy->GetFinestMesh(), &spaceHierarchy->GetFinestFESpace(),
          *essentialTrueDoFs.Last(), orders.Last(), partialAssembly);

      if (method == Method::LORS)
      {
         Operator *opr = solveOperator->GetOperatorAtLevel(0);
         Operator *identityProlongation =
             new IdentityOperator(solveOperator->Height());
         Solver *smoother = solveOperator->ConstructSmoother(
             &spaceHierarchy->GetFinestFESpace(), opr,
             *essentialTrueDoFs.Last());

         solveOperator->MultigridOperator::AddLevel(
             opr, smoother, identityProlongation, false, true, true);
      }
   }
   else
   {
      solveOperator = new PoissonMultigridOperator(
          &spaceHierarchy->GetMeshAtLevel(0),
          &spaceHierarchy->GetFESpaceAtLevel(0), *essentialTrueDoFs[0],
          orders[0], partialAssembly);

      for (int level = 1; level < spaceHierarchy->GetNumLevels(); ++level)
      {
         solveOperator->AddLevel(&spaceHierarchy->GetFESpaceAtLevel(level - 1),
                                 &spaceHierarchy->GetFESpaceAtLevel(level),
                                 *essentialTrueDoFs[level]);
      }
   }

   MultigridSolver *preconditioner = new MultigridSolver(
       solveOperator, MultigridSolver::CycleType::VCYCLE, 3, 3);

   tic_toc.Stop();
   if (myid == 0)
   {
      cout << " done. Setup time: " << tic_toc.RealTime() << "s" << endl;
   }

   ParGridFunction x(&spaceHierarchy->GetFinestFESpace());
   x = 0.0;

   if (myid == 0)
   {
      cout << "Assembling rhs..." << flush;
   }
   tic_toc.Clear();
   tic_toc.Start();
   ParLinearForm *b = new ParLinearForm(&spaceHierarchy->GetFinestFESpace());
   ConstantCoefficient one(1.0);
   b->AddDomainIntegrator(new DomainLFIntegrator(one));
   b->Assemble();
   tic_toc.Stop();
   if (myid == 0)
   {
      cout << " done, " << tic_toc.RealTime() << "s" << endl;
   }

   Vector X, B;
   solveOperator->FormLinearSystem(*essentialTrueDoFs.Last(), x, *b, X, B);

   tic_toc.Clear();
   tic_toc.Start();

   CGSolver pcg(MPI_COMM_WORLD);
   pcg.SetPrintLevel(1);
   pcg.SetMaxIter(100);
   pcg.SetRelTol(1e-6);
   pcg.SetAbsTol(0.0);
   pcg.SetOperator(*solveOperator);
   pcg.SetPreconditioner(*preconditioner);
   pcg.Mult(B, X);

   tic_toc.Stop();

   if (myid == 0)
   {
      cout << "Time to solution: " << tic_toc.RealTime() << "s" << endl;
   }

   solveOperator->RecoverFEMSolution(X, *b, x);

   delete preconditioner;
   delete solveOperator;

   if (visualization)
   {
      char vishost[] = "localhost";
      int visport = 19916;
      socketstream sol_sock(vishost, visport);
      sol_sock << "parallel " << num_procs << " " << myid << "\n";
      sol_sock.precision(8);
      sol_sock << "solution\n" << spaceHierarchy->GetFinestMesh() << x << flush;
   }

   delete b;
   delete spaceHierarchy;

   MFEM_FORALL(i, essentialTrueDoFs.Size(), delete essentialTrueDoFs[i];);
   MFEM_FORALL(i, feCollectons.Size(), delete feCollectons[i];);

   MPI_Finalize();

   return 0;
}