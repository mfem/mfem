#include "mfem.hpp"
#include <fstream>
#include <iostream>

#include "multigrid.hpp"
#include "eigenvalue.hpp"

using namespace std;
using namespace mfem;

class PoissonMultigridOperator : public MultigridOperator
{
private:
   HypreBoomerAMG* amg;
   Array<ParBilinearForm*> forms;
   bool partialAssembly;

   Operator* ConstructOperator(ParFiniteElementSpace* fespace, const Array<int>& essentialDofs)
   {
      ParBilinearForm* form = new ParBilinearForm(fespace);
      if (NumLevels() != 0 && partialAssembly)
      {
         form->SetAssemblyLevel(AssemblyLevel::PARTIAL);
      }
      ConstantCoefficient one(1.0);
      form->AddDomainIntegrator(new DiffusionIntegrator(one));
      form->Assemble();

      OperatorPtr opr;
      
      if (NumLevels() != 0 && partialAssembly)
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

   Solver* ConstructCoarseSolver(Operator* opr)
   {
      HypreParMatrix& hypreCoarseMat = dynamic_cast<HypreParMatrix&>(*opr);
      amg = new HypreBoomerAMG(hypreCoarseMat);
      amg->SetPrintLevel(-1);
      return amg;
   }

   Solver* ConstructSmoother(ParFiniteElementSpace* fespace, Operator* opr, const Array<int>& essentialDofs)
   {
      Solver* smoother = nullptr;

      if (partialAssembly)
      {
         Vector diag(fespace->GetTrueVSize());
         forms.Last()->AssembleDiagonal(diag);
         
         Vector ev(opr->Width());
         OperatorJacobiSmoother invDiagOperator(diag, essentialDofs, 1.0);
         ProductOperator diagPrecond(&invDiagOperator, opr, false, false);
         double estLargestEigenvalue = PowerMethod::EstimateLargestEigenvalue(MPI_COMM_WORLD, diagPrecond, ev, 10, 1e-8);
         smoother = new OperatorChebyshevSmoother(opr, diag, essentialDofs, 3, estLargestEigenvalue);
      }
      else
      {
         smoother = new HypreSmoother(static_cast<HypreParMatrix&>(*opr));
      }

      return smoother;
   }

public:
   PoissonMultigridOperator(ParFiniteElementSpace* fespace, const Array<int>& essentialDofs, bool partialAssembly_)
      : MultigridOperator(), amg(nullptr), partialAssembly(partialAssembly_)
   {
      Operator* coarseOpr = ConstructOperator(fespace, essentialDofs);
      Solver* coarseSolver = ConstructCoarseSolver(coarseOpr);

      AddCoarseLevel(coarseOpr, coarseSolver, false, true);
   }

   ~PoissonMultigridOperator()
   {
      MFEM_FORALL(i, forms.Size(), delete forms[i]; );
   }

   void AddLevel(ParFiniteElementSpace* lFEspace, ParFiniteElementSpace* hFEspace, const Array<int>& essentialDofs)
   {
      Operator* opr = ConstructOperator(hFEspace, essentialDofs);
      Solver* smoother = ConstructSmoother(hFEspace, opr, essentialDofs);
      Operator* P = new TrueTransferOperator(*lFEspace, *hFEspace);
      MultigridOperator::AddLevel(opr, smoother, P, partialAssembly, true, true);
   }

   void FormLinearSystem(const Array<int> &ess_tdof_list,
                         Vector &x, Vector &b,
                         Vector &X, Vector &B,
                         int copy_interior = 0)
   {
      OperatorPtr dummy;
      forms.Last()->FormLinearSystem(ess_tdof_list, x, b, dummy, X, B, copy_interior);
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
   const char *mesh_file = "../../data/fichera.mesh";
   int ref_levels = 2;
   int mg_levels = 2;
   int order = 1;
   const char *basis_type = "G"; // Gauss-Lobatto
   bool visualization = 1;
   bool partialAssembly = false;
   bool pMultigrid = false;
   bool boomerAMG = false;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&order, "-o", "--order",
                  "Order of the finite element spaces");
   args.AddOption(&ref_levels, "-r", "--refine",
                  "Number of times to refine the initial mesh uniformly;"
                  "This mesh will be the coarse mesh in the multigrid hierarchy");
   args.AddOption(&mg_levels, "-l", "--levels",
                  "Number of levels in the multigrid hierarchy;");
   args.AddOption(&basis_type, "-b", "--basis-type",
                  "Basis: G - Gauss-Lobatto, P - Positive, U - Uniform");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&partialAssembly, "-pa", "--partialassembly", "-no-pa",
                  "--no-partialassembly",
                  "Enable or disable partial assembly.");
   args.AddOption(&pMultigrid, "-pmg", "--pmultigrid", "-no-pmg",
                  "--no-pmultigrid",
                  "Enable or p multigrid.");
   args.AddOption(&boomerAMG, "-boomeramg", "--boomeramg", "-no-boomeramg",
                  "--no-boomeramg",
                  "Enable or disable usage of BoomerAMG at finest level");
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

   // See class BasisType in fem/fe_coll.hpp for available basis types
   int basis = BasisType::GetType(basis_type[0]);
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

   Array<FiniteElementCollection*> feCollectons;
   feCollectons.Append(new H1_FECollection(order, dim, basis));

   // Set up coarse grid finite element space
   ParFiniteElementSpace* fespace = new ParFiniteElementSpace(pmesh, feCollectons.Last());
   HYPRE_Int size = fespace->GlobalTrueVSize();

   if (myid == 0)
   {
      cout << "Number of finite element unknowns on level 0: " << size << "; FE order: " << order << endl;
   }

   Array<Array<int>*> essentialTrueDoFs;
   essentialTrueDoFs.Append(new Array<int>());
   fespace->GetEssentialTrueDofs(ess_bdr, *essentialTrueDoFs.Last());

   ParSpaceHierarchy* spaceHierarchy = new ParSpaceHierarchy(pmesh, fespace, true, true);
   PoissonMultigridOperator* oprMultigrid = new PoissonMultigridOperator(fespace, *essentialTrueDoFs.Last(), partialAssembly);

   for(int level = 1; level < mg_levels; ++level)
   {
      tic_toc.Clear();
      tic_toc.Start();

      if (pMultigrid)
      {
         order = std::pow(2, level);
         feCollectons.Append(new H1_FECollection(order, dim, basis));
         spaceHierarchy->AddOrderRefinedLevel(feCollectons.Last());
      }
      else
      {
         spaceHierarchy->AddUniformlyRefinedLevel();
      }

      size = spaceHierarchy->GetFinestFESpace().GlobalTrueVSize();
      if (myid == 0)
      {
         cout << "Number of finite element unknowns on level " << level << ": " << size << "; FE order: " << order << endl;
      }
      
      essentialTrueDoFs.Append(new Array<int>());
      spaceHierarchy->GetFinestFESpace().GetEssentialTrueDofs(ess_bdr, *essentialTrueDoFs.Last());

      oprMultigrid->AddLevel(&spaceHierarchy->GetFESpaceAtLevel(level - 1), &spaceHierarchy->GetFinestFESpace(), *essentialTrueDoFs.Last());
      tic_toc.Stop();
      if (myid == 0)
      {
         cout << "Assembly time on level " << level << ": " << tic_toc.RealTime() << "s" << endl;
      }
   }

   ParGridFunction x(&spaceHierarchy->GetFinestFESpace());
   x = 0.0;

   if (myid == 0)
   {
      cout << "Assembling rhs..." << flush;
   }
   tic_toc.Clear();
   tic_toc.Start();
   ParLinearForm* b = new ParLinearForm(&spaceHierarchy->GetFinestFESpace());
   ConstantCoefficient one(1.0);
   b->AddDomainIntegrator(new DomainLFIntegrator(one));
   b->Assemble();
   tic_toc.Stop();
   if (myid == 0)
   {
      cout << "\t\t\t\tdone, " << tic_toc.RealTime() << "s." << endl;
   }

   Vector X, B;
   oprMultigrid->FormLinearSystem(*essentialTrueDoFs.Last(), x, *b, X, B);

   Vector r(X.Size());
   MultigridSolver* mgCycle = new MultigridSolver(oprMultigrid, MultigridSolver::CycleType::VCYCLE, 3, 3);

   tic_toc.Clear();
   tic_toc.Start();

   ParMesh* pmesh_lor = nullptr;
   H1_FECollection* fec_lor = nullptr;
   ParFiniteElementSpace* fespace_lor = nullptr;
   ParBilinearForm* a_pc = nullptr;
   HypreBoomerAMG* amg = nullptr;

   if (boomerAMG)
   {
      int basis_lor = basis;
      if (basis == BasisType::Positive) { basis_lor=BasisType::ClosedUniform; }
      pmesh_lor = new ParMesh(&spaceHierarchy->GetFinestMesh(), order, basis_lor);
      fec_lor = new H1_FECollection(1, dim);
      fespace_lor = new ParFiniteElementSpace(pmesh_lor, fec_lor);

      a_pc = new ParBilinearForm(fespace_lor);
      HypreParMatrix A_pc;
      a_pc->AddDomainIntegrator(new DiffusionIntegrator(one));
      a_pc->UsePrecomputedSparsity();
      a_pc->Assemble();
      a_pc->FormSystemMatrix(*essentialTrueDoFs.Last(), A_pc);

      amg = new HypreBoomerAMG(A_pc);
   }

   CGSolver pcg(MPI_COMM_WORLD);
   pcg.SetPrintLevel(1);
   pcg.SetMaxIter(100);
   pcg.SetRelTol(1e-6);
   pcg.SetAbsTol(0.0);
   pcg.SetOperator(*oprMultigrid->GetOperatorAtFinestLevel());

   if (boomerAMG)
   {
      pcg.SetPreconditioner(*amg);
   }
   else
   {
      pcg.SetPreconditioner(*mgCycle);
   }
   
   pcg.Mult(B, X);

   if (boomerAMG)
   {
      delete amg;
      delete a_pc;
      delete fespace_lor;
      delete fec_lor;
      delete pmesh_lor;
   }

   tic_toc.Stop();

   if (myid == 0)
   {
      cout << "Time to solution: " << tic_toc.RealTime() << "s" << endl;
   }

   // oprMultigrid->Mult(X, r);
   // subtract(B, r, r);

   // double beginRes = sqrt(InnerProduct(MPI_COMM_WORLD, r, r));
   // double prevRes = beginRes;
   // const int printWidth = 11;

   // if (myid == 0)
   // {
   //    cout << std::setw(3) << "It";
   //    cout << std::setw(printWidth) << "Absres";
   //    cout << std::setw(printWidth) << "Relres";
   //    cout << std::setw(printWidth) << "Conv";
   //    cout << std::setw(printWidth) << "Time [s]" << endl;

   //    cout << std::setw(3) << 0;
   //    cout << std::scientific << std::setprecision(3) << std::setw(printWidth) << beginRes;
   //    cout << std::scientific << std::setprecision(3) << std::setw(printWidth) << 1.0;
   //    cout << std::scientific << std::setprecision(3) << std::setw(printWidth) << 0.0;
   //    cout << std::scientific << std::setprecision(3) << std::setw(printWidth) << 0.0 << endl;
   // }

   // for (int iter = 0; iter < 10; ++iter)
   // {
   //    tic_toc.Clear();
   //    tic_toc.Start();
   //    mgCycle->Mult(B, X);
   //    tic_toc.Stop();

   //    oprMultigrid->Mult(X, r);
   //    subtract(B, r, r);

   //    double res = sqrt(InnerProduct(MPI_COMM_WORLD, r, r));
   //    if (myid == 0)
   //    {
   //       cout << std::setw(3) << iter + 1;
   //       cout << std::scientific << std::setprecision(3) << std::setw(printWidth) << res;
   //       cout << std::scientific << std::setprecision(3) << std::setw(printWidth) << res/beginRes;
   //       cout << std::scientific << std::setprecision(3) << std::setw(printWidth) << res/prevRes;
   //       cout << std::scientific << std::setprecision(3) << std::setw(printWidth) << tic_toc.RealTime() << endl;
   //    }

   //    if (res < 1e-10 * beginRes)
   //    {
   //       break;
   //    }

   //    prevRes = res;
   // }

   oprMultigrid->RecoverFEMSolution(X, *b, x);

   if (visualization)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      socketstream sol_sock(vishost, visport);
      sol_sock << "parallel " << num_procs << " " << myid << "\n";
      sol_sock.precision(8);
      sol_sock << "solution\n" << spaceHierarchy->GetFinestMesh() << x << flush;
   }

   delete b;
   delete mgCycle;
   delete oprMultigrid;
   delete spaceHierarchy;
   
   MFEM_FORALL(i, essentialTrueDoFs.Size(), delete essentialTrueDoFs[i]; );
   MFEM_FORALL(i, feCollectons.Size(), delete feCollectons[i]; );

   MPI_Finalize();

   return 0;
}