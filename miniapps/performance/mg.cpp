#include "mfem.hpp"
#include <fstream>
#include <iostream>

#include "multigrid.hpp"
#include "eigenvalue.hpp"

using namespace std;
using namespace mfem;

class TransferOperator : public Operator
{
private:
   const FiniteElementSpace& hFESpace;
   const FiniteElementSpace& lFESpace;

public:
   TransferOperator(const FiniteElementSpace& hFESpace_,
                    const FiniteElementSpace& lFESpace_)
      : Operator(hFESpace_.GetVSize(), lFESpace_.GetVSize()), hFESpace(hFESpace_), lFESpace(lFESpace_)
   { }

   virtual ~TransferOperator() {}

   virtual void Mult(const Vector &x, Vector &y) const
   {
      Mesh* mesh = hFESpace.GetMesh();
      Array<int> l_dofs, h_dofs;
      DenseMatrix loc_prol;
      Vector subY, subX;

      Geometry::Type cached_geom = Geometry::INVALID;
      const FiniteElement *h_fe = NULL;
      const FiniteElement *l_fe = NULL;
      IsoparametricTransformation T;

      for (int i = 0; i < mesh->GetNE(); i++)
      {
         hFESpace.GetElementDofs(i, h_dofs);
         lFESpace.GetElementDofs(i, l_dofs);

         const Geometry::Type geom = mesh->GetElementBaseGeometry(i);
         if (geom != cached_geom)
         {
            h_fe = hFESpace.GetFE(i);
            l_fe = lFESpace.GetFE(i);
            T.SetIdentityTransformation(h_fe->GetGeomType());
            h_fe->Project(*l_fe, T, loc_prol);
            subY.SetSize(loc_prol.Height());
            cached_geom = geom;
         }

         x.GetSubVector(l_dofs, subX);
         loc_prol.Mult(subX, subY);
         y.SetSubVector(h_dofs, subY);
      }
   }

   virtual void MultTranspose(const Vector &x, Vector &y) const
   {
      y = 0.0;

      Mesh* mesh = hFESpace.GetMesh();
      Array<int> l_dofs, h_dofs;
      DenseMatrix loc_prol;
      Vector subY, subX;

      Array<char> processed(hFESpace.GetVSize());
      processed = 0;

      Geometry::Type cached_geom = Geometry::INVALID;
      const FiniteElement *h_fe = NULL;
      const FiniteElement *l_fe = NULL;
      IsoparametricTransformation T;

      for (int i = 0; i < mesh->GetNE(); i++)
      {
         hFESpace.GetElementDofs(i, h_dofs);
         lFESpace.GetElementDofs(i, l_dofs);

         const Geometry::Type geom = mesh->GetElementBaseGeometry(i);
         if (geom != cached_geom)
         {
            h_fe = hFESpace.GetFE(i);
            l_fe = lFESpace.GetFE(i);
            T.SetIdentityTransformation(h_fe->GetGeomType());
            h_fe->Project(*l_fe, T, loc_prol);
            subY.SetSize(loc_prol.Width());
            cached_geom = geom;
         }

         x.GetSubVector(h_dofs, subX);
         for (int p = 0; p < h_dofs.Size(); ++p)
         {
            if (processed[h_dofs[p]])
            {
               subX[p] = 0.0;
            }
            else
            {
               processed[h_dofs[p]] = 1;
            }
         }

         loc_prol.MultTranspose(subX, subY);
         y.AddElementVector(l_dofs, subY);
      }
   }
};

class TrueTransferOperator : public Operator
{
 private:
   TransferOperator *transferOperator;
   TripleProductOperator *opr;

 public:
   TrueTransferOperator(const FiniteElementSpace &hFESpace_,
                        const FiniteElementSpace &lFESpace_)
   {
      transferOperator = new TransferOperator(hFESpace_, lFESpace_);

      opr = new TripleProductOperator(
          hFESpace_.GetRestrictionMatrix(), transferOperator,
          lFESpace_.GetProlongationMatrix(), false, false, false);
   }

   ~TrueTransferOperator()
   {
      delete opr;
      delete transferOperator;
   }

   virtual void Mult(const Vector &x, Vector &y) const override
   {
      opr->Mult(x, y);
   }

   virtual void MultTranspose(const Vector &x, Vector &y) const override
   {
      opr->MultTranspose(x, y);
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
   int order = 3;
   const char *basis_type = "G"; // Gauss-Lobatto
   bool visualization = 1;
   const bool partialAssembly = false;
   const bool pMultigrid = false;

   OptionsParser args(argc, argv);
   // args.AddOption(&mesh_file, "-m", "--mesh",
   //                "Mesh file to use.");
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
   // Mesh *mesh = new Mesh(1, 1, 1, Element::HEXAHEDRON, true, 1.0, 1.0, 1.0, false);
   // Mesh *mesh = new Mesh(1, 1, Element::QUADRILATERAL, true, 1.0, 1.0, false);
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

   FiniteElementCollection *fec = new H1_FECollection(order, dim, basis);

   // Set up coarse grid finite element space
   ParFiniteElementSpace* fespace = new ParFiniteElementSpace(pmesh, fec);
   HYPRE_Int size = fespace->GlobalTrueVSize();

   if (myid == 0)
   {
      cout << "Number of finite element unknowns on level 0: " << size << "; FE order: " << order << endl;
   }

   Array<int>* essentialTrueDoFs = new Array<int>();
   fespace->GetEssentialTrueDofs(ess_bdr, *essentialTrueDoFs);

   ParBilinearForm* coarseForm = new ParBilinearForm(fespace);
   // if (partialAssembly) { coarseForm->SetAssemblyLevel(AssemblyLevel::PARTIAL); }
   ConstantCoefficient one(1.0);
   coarseForm->AddDomainIntegrator(new DiffusionIntegrator(one));
   coarseForm->Assemble();

   OperatorPtr coarseOpr;
   coarseForm->FormSystemMatrix(*essentialTrueDoFs, coarseOpr);
   coarseOpr.SetOperatorOwner(false);

   HypreParMatrix& hypreCoarseMat = dynamic_cast<HypreParMatrix&>(*coarseOpr);
   HypreBoomerAMG *amg = new HypreBoomerAMG(hypreCoarseMat);
   amg->SetPrintLevel(-1);
   HyprePCG *coarseSolver = new HyprePCG(hypreCoarseMat);
   coarseSolver->SetTol(1e-8);
   coarseSolver->SetMaxIter(500);
   coarseSolver->SetPrintLevel(-1);
   coarseSolver->SetPreconditioner(*amg);

   Array<ParBilinearForm*> forms;
   forms.Append(coarseForm);

   ParSpaceHierarchy spaceHierarchy(pmesh, fespace, true, true);
   MultigridOperator oprMultigrid(coarseOpr.Ptr(), coarseSolver, true, true);

   for(int level = 1; level < mg_levels; ++level)
   {
      int newOrder = order;

      if (pMultigrid)
      {
         newOrder = std::pow(2, level);
         FiniteElementCollection *fecHighOrder = new H1_FECollection(newOrder, dim, basis);
         spaceHierarchy.AddOrderRefinedLevel(fecHighOrder);
      }
      else
      {
         spaceHierarchy.AddUniformlyRefinedLevel();
      }

      size = spaceHierarchy.GetFinestFESpace().GlobalTrueVSize();
      if (myid == 0)
      {
         cout << "Number of finite element unknowns on level " << level << ": " << size << "; FE order: " << newOrder << endl;
      }

      // Create form and operator on next level
      ParBilinearForm* form = new ParBilinearForm(&spaceHierarchy.GetFinestFESpace());
      if (partialAssembly) { form->SetAssemblyLevel(AssemblyLevel::PARTIAL); }
      form->AddDomainIntegrator(new DiffusionIntegrator(one));
      form->Assemble();
      forms.Append(form);

      // Create operator with eliminated essential dofs
      OperatorPtr opr;
      essentialTrueDoFs = new Array<int>();
      spaceHierarchy.GetFinestFESpace().GetEssentialTrueDofs(ess_bdr, *essentialTrueDoFs);

      if (partialAssembly)
      {
         opr.SetType(Operator::ANY_TYPE);
      }
      else
      {
         opr.SetType(Operator::Hypre_ParCSR);
      }

      form->FormSystemMatrix(*essentialTrueDoFs, opr);
      opr.SetOperatorOwner(false);

      // Create prolongation
      OperatorPtr P;

      if (pMultigrid)
      {
         Operator* Pt = new TrueTransferOperator(spaceHierarchy.GetFinestFESpace(), spaceHierarchy.GetFESpaceAtLevel(level - 1));
         P.SetType(Operator::ANY_TYPE);
         P.Reset(Pt, false);
      }
      else
      {
         P.SetType(Operator::ANY_TYPE);
         spaceHierarchy.GetFinestFESpace().GetTrueTransferOperator(spaceHierarchy.GetFESpaceAtLevel(level - 1), P);

         P.SetOperatorOwner(false);
      }

      // Create smoother
      Solver* smoother = nullptr;

      if (partialAssembly)
      {
         Vector diag(spaceHierarchy.GetFinestFESpace().GetTrueVSize());
         form->AssembleDiagonal(diag);
         
         Vector ev(opr->Width());
         OperatorJacobiSmoother invDiagOperator(diag, *essentialTrueDoFs, 1.0);
         ProductOperator diagPrecond(&invDiagOperator, opr.Ptr(), false, false);
         double estLargestEigenvalue = PowerMethod::EstimateLargestEigenvalue(MPI_COMM_WORLD, diagPrecond, ev, 10, 1e-8);
         smoother = new OperatorChebyshevSmoother(opr.Ptr(), diag, *essentialTrueDoFs, 3, estLargestEigenvalue);
      }
      else
      {
         smoother = new HypreSmoother(*opr.As<HypreParMatrix>());
      }

      oprMultigrid.AddLevel(opr.Ptr(), smoother, P.Ptr(), true, true, true);
   }

   ParGridFunction x(&spaceHierarchy.GetFinestFESpace());
   x = 0.0;

   if (myid == 0)
   {
      cout << "Assembling rhs..." << flush;
   }
   tic_toc.Clear();
   tic_toc.Start();
   ParLinearForm b(&spaceHierarchy.GetFinestFESpace());
   b.AddDomainIntegrator(new DomainLFIntegrator(one));
   b.Assemble();
   tic_toc.Stop();
   if (myid == 0)
   {
      cout << "\t\t\t\tdone, " << tic_toc.RealTime() << "s." << endl;
   }

   Vector X, B;
   OperatorPtr dummy;
   forms[spaceHierarchy.GetFinestLevelIndex()]->FormLinearSystem(*essentialTrueDoFs, x, b, dummy, X, B);

   Vector r(X.Size());
   MultigridSolver mgCycle(oprMultigrid, MultigridSolver::CycleType::VCYCLE, 3, 3);

   // HypreParMatrix& hypreMat = dynamic_cast<HypreParMatrix&>(*oprMultigrid.GetOperatorAtFinestLevel());
   // HypreBoomerAMG *amg = new HypreBoomerAMG(hypreMat);
   // HyprePCG *pcg = new HyprePCG(hypreMat);
   // pcg->SetTol(1e-8);
   // pcg->SetMaxIter(500);
   // pcg->SetPrintLevel(2);
   // pcg->SetPreconditioner(*amg);
   // pcg->Mult(B, X);

   // CGSolver pcg(MPI_COMM_WORLD);
   // pcg.SetPrintLevel(1);
   // pcg.SetMaxIter(10);
   // pcg.SetRelTol(1e-5);
   // pcg.SetAbsTol(0.0);
   // pcg.SetOperator(*oprMultigrid.GetOperatorAtFinestLevel());
   // pcg.SetPreconditioner(mgCycle);
   // pcg.Mult(B, X);

   oprMultigrid.Mult(X, r);
   subtract(B, r, r);

   double beginRes = InnerProduct(MPI_COMM_WORLD, r, r);
   double prevRes = beginRes;
   const int printWidth = 11;

   if (myid == 0)
   {
      cout << std::setw(3) << "It";
      cout << std::setw(printWidth) << "Absres";
      cout << std::setw(printWidth) << "Relres";
      cout << std::setw(printWidth) << "Conv";
      cout << std::setw(printWidth) << "Time [s]" << endl;

      cout << std::setw(3) << 0;
      cout << std::scientific << std::setprecision(3) << std::setw(printWidth) << beginRes;
      cout << std::scientific << std::setprecision(3) << std::setw(printWidth) << 1.0;
      cout << std::scientific << std::setprecision(3) << std::setw(printWidth) << 0.0;
      cout << std::scientific << std::setprecision(3) << std::setw(printWidth) << 0.0 << endl;
   }

   for (int iter = 0; iter < 10; ++iter)
   {
      tic_toc.Clear();
      tic_toc.Start();
      mgCycle.Mult(B, X);
      tic_toc.Stop();

      oprMultigrid.Mult(X, r);
      subtract(B, r, r);

      double res = InnerProduct(MPI_COMM_WORLD, r, r);
      if (myid == 0)
      {
         cout << std::setw(3) << iter + 1;
         cout << std::scientific << std::setprecision(3) << std::setw(printWidth) << res;
         cout << std::scientific << std::setprecision(3) << std::setw(printWidth) << res/beginRes;
         cout << std::scientific << std::setprecision(3) << std::setw(printWidth) << res/prevRes;
         cout << std::scientific << std::setprecision(3) << std::setw(printWidth) << tic_toc.RealTime() << endl;
      }

      if (res < 1e-10 * beginRes)
      {
         break;
      }

      prevRes = res;
   }

   forms[spaceHierarchy.GetFinestLevelIndex()]->RecoverFEMSolution(X, b, x);

   if (visualization)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      socketstream sol_sock(vishost, visport);
      sol_sock << "parallel " << num_procs << " " << myid << "\n";
      sol_sock.precision(8);
      sol_sock << "solution\n" << spaceHierarchy.GetFinestMesh() << x << flush;
   }

   // Missing a bunch of deletes

   MPI_Finalize();

   return 0;
}