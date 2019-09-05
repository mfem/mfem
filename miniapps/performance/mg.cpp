#include "mfem-performance.hpp"
#include <fstream>
#include <iostream>

#include "multigrid.hpp"
#include "eigenvalue.hpp"

using namespace std;
using namespace mfem;

// Define template parameters for optimized build.
const Geometry::Type geom     = Geometry::CUBE; // mesh elements  (default: hex)
const int            mesh_p   = 1;              // mesh curvature (default: 3)
const int            sol_p    = 3;              // solution order (default: 3)
const int            rdim     = Geometry::Constants<geom>::Dimension;
const int            ir_order = 2*sol_p+rdim-1;

// Static mesh type
typedef H1_FiniteElement<geom,mesh_p>         mesh_fe_t;
typedef H1_FiniteElementSpace<mesh_fe_t>      mesh_fes_t;
typedef TMesh<mesh_fes_t>                     mesh_t;

// Static solution finite element space type
typedef H1_FiniteElement<geom,sol_p>          sol_fe_t;
typedef H1_FiniteElementSpace<sol_fe_t>       sol_fes_t;

// Static quadrature, coefficient and integrator types
typedef TIntegrationRule<geom,ir_order>       int_rule_t;
typedef TConstantCoefficient<>                coeff_t;
typedef TIntegrator<coeff_t,TDiffusionKernel> integ_t;

// Static bilinear form type, combining the above types
typedef TBilinearForm<mesh_t,sol_fes_t,int_rule_t,integ_t> HPCBilinearForm;

void getEssentialTrueDoFs(Mesh* mesh, FiniteElementSpace* fespace, Array<int>& ess_tdof_list)
{
   if (mesh->bdr_attributes.Size())
   {
      Array<int> ess_bdr(mesh->bdr_attributes.Max());
      ess_bdr = 1;
      fespace->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
   }
}

int main(int argc, char *argv[])
{
   int num_procs, myid;
   MPI_Init(&argc, &argv);
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
   MPI_Comm_rank(MPI_COMM_WORLD, &myid);

   // 1. Parse command-line options.
   const char *mesh_file = "../../data/fichera.mesh";
   int ref_levels = 2;
   int mg_levels = 3;
   const int order = sol_p;
   const char *basis_type = "G"; // Gauss-Lobatto
   bool visualization = 1;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
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

   if (myid == 0)
   {
      cout << "High-performance version using integration rule with "
           << int_rule_t::qpts << " points ..." << endl;
   }
   if (!mesh_t::MatchesGeometry(*mesh))
   {
      if (myid == 0)
      {
         cout << "The given mesh does not match the optimized 'geom' parameter.\n"
               << "Recompile with suitable 'geom' value." << endl;
      }
      delete mesh;
      MPI_Finalize();
      return 4;
   }
   else if (!mesh_t::MatchesNodes(*mesh))
   {
      if (myid == 0)
      {
         cout << "Switching the mesh curvature to match the "
              << "optimized value (order " << mesh_p << ") ..." << endl;
      }
      mesh->SetCurvature(mesh_p, false, -1, Ordering::byNODES);
   }

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
      cout << "Number of finite element unknowns on level 0: " << size << endl;
   }

   Array<int>* essentialTrueDoFs = new Array<int>();
   getEssentialTrueDoFs(pmesh, fespace, *essentialTrueDoFs);

   // Construct hierarchy of finite element spaces
   ParSpaceHierarchy spaceHierarchy(pmesh, fespace);

   for (int level = 1; level < mg_levels; ++level)
   {
      ParMesh* nextMesh = new ParMesh(*spaceHierarchy.GetMesh(level - 1));
      nextMesh->UniformRefinement();
      fespace = new ParFiniteElementSpace(nextMesh, fec);
      spaceHierarchy.addLevel(nextMesh, fespace);

      size = fespace->GlobalTrueVSize();

      if (myid == 0)
      {
         cout << "Number of finite element unknowns on level " << level << ": " << size << endl;
      }
   }

   HPCBilinearForm* coarseForm = new HPCBilinearForm(integ_t(coeff_t(1.0)), *spaceHierarchy.GetFESpace(0));
   coarseForm->Assemble();
   Operator* coarseOpr = nullptr;
   coarseForm->FormSystemOperator(*essentialTrueDoFs, coarseOpr);

   CGSolver* coarseSolver = new CGSolver(MPI_COMM_WORLD);
   coarseSolver->SetPrintLevel(-1);
   coarseSolver->SetMaxIter(500);
   coarseSolver->SetRelTol(1e-8);
   coarseSolver->SetAbsTol(1e-14);
   coarseSolver->SetOperator(*coarseOpr);

   OperatorMultigrid oprMultigrid(coarseForm, coarseOpr, coarseSolver, essentialTrueDoFs);

   for(int level = 1; level < spaceHierarchy.GetNumLevels(); ++level)
   {
      // Operator
      tic_toc.Clear();
      tic_toc.Start();
      if (myid == 0)
      {
         cout << "Partially assemble HPC form on level " << level << "..." << flush;
      }
      HPCBilinearForm* form = new HPCBilinearForm(integ_t(coeff_t(1.0)), *spaceHierarchy.GetFESpace(level));
      form->Assemble();
      Operator* opr = nullptr;
      essentialTrueDoFs = new Array<int>();
      getEssentialTrueDoFs(spaceHierarchy.GetMesh(level), spaceHierarchy.GetFESpace(level), *essentialTrueDoFs);
      form->FormSystemOperator(*essentialTrueDoFs, opr);
      tic_toc.Stop();
      if (myid == 0)
      {
         cout << "\tdone, " << tic_toc.RealTime() << "s." << endl;
      }

      // Smoother
      tic_toc.Clear();
      tic_toc.Start();
      if (myid == 0)
      {
         cout << "Partially assemble diagonal on level " << level << "..." << flush;
      }
      Vector diag(spaceHierarchy.GetFESpace(level)->GetTrueVSize());
      ParBilinearForm paform(spaceHierarchy.GetFESpace(level));
      paform.SetAssemblyLevel(AssemblyLevel::PARTIAL);
      ConstantCoefficient one(1.0);
      paform.AddDomainIntegrator(new DiffusionIntegrator(one));
      paform.Assemble();
      paform.AssembleDiagonal(diag);
      OperatorJacobiSmoother* pa_smoother_one = new OperatorJacobiSmoother(diag, *essentialTrueDoFs, 1.0);
      tic_toc.Stop();
      if (myid == 0)
      {
         cout << "\tdone, " << tic_toc.RealTime() << "s." << endl;
      }

      // Prolongation
      tic_toc.Clear();
      tic_toc.Start();
      if (myid == 0)
      {
         cout << "Constructing prolongation matrix on level " << level << "..." << flush;
      }
      OperatorHandle* P = new OperatorHandle(Operator::Hypre_ParCSR);
      spaceHierarchy.GetFESpace(level)->GetTrueTransferOperator(*spaceHierarchy.GetFESpace(level - 1), *P);
      Operator* prolongation = P->Ptr();
      tic_toc.Stop();
      if (myid == 0)
      {
         cout << "\tdone, " << tic_toc.RealTime() << "s." << endl;
      }

      tic_toc.Clear();
      tic_toc.Start();
      if (myid == 0)
      {
         cout << "Estimating eigenvalues on level " << level << "..." << flush;
      }
      ParGridFunction ev(spaceHierarchy.GetFinestFESpace());
      ProductOperator DinvA(pa_smoother_one, opr, false, false);
      double eigval = PowerMethod::EstimateLargestEigenvalue(DinvA, ev, 20, 1e-8);
      tic_toc.Stop();
      if (myid == 0)
      {
         cout << "\t\tdone, " << tic_toc.RealTime() << "s." << endl;
      }

      OperatorChebyshevSmoother* pa_smoother = new OperatorChebyshevSmoother(opr, diag, *essentialTrueDoFs, 5, eigval);

      oprMultigrid.AddLevel(form, opr, pa_smoother, prolongation, essentialTrueDoFs);
   }

   ParGridFunction x(spaceHierarchy.GetFinestFESpace());
   x = 0.0;

   if (myid == 0)
   {
      cout << "Assembling rhs..." << flush;
   }
   tic_toc.Clear();
   tic_toc.Start();
   ParLinearForm b(spaceHierarchy.GetFinestFESpace());
   ConstantCoefficient one(1.0);
   b.AddDomainIntegrator(new DomainLFIntegrator(one));
   b.Assemble();
   tic_toc.Stop();
   if (myid == 0)
   {
      cout << "\t\t\t\tdone, " << tic_toc.RealTime() << "s." << endl;
   }

   Vector X, B;
   Operator* dummy = nullptr;

   oprMultigrid.GetFormAtFinestLevel()->FormLinearSystem(*oprMultigrid.GetEssentialDoFsAtFinestLevel(), x, b, dummy, X, B);

   Vector r(X.Size());
   SolverMultigrid vCycle(oprMultigrid);

   oprMultigrid.Mult(X, r);
   subtract(B, r, r);

   double beginRes = r * r;
   double prevRes = beginRes;

   for (int iter = 0; iter < 10; ++iter)
   {
      vCycle.Mult(B, X);

      oprMultigrid.Mult(X, r);
      subtract(B, r, r);

      double res = r * r;
      if (myid == 0)
      {
         cout << "abs/rel/conv " << std::scientific << std::setprecision(3) << res << "\t" << res/beginRes << "\t" << res/prevRes << endl;
      }

      if (res < 1e-10 * beginRes)
      {
         break;
      }

      prevRes = res;
   }

   oprMultigrid.GetFormAtFinestLevel()->RecoverFEMSolution(X, b, x);

   if (visualization)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      socketstream sol_sock(vishost, visport);
      sol_sock << "parallel " << num_procs << " " << myid << "\n";
      sol_sock.precision(8);
      sol_sock << "solution\n" << *spaceHierarchy.GetFinestMesh() << x << flush;
   }

   // Missing a bunch of deletes
   
   MPI_Finalize();

   return 0;
}