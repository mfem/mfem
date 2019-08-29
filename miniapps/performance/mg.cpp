#include "mfem-performance.hpp"
#include <fstream>
#include <iostream>

#include "multigrid.hpp"

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

int main(int argc, char *argv[])
{
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
      args.PrintUsage(cout);
      return 1;
   }
   args.PrintOptions(cout);

   // See class BasisType in fem/fe_coll.hpp for available basis types
   int basis = BasisType::GetType(basis_type[0]);
   cout << "Using " << BasisType::Name(basis) << " basis ..." << endl;

   // 2. Read the mesh from the given mesh file. We can handle triangular,
   //    quadrilateral, tetrahedral, hexahedral, surface and volume meshes with
   //    the same code.
   Mesh *mesh = new Mesh(mesh_file, 1, 1);
   // Mesh *mesh = new Mesh(1, 1, 1, Element::HEXAHEDRON, true, 1.0, 1.0, 1.0, false);
   int dim = mesh->Dimension();

   cout << "High-performance version using integration rule with "
         << int_rule_t::qpts << " points ..." << endl;
   if (!mesh_t::MatchesGeometry(*mesh))
   {
      cout << "The given mesh does not match the optimized 'geom' parameter.\n"
            << "Recompile with suitable 'geom' value." << endl;
      delete mesh;
      return 4;
   }
   else if (!mesh_t::MatchesNodes(*mesh))
   {
      cout << "Switching the mesh curvature to match the "
            << "optimized value (order " << mesh_p << ") ..." << endl;
      mesh->SetCurvature(mesh_p, false, -1, Ordering::byNODES);
   }

   FiniteElementCollection *fec = new H1_FECollection(order, dim, basis);

   // Initial refinements of the coarse grid
   for (int i = 0; i < ref_levels; ++i)
   {
      mesh->UniformRefinement();
   }

   // Set up coarse grid finite element space
   FiniteElementSpace* fespace = new FiniteElementSpace(mesh, fec);
   Array<int>* ess_tdof_list = new Array<int>();
   if (mesh->bdr_attributes.Size())
   {
      Array<int> ess_bdr(mesh->bdr_attributes.Max());
      ess_bdr = 1;
      fespace->GetEssentialTrueDofs(ess_bdr, *ess_tdof_list);
   }

   // Construct hierarchy of finite element spaces
   SpaceHierarchy spaceHierarchy(mesh, fespace, ess_tdof_list);

   for (int level = 1; level < mg_levels; ++level)
   {
      Mesh* nextMesh = new Mesh(*spaceHierarchy.GetMesh(level - 1));
      nextMesh->UniformRefinement();
      fespace = new FiniteElementSpace(nextMesh, fec);
      ess_tdof_list = new Array<int>();
      if (nextMesh->bdr_attributes.Size())
      {
         Array<int> ess_bdr(nextMesh->bdr_attributes.Max());
         ess_bdr = 1;
         fespace->GetEssentialTrueDofs(ess_bdr, *ess_tdof_list);
      }

      spaceHierarchy.addLevel(nextMesh, fespace, ess_tdof_list);

      cout << "Number of finite element unknowns on level " << level << ": " << fespace->GetTrueVSize() << endl;
   }

   HPCBilinearForm* coarseForm = new HPCBilinearForm(integ_t(coeff_t(1.0)), *spaceHierarchy.GetFESpace(0));
   coarseForm->Assemble();
   Operator* coarseOpr = nullptr;
   coarseForm->FormSystemOperator(*spaceHierarchy.GetEssentialDoFs(0), coarseOpr);

   CGSolver* coarseSolver = new CGSolver();
   coarseSolver->SetPrintLevel(-1);
   coarseSolver->SetMaxIter(500);
   coarseSolver->SetRelTol(1e-8);
   coarseSolver->SetAbsTol(1e-14);
   coarseSolver->SetOperator(*coarseOpr);

   OperatorMultigrid oprMultigrid(spaceHierarchy, coarseForm, coarseOpr, coarseSolver);

   for(int level = 1; level < spaceHierarchy.GetNumLevels(); ++level)
   {
      // Operator
      HPCBilinearForm* form = new HPCBilinearForm(integ_t(coeff_t(1.0)), *spaceHierarchy.GetFESpace(level));
      form->Assemble();
      Operator* opr = nullptr;
      form->FormSystemOperator(*spaceHierarchy.GetEssentialDoFs(level), opr);

      // Smoother
      Vector diag(spaceHierarchy.GetFESpace(level)->GetVSize());
      BilinearForm paform(spaceHierarchy.GetFESpace(level));
      paform.SetAssemblyLevel(AssemblyLevel::PARTIAL);
      ConstantCoefficient one(1.0);
      paform.AddDomainIntegrator(new DiffusionIntegrator(one));
      paform.Assemble();
      paform.AssembleDiagonal(diag);
      OperatorJacobiSmoother* pa_smoother = new OperatorJacobiSmoother(diag, *spaceHierarchy.GetEssentialDoFs(level), 2.0/3.0);

      // Prolongation
      OperatorHandle Tr(Operator::MFEM_SPARSEMAT);
      spaceHierarchy.GetFESpace(level)->GetTransferOperator(*spaceHierarchy.GetFESpace(level - 1), Tr);
      SparseMatrix* mat; 
      Tr.Get(mat);
      SparseMatrix* prolongation = new SparseMatrix(*mat);

      oprMultigrid.AddLevel(form, opr, pa_smoother, prolongation);
   }

   GridFunction x(spaceHierarchy.GetFinestFESpace());
   x = 0.0;

   LinearForm b(spaceHierarchy.GetFinestFESpace());
   ConstantCoefficient one(1.0);
   b.AddDomainIntegrator(new DomainLFIntegrator(one));
   b.Assemble();

   Vector X, B;
   Operator* dummy = nullptr;

   oprMultigrid.GetFormAtLevel(mg_levels - 1)->FormLinearSystem(*spaceHierarchy.GetEssentialDoFs(mg_levels - 1), x, b, dummy, X, B);

   Vector r(x.Size());
   SolverMultigrid vCycle(oprMultigrid);

   oprMultigrid.Mult(x, r);
   subtract(b, r, r);

   double beginRes = r * r;
   double prevRes = beginRes;

   for (int iter = 0; iter < 10; ++iter)
   {
      vCycle.Mult(b, x);

      oprMultigrid.Mult(x, r);
      subtract(b, r, r);

      double res = r * r;
      cout << "abs/rel/conv " << res << "\t" << res/beginRes << "\t" << res/prevRes << endl;

      prevRes = res;
   }

   oprMultigrid.GetFormAtLevel(mg_levels - 1)->RecoverFEMSolution(X, b, x);

   if (visualization)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      socketstream sol_sock(vishost, visport);
      sol_sock.precision(8);
      sol_sock << "solution\n" << *spaceHierarchy.GetFinestMesh() << x << flush;
   }

   return 0;
}
