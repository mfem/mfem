/// Example 2: AD Minimal Surface
#include "mfem.hpp"
#include "logger.hpp"
#include "ad_intg.hpp"

using namespace std;
using namespace mfem;

struct MinimalSurfaceEnergy : public ADFunction
{
public:
   real_t eps=0.5; // regularization
   MinimalSurfaceEnergy(int dim): ADFunction(dim) {}
   AD_IMPL(T, V, M, gradu,
   {
      T h1_norm(gradu*gradu);
      // sqrt(1+ ||grad u||^2)
      // dJ/du = 0 -> minimal surface
      return sqrt(h1_norm + 1.0) + eps*h1_norm;
   });
};

int main(int argc, char *argv[])
{
   // file name to be saved
   std::stringstream filename;
   filename << "ad-minimalsurface-";

   int order = 1;
   int ref_levels = 3;
   bool visualization = false;
   bool paraview = false;

   OptionsParser args(argc, argv);
   args.AddOption(&order, "-o", "--order",
                  "Finite element polynomial degree");
   args.AddOption(&ref_levels, "-r", "--ref", "Refinement levels");
   args.AddOption(&visualization, "-vis", "--visualization",
                  "-no-vis", "--no-visualization",
                  "Enable visualization, default is false");
   args.AddOption(&paraview, "-pv", "--paraview",
                  "-no-pv", "--no-paraview",
                  "Enable Paraview Export. Default is false");
   args.ParseCheck();

   // Mesh mesh = rhs_fun_circle
   Mesh mesh = Mesh::MakeCartesian2D(10, 10,
                                     Element::QUADRILATERAL);
   const int dim = mesh.Dimension();
   for (int i = 0; i < ref_levels; i++)
   {
      mesh.UniformRefinement();
   }
   FunctionCoefficient bdry_cf([](const Vector &x)
   {
      real_t theta = std::atan2(x(1)-0.5, x(0)-0.5);
      real_t r = std::sqrt(std::pow(x(0)-0.5, 2.0) + std::pow(x(1)-0.5, 2.0));
      return r*std::cos(2*theta);
   });

   H1_FECollection fec(order, dim);
   FiniteElementSpace fes(&mesh, &fec);
   Array<int> is_bdr_ess(mesh.bdr_attributes.Max());
   is_bdr_ess = 1;

   MinimalSurfaceEnergy energy(dim);

   NonlinearForm nlf(&fes);
   nlf.AddDomainIntegrator(new ADNonlinearFormIntegrator<ADEval::GRAD>
                           (energy));
   nlf.SetEssentialBC(is_bdr_ess);

   GridFunction x(&fes);
   x = 0.0;
   x.ProjectBdrCoefficient(bdry_cf, is_bdr_ess);
   NewtonSolver solver;
   UMFPackSolver lin_solver;
   solver.SetSolver(lin_solver);
   solver.SetOperator(nlf);
   solver.SetAbsTol(1e-10);
   solver.SetRelTol(1e-10);
   IterativeSolver::PrintLevel print_level;
   print_level.iterations = 1;
   solver.SetPrintLevel(print_level);
   solver.SetMaxIter(100);
   solver.iterative_mode = true;
   Vector dummy(0);
   GLVis glvis("localhost", 19916);
   glvis.Append(x, "x", "Rjc");
   glvis.Update();
   for (int i=0; i<30; i++)
   {
      solver.Mult(dummy, x);
      glvis.Update();
      energy.eps *= 0.5;
   }
   return 0;
}
