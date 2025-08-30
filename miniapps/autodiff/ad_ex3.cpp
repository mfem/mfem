/// Example 3: AD Linear Elasticity with Vector FE
#include "mfem.hpp"
#include "logger.hpp"
#include "ad_intg.hpp"

using namespace std;
using namespace mfem;


int main(int argc, char *argv[])
{
   // file name to be saved
   std::stringstream filename;
   filename << "ad-elasticity";

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
   VectorFunctionCoefficient load_cf(dim, [dim](const Vector &x, Vector &y)
   {
      y.SetSize(dim);
      y = 1.0;
   });

   H1_FECollection fec(order, dim);
   FiniteElementSpace fes(&mesh, &fec, dim);
   Array<int> is_bdr_ess(mesh.bdr_attributes.Max());
   is_bdr_ess = 0;
   is_bdr_ess[3] = 1;
   Array<int> ess_tdof_list;
   fes.GetEssentialTrueDofs(is_bdr_ess, ess_tdof_list);

   real_t lambda(1.0), mu(1.0);
   LinearElasticityEnergy energy(dim, lambda, mu);

   NonlinearForm nlf(&fes);
   nlf.AddDomainIntegrator(
      new ADNonlinearFormIntegrator<ADEval::GRAD | ADEval::VECTOR>(energy));
   nlf.SetEssentialBC(is_bdr_ess);
   LinearForm load(&fes);
   load.AddDomainIntegrator(new VectorDomainLFIntegrator(load_cf));
   load.Assemble();
   load.SetSubVector(ess_tdof_list, 0.0);

   GridFunction x(&fes);
   x = 0.0;
   SparseMatrix &op = static_cast<SparseMatrix&>(nlf.GetGradient(x));
   CGSolver lin_solver;
   GSSmoother prec;
   lin_solver.SetPreconditioner(prec);
   lin_solver.SetOperator(op);
   lin_solver.SetRelTol(1e-12);
   lin_solver.SetAbsTol(0.0);
   lin_solver.SetMaxIter(1e04);
   lin_solver.Mult(load, x);

   if (visualization)
   {
      GLVis glvis("localhost", 19916);
      glvis.Append(x, "x", "Rjc");
   }
   if (paraview)
   {
      std::stringstream pvloc;
      pvloc << "ParaView/" << filename.str();
      ParaViewDataCollection paraview_dc(pvloc.str(), &mesh);
      paraview_dc.SetLevelsOfDetail(order);
      paraview_dc.SetDataFormat(VTKFormat::BINARY);
      paraview_dc.SetHighOrderOutput(true);
      paraview_dc.RegisterField("solution", &x);
      paraview_dc.SetCycle(0);
      paraview_dc.SetTime(0.0);
      paraview_dc.Save();
   }
   return 0;
}
