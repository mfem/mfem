/// Example 6: Darcy
#include "mfem.hpp"
#include "logger.hpp"
#include "ad_intg.hpp"
#include "tools.hpp"

using namespace std;
using namespace mfem;


struct DarcyFunctional : public ADVectorFunction
{
   int dim;
   // input: q (vector), divq (scalar), u (scalar) -> dim + 1 + 1
   // output: coefficient for w, divw, v -> dim + 1 + 1 (w, v are test functions)
   DarcyFunctional(int dim) : ADVectorFunction(dim + 1 + 1, dim + 1 + 1),
      dim(dim) {}
   // (q, w) + (div w, u) -> res[w] = q, res[divw] = u
   // (div q, v) -> res[v] = div q
   AD_VEC_IMPL(T, V, M, q_divq_u, res,
   {
      res.SetSize(dim + 1 + 1);
      const V q(q_divq_u.GetData(), dim);
      const T divq = q_divq_u[dim];
      const T u = q_divq_u[dim+1];

      V w_cf(res.GetData(), dim);
      T &divw_cf = res[dim];
      T &v_cf = res[dim+1];
      w_cf = q;
      divw_cf = u;
      v_cf = divq;
   });
};

int main(int argc, char *argv[])
{
   Mpi::Init();
   int num_procs = Mpi::WorldSize();
   int myid = Mpi::WorldRank();
   Hypre::Init();
   MPI_Comm comm = MPI_COMM_WORLD;
   // file name to be saved
   std::stringstream filename;
   filename << "ad-darcy-";

   int order = 2;
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
   if (myid != 0) { out.Disable(); }

   // Mesh mesh = rhs_fun_circle
   Mesh ser_mesh = Mesh::MakeCartesian2D(2, 2,
                                         Element::QUADRILATERAL);
   const int dim = ser_mesh.Dimension();
   for (int i = 0; i < ref_levels; i++)
   {
      ser_mesh.UniformRefinement();
   }
   ParMesh mesh(comm, ser_mesh);

   FunctionCoefficient load_cf([](const Vector &x)
   {
      return 2*M_PI * M_PI * std::sin(M_PI * x(0)) * std::sin(M_PI * x(1));
   });
   DarcyFunctional darcy_functional(dim);

   RT_FECollection flux_fec(order, dim);
   L2_FECollection potential_fec(order, dim);
   ParFiniteElementSpace flux_fes(&mesh, &flux_fec);
   ParFiniteElementSpace potential_fes(&mesh, &potential_fec);
   QuadratureSpace visspace(&mesh, order+3);
   const IntegrationRule &ir = IntRules.Get(Geometry::Type::SQUARE, 3*order + 3);

   Array<int> offsets(3);
   offsets[0] = 0;
   offsets[1] = flux_fes.GetTrueVSize();
   offsets[2] = potential_fes.GetTrueVSize();
   offsets.PartialSum();
   BlockVector flux_and_potential(offsets);
   BlockVector rhs(offsets);

   ParGridFunction flux(&flux_fes), potential(&potential_fes);

   flux = 0.0; flux.GetTrueDofs(flux_and_potential.GetBlock(0));
   potential = 0.0; potential.GetTrueDofs(flux_and_potential.GetBlock(1));

   Array<ParFiniteElementSpace*> fespaces{&flux_fes, &potential_fes};
   ParBlockNonlinearForm bnlf(fespaces);
   constexpr ADEval flux_mode = ADEval::VECFE | ADEval::VALUE | ADEval::DIV;
   constexpr ADEval potential_mode = ADEval::VALUE;
   bnlf.AddDomainIntegrator(
      new ADBlockNonlinearFormIntegrator<flux_mode, potential_mode>
      (darcy_functional, &ir)
   );

   ParLinearForm b(&potential_fes);
   b.AddDomainIntegrator(new DomainLFIntegrator(load_cf));
   b.Assemble();
   b.ParallelAssemble(rhs.GetBlock(1));
   rhs.GetBlock(1).Neg();
   rhs.GetBlock(0) = 0.0;

   MUMPSMonoSolver lin_solver(comm);
   lin_solver.SetOperator(bnlf.GetGradient(flux_and_potential));
   lin_solver.Mult(rhs, flux_and_potential);
   flux.SetFromTrueDofs(flux_and_potential.GetBlock(0));
   potential.SetFromTrueDofs(flux_and_potential.GetBlock(1));

   GLVis glvis("localhost", 19916, 400, 350, 3);
   glvis.Append(flux, "flux", "RjclQmm");
   glvis.Append(potential, "potential", "Rjclmm");

   FunctionCoefficient exact_potential([](const Vector &x)
   {
      return std::sin(M_PI * x(0)) * std::sin(M_PI * x(1));
   });
   VectorFunctionCoefficient exact_flux(dim, [](const Vector &x, Vector &q)
   {
      q.SetSize(x.Size());
      q[0] = M_PI*std::cos(M_PI*x[0])*std::sin(M_PI*x[1]);
      q[1] = M_PI*std::sin(M_PI*x[0])*std::cos(M_PI*x[1]);
   });
   out << "L2 Error in Potential: "
       << potential.ComputeL2Error(exact_potential) << std::endl;
   out << "L2 Error in Flux: "
       << flux.ComputeL2Error(exact_flux) << std::endl;

   return 0;
}
