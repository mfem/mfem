#include "mfem.hpp"
#include "topopt.hpp"
#include "topopt_problems.hpp"

using namespace mfem;

int main(int argc, char *argv[])
{
   // 1. Initialize MPI and HYPRE.
   Mpi::Init(argc, argv);
   int num_procs = Mpi::WorldSize();
   int myid = Mpi::WorldRank();
   Hypre::Init();

   // 2. Parse command-line options.
   TopoptProblem prob = Cantilever2;
   int ser_ref_levels = 2;
   int par_ref_levels = 4;
   int order_control = 0;
   int order_filter = 1;
   int order_state = 1;
   bool glvis = true;
   bool paraview = true;

   real_t exponent = 3.0;
   real_t rho0 = 1e-06;
   // problem dependent parmeters.
   // Set to -1 to use default value
   real_t E = -1.0;
   real_t nu = -1.0;
   real_t r_min = -1.0;
   real_t max_vol = -1.0;
   real_t min_vol = -1.0;

   OptionsParser args(argc, argv);
   args.AddOption((int*)&prob, "-p", "--problem",
                  "Topology optimization problem. See, topopt_problems.hpp");
   args.AddOption(&ser_ref_levels, "-rs", "--refine-serial",
                  "The number of uniform mesh refinement in serial");
   args.AddOption(&par_ref_levels, "-rp", "--refine-parallel",
                  "The number of uniform mesh refinement in parallel");
   args.AddOption(&order_control, "-oc", "--order-control",
                  "Finite element order (polynomial degree) for control variable.");
   args.AddOption(&order_filter, "-of", "--order-filter",
                  "Finite element order (polynomial degree) for filter variable.");
   args.AddOption(&order_state, "-os", "--order-state",
                  "Finite element order (polynomial degree) for state variable.");
   args.AddOption(&glvis, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&paraview, "-pv", "--paraview", "-no-pv",
                  "--no-paraview",
                  "Enable or disable paraview export.");
   args.Parse();
   if (!args.Good())
   {
      if (myid == 0)
      {
         args.PrintUsage(out);
      }
      return 1;
   }

   real_t lambda(-1.0), mu(-1.0);
   if (lambda > 0) { lambda = E*nu/((1+nu)*(1-2*nu)); }
   if (mu > 0) { mu = E/(2*(1+nu)); }
   Array2D<int> ess_bdr_state;
   Array<int> ess_bdr_filter;
   real_t tot_vol;
   std::unique_ptr<ParMesh> mesh((ParMesh*)GetTopoptMesh(
                                    prob, r_min, tot_vol, min_vol, max_vol,
                                    lambda, mu, ser_ref_levels, par_ref_levels));
   if (myid == 0)
   {
      args.PrintOptions(out);
   }

   const int dim = mesh->SpaceDimension();

   L2_FECollection fec_control(order_control, dim);
   H1_FECollection fec_filter(order_filter, dim);
   H1_FECollection fec_state(order_state, dim);

   ParFiniteElementSpace fes_control(mesh.get(), &fec_control);
   ParFiniteElementSpace fes_filter(mesh.get(), &fec_filter);
   ParFiniteElementSpace fes_state(mesh.get(), &fec_state, dim, Ordering::byNODES);

   ParGridFunction control_gf(&fes_control);
   ParGridFunction control_old_gf(&fes_control);
   ParGridFunction control_eps_gf(&fes_control);
   ParGridFunction filter_gf(&fes_filter);
   ParGridFunction state_gf(&fes_state);
   std::unique_ptr<ParGridFunction> adj_state_gf;
   if (prob < 0) { adj_state_gf.reset(new ParGridFunction(&fes_state)); }

   ParGridFunction grad_gf(&fes_control);
   ParGridFunction grad_old_gf(&fes_control);
   ParGridFunction grad_filter_gf(&fes_filter);

   MappedGFCoefficient simp_cf(filter_gf,
   new std::function<real_t(const real_t)>([exponent, rho0](const real_t x) {return simp(x, exponent, rho0);}));
   MappedGFCoefficient der_simp_cf(filter_gf,
   new std::function<real_t(const real_t)>([exponent, rho0](const real_t x) {return der_simp(x, exponent, rho0);}));
   ProductCoefficient lambda_simp_cf(lambda, simp_cf), mu_simp_cf(mu, simp_cf);
   ConstantCoefficient lambda_cf(lambda), mu_cf(mu);


   FermiDiracEntropy entropy;
   MappedGFCoefficient density_cf = entropy.GetEntropyCoeff(control_gf);
   MappedPairedGFCoefficient bregman_diff_old
      = entropy.GetBregman_dual(control_old_gf, control_gf);
   MappedPairedGFCoefficient bregman_diff_eps
      = entropy.GetBregman_dual(control_eps_gf, control_gf);

   DesignDensity density(fes_control, tot_vol, min_vol, max_vol, &entropy);

   HelmholtzFilter filter(fes_filter, ess_bdr_filter, r_min, true);
   filter.GetLinearForm()->AddDomainIntegrator(new DomainLFIntegrator(density_cf));
   StrainEnergyDensityCoefficient energy(lambda_cf, mu_cf, der_simp_cf, state_gf,
                                         adj_state_gf.get());
   filter.GetAdjLinearForm()->AddDomainIntegrator(new DomainLFIntegrator(energy));

   ElasticityProblem elasticity(fes_state, ess_bdr_state, lambda_simp_cf,
                                mu_simp_cf, prob < 0);

   SetupTopoptProblem(prob, elasticity, filter_gf, state_gf);


}
