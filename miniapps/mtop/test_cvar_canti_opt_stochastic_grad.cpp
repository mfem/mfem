#include "mfem.hpp"
#include "mtop_solvers.hpp"

#include <array>
#include <bitset>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "test_cvar_canti_opt_stochastic_initialization_helpers.hpp"
#include "test_cvar_canti_opt_stochastic_simpl_helpers.hpp"

using namespace std;
using namespace mfem;

namespace
{

void SetScenarioDispBCs(IsoLinElasticSolver &elsolver,
                        const std::bitset<N> &bits,
                        const std::array<int, N> &bit_index_to_mesh_bc_index)
{
   elsolver.DelDispBC();
   for (int i = 0; i < N; i++)
   {
      if (!bits.test(i))
      {
         elsolver.AddDispBC(bit_index_to_mesh_bc_index[i], 4, 0.0);
      }
   }
}

real_t ComputeCompliance(ParFiniteElementSpace *filter_fes,
                         Coefficient &compliance_coeff,
                         ParGridFunction &onegf)
{
   ParLinearForm compl_form(filter_fes);
   compl_form.AddDomainIntegrator(new DomainLFIntegrator(compliance_coeff));
   compl_form.Assemble();
   return compl_form(onegf);
}

} // namespace

int main(int argc, char *argv[])
{
   Mpi::Init();
   int myid = Mpi::WorldRank();
   Hypre::Init();

   // const char *mesh_file = "canti_2D_6.msh";   // SYMMETRIC MESH
   const char *mesh_file = "canti_n_2D_6.msh"; // ASYMMETRIC MESH
   int order = 2;
   const char *device_config = "cpu";
   real_t filter_radius = real_t(0.1);
   real_t E_min = 1e-3;
   real_t E_max = 1.0;
   real_t poisson_ratio = 0.2;
   real_t odens_init = 0.30;

   // canti_2D_6.msh (symmetric)
   // const int fixed_node_bc_index = 1;
   // const int outer_boundary_bc_index = 8;
   // const std::array<int, N> bit_index_to_mesh_bc_index = {2, 3, 4, 5, 6, 7};

   // canti_n_2D_6.msh (asymmetric)
   const int fixed_node_bc_index = 5;
   const int outer_boundary_bc_index = 8;
   const std::array<int, N> bit_index_to_mesh_bc_index = {2, 3, 4, 6, 7, 1};

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh", "Mesh file to use.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree) or -1 for isoparametric space.");
   args.AddOption(&device_config, "-d", "--device",
                  "Device configuration string, see Device::Configure().");
   args.AddOption(&filter_radius, "-fr", "--filter", "Set the filter radius.");
   args.AddOption(&odens_init, "-rho", "--initial-density",
                  "Initial constant design density in [0,1].");
   args.Parse();
   if (!args.Good())
   {
      if (myid == 0) { args.PrintUsage(cout); }
      return 1;
   }
   if (myid == 0) { args.PrintOptions(cout); }

   if (myid == 0)
   {
      std::cout << std::scientific << std::setprecision(8);
   }

   Device device(device_config);
   if (myid == 0) { device.Print(); }

   Mesh mesh(mesh_file, 1, 1);
   int dim = mesh.Dimension();

   {
      int ref_levels = (int)floor(log(1000. / mesh.GetNE()) / log(2.) / dim) + 1;
      for (int l = 0; l < ref_levels; l++)
      {
         mesh.UniformRefinement();
      }
   }

   ParMesh pmesh(MPI_COMM_WORLD, mesh);
   mesh.Clear();
   {
      int par_ref_levels = 1;
      for (int l = 0; l < par_ref_levels; l++)
      {
         pmesh.UniformRefinement();
      }
   }

   FilterOperator *filt = new FilterOperator(filter_radius, &pmesh);
   filt->AddBC(fixed_node_bc_index, 1.0);
   for (int i = 0; i < N; i++)
   {
      filt->AddBC(bit_index_to_mesh_bc_index[i], 1.0);
   }
   filt->AddBC(outer_boundary_bc_index, 0.0);
   filt->Assemble();

   ParGridFunction odens(filt->GetDesignFES());
   ParGridFunction fdens(filt->GetFilterFES());
   odens = odens_init;
   odens.SetTrueVector();

   Vector odv(filt->GetDesignFES()->GetTrueVSize());
   Vector fdv(filt->GetFilterFES()->GetTrueVSize());
   odens.GetTrueDofs(odv);
   filt->Mult(odv, fdv);
   fdens.SetFromTrueDofs(fdv);

   IsoLinElasticSolver *elsolver = new IsoLinElasticSolver(&pmesh, order);
   elsolver->AddSurfLoad(fixed_node_bc_index, 0.0, 1.0);

   IsoComplCoef icc;
   icc.SetGridFunctions(&fdens, &(elsolver->GetDisplacements()));
   icc.SetMaterial(E_min, E_max, poisson_ratio);

   elsolver->SetMaterial(*(icc.GetE()), *(icc.GetNu()));

   ConstantCoefficient one(1.0);
   ConstantCoefficient zero(0.0);
   ParGridFunction onegf(filt->GetFilterFES());
   onegf = 1.0;

   ParBilinearForm mass(filt->GetDesignFES());
   mass.AddDomainIntegrator(new InverseIntegrator(new MassIntegrator(one)));
   mass.Assemble();
   HypreParMatrix Minv;
   Array<int> empty;
   mass.FormSystemMatrix(empty, Minv);

   ParGridFunction odens_gradient_single(filt->GetDesignFES());
   Vector tmp_grad(filt->GetDesignFES()->GetTrueVSize());
   Vector ograd(filt->GetDesignFES()->GetTrueVSize());

   std::vector<std::pair<std::string, std::bitset<N>>> scenarios;
   scenarios.push_back({"no_failure", std::bitset<N>(std::string("000000"))});
   scenarios.push_back({"single_failure_bit0", std::bitset<N>(std::string("000001"))});
   scenarios.push_back({"single_failure_bit2", std::bitset<N>(std::string("000100"))});

   // Small trial step sizes for descent checks with the helper-computed update direction.
   std::vector<real_t> epsilons = {1e-2, 1e-3, 1e-4};

   if (myid == 0)
   {
      std::cout << "Running stochastic-scenario gradient check with "
                << scenarios.size() << " scenarios." << std::endl;
   }

   bool local_descent_check_failed = false;

   for (size_t s = 0; s < scenarios.size(); s++)
   {
      const std::string &scenario_name = scenarios[s].first;
      const std::bitset<N> &bits = scenarios[s].second;

      SetScenarioDispBCs(*elsolver, bits, bit_index_to_mesh_bc_index);
      elsolver->Assemble();
      elsolver->FSolve();

      // Refresh pointers after each scenario assembly/solve.
      ParGridFunction &sol = elsolver->GetDisplacements();
      icc.SetGridFunctions(&fdens, &sol);

      const real_t base_compliance = ComputeCompliance(filt->GetFilterFES(), icc, onegf);

      real_t grad_norm_after_afilter = 0.0;
      real_t grad_norm_after_smoothing = 0.0;
      bool grad_ok = compute_smoothed_design_gradient(*filt, icc, Minv,
                                                      odens_gradient_single, tmp_grad,
                                                      grad_norm_after_afilter,
                                                      grad_norm_after_smoothing);
      if (!grad_ok)
      {
         if (myid == 0)
         {
            std::cout << "[" << scenario_name << "] gradient had non-finite intermediate values; skipping." << std::endl;
         }
         continue;
      }

      odens_gradient_single.GetTrueDofs(ograd);

      Vector descent_step(odens.GetTrueVector().Size());
      Vector perturbed_odv(odens.GetTrueVector().Size());
      Vector perturbed_fdv(fdens.GetTrueVector().Size());

      // Test the exact direction the optimizer would use: helper output in true-dof space.
      descent_step = ograd;
      const real_t descent_norm_sq = InnerProduct(pmesh.GetComm(), descent_step, descent_step);
      if (!std::isfinite(descent_norm_sq) || descent_norm_sq <= 0.0)
      {
         if (myid == 0)
         {
            std::cout << "[" << scenario_name
                      << "] smoothed descent direction has non-finite or zero norm; skipping." << std::endl;
         }
         continue;
      }
      descent_step /= std::sqrt(descent_norm_sq);

      odens.GetTrueDofs(odv);

      if (myid == 0)
      {
         std::cout << "\nScenario: " << scenario_name
                   << " bits=" << bits.to_string()
                   << " base_compliance=" << base_compliance
                   << " grad_norm_after_afilter=" << grad_norm_after_afilter
                   << " grad_norm_after_smoothing=" << grad_norm_after_smoothing
                   << std::endl;
      }

      for (size_t e = 0; e < epsilons.size(); e++)
      {
         const real_t eps = epsilons[e];

         perturbed_odv = odv;
         perturbed_odv.Add(eps, descent_step);

         // Keep densities in physical range for robust elasticity solves.
         for (int i = 0; i < perturbed_odv.Size(); i++)
         {
            perturbed_odv[i] = std::max(real_t(1e-6), std::min(real_t(1.0 - 1e-6), perturbed_odv[i]));
         }

         filt->Mult(perturbed_odv, perturbed_fdv);
         fdens.SetFromTrueDofs(perturbed_fdv);

         elsolver->FSolve();
         const real_t perturbed_compliance = ComputeCompliance(filt->GetFilterFES(), icc, onegf);
         const real_t fd_slope = (perturbed_compliance - base_compliance) / eps;
         const bool is_descent = (fd_slope < 0.0);

         if (myid == 0)
         {
            std::cout << "  eps=" << eps
                      << " base=" << base_compliance
                      << " perturbed=" << perturbed_compliance
                      << " fd_slope_along_smoothed=" << fd_slope
                      << " descent_ok=" << (is_descent ? "true" : "false")
                      << std::endl;

            if (!is_descent)
            {
               std::cout << "  ERROR: non-descent step detected for scenario "
                         << scenario_name << " at eps=" << eps << std::endl;
            }
         }

         if (!is_descent)
         {
            local_descent_check_failed = true;
         }
      }

      // Restore base state before the next scenario.
      fdens.SetFromTrueDofs(fdv);
      elsolver->FSolve();
   }

   delete elsolver;
   delete filt;

   int local_fail = local_descent_check_failed ? 1 : 0;
   int global_fail = 0;
   MPI_Allreduce(&local_fail, &global_fail, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);

   if (myid == 0)
   {
      if (global_fail)
      {
         std::cout << "Gradient descent check FAILED." << std::endl;
      }
      else
      {
         std::cout << "Gradient descent check PASSED." << std::endl;
      }
   }

   Mpi::Finalize();
   return global_fail ? 1 : 0;
}
