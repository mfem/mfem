// Thermal Compliance minimization with projected mirror descent OC in parallel
//
//                  minimize F(ρ) = ∫_Ω f⋅u dx over ρ ∈ L¹(Ω)
//
//                  subject to
//
//                    -∇⋅(K ∇u) = f             in Ω + BCs
//                    -ϵ²Δρ̃ + ρ̃ = ρ             in Ω + Neumann BCs
//                    0 ≤ ρ ≤ 1                 in Ω
//                    ∫_Ω ρ dx = θ vol(Ω)
//
//              Here, r(ρ̃) = ρ₀ + ρ̃³ (1-ρ₀) is the solid isotropic material
//              penalization (SIMP) law, K is the diffusion coefficient,
//              ϵ > 0 is the design length scale, and 0 < θ < 1 is the volume fraction.
//
//              Update is done by
//
//              ρ_new = sigmoid(ψ_new) = sigmoid(ψ_cur - α ∇F(ρ_cur) + c)
//
//              where c is a constant volume correction. The step size α is
//              determined by a generalized Barzilai-Borwein method with
//              Armijo condition check
//
//              BB:        α_init = |(δψ, δρ) / (δ∇F(ρ), δρ)|
//
//              Armijo:   F(ρ(α)) ≤ F(ρ_cur) + c_1 (∇F(ρ_cur), ρ(α) - ρ_cur)
//                        with ρ(α) = sigmoid(ψ_cur - α∇F(ρ_cur) + c)
//
//
// [1] Andreassen, E., Clausen, A., Schevenels, M., Lazarov, B. S., & Sigmund, O.
//    (2011). Efficient topology optimization in MATLAB using 88 lines of
//    code. Structural and Multidisciplinary Optimization, 43(1), 1-16.
// [2] Keith, B. and Surowiec, T. (2023) Proximal Galerkin: A structure-
//     preserving finite element method for pointwise bound constraints.
//     arXiv:2307.12444 [math.NA]
// [3] Lazarov, B. S., & Sigmund, O. (2011). Filters in topology optimization
//     based on Helmholtz‐type differential equations. International Journal
//     for Numerical Methods in Engineering, 86(6), 765-781.

#include "mfem.hpp"
#include <iostream>
#include <fstream>
#include "topopt.hpp"
#include "helper.hpp"
#include "prob_thermal.hpp"

using namespace std;
using namespace mfem;

int main(int argc, char *argv[])
{
   Mpi::Init();
   int num_procs = Mpi::WorldSize();
   int myid = Mpi::WorldRank();
   Hypre::Init();
   if (Mpi::Root()) { mfem::out << "Parallel run using " << num_procs << " processes" << std::endl; }

   // 1. Parse command-line options.
   int seq_ref_levels = 0;
   int par_ref_levels = 6;
   int order = 1;
   // filter radius. Use problem-dependent default value if not provided.
   // See switch statements below
   double filter_radius = -1;
   // Volume fraction. Use problem-dependent default value if not provided.
   // See switch statements below
   double vol_fraction = -1;
   int max_it = 5e2;
   double rho_min = 1e-06;
   double exponent = 3.0;
   double K=1.0;
   double c1 = 1e-04;
   bool glvis_visualization = true;
   bool save = false;
   bool paraview = true;
   double tol_stationarity = 1e-04;
   double tol_compliance = 5e-05;

   ostringstream filename_prefix;
   filename_prefix << "PMD_OC-";

   int problem = ThermalProblem::HeatSink;

   OptionsParser args(argc, argv);
   args.AddOption(&seq_ref_levels, "-rs", "--seq-refine",
                  "Number of times to refine the sequential mesh uniformly.");
   args.AddOption(&par_ref_levels, "-rp", "--par-refine",
                  "Number of times to refine the parallel mesh uniformly.");
   args.AddOption(&problem, "-p", "--problem",
                  "Problem number: 0) HeatSink");
   args.AddOption(&order, "-o", "--order",
                  "Order (degree) of the finite elements.");
   args.AddOption(&filter_radius, "-fr", "--filter-radius",
                  "Length scale for ρ.");
   args.AddOption(&max_it, "-mi", "--max-it",
                  "Maximum number of gradient descent iterations.");
   args.AddOption(&vol_fraction, "-vf", "--volume-fraction",
                  "Volume fraction for the material density.");
   args.AddOption(&rho_min, "-rmin", "--rho-min",
                  "Minimum of density coefficient.");
   args.AddOption(&glvis_visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.Parse();
   if (!args.Good()) {if (Mpi::Root()) args.PrintUsage(mfem::out);}


   std::unique_ptr<Mesh> mesh;
   Array2D<int> ess_bdr;
   Array<int> ess_bdr_filter;
   std::unique_ptr<Coefficient> vforce_cf;
   std::string prob_name;
   GetThermalProblem((ThermalProblem)problem, filter_radius, vol_fraction,
                     mesh, vforce_cf,
                     ess_bdr, ess_bdr_filter,
                     prob_name, seq_ref_levels, par_ref_levels);
   filename_prefix << prob_name << "-" << seq_ref_levels + par_ref_levels;
   int dim = mesh->Dimension();
   const int num_el = mesh->GetNE();
   std::unique_ptr<ParMesh> pmesh(static_cast<ParMesh*>(mesh.release()));

   if (Mpi::Root())
      mfem::out << "\n"
                << "Thermal Compliance Minimization with Projected Mirror Descent OC.\n"
                << "Problem: " << filename_prefix.str() << "\n"
                << "The number of elements: " << num_el << "\n"
                << "Order: " << order << "\n"
                << "Volume Fraction: " << vol_fraction << "\n"
                << "Filter Radius: " << filter_radius << "\n"
                << "Maximum iteration: " << max_it << "\n"
                << "GLVis: " << glvis_visualization << "\n"
                << "Paraview: " << paraview << std::endl;

   if (glvis_visualization && dim == 3)
   {
      glvis_visualization = false;
      paraview = true;
      if (Mpi::Root()) { mfem::out << "GLVis for 3D is disabled. Use ParaView" << std::endl; }
   }
   pmesh->SetAttributes();

   if (save)
   {
      ostringstream meshfile;
      meshfile << filename_prefix.str() << "." << setfill('0') << setw(6) << myid;
      ofstream mesh_ofs(meshfile.str().c_str());
      mesh_ofs.precision(8);
      pmesh->Print(mesh_ofs);
   }

   // 4. Define the necessary finite element spaces on the mesh.
   H1_FECollection state_fec(order, dim); // space for u
   H1_FECollection filter_fec(order, dim); // space for ρ̃
   L2_FECollection control_fec(order-1, dim,
                               BasisType::GaussLobatto); // space for ψ
   ParFiniteElementSpace state_fes(pmesh.get(), &state_fec);
   ParFiniteElementSpace filter_fes(pmesh.get(), &filter_fec);
   ParFiniteElementSpace control_fes(pmesh.get(), &control_fec);

   int state_size = state_fes.GlobalTrueVSize();
   int control_size = control_fes.GlobalTrueVSize();
   int filter_size = filter_fes.GlobalTrueVSize();
   if (Mpi::Root())
   {
      mfem::out << "\n"
                << "Number of state unknowns: " << state_size << "\n"
                << "Number of filter unknowns: " << filter_size << "\n"
                << "Number of control unknowns: " << control_size << std::endl;
   }

   // 5. Set the initial guess for ρ.
   SIMPProjector simp_rule(exponent, rho_min);
   HelmholtzFilter filter(filter_fes, filter_radius/(2.0*sqrt(3.0)),
                          ess_bdr_filter);
   LatentDesignDensity density(control_fes, filter, vol_fraction,
                               ShannonEntropy, log_d, exp_d, false, true);

   ConstantCoefficient K_cf(K);
   ParametrizedDiffusionEquation diffusion(state_fes,
                                           density.GetFilteredDensity(), simp_rule, K_cf, *vforce_cf, ess_bdr);
   TopOptProblem optprob(diffusion.GetLinearForm(), diffusion, density, false,
                         true);

   ParGridFunction &u = *dynamic_cast<ParGridFunction*>(&optprob.GetState());
   ParGridFunction &rho_filter = *dynamic_cast<ParGridFunction*>
                                 (&density.GetFilteredDensity());
   // 10. Connect to GLVis. Prepare for VisIt output.
   char vishost[] = "localhost";
   int  visport   = 19916;
   socketstream sout_SIMP, sout_r;
   std::unique_ptr<ParGridFunction> designDensity_gf, rho_gf;
   if (glvis_visualization)
   {
      MPI_Barrier(MPI_COMM_WORLD);
      designDensity_gf.reset(new ParGridFunction(&filter_fes));
      rho_gf.reset(new ParGridFunction(&filter_fes));
      designDensity_gf->ProjectCoefficient(simp_rule.GetPhysicalDensity(
                                              density.GetFilteredDensity()));
      rho_gf->ProjectCoefficient(density.GetDensityCoefficient());
      sout_SIMP.open(vishost, visport);
      if (sout_SIMP.is_open())
      {
         sout_SIMP << "parallel " << num_procs << " " << myid << "\n";
         sout_SIMP.precision(8);
         sout_SIMP << "solution\n" << *pmesh << *designDensity_gf
                   << "window_title 'Design density r(ρ̃) - PMD_OC "
                   << problem << "'\n"
                   << "keys Rjl***************\n"
                   << flush;
         MPI_Barrier(MPI_COMM_WORLD); // try to prevent streams from mixing
      }
      sout_r.open(vishost, visport);
      if (sout_r.is_open())
      {
         sout_r << "parallel " << num_procs << " " << myid << "\n";
         sout_r.precision(8);
         sout_r << "solution\n" << *pmesh << *rho_gf
                << "window_title 'Raw density ρ - PMD_OC "
                << problem << "'\n"
                << "keys Rjl***************\n"
                << flush;
         MPI_Barrier(MPI_COMM_WORLD); // try to prevent streams from mixing
      }
   }
   std::unique_ptr<ParaViewDataCollection> pd;
   if (paraview)
   {
      pd.reset(new ParaViewDataCollection(filename_prefix.str(), pmesh.get()));
      pd->SetPrefixPath("ParaView");
      pd->RegisterField("state", &u);
      pd->RegisterField("psi", &density.GetGridFunction());
      pd->RegisterField("frho", &rho_filter);
      pd->SetLevelsOfDetail(order);
      pd->SetDataFormat(VTKFormat::BINARY);
      pd->SetHighOrderOutput(true);
      pd->SetCycle(0);
      pd->SetTime(0);
      pd->Save();
   }

   // 11. Iterate
   ParGridFunction &grad(*dynamic_cast<ParGridFunction*>(&optprob.GetGradient()));
   ParGridFunction &psi(*dynamic_cast<ParGridFunction*>
                        (&density.GetGridFunction()));
   ParGridFunction old_d(&control_fes), old_psi(&control_fes);

   ParLinearForm diff_rho_form(&control_fes);
   std::unique_ptr<Coefficient> diff_rho(optprob.GetDensityDiffCoeff(old_psi));
   diff_rho_form.AddDomainIntegrator(new DomainLFIntegrator(*diff_rho));

   if (Mpi::Root())
      mfem::out << "\n"
                << "Initialization Done." << "\n"
                << "Start Projected Mirror Descent OC Step." << "\n" << std::endl;

   double compliance = optprob.Eval();
   double step_size(0), volume(density.GetDomainVolume()*vol_fraction),
          stationarityError(infinity()), stationarityError_bregman(infinity());
   int num_reeval(0);
   double old_compliance;

   TableLogger logger;
   logger.Append(std::string("Volume"), volume);
   logger.Append(std::string("Compliance"), compliance);
   logger.Append(std::string("Stationarity"), stationarityError);
   logger.Append(std::string("Re-evel"), num_reeval);
   logger.Append(std::string("Step Size"), step_size);
   logger.Append(std::string("Stationarity-Bregman"), stationarityError_bregman);
   logger.SaveWhenPrint(filename_prefix.str());
   logger.Print();

   optprob.UpdateGradient();
   bool converged = false;
   ParGridFunction d(&control_fes), old_d(&control_fes);
   for (int k = 0; k < max_it; k++)
   {
      old_d = d;
      d = grad;
      d.ApplyMap([](double x) {return x > -1e-13 ? -std::log(1e-13) : -std::log(-x); });
      // Step 1. Compute Step size
      if (k == 0) { step_size = 1.0; }
      else
      {
         diff_rho_form.Assemble();
         old_psi -= psi;
         old_d -= d;
         step_size = std::fabs(diff_rho_form(old_psi)  / diff_rho_form(old_d));
      }

      // Step 2. Store old data
      old_compliance = compliance;
      old_psi = psi;

      // Step 3. Step and upate gradient
      num_reeval = Step_Armijo(optprob, old_psi, d, diff_rho_form, c1, step_size);
      compliance = optprob.GetValue();
      volume = density.GetVolume();
      optprob.UpdateGradient();
      
      // Step 4. Visualization
      if (glvis_visualization)
      {
         if (sout_SIMP.is_open())
         {
            designDensity_gf->ProjectCoefficient(simp_rule.GetPhysicalDensity(
                                                    density.GetFilteredDensity()));
            sout_SIMP << "parallel " << num_procs << " " << myid << "\n";
            sout_SIMP << "solution\n" << *pmesh << *designDensity_gf
                      << flush;
            MPI_Barrier(MPI_COMM_WORLD); // try to prevent streams from mixing
         }
         if (sout_r.is_open())
         {
            rho_gf->ProjectCoefficient(density.GetDensityCoefficient());
            sout_r << "parallel " << num_procs << " " << myid << "\n";
            sout_r << "solution\n" << *pmesh << *rho_gf
                   << flush;
            MPI_Barrier(MPI_COMM_WORLD); // try to prevent streams from mixing
         }
      }
      if (paraview)
      {
         pd->SetCycle(k);
         pd->SetTime(k);
         pd->Save();
      }

      // Check convergence
      stationarityError = density.StationarityErrorL2(grad);
      stationarityError_bregman = density.StationarityError(grad);

      logger.Print();

      if (stationarityError < tol_stationarity &&
          std::fabs(old_compliance - compliance) < tol_compliance)
      {
         converged = true;
         if (Mpi::Root()) { mfem::out << "Total number of iteration = " << k + 1 << std::endl; }
         break;
      }
   }
   if (!converged)
   {
      if (Mpi::Root()) { mfem::out << "Total number of iteration = " << max_it << std::endl; }
      if (Mpi::Root()) { mfem::out << "Maximum iteration reached." << std::endl; }
   }
   if (save)
   {
      ostringstream solfile, solfile2;
      solfile << filename_prefix.str() << "-" << seq_ref_levels << "-" <<
              par_ref_levels << "-0." << setfill('0') << setw(6) << myid;
      solfile2 << filename_prefix.str() << "-" << seq_ref_levels << "-" <<
               par_ref_levels << "-f." << setfill('0') << setw(6) << myid;
      ofstream sol_ofs(solfile.str().c_str());
      sol_ofs.precision(8);
      sol_ofs << psi;

      ofstream sol_ofs2(solfile2.str().c_str());
      sol_ofs2.precision(8);
      sol_ofs2 << density.GetFilteredDensity();
   }

   return 0;
}
