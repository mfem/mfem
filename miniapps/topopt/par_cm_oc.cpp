// Compliancemi Mechanism with projected mirror descent in parallel
//
//                  minimize F(ρ) = ∫_{Γ_out} d⋅u dx over ρ ∈ L¹(Ω)
//
//                  subject to
//
//                    -Div(r(ρ̃)Cε(u))   = 0       in Ω
//                    Cε(u):n + k_in  u = t       on Γ_in
//                    Cε(u):n + k_out u = 0       on Γ_out
//                    -ϵ²Δρ̃ + ρ̃ = ρ               in Ω + Neumann BCs
//                    0 ≤ ρ ≤ 1                   in Ω
//                    ∫_Ω ρ dx = θ vol(Ω)
//
//              Here, r(ρ̃) = ρ₀ + ρ̃³ (1-ρ₀) is the solid isotropic material
//              penalization (SIMP) law, C is the elasticity tensor for an
//              isotropic linearly elastic material, k_* are spring constants,
//              Γ_* are input/output ports, ϵ > 0 is the design
//              length scale, and 0 < θ < 1 is the volume fraction.
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
#include "prob_elasticity.hpp"

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
   int max_it = 2e2;
   double rho_min = 1e-06;
   double exponent = 3.0;
   double E = 1.0;
   double nu = 0.3;
   double c1 = 1e-04;
   bool glvis_visualization = true;
   bool save = false;
   bool paraview = true;
   double tol_stationarity = 0;
   double tol_compliance = 5e-05;
   double cx = 1.0;
   double cy = 0.0;

   ostringstream filename_prefix;
   filename_prefix << "PMD-";

   int problem = CompliantMechanismProblem::ForceInverter;

   OptionsParser args(argc, argv);
   args.AddOption(&seq_ref_levels, "-rs", "--seq-refine",
                  "Number of times to refine the sequential mesh uniformly.");
   args.AddOption(&par_ref_levels, "-rp", "--par-refine",
                  "Number of times to refine the parallel mesh uniformly.");
   args.AddOption(&problem, "-p", "--problem",
                  "Problem number: 0) ForceInverter, 1) ForceInverter3.");
   args.AddOption(&order, "-o", "--order",
                  "Order (degree) of the finite elements.");
   args.AddOption(&filter_radius, "-fr", "--filter-radius",
                  "Length scale for ρ.");
   args.AddOption(&max_it, "-mi", "--max-it",
                  "Maximum number of gradient descent iterations.");
   args.AddOption(&vol_fraction, "-vf", "--volume-fraction",
                  "Volume fraction for the material density.");
   args.AddOption(&E, "-E", "--E",
                  "Lamé constant λ.");
   args.AddOption(&nu, "-nu", "--nu",
                  "Lamé constant μ.");
   args.AddOption(&rho_min, "-rmin", "--rho-min",
                  "Minimum of density coefficient.");
   args.AddOption(&glvis_visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&cx, "-cx", "--connection-x",
                  "Connection point x-coordinate for ports");
   args.AddOption(&cy, "-cy", "--connection-y",
                  "Connection point y-coordinate for ports");
   args.Parse();
   if (!args.Good()) {if (Mpi::Root()) args.PrintUsage(mfem::out);}


   std::unique_ptr<Mesh> mesh;
   Array2D<int> ess_bdr;
   Array<int> ess_bdr_filter;
   Array<int> bdr_in, bdr_out;
   double k_in, k_out;
   int idx_in, idx_out;
   std::unique_ptr<VectorCoefficient> t_in;
   Vector d_in, d_out;
   std::string prob_name;
   GetCompliantMechanismProblem((CompliantMechanismProblem)problem, filter_radius,
                                vol_fraction,
                                mesh, k_in, k_out, d_in, d_out, t_in, bdr_in, bdr_out,
                                ess_bdr, ess_bdr_filter,
                                prob_name, seq_ref_levels, par_ref_levels);
   filename_prefix << prob_name << "-" << seq_ref_levels + par_ref_levels;
   int dim = mesh->Dimension();
   const int num_el = mesh->GetNE();
   std::unique_ptr<ParMesh> pmesh(static_cast<ParMesh*>(mesh.release()));

   if (Mpi::Root())
      mfem::out << "\n"
                << "Compliance Mechanism with Projected Mirror Descent.\n"
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
   ParFiniteElementSpace state_fes(pmesh.get(), &state_fec,dim);
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
   PrimalDesignDensity density(control_fes, filter, vol_fraction);

   ConstantCoefficient E_cf(E), nu_cf(nu);
   Vector zero_d(pmesh->SpaceDimension()); zero_d = 0.0;
   VectorConstantCoefficient zero_cf(zero_d);
   ParametrizedElasticityEquation elasticity(state_fes,
                                             density.GetFilteredDensity(), simp_rule, E_cf, nu_cf, zero_cf, ess_bdr);
   VectorConstantCoefficient d_in_cf(d_in), d_out_cf(d_out);
   d_out.Neg();
   VectorConstantCoefficient neg_d_out_cf(d_out);
   ConstantCoefficient k_in_cf(k_in), k_out_cf(k_out);
   elasticity.GetBilinearForm().AddBdrFaceIntegrator(new
                                                     VectorBdrDirectionalMassIntegrator(
                                                        k_in_cf, d_in_cf, dim), bdr_in);
   elasticity.GetBilinearForm().AddBdrFaceIntegrator(new
                                                     VectorBdrDirectionalMassIntegrator(
                                                        k_out_cf, d_out_cf, dim), bdr_out);
   elasticity.GetLinearForm().AddBoundaryIntegrator(new VectorBoundaryLFIntegrator(
                                                       *t_in), bdr_in);
   elasticity.SetLinearFormStationary();
   ParLinearForm obj(&state_fes);
   obj.AddBdrFaceIntegrator(new VectorBoundaryLFIntegrator(neg_d_out_cf), bdr_out);
   obj.Assemble();
   TopOptProblem optprob(obj, elasticity, density, true, false);

   ParGridFunction &u = *dynamic_cast<ParGridFunction*>(&optprob.GetState());
   ParGridFunction &rho_filter = *dynamic_cast<ParGridFunction*>
                                 (&density.GetFilteredDensity());
   ParGridFunction &grad(*dynamic_cast<ParGridFunction*>(&optprob.GetGradient()));
   ParGridFunction &rho(*dynamic_cast<ParGridFunction*>
                        (&density.GetGridFunction()));
   {
      Vector center({cx, cy});
      Array<Vector*> ports(3);
      ports[0] = new Vector(2); (*ports[0])(0) = 0.0; (*ports[0])(1) = 0.0;
      ports[1] = new Vector(2); (*ports[1])(0) = 0.0; (*ports[1])(1) = 1.0;
      ports[2] = new Vector(2); (*ports[2])(0) = 2.0; (*ports[2])(1) = 1.0;

      initialDesign(rho, center, ports, vol_fraction*density.GetDomainVolume(),
                    2.0, -5, 5);
      rho.ApplyMap(sigmoid);
   }
   rho_filter = 1.0;
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
                   << "window_title 'Design density r(ρ̃) - OC "
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
                << "window_title 'Raw density ρ - OC "
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
      pd->RegisterField("rho", &density.GetGridFunction());
      pd->RegisterField("frho", &rho_filter);
      pd->SetLevelsOfDetail(order + 3);
      pd->SetDataFormat(VTKFormat::BINARY);
      pd->SetCycle(0);
      pd->SetTime(0);
      pd->Save();
   }

   // 11. Iterate
   ParGridFunction old_rho(&control_fes);

   ParLinearForm diff_rho_form(&control_fes);
   std::unique_ptr<Coefficient> diff_rho(optprob.GetDensityDiffCoeff(old_rho));
   diff_rho_form.AddDomainIntegrator(new DomainLFIntegrator(*diff_rho));

   if (Mpi::Root())
      mfem::out << "\n"
                << "Initialization Done." << "\n"
                << "Start Optimality Criteria Method." << "\n" << std::endl;

   double compliance = optprob.Eval();
   double volume(density.GetDomainVolume()*vol_fraction),
          stationarityError(infinity());
   double old_compliance;

   TableLogger logger;
   logger.Append(std::string("Volume"), volume);
   logger.Append(std::string("Compliance"), compliance);
   logger.Append(std::string("Stationarity"), stationarityError);
   logger.SaveWhenPrint(filename_prefix.str());
   logger.Print();

   optprob.UpdateGradient();
   ParGridFunction B(&control_fes), lower(&control_fes), upper(&control_fes);
   ParBilinearForm inv_mass(&control_fes);
   inv_mass.AddDomainIntegrator(new InverseIntegrator(new MassIntegrator));
   inv_mass.Assemble();
   bool converged = false;
   double mv = 0.2;
   for (int k = 0; k < max_it; k++)
   {
      // Store old data
      old_compliance = compliance;

      B = grad;
      inv_mass.Mult(grad, B);
      B.ApplyMap([](double x) {return std::pow(std::max(-x, 1e-06), 0.3); });
      lower = rho;
      lower -= mv; lower.Clip(0, 1);
      upper = rho;
      upper += mv; upper.Clip(0, 1);
      double l1(0.0), l2(1e09);
      old_rho = rho;
      while ((l2 - l1) > 1e-5)
      {
         double lmid = 0.5*(l1 + l2);
         rho = old_rho;
         rho *= B;
         rho *= 1.0 / std::sqrt(lmid);
         rho.Clip(lower, upper);
         volume = density.ComputeVolume();
         if (volume > density.GetDomainVolume()*vol_fraction)
         {
            l1 = lmid;
         }
         else
         {
            l2 = lmid;
         }
      }
      // Eval and update gradient
      compliance = optprob.Eval();
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
      stationarityError = density.StationarityError(grad);

      logger.Print();

      if (stationarityError < tol_stationarity &&
          std::fabs((old_compliance - compliance)/old_compliance) < tol_compliance)
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
              par_ref_levels <<
              "-0." << setfill('0') << setw(6) << myid;
      solfile2 << filename_prefix.str() << "-" << seq_ref_levels << "-" <<
               par_ref_levels <<
               "-f." << setfill('0') << setw(6) << myid;
      ofstream sol_ofs(solfile.str().c_str());
      sol_ofs.precision(8);
      sol_ofs << rho;

      ofstream sol_ofs2(solfile2.str().c_str());
      sol_ofs2.precision(8);
      sol_ofs2 << density.GetFilteredDensity();
   }

   return 0;
}
