// Copyright (c) 2010-2025, Lawrence Livermore National Security, LLC.
//
// Solve a damped frequency-domain cantilever problem in two or three
// dimensions and optionally write its complex displacement for ParaView.

#include "frequency_domain_driver_utils.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>

using namespace mfem;

namespace
{

/// Evaluate sqrt(|u_real|^2 + |u_imaginary|^2).
class ComplexDisplacementMagnitude : public Coefficient
{
public:
   ComplexDisplacementMagnitude(const ParGridFunction &real_displacement,
                                const ParGridFunction &imaginary_displacement)
      : real_displacement_(real_displacement),
        imaginary_displacement_(imaginary_displacement) { }

   real_t Eval(ElementTransformation &transformation,
               const IntegrationPoint &point) override
   {
      real_displacement_.GetVectorValue(transformation, point, real_value_);
      imaginary_displacement_.GetVectorValue(transformation, point,
                                             imaginary_value_);
      return std::sqrt(real_value_*real_value_ +
                       imaginary_value_*imaginary_value_);
   }

private:
   const ParGridFunction &real_displacement_;
   const ParGridFunction &imaginary_displacement_;
   Vector real_value_;
   Vector imaginary_value_;
};

/// Construct a force supported inside a circle or sphere.
std::shared_ptr<VectorCoefficient> MakeLocalizedForce(
   const int dimension, const Vector &center, const real_t radius,
   const int component, const real_t amplitude)
{
   return std::make_shared<VectorFunctionCoefficient>(
             dimension,
             [center, radius, component, amplitude](const Vector &position,
                                                    Vector &value)
   {
      value.SetSize(position.Size());
      value = 0.0;
      real_t distance_squared = 0.0;
      for (int d = 0; d < position.Size(); ++d)
      {
         const real_t difference = position(d) - center(d);
         distance_squared += difference*difference;
      }
      if (distance_squared <= radius*radius)
      {
         value(component) = amplitude;
      }
   });
}

} // namespace

int main(int argc, char *argv[])
{
   Mpi::Init(argc, argv);
   Hypre::Init();

   const char *device_configuration = "cpu";
   const char *excitation_name = "volume";
   const char *output_prefix = "ParaView";
   const char *csv_path = "";
   bool visualization = false;
   int dimension = 2;
   int nx = 24;
   int ny = 6;
   int nz = 6;
   int order = 2;
   int serial_refinements = 0;
   int parallel_refinements = 0;
   int excitation_component = -1;
   real_t length = 4.0;
   real_t height = 1.0;
   real_t width = 1.0;
   real_t frequency = 0.05;
   real_t frequency_factor = -1.0;
   real_t lambda = 2.3;
   real_t mu = 1.7;
   real_t density = 0.9;
   real_t amplitude_real = 1.0;
   real_t amplitude_imaginary = 0.0;
   real_t load_radius = 0.2;
   real_t load_offset = 0.3;
   frequency_domain::SolverOptions solver_options;
   frequency_domain::DampingOptions damping_options;
   frequency_domain::EigenOptions eigen_options;

   OptionsParser args(argc, argv);
   args.AddOption(&dimension, "-dim", "--dimension",
                  "Spatial dimension: 2 or 3.");
   args.AddOption(&device_configuration, "-d", "--device",
                  "MFEM device configuration.");
   args.AddOption(&excitation_name, "-exc", "--excitation",
                  "Excitation mode: volume, surface, both, or support.");
   args.AddOption(&output_prefix, "-out", "--output-prefix",
                  "ParaView output directory.");
   args.AddOption(&csv_path, "-csv", "--csv", "Optional CSV output file.");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization", "Enable or disable ParaView output.");
   args.AddOption(&nx, "-nx", "--x-elements",
                  "Number of elements along the beam.");
   args.AddOption(&ny, "-ny", "--y-elements",
                  "Number of elements through the beam height.");
   args.AddOption(&nz, "-nz", "--z-elements",
                  "Number of elements through the 3D beam width.");
   args.AddOption(&order, "-o", "--order", "H1 polynomial degree.");
   args.AddOption(&serial_refinements, "-rs", "--serial-refinements",
                  "Number of serial refinements.");
   args.AddOption(&parallel_refinements, "-rp", "--parallel-refinements",
                  "Number of parallel refinements.");
   args.AddOption(&length, "-lx", "--length", "Beam length.");
   args.AddOption(&height, "-ly", "--height", "Beam height.");
   args.AddOption(&width, "-lz", "--width", "Three-dimensional beam width.");
   args.AddOption(&frequency, "-f", "--frequency",
                  "Angular excitation frequency.");
   args.AddOption(&frequency_factor, "-ff", "--frequency-factor",
                  "Set frequency to this multiple of the lowest eigenfrequency; "
                  "a positive value overrides --frequency.");
   args.AddOption(&lambda, "-la", "--lambda", "First Lame coefficient.");
   args.AddOption(&mu, "-mu", "--mu", "Shear modulus.");
   args.AddOption(&density, "-rho", "--density", "Mass density.");
   args.AddOption(&amplitude_real, "-ar", "--amplitude-real",
                  "Real excitation amplitude.");
   args.AddOption(&amplitude_imaginary, "-ai", "--amplitude-imaginary",
                  "Imaginary excitation amplitude.");
   args.AddOption(&excitation_component, "-c", "--component",
                  "Zero-based excitation component; -1 selects the last.");
   args.AddOption(&load_radius, "-r", "--load-radius",
                  "Radius of the localized volume excitation.");
   args.AddOption(&load_offset, "-off", "--load-offset",
                  "Distance from the free end to the localized-load center.");
   solver_options.AddOptions(args);
   damping_options.AddOptions(args);
   eigen_options.AddOptions(args);
   args.ParseCheck();

   const std::string excitation(excitation_name);
   const bool use_volume = excitation == "volume" || excitation == "both";
   const bool use_surface = excitation == "surface" || excitation == "both";
   const bool use_support = excitation == "support";
   MFEM_VERIFY(dimension == 2 || dimension == 3,
               "The spatial dimension must be 2 or 3.");
   MFEM_VERIFY(use_volume || use_surface || use_support,
               "Excitation must be volume, surface, both, or support.");
   MFEM_VERIFY(nx > 0 && ny > 0 && (dimension == 2 || nz > 0),
               "Mesh element counts must be positive.");
   MFEM_VERIFY(order > 0, "The polynomial degree must be positive.");
   MFEM_VERIFY(serial_refinements >= 0 && parallel_refinements >= 0,
               "Refinement counts must be nonnegative.");
   MFEM_VERIFY(length > 0.0 && height > 0.0 &&
               (dimension == 2 || width > 0.0),
               "Beam dimensions must be positive.");
   MFEM_VERIFY(frequency >= 0.0, "Frequency must be nonnegative.");
   MFEM_VERIFY(lambda > 0.0 && mu > 0.0 && density > 0.0,
               "Material parameters must be positive.");
   MFEM_VERIFY(frequency_factor == -1.0 || frequency_factor > 0.0,
               "Frequency factor must be positive when supplied.");
   solver_options.Validate();
   damping_options.Validate();
   eigen_options.Validate();
   MFEM_VERIFY(excitation_component >= -1,
               "The excitation component must be -1 or nonnegative.");
   MFEM_VERIFY(amplitude_real != 0.0 || amplitude_imaginary != 0.0,
               "At least one excitation amplitude must be nonzero.");

   if (excitation_component < 0) { excitation_component = dimension - 1; }
   MFEM_VERIFY(excitation_component < dimension,
               "Excitation component is outside the spatial dimension.");
   if (use_volume)
   {
      const real_t transverse_size = dimension == 2 ?
                                     height : std::min(height, width);
      MFEM_VERIFY(load_radius > 0.0 && 2.0*load_radius < transverse_size,
                  "The localized load must fit inside the beam cross-section.");
      MFEM_VERIFY(load_offset > load_radius &&
                  load_offset + load_radius < length,
                  "The localized load must fit inside the beam length.");
   }

   Device device(device_configuration);
   Mesh serial_mesh;
   if (dimension == 2)
   {
      serial_mesh = Mesh::MakeCartesian2D(
                       nx, ny, Element::QUADRILATERAL, true, length, height);
   }
   else
   {
      serial_mesh = Mesh::MakeCartesian3D(
                       nx, ny, nz, Element::HEXAHEDRON,
                       length, height, width);
   }
   for (int i = 0; i < serial_refinements; ++i)
   {
      serial_mesh.UniformRefinement();
   }

   ParMesh mesh(MPI_COMM_WORLD, serial_mesh);
   for (int i = 0; i < parallel_refinements; ++i)
   {
      mesh.UniformRefinement();
   }

   H1_FECollection collection(order, dimension);
   ParFiniteElementSpace space(&mesh, &collection, dimension,
                               Ordering::byNODES);
   // Cartesian meshes number the x-min/x-max boundaries differently in 2D
   // and 3D. Eigenmodes always use the homogeneous form of the support.
   const int support_attribute = dimension == 2 ? 4 : 5;
   const int free_surface_attribute = dimension == 2 ? 2 : 3;
   Array<int> essential_boundary(mesh.bdr_attributes.Max());
   essential_boundary = 0;
   essential_boundary[support_attribute - 1] = 1;
   ConstantCoefficient lambda_coefficient(lambda);
   ConstantCoefficient mu_coefficient(mu);
   ConstantCoefficient density_coefficient(density);
   const frequency_domain::EigenDiagnostics eigen =
      frequency_domain::ComputeEigenDiagnostics(
         space, lambda_coefficient, mu_coefficient, density_coefficient,
         essential_boundary, damping_options, eigen_options);
   frequency = frequency_domain::ResolveFrequency(
                  frequency, frequency_factor, eigen);

   FrequencyDomainLinearElasticitySolver solver(space);
   solver.SetLameMaterial(
      std::make_shared<ConstantCoefficient>(lambda),
      std::make_shared<ConstantCoefficient>(mu),
      std::make_shared<ConstantCoefficient>(density));
   solver.SetFrequency(frequency);
   damping_options.Apply(solver);
   solver_options.Apply(solver);
   Vector real_direction(dimension);
   Vector imaginary_direction(dimension);
   real_direction = 0.0;
   imaginary_direction = 0.0;
   real_direction(excitation_component) = amplitude_real;
   imaginary_direction(excitation_component) = amplitude_imaginary;

   if (use_support)
   {
      solver.AddDisplacementBC(
         support_attribute,
         std::make_shared<VectorConstantCoefficient>(real_direction),
         std::make_shared<VectorConstantCoefficient>(imaginary_direction));
   }
   else
   {
      solver.AddBoundaryID(support_attribute);
      if (use_volume)
      {
         Vector center(dimension);
         center(0) = length - load_offset;
         center(1) = 0.5*height;
         if (dimension == 3) { center(2) = 0.5*width; }
         solver.AddVolumeLoad(
            1,
            MakeLocalizedForce(dimension, center, load_radius,
                               excitation_component, amplitude_real),
            MakeLocalizedForce(dimension, center, load_radius,
                               excitation_component, amplitude_imaginary));
      }
      if (use_surface)
      {
         solver.AddBoundaryLoad(
            free_surface_attribute,
            std::make_shared<VectorConstantCoefficient>(real_direction),
            std::make_shared<VectorConstantCoefficient>(imaginary_direction));
      }
   }

   StopWatch assembly_timer;
   assembly_timer.Start();
   solver.Assemble();
   assembly_timer.Stop();

   ParComplexGridFunction displacement(&space);
   displacement = std::complex<real_t>(0.0, 0.0);
   StopWatch solve_timer;
   solve_timer.Start();
   solver.Solve(displacement);
   solve_timer.Stop();

   const std::string collection_name =
      "frequency_domain_cantilever_" + std::to_string(dimension) + "d_" +
      excitation;
   double visualization_time = 0.0;
   if (visualization)
   {
      StopWatch visualization_timer;
      visualization_timer.Start();
      ParFiniteElementSpace scalar_space(&mesh, &collection);
      ParGridFunction displacement_amplitude(&scalar_space);
      ComplexDisplacementMagnitude magnitude(displacement.real(),
                                             displacement.imag());
      displacement_amplitude.ProjectCoefficient(magnitude);
      ParaViewDataCollection paraview(collection_name, &mesh);
      paraview.SetPrefixPath(output_prefix);
      paraview.SetLevelsOfDetail(order);
      paraview.SetDataFormat(VTKFormat::BINARY);
      paraview.SetHighOrderOutput(true);
      paraview.SetCycle(0);
      paraview.SetTime(frequency);
      paraview.RegisterField("displacement_real", &displacement.real());
      paraview.RegisterField("displacement_imaginary", &displacement.imag());
      paraview.RegisterField("displacement_amplitude", &displacement_amplitude);
      paraview.Save();
      visualization_timer.Stop();
      visualization_time = visualization_timer.RealTime();
   }

   const double local_times[] =
   {
      assembly_timer.RealTime(), solver.GetAssemblyTime(),
      solver.GetPreconditionerAssemblyTime(), solver.GetSolverSetupTime(),
      solve_timer.RealTime(), solver.GetLinearSolveTime(),
      solver.GetLoadAssemblyTime(), solver.GetSolutionDistributionTime(),
      visualization_time, eigen.solve_time
   };
   double maximum_times[10];
   frequency_domain::GlobalMaximum(local_times, maximum_times, 10,
                                   mesh.GetComm());
   const double assembly_time = maximum_times[0];
   const double operator_assembly_time = maximum_times[1];
   const double preconditioner_assembly_time = maximum_times[2];
   const double solver_setup_time = maximum_times[3];
   const double solve_time = maximum_times[4];
   const double linear_solve_time = maximum_times[5];
   const double load_assembly_time = maximum_times[6];
   const double distribution_time = maximum_times[7];
   visualization_time = maximum_times[8];
   const double eigen_solve_time = maximum_times[9];
   const int iterations = solver.GetNumIterations();
   const int preconditioner_applications =
      solver.GetNumPreconditionerApplications();
   const int h_inverse_applications = solver.GetNumHInverseApplications();
   const int h_inverse_iterations = solver.GetNumHInverseIterations();
   const bool converged = solver.GetConverged();
   const auto active_solver = solver.GetActiveLinearSolverType();
   const real_t initial_residual = solver.GetInitialNorm();
   const real_t final_residual = solver.GetFinalNorm();
   const real_t relative_residual =
      initial_residual > 0.0 ? final_residual/initial_residual : -1.0;
   const real_t frequency_ratio = frequency/eigen.omega1;
   const real_t loss_factor = frequency_domain::LossFactor(frequency, eigen);
   const real_t h_indicator = frequency_domain::HIndicator(frequency, eigen);
   const long long global_elements = mesh.GetGlobalNE();
   const HYPRE_BigInt global_displacement_dofs = space.GlobalTrueVSize();
   const HYPRE_BigInt global_total_dofs = 2*global_displacement_dofs;
   const bool csv_requested = std::string(csv_path).size() > 0;
   std::unique_ptr<std::ofstream> csv;
   int csv_open = 1;
   if (csv_requested && Mpi::Root())
   {
      csv.reset(new std::ofstream(csv_path));
      csv_open = *csv ? 1 : 0;
   }
   if (csv_requested)
   {
      MPI_Bcast(&csv_open, 1, MPI_INT, 0, mesh.GetComm());
      MFEM_VERIFY(csv_open, "Unable to open CSV output file: " << csv_path);
   }
   if (Mpi::Root())
   {
      std::cout << std::setprecision(12)
                << "Cantilever dimension: " << dimension << '\n'
                << "Excitation: " << excitation << '\n'
                << "Damping model: " << damping_options.model << '\n';
      if (std::string(damping_options.model) == "rayleigh")
      {
         std::cout << "Rayleigh damping alpha, beta: "
                   << damping_options.alpha << ", "
                   << damping_options.beta << '\n';
      }
      else
      {
         std::cout << "Independent damping c_M, lambda_C, mu_C: "
                   << damping_options.mass << ", "
                   << damping_options.lambda << ", "
                   << damping_options.mu << '\n';
      }
      std::cout << "Lowest eigenvalue: " << eigen.lambda1 << '\n'
                << "Lowest angular eigenfrequency: " << eigen.omega1 << '\n'
                << "First-mode damping coefficient c1: "
                << eigen.modal_damping << '\n'
                << "First-mode damping ratio zeta1: "
                << eigen.damping_ratio << '\n'
                << "Angular frequency: " << frequency << '\n'
                << "Frequency / lowest eigenfrequency: "
                << frequency_ratio << '\n'
                << "Relative damping eta1(omega): " << loss_factor << '\n'
                << "First-mode H indicator: " << h_indicator << '\n'
                << "Global elements: " << global_elements << '\n'
                << "Global displacement DOFs: "
                << global_displacement_dofs << '\n'
                << "Global total real-block DOFs: "
                << global_total_dofs << '\n'
                << "Requested linear solver: "
                << frequency_domain::LinearSolverName(
                      solver.GetLinearSolverType()) << '\n'
                << "Active linear solver: "
                << frequency_domain::LinearSolverName(
                      active_solver) << '\n'
                << "Preconditioner: "
                << (active_solver ==
                    FrequencyDomainLinearElasticitySolver::LinearSolverType::
                    MUMPS ? "none" :
                    frequency_domain::PreconditionerName(
                       solver.GetPreconditionerType())) << '\n'
                << "H inverse: "
                << (active_solver ==
                    FrequencyDomainLinearElasticitySolver::LinearSolverType::
                    MUMPS ? "none" :
                    frequency_domain::HInverseName(
                       solver.GetHInverseType())) << '\n'
                << "Fixed H-inverse AMG cycles: "
                << solver.GetHInverseAMGCycles() << '\n'
                << "LOR ordering: "
                << frequency_domain::LOROrderingName(
                      solver.GetLOROrdering()) << '\n'
                << "Outer relative/absolute tolerance: "
                << solver.GetRelTol() << ", " << solver.GetAbsTol() << '\n'
                << "Outer maximum iterations/restart dimension: "
                << solver.GetMaxIter() << ", " << solver.GetKDim() << '\n'
                << "H-inverse relative/absolute tolerance: "
                << solver.GetPreconditionerRelTol() << ", "
                << solver.GetPreconditionerAbsTol() << '\n'
                << "H-inverse maximum iterations: "
                << solver.GetPreconditionerMaxIter() << '\n'
                << "Converged: " << (converged ? "yes" : "no") << '\n'
                << "Outer iterations: " << iterations << '\n'
                << "Block-preconditioner applications: "
                << preconditioner_applications << '\n'
                << "H-inverse applications: " << h_inverse_applications
                << '\n'
                << "Accumulated H-inverse iterations/cycles: "
                << h_inverse_iterations << '\n'
                << "Initial residual norm: " << initial_residual << '\n'
                << "Final residual norm: " << final_residual << '\n'
                << "Relative residual norm: " << relative_residual << '\n'
                << "Eigenvalue solve time (max): " << eigen_solve_time
                << " s\n"
                << "Total assembly time (max): " << assembly_time << " s\n"
                << "Operator assembly time (max): "
                << operator_assembly_time << " s\n"
                << "Preconditioner assembly time (max): "
                << preconditioner_assembly_time << " s\n"
                << "Solver setup time (max): " << solver_setup_time << " s\n"
                << "Solve time (max): " << solve_time << " s\n"
                << "Load assembly time (max): " << load_assembly_time
                << " s\n"
                << "Linear solve time (max): " << linear_solve_time
                << " s\n"
                << "Solution distribution time (max): "
                << distribution_time << " s\n"
                << "ParaView output time (max): " << visualization_time
                << " s\n";
      if (visualization)
      {
         std::string output_path(output_prefix);
         if (!output_path.empty() && output_path.back() != '/')
         {
            output_path += '/';
         }
         output_path += collection_name + "/" + collection_name + ".pvd";
         std::cout << "ParaView collection: " << output_path << '\n';
      }

      if (csv)
      {
         *csv << std::setprecision(17)
             << "dimension,excitation,damping_model,damping_alpha,"
                "damping_beta,mass_damping,damping_lambda,damping_mu,"
                "frequency,lambda1,omega1,"
                "frequency_ratio,modal_damping,damping_ratio,loss_factor,"
                "h_indicator,elements,displacement_dofs,total_dofs,"
                "requested_solver,active_solver,"
                "preconditioner,h_inverse,h_amg_cycles,lor_ordering,"
                "relative_tolerance,"
                "absolute_tolerance,max_iterations,kdim,"
                "preconditioner_relative_tolerance,"
                "preconditioner_absolute_tolerance,"
                "preconditioner_max_iterations,converged,outer_iterations,"
                "preconditioner_applications,h_inverse_applications,"
                "h_inverse_iterations,initial_residual,final_residual,"
                "relative_residual,eigen_time,assembly_time,operator_time,"
                "preconditioner_setup_time,solver_setup_time,solve_time,"
                "load_time,linear_solve_time,distribution_time,"
                "visualization_time\n"
             << dimension << ',' << excitation << ',' << damping_options.model
             << ',' << damping_options.alpha << ',' << damping_options.beta
             << ',' << damping_options.mass << ',' << damping_options.lambda
             << ',' << damping_options.mu << ',' << frequency << ','
             << eigen.lambda1 << ','
             << eigen.omega1 << ',' << frequency_ratio << ','
             << eigen.modal_damping << ',' << eigen.damping_ratio << ','
             << loss_factor << ',' << h_indicator << ',' << global_elements
             << ',' << global_displacement_dofs << ',' << global_total_dofs
             << ','
             << frequency_domain::LinearSolverName(
                   solver.GetLinearSolverType()) << ','
             << frequency_domain::LinearSolverName(
                   active_solver) << ','
             << (active_solver ==
                 FrequencyDomainLinearElasticitySolver::LinearSolverType::
                 MUMPS ? "none" : frequency_domain::PreconditionerName(
                    solver.GetPreconditionerType())) << ','
             << (active_solver ==
                 FrequencyDomainLinearElasticitySolver::LinearSolverType::
                 MUMPS ? "none" : frequency_domain::HInverseName(
                    solver.GetHInverseType())) << ','
             << solver.GetHInverseAMGCycles() << ','
             << frequency_domain::LOROrderingName(solver.GetLOROrdering())
             << ',' << solver.GetRelTol() << ',' << solver.GetAbsTol() << ','
             << solver.GetMaxIter() << ',' << solver.GetKDim() << ','
             << solver.GetPreconditionerRelTol() << ','
             << solver.GetPreconditionerAbsTol() << ','
             << solver.GetPreconditionerMaxIter() << ','
             << (converged ? 1 : 0) << ',' << iterations << ','
             << preconditioner_applications << ',' << h_inverse_applications
             << ',' << h_inverse_iterations << ',' << initial_residual
             << ',' << final_residual << ',' << relative_residual << ','
             << eigen_solve_time << ',' << assembly_time << ','
             << operator_assembly_time << ',' << preconditioner_assembly_time
             << ',' << solver_setup_time << ',' << solve_time << ','
             << load_assembly_time << ',' << linear_solve_time << ','
             << distribution_time << ',' << visualization_time << '\n';
      }
   }

   if (csv_requested)
   {
      int csv_written = 1;
      if (Mpi::Root())
      {
         csv->flush();
         csv_written = *csv ? 1 : 0;
      }
      MPI_Bcast(&csv_written, 1, MPI_INT, 0, mesh.GetComm());
      MFEM_VERIFY(csv_written,
                  "Unable to write CSV output file: " << csv_path);
   }

   return converged ? EXIT_SUCCESS : EXIT_FAILURE;
}
