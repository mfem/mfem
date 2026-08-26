// Copyright (c) 2010-2025, Lawrence Livermore National Security, LLC.
//
// Solve a damped frequency-domain cantilever problem in two or three
// dimensions and write its complex displacement for ParaView.

#include "frequency_domain_elasticity_solver.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdlib>
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

/// Return the maximum elapsed time over all MPI ranks.
double GlobalMaximum(const double local_value, const MPI_Comm communicator)
{
   double global_value = 0.0;
   MPI_Reduce(&local_value, &global_value, 1, MPI_DOUBLE, MPI_MAX, 0,
              communicator);
   return global_value;
}

} // namespace

int main(int argc, char *argv[])
{
   Mpi::Init(argc, argv);
   Hypre::Init();

   const char *device_configuration = "cpu";
   const char *excitation_name = "volume";
   const char *output_prefix = "ParaView";
   int dimension = 2;
   int nx = 24;
   int ny = 6;
   int nz = 6;
   int order = 2;
   int serial_refinements = 0;
   int parallel_refinements = 0;
   int excitation_component = -1;
   int max_iterations = 500;
   real_t length = 4.0;
   real_t height = 1.0;
   real_t width = 1.0;
   real_t frequency = 0.05;
   real_t lambda = 2.3;
   real_t mu = 1.7;
   real_t density = 0.9;
   real_t damping_alpha = 0.08;
   real_t damping_beta = 0.015;
   real_t amplitude_real = 1.0;
   real_t amplitude_imaginary = 0.0;
   real_t load_radius = 0.2;
   real_t load_offset = 0.3;
   real_t relative_tolerance = 1.0e-10;

   OptionsParser args(argc, argv);
   args.AddOption(&dimension, "-dim", "--dimension",
                  "Spatial dimension: 2 or 3.");
   args.AddOption(&device_configuration, "-d", "--device",
                  "MFEM device configuration.");
   args.AddOption(&excitation_name, "-exc", "--excitation",
                  "Excitation mode: volume, surface, both, or support.");
   args.AddOption(&output_prefix, "-out", "--output-prefix",
                  "ParaView output directory.");
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
   args.AddOption(&lambda, "-la", "--lambda", "First Lame coefficient.");
   args.AddOption(&mu, "-mu", "--mu", "Shear modulus.");
   args.AddOption(&density, "-rho", "--density", "Mass density.");
   args.AddOption(&damping_alpha, "-da", "--damping-alpha",
                  "Rayleigh mass-damping coefficient.");
   args.AddOption(&damping_beta, "-db", "--damping-beta",
                  "Rayleigh stiffness-damping coefficient.");
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
   args.AddOption(&relative_tolerance, "-rtol", "--relative-tolerance",
                  "Outer relative solver tolerance.");
   args.AddOption(&max_iterations, "-mi", "--max-iterations",
                  "Outer maximum iteration count.");
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
   MFEM_VERIFY(damping_alpha >= 0.0 && damping_beta >= 0.0,
               "Rayleigh damping parameters must be nonnegative.");
   MFEM_VERIFY(relative_tolerance >= 0.0 && max_iterations > 0,
               "Solver controls are invalid.");
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
   FrequencyDomainLinearElasticitySolver solver(space);
   solver.SetLameMaterial(
      std::make_shared<ConstantCoefficient>(lambda),
      std::make_shared<ConstantCoefficient>(mu),
      std::make_shared<ConstantCoefficient>(density));
   solver.SetFrequency(frequency);
   solver.SetRayleighDamping(damping_alpha, damping_beta);
   solver.SetPreconditionerType(
      FrequencyDomainLinearElasticitySolver::PreconditionerType::PRESB);
   solver.SetHInverseType(
      FrequencyDomainLinearElasticitySolver::HInverseType::LORMonolithicAMG);
   solver.SetLinearSolverType(
      FrequencyDomainLinearElasticitySolver::LinearSolverType::Automatic);
   solver.SetRelTol(relative_tolerance);
   solver.SetMaxIter(max_iterations);
   solver.SetPrintLevel(0);

   // Cartesian meshes number the x-min/x-max boundaries differently in 2D
   // and 3D.
   const int support_attribute = dimension == 2 ? 4 : 5;
   const int free_surface_attribute = dimension == 2 ? 2 : 3;
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

   ParFiniteElementSpace scalar_space(&mesh, &collection);
   ParGridFunction displacement_amplitude(&scalar_space);
   ComplexDisplacementMagnitude magnitude(displacement.real(),
                                          displacement.imag());
   displacement_amplitude.ProjectCoefficient(magnitude);

   const std::string collection_name =
      "frequency_domain_cantilever_" + std::to_string(dimension) + "d_" +
      excitation;
   ParaViewDataCollection paraview(collection_name, &mesh);
   paraview.SetPrefixPath(output_prefix);
   paraview.SetLevelsOfDetail(order);
   paraview.SetDataFormat(VTKFormat::BINARY);
   paraview.SetHighOrderOutput(true);
   paraview.SetCycle(0);
   paraview.SetTime(0.0);
   paraview.RegisterField("displacement_real", &displacement.real());
   paraview.RegisterField("displacement_imaginary", &displacement.imag());
   paraview.RegisterField("displacement_amplitude", &displacement_amplitude);
   paraview.Save();

   const double assembly_time = GlobalMaximum(assembly_timer.RealTime(),
                                               mesh.GetComm());
   const double operator_assembly_time = GlobalMaximum(
      solver.GetAssemblyTime(), mesh.GetComm());
   const double preconditioner_assembly_time = GlobalMaximum(
      solver.GetPreconditionerAssemblyTime(), mesh.GetComm());
   const double solver_setup_time = GlobalMaximum(
      solver.GetSolverSetupTime(), mesh.GetComm());
   const double solve_time = GlobalMaximum(solve_timer.RealTime(),
                                            mesh.GetComm());
   const int iterations = solver.GetNumIterations();
   const long long global_elements = mesh.GetGlobalNE();
   const HYPRE_BigInt global_displacement_dofs = space.GlobalTrueVSize();
   if (Mpi::Root())
   {
      std::string output_path(output_prefix);
      if (!output_path.empty() && output_path.back() != '/')
      {
         output_path += '/';
      }
      output_path += collection_name + "/" + collection_name + ".pvd";
      std::cout << "Cantilever dimension: " << dimension << '\n'
                << "Excitation: " << excitation << '\n'
                << "Angular frequency: " << frequency << '\n'
                << "Global elements: " << global_elements << '\n'
                << "Global displacement DOFs: "
                << global_displacement_dofs << '\n'
                << "Outer iterations: " << iterations << '\n'
                << "Total assembly time (max): " << assembly_time << " s\n"
                << "Operator assembly time (max): "
                << operator_assembly_time << " s\n"
                << "Preconditioner assembly time (max): "
                << preconditioner_assembly_time << " s\n"
                << "Solver setup time (max): " << solver_setup_time << " s\n"
                << "Solve time (max): " << solve_time << " s\n"
                << "ParaView collection: " << output_path << '\n';
   }

   return EXIT_SUCCESS;
}
