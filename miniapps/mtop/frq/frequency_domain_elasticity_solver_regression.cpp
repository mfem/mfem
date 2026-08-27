// Regression coverage for FrequencyDomainLinearElasticitySolver.

#include "frequency_domain_elasticity_solver.hpp"

#include <cmath>
#include <complex>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>

using namespace mfem;

namespace
{

/// Return the largest value held by any rank in @a communicator.
real_t GlobalMaximum(const real_t local_value, const MPI_Comm communicator)
{
   real_t global_value = 0.0;
   MPI_Allreduce(&local_value, &global_value, 1, MPITypeMap<real_t>::mpi_type,
                 MPI_MAX, communicator);
   return global_value;
}

/// Print and return one communicator-wide scalar comparison.
bool CheckValue(const std::string &name, const real_t value,
                const real_t tolerance, const MPI_Comm communicator)
{
   const real_t global_value = GlobalMaximum(value, communicator);
   const bool passed = global_value <= tolerance;
   if (Mpi::Root())
   {
      std::cout << (passed ? "PASS  " : "FAIL  ") << name
                << ": " << global_value << '\n';
   }
   return passed;
}

/// Fill a true-dof vector deterministically without assuming host execution.
void FillExactVector(Vector &vector)
{
   vector.UseDevice(true);
   real_t *values = vector.HostWrite();
   for (int i = 0; i < vector.Size(); ++i)
   {
      values[i] = 0.2*std::sin(0.37*(i + 1))
                  + 0.1*std::cos(0.19*(i + 2));
   }
}

/// Impose homogeneous values in both complex blocks of a test vector.
void ZeroEssentialEntries(const Array<int> &essential_tdofs, Vector &vector)
{
   const int block_size = vector.Size()/2;
   real_t *values = vector.HostReadWrite();
   for (int i = 0; i < essential_tdofs.Size(); ++i)
   {
      values[essential_tdofs[i]] = 0.0;
      values[block_size + essential_tdofs[i]] = 0.0;
   }
}

/// Vector coefficient whose value can be changed between solver applications.
class MutableVectorCoefficient : public VectorCoefficient
{
public:
   explicit MutableVectorCoefficient(const Vector &value)
      : VectorCoefficient(value.Size()), value_(value) { }

   void SetValue(const Vector &value)
   {
      MFEM_VERIFY(value.Size() == GetVDim(),
                  "Mutable coefficient dimension changed.");
      value_ = value;
   }

   void Eval(Vector &value, ElementTransformation &,
             const IntegrationPoint &) override
   {
      value = value_;
   }

private:
   Vector value_;
};

/// Apply the common low-frequency material and homogeneous boundary setup.
void ConfigureSolver(FrequencyDomainLinearElasticitySolver &solver)
{
   solver.SetLambda(2.3);
   solver.SetMu(1.7);
   solver.SetDensity(0.9);
   solver.SetFrequency(0.2);
   solver.SetRayleighDamping(0.08, 0.015);
   solver.AddBoundaryID(1);
   solver.SetRelTol(1.0e-11);
   solver.SetAbsTol(1.0e-14);
   solver.SetMaxIter(500);
   solver.SetKDim(50);
   solver.SetPrintLevel(-1);
   solver.SetPreconditionerRelTol(1.0e-5);
   solver.SetPreconditionerAbsTol(1.0e-14);
   solver.SetPreconditionerMaxIter(100);
   solver.SetPreconditionerPrintLevel(-1);
}

/// Verify a forward or transpose solve against a manufactured true-dof state.
bool CheckManufacturedSolve(
   const std::string &name, FrequencyDomainLinearElasticitySolver &solver,
   const bool transpose, const real_t tolerance)
{
   solver.Assemble();
   Vector exact(solver.Height());
   FillExactVector(exact);
   ZeroEssentialEntries(solver.GetEssentialTrueDofs(), exact);

   Vector rhs(solver.Height());
   if (transpose)
   {
      solver.GetFrequencyDomainOperator().MultTranspose(exact, rhs);
   }
   else
   {
      solver.GetFrequencyDomainOperator().Mult(exact, rhs);
   }

   Vector solution(solver.Width());
   solution.UseDevice(true);
   solution = 0.0;
   if (transpose)
   {
      solver.MultTranspose(rhs, solution);
   }
   else
   {
      solver.Mult(rhs, solution);
   }
   solution -= exact;
   return CheckValue(name, solution.Normlinf(), tolerance,
                     solver.GetFESpace().GetComm());
}

/// Configure the unconstrained operator used to manufacture nonzero BC data.
void ConfigureReferenceOperator(FrequencyDomainElasticityOperator &op)
{
   op.SetLameMaterial(std::make_shared<ConstantCoefficient>(2.3),
                      std::make_shared<ConstantCoefficient>(1.7),
                      std::make_shared<ConstantCoefficient>(0.9));
   op.SetFrequency(0.2);
   op.SetRayleighDamping(0.08, 0.015);
   op.Assemble();
}

/// Verify complex nonzero displacement lifting against an unconstrained RHS.
bool CheckComplexBoundaryLifting(ParFiniteElementSpace &space)
{
   FrequencyDomainElasticityOperator reference(space);
   ConfigureReferenceOperator(reference);

   FrequencyDomainLinearElasticitySolver solver(space);
   ConfigureSolver(solver);
   solver.ClearBoundaryConditions();

   Vector real_boundary(space.GetVDim());
   Vector imaginary_boundary(space.GetVDim());
   real_boundary = 0.0;
   imaginary_boundary = 0.0;
   real_boundary(0) = 0.035;
   imaginary_boundary(0) = -0.012;
   if (space.GetVDim() > 1)
   {
      real_boundary(1) = -0.021;
      imaginary_boundary(1) = 0.027;
   }
   std::shared_ptr<MutableVectorCoefficient> real_coefficient(
      new MutableVectorCoefficient(real_boundary));
   std::shared_ptr<MutableVectorCoefficient> imaginary_coefficient(
      new MutableVectorCoefficient(imaginary_boundary));
   solver.AddDisplacementBC(1, real_coefficient, imaginary_coefficient);

   ParComplexGridFunction exact_grid_function(&space);
   FillExactVector(exact_grid_function);
   exact_grid_function.real().SyncMemory(exact_grid_function);
   exact_grid_function.imag().SyncMemory(exact_grid_function);
   Array<int> marker(space.GetParMesh()->bdr_attributes.Max());
   marker = 0;
   marker[0] = 1;
   exact_grid_function.real().ProjectBdrCoefficient(
      *real_coefficient, marker);
   exact_grid_function.imag().ProjectBdrCoefficient(
      *imaginary_coefficient, marker);
   exact_grid_function.real().SyncAliasMemory(exact_grid_function);
   exact_grid_function.imag().SyncAliasMemory(exact_grid_function);

   Vector exact_true(2*space.GetTrueVSize());
   exact_true.UseDevice(true);
   exact_grid_function.ParallelProject(exact_true);
   Vector rhs(reference.Height());
   reference.Mult(exact_true, rhs);
   Vector solution;
   solver.Mult(rhs, solution);
   solution -= exact_true;
   const bool initial_solve =
      CheckValue("complex displacement lifting", solution.Normlinf(),
                 1.0e-7, space.GetComm());

   // A homogeneous transpose solve must not overwrite the cached nonzero
   // boundary values used by later forward solves.
   Vector transpose_exact(solver.Height());
   FillExactVector(transpose_exact);
   ZeroEssentialEntries(solver.GetEssentialTrueDofs(), transpose_exact);
   Vector transpose_rhs(solver.Height());
   solver.GetFrequencyDomainOperator().MultTranspose(transpose_exact,
                                                     transpose_rhs);
   Vector transpose_solution;
   solver.MultTranspose(transpose_rhs, transpose_solution);

   Vector repeated_solution;
   solver.Mult(rhs, repeated_solution);
   repeated_solution -= exact_true;
   const bool repeated_solve =
      CheckValue("complex displacement lifting after transpose",
                 repeated_solution.Normlinf(), 1.0e-7, space.GetComm());

   // A coefficient changed in place must be explicitly invalidated without
   // forcing the matrix and preconditioner to be rebuilt.
   real_boundary(0) += 0.013;
   imaginary_boundary(0) -= 0.009;
   real_coefficient->SetValue(real_boundary);
   imaginary_coefficient->SetValue(imaginary_boundary);
   solver.BoundaryValuesChanged();
   exact_grid_function.real().ProjectBdrCoefficient(
      *real_coefficient, marker);
   exact_grid_function.imag().ProjectBdrCoefficient(
      *imaginary_coefficient, marker);
   exact_grid_function.real().SyncAliasMemory(exact_grid_function);
   exact_grid_function.imag().SyncAliasMemory(exact_grid_function);
   exact_grid_function.ParallelProject(exact_true);
   reference.Mult(exact_true, rhs);
   Vector updated_solution;
   solver.Mult(rhs, updated_solution);
   updated_solution -= exact_true;
   const bool updated_solve =
      CheckValue("updated complex displacement lifting",
                 updated_solution.Normlinf(), 1.0e-7, space.GetComm());
   return initial_solve && repeated_solve && updated_solve;
}

/// Verify real, imaginary-only, volume, and traction load assembly by residual.
bool CheckComplexLoadInterface(ParFiniteElementSpace &space)
{
   FrequencyDomainLinearElasticitySolver solver(space);
   ConfigureSolver(solver);

   Vector real_load(space.GetVDim());
   Vector imaginary_load(space.GetVDim());
   real_load = 0.0;
   imaginary_load = 0.0;
   real_load(0) = 0.4;
   imaginary_load(space.GetVDim() - 1) = -0.3;
   std::shared_ptr<VectorCoefficient> real_coefficient(
      new VectorConstantCoefficient(real_load));
   std::shared_ptr<VectorCoefficient> imaginary_coefficient(
      new VectorConstantCoefficient(imaginary_load));
   solver.AddVolumeLoad(1, std::shared_ptr<VectorCoefficient>(),
                        imaginary_coefficient);
   solver.AddBoundaryLoad(2, real_coefficient, imaginary_coefficient);

   ParComplexGridFunction solution(&space);
   solution = std::complex<real_t>(0.0, 0.0);
   solver.Solve(solution);

   Array<int> domain_marker(space.GetParMesh()->attributes.Max());
   domain_marker = 0;
   domain_marker[0] = 1;
   Array<int> boundary_marker(space.GetParMesh()->bdr_attributes.Max());
   boundary_marker = 0;
   boundary_marker[1] = 1;

   ParLinearForm real_form(&space);
   ParLinearForm imaginary_form(&space);
   imaginary_form.AddDomainIntegrator(
      new VectorDomainLFIntegrator(*imaginary_coefficient), domain_marker);
   real_form.AddBoundaryIntegrator(
      new VectorBoundaryLFIntegrator(*real_coefficient), boundary_marker);
   imaginary_form.AddBoundaryIntegrator(
      new VectorBoundaryLFIntegrator(*imaginary_coefficient), boundary_marker);
   real_form.Assemble();
   imaginary_form.Assemble();

   Vector reference_rhs(2*space.GetTrueVSize());
   Vector real_rhs;
   Vector imaginary_rhs;
   real_rhs.MakeRef(reference_rhs, 0, space.GetTrueVSize());
   imaginary_rhs.MakeRef(reference_rhs, space.GetTrueVSize(),
                         space.GetTrueVSize());
   real_form.ParallelAssemble(real_rhs);
   imaginary_form.ParallelAssemble(imaginary_rhs);
   real_rhs.SyncAliasMemory(reference_rhs);
   imaginary_rhs.SyncAliasMemory(reference_rhs);
   ZeroEssentialEntries(solver.GetEssentialTrueDofs(), reference_rhs);

   Vector solution_true(2*space.GetTrueVSize());
   solution.ParallelProject(solution_true);
   Vector residual(reference_rhs.Size());
   solver.GetFrequencyDomainOperator().Mult(solution_true, residual);
   residual -= reference_rhs;
   return CheckValue("complex load assembly residual", residual.Normlinf(),
                     1.0e-7, space.GetComm());
}

/// Exercise ownership transfer when one object supplies multiple coefficients.
bool CheckAliasedOwnershipTransfer(ParFiniteElementSpace &space)
{
   {
      FrequencyDomainLinearElasticitySolver solver(space);
      ConstantCoefficient *material = new ConstantCoefficient(1.0);
      solver.SetLameMaterial(*material, *material, *material, true);

      Vector load_value(space.GetVDim());
      load_value = 0.25;
      VectorConstantCoefficient *load =
         new VectorConstantCoefficient(load_value);
      solver.AddVolumeLoad(1, *load, *load, true);
   }
   return CheckValue("aliased coefficient ownership transfer", 0.0, 0.0,
                     space.GetComm());
}

} // namespace

/// Exercise all initial solver/preconditioner combinations on a small mesh.
int main(int argc, char *argv[])
{
   Mpi::Init(argc, argv);
   Hypre::Init();

   const char *device_configuration = "cpu";
   int order = 2;
   OptionsParser args(argc, argv);
   args.AddOption(&device_configuration, "-d", "--device",
                  "MFEM device configuration.");
   args.AddOption(&order, "-o", "--order", "Finite element order.");
   args.ParseCheck();
   Device device(device_configuration);

   Mesh serial_mesh = Mesh::MakeCartesian2D(
                         3, 2, Element::QUADRILATERAL, true, 1.0, 0.75);
   ParMesh mesh(MPI_COMM_WORLD, serial_mesh);
   H1_FECollection collection(order, mesh.Dimension());
   ParFiniteElementSpace space(&mesh, &collection, mesh.SpaceDimension(),
                               Ordering::byNODES);
   ParFiniteElementSpace material_space(&mesh, &collection);
   ParGridFunction lambda_field(&material_space);
   ParGridFunction mu_field(&material_space);
   ParGridFunction density_field(&material_space);
   ParGridFunction mass_damping_field(&material_space);
   ParGridFunction damping_lambda_field(&material_space);
   ParGridFunction damping_mu_field(&material_space);
   lambda_field = 2.3;
   mu_field = 1.7;
   density_field = 0.9;
   mass_damping_field = 0.025;
   damping_lambda_field = 0.012;
   damping_mu_field = 0.009;

   int failures = 0;
   {
      FrequencyDomainLinearElasticitySolver solver(space);
      ConfigureSolver(solver);
      solver.SetPreconditionerType(
         FrequencyDomainLinearElasticitySolver::PreconditionerType::PRESB);
      solver.SetHInverseType(
         FrequencyDomainLinearElasticitySolver::HInverseType::
         LORMonolithicAMG);
      failures += !CheckManufacturedSolve("PRESB/GMRES forward", solver,
                                          false, 1.0e-7);
      failures += !CheckManufacturedSolve("PRESB/GMRES transpose", solver,
                                          true, 1.0e-7);
   }
   {
      FrequencyDomainLinearElasticitySolver solver(space);
      ConfigureSolver(solver);
      solver.SetPreconditionerType(
         FrequencyDomainLinearElasticitySolver::PreconditionerType::
         BlockDiagonal);
      solver.SetHInverseType(
         FrequencyDomainLinearElasticitySolver::HInverseType::
         LORMonolithicAMG);
      solver.SetLOROrdering(Ordering::byVDIM);
      failures += !CheckManufacturedSolve("block diagonal/MINRES forward",
                                          solver, false, 1.0e-7);
      failures += !CheckManufacturedSolve("block diagonal/MINRES transpose",
                                          solver, true, 1.0e-7);
   }
   {
      FrequencyDomainLinearElasticitySolver solver(space);
      ConfigureSolver(solver);
      solver.SetLameMaterialFields(lambda_field, mu_field, density_field);
      solver.SetDampingCoefficientFields(
         mass_damping_field, damping_lambda_field, damping_mu_field);
      solver.SetHInverseType(
         FrequencyDomainLinearElasticitySolver::HInverseType::
         LORMonolithicCGAMG);
      solver.Assemble();
      // Inspection getters must remain safe when called only on rank zero.
      int rank = 0;
      MPI_Comm_rank(space.GetComm(), &rank);
      int selected_fgmres = 0;
      if (rank == 0)
      {
         // SetOperator must not perform collective lazy assembly.
         solver.SetOperator(solver.GetFrequencyDomainOperator());
         selected_fgmres =
            solver.GetActiveLinearSolverType() ==
            FrequencyDomainLinearElasticitySolver::LinearSolverType::FGMRES;
      }
      MPI_Bcast(&selected_fgmres, 1, MPI_INT, 0, space.GetComm());
      failures += !CheckValue("automatic FGMRES selection",
                              selected_fgmres ? 0.0 : 1.0, 0.0,
                              space.GetComm());
      failures += !CheckManufacturedSolve("nested CG/AMG forward", solver,
                                          false, 2.0e-7);
   }

   failures += !CheckComplexBoundaryLifting(space);
   failures += !CheckComplexLoadInterface(space);
   failures += !CheckAliasedOwnershipTransfer(space);

#ifdef MFEM_USE_MUMPS
   {
      FrequencyDomainLinearElasticitySolver solver(space);
      ConfigureSolver(solver);
      solver.SetHInverseType(
         FrequencyDomainLinearElasticitySolver::HInverseType::MUMPS);
      failures += !CheckManufacturedSolve("PRESB/MUMPS-H forward", solver,
                                          false, 1.0e-9);
   }
   {
      FrequencyDomainLinearElasticitySolver solver(space);
      ConfigureSolver(solver);
      solver.SetLinearSolverType(
         FrequencyDomainLinearElasticitySolver::LinearSolverType::MUMPS);
      failures += !CheckManufacturedSolve("MUMPS forward", solver, false,
                                          1.0e-10);
      failures += !CheckManufacturedSolve("MUMPS transpose", solver, true,
                                          1.0e-10);
   }
#endif

   int global_failures = 0;
   MPI_Allreduce(&failures, &global_failures, 1, MPI_INT, MPI_MAX,
                 space.GetComm());
   if (Mpi::Root())
   {
      std::cout << (global_failures == 0 ?
                    "ALL TESTS PASSED\n" : "TESTS FAILED\n");
   }
   return global_failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
