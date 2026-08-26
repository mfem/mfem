// Compare the frequency-domain elasticity PA actions with assembled Hypre
// matrices on a small distributed elasticity problem.

#include "frequency_domain_elasticity.hpp"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>

using namespace mfem;

namespace
{

/// Fill a distributed true-dof vector with deterministic nonzero data.
void FillVector(Vector &vector)
{
   for (int i = 0; i < vector.Size(); ++i)
   {
      vector(i) = std::sin(0.17*(i + 1)) + 0.25*std::cos(0.31*(i + 1));
   }
}

/// Compare two operator actions using the maximum local infinity norm.
bool CompareAction(const std::string &name, const Operator &matrix_free,
                   const Operator &assembled, const Vector &input,
                   const MPI_Comm communicator, const real_t tolerance)
{
   MFEM_VERIFY(matrix_free.Width() == input.Size() &&
               assembled.Width() == input.Size() &&
               matrix_free.Height() == assembled.Height(),
               "Compared operator dimensions must agree.");
   Vector matrix_free_result(matrix_free.Height());
   Vector assembled_result(assembled.Height());
   matrix_free.Mult(input, matrix_free_result);
   assembled.Mult(input, assembled_result);
   matrix_free_result -= assembled_result;

   const double local_error = matrix_free_result.Normlinf();
   double global_error = 0.0;
   MPI_Allreduce(&local_error, &global_error, 1, MPI_DOUBLE, MPI_MAX,
                 communicator);
   if (Mpi::Root())
   {
      std::cout << (global_error <= tolerance ? "PASS  " : "FAIL  ")
                << name << ": |error|_inf = " << global_error << '\n';
   }
   return global_error <= tolerance;
}

} // namespace

/// Build a small elasticity problem and compare every exposed PA operator
/// against its independently assembled Hypre representation.
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

   ConstantCoefficient lambda(2.3);
   ConstantCoefficient mu(1.7);
   ConstantCoefficient density(0.9);
   FrequencyDomainElasticityOperator vibration(space);
   vibration.SetLameMaterial(lambda, mu, density);
   vibration.SetRayleighDamping(0.08, 0.015);
   vibration.SetFrequency(1.4);

   Array<int> essential_boundary(mesh.bdr_attributes.Max());
   essential_boundary = 0;
   essential_boundary[0] = 1;
   Array<int> essential_tdofs;
   space.GetEssentialTrueDofs(essential_boundary, essential_tdofs);
   vibration.SetEssentialTrueDofs(essential_tdofs);
   vibration.Assemble();

   Vector component_input(space.GetTrueVSize());
   Vector complex_input(2*space.GetTrueVSize());
   FillVector(component_input);
   FillVector(complex_input);

   int failures = 0;
   const real_t tolerance = 1.0e-10;
   std::unique_ptr<HypreParMatrix> W1 = vibration.FormW1Matrix();
   std::unique_ptr<HypreParMatrix> W2 = vibration.FormW2Matrix();
   std::unique_ptr<HypreParMatrix> T = vibration.FormTMatrix();
   std::unique_ptr<HypreParMatrix> W = vibration.FormWMatrix();
   std::unique_ptr<HypreParMatrix> H = vibration.FormHMatrix();
   failures += !CompareAction("W1=K", vibration.GetW1Operator(), *W1,
                              component_input, mesh.GetComm(), tolerance);
   failures += !CompareAction("W2=omega^2 M", vibration.GetW2Operator(), *W2,
                              component_input, mesh.GetComm(), tolerance);
   failures += !CompareAction("T=omega C", vibration.GetTOperator(), *T,
                              component_input, mesh.GetComm(), tolerance);
   failures += !CompareAction("W=K-omega^2 M", vibration.GetWOperator(), *W,
                              component_input, mesh.GetComm(), tolerance);
   failures += !CompareAction("H=W+T", vibration.GetHOperator(), *H,
                              component_input, mesh.GetComm(), tolerance);

   std::unique_ptr<ComplexOperator> block_operator =
      vibration.FormBlockOperator();
   std::unique_ptr<ComplexHypreParMatrix> assembled_complex =
      vibration.FormAssembledComplexOperator();
   failures += !CompareAction("matrix-free complex block operator",
                              vibration, *block_operator, complex_input,
                              mesh.GetComm(), tolerance);
   failures += !CompareAction("assembled complex block operator",
                              vibration, *assembled_complex, complex_input,
                              mesh.GetComm(), tolerance);

   std::unique_ptr<ComplexOperator> symmetric_block_operator =
      vibration.FormBlockOperator(ComplexOperator::BLOCK_SYMMETRIC);
   std::unique_ptr<ComplexHypreParMatrix> assembled_symmetric =
      vibration.FormAssembledComplexOperator(
         ComplexOperator::BLOCK_SYMMETRIC);
   failures += !CompareAction("symmetric complex block operator",
                              *symmetric_block_operator,
                              *assembled_symmetric, complex_input,
                              mesh.GetComm(), tolerance);

#ifdef MFEM_USE_MUMPS
   // Materialize the real system matrix only for the direct solver. The
   // HERMITIAN convention [W,-T;T,W] is generally unsymmetric.
   std::unique_ptr<HypreParMatrix> real_matrix(
      assembled_complex->GetSystemMatrix());
   Vector direct_rhs(real_matrix->Height());
   real_matrix->Mult(complex_input, direct_rhs);
   Vector direct_solution(real_matrix->Width());
   direct_solution = 0.0;
   MUMPSSolver direct_solver(real_matrix->GetComm());
   direct_solver.SetPrintLevel(0);
   direct_solver.SetMatrixSymType(MUMPSSolver::MatType::UNSYMMETRIC);
   direct_solver.SetOperator(*real_matrix);
   direct_solver.Mult(direct_rhs, direct_solution);
   direct_solution -= complex_input;
   const double local_direct_error = direct_solution.Normlinf();
   double global_direct_error = 0.0;
   MPI_Allreduce(&local_direct_error, &global_direct_error, 1, MPI_DOUBLE,
                 MPI_MAX, mesh.GetComm());
   failures += global_direct_error > tolerance;
   if (Mpi::Root())
   {
      std::cout << (global_direct_error <= tolerance ? "PASS  " : "FAIL  ")
                << "MUMPS direct solve: |error|_inf = "
                << global_direct_error << '\n';
   }
#endif

   // Select a damping operator with independent mass, lambda, and mu
   // coefficients, then compare both its PA and assembled actions.
   block_operator.reset();
   symmetric_block_operator.reset();
   ConstantCoefficient mass_damping(0.11);
   ConstantCoefficient damping_lambda(0.07);
   ConstantCoefficient damping_mu(0.04);
   vibration.SetDampingCoefficients(mass_damping, damping_lambda,
                                    damping_mu);
   std::unique_ptr<HypreParMatrix> coefficient_T =
      vibration.FormTMatrix();
   std::unique_ptr<HypreParMatrix> coefficient_H =
      vibration.FormHMatrix();
   failures += !CompareAction("coefficient damping T",
                              vibration.GetTOperator(), *coefficient_T,
                              component_input, mesh.GetComm(), tolerance);
   failures += !CompareAction("coefficient damping H",
                              vibration.GetHOperator(), *coefficient_H,
                              component_input, mesh.GetComm(), tolerance);
   std::unique_ptr<ComplexHypreParMatrix> coefficient_damped_matrix =
      vibration.FormAssembledComplexOperator();
   failures += !CompareAction("coefficient-damped complex operator",
                              vibration, *coefficient_damped_matrix,
                              complex_input, mesh.GetComm(), tolerance);

   // Replace the complete material and verify that stale PA and Hypre data are
   // rebuilt from the new engineering parameters.
   ConstantCoefficient young_modulus(6.0);
   ConstantCoefficient poisson_ratio(0.27);
   ConstantCoefficient updated_density(1.1);
   vibration.SetEngineeringMaterial(young_modulus, poisson_ratio,
                                    updated_density);
   std::unique_ptr<ComplexHypreParMatrix> updated_matrix =
      vibration.FormAssembledComplexOperator();
   failures += !CompareAction("updated isotropic material", vibration,
                              *updated_matrix, complex_input,
                              mesh.GetComm(), tolerance);

   int global_failures = 0;
   MPI_Allreduce(&failures, &global_failures, 1, MPI_INT, MPI_MAX,
                 mesh.GetComm());
   if (Mpi::Root())
   {
      std::cout << (global_failures == 0 ?
                    "ALL TESTS PASSED\n" : "TESTS FAILED\n");
   }
   return global_failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
