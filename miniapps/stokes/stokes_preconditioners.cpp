#include "stokes_preconditioners.hpp"

namespace mfem::stokes::detail
{

P1AMGPreconditioner::P1AMGPreconditioner(
   const HypreParMatrix &matrix_, ParFiniteElementSpace &space,
   const std::vector<Array<int>> &boundary_markers, int smoothing_steps,
   bool elasticity, bool refine, const CoarseAssembler &assemble)
   : Solver(matrix_.Height()), matrix(matrix_),
     coarse_collection(1, space.GetParMesh()->Dimension()),
     coarse_space(space.GetParMesh(), &coarse_collection, space.GetVDim(),
                  space.GetOrdering()),
     transfer(coarse_space, space, true),
     prolongation(dynamic_cast<HypreParMatrix*>(transfer.GetTrueTransferOperator())),
     smoother(matrix, HypreSmoother::l1Jacobi, smoothing_steps),
     residual(height), correction(height),
     coarse_rhs(coarse_space.GetTrueVSize()), coarse_solution(coarse_rhs.Size())
{
   MFEM_VERIFY(prolongation && smoothing_steps > 0, "Invalid P1-AMG setup");
   MFEM_VERIFY(boundary_markers.empty() ||
               boundary_markers.size() == static_cast<size_t>(space.GetVDim()),
               "P1-AMG needs one boundary marker per component");
   Array<int> fine_essential, coarse_essential, selected;
   for (int component = 0; component < static_cast<int>(boundary_markers.size()); ++component)
   {
      space.GetEssentialTrueDofs(boundary_markers[component], selected, component);
      fine_essential.Append(selected);
      coarse_space.GetEssentialTrueDofs(boundary_markers[component], selected, component);
      coarse_essential.Append(selected);
   }
   fine_essential.Sort();
   coarse_essential.Sort();
   prolongation->EliminateRows(fine_essential);
   delete prolongation->EliminateCols(coarse_essential);
   if (assemble) { coarse_matrix = assemble(coarse_space); }
   else { coarse_matrix.reset(RAP(&matrix, prolongation)); }
   MFEM_VERIFY(coarse_matrix && coarse_matrix->Height() == coarse_space.GetTrueVSize() &&
               coarse_matrix->Width() == coarse_space.GetTrueVSize(),
               "Invalid P1-AMG coarse matrix");
   delete coarse_matrix->EliminateRowsCols(coarse_essential);
   coarse_amg = std::make_unique<HypreBoomerAMG>(*coarse_matrix);
   if (elasticity) { coarse_amg->SetElasticityOptions(&coarse_space, refine); }
   coarse_amg->SetMaxIter(1);
   coarse_amg->SetTol(0.0);
   coarse_amg->SetPrintLevel(0);
   coarse_amg->iterative_mode = false;
   smoother.iterative_mode = false;
   coarse_rhs = 0.0;
   coarse_solution = 0.0;
   coarse_amg->Setup(coarse_rhs, coarse_solution);
}

void P1AMGPreconditioner::SetOperator(const Operator &)
{
   MFEM_ABORT("P1AMGPreconditioner has fixed operators");
}

void P1AMGPreconditioner::Mult(const Vector &input, Vector &output) const
{
   output.SetSize(height);
   smoother.Mult(input, output);
   matrix.Mult(output, residual);
   subtract(input, residual, residual);
   prolongation->MultTranspose(residual, coarse_rhs);
   coarse_amg->Mult(coarse_rhs, coarse_solution);
   prolongation->Mult(coarse_solution, correction);
   output += correction;
   matrix.Mult(output, residual);
   subtract(input, residual, residual);
   smoother.Mult(residual, correction);
   output += correction;
}

UpperTriangularPreconditioner::UpperTriangularPreconditioner(
   const Array<int> &offsets_, Solver &velocity_inverse_, Solver &pressure_inverse_,
   const Operator &momentum_pressure_, Solver *gauge_inverse_, const Operator *gauge_column_)
   : Solver(offsets_.Last()), offsets(offsets_),
     velocity_inverse(velocity_inverse_), pressure_inverse(pressure_inverse_),
     momentum_pressure(momentum_pressure_), gauge_inverse(gauge_inverse_),
     gauge_column(gauge_column_), work(offsets),
     velocity_work(velocity_inverse.Height()), pressure_work(pressure_inverse.Height())
{
   MFEM_VERIFY((offsets.Size() == 3 && !gauge_inverse && !gauge_column) ||
               (offsets.Size() == 4 && gauge_inverse && gauge_column),
               "Inconsistent upper-triangular gauge blocks");
}

void UpperTriangularPreconditioner::SetOperator(const Operator &)
{
   MFEM_ABORT("UpperTriangularPreconditioner has fixed blocks");
}

void UpperTriangularPreconditioner::Mult(const Vector &input, Vector &output) const
{
   work.Vector::operator=(input);
   output.SetSize(height);
   BlockVector result(output, offsets);
   if (gauge_inverse)
   {
      gauge_inverse->Mult(work.GetBlock(2), result.GetBlock(2));
      gauge_column->Mult(result.GetBlock(2), pressure_work);
      work.GetBlock(1) -= pressure_work;
   }
   pressure_inverse.Mult(work.GetBlock(1), result.GetBlock(1));
   momentum_pressure.Mult(result.GetBlock(1), velocity_work);
   work.GetBlock(0) -= velocity_work;
   velocity_inverse.Mult(work.GetBlock(0), result.GetBlock(0));
}

}
