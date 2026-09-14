#include "mfem.hpp"

using namespace mfem;

void reconstructH1Field(const ParGridFunction &src, ParGridFunction &dst)
{
   Vector dst_dofs;
   dst.GetTrueDofs(dst_dofs);

   OperatorPtr M_op;
   ParBilinearForm M(dst.ParFESpace());
   M.AddDomainIntegrator(new MassIntegrator());
   M.Assemble();
   M.FormSystemMatrix({}, M_op);

   OperatorPtr Rhs_op;
   Vector src_dofs;
   Vector rhs(dst_dofs.Size());
   ParMixedBilinearForm Rhs(src.ParFESpace(), dst.ParFESpace());
   Rhs.AddDomainIntegrator(new MassIntegrator());
   Rhs.Assemble();
   Rhs.FormRectangularSystemMatrix({}, {}, Rhs_op);
   src.GetTrueDofs(src_dofs);
   Rhs_op->Mult(src_dofs, rhs);

   CGSolver solver(dst.ParFESpace()->GetComm());
   solver.iterative_mode = false;
   solver.SetRelTol(1e-5);
   solver.SetAbsTol(0.0);
   solver.SetMaxIter(30);
   solver.SetPrintLevel(0);
   HypreSmoother prec;
   prec.SetType(HypreSmoother::Jacobi);
   solver.SetPreconditioner(prec);
   solver.SetOperator(*M_op);
   solver.Mult(rhs, dst_dofs);
   MFEM_VERIFY(solver.GetConverged(), "Solver did not converge.");
   dst.SetFromTrueDofs(dst_dofs);
}


void reconstructL2Field(const ParGridFunction& src, ParGridFunction& dst)
{
   // todo: improve checking when dst is L2, p=0
   if (dst.FESpace()->GetOrder(0) == 0)
   {
      dst = src;
      return;
   }

   MassIntegrator mass;
   const FiniteElementSpace& src_fe_space = *(src.FESpace()); // u_hat space
   const FiniteElementSpace& dst_fe_space = *(dst.FESpace()); // u space

   std::unique_ptr<HypreParMatrix> matrix;

   // compute <{u_hat} (xhat dot n), psi_hat>_E
   VectorConstantCoefficient xhat(Vector({1.0,0.0}));
   Vector b_xhat(src_fe_space.GetNE());
   ParBilinearForm B_xhat(src.ParFESpace());
   B_xhat.AddInteriorFaceIntegrator(new DGTraceIntegrator(xhat, 1.0, 0.0));
   B_xhat.AddBdrFaceIntegrator(new DGTraceIntegrator(xhat, 2.0, 0.0)); // note the 2 enforces du/dx=0 at the x boundaries
   B_xhat.Assemble();
   B_xhat.Finalize();
   matrix = std::unique_ptr<HypreParMatrix>(B_xhat.ParallelAssemble());
   matrix->Mult(*src.GetTrueDofs(), b_xhat);

   // compute <{u_hat} (yhat dot n), psi_hat>_E
   VectorConstantCoefficient yhat(Vector({0.0,1.0}));
   Vector b_yhat(src_fe_space.GetNE());
   ParBilinearForm B_yhat(src.ParFESpace());
   B_yhat.AddInteriorFaceIntegrator(new DGTraceIntegrator(yhat, 1.0, 0.0));
   B_yhat.AddBdrFaceIntegrator(new DGTraceIntegrator(yhat, 2.0, 0.0)); // note the 2 enforces du/dy=0 at the y boundaries
   B_yhat.Assemble();
   B_yhat.Finalize();
   matrix = std::unique_ptr<HypreParMatrix>(B_yhat.ParallelAssemble());
   matrix->Mult(*src.GetTrueDofs(), b_yhat);

   MixedDirectionalDerivativeIntegrator partial_x(xhat);
   MixedDirectionalDerivativeIntegrator partial_y(yhat);
   for (int element_ind=0; element_ind < src_fe_space.GetNE(); element_ind++)
   {
      const FiniteElement& src_element = *(src_fe_space.GetFE(element_ind));
      const FiniteElement& dst_element = *(dst_fe_space.GetFE(element_ind));
      ElementTransformation& transform = *(src_fe_space.GetElementTransformation(element_ind));
      DenseMatrix A(dst_element.GetDof());
      Vector b(dst_element.GetDof());
      DenseMatrix Arow;

      // enforce (u, psi_hat)_E = ({u_hat}, psi_hat)_E
      mass.AssembleElementMatrix2(dst_element, src_element, transform, Arow);
      A.SetSubMatrix(0, 0, Arow);
      Vector bmean(b, 0, src_element.GetDof());
      DenseMatrix Bmean;
      mass.AssembleElementMatrix(src_element, transform, Bmean);
      Vector src_dof_values;
      src.GetElementDofValues(element_ind, src_dof_values);
      Bmean.Mult(src_dof_values, bmean);

      // enforce (div[ u xhat ], psi_hat)_E = <{u_hat} (xhat dot n), psi_hat>_E,
      // i.e., (du/dx, psi_hat)_E = <{u_hat} (xhat dot n), psi_hat>_E
      partial_x.AssembleElementMatrix2(dst_element, src_element, transform, Arow);
      A.SetSubMatrix(1, 0, Arow);
      b[1] = b_xhat[element_ind];

      // enforce (div[ u yhat ], psi_hat)_E = <{u_hat} (yhat dot n), psi_hat>_E,
      // i.e., (du/dy, psi_hat)_E = <{u_hat} (yhat dot n), psi_hat>_E
      partial_y.AssembleElementMatrix2(dst_element, src_element, transform, Arow);
      A.SetSubMatrix(2, 0, Arow);
      b[2] = b_yhat[element_ind];

      // enforce (div[ (xhat \otimes yhat) grad[u] ], psi_hat)_E = 0, i.e.,
      // < du/dy (xhat dot n), psi_hat>_E = 0
      // TODO: replace with actual face integration
      A(3,0) = 1.0;
      A(3,1) = -1.0;
      A(3,2) = -1.0;
      A(3,3) = 1.0;
      b[3] = 0.0;

      // solve for u dof values
      A.Invert();
      mfem::Vector solution(dst_element.GetDof());
      A.Mult(b, solution);
      Array<int> dst_dof_indices;
      dst_fe_space.GetElementDofs(element_ind, dst_dof_indices);
      dst.SetSubVector(dst_dof_indices, solution);
   }
   dst.ExchangeFaceNbrData();
}
