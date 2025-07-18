// Copyright (c) 2010-2025, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#include "mfem.hpp"
#include <fstream>
#include <string>
#include <iomanip>

using namespace mfem;

namespace reconstruction
{

// Special integrator for the second derivative
class MixedBidirectionalHessianIntegrator : public MixedScalarIntegrator
{
private:
   VectorCoefficient *dir1_vq, *dir2_vq;
   Vector dir1_vq_ev, dir2_vq_ev;
   DenseMatrix d2shape, shape;

   void CalcTrialShape(const FiniteElement& trial_fe,
                       ElementTransformation& Trans,
                       Vector& trial_shape) override;

   void CalcTestShape(const FiniteElement& test_fe,
                      ElementTransformation& Trans,
                      Vector& test_shape) override;

   inline virtual bool VerifyFiniteElementTypes(const FiniteElement& trial_fe,
                                                const FiniteElement& test_fe) const override
   {
      return (trial_fe.GetDerivType() == mfem::FiniteElement::GRAD &&
              test_fe.GetDerivType() == mfem::FiniteElement::GRAD);
   };
public:
   MixedBidirectionalHessianIntegrator(VectorCoefficient& _dir1_vq,
                                       VectorCoefficient& _dir2_vq)
      : dir1_vq(&_dir1_vq), dir2_vq(&_dir2_vq) {};

   /* void AssembleElementMatrix2(const FiniteElement& trial_fe,
    *                             const FiniteElement& test_fe,
    *                             ElementTransformation& Trans,
    *                             DenseMatrix& elmat) override;
    */
};

void MixedBidirectionalHessianIntegrator::CalcTrialShape(
   const FiniteElement& trial_fe,
   ElementTransformation& Trans,
   Vector& trial_shape)
{
   int dim = Trans.GetDimension();
   dir1_vq->Eval(dir1_vq_ev, Trans, Trans.GetIntPoint());
   dir2_vq->Eval(dir2_vq_ev, Trans, Trans.GetIntPoint());
   d2shape.SetSize(trial_fe.GetDof(), dim*(dim+1)/2);
   // TODO(Gabriel): Requires CalcHessian on L2_Quad e.g.
   // trial_fe.CalcPhysHessian(Trans, d2shape);
   // Something like DenseMatrix::InnerProduct
};

void MixedBidirectionalHessianIntegrator::CalcTestShape(
   const FiniteElement& test_fe,
   ElementTransformation& Trans,
   Vector& test_shape)
{
   test_fe.CalcPhysShape(Trans, test_shape);
}

// reconstruct a field u (represented by dst) from u_hat (represented by src)
// Note current only 2D reconstruction of L^2_1 (piecewise-linear) field from
// L^2_0 (piecewise-constant) field is supported. The reconstruction is done
// by enforcing
//   1) element average: (u, psi_hat)_E = (u_hat, psi_hat)_E
//   2) grad: (div[ u e_i ], psi_hat)_E = <{u_hat} (e_i dot n), psi_hat>_E
//   3) grad^2: (div[ (e_i \otimes e_j) grad[u] ], psi_hat)_E =
//        <(grad[u_hat] dot e_j) e_i, psi_hat>_E ... which is 0
//   4) grad^3: (div[ (e_i \otimes e_j \otimes e_k) hessian[u] ], psi_hat)_E =
//        <[hessian[u_hat] e_k) dot e_j] e_i, psi_hat>_E ... which is 0
// where ( , )_E denotes area integral, < , >_E denotes the surface integral,
// psi_hat the L^2_0 basis function on element E, and e_* is the unit vector
// in the x_* direction
void reconstructField(const ParGridFunction& src, ParGridFunction& dst)
{
   MassIntegrator mass;
   const FiniteElementSpace& src_fe_space = *(src.FESpace()); // u_hat space
   const FiniteElementSpace& dst_fe_space = *(dst.FESpace()); // u space
   const Mesh& mesh = *(src_fe_space.GetMesh());

   std::unique_ptr<HypreParMatrix> matrix;

   // compute <{u_hat} (xhat dot n), psi_hat>_E
   VectorConstantCoefficient xhat(Vector({1.0,0.0}));
   Vector b_xhat(mesh.GetNE());
   ParBilinearForm B_xhat(src.ParFESpace());
   B_xhat.AddInteriorFaceIntegrator(new DGTraceIntegrator(xhat, 1.0, 0.0));
   B_xhat.AddBdrFaceIntegrator(new DGTraceIntegrator(xhat, 2.0,
                                                     0.0)); // note the 2 enforces du/dx=0 at the x boundaries
   B_xhat.Assemble();
   B_xhat.Finalize();
   matrix = std::unique_ptr<HypreParMatrix>(B_xhat.ParallelAssemble());
   matrix->Mult(*src.GetTrueDofs(), b_xhat);

   // compute <{u_hat} (yhat dot n), psi_hat>_E
   VectorConstantCoefficient yhat(Vector({0.0,1.0}));
   Vector b_yhat(mesh.GetNE());
   ParBilinearForm B_yhat(src.ParFESpace());
   B_yhat.AddInteriorFaceIntegrator(new DGTraceIntegrator(yhat, 1.0, 0.0));
   B_yhat.AddBdrFaceIntegrator(new DGTraceIntegrator(yhat, 2.0,
                                                     0.0)); // note the 2 enforces du/dy=0 at the y boundaries
   B_yhat.Assemble();
   B_yhat.Finalize();
   matrix = std::unique_ptr<HypreParMatrix>(B_yhat.ParallelAssemble());
   matrix->Mult(*src.GetTrueDofs(), b_yhat);

   MixedDirectionalDerivativeIntegrator partial_x(xhat);
   MixedDirectionalDerivativeIntegrator partial_y(yhat);
   MatrixConstantCoefficient xhatyhat(DenseMatrix({{0.0, 1.0},{0.0, 0.0}}));
   DiffusionIntegrator partial_xy(xhatyhat);
   // TODO(Gabriel): Requires CalcHessian for L2_Quad...
   // MixedBidirectionalHessianIntegrator partial_xy(xhat,yhat);
   for (int element_ind=0; element_ind < mesh.GetNE(); element_ind++)
   {
      const FiniteElement& src_element = *(src_fe_space.GetFE(element_ind));
      const FiniteElement& dst_element = *(dst_fe_space.GetFE(element_ind));
      ElementTransformation& transform = *(src_fe_space.GetElementTransformation(
                                              element_ind));
      DenseMatrix A(dst_element.GetDof());
      Vector b(dst_element.GetDof());
      DenseMatrix Arow;

      // enforce (u, psi_hat)_E = (u_hat, psi_hat)_E
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
      Vector temp;
      partial_xy.AssembleElementMatrix2(dst_element, dst_element, transform, Arow);
      Arow.GetDiag(temp);
      Arow.SetSize(src_element.GetDof(), dst_element.GetDof());
      Arow.SetRow(0, temp);
      A.SetSubMatrix(A.Height()-1, 0, Arow);
      b[3] = 0.0;

      // solve for u dof values
      A.Invert();
      Vector solution(dst_element.GetDof());
      A.Mult(b, solution);
      Array<int> dst_dof_indices;
      dst_fe_space.GetElementDofs(element_ind, dst_dof_indices);
      dst.SetSubVector(dst_dof_indices, solution);
   }
   dst.ExchangeFaceNbrData();
}

}  // end namespace reconstruction

using namespace reconstruction;

int main(int argc, char* argv[])
{
   Mpi::Init(argc, argv);

   // Default command-line options
   int ser_ref_levels = 0;
   int par_ref_levels = 0;

   int order_original = 3;
   int order_averages = 0;
   int order_reconstruction = 1;

   bool save_to_file = true;

   // Parse options
   OptionsParser args(argc, argv);
   args.AddOption(&ser_ref_levels, "-rs", "--refine",
                  "Number of serial refinement steps.");
   args.AddOption(&par_ref_levels, "-rp", "--refine",
                  "Number of parallel refinement steps.");

   args.AddOption(&order_original, "-O", "--order-original",
                  "Order original broken space.");
   // TODO(Gabriel): Not implemented yet
   // args.AddOption(&order_averages, "-A", "--order-averages",
   //                "Order averaged broken space.");
   args.AddOption(&order_reconstruction, "-R", "--order-reconstruction",
                  "Order of reconstruction broken space.");

   args.AddOption(&save_to_file, "-s", "--save", "-no-s",
                  "--no-save", "Show or not show approximation error.");

   args.ParseCheck();
   MFEM_VERIFY((ser_ref_levels >= 0) && (par_ref_levels >= 0), "")

   if (Mpi::Root())
   {
      mfem::out << "Number of serial refs.:  " << ser_ref_levels << "\n";
      mfem::out << "Number of parallel refs: " << par_ref_levels << "\n";
      mfem::out << "Original order:          " << order_original << "\n";
      mfem::out << "Original averages:       " << order_averages << "\n";
      mfem::out << "Original reconstruction: " << order_reconstruction << "\n";
   }

   // define u(x,y) to be represented
   const int k_x = 1;
   const int k_y = 2;
   std::function<real_t(const Vector &)> u_function =
      [=](const Vector& x)
   {
      return std::cos(2*M_PI*k_x * x(0)) * std::sin(2*M_PI*k_y * x(1));
   };
   FunctionCoefficient u_coefficient(u_function);

   // create simple 2D mesh
   const int num_x = 2;
   const int num_y = 2;
   Mesh serial_mesh = Mesh::MakeCartesian2D(num_x, num_y,
                                            Element::QUADRILATERAL);
   for (int i = 0; i < ser_ref_levels; ++i) { serial_mesh.UniformRefinement(); }
   ParMesh mesh(MPI_COMM_WORLD, serial_mesh);
   for (int i = 0; i < par_ref_levels; ++i) { mesh.UniformRefinement(); }

   // compute finite element representation of u(x,y)
   L2_FECollection fe_collection_original(order_original, mesh.Dimension());
   L2_FECollection fe_collection_averages(order_averages, mesh.Dimension());
   L2_FECollection fe_collection_reconstruction(order_reconstruction,
                                                mesh.Dimension());

   ParFiniteElementSpace fe_space_original(&mesh, &fe_collection_original);
   ParFiniteElementSpace fe_space_averages(&mesh, &fe_collection_averages);
   ParFiniteElementSpace fe_space_reconstruction(&mesh,
                                                 &fe_collection_reconstruction);

   ParGridFunction u_original(&fe_space_original);
   ParGridFunction u_averages(&fe_space_averages);
   ParGridFunction u_rec_avg(&fe_space_averages);
   ParGridFunction diff(&fe_space_averages);
   ParGridFunction u_reconstruction(&fe_space_reconstruction);

   // compute element averages
   u_original.ProjectCoefficient(u_coefficient);
   u_original.GetElementAverages(u_averages);

   // compute reconstruction
   reconstructField(u_averages, u_reconstruction);

   u_reconstruction.GetElementAverages(u_rec_avg);

   // evaluate reconstruction
   char vishost[] = "localhost";
   int visport = 19916;
   socketstream glvis_original(vishost, visport);
   socketstream glvis_averages(vishost, visport);
   socketstream glvis_rec_avg(vishost, visport);
   socketstream glvis_reconstruction(vishost, visport);
   if (glvis_original && glvis_averages && glvis_reconstruction)
   {
      //glvis_original.precision(8);
      glvis_original << "parallel " << mesh.GetNRanks()
                     << " " << mesh.GetMyRank() << "\n"
                     << "solution\n" << mesh << u_original
                     << "window_title 'original'\n" << std::flush;
      MPI_Barrier(mesh.GetComm());
      //glvis_averages.precision(8);
      glvis_averages << "parallel " << mesh.GetNRanks()
                     << " " << mesh.GetMyRank() << "\n"
                     << "solution\n" << mesh << u_averages
                     << "window_title 'averages'\n" << std::flush;
      MPI_Barrier(mesh.GetComm());
      //glvis_reconstruction.precision(8);
      glvis_reconstruction << "parallel " << mesh.GetNRanks()
                           << " " << mesh.GetMyRank() << "\n"
                           << "solution\n" << mesh << u_reconstruction
                           << "window_title 'reconstruction'\n" << std::flush;
      //glvis_reconstruction.precision(8);
      glvis_rec_avg << "parallel " << mesh.GetNRanks()
                    << " " << mesh.GetMyRank() << "\n"
                    << "solution\n" << mesh << u_rec_avg
                    << "window_title 'rec average'\n" << std::flush;
   }
   else
   {
      MFEM_WARNING("Cannot connect to glvis server, disabling visualization.")
   }

   real_t error = u_reconstruction.ComputeL2Error(u_coefficient);
   subtract(u_rec_avg, u_averages, diff);
   ConstantCoefficient zeros(0.0);
   real_t error_avg = diff.ComputeL2Error(zeros);

   Vector el_error(mesh.GetNE());
   ConstantCoefficient ones(1.0);
   ParGridFunction zero(&fe_space_reconstruction);
   zero = 0.0;
   zero.ComputeElementLpErrors(2.0, ones, el_error);
   real_t hmax = el_error.Max();

   if (Mpi::Root())
   {
      mfem::out << "\n|| u_h - u ||_{L^2} = " << error << "\n" << std::endl;
      mfem::out << "\n|| Avg(Rec(u_h)) - u_h ||_{L^2} = " << error_avg << "\n" <<
                std::endl;
   }

   if (save_to_file && Mpi::Root())
   {
      std::ofstream file;
      file.open("convergence.csv", std::ios::out | std::ios::app);
      if (!file.is_open())
      {
         mfem_error("Failed to open");
      }
      file << std::scientific << std::setprecision(16);
      file << error
           << "," << fe_space_averages.GetNConformingDofs()
           << "," << fe_space_reconstruction.GetNConformingDofs()
           << "," << hmax
           << "," << mesh.GetNE() << std::endl;
      file.close();
   }

   Mpi::Finalize();

   return 0;
}
