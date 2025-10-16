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
#include <string>
#include <map>

using namespace mfem;

void reconstructField(const ParGridFunction& src, ParGridFunction& dst);

void L2Reconstruction(const ParGridFunction& src, ParGridFunction& dst);

int main(int argc, char* argv[])
{
   Mpi::Init(argc, argv);

   // Default command-line options
   int refinement_levels = 0;

   int order_representation = 3;
   int order_reconstruction = 1;

   int field_profile = 0;
   real_t field_kx = 2.0;
   real_t field_ky = 4.0;

   bool visualization = true;
   int visport = 19916;

   // example field profiles
   using profile_t = std::function<real_t(const Vector&,const Vector&)>;
   std::vector<std::pair<std::string, profile_t>> field_profiles;
   // plane profile
   field_profiles.push_back(std::make_pair(
      "1 + kx x + ky y",
      [](const Vector &x, const Vector &k)
      {
         return 1.0 + x*k;
      }));
   // sinusoidal profile
   field_profiles.push_back(std::make_pair(
      "sin(2pi kx x) sin(2pi ky y)",
      [](const Vector &x, const Vector &k)
      {
         real_t result = 1.0;
         for(int i=0; i < x.Size(); i++) result *= std::sin(2.0*M_PI*k(i)*x(i));
         return result;
      }));
   // exponential-sinusoidal profile
   field_profiles.push_back(std::make_pair(
      "exp(r) cos(kx x) sin(ky y)",
      [](const Vector &x, const Vector &k)
      {
         real_t result = 1.0;
         for(int i=0; i < x.Size(); i++)
            result *= std::exp(x.Norml2()) * std::sin(2.0*M_PI*k(i)*x(i));
         return result;
      }));
   // create CLI help string for profiles
   std::string field_profiles_help = "Profile of field to be reconstructed:";
   for (int i=0; i < field_profiles.size(); i++)
      field_profiles_help += "\n\t" + std::to_string(i) + ": " + field_profiles[i].first;

   // Parse options
   OptionsParser args(argc, argv);
   args.AddOption(&refinement_levels, "-r", "--refine",
                  "Number of times to refine the mesh uniformly.");
   args.AddOption(&order_representation, "-orep", "--order-representation",
                  "Order of the finite element space to represent the field.");
   args.AddOption(&order_reconstruction, "-orec", "--order-reconstruction",
                  "Order of the finite element space to reconstruct the field.");
   args.AddOption(&field_profile, "-f", "--field-profile", field_profiles_help.c_str());
   args.AddOption(&field_kx, "-fx", "--field-kx",
                  "Value of kx in field profile");
   args.AddOption(&field_ky, "-fy", "--field-ky",
                  "Value of ky in field profile.");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization", "Enable or disable GLVis visualization.");
   args.AddOption(&visport, "-visp", "--visualization-port",
                  "Use custom port number for GLVis.");
   args.ParseCheck();

   // define u(x,y) to be represented
   profile_t u_function = field_profiles[field_profile].second;
   const Vector k({field_kx, field_ky});
   std::function<real_t(const Vector&)> u_function_wrapper =
      [&](const Vector &x) { return u_function(x, k); };
   FunctionCoefficient u_coefficient(u_function_wrapper);

   // create simple 2D mesh
   const int num_x = 2;
   const int num_y = 2;
   Mesh serial_mesh = Mesh::MakeCartesian2D(num_x, num_y, Element::QUADRILATERAL);
   for (int i = 0; i < refinement_levels; i++)
   { serial_mesh.UniformRefinement(); }
   serial_mesh.EnsureNCMesh();
   ParMesh mesh(MPI_COMM_WORLD, serial_mesh);
   serial_mesh.Clear();

   // create FEM things
   L2_FECollection fe_collection_averages(0, mesh.Dimension());
   ParFiniteElementSpace fe_space_averages(&mesh, &fe_collection_averages);
   ParGridFunction u_averages(&fe_space_averages);

   L2_FECollection fe_collection_representation(order_representation, mesh.Dimension());
   ParFiniteElementSpace fe_space_representation(&mesh, &fe_collection_representation);
   ParGridFunction u(&fe_space_representation);

   L2_FECollection fe_collection_reconstruction(order_reconstruction, mesh.Dimension());
   ParFiniteElementSpace fe_space_reconstruction(&mesh, &fe_collection_reconstruction);
   ParGridFunction u_reconstruction(&fe_space_reconstruction);
   ParGridFunction u_reconstruction_averages(&fe_space_averages);

   // compute element averages
   u.ProjectCoefficient(u_coefficient);
   u.GetElementAverages(u_averages);

   // compute reconstruction
   L2Reconstruction(u_averages, u_reconstruction);

   // evaluate reconstruction
   u_reconstruction.GetElementAverages(u_reconstruction_averages);
   char vishost[] = "localhost";
   socketstream glvis_original(vishost, visport);
   socketstream glvis_averages(vishost, visport);
   socketstream glvis_reconstruction_averages(vishost, visport);
   socketstream glvis_reconstruction(vishost, visport);
   if (glvis_original &&
       glvis_averages &&
       glvis_reconstruction_averages &&
       glvis_reconstruction &&
       visualization)
   {
      glvis_original.precision(8);
      glvis_original << "solution\n" << mesh << u
         << "window_title 'original'\n" << std::flush;
      glvis_averages.precision(8);
      glvis_averages << "solution\n" << mesh << u_averages
         << "window_title 'averages'\n" << std::flush;
      glvis_reconstruction.precision(8);
      glvis_reconstruction << "solution\n" << mesh << u_reconstruction
         << "window_title 'reconstruction'\n" << std::flush;
      glvis_reconstruction.precision(8);
      glvis_reconstruction_averages << "solution\n" << mesh << u_reconstruction_averages
         << "window_title 'rec average'\n" << std::flush;
   }

   real_t error = u_reconstruction.ComputeL2Error(u_coefficient);
   //subtract(u_reconstruction_averages, u_averages, diff);
   ConstantCoefficient zeros(0.0);
   real_t error_avg = 0.0;//diff.ComputeL2Error(zeros);

   Vector el_error(mesh.GetNE());
   ConstantCoefficient ones(1.0);
   GridFunction zero(&fe_space_reconstruction);
   zero = 0.0;
   zero.ComputeElementLpErrors(2.0, ones, el_error);
   real_t hmax = el_error.Max();

   if (Mpi::Root())
   {
      mfem::out << "\n|| Avg(Rec(u_h)) - u_h ||_{L^2} = " << error_avg
                << std::endl;
      mfem::out << "|| u_h - u ||_{L^2} = \n" << error << std::endl;
   }

   Mpi::Finalize();

   return 0;
}


/// @brief Saturates a neighborhood with hopes to well-pose the least squares problem.
/// @todo Make friends with @a MCMesh
void SaturateNeighborhood(NCMesh& mesh, const int element_idx,
                          const int target_ndofs, const int contributed_ndofs,
                          Array<int>& neighbors)
{
   Array<int> temp;
   mesh.FindNeighbors(element_idx, neighbors);
   neighbors.Append(element_idx);
   MFEM_VERIFY(mesh.GetNumElements() * contributed_ndofs >= target_ndofs,
               "Mesh too small!");
   while (neighbors.Size() * contributed_ndofs < target_ndofs)
   {
      mesh.NeighborExpand(neighbors, temp);
      neighbors = temp;
   }
   neighbors.Unique();
}

void L2Reconstruction(const ParGridFunction& src, ParGridFunction& dst)
{
   const real_t RTOL = 1.0e-5;

   Mesh *mesh = dst.FESpace()->GetMesh();
   FiniteElementSpace *fes = dst.FESpace();
   NCMesh *ncmesh = fes->GetMesh()->ncmesh;

   for (int element_idx = 0; element_idx < fes->GetNE(); element_idx++)
   {
      const FiniteElement* element = fes->GetFE(element_idx);
      Array<int> element_dofs;
      fes->GetElementDofs(element_idx, element_dofs);
      const int N = element_dofs.Size();
      IsoparametricTransformation trans;
      fes->GetElementTransformation(element_idx, &trans);

      DenseMatrix A(N+1,N+1);
      A = 0.0;
      Vector b(N+1);
      b = 0.0;
      IntegrationPoint int_point_average;
      int_point_average.Set2(0.5, 0.5); // TODO: generalize to 3D
      const real_t element_average = src.GetValue(element_idx, int_point_average);
      b(N) = element_average;

      IsoparametricTransformation neighbor_trans;
      Array<int> neighbor_element_idxs;
      ncmesh->FindNeighbors(element_idx, neighbor_element_idxs);
      for (int i=0; i < element->GetOrder(); i++)
      {
         Array<int> temp;;
         ncmesh->NeighborExpand(neighbor_element_idxs, temp);
         neighbor_element_idxs = temp;
      }
      // DEBUG
      std::vector<Vector> alphas;
      std::vector<Vector> centers;
      for (const int neighbor_element_idx : neighbor_element_idxs)
      {
         const FiniteElement* neighbor_element = fes->GetFE(neighbor_element_idx);
         fes->GetElementTransformation(neighbor_element_idx, &neighbor_trans);
         const real_t neighbor_average = src.GetValue(neighbor_element_idx, int_point_average);

         // compute average of u over neighbor element (denoted alpha)
         Vector alpha(N);
         real_t volume = 0.0;
         alpha = 0.0;
         const int int_order = neighbor_element->GetOrder() + neighbor_trans.OrderW();
         const IntegrationRule& int_rule =
            IntRules.Get(neighbor_element->GetGeomType(), int_order);
         for (int k=0; k < int_rule.GetNPoints(); k++)
         {
            IntegrationPoint int_point = int_rule.IntPoint(k);
            if (element_idx != neighbor_element_idx)
            {
               Vector physical_point;
               neighbor_trans.Transform(int_point, physical_point);
               InverseElementTransformation inv_trans(&trans);
               inv_trans.SetSolverType(inv_trans.Newton);
               inv_trans.SetInitialGuessType(inv_trans.Center);
               inv_trans.SetPhysicalRelTol(1e-12);
               IntegrationPoint inv_int_point;
               const int result = inv_trans.Transform(physical_point, inv_int_point);
               MFEM_VERIFY(result != inv_trans.Unknown, "InverseTransform failed.");
               int_point.Set2(inv_int_point.x, inv_int_point.y); // TODO: generalize to 3D
            }
            Vector shape(N);
            element->CalcShape(int_point, shape);
            trans.SetIntPoint(&int_point);
            const real_t detJ = trans.Weight();
            alpha.Add(int_point.weight * detJ, shape);
            volume += int_point.weight * detJ;
         }
         alpha *= 1.0/volume;
         // DEBUG
         Vector tmp(2);
         neighbor_trans.Transform(int_point_average, tmp);
         centers.push_back(tmp);
         alphas.push_back(alpha);

         // update Q block (of A) with Q_neighbor = alpha \otimes alpha
         DenseMatrix Q_neighbor(N,N);
         MultVVt(alpha, Q_neighbor);
         A.AddSubMatrix(0, Q_neighbor);
         // update c block (of b) with c_neighbor = neighbor_average * alpha
         Vector c_neighbor = alpha;
         c_neighbor *= neighbor_average;
         b.AddSubVector(c_neighbor, 0);
         // for the original element, set e blocks (of A) with e = alpha
         if (element_idx == neighbor_element_idx)
         {
            for (int k=0; k < N; k++)
            {
               A(k,N) = -alpha(k);
               A(N,k) = alpha(k);
            }
         }
      }

      Vector y(N+1);
      DenseMatrixInverse A_inverse(A);
      A_inverse.Factor();
      A_inverse.Mult(b, y);
      // DEBUG
      if (std::abs(y(N)) > 1e-8 && false)
      {
         mfem::out << "\nnumber of neighbors: " << neighbor_element_idxs.Size() << std::endl;
         mfem::out << "y: ";
         y.Print();
         DenseMatrix Q(N,N);
         A.GetSubMatrix(0, N, Q);
         const Vector x(y, 0, N);
         const Vector c(b, 0, N);
         Vector e(N);
         for (int i=0; i < N; i++)
         {
            e(i) = A(N,i);
         }
         Vector Qx(N);
         Q.Mult(x,Qx);
         mfem::out << "f(x) = " << Qx*x*0.5 - c*x << "\t eTx = " << e*x << std::endl;
         Vector tmp(2);
         trans.Transform(int_point_average, tmp);
         mfem::out << "Current element: ";
         tmp.Print();
         for (int i=0; i < alphas.size(); i++)
         {
            mfem::out << "Neighbor element: ";
            centers[i].Print();
            alphas[i].Print();
            mfem::out << "alpha*x: " << alphas[i]*x << std::endl;
         }
         Vector x2(N);
         x2 = element_average;
         Q.Mult(x2, Qx);
         mfem::out << "f(1) = " << Qx*x2*0.5 - c*x2 << "\t eTx = " << e*x2 << std::endl;
         mfem::out << "A: ";
         mfem::out << "average: " << element_average << std::endl;
      }
      const Vector x(y, 0, N);
      dst.SetSubVector(element_dofs, x);
   }
}


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

