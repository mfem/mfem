#include "mfem.hpp"

using namespace mfem;

// Draft bilinear integrator if needed
class AsymmetricMassIntegrator : public MassIntegrator
{
   // TODO: Kernel implementation, requires kernel_dispatch.hpp
   // TODO: MFEM_THREAD_SAFE
   // Vector tr_shape, te_shape;

protected:
   const mfem::FiniteElementSpace *tr_fes, *te_fes;

public:

   AsymmetricMassIntegrator() {};

   void AsymmetricElementMatrix(const FiniteElement &trial_fe,
                                const FiniteElement &test_fe, ElementTransformation &trial_tr,
                                ElementTransformation &test_tr, DenseMatrix &el_mat);
};

void AsymmetricMassIntegrator::AsymmetricElementMatrix(const FiniteElement
                                                       &trial_fe,
                                                       const FiniteElement &test_fe,
                                                       ElementTransformation &trial_tr,
                                                       ElementTransformation &test_tr,
                                                       DenseMatrix &elmat)
{
   int tr_ndof = trial_fe.GetDof();
   int te_ndof = test_fe.GetDof();

   elmat.SetSize(te_ndof, tr_ndof);
   elmat = 0.0;

   Vector tr_shape(tr_ndof);
   Vector te_shape(te_ndof);

   /* Notes:
    * Let T_K : ref_K -> K from reference to physical
    * Let phi_i_K a shape function on the element K
    * and phi_i a shape function on the reference element ref_K
    * Let N(K) be an adjacent element to K
    *
    * Then phi_i_K(x) = phi_i(inv_T_K(x)), x in Omega.
    * Moreover,
    * int_N(K) phi_i_K(x) v(x) dx
    * = int_N(K) phi_i(inv_T_K(x)) v(x) dx
    * = int_ref_K phi_i(inv_T_K( T_N(K)(y) )) v( T_N(K)(y) ) |Jac T_N(K)| dy.
    */

   // GetRule is public!
   const IntegrationRule *ir = GetIntegrationRule(trial_fe, test_fe, trial_tr);

   DenseMatrix physical_pts, physical_pts_trial;
   Vector physical_ip;
   test_tr.Transform(*ir, physical_pts);
   // Vector physical_ip_trial;
   // trial_tr.Transform(*ir, physical_pts_trial);

   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);
      IntegrationPoint tip;

      physical_pts.GetColumn(i, physical_ip);
      // physical_pts_trial.GetColumn(i, physical_ip_trial);

      /* DEBUG
       * mfem::out << i << "th Physical pt from test_tr" << std::endl;
       * physical_ip.Print();
       * mfem::out << i << "th Physical pt from trial_tr" << std::endl;
       * physical_ip_trial.Print();
       */

      // Pullback without projections
      InverseElementTransformation inv_trial_tr(&trial_tr);
      inv_trial_tr.SetPhysicalRelTol(1e-16);
      inv_trial_tr.SetMaxIter(50);
      inv_trial_tr.SetSolverType(InverseElementTransformation::SolverType::Newton);
      inv_trial_tr.SetInitialGuessType(
         InverseElementTransformation::InitGuessType::ClosestPhysNode);
      // inv_trial_tr.SetPrintLevel(4); // desperate measures
      inv_trial_tr.Transform(physical_ip, tip);

      trial_tr.SetIntPoint(&tip);
      test_tr.SetIntPoint(&ip);

      // Onto the test element
      trial_fe.CalcPhysShape(trial_tr, tr_shape);
      test_fe.CalcPhysShape(test_tr, te_shape);

      te_shape *= test_tr.Weight() * ip.weight;
      AddMultVWt(te_shape, tr_shape, elmat);
   }
}

// Small least square solver
// TODO: it could be implemented as a mfem::Solver, as a method...
real_t _shift = 0.0;
int _print_level = -1;
int _max_iter = 100;
void LSSolver(const DenseMatrix& A, const Vector& b, Vector& x)
{
   x.SetSize(A.Width());

   Vector Atb(A.Width());
   A.MultTranspose(b, Atb);

   TransposeOperator At(&A);
   ProductOperator AtA(&At, &A, false, false);
   IdentityOperator I(AtA.Height());
   SumOperator AtA_reg(&AtA, 1.0, &I, _shift, false, false);

   CG(AtA_reg, Atb, x, _print_level, _max_iter);
}

int main(int argc, char* argv[])
{
   Mpi::Init(argc, argv);

   // Default command-line options
   int ser_ref_levels = 0;
   int par_ref_levels = 0;
   int order_original = 3;
   int order_reconstruction = 1;
   bool show_error = false;

   // Parse options
   OptionsParser args(argc, argv);
   args.AddOption(&ser_ref_levels, "-rs", "--refine",
                  "Number of serial refinement steps.");
   args.AddOption(&par_ref_levels, "-rp", "--refine",
                  "Number of parallel refinement steps.");
   args.AddOption(&order_original, "-O", "--order-original",
                  "Order original broken space.");
   args.AddOption(&order_reconstruction, "-R", "--order-reconstruction",
                  "Order of reconstruction broken space.");
   args.AddOption(&show_error, "-se", "--show-error", "-no-se",
                  "--no-show-error", "Show approximation error.");

   if (Mpi::Root())
   {
      args.ParseCheck();
      MFEM_VERIFY((ser_ref_levels >= 0) && (par_ref_levels >= 0), "")
      mfem::out << "Number of serial refinements:     " << ser_ref_levels << "\n";
      mfem::out << "Number of parallel refinements:   " << par_ref_levels << "\n";
      mfem::out << "Original order:                   " << order_original << "\n";
      mfem::out << "Original reconstruction:          " << order_reconstruction <<
                "\n";
   }

   // Mesh
   const int num_x = 2;
   const int num_y = 2;
   Mesh serial_mesh = Mesh::MakeCartesian2D(num_x, num_y,
                                            Element::QUADRILATERAL);
   for (int i = 0; i < ser_ref_levels; ++i) { serial_mesh.UniformRefinement(); }
   ParMesh mesh(MPI_COMM_WORLD, serial_mesh);
   for (int i = 0; i < par_ref_levels; ++i) { mesh.UniformRefinement(); }
   NCMesh nc_mesh(static_cast<Mesh*>(&mesh));

   // target function u(x,y)
   const int k_x = 1;
   const int k_y = 2;
   std::function<real_t(const Vector &)> u_function =
      [=](const Vector& x)
   {
      return std::cos(2*M_PI*k_x * x(0)) * std::sin(2*M_PI*k_y * x(1));
   };
   FunctionCoefficient u_coefficient(u_function);

   // Broken spaces
   L2_FECollection fec_original(order_original, mesh.Dimension());
   L2_FECollection fec_averages(0, mesh.Dimension());
   L2_FECollection fec_reconstruction(order_reconstruction, mesh.Dimension());

   ParFiniteElementSpace fes_original(&mesh, &fec_original);
   ParFiniteElementSpace fes_averages(&mesh, &fec_averages);
   ParFiniteElementSpace fes_reconstruction(&mesh, &fec_reconstruction);

   ParGridFunction u_original(&fes_original);
   ParGridFunction u_averages(&fes_averages);
   ParGridFunction u_rec_avg(&fes_averages);
   ParGridFunction u_reconstruction(&fes_reconstruction);

   u_original.ProjectCoefficient(u_coefficient);
   u_original.GetElementAverages(u_averages);

   // Declare mass integrator
   AsymmetricMassIntegrator mass;

   // Compute local volumes
   ParGridFunction zeros(&fes_averages);
   ConstantCoefficient ones(1.0);
   Vector volumes(mesh.GetNE());
   zeros.ComputeElementL1Errors(ones, volumes);

   Array<int> ngh_e;
   for (int e_idx = 0; e_idx < mesh.GetNE(); e_idx++ )
   {
      nc_mesh.FindNeighbors(e_idx, ngh_e);
      ngh_e.Append(e_idx);
      int num_ngh_e = ngh_e.Size();

      /* DEBUG
       * mfem::out << "DEBUG:\n\tOn " << e_idx << " elem\n" << std::endl;
       * ngh_e.Print();
       */

      // Define small matrix
      int fe_rec_e_ndof = fes_reconstruction.GetFE(e_idx)->GetDof();
      DenseMatrix local_mass_mat(num_ngh_e, fe_rec_e_ndof);

      for (int i = 0; i < num_ngh_e; i++)
      {
         int ngh_e_idx = ngh_e[i];
         /* DEBUG
          * mfem::out << "DEBUG:\n\t\tOn " << i << "th neigh elem ("<< ngh_e_idx << ")\n" << std::endl;
          */

         IsoparametricTransformation trial_tr, test_tr; // TODO: pointers
         fes_reconstruction.GetElementTransformation(ngh_e_idx, &trial_tr);
         fes_averages.GetElementTransformation(e_idx, &test_tr);

         DenseMatrix ngh_elem_mat;
         mass.AsymmetricElementMatrix(*fes_reconstruction.GetFE(ngh_e_idx),
                                      *fes_averages.GetFE(e_idx),
                                      trial_tr, test_tr, ngh_elem_mat);

         // TODO: Extend GetRow API, allow composition
         // TODO: This works as fes_avg is lowest order!
         Vector ngh_vec;
         ngh_elem_mat.GetRow(0, ngh_vec);
         local_mass_mat.SetRow(i, ngh_vec);

         /* DEBUG
          * mfem::out << " Sizes: " << ngh_elem_mat.Height() << " x " <<
          *           ngh_elem_mat.Width() << std::endl;
          * ngh_elem_mat.Print();
          */
      }

      // Get local volumes and scale
      Vector local_volumes(num_ngh_e);
      volumes.GetSubVector(ngh_e, local_volumes);
      local_mass_mat.InvLeftScaling(local_volumes);

      /* DEBUG
       * mfem::out << e_idx << "th Element local LS mat" << std::endl;
       * least_sqrs_e.Print();
       */
      Vector local_u_avg;
      u_averages.GetSubVector(ngh_e, local_u_avg);

      Vector local_u_rec;
      LSSolver(local_mass_mat, local_u_avg, local_u_rec);

      /* DEBUG
       * mfem::out << "I solved locally!" << std::endl;
       * local_u_rec.Print();
       */

      Array<int> local_dofs;
      fes_reconstruction.GetElementDofs(e_idx, local_dofs);
      u_reconstruction.SetSubVector(local_dofs,local_u_rec);

      // ENDING LOOP
      ngh_e.DeleteAll();
      local_dofs.DeleteAll();
   }
   u_reconstruction.GetElementAverages(u_rec_avg);

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
      MPI_Barrier(mesh.GetComm());
      //glvis_reconstruction.precision(8);
      glvis_rec_avg << "parallel " << mesh.GetNRanks()
                    << " " << mesh.GetMyRank() << "\n"
                    << "solution\n" << mesh << u_rec_avg
                    << "window_title 'rec average'\n" << std::flush;
   }

   // Error studies
   Vector diff(u_averages.Size());
   real_t error = u_reconstruction.ComputeL2Error(u_coefficient);
   subtract(u_rec_avg, u_averages, diff);
   real_t error_avg = diff.Norml2();

   // TODO: Workaround local meshsize
   // Vector el_error(mesh.GetNE());
   // ConstantCoefficient ones(1.0);
   // ParGridFunction zero(&fe_space_reconstruction);
   // zero = 0.0;
   // zero.ComputeElementLpErrors(2.0, ones, el_error);
   // real_t hmax = el_error.Max();

   if (show_error && Mpi::Root())
   {
      mfem::out << "\n|| Rec(u_h) - u ||_{L^2} = " << error << "\n" << std::endl;
      mfem::out << "\n|| Avg(Rec(u_h)) - u_h ||_{ell^2} = " << error_avg << "\n" <<
                std::endl;
   }

   mfem::out << "Survived!" << std::endl;

   Mpi::Finalize();
   return 0;
}
