#include "mfem.hpp"

using namespace mfem;

/** Class for an asymmetric (out-of-element) mass integrator $a_K(u,v) := (E(u),v)_K$,
 *  where $E$ is an extension of $u$ (with original domain $\hat{K}$) to $K$.
 *
 *  @note Currently we don't need to derive from MassIntegrator, as GetRule is public.
 *  This could be it's own standalone function.
 *  @dev Kernel implementation: requires kernel_dispatch.hpp. MFEM_THREAD_SAFE...
 */
class AsymmetricMassIntegrator : public MassIntegrator
{
private:
   Vector ngh_shape, self_shape;
   Vector physical_ngh_ip;
   DenseMatrix physical_ngh_pts;
   IntegrationPoint ngh_ip;
   int ngh_ndof, self_ndof;

protected:
   const mfem::FiniteElementSpace *tr_fes, *te_fes;

public:

   AsymmetricMassIntegrator() {};

   /// Assembles element mass matrix, extending @a ngh_fe to @a self_fe.
   void AsymmetricElementMatrix(const FiniteElement &ngh_fe,
                                const FiniteElement &self_fe,
                                ElementTransformation &ngh_tr,
                                ElementTransformation &self_tr,
                                DenseMatrix &el_mat);
};

void AsymmetricMassIntegrator::AsymmetricElementMatrix(const FiniteElement
                                                       &ngh_fe,
                                                       const FiniteElement &self_fe,
                                                       ElementTransformation &ngh_tr,
                                                       ElementTransformation &self_tr,
                                                       DenseMatrix &elmat)
{
   ngh_ndof = ngh_fe.GetDof();
   self_ndof = self_fe.GetDof();

   self_shape.SetSize(self_ndof);
   ngh_shape.SetSize(ngh_ndof);
   elmat.SetSize(self_ndof, ngh_ndof);
   elmat = 0.0;

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

   const int int_order = self_fe.GetOrder() + ngh_fe.GetOrder() + self_tr.OrderW();
   const IntegrationRule &ir = IntRules.Get(self_fe.GetGeomType(), int_order);

   ngh_tr.Transform(ir, physical_ngh_pts);

   for (int i = 0; i < ir.GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir.IntPoint(i);
      physical_ngh_pts.GetColumn(i, physical_ngh_ip);

      // Pullback physical neighboring integration points to
      // "extended" reference element under self_tr
      InverseElementTransformation inv_tr(&self_tr);
      inv_tr.SetPhysicalRelTol(1e-16);
      inv_tr.SetMaxIter(50);
      inv_tr.SetSolverType(InverseElementTransformation::SolverType::Newton);
      inv_tr.SetInitialGuessType(
         InverseElementTransformation::InitGuessType::ClosestPhysNode);
      inv_tr.Transform(physical_ngh_ip, ngh_ip);

      ngh_tr.SetIntPoint(&ngh_ip);
      self_tr.SetIntPoint(&ip);

      // Compute shape functions on self_fe
      ngh_fe.CalcPhysShape(ngh_tr, ngh_shape);
      self_fe.CalcPhysShape(self_tr, self_shape);

      self_shape *= self_tr.Weight() * ip.weight;
      AddMultVWt(self_shape, ngh_shape, elmat);
   }
}

// Small least square solver
// TODO: it could be implemented as a mfem::Solver, as a method...
int _print_level = -1;
int _max_iter = 100;

/// @brief Dense small least squares solver
void LSSolver(const DenseMatrix& A, const Vector& b, Vector& x,
              real_t shift = 0.0)
{
   x.SetSize(A.Width());

   Vector Atb(A.Width());
   A.MultTranspose(b, Atb);

   TransposeOperator At(&A);
   ProductOperator AtA(&At, &A, false, false);
   IdentityOperator I(AtA.Height());
   SumOperator AtA_reg(&AtA, 1.0, &I, shift, false, false);

   MINRES(AtA_reg, Atb, x, _print_level, _max_iter);
}

/// @brief Dense small least squares solver, with constrains @a C with value @a c
void LSSolver(const DenseMatrix& A, const DenseMatrix& C,
              const Vector& b, const Vector& c,
              Vector& x, Vector& y,
              real_t shift = 0.0)
{
   TransposeOperator At(&A);
   ProductOperator AtA(&At, &A, false, false);
   IdentityOperator I(AtA.Height());
   SumOperator AtA_reg(&AtA, 1.0, &I, shift, false, false);

   TransposeOperator Ct(&C);

   // Block matrix
   Array<int> offsets(3);
   offsets[0] = 0;
   offsets[1] = AtA_reg.Width();
   offsets[2] = AtA_reg.Width() + Ct.Width();

   BlockOperator block_mat(offsets);
   block_mat.SetBlock(0, 0, &AtA_reg);
   block_mat.SetBlock(0, 1, &Ct);
   block_mat.SetBlock(1, 0, const_cast<DenseMatrix*>(&C));

   // Block vectors
   BlockVector rhs(offsets), z(offsets);
   rhs = 0.0;
   z = 0.0;

   Vector Atb(A.Width());
   A.MultTranspose(b, Atb);

   rhs.SetVector(Atb, offsets[0]);
   rhs.SetVector(c, offsets[1]);

   MINRES(block_mat, rhs, z, _print_level, _max_iter);

   x.SetSize(A.Width());
   y.SetSize(C.Width());

   z.GetBlockView(0,x);
   z.GetBlockView(1,y);
}

int main(int argc, char* argv[])
{
   Mpi::Init(argc, argv);

   // Default command-line options
   int ser_ref_levels = 0;
   int par_ref_levels = 0;
   int order_original = 3;
   int order_reconstruction = 1;
   bool preserve_volumes = true;

   bool show_error = true;
   bool save_to_file = false;

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
   args.AddOption(&preserve_volumes, "-V", "--preserve-volumes", "-no-V",
                  "--no-preserve-volumes", "Preserve averages (volumes) by"
                  " solving a constrained least squares problem");
   args.AddOption(&show_error, "-se", "--show-error", "-no-se",
                  "--no-show-error", "Show approximation error.");
   args.AddOption(&save_to_file, "-s", "--save", "-no-s",
                  "--no-save", "Show or not show approximation error.");

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
   auto ngh_tr = new IsoparametricTransformation;
   auto self_tr = new IsoparametricTransformation;
   for (int e_idx = 0; e_idx < mesh.GetNE(); e_idx++ )
   {
      nc_mesh.FindNeighbors(e_idx, ngh_e);
      ngh_e.Append(e_idx);
      const int num_ngh_e = ngh_e.Size();

      // Define small matrix
      const int fe_rec_e_ndof = fes_reconstruction.GetFE(e_idx)->GetDof();
      DenseMatrix local_mass_mat(num_ngh_e, fe_rec_e_ndof);

      for (int i = 0; i < num_ngh_e; i++)
      {
         int ngh_e_idx = ngh_e[i];

         fes_reconstruction.GetElementTransformation(ngh_e_idx, ngh_tr);
         fes_averages.GetElementTransformation(e_idx, self_tr);

         DenseMatrix ngh_elem_mat;
         mass.AsymmetricElementMatrix(*fes_reconstruction.GetFE(ngh_e_idx),
                                      *fes_averages.GetFE(e_idx),
                                      *ngh_tr, *self_tr, ngh_elem_mat);

         if (ngh_elem_mat.Height()!=1) { mfem_error("High order case not implemented yet!"); }
         Vector ngh_vec;
         ngh_elem_mat.GetRow(0, ngh_vec);
         local_mass_mat.SetRow(i, ngh_vec);
      }

      // Get local volumes and scale patch matrix
      Vector local_volumes(num_ngh_e);
      volumes.GetSubVector(ngh_e, local_volumes);
      local_mass_mat.InvLeftScaling(local_volumes);

      // Solve
      Vector local_u_avg, local_u_rec;
      u_averages.GetSubVector(ngh_e, local_u_avg);
      if (preserve_volumes)
      {
         Vector _mult, exact_average_e(1);
         exact_average_e = u_averages(e_idx);

         DenseMatrix avg_mat(1, local_mass_mat.Width());
         Vector avg_self(local_mass_mat.Height());
         local_mass_mat.GetRow(local_mass_mat.Height()-1, avg_self);
         avg_mat.SetRow(0, avg_self);

         LSSolver(local_mass_mat, avg_mat,
                  local_u_avg, exact_average_e,
                  local_u_rec, _mult);
      }
      else
      {
         LSSolver(local_mass_mat, local_u_avg, local_u_rec);
      }

      // Integrate into global solution
      Array<int> local_dofs;
      fes_reconstruction.GetElementDofs(e_idx, local_dofs);
      u_reconstruction.SetSubVector(local_dofs,local_u_rec);

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

   if (show_error && Mpi::Root())
   {
      mfem::out << "\n|| Rec(u_h) - u ||_{L^2} = " << error << "\n" << std::endl;
      mfem::out << "\n|| Avg(Rec(u_h)) - u_h ||_{ell^2} = " << error_avg << "\n" <<
                std::endl;
   }

   if (save_to_file && Mpi::Root())
   {
      Vector el_error(mesh.GetNE());
      ConstantCoefficient ones(1.0);
      ParGridFunction zero(&fes_reconstruction);
      zero = 0.0;
      zero.ComputeElementLpErrors(2.0, ones, el_error);
      real_t hmax = el_error.Max();

      std::ofstream file;
      file.open("convergence.csv", std::ios::out | std::ios::app);
      if (!file.is_open())
      {
         mfem_error("Failed to open");
      }
      file << std::scientific << std::setprecision(16);
      file << error
           << "," << fes_averages.GetNConformingDofs()
           << "," << fes_reconstruction.GetNConformingDofs()
           << "," << hmax
           << "," << mesh.GetNE() << std::endl;
      file.close();
   }

   delete ngh_tr;
   delete self_tr;

   Mpi::Finalize();
   return 0;
}
