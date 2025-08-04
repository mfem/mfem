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

namespace mfem::reconstruction
{

/// Enumerator for solver types
enum SolverType
{
   direct,
   cg,
   bicgstab,
   minres,
   num_solvers  // last
};

/// @brief Get face averages at quadrature points, associated to
/// the face associated to @a face_trans
void ComputeFaceAverage(const FiniteElementSpace& fes,
                        FaceElementTransformations& face_trans,
                        const GridFunction& global_function,
                        Vector& face_values)
{
   const bool has_other = (face_trans.Elem2No >= 0);
   const FiniteElement* fe_self = fes.GetFE(face_trans.Elem1No);
   const FiniteElement* fe_other = has_other?fes.GetFE(face_trans.Elem2No):
                                   fe_self;

   const int ndof_self = fe_self->GetDof();
   const int ndof_other = has_other?fe_other->GetDof():0;

   const int order = face_trans.OrderW() + fe_self->GetOrder() +
                     fe_other->GetOrder();
   const IntegrationRule& ir = IntRules.Get(face_trans.GetGeometryType(), order);

   Array<int> dofs_self, dofs_other;
   fes.GetElementDofs(face_trans.Elem1No, dofs_self);
   if (has_other) { fes.GetElementDofs(face_trans.Elem2No, dofs_other); }

   Vector shape_self(ndof_self), shape_other(ndof_other);
   Vector dofs_at_self, dofs_at_other;
   global_function.GetSubVector(dofs_self, dofs_at_self);
   global_function.GetSubVector(dofs_other, dofs_at_other);

   face_values.SetSize(ir.GetNPoints());
   face_values = 0.0;

   for (int p = 0; p < ir.GetNPoints(); p++)
   {
      const IntegrationPoint& ip = ir.IntPoint(p);
      face_trans.SetAllIntPoints(&ip);

      fe_self->CalcPhysShape(*face_trans.Elem1, shape_self);
      if (has_other) { fe_other->CalcPhysShape(*face_trans.Elem2, shape_other); }

      face_values(p) = shape_self*dofs_at_self;
      if (has_other) { face_values(p) += shape_other*dofs_at_other; }
   }
   if (has_other) { face_values *= 0.5; }
}

/* void ComputeFaceAverages(const FiniteElement& fe_self,
 *                          const FiniteElement& fe_other,
 *                          FaceElementTransformations& Trans)
 * {
 *    int ndofs_self, ndofs_other;
 *    ndofs_self = fe_self.GetDof();
 *    ndofs_other = 0;
 *    if (Trans.Elem2No >= 0) { ndofs_other = fe_other.GetDof(); }
 *
 *    Vector shape_self(ndofs_self), shape_other(ndofs_other);
 *
 *    const IntegrationRule* ir = &IntRules.Get(Trans.GetGeometryType(),
 *                                              Trans.Elem1->OrderW() + fe_self.GetOrder() + fe_other.GetOrder());
 *
 *    for (int idx = 0; idx < ir->GetNPoints(); idx++)
 *    {
 *       const IntegrationPoint& ip = ir->IntPoint(idx);
 *       real_t w = ip.weight;
 *       Trans.SetAllIntPoints(&ip);
 *
 *       const IntegrationPoint &ip_self = Trans.GetElement1IntPoint();
 *       const IntegrationPoint &ip_other = Trans.GetElement2IntPoint();
 *
 *       fe_self.CalcPhysShape(*Trans.Elem1, shape_self);
 *       if (ndofs_other) { fe_other.CalcPhysShape(*Trans.Elem2, shape_other); }
 *
 *       // TODO DEBUG
 *       mfem::out << "Shape functions on el1 (in):  " << std::endl;
 *       shape_self.Print(mfem::out,1);
 *       if (Trans.Elem2No >= 0) { mfem::out << "Single!!!!" << std::endl; }
 *       mfem::out << "Shape functions on el2 (out): " << std::endl;
 *       shape_other.Print(mfem::out,1);
 *    }
 * }
 */

} // end namespace mfem::reconstruction

using namespace mfem;
using namespace reconstruction;

int main(int argc, char* argv[])
{
   Mpi::Init(argc, argv);

   // Default command-line options
   int ser_ref_levels = 0;
   int par_ref_levels = 0;

   int order_smooth = 3;
   int order_src = 0;
   int order_dst = 1;

   SolverType solver_type = direct;
   real_t solver_reg = 0.0;
   real_t solver_rtol = 1.0e-30;
   real_t solver_atol = 0.0;
   int solver_maxiter = 1000;
   // TODO(Gabriel): Not implemented yet
   // int solver_plevel = 3;

   bool save_to_file = false;

   bool visualization = true;
   int visport = 19916;

   // Parse options
   OptionsParser args(argc, argv);
   args.AddOption(&ser_ref_levels, "-rs", "--refine",
                  "Number of serial refinement steps.");
   args.AddOption(&par_ref_levels, "-rp", "--refine",
                  "Number of parallel refinement steps.");

   args.AddOption(&order_smooth, "-O", "--order-original",
                  "Order original broken space.");
   // TODO(Gabriel): Not implemented yet
   // args.AddOption(&order_src, "-A", "--order-averages",
   //                "Order averaged broken space.");
   args.AddOption(&order_dst, "-R", "--order-reconstruction",
                  "Order of reconstruction broken space.");

   args.AddOption((int*)&solver_type, "-S", "--solver",
                  "Solvers to be considered:"
                  "\n\t0: Direct Solver"
                  "\n\t1: CG - Conjugate Gradient"
                  "\n\t2: BiCGSTAB -Biconjugate gradient stabilized"
                  "\n\t3: MINRES - Minimal residual");
   args.AddOption(&solver_reg, "-Sreg", "--solver-reg",
                  "Add regularization term to the least squares problem");
   args.AddOption(&solver_rtol, "-Srtol", "--solver-rtol",
                  "Relative tolerance for the iterative solver");
   args.AddOption(&solver_atol, "-Satol", "--solver-atol",
                  "Absolute tolerance for the iterative solver");
   args.AddOption(&solver_maxiter, "-Smi", "--solver-maxiter",
                  "Maximum number of iterations for the solver");
   // TODO(Gabriel): Not implemented yet
   // args.AddOption(&solver_plevel, "-Sp", "--solver-print",
   //                "Print level for the iterative solver");

   args.AddOption(&save_to_file, "-s", "--save", "-no-s",
                  "--no-save", "Show or not show approximation error.");

   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization", "Enable or disable GLVis visualization.");
   args.AddOption(&visport, "-p", "--send-port", "Socket for GLVis.");

   args.ParseCheck();
   MFEM_VERIFY((ser_ref_levels >= 0) && (par_ref_levels >= 0), "")
   MFEM_VERIFY((solver_reg>=0.0), "")
   MFEM_VERIFY(order_smooth > order_src, "")
   MFEM_VERIFY((0 <= solver_type) && (solver_type < num_solvers),
               "invalid solver type: " << solver_type);

   if (Mpi::Root())
   {
      mfem::out << "Number of serial refs.:  " << ser_ref_levels << "\n";
      mfem::out << "Number of parallel refs: " << par_ref_levels << "\n";
      mfem::out << "Original order:          " << order_smooth << "\n";
      mfem::out << "Original averages:       " << order_src << "\n";
      mfem::out << "Original reconstruction: " << order_dst << "\n";
   }

   // Mesh
   const int num_x = 2;
   const int num_y = 2;
   Mesh serial_mesh = Mesh::MakeCartesian2D(num_x, num_y, Element::QUADRILATERAL);
   for (int i = 0; i < ser_ref_levels; ++i) { serial_mesh.UniformRefinement(); }
   ParMesh mesh(MPI_COMM_WORLD, serial_mesh);
   for (int i = 0; i < par_ref_levels; ++i) { mesh.UniformRefinement(); }

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
   L2_FECollection fec_smooth(order_smooth, mesh.Dimension());
   L2_FECollection fec_src(order_src, mesh.Dimension());
   L2_FECollection fec_dst(order_dst, mesh.Dimension());

   ParFiniteElementSpace fes_smooth(&mesh, &fec_smooth);
   ParFiniteElementSpace fes_src(&mesh, &fec_src);
   ParFiniteElementSpace fes_dst(&mesh, &fec_dst);

   ParGridFunction u_smooth(&fes_smooth);
   ParGridFunction u_src(&fes_src);
   ParGridFunction u_rec_avg(&fes_src);
   ParGridFunction diff(&fes_src);
   ParGridFunction u_dst(&fes_dst);

   u_smooth.ProjectCoefficient(u_coefficient);
   u_smooth.GetElementAverages(u_src);

   FaceElementTransformations* face_trans = nullptr;
   Array<int> faces_e, orientation_e, face_dofs;
   Vector avg_at_face, u_avg_at_face_dofs;

   for (int e_idx = 0; e_idx < mesh.GetNE(); e_idx++)
   {
      // Local average? Local projection?


      mesh.GetElementEdges(e_idx, faces_e, orientation_e);
      for (int i = 0; i < faces_e.Size(); i++)
      {
         const int f_idx = faces_e[i];
         face_trans = mesh.GetFaceElementTransformations(f_idx);
         fes_src.GetFaceDofs(f_idx, face_dofs);

         u_src.GetSubVector(face_dofs, u_avg_at_face_dofs);
         ComputeFaceAverage(fes_src, *face_trans,
                            u_src, avg_at_face);

         /* DEBUG:
          * mfem::out << "Face " << f_idx << std::endl;
          * avg_at_face.Print(mfem::out,1);
          * mfem::out << "Values of u (avg) at face_dofs" << std::endl;
          * u_avg_at_face_dofs.Print(mfem::out,1);
          * mfem::out << " ---  " << std::endl; */
      }
   }

   char vishost[] = "localhost";
   socketstream glvis_smooth(vishost, visport);
   socketstream glvis_src(vishost, visport);
   socketstream glvis_rec_avg(vishost, visport);
   socketstream glvis_dst(vishost, visport);

   if (glvis_smooth &&
       glvis_src &&
       glvis_rec_avg &&
       glvis_dst &&
       visualization)
   {
      //glvis_smooth.precision(8);
      glvis_smooth << "parallel " << mesh.GetNRanks()
                   << " " << mesh.GetMyRank() << "\n"
                   << "solution\n" << mesh << u_smooth
                   << "window_title 'original'\n" << std::flush;
      MPI_Barrier(mesh.GetComm());
      //glvis_src.precision(8);
      glvis_src << "parallel " << mesh.GetNRanks()
                << " " << mesh.GetMyRank() << "\n"
                << "solution\n" << mesh << u_src
                << "window_title 'averages'\n" << std::flush;
      MPI_Barrier(mesh.GetComm());
      //glvis_dst.precision(8);
      glvis_dst << "parallel " << mesh.GetNRanks()
                << " " << mesh.GetMyRank() << "\n"
                << "solution\n" << mesh << u_dst
                << "window_title 'reconstruction'\n" << std::flush;
      MPI_Barrier(mesh.GetComm());
      //glvis_dst.precision(8);
      glvis_rec_avg << "parallel " << mesh.GetNRanks()
                    << " " << mesh.GetMyRank() << "\n"
                    << "solution\n" << mesh << u_rec_avg
                    << "window_title 'rec average'\n" << std::flush;
   }
   else if (visualization)
   {
      MFEM_WARNING("Cannot connect to glvis server, disabling visualization.")
   }

   Mpi::Finalize();
   return 0;
}
