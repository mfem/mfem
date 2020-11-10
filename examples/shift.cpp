//                       MFEM
//
// Compile with: make shift
//
// Sample runs:  shift
// make shift;./shift -m ../data/inline-quad.mesh  -rs 2

#include "../mfem.hpp"
#include <fstream>
#include <iostream>
#include "shift.hpp"

using namespace std;
using namespace mfem;

double velinit(const Vector &x)
{
   double dx = x(0) - 0.5,
          dy = x(1) - 0.5,
          rv = dx*dx + dy*dy;
   return rv > 0.22 ? 0. : 0.5;
}

int main(int argc, char *argv[])
{
   /// 1. Parse command-line options.
   const char *mesh_file = "../data/square-disc.mesh";
   int order = 2;
   bool static_cond = false;
   bool pa = false;
   const char *device_config = "cpu";
   bool visualization = true;
   int ser_ref_levels = 0;
   double dbc_val = 0.0;
   bool smooth = true;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree) or -1 for"
                  " isoparametric space.");
   args.AddOption(&static_cond, "-sc", "--static-condensation", "-no-sc",
                  "--no-static-condensation", "Enable static condensation.");
   args.AddOption(&pa, "-pa", "--partial-assembly", "-no-pa",
                  "--no-partial-assembly", "Enable Partial Assembly.");
   args.AddOption(&device_config, "-d", "--device",
                  "Device configuration string, see Device::Configure().");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&ser_ref_levels, "-rs", "--refine-serial",
                  "Number of times to refine the mesh uniformly in serial.");
   args.AddOption(&smooth, "-opt", "--opt", "-no-opt",
                  "--no-opt",
                  "Optimize the mesh.");

   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(cout);
      return 1;
   }
   args.PrintOptions(cout);

   // 2. Enable hardware devices such as GPUs, and programming models such as
   //    CUDA, OCCA, RAJA and OpenMP based on command line options.
   Device device(device_config);
   device.Print();

   // 3. Read the mesh from the given mesh file. We can handle triangular,
   //    quadrilateral, tetrahedral, hexahedral, surface and volume meshes with
   //    the same code.
   Mesh mesh(mesh_file, 1, 1);
   int dim = mesh.Dimension();
   for (int lev = 0; lev < ser_ref_levels; lev++) { mesh.UniformRefinement(); }

   // 5. Define a finite element space on the mesh. Here we use continuous
   //    Lagrange finite elements of the specified order. If order < 1, we
   //    instead use an isoparametric/isogeometric space.
   FiniteElementCollection *fec_mesh;
   if (order > 0)
   {
      fec_mesh = new H1_FECollection(order, dim);
   }
   else
   {
      fec_mesh = new H1_FECollection(order = 1, dim);
   }
   FiniteElementSpace *fespace_mesh = new FiniteElementSpace(&mesh, fec_mesh, dim);
   mesh.SetNodalFESpace(fespace_mesh);
   GridFunction x_mesh(fespace_mesh);
   mesh.SetNodalGridFunction(&x_mesh);

   // 5. Define a finite element space on the mesh. Here we use continuous
   //    Lagrange finite elements of the specified order. If order < 1, we
   //    instead use an isoparametric/isogeometric space.
   FiniteElementCollection *fec;
   bool delete_fec;
   if (order > 0)
   {
      fec = new H1_FECollection(order, dim);
      delete_fec = true;
   }
   else
   {
      fec = new H1_FECollection(order = 1, dim);
      delete_fec = true;
   }
   FiniteElementSpace fespace(&mesh, fec);
   L2_FECollection fecl2 = L2_FECollection(0, dim);
   FiniteElementSpace fesl2(&mesh, &fecl2);
   cout << "Number of finite element unknowns: "
        << fespace.GetTrueVSize() << endl;

   // 8. Define the solution vector x as a finite element grid function
   //    corresponding to fespace. Initialize x with initial guess of zero,
   //    which satisfies the boundary conditions.
   GridFunction x(&fespace), dist(&fespace);
   x = 0.0;
   GridFunction gfl2(&fesl2);

   FunctionCoefficient dist_fun_coef(dist_fun);

   x.ProjectCoefficient(dist_fun_coef);

   if (visualization && false)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      socketstream sol_sock(vishost, visport);
      sol_sock.precision(8);
      sol_sock << "solution\n" << mesh << x << flush;
      sol_sock << "window_title 'Distance function'\n"
               << "window_geometry "
               << 0 << " " << 0 << " " << 350 << " " << 350 << "\n"
               << "keys Rjmpc" << endl;
   }

   // optimize the mesh
   if (smooth) { optimize_mesh_with_distfun(mesh, x_mesh, x); }
   // project dist_fun at original smoothed mesh
   x.ProjectCoefficient(dist_fun_coef);
   dist.ProjectCoefficient(dist_fun_coef);

   if (visualization && false)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      socketstream sol_sock(vishost, visport);
      sol_sock.precision(8);
      sol_sock << "solution\n" << mesh << x << flush;
      sol_sock << "window_title 'Distance function'\n"
               << "window_geometry "
               << 0 << " " << 0 << " " << 350 << " " << 350 << "\n"
               << "keys Rjmpc" << endl;
   }

   int max_attr     = mesh.attributes.Max();
   IntegrationRules IntRulesLo(0, Quadrature1D::GaussLobatto);

   // Set trim flag based on the distance field
   // 0 if completely in the domain
   // 1 if completely outside the domain
   // 2 if partially inside the domain
   Array<int> trim_flag(mesh.GetNE());
   trim_flag = 0;
   Vector vals;
   for (int i = 0; i < mesh.GetNE(); i++)
   {
      ElementTransformation *Tr = mesh.GetElementTransformation(i);
      const IntegrationRule &ir =
         IntRulesLo.Get(mesh.GetElementBaseGeometry(i), 4*Tr->OrderJ());
      x.GetValues(i, ir, vals);

      /// Trim outside the boundary
      //        double minv = vals.Min();
      //        if (minv < 0.) {
      //            mesh.SetAttribute(i, max_attr+1);
      //            trim_flag[i] = 1;
      //        }

      /// Trim inside the boundary - this is better since we can direclty
      /// interpolate
      int count = 0;
      for (int j = 0; j < ir.GetNPoints(); j++)
      {
         double val = vals(j);
         if (val <= 0.) { count++; }
      }
      if (count == ir.GetNPoints())
      {
         mesh.SetAttribute(i, max_attr+1);
         trim_flag[i] = 1;
      }
      else if (count > 0)
      {
         trim_flag[i] = 2;
      }
   }

   Array<int> elstotrim;
   Array<int> sbm_int_face;
   Array<int> sbm_int_face_el;
   Array<int> sbm_int_flag; // 1 if int face, 2 if bdr face
   // Get SBM faces
   for (int i = 0; i < mesh.GetNumFaces(); i++)
   {
      FaceElementTransformations *tr;
      tr = mesh.GetInteriorFaceTransformations (i);
      if (tr != NULL)
      {
         int ne1 = tr->Elem1No;
         int ne2 = tr->Elem2No;
         int te1 = trim_flag[ne1], te2 = trim_flag[ne2];
         if (te1 == 1 && te2 != 1)
         {
            sbm_int_face.Append(i);
            sbm_int_face_el.Append(ne2);
            sbm_int_flag.Append(1);
         }
         if (te2 == 1 && te1 != 1)
         {
            sbm_int_face.Append(i);
            sbm_int_face_el.Append(ne1);
            sbm_int_flag.Append(1);
         }
      }
   }

   for (int i = 0; i < mesh.GetNBE(); i++)
   {
      FaceElementTransformations *tr;
      tr = mesh.GetBdrFaceTransformations (i);
      if (tr != NULL)
      {
         int ne1 = tr->Elem1No;
         int te1 = trim_flag[ne1];
         if (te1 == 1)
         {
            elstotrim.Append(ne1);
         }
         else if (te1 == 2)
         {
            sbm_int_face.Append(i);
            sbm_int_face_el.Append(ne1);
            sbm_int_flag.Append(2);
         }
      }
   }

   elstotrim.Sort();
   elstotrim.Unique();

   for (int i = 0; i < gfl2.Size(); i++)
   {
      gfl2(i) = trim_flag[i]*1.;
   }

   if (visualization)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      socketstream sol_sock(vishost, visport);
      sol_sock.precision(8);
      sol_sock << "solution\n" << mesh << gfl2 << flush;
      sol_sock << "window_title 'Distance function'\n"
               << "window_geometry "
               << 0 << " " << 0 << " " << 350 << " " << 350 << "\n"
               << "keys Rjmpc" << endl;
   }
   //MFEM_ABORT(" ");

   // Get the distance field vector
   GridFunction x_dx(&fespace), x_dy(&fespace),
                x_dx_dy(fespace_mesh);
   x.GetDerivative(1, 0, x_dx);
   x.GetDerivative(1, 1, x_dy);
   // set vector magnitude
   for (int i = 0; i < x_dx.Size(); i++)
   {
      double dxv = x_dx(i),
             dyv = x_dy(i);
      double mag = dxv*dxv + dyv*dyv;
      if (mag > 0) { mag = pow(mag, 0.5); }
      x_dx(i) *= x(i)/mag;
      x_dy(i) *= x(i)/mag;
   }

   // copy to vector GridFunction
   for (int i = 0; i < x_dx_dy.Size()/dim; i++)
   {
      x_dx_dy(i) = x_dx(i);
      x_dx_dy(i + x_dx_dy.Size()/dim) = x_dy(i);
   }
   x_dx_dy *= -1; // true = surrogate + d

   VectorFunctionCoefficient dis_vec(dim, dist_vec_c);
   x_dx_dy.ProjectCoefficient(dis_vec);

   // Visualize vector GridFunction
   if (visualization)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      socketstream sol_sock(vishost, visport);
      sol_sock.precision(8);
      sol_sock << "solution\n" << mesh << x_dx_dy << flush;
      sol_sock << "window_title 'DDDerivative distfun'\n"
               << "window_geometry "
               << 350 << " " << 350 << " " << 350 << " " << 350 << "\n"
               << "keys Rjmpc" << endl;
   }
   //MFEM_ABORT(" ");

   {
      ofstream mesh_ofs("refined.mesh");
      mesh_ofs.precision(8);
      mesh.Print(mesh_ofs);
      ofstream sol_ofs("dist.gf");
      sol_ofs.precision(8);
      x_dx_dy.Save(sol_ofs);
   }

   x = 0;


   // 6. Determine the list of true (i.e. conforming) essential boundary dofs.
   //    In this example, the boundary conditions are defined by marking all
   //    the boundary attributes from the mesh as essential (Dirichlet) and
   //    converting them to a list of true dofs.
   Array<int> ess_tdof_list;
   Array<int> ess_bdr(mesh.bdr_attributes.Max());
   if (mesh.bdr_attributes.Size())
   {
      ess_bdr = 1;
      //       ess_bdr[4] = 0;
      if (ess_bdr.Size() == 5) { ess_bdr[4] = 0; }
   }
   Array<int> dofs;
   Array<int> solvedofs;
   // approach 2 - get dofs of all elements that are not marked

   // First get all essential vdofs at boundary
   Array<int> ess_vdofs_bdr;
   fespace.GetEssentialVDofs(ess_bdr, ess_vdofs_bdr);

   // now get all dofs that are not part of the untrimmed elements
   Array<int> ess_vdofs_hole(ess_vdofs_bdr.Size());
   ess_vdofs_hole = -1;
   for (int e = 0; e < trim_flag.Size(); e++)
   {
      if (trim_flag[e] != 1)
      {
         fespace.GetElementDofs(e, dofs);
         solvedofs.Append(dofs);
      }
   }

   for (int s = 0; s < solvedofs.Size(); s++)
   {
      ess_vdofs_hole[solvedofs[s]] = 0;
   }


   // now combine the two lists
   for (int i = 0; i < ess_vdofs_hole.Size(); i++)
   {
      if (ess_vdofs_bdr[i] == -1) { ess_vdofs_hole[i] = -1; }
   }

   // convert to tdofs
   const SparseMatrix *R = fespace.GetConformingRestriction();
   Array<int> ess_tdofs;
   if (!R)
   {
      ess_tdofs.MakeRef(ess_vdofs_hole);
   }
   else
   {
      R->BooleanMult(ess_vdofs_hole, ess_tdofs);
   }
   fespace.MarkerToList(ess_tdofs, ess_tdof_list);

   double alpha = 10.;

   VectorGridFunctionCoefficient dist_vec(&x_dx_dy);


   // 7. Set up the linear form b(.) which corresponds to the right-hand side of
   //    the FEM linear system, which in this case is (1,phi_i) where phi_i are
   //    the basis functions in the finite element fespace.
   LinearForm b(&fespace);
   ConstantCoefficient one(1.0);
   b.AddDomainIntegrator(new DomainLFIntegrator(one), trim_flag);
   ConstantCoefficient dbcCoef(dbc_val);
   b.AddShiftedBdrFaceIntegrator(new SBM2LFIntegrator(dbcCoef, alpha, dist_vec),
                                 sbm_int_face, sbm_int_face_el, sbm_int_flag);
   //    b.AddBdrFaceIntegrator(new SBM2LFIntegrator(dbcCoef, alpha, dist_vec), ess_fake_bdr);
   b.Assemble();

   // 9. Set up the bilinear form a(.,.) on the finite element space
   //    corresponding to the Laplacian operator -Delta, by adding the Diffusion
   //    domain integrator.
   BilinearForm a(&fespace);
   //(nabla u, nabla w) - Term 1
   a.AddDomainIntegrator(new DiffusionIntegrator(one), trim_flag);
   // -(nabla u.n, w)-(nabla w.n, u) - Term 2 and 3
   //a.AddBdrFaceIntegrator(new DGDiffusionIntegrator(-1., 0.), ess_fake_bdr);
   // Terms 2 and 3 included via DGDiffusionIntegrator inside SBMIntegrator
   // <nabla u.d, nabla w.n> Term 4 and
   // <alpha h^{-1} u, w> Term 5
   // <alpha h^{-1} u, grad w.d> Term 6
   // <alpha h^{-1} grad u.d, w> Term 7
   // < alpha h^{-1} grad u.d, grad w.d> Term 8

   a.AddShiftedBdrFaceIntegrator(new SBM2Integrator(alpha, dist_vec),
                                 sbm_int_face, sbm_int_face_el, sbm_int_flag);
   //    a.AddBdrFaceIntegrator(new SBM2Integrator(alpha, dist_vec), ess_fake_bdr);
   FunctionCoefficient vel_init_coef(velinit);
   x.ProjectCoefficient(vel_init_coef);
   x = 0;

   // 10. Assemble the bilinear form and the corresponding linear system,
   //     applying any necessary transformations such as: eliminating boundary
   //     conditions, applying conforming constraints for non-conforming AMR,
   //     static condensation, etc.
   if (static_cond) { a.EnableStaticCondensation(); }

   a.Assemble();

   OperatorPtr A;
   Vector B, X;
   a.FormLinearSystem(ess_tdof_list, x, b, A, X, B);

   cout << "Size of linear system: " << A->Height() << endl;

   // 11. Solve the linear system A X = B.
   if (!pa)
   {
#ifndef MFEM_USE_SUITESPARSE
      // Use a simple symmetric Gauss-Seidel preconditioner with PCG.
      GSSmoother M((SparseMatrix&)(*A));
      //PCG(*A, M, B, X, 1, 200, 1e-12, 0.0);
      int max_iter = 200;
      double tol = 1.e-12;
      GMRES(*A, M, B, X, 1, 500, 50, 1e-12, 0.0);
      //MINRES(*A, M, B, X, 1, 200, 1e-12, 0.0);
#else
      // If MFEM was compiled with SuiteSparse, use UMFPACK to solve the system.
      UMFPackSolver umf_solver;
      umf_solver.Control[UMFPACK_ORDERING] = UMFPACK_ORDERING_METIS;
      umf_solver.SetOperator(*A);
      umf_solver.Mult(B, X);
#endif
   }
   else // Jacobi preconditioning in partial assembly mode
   {
      if (UsesTensorBasis(fespace))
      {
         OperatorJacobiSmoother M(a, ess_tdof_list);
         PCG(*A, M, B, X, 1, 400, 1e-12, 0.0);
      }
      else
      {
         CG(*A, B, X, 1, 400, 1e-12, 0.0);
      }
   }

   //    // 12. Recover the solution as a finite element grid function.
   a.RecoverFEMSolution(X, b, x);

   for (int i = 0; i < elstotrim.Size(); i++)
   {

   }
   //    for (int i = 0; i < x.Size(); i++) {
   //        if (dist(i) < 0.) { x(i) = 0.; }
   //    }


   // 13. Save the refined mesh and the solution. This output can be viewed later
   //     using GLVis: "glvis -m refined.mesh -g sol.gf".
   ofstream mesh_ofs("sbm.mesh");
   mesh_ofs.precision(8);
   mesh.Print(mesh_ofs);
   ofstream sol_ofs("sbm.gf");
   sol_ofs.precision(8);
   x.Save(sol_ofs);

   VisItDataCollection visit_dc("sbm", &mesh);
   visit_dc.RegisterField("sbmsol", &x);
   visit_dc.Save();

   // 14. Send the solution by socket to a GLVis server.
   if (visualization)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      socketstream sol_sock(vishost, visport);
      sol_sock.precision(8);
      sol_sock << "solution\n" << mesh << x << flush;
      sol_sock << "window_title 'Solution'\n"
               << "window_geometry "
               << 700 << " " << 0 << " " << 350 << " " << 350 << "\n"
               << "keys Rj" << endl;
   }

   // 15. Free the used memory.
   if (delete_fec)
   {
      delete fec;
   }

   return 0;
}
