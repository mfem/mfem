//                                MFEM Example 36
//
// Compile with: make ex36
//
// Sample runs:  ex36
//
// Description:  This example code demonstrates the use of MFEM to define a
//               discontinuous Galerkin (DG) finite element discretization of
//               the Laplace problem -Delta u = f with Dirichlet boundary
//               conditions. Finite element spaces of any order, including zero
//               on regular grids, are supported. The example highlights the
//               use of coupling solution domains though custom physics defined
//               on internal boundaries.
//
//               We recommend viewing examples 1, 14, and 34 before viewing this
//               example.

#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

class InteriorMassIntegrator : public BilinearFormIntegrator
{
public:
   InteriorMassIntegrator(Coefficient &Q)
      : Q(Q)
   {}

   void AssembleFaceMatrix(const FiniteElement &el1,
                           const FiniteElement &el2,
                           FaceElementTransformations &trans,
                           DenseMatrix &elmat) override;

   using BilinearFormIntegrator::AssembleFaceMatrix;

private:
   Coefficient &Q;

#ifndef MFEM_THREAD_SAFE
   Vector shape1;
   Vector shape2;
   DenseMatrix elmat11;
   DenseMatrix elmat12;
   DenseMatrix elmat21;
   DenseMatrix elmat22;
#endif
};

Mesh generate_mesh(int ref, int internal_bdr_attr = 5);

int main(int argc, char *argv[])
{
   // 1. Parse command-line options.
   int ref_levels = 0;
   int order = 1;
   int sol_order = 3;
   double jump = -2;
   double sigma = -1.0;
   double kappa = -1.0;
   double eta = 0.0;
   bool visualization = 1;

   OptionsParser args(argc, argv);
   args.AddOption(&ref_levels, "-r", "--refine",
                  "Number of times to refine the mesh uniformly, -1 for auto.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree) >= 0.");
   args.AddOption(&sigma, "-s", "--sigma",
                  "One of the three DG penalty parameters, typically +1/-1."
                  " See the documentation of class DGDiffusionIntegrator.");
   args.AddOption(&kappa, "-k", "--kappa",
                  "One of the three DG penalty parameters, should be positive."
                  " Negative values are replaced with (order+1)^2.");
   args.AddOption(&eta, "-e", "--eta", "BR2 penalty parameter.");
   args.AddOption(&sol_order, "-so", "--solution_order",
                  "Polynomial order of the exact solution >= 0.");
   args.AddOption(&jump, "-j", "--jump",
                  "Value of the discontinuity between the material regions.");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(cout);
      return 1;
   }
   if (kappa < 0)
   {
      kappa = (order+1)*(order+1);
   }
   if (sol_order < 0)
   {
      sol_order = 1;
   }
   args.PrintOptions(cout);

   // 2. Construct the (serial) mesh and refine it if requested.
   auto mesh = generate_mesh(ref_levels);

   int dim = mesh.Dimension();

   if (mesh.NURBSext)
   {
      mesh.SetCurvature(max(order, 1));
   }

   // 3. Define a finite element space on the mesh. Here we use discontinuous
   //    finite elements of the specified order >= 0.
   DG_FECollection fec(order, dim);
   FiniteElementSpace fespace(&mesh, &fec);
   cout << "Number of unknowns: " << fespace.GetVSize() << endl;

   // 4. Set up the linear form b(.) which corresponds to the right-hand side of
   //    the FEM linear system.
   LinearForm b(&fespace);

   Array<int> p1_attr_marker(mesh.attributes.Max());
   p1_attr_marker = 0;
   p1_attr_marker[0] = 1;

   FunctionCoefficient p1_source([sol_order](const Vector &p)
   {
      const double x = p(0);
      const double val = -(sol_order - 1)*sol_order*pow(x, sol_order-2);
      return val;
   });
   b.AddDomainIntegrator(new DomainLFIntegrator(p1_source), p1_attr_marker);

   Array<int> p2_attr_marker(mesh.attributes.Max());
   p2_attr_marker = 0;
   p2_attr_marker[1] = 1;

   FunctionCoefficient p2_source([sol_order](const Vector &p)
   {
      const double x = p(0);
      double val = -(sol_order - 1)*sol_order*pow(x - 2, sol_order-2);
      if (sol_order % 2 == 0)
      {
         val *= -1.0;
      }
      return val;
   });
   b.AddDomainIntegrator(new DomainLFIntegrator(p2_source), p2_attr_marker);

   ConstantCoefficient one(1.0);

   Array<int> p1_bdr_attr_marker(mesh.bdr_attributes.Max());
   p1_bdr_attr_marker = 0;
   p1_bdr_attr_marker[0] = 1;

   ConstantCoefficient left_bc_val(0.0);
   b.AddBdrFaceIntegrator(
      new DGDirichletLFIntegrator(left_bc_val, one, sigma, kappa),
      p1_bdr_attr_marker);

   Array<int> p2_bdr_attr_marker(mesh.bdr_attributes.Max());
   p2_bdr_attr_marker = 0;
   p2_bdr_attr_marker[1] = 1;

   ConstantCoefficient right_bc_val(2.0 + jump);
   b.AddBdrFaceIntegrator(
      new DGDirichletLFIntegrator(right_bc_val, one, sigma, kappa),
      p2_bdr_attr_marker);

   b.Assemble();

   // 5. Define the solution vector x as a finite element grid function
   //    corresponding to fespace. Initialize x with initial guess of zero.
   GridFunction x(&fespace);
   x = 0.0;

   // 6. Set up the bilinear form a(.,.) on the finite element space
   //    corresponding to the Laplacian operator -Delta, by adding the Diffusion
   //    domain integrator and the interior and boundary DG face integrators.
   //    Note that boundary conditions are imposed weakly in the form, so there
   //    is no need for dof elimination. After assembly and finalizing we
   //    extract the corresponding sparse matrix A.
   BilinearForm a(&fespace);
   a.AddDomainIntegrator(new DiffusionIntegrator(one));

   a.AddInteriorFaceIntegrator(new DGDiffusionIntegrator(one, sigma, kappa));
   a.AddBdrFaceIntegrator(new DGDiffusionIntegrator(one, sigma, kappa),
                          p1_bdr_attr_marker);
   a.AddBdrFaceIntegrator(new DGDiffusionIntegrator(one, sigma, kappa),
                          p2_bdr_attr_marker);
   if (eta > 0)
   {
      a.AddInteriorFaceIntegrator(new DGDiffusionBR2Integrator(fespace, eta));
      a.AddBdrFaceIntegrator(new DGDiffusionBR2Integrator(fespace, eta));
   }

   // 7. Negate the DG interface terms along the internal boundary so that the
   //    only coupling between domains is from the chosen model (constant flux
   //    in this case).
   Array<int> internal_bdr_attr_marker(mesh.bdr_attributes.Max());
   internal_bdr_attr_marker = 0;
   internal_bdr_attr_marker[4] = 1;

   ProductCoefficient neg_one(-1.0, one);
   a.AddInternalBoundaryFaceIntegrator(new DGDiffusionIntegrator(neg_one, sigma,
                                                                 kappa),
                                       internal_bdr_attr_marker);
   if (eta > 0)
   {
      a.AddInternalBoundaryFaceIntegrator(new DGDiffusionBR2Integrator(fespace,
                                                                       neg_one, eta),
                                          internal_bdr_attr_marker);
   }

   ConstantCoefficient mass_coeff(sol_order / jump);
   a.AddInternalBoundaryFaceIntegrator(new InteriorMassIntegrator(mass_coeff),
                                       internal_bdr_attr_marker);

   a.Assemble();
   a.Finalize();
   const SparseMatrix &A = a.SpMat();

#ifndef MFEM_USE_SUITESPARSE
   // 8. Define a simple symmetric Gauss-Seidel preconditioner and use it to
   //    solve the system Ax=b with PCG in the symmetric case, and GMRES in the
   //    non-symmetric one.
   GSSmoother M(A);
   if (sigma == -1.0 && !(jump < 0))
   {
      PCG(A, M, b, x, 1, 500, 1e-12, 0.0);
   }
   else
   {
      GMRES(A, M, b, x, 1, 500, 500, 1e-24, 0.0);
   }
#else
   // 8. If MFEM was compiled with SuiteSparse, use UMFPACK to solve the system.
   UMFPackSolver umf_solver;
   umf_solver.Control[UMFPACK_ORDERING] = UMFPACK_ORDERING_METIS;
   umf_solver.SetOperator(A);
   umf_solver.Mult(b, x);
#endif

   // 9. Save the refined mesh and the solution. This output can be viewed later
   //    using GLVis: "glvis -m refined.mesh -g sol.gf".
   ofstream mesh_ofs("refined.mesh");
   mesh_ofs.precision(8);
   mesh.Print(mesh_ofs);
   ofstream sol_ofs("sol.gf");
   sol_ofs.precision(8);
   x.Save(sol_ofs);

   // 10. Send the solution by socket to a GLVis server.
   if (visualization)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      socketstream sol_sock(vishost, visport);
      sol_sock.precision(8);
      sol_sock << "solution\n" << mesh << x << flush;
   }

   return 0;
}

void InteriorMassIntegrator::AssembleFaceMatrix(
   const FiniteElement &el1,
   const FiniteElement &el2,
   FaceElementTransformations &trans,
   DenseMatrix &elmat)
{
   int ndof1 = el1.GetDof();
   int ndof2 = el2.GetDof();
   int ndof = ndof1 + ndof2;

#ifdef MFEM_THREAD_SAFE
   Vector shape1;
   Vector shape2;
   DenseMatrix elmat11;
   DenseMatrix elmat12;
   DenseMatrix elmat21;
   DenseMatrix elmat22;
#endif
   shape1.SetSize(ndof1);
   shape2.SetSize(ndof2);

   elmat11.SetSize(ndof1);
   elmat12.SetSize(ndof1, ndof2);
   elmat21.SetSize(ndof2, ndof1);
   elmat22.SetSize(ndof2);

   const auto *ir = IntRule;
   if (ir == NULL)
   {
      int order = 2 * max(el1.GetOrder(), el2.GetOrder());
      ir = &IntRules.Get(trans.GetGeometryType(), order);
   }

   elmat.SetSize(ndof);
   elmat = 0.0;
   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      const auto &ip = ir->IntPoint(i);

      // Set the integration point in the face and the neighboring element
      trans.SetAllIntPoints(&ip);

      const double w = ip.weight * trans.Weight();

      // Access the neighboring element's integration point
      const auto &eip1 = trans.GetElement1IntPoint();
      const auto &eip2 = trans.GetElement2IntPoint();

      el1.CalcShape(eip1, shape1);
      el2.CalcShape(eip2, shape2);

      const double Q_val = Q.Eval(trans, ip);

      elmat11 = 0.0;
      AddMult_a_VVt(Q_val * w, shape1, elmat11);

      elmat12 = 0.0;
      AddMult_a_VWt(-Q_val * w, shape2, shape1, elmat12);

      elmat21 = 0.0;
      AddMult_a_VWt(-Q_val * w, shape1, shape2, elmat21);

      elmat22 = 0.0;
      AddMult_a_VVt(Q_val * w, shape2, elmat22);

      for (int j = 0; j < ndof1; ++j)
      {
         for (int k = 0; k < ndof1; ++k)
         {
            elmat(j, k) += elmat11(j, k);
         }
      }

      for (int j = 0; j < ndof1; ++j)
      {
         for (int k = 0; k < ndof2; ++k)
         {
            elmat(j, k + ndof1) += elmat12(j, k);
            elmat(k + ndof1, j) += elmat21(k, j);
         }
      }

      for (int j = 0; j < ndof2; ++j)
      {
         for (int k = 0; k < ndof2; ++k)
         {
            elmat(j + ndof1, k + ndof1) += elmat22(j, k);
         }
      }
   }
}

Mesh generate_mesh(int ref, int internal_bdr_attr)
{
   int nxy = 4 * (ref+1);
   auto mesh = Mesh::MakeCartesian2D(nxy, nxy, Element::TRIANGLE, true, 2.0, 1.0);
   // auto mesh = Mesh::MakeCartesian2D(nxy, nxy, Element::QUADRILATERAL, true, 2.0, 1.0);

   // assign element attributes to left and right sides
   for (int i = 0; i < mesh.GetNE(); ++i)
   {
      auto *elem = mesh.GetElement(i);

      Array<int> verts;
      elem->GetVertices(verts);

      bool left = true;
      for (int j = 0; j < verts.Size(); ++j)
      {
         auto *vtx = mesh.GetVertex(verts[j]);
         if (vtx[0] <= 1.0)
         {
            left = left;
         }
         else
         {
            left = false;
         }
      }
      if (left)
      {
         elem->SetAttribute(1);
      }
      else
      {
         elem->SetAttribute(2);
      }
   }

   // assign boundary element attributes to left and right sides
   for (int i = 0; i < mesh.GetNBE(); ++i)
   {
      auto *elem = mesh.GetBdrElement(i);

      Array<int> verts;
      elem->GetVertices(verts);

      bool left = true;
      bool right = true;
      bool top = true;
      bool bottom = true;
      for (int j = 0; j < verts.Size(); ++j)
      {
         auto *vtx = mesh.GetVertex(verts[j]);
         left = left && abs(vtx[0] - 0.0) < 1e-12;
         right = right && abs(vtx[0] - 2.0) < 1e-12;
         top = top && abs(vtx[1] - 1.0) < 1e-12;
         bottom = bottom && abs(vtx[1] - 0.0) < 1e-12;
      }
      if (left)
      {
         elem->SetAttribute(1);
      }
      else if (right)
      {
         elem->SetAttribute(2);
      }
      else if (top)
      {
         elem->SetAttribute(3);
      }
      else if (bottom)
      {
         elem->SetAttribute(4);
      }
   }

   // add internal boundary elements
   for (int i = 0; i < mesh.GetNumFaces(); ++i)
   {
      int e1, e2;
      mesh.GetFaceElements(i, &e1, &e2);
      if (e1 >= 0 && e2 >= 0 && mesh.GetAttribute(e1) != mesh.GetAttribute(e2))
      {
         // This is the internal face between attributes.
         auto *new_elem = mesh.GetFace(i)->Duplicate(&mesh);
         new_elem->SetAttribute(internal_bdr_attr);
         mesh.AddBdrElement(new_elem);
      }
   }

   mesh.FinalizeTopology(); // Finalize to build relevant tables
   mesh.Finalize();
   mesh.SetAttributes();

   return mesh;
}