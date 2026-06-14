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
#include "unit_tests.hpp"

using namespace mfem;

namespace mfem
{
namespace unit_tests
{

double LinearFunction2D(const Vector &x)
{
   return 2.0 * x[0] + 3.0 * x[1];
}

double LinearFunction3D(const Vector &x)
{
   return 2.0 * x[0] + 3.0 * x[1] + 4.0 * x[2];
}

double SmoothFunction2D(const Vector &x)
{
   const real_t pi = 4.0*atan(1.0);
   return sin(pi*x[0])*sin(pi*x[1]);
}

void GradSmoothFunction2D(const Vector &x, Vector &grad)
{
   const real_t pi = 4.0*atan(1.0);
   grad.SetSize(2);
   grad[0] = pi*cos(pi*x[0])*sin(pi*x[1]);
   grad[1] = pi*sin(pi*x[0])*cos(pi*x[1]);
}

void CheckRows(const Vector &values, int first, int last, real_t expected)
{
   for (int i = first; i < last; i++)
   {
      REQUIRE(values(i) == MFEM_Approx(expected));
   }
}

void CheckNodes2D(const FiniteElement &fe)
{
   const IntegrationRule &nodes = fe.GetNodes();
   bool has_nonzero_node = false;

   REQUIRE(nodes.GetNPoints() == fe.GetDof());
   for (int i = 0; i < nodes.GetNPoints(); i++)
   {
      const IntegrationPoint &ip = nodes.IntPoint(i);
      REQUIRE(ip.x >= -1e-12);
      REQUIRE(ip.x <= 1.0 + 1e-12);
      REQUIRE(ip.y >= -1e-12);
      REQUIRE(ip.y <= 1.0 + 1e-12);
      has_nonzero_node = has_nonzero_node || ip.x != 0.0 || ip.y != 0.0;
   }
   REQUIRE(has_nonzero_node);
}

void CheckNodes3D(const FiniteElement &fe)
{
   const IntegrationRule &nodes = fe.GetNodes();
   bool has_nonzero_node = false;

   REQUIRE(nodes.GetNPoints() == fe.GetDof());
   for (int i = 0; i < nodes.GetNPoints(); i++)
   {
      const IntegrationPoint &ip = nodes.IntPoint(i);
      REQUIRE(ip.x >= -1e-12);
      REQUIRE(ip.x <= 1.0 + 1e-12);
      REQUIRE(ip.y >= -1e-12);
      REQUIRE(ip.y <= 1.0 + 1e-12);
      REQUIRE(ip.z >= -1e-12);
      REQUIRE(ip.z <= 1.0 + 1e-12);
      has_nonzero_node = has_nonzero_node ||
                         ip.x != 0.0 || ip.y != 0.0 || ip.z != 0.0;
   }
   REQUIRE(has_nonzero_node);
}

void CheckProjectGrad2D(FiniteElementSpace &h1_fes,
                        FiniteElementSpace &nurbs_fes,
                        GridFunction &h1_gf)
{
   Array<int> h1_dofs;
   Vector h1_loc, projected_grad;

   for (int el = 0; el < h1_fes.GetNE(); el++)
   {
      ElementTransformation *T = h1_fes.GetElementTransformation(el);
      const FiniteElement *h1_fe = h1_fes.GetFE(el);
      const FiniteElement *nurbs_fe = nurbs_fes.GetFE(el);

      REQUIRE(T != nullptr);
      REQUIRE(h1_fe != nullptr);
      REQUIRE(nurbs_fe != nullptr);
      REQUIRE(nurbs_fe->GetGeomType() == Geometry::SQUARE);

      const NURBS_HCurl2DFiniteElement *hc_fe =
         dynamic_cast<const NURBS_HCurl2DFiniteElement*>(nurbs_fe);
      REQUIRE(hc_fe != nullptr);
      CheckNodes2D(*nurbs_fe);

      // The scalar NURBS mesh stores order p+1 for the component spaces used
      // to build H(curl), while the local H(curl) blocks have order p in the
      // differentiated component and p+1 in the transverse component.
      const int p = nurbs_fe->GetOrder() - 1;
      const int ndof_x = (p + 1)*(p + 2);
      const int ndof_y = (p + 2)*(p + 1);

      DenseMatrix grad;
      hc_fe->ProjectGrad(*h1_fe, *T, grad);
      REQUIRE(grad.Height() == nurbs_fe->GetDof());
      REQUIRE(grad.Width() == h1_fe->GetDof());
      REQUIRE(ndof_x + ndof_y == grad.Height());

      h1_fes.GetElementDofs(el, h1_dofs);
      h1_gf.GetSubVector(h1_dofs, h1_loc);
      REQUIRE(h1_loc.Size() == grad.Width());
      projected_grad.SetSize(grad.Height());
      grad.Mult(h1_loc, projected_grad);

      CheckRows(projected_grad, 0, ndof_x, 2.0);
      CheckRows(projected_grad, ndof_x, ndof_x + ndof_y, 3.0);
   }
}

void CheckProjectGrad3D(FiniteElementSpace &h1_fes,
                        FiniteElementSpace &nurbs_fes,
                        GridFunction &h1_gf)
{
   Array<int> h1_dofs;
   Vector h1_loc, projected_grad;

   for (int el = 0; el < h1_fes.GetNE(); el++)
   {
      ElementTransformation *T = h1_fes.GetElementTransformation(el);
      const FiniteElement *h1_fe = h1_fes.GetFE(el);
      const FiniteElement *nurbs_fe = nurbs_fes.GetFE(el);

      REQUIRE(T != nullptr);
      REQUIRE(h1_fe != nullptr);
      REQUIRE(nurbs_fe != nullptr);
      REQUIRE(nurbs_fe->GetGeomType() == Geometry::CUBE);

      const NURBS_HCurl3DFiniteElement *hc_fe =
         dynamic_cast<const NURBS_HCurl3DFiniteElement*>(nurbs_fe);
      REQUIRE(hc_fe != nullptr);
      CheckNodes3D(*nurbs_fe);

      const int p = nurbs_fe->GetOrder() - 1;
      const int ndof_x = (p + 1)*(p + 2)*(p + 2);
      const int ndof_y = (p + 2)*(p + 1)*(p + 2);
      const int ndof_z = (p + 2)*(p + 2)*(p + 1);

      DenseMatrix grad;
      hc_fe->ProjectGrad(*h1_fe, *T, grad);
      REQUIRE(grad.Height() == nurbs_fe->GetDof());
      REQUIRE(grad.Width() == h1_fe->GetDof());
      REQUIRE(ndof_x + ndof_y + ndof_z == grad.Height());

      h1_fes.GetElementDofs(el, h1_dofs);
      h1_gf.GetSubVector(h1_dofs, h1_loc);
      REQUIRE(h1_loc.Size() == grad.Width());
      projected_grad.SetSize(grad.Height());
      grad.Mult(h1_loc, projected_grad);

      CheckRows(projected_grad, 0, ndof_x, 2.0);
      CheckRows(projected_grad, ndof_x, ndof_x + ndof_y, 3.0);
      CheckRows(projected_grad, ndof_x + ndof_y,
                ndof_x + ndof_y + ndof_z, 4.0);
   }
}

void CheckProject2D(FiniteElementSpace &h1_fes,
                    FiniteElementSpace &nurbs_fes)
{
   Vector vector_dofs, projected;

   for (int el = 0; el < h1_fes.GetNE(); el++)
   {
      ElementTransformation *T = h1_fes.GetElementTransformation(el);
      const FiniteElement *h1_fe = h1_fes.GetFE(el);
      const FiniteElement *nurbs_fe = nurbs_fes.GetFE(el);

      REQUIRE(T != nullptr);
      REQUIRE(h1_fe != nullptr);
      REQUIRE(nurbs_fe != nullptr);

      const NURBS_HCurl2DFiniteElement *hc_fe =
         dynamic_cast<const NURBS_HCurl2DFiniteElement*>(nurbs_fe);
      REQUIRE(hc_fe != nullptr);

      const int p = nurbs_fe->GetOrder() - 1;
      const int ndof_x = (p + 1)*(p + 2);
      const int ndof_y = (p + 2)*(p + 1);

      DenseMatrix I;
      hc_fe->Project(*h1_fe, *T, I);
      REQUIRE(I.Height() == nurbs_fe->GetDof());
      REQUIRE(I.Width() == 2*h1_fe->GetDof());
      REQUIRE(ndof_x + ndof_y == I.Height());

      vector_dofs.SetSize(I.Width());
      for (int i = 0; i < h1_fe->GetDof(); i++)
      {
         vector_dofs(i) = 5.0;
         vector_dofs(i + h1_fe->GetDof()) = 7.0;
      }
      projected.SetSize(I.Height());
      I.Mult(vector_dofs, projected);

      CheckRows(projected, 0, ndof_x, 5.0);
      CheckRows(projected, ndof_x, ndof_x + ndof_y, 7.0);
   }
}

void CheckProject3D(FiniteElementSpace &h1_fes,
                    FiniteElementSpace &nurbs_fes)
{
   Vector vector_dofs, projected;

   for (int el = 0; el < h1_fes.GetNE(); el++)
   {
      ElementTransformation *T = h1_fes.GetElementTransformation(el);
      const FiniteElement *h1_fe = h1_fes.GetFE(el);
      const FiniteElement *nurbs_fe = nurbs_fes.GetFE(el);

      REQUIRE(T != nullptr);
      REQUIRE(h1_fe != nullptr);
      REQUIRE(nurbs_fe != nullptr);

      const NURBS_HCurl3DFiniteElement *hc_fe =
         dynamic_cast<const NURBS_HCurl3DFiniteElement*>(nurbs_fe);
      REQUIRE(hc_fe != nullptr);

      const int p = nurbs_fe->GetOrder() - 1;
      const int ndof_x = (p + 1)*(p + 2)*(p + 2);
      const int ndof_y = (p + 2)*(p + 1)*(p + 2);
      const int ndof_z = (p + 2)*(p + 2)*(p + 1);

      DenseMatrix I;
      hc_fe->Project(*h1_fe, *T, I);
      REQUIRE(I.Height() == nurbs_fe->GetDof());
      REQUIRE(I.Width() == 3*h1_fe->GetDof());
      REQUIRE(ndof_x + ndof_y + ndof_z == I.Height());

      vector_dofs.SetSize(I.Width());
      for (int i = 0; i < h1_fe->GetDof(); i++)
      {
         vector_dofs(i) = 5.0;
         vector_dofs(i + h1_fe->GetDof()) = 7.0;
         vector_dofs(i + 2*h1_fe->GetDof()) = 11.0;
      }
      projected.SetSize(I.Height());
      I.Mult(vector_dofs, projected);

      CheckRows(projected, 0, ndof_x, 5.0);
      CheckRows(projected, ndof_x, ndof_x + ndof_y, 7.0);
      CheckRows(projected, ndof_x + ndof_y,
                ndof_x + ndof_y + ndof_z, 11.0);
   }
}

real_t ComputeProjectGradL2Error(FiniteElementSpace &h1_fes,
                                 FiniteElementSpace &hcurl_fes,
                                 GridFunction &h1_gf,
                                 VectorCoefficient &grad_coeff)
{
   // Integrate the local H(curl) field produced by ProjectGrad directly. This
   // keeps the convergence check focused on the element projector.
   Array<int> h1_dofs;
   Vector h1_loc, hcurl_loc, approx, exact;
   DenseMatrix grad, vshape;
   real_t error = 0.0;

   for (int el = 0; el < h1_fes.GetNE(); el++)
   {
      ElementTransformation *T = h1_fes.GetElementTransformation(el);
      const FiniteElement *h1_fe = h1_fes.GetFE(el);
      const FiniteElement *hcurl_fe = hcurl_fes.GetFE(el);

      REQUIRE(T != nullptr);
      REQUIRE(h1_fe != nullptr);
      REQUIRE(hcurl_fe != nullptr);

      const NURBS_HCurl2DFiniteElement *hc_fe =
         dynamic_cast<const NURBS_HCurl2DFiniteElement*>(hcurl_fe);
      REQUIRE(hc_fe != nullptr);

      hc_fe->ProjectGrad(*h1_fe, *T, grad);

      h1_fes.GetElementDofs(el, h1_dofs);
      h1_gf.GetSubVector(h1_dofs, h1_loc);
      REQUIRE(h1_loc.Size() == grad.Width());
      hcurl_loc.SetSize(grad.Height());
      grad.Mult(h1_loc, hcurl_loc);

      const IntegrationRule &ir =
         IntRules.Get(hcurl_fe->GetGeomType(), 2*hcurl_fe->GetOrder() + 4);
      vshape.SetSize(hcurl_fe->GetDof(), hcurl_fe->GetRangeDim());
      approx.SetSize(hcurl_fe->GetRangeDim());
      for (int q = 0; q < ir.GetNPoints(); q++)
      {
         const IntegrationPoint &ip = ir.IntPoint(q);
         T->SetIntPoint(&ip);
         hcurl_fe->CalcVShape(*T, vshape);
         vshape.MultTranspose(hcurl_loc, approx);
         grad_coeff.Eval(exact, *T, ip);
         approx -= exact;
         error += ip.weight*T->Weight()*(approx*approx);
      }
   }

   return sqrt(error);
}

real_t ComputeProjectGradError2D(int ref)
{
   const int order = 2;

   Mesh mesh("data/square-nurbs.mesh");
   REQUIRE(mesh.NURBSext != nullptr);
   for (int l = 0; l < ref; l++)
   {
      mesh.UniformRefinement();
   }

   NURBSFECollection h1_fec(order);
   FiniteElementSpace h1_fes(&mesh, &h1_fec);

   NURBS_HCurlFECollection hcurl_fec(order, 2);
   FiniteElementSpace hcurl_fes(&mesh, &hcurl_fec);

   FunctionCoefficient f_coeff(SmoothFunction2D);
   GridFunction h1_gf(&h1_fes);
   h1_gf.ProjectCoefficient(f_coeff);

   VectorFunctionCoefficient grad_coeff(2, GradSmoothFunction2D);
   return ComputeProjectGradL2Error(h1_fes, hcurl_fes, h1_gf, grad_coeff);
}

TEST_CASE("NURBS ProjectGrad 2D", "[NURBSProjectGrad2D]")
{
   const int order = 2;

   Mesh mesh("data/square-nurbs.mesh");
   REQUIRE(mesh.NURBSext != nullptr);

   NURBSFECollection h1_fec(order);
   FiniteElementSpace h1_fes(&mesh, &h1_fec);

   NURBS_HCurlFECollection nurbs_fec(order, 2);
   FiniteElementSpace nurbs_fes(&mesh, &nurbs_fec);

   FunctionCoefficient f_coeff(LinearFunction2D);
   GridFunction h1_gf(&h1_fes);
   h1_gf.ProjectCoefficient(f_coeff);

   CheckProjectGrad2D(h1_fes, nurbs_fes, h1_gf);
}

TEST_CASE("NURBS ProjectGrad 3D", "[NURBSProjectGrad3D]")
{
   const int order = 2;

   Mesh mesh("data/cube-nurbs.mesh");
   REQUIRE(mesh.NURBSext != nullptr);

   NURBSFECollection h1_fec(order);
   FiniteElementSpace h1_fes(&mesh, &h1_fec);

   NURBS_HCurlFECollection nurbs_fec(order, 3);
   FiniteElementSpace nurbs_fes(&mesh, &nurbs_fec);

   FunctionCoefficient f_coeff(LinearFunction3D);
   GridFunction h1_gf(&h1_fes);
   h1_gf.ProjectCoefficient(f_coeff);

   CheckProjectGrad3D(h1_fes, nurbs_fes, h1_gf);
}

TEST_CASE("NURBS Project 2D", "[NURBSProject2D]" )
{
   const int order = 2;

   Mesh mesh("data/square-nurbs.mesh");
   REQUIRE(mesh.NURBSext != nullptr);

   NURBSFECollection h1_fec(order);
   FiniteElementSpace h1_fes(&mesh, &h1_fec);

   NURBS_HCurlFECollection nurbs_fec(order, 2);
   FiniteElementSpace nurbs_fes(&mesh, &nurbs_fec);

   CheckProject2D(h1_fes, nurbs_fes);
}

TEST_CASE("NURBS Project 3D", "[NURBSProject3D]" )
{
   const int order = 2;

   Mesh mesh("data/cube-nurbs.mesh");
   REQUIRE(mesh.NURBSext != nullptr);

   NURBSFECollection h1_fec(order);
   FiniteElementSpace h1_fes(&mesh, &h1_fec);

   NURBS_HCurlFECollection nurbs_fec(order, 3);
   FiniteElementSpace nurbs_fes(&mesh, &nurbs_fec);

   CheckProject3D(h1_fes, nurbs_fes);
}

TEST_CASE("NURBS ProjectGrad convergence 2D",
          "[NURBSProjectGradConvergence2D]")
{
   const real_t err0 = ComputeProjectGradError2D(0);
   const real_t err1 = ComputeProjectGradError2D(1);
   const real_t err2 = ComputeProjectGradError2D(2);
   const real_t err3 = ComputeProjectGradError2D(3);

   CAPTURE(err0);
   CAPTURE(err1);
   CAPTURE(err2);
   CAPTURE(err3);

   REQUIRE(err1 < err0);
   REQUIRE(err2 < err1);
   REQUIRE(err3 < err2);
   REQUIRE(err3 < 0.35*err0);
}

} // namespace unit_tests
} // namespace mfem
