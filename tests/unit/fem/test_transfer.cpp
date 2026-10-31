// Copyright (c) 2010-2026, Lawrence Livermore National Security, LLC. Produced
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

#include <memory>

using namespace mfem;

int RandomPRefinement(FiniteElementSpace & fes)
{
   Mesh *mesh = fes.GetMesh();
   int maxorder = 0;
   for (int i = 0; i < mesh->GetNE(); i++)
   {
      const int order = fes.GetElementOrder(i);
      maxorder = std::max(maxorder,order);
      if ((double) rand() / RAND_MAX < 0.5)
      {
         fes.SetElementOrder(i,order+1);
         maxorder = std::max(maxorder,order+1);
      }
   }
   fes.Update(false);
   return maxorder;
}


int dimension;
int coeff_order;
double coeff(const Vector& X)
{
   double x = X[0];
   double y = X[1];
   double z = 0.;
   if (dimension == 2)
   {
      if (coeff_order == 1)
      {
         return 1.1 * x + 2.0 * y;
      }
      else
      {
         return (1.-x)*x*(1.-y)*y;
      }
   }
   else
   {
      z = X[2];
      if (coeff_order == 1)
      {
         return 1.1 * x + 2.0 * y + 3.0 * z;
      }
      else
      {
         return (1.-x)*x*(1.-y)*y*(1.-z)*z;
      }
   }
}

void vectorcoeff(const Vector& x, Vector& y)
{
   y(0) = coeff(x);
   y(1) = -coeff(x);
   if (dimension == 3)
   {
      y(2) = 2.0 * coeff(x);
   }
}

enum class VecSpace { H1, VectorH1nodes, VectorH1vdim, ND, RT };

std::string VecSpaceName(VecSpace vectorspace)
{
   switch (vectorspace)
   {
      case VecSpace::H1: return "H1";
      case VecSpace::VectorH1nodes: return "Vector H1 by nodes";
      case VecSpace::VectorH1vdim: return "Vector H1 by vdim";
      case VecSpace::ND: return "Nedelec";
      case VecSpace::RT: return "Raviart-Thomas";
   }
   return "";
}

TEST_CASE("Transfer", "[Transfer]")
{
   auto vectorspace = GENERATE(VecSpace::H1, VecSpace::VectorH1nodes,
                               VecSpace::VectorH1vdim, VecSpace::ND, VecSpace::RT);
   auto geometric = GENERATE(true, false);
   auto simplex = GENERATE(true, false);
   dimension = GENERATE(2, 3);

   int order = 2;
   int ne = 2;

   int fineOrder = geometric ? order : 2*order;

   // Log test case information
   int total_ne = static_cast<int>(std::pow(ne, dimension));
   CAPTURE(VecSpaceName(vectorspace), dimension, simplex, total_ne, order,
           fineOrder, geometric);

   Mesh mesh;
   if (dimension == 2)
   {
      Element::Type type = simplex ? Element::TRIANGLE : Element::QUADRILATERAL;
      mesh = Mesh::MakeCartesian2D(ne, ne, type, 1, 1.0, 1.0);
   }
   else
   {
      Element::Type type = simplex ? Element::TETRAHEDRON : Element::HEXAHEDRON;
      mesh = Mesh::MakeCartesian3D(ne, ne, ne, type, 1.0, 1.0, 1.0);
   }
   FiniteElementCollection *c_fec = nullptr;
   FiniteElementCollection *f_fec = nullptr;

   switch (vectorspace)
   {
      case VecSpace::H1:
      case VecSpace::VectorH1nodes:
      case VecSpace::VectorH1vdim:
         c_fec = new H1_FECollection(order, dimension);
         f_fec = geometric ? c_fec : new H1_FECollection(fineOrder, dimension);
         break;
      case VecSpace::ND:
         c_fec = new ND_FECollection(order+1, dimension);
         f_fec = geometric ? c_fec : new ND_FECollection(fineOrder, dimension);
         break;
      case VecSpace::RT:
         c_fec = new RT_FECollection(order, dimension);
         f_fec = geometric ? c_fec : new RT_FECollection(fineOrder, dimension);
         break;
   }

   Mesh fineMesh(mesh);
   if (geometric)
   {
      fineMesh.UniformRefinement();
   }

   const int vdim = (vectorspace == VecSpace::VectorH1nodes
                     || vectorspace == VecSpace::VectorH1vdim) ? dimension : 1;
   Ordering::Type ordering = (vectorspace == VecSpace::VectorH1vdim)
                             ? Ordering::byVDIM : Ordering::byNODES;
   FiniteElementSpace *c_fespace =
      new FiniteElementSpace(&mesh, c_fec, vdim, ordering);
   FiniteElementSpace *f_fespace =
      new FiniteElementSpace(&fineMesh, f_fec, vdim, ordering);

   Operator* referenceOperator = nullptr;

   if (!geometric)
   {
      referenceOperator = new PRefinementTransferOperator(*c_fespace,
                                                          *f_fespace);
   }
   else
   {
      OperatorPtr P(Operator::ANY_TYPE);
      f_fespace->GetTransferOperator(*c_fespace, P);
      P.SetOperatorOwner(false);
      referenceOperator = P.Ptr();
   }

   TransferOperator testTransferOperator(*c_fespace, *f_fespace);
   GridFunction X(c_fespace);
   GridFunction X_cmp(c_fespace);
   GridFunction Y_exact(f_fespace);
   GridFunction Y_std(f_fespace);
   GridFunction Y_test(f_fespace);
   coeff_order = 1;
   if (vectorspace == VecSpace::H1)
   {
      FunctionCoefficient funcCoeff(&coeff);
      X.ProjectCoefficient(funcCoeff);
      Y_exact.ProjectCoefficient(funcCoeff);
   }
   else
   {
      VectorFunctionCoefficient funcCoeff(dimension, &vectorcoeff);
      X.ProjectCoefficient(funcCoeff);
      Y_exact.ProjectCoefficient(funcCoeff);
   }

   Y_std = 0.0;
   Y_test = 0.0;

   referenceOperator->Mult(X, Y_std);

   Y_std -= Y_exact;
   REQUIRE(Y_std.Norml2() < 1e-12 * Y_exact.Norml2());

   testTransferOperator.Mult(X, Y_test);

   Y_test -= Y_exact;
   REQUIRE(Y_test.Norml2() < 1e-12 * Y_exact.Norml2());

   referenceOperator->MultTranspose(Y_exact, X);
   testTransferOperator.MultTranspose(Y_exact, X_cmp);

   X -= X_cmp;
   REQUIRE(X.Norml2() < 1e-12 * X_cmp.Norml2());

   delete referenceOperator;
   delete f_fespace;
   delete c_fespace;
   if (geometric == 0)
   {
      delete f_fec;
   }
   delete c_fec;
}

TEST_CASE("InterpolationGridTransfer Round Trip", "[Transfer]")
{
   auto vectorspace = GENERATE(VecSpace::H1, VecSpace::VectorH1nodes,
                               VecSpace::VectorH1vdim, VecSpace::ND, VecSpace::RT);
   auto simplex = GENERATE(true, false);
   dimension = GENERATE(2, 3);
   const int order = 2;
   const int ne = 2;
   CAPTURE(VecSpaceName(vectorspace), dimension, simplex, order);

   Mesh mesh;
   if (dimension == 2)
   {
      Element::Type type = simplex ? Element::TRIANGLE : Element::QUADRILATERAL;
      mesh = Mesh::MakeCartesian2D(ne, ne, type, 1, 1.0, 1.0);
   }
   else
   {
      Element::Type type = simplex ? Element::TETRAHEDRON : Element::HEXAHEDRON;
      mesh = Mesh::MakeCartesian3D(ne, ne, ne, type, 1.0, 1.0, 1.0);
   }
   Mesh fineMesh(mesh);
   fineMesh.UniformRefinement();

   std::unique_ptr<FiniteElementCollection> fec;
   switch (vectorspace)
   {
      case VecSpace::H1:
      case VecSpace::VectorH1nodes:
      case VecSpace::VectorH1vdim:
         fec.reset(new H1_FECollection(order, dimension));
         break;
      case VecSpace::ND:
         fec.reset(new ND_FECollection(order, dimension));
         break;
      case VecSpace::RT:
         fec.reset(new RT_FECollection(order - 1, dimension));
         break;
   }

   const int vdim = (vectorspace == VecSpace::VectorH1nodes
                     || vectorspace == VecSpace::VectorH1vdim) ? dimension : 1;
   Ordering::Type ordering = (vectorspace == VecSpace::VectorH1vdim)
                             ? Ordering::byVDIM : Ordering::byNODES;
   FiniteElementSpace c_fespace(&mesh, fec.get(), vdim, ordering);
   FiniteElementSpace f_fespace(&fineMesh, fec.get(), vdim, ordering);

   GridFunction X(&c_fespace), Y(&f_fespace), X_rt(&c_fespace);
   coeff_order = 1;
   if (vectorspace == VecSpace::H1)
   {
      FunctionCoefficient funcCoeff(&coeff);
      X.ProjectCoefficient(funcCoeff);
   }
   else
   {
      VectorFunctionCoefficient funcCoeff(dimension, &vectorcoeff);
      X.ProjectCoefficient(funcCoeff);
   }

   // The backward operator is a left inverse of the forward operator, so a
   // forward/backward round trip must reproduce any coarse field.
   InterpolationGridTransfer transfer(c_fespace, f_fespace);
   transfer.ForwardOperator().Mult(X, Y);
   transfer.BackwardOperator().Mult(Y, X_rt);

   X_rt -= X;
   REQUIRE(X_rt.Norml2() < 1e-12 * X.Norml2());
}

TEST_CASE("Variable Order Transfer", "[Transfer][VariableOrder]")
{
   auto vectorspace = GENERATE(VecSpace::H1, VecSpace::VectorH1nodes,
                               VecSpace::VectorH1vdim, VecSpace::ND,
                               VecSpace::RT);
   dimension = GENERATE(2, 3);

   int ne = 2;
   int order = 2;

   // Log test case information
   int total_ne = static_cast<int>(pow(ne, dimension));
   CAPTURE(VecSpaceName(vectorspace), dimension, total_ne, order);

   Mesh mesh;
   if (dimension == 2)
   {
      Element::Type type = Element::QUADRILATERAL;
      mesh = Mesh::MakeCartesian2D(ne, ne, type, 1, 1.0, 1.0);
   }
   else
   {
      Element::Type type = Element::HEXAHEDRON;
      mesh = Mesh::MakeCartesian3D(ne, ne, ne, type, 1.0, 1.0, 1.0);
   }
   FiniteElementCollection* c_fec = nullptr;
   FiniteElementCollection* f_fec = nullptr;
   switch (vectorspace)
   {
      case VecSpace::H1:
      case VecSpace::VectorH1nodes:
      case VecSpace::VectorH1vdim:
         c_fec = new H1_FECollection(order, dimension);
         f_fec = new H1_FECollection(order, dimension);
         break;
      case VecSpace::ND:
         c_fec = new ND_FECollection(order+1, dimension);
         f_fec = new ND_FECollection(order+1, dimension);
         break;
      case VecSpace::RT:
         c_fec = new RT_FECollection(order, dimension);
         f_fec = new RT_FECollection(order, dimension);
         break;
   }

   mesh.EnsureNCMesh();
   mesh.RandomRefinement(0.5);

   const int vdim = (vectorspace == VecSpace::VectorH1nodes
                     || vectorspace == VecSpace::VectorH1vdim) ? dimension : 1;
   Ordering::Type ordering = (vectorspace == VecSpace::VectorH1vdim)
                             ? Ordering::byVDIM : Ordering::byNODES;

   FiniteElementSpace *c_fespace =
      new FiniteElementSpace(&mesh, c_fec, vdim, ordering);
   FiniteElementSpace *f_fespace =
      new FiniteElementSpace(&mesh, f_fec, vdim, ordering);

   RandomPRefinement(*f_fespace);

   Operator* referenceOperator = nullptr;

   referenceOperator = new PRefinementTransferOperator(*c_fespace,
                                                       *f_fespace);

   TransferOperator testTransferOperator(*c_fespace, *f_fespace);
   GridFunction X(c_fespace); X = 0.;
   GridFunction X_cmp(c_fespace); X_cmp = 0.;
   GridFunction Y_exact(f_fespace); Y_exact = 0.;
   GridFunction Y_std(f_fespace); Y_std = 0.;
   GridFunction Y_test(f_fespace); Y_test = 0.;
   coeff_order = std::min(2,order);
   if (vectorspace == VecSpace::H1)
   {
      FunctionCoefficient funcCoeff(&coeff);
      X.ProjectCoefficient(funcCoeff);
      Y_exact.ProjectCoefficient(funcCoeff);
   }
   else
   {
      VectorFunctionCoefficient funcCoeff(dimension, &vectorcoeff);
      X.ProjectCoefficient(funcCoeff);
      Y_exact.ProjectCoefficient(funcCoeff);
   }

   Y_std = 0.0;
   Y_test = 0.0;

   referenceOperator->Mult(X, Y_std);
   Y_std -= Y_exact;
   REQUIRE(Y_std.Norml2() < 1e-12 * Y_exact.Norml2());

   testTransferOperator.Mult(X, Y_test);

   Y_test -= Y_exact;
   REQUIRE(Y_test.Norml2() < 1e-12 * Y_exact.Norml2());

   referenceOperator->MultTranspose(Y_exact, X);
   testTransferOperator.MultTranspose(Y_exact, X_cmp);

   X -= X_cmp;
   REQUIRE(X.Norml2() < 1e-12 * X_cmp.Norml2());

   delete referenceOperator;
   delete f_fespace;
   delete c_fespace;
   delete f_fec;
   delete c_fec;
}

TEST_CASE("Variable Order True Transfer", "[Transfer][VariableOrder]")
{
   auto vectorspace = GENERATE(VecSpace::H1, VecSpace::VectorH1nodes,
                               VecSpace::VectorH1vdim);
   dimension = GENERATE(2, 3);

   int ne = 2;
   int order = 2;

   // Log test case information
   CAPTURE(VecSpaceName(vectorspace), dimension, order);

   Mesh mesh;
   if (dimension == 2)
   {
      Element::Type type = Element::QUADRILATERAL;
      mesh = Mesh::MakeCartesian2D(ne, ne, type, 1, 1.0, 1.0);
   }
   else
   {
      Element::Type type = Element::HEXAHEDRON;
      mesh = Mesh::MakeCartesian3D(ne, ne, ne, type, 1.0, 1.0, 1.0);
   }
   FiniteElementCollection *c_fec = nullptr;
   FiniteElementCollection *f_fec = nullptr;
   c_fec = new H1_FECollection(order, dimension);
   f_fec = new H1_FECollection(order, dimension);
   mesh.EnsureNCMesh();
   mesh.RandomRefinement(0.5);
   const int vdim = (vectorspace == VecSpace::VectorH1nodes
                     || vectorspace == VecSpace::VectorH1vdim) ? dimension : 1;
   Ordering::Type ordering = (vectorspace == VecSpace::VectorH1vdim)
                             ? Ordering::byVDIM : Ordering::byNODES;

   FiniteElementSpace *c_fespace =
      new FiniteElementSpace(&mesh, c_fec, vdim, ordering);
   FiniteElementSpace *f_fespace =
      new FiniteElementSpace(&mesh, f_fec, vdim, ordering);

   RandomPRefinement(*f_fespace);

   const SparseMatrix *Rc = c_fespace->GetRestrictionMatrix();
   TrueTransferOperator T(*c_fespace, *f_fespace);
   GridFunction xc(c_fespace);
   Vector Xc(c_fespace->GetTrueVSize());
   Vector Diff(c_fespace->GetTrueVSize());
   Vector Yc(c_fespace->GetTrueVSize());
   Vector Xf(f_fespace->GetTrueVSize());
   Vector Yf(f_fespace->GetTrueVSize());

   coeff_order = 2;

   BilinearFormIntegrator *massc = nullptr;
   BilinearFormIntegrator *massf = nullptr;

   if (vectorspace == VecSpace::H1)
   {
      FunctionCoefficient funcCoeff(&coeff);
      xc.ProjectCoefficient(funcCoeff);
      massc = new MassIntegrator;
      massf = new MassIntegrator;
   }
   else
   {
      VectorFunctionCoefficient funcCoeff(dimension, &vectorcoeff);
      xc.ProjectCoefficient(funcCoeff);
      massc = new VectorMassIntegrator;
      massf = new VectorMassIntegrator;
   }
   if (Rc)
   {
      Rc->Mult(xc,Xc);
   }
   else
   {
      Xc.MakeRef(xc,0);
   }
   T.Mult(Xc, Xf);

   BilinearForm mc(c_fespace);
   mc.AddDomainIntegrator(massc);
   mc.Assemble();
   SparseMatrix Mc;
   Array<int> empty;
   mc.FormSystemMatrix(empty, Mc);

   BilinearForm mf(f_fespace);
   mf.AddDomainIntegrator(massf);
   mf.Assemble();
   SparseMatrix Mf;
   mf.FormSystemMatrix(empty, Mf);

   Mf.Mult(Xf,Yf);

   T.MultTranspose(Yf,Yc);

   GSSmoother M(Mc);
   Diff = 0.0;
   PCG(Mc, M, Yc, Diff, 0, 500, 1e-24, 0.0);

   Diff -= Xc;
   REQUIRE(Diff.Norml2() < 1e-10);

   delete f_fespace;
   delete c_fespace;
   delete f_fec;
   delete c_fec;
}

TEST_CASE("Restriction Transpose Operator")
{
   int order = GENERATE(1, 2);
   auto mesh_fname = GENERATE("../../data/amr-quad.mesh",
                              "../../data/fichera-amr.mesh");

   Mesh mesh = Mesh::LoadFromFile(mesh_fname);
   H1_FECollection fec(order, mesh.Dimension());
   FiniteElementSpace fes(&mesh, &fec);

   BilinearForm a(&fes);

   const Operator *R = fes.GetRestrictionOperator();
   const Operator *Rt = fes.GetRestrictionTransposeOperator();
   const Operator *Rt_2 = a.GetOutputRestrictionTranspose();

   REQUIRE(R);
   REQUIRE(Rt);
   REQUIRE(Rt_2);

   Vector x(R->Height()), y1(R->Width()), y2(Rt->Height()), y3(Rt_2->Height());

   x.Randomize(1);

   R->MultTranspose(x, y1);
   Rt->Mult(x, y2);
   Rt_2->Mult(x, y3);

   y2 -= y1;
   REQUIRE(y2.Normlinf() == MFEM_Approx(0.0));

   y3 -= y1;
   REQUIRE(y3.Normlinf() == MFEM_Approx(0.0));
}


real_t sin_func(const Vector &x)
{
   return sin(M_PI * x.Sum());
}

void sin_vfunc(const Vector &x, Vector &y)
{
   y.SetSize(x.Size());
   for (int i = 0; i < y.Size(); i++)
   {
      y(i) = sin(M_PI * x[i]);
   }
}

namespace
{

struct TraceCollections
{
   std::unique_ptr<FiniteElementCollection> c_fec;
   std::unique_ptr<FiniteElementCollection> f_fec;
   std::unique_ptr<FiniteElementCollection> c_trace_fec;
   std::unique_ptr<FiniteElementCollection> f_trace_fec;
};

TraceCollections MakeTraceCollections(const VecSpace vectorspace,
                                      const int order, const int dim)
{
   TraceCollections fec;
   switch (vectorspace)
   {
      case VecSpace::H1:
      case VecSpace::VectorH1nodes:
      case VecSpace::VectorH1vdim:
         fec.c_fec = std::make_unique<H1_FECollection>(order, dim);
         fec.f_fec = std::make_unique<H1_FECollection>(order+1, dim);
         fec.c_trace_fec = std::make_unique<H1_Trace_FECollection>(order, dim);
         fec.f_trace_fec = std::make_unique<H1_Trace_FECollection>(order+1, dim);
         break;
      case VecSpace::ND:
         fec.c_fec = std::make_unique<ND_FECollection>(order, dim);
         fec.f_fec = std::make_unique<ND_FECollection>(order+1, dim);
         fec.c_trace_fec = std::make_unique<ND_Trace_FECollection>(order, dim);
         fec.f_trace_fec = std::make_unique<ND_Trace_FECollection>(order+1, dim);
         break;
      case VecSpace::RT:
         fec.c_fec = std::make_unique<RT_FECollection>(order-1, dim);
         fec.f_fec = std::make_unique<RT_FECollection>(order, dim);
         fec.c_trace_fec = std::make_unique<RT_Trace_FECollection>(order-1, dim);
         fec.f_trace_fec = std::make_unique<RT_Trace_FECollection>(order, dim);
         break;
   }
   return fec;
}

template <typename MeshT, typename FESpaceT, typename GridFunctionT>
void CheckTracePRefinementTrueTransfer(MeshT &mesh,
                                       FESpaceT &c_fes, FESpaceT &f_fes,
                                       FESpaceT &c_trace_fes, FESpaceT &f_trace_fes,
                                       const VecSpace vectorspace,
                                       const int dim,
                                       const bool assembleP)
{
   GridFunctionT x_c(&c_fes); x_c = 0.0;
   GridFunctionT x_f(&f_fes); x_f = 0.0;
   GridFunctionT x_trace_c(&c_trace_fes); x_trace_c = 0.0;
   GridFunctionT x_trace_f(&f_trace_fes); x_trace_f = 0.0;

   if (vectorspace == VecSpace::H1)
   {
      FunctionCoefficient cf(sin_func);
      x_c.ProjectCoefficient(cf);
      x_trace_c.ProjectTraceCoefficient(cf);
   }
   else
   {
      VectorFunctionCoefficient vec_cf(dim, &sin_vfunc);
      x_c.ProjectCoefficient(vec_cf);
      switch (vectorspace)
      {
         case VecSpace::VectorH1nodes:
         case VecSpace::VectorH1vdim:
            x_trace_c.ProjectTraceCoefficient(vec_cf);
            break;
         case VecSpace::ND:
            x_trace_c.ProjectTraceCoefficientTangent(vec_cf);
            break;
         case VecSpace::RT:
            x_trace_c.ProjectTraceCoefficientNormal(vec_cf);
            break;
         default:
            break;
      }
   }

   // Generate transfer operators for field and trace spaces
   PRefinementTransferOperator P(c_fes, f_fes, assembleP);
   PRefinementTransferOperator P_trace(c_trace_fes, f_trace_fes, assembleP);

   Vector x_c_true(c_fes.GetTrueVSize());
   Vector x_f_true(f_fes.GetTrueVSize());
   Vector x_trace_c_true(c_trace_fes.GetTrueVSize());
   Vector x_trace_f_true(f_trace_fes.GetTrueVSize());
   x_c.GetTrueDofs(x_c_true);
   x_trace_c.GetTrueDofs(x_trace_c_true);
   P.GetTrueTransferOperator()->Mult(x_c_true, x_f_true);
   P_trace.GetTrueTransferOperator()->Mult(x_trace_c_true, x_trace_f_true);

   x_f.SetFromTrueDofs(x_f_true);
   x_trace_f.SetFromTrueDofs(x_trace_f_true);

   // zero out interior dofs before and after p-ref to compare with trace
   Array<int> vdofs;
   for (int i = 0; i < mesh.GetNE(); i++)
   {
      c_fes.GetElementInteriorVDofs(i, vdofs);
      x_c.SetSubVector(vdofs, 0.0);
      f_fes.GetElementInteriorVDofs(i, vdofs);
      x_f.SetSubVector(vdofs, 0.0);
   }

   // Embed the trace dofs to a field GridFunction for comparison
   Array<int> face_vdofs, trace_vdofs;
   Vector values;
   GridFunctionT x_embedded_trace_c(&c_fes); x_embedded_trace_c = 0.0;
   GridFunctionT x_embedded_trace_f(&f_fes); x_embedded_trace_f = 0.0;
   for (int i = 0; i < mesh.GetNumFaces(); i++)
   {
      c_trace_fes.GetFaceVDofs(i, trace_vdofs);
      x_trace_c.GetSubVector(trace_vdofs, values);
      c_fes.GetFaceVDofs(i, face_vdofs);
      x_embedded_trace_c.SetSubVector(face_vdofs, values);

      f_trace_fes.GetFaceVDofs(i, trace_vdofs);
      x_trace_f.GetSubVector(trace_vdofs, values);
      f_fes.GetFaceVDofs(i, face_vdofs);
      x_embedded_trace_f.SetSubVector(face_vdofs, values);
   }

   x_embedded_trace_c -= x_c;
   REQUIRE(x_embedded_trace_c.Norml2() == MFEM_Approx(0.0));
   x_embedded_trace_f -= x_f;
   REQUIRE(x_embedded_trace_f.Norml2() == MFEM_Approx(0.0));
}

} // namespace


TEST_CASE("Trace PRefinement Serial TrueTransfer", "[Transfer]")
{
   auto simplex = GENERATE(true, false);
   dimension = GENERATE(2, 3);
   constexpr int ne = 4;
   auto order = GENERATE(1,2,3);
   auto vectorspace = GENERATE(VecSpace::H1, VecSpace::VectorH1nodes,
                               VecSpace::VectorH1vdim,
                               VecSpace::ND,VecSpace::RT);
   auto assembleP = GENERATE(false, true);
   auto amr = GENERATE(false, true);

   // Log test case information
   const int total_ne = static_cast<int>(std::pow(ne, dimension));
   CAPTURE(VecSpaceName(vectorspace),dimension, simplex, total_ne, order,
           assembleP);

   Mesh mesh;
   if (dimension == 2)
   {
      Element::Type type = simplex ? Element::TRIANGLE : Element::QUADRILATERAL;
      mesh = Mesh::MakeCartesian2D(ne, ne, type, 1, 1.0, 1.0);
   }
   else
   {
      Element::Type type = simplex ? Element::TETRAHEDRON : Element::HEXAHEDRON;
      mesh = Mesh::MakeCartesian3D(ne, ne, ne, type, 1.0, 1.0, 1.0);
   }

   if (amr) { mesh.RandomRefinement(0.5); }

   auto fec = MakeTraceCollections(vectorspace, order, dimension);

   const int vdim = (vectorspace == VecSpace::VectorH1nodes
                     || vectorspace == VecSpace::VectorH1vdim) ? dimension : 1;
   Ordering::Type ordering = (vectorspace == VecSpace::VectorH1vdim)
                             ? Ordering::byVDIM : Ordering::byNODES;

   FiniteElementSpace c_fes(&mesh, fec.c_fec.get(), vdim, ordering);
   FiniteElementSpace f_fes(&mesh, fec.f_fec.get(), vdim, ordering);
   FiniteElementSpace c_trace_fes(&mesh, fec.c_trace_fec.get(), vdim, ordering);
   FiniteElementSpace f_trace_fes(&mesh, fec.f_trace_fec.get(), vdim, ordering);

   CheckTracePRefinementTrueTransfer<Mesh, FiniteElementSpace, GridFunction>(
      mesh, c_fes, f_fes, c_trace_fes, f_trace_fes, vectorspace, dimension,
      assembleP);
}


TEST_CASE("Uniform pyramid refinement transfer", "[Transfer][Pyramid]")
{
   dimension = 3;
   const int pyr_type = GENERATE(0, 1);
   CAPTURE(pyr_type);

   Mesh coarse_mesh("../../data/ref-pyramid.mesh");
   Mesh fine_mesh(coarse_mesh);
   H1_FECollection fec(2, dimension, BasisType::GaussLobatto, pyr_type);
   FiniteElementSpace fes(&fine_mesh, &fec);
   GridFunction x(&fes);
   FunctionCoefficient func_coeff([](const Vector &X)
   {
      return X[0]*X[0] + X[1]*X[1] + X[2]*X[2] +
             X[0]*X[1] + X[1]*X[2];
   });
   x.ProjectCoefficient(func_coeff);
   // Both order-two pyramid bases represent this quadratic field exactly.
   REQUIRE(x.ComputeL2Error(func_coeff) < 1e-12);

   fine_mesh.UniformRefinement();

   // One pyramid refines into ten children in total.
   REQUIRE(fine_mesh.GetNE() == 10);
   int num_pyramids = 0;
   int num_tets = 0;
   for (int i = 0; i < fine_mesh.GetNE(); i++)
   {
      num_pyramids += fine_mesh.GetElementType(i) == Element::PYRAMID;
      num_tets += fine_mesh.GetElementType(i) == Element::TETRAHEDRON;
   }
   // Six children retain the parent's pyramid geometry.
   REQUIRE(num_pyramids == 6);
   // The other four children are tetrahedra.
   REQUIRE(num_tets == 4);

   const CoarseFineTransformations &trans = fine_mesh.GetRefinementTransforms();
   // Each child has a refinement embedding.
   REQUIRE(trans.embeddings.Size() == 10);
   const DenseMatrixStack &pmats =
      trans.point_matrices[Geometry::PYRAMID];
   // The pyramid parent has a point matrix for each child.
   REQUIRE(pmats.SizeK() == 10);
   for (int i = 0; i < trans.embeddings.Size(); i++)
   {
      // Every child belongs to the sole coarse element.
      REQUIRE(trans.embeddings[i].parent == 0);
      // The embedding records the parent's geometry, even for tetrahedra.
      REQUIRE(trans.embeddings[i].geom == Geometry::PYRAMID);
      // Child order matches the point-matrix stack index.
      REQUIRE(trans.embeddings[i].matrix == i);
      // Embeddings use three reference-space coordinates.
      REQUIRE(pmats.SizeI(i) == 3);
      // Pyramid children have five vertices; tetrahedra have four.
      REQUIRE(pmats.SizeJ(i) == (i < 6 ? 5 : 4));
   }

   fes.Update();
   x.Update();

   GridFunction exact(&fes);
   exact.ProjectCoefficient(func_coeff);
   x -= exact;
   // Updating the FE space and grid function preserves the quadratic field.
   REQUIRE(x.Normlinf() == Approx(0.0).margin(1e-12));

   FiniteElementSpace coarse_fes(&coarse_mesh, &fec);
   FiniteElementSpace fine_fes(&fine_mesh, &fec);
   GridFunction fine_x(&fine_fes);
   GridFunction coarse_x(&coarse_fes);
   GridFunction coarse_exact(&coarse_fes);
   fine_x.ProjectCoefficient(func_coeff);
   coarse_exact.ProjectCoefficient(func_coeff);

   InterpolationGridTransfer transfer(coarse_fes, fine_fes);
   transfer.BackwardOperator().Mult(fine_x, coarse_x);
   coarse_x -= coarse_exact;
   // Backward interpolation recovers the field on the coarse pyramid.
   REQUIRE(coarse_x.Normlinf() == Approx(0.0).margin(1e-12));

   OperatorHandle matrix_free(Operator::ANY_TYPE);
   OperatorHandle sparse(Operator::MFEM_SPARSEMAT);
   fine_fes.GetTransferOperator(coarse_fes, matrix_free);
   fine_fes.GetTransferOperator(coarse_fes, sparse);

   Vector coarse_values(coarse_fes.GetVSize());
   for (int i = 0; i < coarse_values.Size(); i++)
   {
      coarse_values(i) = (i%7 - 3)/7.0;
   }
   Vector matrix_free_values(fine_fes.GetVSize());
   Vector sparse_values(fine_fes.GetVSize());
   matrix_free->Mult(coarse_values, matrix_free_values);
   sparse->Mult(coarse_values, sparse_values);
   sparse_values -= matrix_free_values;
   // Sparse and matrix-free transfers agree for arbitrary coarse DOF values.
   REQUIRE(sparse_values.Normlinf() == Approx(0.0).margin(1e-12));
}

TEST_CASE("Uniform refinement transfer with mixed pyramid parents",
          "[Transfer][Pyramid]")
{
   const int pyr_type = GENERATE(0, 1);
   CAPTURE(pyr_type);

   Mesh coarse_mesh("../../data/l-mixed.mesh");
   Mesh fine_mesh(coarse_mesh);
   // Coarse elements have valid orientations.
   REQUIRE(coarse_mesh.CheckElementOrientation(false) == 0);
   // The coarse boundary faces are oriented consistently.
   REQUIRE(coarse_mesh.CheckBdrElementOrientation(false) == 0);
   H1_FECollection fec(2, 3, BasisType::GaussLobatto, pyr_type);
   FiniteElementSpace coarse_fes(&coarse_mesh, &fec);

   fine_mesh.UniformRefinement();
   // Refinement preserves positive element orientations.
   REQUIRE(fine_mesh.CheckElementOrientation(false) == 0);
   // Refinement preserves boundary-face orientations.
   REQUIRE(fine_mesh.CheckBdrElementOrientation(false) == 0);
   // Four pyramids, four wedges, and a final pyramid have different child
   // counts, so the final parent's children begin at offset 72.
   // The mixed coarse mesh has nine parents.
   REQUIRE(coarse_mesh.GetNE() == 9);
   // Five pyramids and four wedges produce 82 children.
   REQUIRE(fine_mesh.GetNE() == 82);
   const CoarseFineTransformations &trans = fine_mesh.GetRefinementTransforms();
   // Every fine element has an embedding.
   REQUIRE(trans.embeddings.Size() == fine_mesh.GetNE());

   int child_offset = 0;
   for (int parent = 0; parent < coarse_mesh.GetNE(); parent++)
   {
      const Geometry::Type geom = coarse_mesh.GetElementBaseGeometry(parent);
      const bool pyramid = parent < 4 || parent == 8;
      // The interleaved final pyramid must retain its own parent geometry.
      REQUIRE(geom == (pyramid ? Geometry::PYRAMID : Geometry::PRISM));
      const int num_children = pyramid ? 10 : 8;
      const DenseMatrixStack &pmats = trans.point_matrices[geom];
      for (int local = 0; local < num_children; local++)
      {
         const int child = child_offset + local;
         const Embedding &embedding = trans.embeddings[child];
         const Geometry::Type child_geom = pyramid ?
            (local < 6 ? Geometry::PYRAMID : Geometry::TETRAHEDRON) :
            Geometry::PRISM;
         CAPTURE(parent, child, local);
         // Children retain the correct parent across changing offsets.
         REQUIRE(embedding.parent == parent);
         // Embeddings record the parent geometry, including tetrahedral children.
         REQUIRE(embedding.geom == geom);
         // Each parent's child order selects the matching reference embedding.
         REQUIRE(embedding.matrix == local);
         // Child geometries follow the pyramid or wedge refinement pattern.
         REQUIRE(fine_mesh.GetElementBaseGeometry(child) == child_geom);
         // Reference point matrices have three coordinates per vertex.
         REQUIRE(pmats.SizeI(local) == 3);
         // Matrix widths match each child's number of vertices.
         REQUIRE(pmats.SizeJ(local) ==
                 (child_geom == Geometry::PYRAMID ? 5 :
                  child_geom == Geometry::TETRAHEDRON ? 4 : 6));
      }
      child_offset += num_children;
   }
   // All children are accounted for, including the last pyramid's ten.
   REQUIRE(child_offset == fine_mesh.GetNE());

   FiniteElementSpace fine_fes(&fine_mesh, &fec);
   OperatorHandle matrix_free(Operator::ANY_TYPE);
   OperatorHandle sparse(Operator::MFEM_SPARSEMAT);
   fine_fes.GetTransferOperator(coarse_fes, matrix_free);
   fine_fes.GetTransferOperator(coarse_fes, sparse);

   Vector coarse_values(coarse_fes.GetVSize());
   for (int i = 0; i < coarse_values.Size(); i++)
   {
      coarse_values(i) = (i%11 - 5)/11.0;
   }
   Vector fine_values(fine_fes.GetVSize());
   Vector sparse_values(fine_fes.GetVSize());
   matrix_free->Mult(coarse_values, fine_values);
   sparse->Mult(coarse_values, sparse_values);
   sparse_values -= fine_values;
   // Explicit and matrix-free transfers agree for arbitrary coarse values.
   REQUIRE(sparse_values.Normlinf() == Approx(0.0).margin(1e-11));

   Mesh update_mesh(coarse_mesh);
   FiniteElementSpace update_fes(&update_mesh, &fec);
   GridFunction update_values(&update_fes);
   update_values = coarse_values;
   update_mesh.UniformRefinement();
   update_fes.Update();
   update_values.Update();
   update_values -= fine_values;
   // Updating a mixed-parent FE space applies the same refinement transfer.
   REQUIRE(update_values.Normlinf() == Approx(0.0).margin(1e-11));

   InterpolationGridTransfer transfer(coarse_fes, fine_fes);
   Vector coarse_recovered(coarse_fes.GetVSize());
   transfer.BackwardOperator().Mult(fine_values, coarse_recovered);
   coarse_recovered -= coarse_values;
   // Derefinement is a left inverse across both parent reference types.
   REQUIRE(coarse_recovered.Normlinf() == Approx(0.0).margin(1e-11));

   L2_FECollection dg_fec(2, 3, BasisType::GaussLegendre,
                          FiniteElement::VALUE, pyr_type);
   FiniteElementSpace dg_fes(&coarse_mesh, &dg_fec);
   GridFunction parent_values(&dg_fes);
   Array<int> parent_dofs;
   for (int parent = 0; parent < coarse_mesh.GetNE(); parent++)
   {
      dg_fes.GetElementDofs(parent, parent_dofs);
      for (int i = 0; i < parent_dofs.Size(); i++)
      {
         parent_values(parent_dofs[i]) = parent + 1.0;
      }
   }
   GridFunctionCoefficient parent_coefficient(&parent_values);
   for (int child = 0; child < fine_mesh.GetNE(); child++)
   {
      ElementTransformation &T = *fine_mesh.GetElementTransformation(child);
      const IntegrationRule &ir =
         IntRules.Get(fine_mesh.GetElementBaseGeometry(child), 3);
      const int parent = trans.embeddings[child].parent;
      for (int q = 0; q < ir.GetNPoints(); q++)
      {
         const IntegrationPoint &ip = ir.IntPoint(q);
         T.SetIntPoint(&ip);
         // A coarse-grid coefficient selects the right parent for every child.
         REQUIRE(parent_coefficient.Eval(T, ip) ==
                 Approx(parent + 1.0).margin(1e-12));
      }
   }

}

TEST_CASE("L2 projection on mixed pyramid parents", "[Transfer][Pyramid]")
{
   const int pyr_type = GENERATE(0, 1);
   const bool continuous = GENERATE(false, true);
   CAPTURE(pyr_type, continuous);

   Mesh coarse_mesh("../../data/l-mixed.mesh");
   Mesh fine_mesh(coarse_mesh);
   fine_mesh.UniformRefinement();

   // The order-two H1 lumped mass has zero rows on this wedge-containing mesh;
   // order one gives a well-defined continuous projection. The L2 case retains
   // order two to exercise the different pyramid and tetrahedron DOF counts.
   std::unique_ptr<FiniteElementCollection> fec;
   if (continuous)
   {
      fec.reset(new H1_FECollection(1, 3, BasisType::GaussLobatto, pyr_type));
   }
   else
   {
      fec.reset(new L2_FECollection(2, 3, BasisType::GaussLegendre,
                                    FiniteElement::VALUE, pyr_type));
   }
   FiniteElementSpace coarse_fes(&coarse_mesh, fec.get());
   FiniteElementSpace fine_fes(&fine_mesh, fec.get());
   ConstantCoefficient one(1.0);
   GridFunction coarse_constant(&coarse_fes);
   GridFunction fine_constant(&fine_fes);
   GridFunction projected(&fine_fes);
   coarse_constant.ProjectCoefficient(one);
   fine_constant.ProjectCoefficient(one);

   L2ProjectionGridTransfer standard(coarse_fes, fine_fes);
   standard.ForwardOperator().Mult(coarse_constant, projected);
   projected -= fine_constant;
   // Standard projection preserves constants across both parent geometries.
   REQUIRE(projected.Normlinf() == Approx(0.0).margin(1e-11));

   FunctionCoefficient probe([](const Vector &X)
   {
      return X[0] + 2.0*X[1] - X[2] + X[0]*X[1];
   });
   GridFunction coarse_probe(&coarse_fes);
   GridFunction fine_standard(&fine_fes);
   GridFunction fine_ea(&fine_fes);
   coarse_probe.ProjectCoefficient(probe);
   L2ProjectionGridTransfer ea_requested(coarse_fes, fine_fes);
   ea_requested.UseEA(true);
   standard.ForwardOperator().Mult(coarse_probe, fine_standard);
   ea_requested.ForwardOperator().Mult(coarse_probe, fine_ea);
   fine_ea -= fine_standard;
   // EA requests fall back to standard assembly on the mixed mesh.
   REQUIRE(fine_ea.Normlinf() == Approx(0.0).margin(1e-11));
}

TEST_CASE("Uniform refinement with legacy RT0 collection", "[Transfer]")
{
   const bool simplex = GENERATE(false, true);
   CAPTURE(simplex);

   const Element::Type element_type = simplex ? Element::TRIANGLE :
                                      Element::QUADRILATERAL;
   Mesh mesh = Mesh::MakeCartesian2D(
                  2, 2, element_type, true, 1.0, 1.0);
   RT0_2DFECollection fec;
   FiniteElementSpace fes(&mesh, &fec);

   Vector constant(2);
   constant[0] = 1.0;
   constant[1] = -2.0;
   VectorConstantCoefficient coefficient(constant);
   GridFunction x(&fes);
   x.ProjectCoefficient(coefficient);

   mesh.UniformRefinement();
   fes.Update();
   x.Update();

   GridFunction exact(&fes);
   exact.ProjectCoefficient(coefficient);
   x -= exact;
   // Same-geometry refinement transfers legacy RT0 values correctly.
   REQUIRE(x.Normlinf() == Approx(0.0).margin(1e-12));
}

TEST_CASE("L2 projection transfer with mixed pyramid children",
          "[Transfer][Pyramid]")
{
   dimension = 3;
   coeff_order = 1;
   const int pyr_type = GENERATE(0, 1);
   const bool continuous = GENERATE(false, true);
   const bool use_ea = GENERATE(false, true);
   CAPTURE(pyr_type, continuous, use_ea);

   std::unique_ptr<FiniteElementCollection> fec;
   if (continuous)
   {
      fec.reset(new H1_FECollection(2, dimension, BasisType::GaussLobatto,
                                    pyr_type));
   }
   else
   {
      fec.reset(new L2_FECollection(2, dimension, BasisType::GaussLegendre,
                                    FiniteElement::VALUE, pyr_type));
   }

   Mesh coarse_mesh("../../data/ref-pyramid.mesh");
   Mesh fine_mesh(coarse_mesh);
   fine_mesh.UniformRefinement();
   FiniteElementSpace coarse_fes(&coarse_mesh, fec.get());
   FiniteElementSpace fine_fes(&fine_mesh, fec.get());
   FunctionCoefficient func_coeff(&coeff);
   // The H1 transfer uses a lumped fine-mesh mass matrix. It preserves
   // constants exactly, but does not generally reproduce nonconstant fields.
   ConstantCoefficient constant_coeff(1.0);
   Coefficient *test_coeff = continuous ?
                             static_cast<Coefficient*>(&constant_coeff) :
                             static_cast<Coefficient*>(&func_coeff);
   FunctionCoefficient coarse_probe_coeff([](const Vector &X)
   {
      return X[0] + 2.0*X[1] - X[2] + X[0]*X[1];
   });
   FunctionCoefficient fine_probe_coeff([](const Vector &X)
   {
      return X[0]*X[0] - X[1]*X[2] + 0.5*X[2];
   });

   GridFunction coarse_x(&coarse_fes);
   GridFunction fine_x(&fine_fes);
   GridFunction fine_exact(&fine_fes);
   GridFunction coarse_recovered(&coarse_fes);
   GridFunction coarse_exact(&coarse_fes);
   GridFunction coarse_probe(&coarse_fes);
   GridFunction fine_probe(&fine_fes);
   GridFunction coarse_transpose(&coarse_fes);
   GridFunction fine_transpose(&fine_fes);
   coarse_x.ProjectCoefficient(*test_coeff);
   fine_exact.ProjectCoefficient(*test_coeff);
   coarse_exact.ProjectCoefficient(*test_coeff);
   coarse_probe.ProjectCoefficient(coarse_probe_coeff);
   fine_probe.ProjectCoefficient(fine_probe_coeff);

   L2ProjectionGridTransfer transfer(coarse_fes, fine_fes);
   transfer.UseEA(use_ea);
   const Operator &forward = transfer.ForwardOperator();
   forward.Mult(coarse_x, fine_x);
   fine_x -= fine_exact;
   // The chosen H1 constant or L2 linear field transfers exactly.
   REQUIRE(fine_x.Normlinf() == Approx(0.0).margin(1e-11));
   forward.Mult(coarse_probe, fine_x);
   forward.MultTranspose(fine_probe, coarse_transpose);
   // Forward multiplication and its transpose are adjoints on nonconstant data.
   REQUIRE(InnerProduct(fine_x, fine_probe) ==
           Approx(InnerProduct(coarse_probe, coarse_transpose)).epsilon(1e-11));

   if (use_ea)
   {
      L2ProjectionGridTransfer standard_transfer(coarse_fes, fine_fes);
      GridFunction fine_standard(&fine_fes);
      standard_transfer.ForwardOperator().Mult(coarse_probe, fine_standard);
      fine_standard -= fine_x;
      // Unsupported EA agrees with standard assembly on nonconstant data.
      REQUIRE(fine_standard.Normlinf() == Approx(0.0).margin(1e-11));
   }

   const Operator &backward = transfer.BackwardOperator();
   backward.Mult(fine_exact, coarse_recovered);
   coarse_recovered -= coarse_exact;
   // Backward transfer recovers the chosen field on the coarse mesh.
   REQUIRE(coarse_recovered.Normlinf() == Approx(0.0).margin(1e-11));
   backward.Mult(fine_probe, coarse_recovered);
   backward.MultTranspose(coarse_probe, fine_transpose);
   // Backward multiplication and its transpose are also adjoints.
   REQUIRE(InnerProduct(coarse_recovered, coarse_probe) ==
           Approx(InnerProduct(fine_probe, fine_transpose)).epsilon(1e-11));
}

TEST_CASE("L2 projection EA falls back for triangle bases", "[Transfer]")
{
   dimension = 2;
   coeff_order = 1;
   const bool continuous = GENERATE(false, true);
   CAPTURE(continuous);

   std::unique_ptr<FiniteElementCollection> fec;
   if (continuous)
   {
      fec.reset(new H1_FECollection(2, dimension));
   }
   else
   {
      fec.reset(new L2_FECollection(2, dimension));
   }

   Mesh coarse_mesh = Mesh::MakeCartesian2D(
                         1, 1, Element::TRIANGLE, true, 1.0, 1.0);
   Mesh fine_mesh(coarse_mesh);
   fine_mesh.UniformRefinement();
   FiniteElementSpace coarse_fes(&coarse_mesh, fec.get());
   FiniteElementSpace fine_fes(&fine_mesh, fec.get());
   // The coarse triangle space cannot use the tensor-basis EA kernels.
   REQUIRE_FALSE(UsesTensorBasis(coarse_fes));
   // Refinement does not change that basis restriction.
   REQUIRE_FALSE(UsesTensorBasis(fine_fes));

   FunctionCoefficient func_coeff(&coeff);
   GridFunction coarse_x(&coarse_fes);
   GridFunction fine_ref(&fine_fes);
   GridFunction fine_ea(&fine_fes);
   GridFunction coarse_ref(&coarse_fes);
   GridFunction coarse_ea(&coarse_fes);
   coarse_x.ProjectCoefficient(func_coeff);

   L2ProjectionGridTransfer standard_transfer(coarse_fes, fine_fes);
   L2ProjectionGridTransfer ea_transfer(coarse_fes, fine_fes);
   ea_transfer.UseEA(true);
   standard_transfer.ForwardOperator().Mult(coarse_x, fine_ref);
   ea_transfer.ForwardOperator().Mult(coarse_x, fine_ea);
   standard_transfer.BackwardOperator().Mult(fine_ref, coarse_ref);
   ea_transfer.BackwardOperator().Mult(fine_ref, coarse_ea);

   fine_ea -= fine_ref;
   coarse_ea -= coarse_ref;
   // Requesting EA gives the standard forward transfer result.
   REQUIRE(fine_ea.Normlinf() == Approx(0.0).margin(1e-12));
   // The backward operator uses the same standard-assembly fallback.
   REQUIRE(coarse_ea.Normlinf() == Approx(0.0).margin(1e-12));
}

TEST_CASE("Uniform refinement uses active parent geometries", "[Transfer]")
{
   QuadraticFECollection fec;
   Mesh mesh = Mesh::MakeCartesian3D(
                  1, 1, 1, Element::HEXAHEDRON, 1.0, 1.0, 1.0);
   FiniteElementSpace fes(&mesh, &fec);
   GridFunction x(&fes);
   x = 1.0;

   mesh.UniformRefinement();
   const CoarseFineTransformations &trans = mesh.GetRefinementTransforms();
   Array<Geometry::Type> parent_geometries;
   trans.GetParentGeometries(parent_geometries);
   // The hex-only mesh has one active parent geometry.
   REQUIRE(parent_geometries.Size() == 1);
   // That active geometry is a cube.
   REQUIRE(parent_geometries[0] == Geometry::CUBE);
   // An unused pyramid point-matrix stack is nevertheless populated.
   REQUIRE(trans.point_matrices[Geometry::PYRAMID].SizeK() > 0);

   fes.Update();
   x.Update();

   Mesh coarse_mesh = Mesh::MakeCartesian3D(
                         1, 1, 1, Element::HEXAHEDRON, 1.0, 1.0, 1.0);
   Mesh fine_mesh(coarse_mesh);
   fine_mesh.UniformRefinement();
   FiniteElementSpace coarse_fes(&coarse_mesh, &fec);
   FiniteElementSpace fine_fes(&fine_mesh, &fec);
   InterpolationGridTransfer transfer(coarse_fes, fine_fes);
   const Operator &forward = transfer.ForwardOperator();
   const Operator &backward = transfer.BackwardOperator();
   // The forward operator maps into the refined FE space.
   REQUIRE(forward.Height() == fine_fes.GetVSize());
   // Its input comes from the coarse FE space.
   REQUIRE(forward.Width() == coarse_fes.GetVSize());
   // The backward operator maps into the coarse FE space.
   REQUIRE(backward.Height() == coarse_fes.GetVSize());
   // Its input comes from the refined FE space.
   REQUIRE(backward.Width() == fine_fes.GetVSize());

   InterpolationGridTransfer assembled_transfer(coarse_fes, fine_fes);
   assembled_transfer.SetOperatorType(Operator::MFEM_SPARSEMAT);
   const Operator &assembled_forward = assembled_transfer.ForwardOperator();
   // Sparse assembly also creates all refined output rows.
   REQUIRE(assembled_forward.Height() == fine_fes.GetVSize());
   // Sparse assembly retains the correct coarse input size.
   REQUIRE(assembled_forward.Width() == coarse_fes.GetVSize());
}

#ifdef MFEM_USE_MPI

TEST_CASE("Parallel Transfer", "[Transfer][Parallel]")
{
   auto simplex = GENERATE(true, false);
   auto geometric = GENERATE(true, false);
   dimension = GENERATE(2, 3);
   int ne = 4;
   int order = 2;

   int fineOrder = geometric ? order : 2 * order;
   int num_procs, myid;
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
   MPI_Comm_rank(MPI_COMM_WORLD, &myid);

   // Log test case information
   int total_ne = static_cast<int>(std::pow(ne, dimension));
   CAPTURE(dimension, simplex, total_ne, order, fineOrder, geometric);

   coeff_order = 1;

   Mesh mesh;
   if (dimension == 2)
   {
      Element::Type type = simplex ? Element::TRIANGLE : Element::QUADRILATERAL;
      mesh = Mesh::MakeCartesian2D(ne, ne, type, 1, 1.0, 1.0);
   }
   else
   {
      Element::Type type = simplex ? Element::TETRAHEDRON : Element::HEXAHEDRON;
      mesh = Mesh::MakeCartesian3D(ne, ne, ne, type, 1.0, 1.0, 1.0);
   }

   Mesh fineMesh(mesh);
   if (geometric)
   {
      fineMesh.UniformRefinement();
   }

   ParMesh *pmesh = new ParMesh(MPI_COMM_WORLD, mesh);
   ParMesh pfineMesh(MPI_COMM_WORLD, mesh);
   if (geometric)
   {
      pfineMesh.UniformRefinement();
   }

   FiniteElementCollection *c_h1_fec =
      new H1_FECollection(order, dimension);
   FiniteElementCollection *f_h1_fec = geometric ? c_h1_fec : new
                                       H1_FECollection(fineOrder, dimension);

   constexpr int vdim = 1;

   double referenceRestrictionValue = 0.0;

   // Compute reference values in serial
   {
      FiniteElementSpace* c_h1_fespace = new FiniteElementSpace(&mesh, c_h1_fec,
                                                                vdim);
      FiniteElementSpace* f_h1_fespace = new FiniteElementSpace(&fineMesh,
                                                                f_h1_fec, vdim);

      Operator* transferOperator = new TransferOperator(*c_h1_fespace,
                                                        *f_h1_fespace);
      GridFunction X(c_h1_fespace);
      GridFunction Y(f_h1_fespace);

      FunctionCoefficient funcCoeff(&coeff);
      Y.ProjectCoefficient(funcCoeff);
      X = 0.0;

      transferOperator->MultTranspose(Y, X);

      referenceRestrictionValue = std::sqrt(InnerProduct(X, X));

      delete transferOperator;
      delete f_h1_fespace;
      delete c_h1_fespace;
   }

   ParFiniteElementSpace* c_h1_fespace = new ParFiniteElementSpace(pmesh,
                                                                   c_h1_fec,
                                                                   vdim);
   ParFiniteElementSpace* f_h1_fespace = new ParFiniteElementSpace(&pfineMesh,
                                                                   f_h1_fec,
                                                                   vdim);

   Operator* transferOperator = new TrueTransferOperator(*c_h1_fespace,
                                                         *f_h1_fespace);
   ParGridFunction X(c_h1_fespace);
   ParGridFunction Y_exact(f_h1_fespace);
   ParGridFunction Y(f_h1_fespace);
   FunctionCoefficient funcCoeff(&coeff);
   X.ProjectCoefficient(funcCoeff);
   Y_exact.ProjectCoefficient(funcCoeff);

   Y = 0.0;

   Vector X_true(c_h1_fespace->GetTrueVSize());
   Vector Y_true(f_h1_fespace->GetTrueVSize());

   c_h1_fespace->GetRestrictionMatrix()->Mult(X, X_true);
   transferOperator->Mult(X_true, Y_true);
   f_h1_fespace->GetProlongationMatrix()->Mult(Y_true, Y);

   Y -= Y_exact;
   REQUIRE(Y.Norml2() < 1e-12 * Y_exact.Norml2());

   f_h1_fespace->GetRestrictionMatrix()->Mult(Y_exact, Y_true);
   transferOperator->MultTranspose(Y_true, X_true);

   double restrictionValue = std::sqrt(InnerProduct(MPI_COMM_WORLD, X_true,
                                                    X_true));
   REQUIRE(std::abs(restrictionValue - referenceRestrictionValue) < 1e-12 *
           std::abs(referenceRestrictionValue));

   delete transferOperator;
   delete f_h1_fespace;
   delete c_h1_fespace;
   if (!geometric) { delete f_h1_fec; }
   delete c_h1_fec;
   delete pmesh;
}

TEST_CASE("Parallel InterpolationGridTransfer Round Trip",
          "[Transfer][Parallel]")
{
   auto simplex = GENERATE(true, false);
   dimension = 3;
   const int order = 2;
   const int ne = 2;
   CAPTURE(simplex, order);

   Element::Type type = simplex ? Element::TETRAHEDRON : Element::HEXAHEDRON;
   Mesh mesh = Mesh::MakeCartesian3D(ne, ne, ne, type, 1.0, 1.0, 1.0);
   ParMesh pmesh(MPI_COMM_WORLD, mesh);
   ParMesh pfineMesh(pmesh);
   pfineMesh.UniformRefinement();

   ND_FECollection fec(order, dimension);
   ParFiniteElementSpace c_fespace(&pmesh, &fec);
   ParFiniteElementSpace f_fespace(&pfineMesh, &fec);

   ParGridFunction X(&c_fespace), Y(&f_fespace), X_rt(&c_fespace);
   coeff_order = 1;
   VectorFunctionCoefficient funcCoeff(dimension, &vectorcoeff);
   X.ProjectCoefficient(funcCoeff);

   InterpolationGridTransfer transfer(c_fespace, f_fespace);
   transfer.ForwardOperator().Mult(X, Y);
   transfer.BackwardOperator().Mult(Y, X_rt);

   X_rt -= X;
   const real_t err_norm = std::sqrt(InnerProduct(MPI_COMM_WORLD, X_rt, X_rt));
   const real_t X_norm = std::sqrt(InnerProduct(MPI_COMM_WORLD, X, X));
   REQUIRE(err_norm < 1e-12 * X_norm);
}

TEST_CASE("Trace PRefinement Parallel TrueTransfer", "[Transfer][Parallel]")
{
   auto simplex = GENERATE(true, false);
   dimension = GENERATE(2, 3);
   constexpr int ne = 4;
   auto order = GENERATE(1,2,3);
   auto vectorspace = GENERATE(VecSpace::H1, VecSpace::VectorH1nodes,
                               VecSpace::VectorH1vdim,
                               VecSpace::ND,VecSpace::RT);
   auto assembleP = GENERATE(true, false);

   auto amr = GENERATE(true, false);

   // Log test case information
   const int total_ne = static_cast<int>(std::pow(ne, dimension));
   CAPTURE(VecSpaceName(vectorspace),dimension, simplex, total_ne, order,
           assembleP);

   Mesh mesh;
   if (dimension == 2)
   {
      Element::Type type = simplex ? Element::TRIANGLE : Element::QUADRILATERAL;
      mesh = Mesh::MakeCartesian2D(ne, ne, type, 1, 1.0, 1.0);
   }
   else
   {
      Element::Type type = simplex ? Element::TETRAHEDRON : Element::HEXAHEDRON;
      mesh = Mesh::MakeCartesian3D(ne, ne, ne, type, 1.0, 1.0, 1.0);
   }

   if (amr) { mesh.EnsureNCMesh(true); }

   ParMesh pmesh(MPI_COMM_WORLD, mesh);

   if (amr) { pmesh.RandomRefinement(0.5); }

   auto fec = MakeTraceCollections(vectorspace, order, dimension);

   const int vdim = (vectorspace == VecSpace::VectorH1nodes
                     || vectorspace == VecSpace::VectorH1vdim) ? dimension : 1;
   Ordering::Type ordering = (vectorspace == VecSpace::VectorH1vdim)
                             ? Ordering::byVDIM : Ordering::byNODES;

   ParFiniteElementSpace c_fes(&pmesh, fec.c_fec.get(), vdim, ordering);
   ParFiniteElementSpace f_fes(&pmesh, fec.f_fec.get(), vdim, ordering);
   ParFiniteElementSpace c_trace_fes(&pmesh, fec.c_trace_fec.get(), vdim,
                                     ordering);
   ParFiniteElementSpace f_trace_fes(&pmesh, fec.f_trace_fec.get(), vdim,
                                     ordering);

   CheckTracePRefinementTrueTransfer<ParMesh, ParFiniteElementSpace, ParGridFunction>
   (
      pmesh, c_fes, f_fes, c_trace_fes, f_trace_fes, vectorspace, dimension,
      assembleP);
}



#endif
