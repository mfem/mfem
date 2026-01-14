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

#include <iostream>
#include <cmath>

using namespace mfem;


/**
* Compute the error of the taylor series expansion of the shapefunctions, upto
* and including the hessian term:
*  res = shape(xi) + dshape(xi)*eps*dx + 0.5*hessian(xi)*eps*eps*dx*dx
*        - shape(xi + eps*dx)
*/
real_t TaylorSeriesError(const FiniteElement* fe,
                         const IntegrationPoint &ip,
                         const Vector &dx,
                         const real_t eps)
{
   const int dof = fe->GetDof();
   const int dim = fe->GetDim();
   const int hdim = (dim*(dim+1))/2;

   Vector shape(dof);
   DenseMatrix dshape(dof,dim);
   DenseMatrix hessian(dof,hdim);

   fe->CalcShape(ip, shape);
   fe->CalcDShape(ip, dshape);
   fe->CalcHessian(ip, hessian);

   Vector dx2(hdim);
   if (dim == 1)
   {
      dx2[0] = dx[0]*dx[0];
   }
   else if (dim == 2)
   {
      dx2[0] =   dx[0]*dx[0];
      dx2[1] = 2*dx[0]*dx[1];
      dx2[2] =   dx[1]*dx[1];
   }
   else if (dim == 3)
   {
      dx2[0] =   dx[0]*dx[0];
      dx2[1] = 2*dx[0]*dx[1];
      dx2[2] = 2*dx[0]*dx[2];
      dx2[3] =   dx[1]*dx[1];
      dx2[4] = 2*dx[1]*dx[2];
      dx2[5] =   dx[2]*dx[2];
   }

   Vector res(dof);
   res = shape;
   dshape.AddMult(dx, res, eps);
   hessian.AddMult(dx2, res, 0.5*eps*eps);

   IntegrationPoint ip_eps;
   Vector shape_eps(dof);
   ip_eps.x = ip.x + eps*dx[0];
   if (dim >= 2 ) { ip_eps.y = ip.y + eps*dx[1]; }
   if (dim == 3 ) { ip_eps.z = ip.z + eps*dx[2]; }

   fe->CalcShape(ip_eps, shape_eps);
   res -= shape_eps;
   return res.Norml2();
}

/**
* Check the convergence of the taylor series, of a given element @a fe at
* a given point @a ip in a given direction @a dx.
* For linear and quadratic elements the taylor series is exact.
* For other elements the convergence should be third order.
*/

void CheckTaylorSeries(const FiniteElement* fe,
                       const IntegrationPoint &ip,
                       const Vector &dx)
{
   real_t eps = 0.1;
   constexpr real_t red = 4.0;
   constexpr int steps = 100;
   constexpr real_t tol = 1e-8;

   real_t error = TaylorSeriesError(fe, ip, dx, eps);
   real_t order;
   int i;
   for (i = 0; i < steps; ++i)
   {
      eps /= red;
      real_t err_new = TaylorSeriesError(fe, ip, dx, eps);
      order = log(error/err_new)/log(red);
      error = err_new;
      if (error < tol) { break; }
   }
   mfem::out<<i<<" "<<error<<" "<<order<<std::endl;
   if (i == 0)
   {
      REQUIRE(error == MFEM_Approx(0));
   }
   else
   {
      REQUIRE(order > 2.98);
   }
}

/**
* Test if a given element @a fe has the correct behaviour of the taylor series.
*/
void TestCalcHessian(const FiniteElement* fe)
{
   const int dim = fe->GetDim();

   constexpr int check_res = 2;
   int num_check_dirs = dim;

   // Get a uniform grid of integration points
   RefinedGeometry* ref = GlobGeometryRefiner.Refine(fe->GetGeomType(),
                                                     check_res);
   const IntegrationRule& intRule = ref->RefPts;
   int npoints = intRule.GetNPoints();
   Vector dx(dim);
   for (int i=0; i < npoints; ++i)
   {
      // Get the current integration point from intRule
      IntegrationPoint pt = intRule.IntPoint(i);

      for (int j=0; j < num_check_dirs; ++j)
      {
         dx[0] = sin(2*j + 0.3);
         if (dim >= 2) { dx[1] = cos(5*j + 0.2); }
         if (dim == 3) { dx[2] = sin(3*j + 0.1); }
         CheckTaylorSeries(fe, pt, dx);
      }
   }
}


TEST_CASE("CalcHessian",
          "[Linear1DFiniteElement]"
          "[Linear2DFiniteElement]"
          "[Linear3DFiniteElement]"
          "[BiLinear2DFiniteElement]"
          "[TriLinear3DFiniteElement]"
          "[H1_SegmentElement]"
          "[H1_QuadrilateralElement]"
          "[H1_HexahedronElement]"
          "[H1_TriangleElement]"
          "[H1_TetrahedronElement]"
          "[NURBS1DFiniteElement]"
          "[NURBS2DFiniteElement]"
          "[NURBS3DFiniteElement]")
{

   // Fixed Order Elements
   SECTION("Linear1DFiniteElement")
   {
      mfem::out<<"Linear1DFiniteElement"<<std::endl;
      Linear1DFiniteElement fe;
      TestCalcHessian(&fe);
   }

   SECTION("Linear2DFiniteElement")
   {
      mfem::out<<"Linear2DFiniteElement"<<std::endl;
      Linear2DFiniteElement fe;
      TestCalcHessian(&fe);
   }

   SECTION("Linear3DFiniteElement")
   {
      mfem::out<<"Linear3DFiniteElement"<<std::endl;
      Linear3DFiniteElement fe;
      TestCalcHessian(&fe);
   }

   SECTION("BiLinear2DFiniteElement")
   {
      mfem::out<<"BiLinear2DFiniteElement"<<std::endl;
      BiLinear2DFiniteElement fe;
      TestCalcHessian(&fe);
   }

   SECTION("TriLinear3DFiniteElement")
   {
      mfem::out<<"TriLinear3DFiniteElement"<<std::endl;
      TriLinear3DFiniteElement fe;
      TestCalcHessian(&fe);
   }

   // H1 Elements
   SECTION("H1_SegmentElement")
   {
      int order = GENERATE(1,2,3,4,5);
      mfem::out<<"H1_SegmentElement = "<<order<<std::endl;
      H1_SegmentElement fe(order);
      TestCalcHessian(&fe);
   }

   SECTION("H1_QuadrilateralElement")
   {
      int order = GENERATE(1,2,3,4,5);
      H1_QuadrilateralElement fe(order);
      mfem::out<<"H1_QuadrilateralElement = "<<order<<std::endl;
      TestCalcHessian(&fe);
   }

   SECTION("H1_HexahedronElement")
   {
      int order = GENERATE(1,2,3,4,5);
      mfem::out<<"H1_HexahedronElement = "<<order<<std::endl;
      H1_HexahedronElement fe(order);
      TestCalcHessian(&fe);
   }

   SECTION("H1_TriangleElement")
   {
      int order = GENERATE(1,2,3,4,5);
      mfem::out<<"H1_TriangleElement = "<<order<<std::endl;
      H1_TriangleElement fe(order);
      TestCalcHessian(&fe);
   }

   SECTION("H1_TetrahedronElement")
   {
      int order = GENERATE(1,2,3,4,5);
      mfem::out<<"H1_TetrahedronElement = "<<order<<std::endl;
      H1_TetrahedronElement fe(order);
      TestCalcHessian(&fe);
   }

   // NURBS Elements
   SECTION("NURBS1DFiniteElement")
   {
      int order = GENERATE(1,2,3,4,5);
      mfem::out<<"NURBS1DFiniteElement = "<<order<<std::endl;
      NURBS1DFiniteElement fe(order);
      Array <const KnotVector*> kv(1);
      kv[0] = new KnotVector(order);
      fe.KnotVectors() = kv;
      int IJK[1];
      IJK[0] = 0;
      fe.SetIJK(IJK);
      fe.SetOrder();
      fe.Weights() = 1.0;
      TestCalcHessian(&fe);
      delete kv[0];
   }

   SECTION("NURBS2DFiniteElement")
   {
      int order = GENERATE(1,2,3,4,5);
      mfem::out<<"NURBS2DFiniteElement = "<<order<<std::endl;
      NURBS2DFiniteElement fe(order);
      Array <const KnotVector*> kv(2);
      kv[0] = new KnotVector(order);
      kv[1] = new KnotVector(order);
      fe.KnotVectors() = kv;
      int IJK[2];
      IJK[0] = IJK[1] = 0;
      fe.SetIJK(IJK);
      fe.SetOrder();
      fe.Weights() = 1.0;
      TestCalcHessian(&fe);
      delete kv[0];
      delete kv[1];
   }

   SECTION("NURBS3DFiniteElement")
   {
      int order = GENERATE(1,2,3,4,5);
      mfem::out<<"NURBS3DFiniteElement = "<<order<<std::endl;
      NURBS3DFiniteElement fe(order);
      Array <const KnotVector*> kv(3);
      kv[0] = new KnotVector(order);
      kv[1] = new KnotVector(order);
      kv[2] = new KnotVector(order);
      fe.KnotVectors() = kv;
      int IJK[3];
      IJK[0] = IJK[1] = IJK[2] = 0;
      fe.SetIJK(IJK);
      fe.SetOrder();
      fe.Weights() = 1.0;
      TestCalcHessian(&fe);
      delete kv[0];
      delete kv[1];
      delete kv[2];
   }
}

TEST_CASE("Laplacian",
          "[NURBS2DFiniteElement]"
          "[NURBS3DFiniteElement]")
{
   int order = 4;
   std::string meshName = GENERATE("square-nurbs.mesh",
                                   "cube-nurbs.mesh");
   mfem::out<<"\nCheck laplacian for "<< meshName <<std::endl;
   bool deformed = GENERATE(false,true);
   if (deformed) { mfem::out<<"Mesh is deformed"<<std::endl; }
   bool NURBS = GENERATE(false,true);
   if (NURBS) { mfem::out<<"Using NURBS"<<std::endl; }

   Mesh mesh("../../data/" + meshName, 1, 1);
   const int dim = mesh.Dimension();

   // Rotate mesh
   DenseMatrix Rotate(dim);
   if (dim == 2)
   {
      NURBSPatch::Get2DRotationMatrix(M_PI/7, Rotate);
   }
   else if (dim == 3)
   {
      real_t n[] = {0.0,0.0,1.0};
      NURBSPatch::Get3DRotationMatrix(n, M_PI/7,M_PI/7, Rotate);
   }

   Vector x0(dim), x1(dim);
   for (int i = 0; i <mesh.GetNodes()->Size()/dim; i++)
   {
      mesh.GetNode(i, x0.GetData());
      Rotate.Mult(x0, x1);
      mesh.SetNode(i, x1.GetData());
   }

   // Distort mesh
   real_t distort_scale = 0.05;
   if (deformed)
   {
      Vector dx(mesh.GetNodes()->Size());
      dx.Randomize(1234);
      dx *= 2.0; dx -= 1.0; dx *= distort_scale;
      mesh.MoveNodes(dx);
   }

   if (NURBS)
   {
      // We need a C1 smooth mesh
      mesh.DegreeElevate(1);

      // Refine mesh
      mesh.UniformRefinement();

      // Distort mesh
      distort_scale = 0.01;
      if (deformed)
      {
         Vector dx(mesh.GetNodes()->Size());
         dx.Randomize(1234);
         dx *= 2.0; dx -= 1.0; dx *= distort_scale;
         mesh.MoveNodes(dx);
      }
   }

   // Create Space
   FiniteElementCollection *fe_coll = nullptr;
   NURBSExtension *ext  = nullptr;
   if (NURBS)
   {
      fe_coll = new NURBSFECollection (order);
      ext = new NURBSExtension(mesh.NURBSext, order);
   }
   else
   {
      fe_coll = new H1_FECollection (order);
   }
   FiniteElementSpace fes(&mesh, ext, fe_coll);

   // Compute (grad w, grad phi) + (w, laplace phi) = 0
   SparseMatrix gmat(fes.GetNDofs());
   Vector shape, lshape;
   DenseMatrix dshape, elmat;

   DofTransformation doftrans;
   ElementTransformation *eltrans;
   Array<int> vdofs;
   for (int e = 0; e < fes.GetNE(); e++)
   {
      const int dof = fes.GetFE(e)->GetDof();
      shape.SetSize(dof);
      dshape.SetSize(dof,dim);
      lshape.SetSize(dof);

      elmat.SetSize(dof);
      elmat = 0.0;
      eltrans = fes.GetElementTransformation (e);

      // Integrand involves non-polynomial mapping
      const int intorder = 3*fes.GetFE(e)->GetOrder();
      const IntegrationRule *ir = &IntRules.Get(fes.GetFE(e)->GetGeomType(),
                                                intorder);

      elmat = 0.0;
      for (int i = 0; i < ir->GetNPoints(); i++)
      {
         const IntegrationPoint &ip = ir->IntPoint(i);
         eltrans->SetIntPoint(&ip);
         const real_t w = ip.weight * eltrans->Weight();

         fes.GetFE(e)->CalcShape(ip, shape);
         fes.GetFE(e)->CalcPhysLaplacian(*eltrans, lshape);
         fes.GetFE(e)->CalcPhysDShape(*eltrans, dshape);

         // Check Laplacian
         AddMult_a_AAt (w, dshape, elmat);
         AddMult_a_VWt (w, shape, lshape, elmat);
      }

      // Add to global matrix
      fes.GetElementVDofs (e, vdofs);
      gmat.AddSubMatrix (vdofs, vdofs, elmat, 1);
   }

   // Apply homogeneous essential boundary conditions on entire boundary
   Array<int> ess_dofs;
   fes.GetBoundaryTrueDofs(ess_dofs);
   for (int i=0; i<ess_dofs.Size(); i++)
   {
      gmat.EliminateRowCol(ess_dofs[i], Operator::DiagonalPolicy::DIAG_ZERO);
   }
   gmat.Finalize (1);
   mfem::out<<"Difference between matrices = "<< gmat.MaxNorm()  <<std::endl;
   // Tolerance can be tighter if intorder is increased
   REQUIRE(gmat.MaxNorm() == MFEM_Approx(0.0, 1e-8));

   delete fe_coll;
}

