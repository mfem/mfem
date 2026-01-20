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
#include "catch.hpp"
#include "unit_tests.hpp"

#include <iostream>
#include <cmath>
#include <algorithm>

using namespace mfem;


void FillWithRandomNumbers(Vector &x, real_t a, real_t b)
{
   std::random_device rd;
   std::mt19937 gen(rd());
   std::uniform_real_distribution<> dis(a, b);
   for (int i = 0; i < x.Size(); i++)
   {
      x[i] = dis(gen);
   }
}

void TestDivShape(Mesh *mesh, int order)
{
   const int dim = mesh->Dimension();
   NURBS_HDivFECollection vfe_coll(order,dim);
   FiniteElementSpace fes(mesh, new NURBSExtension(mesh->NURBSext, order),
                          &vfe_coll);

   Vector div_shape;
   DenseTensor dvshape;
   DofTransformation doftrans;
   ElementTransformation *eltrans;
   for (int e = 0; e < fes.GetNE(); e++)
   {
      int dof = fes.GetFE(e)->GetDof();

      div_shape.SetSize(dof);
      dvshape.SetSize(dof,dim,dim);

      eltrans = fes.GetElementTransformation (e);

      int intorder = fes.GetFE(e)->GetOrder();
      const IntegrationRule *ir = &IntRules.Get(fes.GetFE(e)->GetGeomType(),
                                                intorder);

      for (int i = 0; i < ir->GetNPoints(); i++)
      {
         const IntegrationPoint &ip = ir->IntPoint(i);
         eltrans->SetIntPoint(&ip);
         fes.GetFE(e)->CalcPhysDVShape(*eltrans,dvshape);
         fes.GetFE(e)->CalcPhysDivShape(*eltrans, div_shape);

         for (int d=0; d<dof; d++)
         {
            double div = 0;
            for (int j=0; j<dim; j++)
            {
               div += dvshape(d,j,j);
            }
            REQUIRE(fabs(div - div_shape[d]) < 1e-10);
         }
      }
   }
   std::cout <<"End HDiv check: "<<dim<<" "<<order<<std::endl;
   std::cout<< std::flush;
}

void TestCurlShape(Mesh *mesh, int order)
{
   const int dim = mesh->Dimension();
   NURBS_HCurlFECollection vfe_coll(order,dim);
   FiniteElementSpace fes(mesh, new NURBSExtension(mesh->NURBSext, order),
                          &vfe_coll);

   DenseMatrix curl_shape;
   DenseTensor dvshape;
   DofTransformation doftrans;
   ElementTransformation *eltrans;
   Vector curl((dim*(dim-1))/2);
   for (int e = 0; e < fes.GetNE(); e++)
   {
      int dof = fes.GetFE(e)->GetDof();

      curl_shape.SetSize(dof,dim);
      dvshape.SetSize(dof,dim,dim);

      eltrans = fes.GetElementTransformation (e);

      int intorder = fes.GetFE(e)->GetOrder();
      const IntegrationRule *ir = &IntRules.Get(fes.GetFE(e)->GetGeomType(),
                                                intorder);

      for (int i = 0; i < ir->GetNPoints(); i++)
      {
         const IntegrationPoint &ip = ir->IntPoint(i);
         eltrans->SetIntPoint(&ip);
         fes.GetFE(e)->CalcPhysDVShape(*eltrans, dvshape);
         fes.GetFE(e)->CalcPhysCurlShape(*eltrans, curl_shape);

         if (dim == 2)
         {
            for (int d=0; d<dof; d++)
            {
               curl[0] =  -dvshape(d, 0, 1) + dvshape(d, 1, 0);
               REQUIRE(curl[0] - curl_shape(d,0) == MFEM_Approx(0.0));
            }
         }
         else if (dim == 3)
         {
            for (int d=0; d<dof; d++)
            {
               curl[0] = dvshape(d, 2, 1) - dvshape(d, 1, 2);
               curl[1] = dvshape(d, 0, 2) - dvshape(d, 2, 0);
               curl[2] = dvshape(d, 1, 0) - dvshape(d, 0, 1);
               REQUIRE(curl[0] - curl_shape(d,0) == MFEM_Approx(0.0));
               REQUIRE(curl[1] - curl_shape(d,1) == MFEM_Approx(0.0));
               REQUIRE(curl[2] - curl_shape(d,2) == MFEM_Approx(0.0));
            }
         }
      }
   }
   std::cout <<"End HCurl check: "<<dim<<" "<<order<<std::endl;
   std::cout<< std::flush;
}

TEST_CASE("CalcDVShape","[2D/3D][HDiv/HCurl][Order]"
         )
{
   std::string meshName = GENERATE("square-nurbs.mesh", "cube-nurbs.mesh");
   int order = GENERATE(1,2,3,4,5);

   Mesh mesh("../../data/" + meshName, 1, 1);
   mesh.UniformRefinement();

   Vector nodes, dx;
   mesh.GetNodes(nodes);

   dx.SetSize(nodes.Size());
   FillWithRandomNumbers(dx, -0.1, 0.1);
   nodes += dx;

   mesh.SetNodes(nodes);

   TestDivShape(&mesh, order);
   TestCurlShape(&mesh, order);
}
