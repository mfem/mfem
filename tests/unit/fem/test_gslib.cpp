// Copyright (c) 2010-2022, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#include "unit_tests.hpp"
#include "mfem.hpp"

using namespace mfem;
#ifdef MFEM_USE_GSLIB
namespace gslib_test
{

int func_order;

// Scalar function to project
double scalar_func(const Vector &x)
{
   const int dim = x.Size();
   double res = 0.0;
   for (int d = 0; d < dim; d++) { res += std::pow(x(d), func_order); }
   return res;
}

void F_exact(const Vector &p, Vector &F)
{
   F(0) = scalar_func(p);
   for (int i = 1; i < F.Size(); i++) { F(i) = (i+1)*F(0); }
}

enum class Space { H1, L2 };

std::string SpaceName(Space space)
{
   switch (space)
   {
      case Space::H1: return "H1";
      case Space::L2: return "L2";
   }
   return "";
}

TEST_CASE("GSLIBInterpolate", "[GSLIBInterpolate]")
{
   auto space               = GENERATE(Space::H1, Space::L2);
   auto simplex             = GENERATE(true, false);
   int dim                  = GENERATE(2, 3);
   func_order               = GENERATE(1, 2);
   int mesh_order           = GENERATE(1, 2);
   int mesh_node_ordering   = GENERATE(0, 1);
   int point_ordering       = GENERATE(0, 1);
   int ncomp                = GENERATE(1, 2);
   int gf_ordering          = GENERATE(0, 1);
   bool href                = GENERATE(true, false);
   bool pref                = GENERATE(true, false);

   int ne = 4;

   int total_ne = std::pow(ne, dim);
   CAPTURE(space, simplex, dim, func_order, mesh_order, mesh_node_ordering,
           point_ordering, ncomp, gf_ordering, href, pref);

   Mesh mesh;
   if (dim == 2)
   {
      Element::Type type = simplex ? Element::TRIANGLE : Element::QUADRILATERAL;
      mesh = Mesh::MakeCartesian2D(ne, ne, type, 1, 1.0, 1.0);
   }
   else
   {
      Element::Type type = simplex ? Element::TETRAHEDRON : Element::HEXAHEDRON;
      mesh = Mesh::MakeCartesian3D(ne, ne, ne, type, 1.0, 1.0, 1.0);
   }

   if (href || pref) { mesh.EnsureNCMesh(); }
   if (href) { mesh.RandomRefinement(0.5); }

   // Set Mesh NodalFESpace
   H1_FECollection fecm(mesh_order, dim);
   FiniteElementSpace fespacem(&mesh, &fecm, dim, mesh_node_ordering);
   mesh.SetNodalFESpace(&fespacem);

   // Set GridFunction to be interpolated
   FiniteElementCollection *c_fec = nullptr;

   switch (space)
   {
      case Space::H1:
         c_fec = new H1_FECollection(func_order, dim);
         break;
      case Space::L2:
         c_fec = new L2_FECollection(func_order, dim);
         break;
   }

   FiniteElementSpace c_fespace =
      FiniteElementSpace(&mesh, c_fec, ncomp, gf_ordering);
   GridFunction field_vals(&c_fespace);

   VectorFunctionCoefficient F(ncomp, F_exact);
   field_vals.ProjectCoefficient(F);

   // Generate points in the domain
   Vector pos_min, pos_max;
   mesh.GetBoundingBox(pos_min, pos_max, mesh_order);
   const int pts_cnt_1D = 25;
   int pts_cnt = pow(pts_cnt_1D, dim);
   Vector vxyz(pts_cnt * dim);
   if (dim == 2)
   {
      L2_QuadrilateralElement el(pts_cnt_1D - 1, BasisType::ClosedUniform);
      const IntegrationRule &ir = el.GetNodes();
      for (int i = 0; i < ir.GetNPoints(); i++)
      {
         const IntegrationPoint &ip = ir.IntPoint(i);
         if (point_ordering == Ordering::byNODES)
         {
            vxyz(i)           = pos_min(0) + ip.x * (pos_max(0)-pos_min(0));
            vxyz(pts_cnt + i) = pos_min(1) + ip.y * (pos_max(1)-pos_min(1));
         }
         else
         {
            vxyz(i*dim + 0) = pos_min(0) + ip.x * (pos_max(0)-pos_min(0));
            vxyz(i*dim + 1) = pos_min(1) + ip.y * (pos_max(1)-pos_min(1));
         }
      }
   }
   else
   {
      L2_HexahedronElement el(pts_cnt_1D - 1, BasisType::ClosedUniform);
      const IntegrationRule &ir = el.GetNodes();
      for (int i = 0; i < ir.GetNPoints(); i++)
      {
         const IntegrationPoint &ip = ir.IntPoint(i);
         if (point_ordering == Ordering::byNODES)
         {
            vxyz(i)             = pos_min(0) + ip.x * (pos_max(0)-pos_min(0));
            vxyz(pts_cnt + i)   = pos_min(1) + ip.y * (pos_max(1)-pos_min(1));
            vxyz(2*pts_cnt + i) = pos_min(2) + ip.z * (pos_max(2)-pos_min(2));
         }
         else
         {
            vxyz(i*dim + 0) = pos_min(0) + ip.x * (pos_max(0)-pos_min(0));
            vxyz(i*dim + 1) = pos_min(1) + ip.y * (pos_max(1)-pos_min(1));
            vxyz(i*dim + 2) = pos_min(2) + ip.z * (pos_max(2)-pos_min(2));
         }
      }
   }

   // Find and interpolat FE Function values
   Vector interp_vals(pts_cnt*ncomp);
   FindPointsGSLIB finder;
   finder.Setup(mesh);
   finder.SetL2AvgType(FindPointsGSLIB::NONE);
   finder.Interpolate(vxyz, field_vals, interp_vals, point_ordering);
   Array<unsigned int> code_out    = finder.GetCode();
   Vector dist_p_out = finder.GetDist();

   int face_pts = 0, not_found = 0, found = 0;
   double err = 0.0, max_err = 0.0, max_dist = 0.0;
   Vector pos(dim);
   int npt = 0;
   for (int j = 0; j < ncomp; j++)
   {
      for (int i = 0; i < pts_cnt; i++)
      {
         if (code_out[i] < 2)
         {
            if (j == 0) { found++; }
            for (int d = 0; d < dim; d++)
            {
               pos(d) = point_ordering == Ordering::byNODES ?
                        vxyz(d*pts_cnt + i) :
                        vxyz(i*dim + d);
            }
            Vector exact_val(ncomp);
            F_exact(pos, exact_val);
            err = gf_ordering == Ordering::byNODES ?
                  fabs(exact_val(j) - interp_vals[i + j*pts_cnt]) :
                  fabs(exact_val(j) - interp_vals[i*ncomp + j]);
            max_err  = std::max(max_err, err);
            max_dist = std::max(max_dist, dist_p_out(i));
            if (code_out[i] == 1 && j == 0) { face_pts++; }
         }
         else { if (j == 0) { not_found++; } }
         npt++;
      }
   }

   REQUIRE(max_err < 1e-12);
   REQUIRE(max_dist < 1e-10);
   REQUIRE(not_found == 0);

   finder.FreeData();
   delete c_fec;
}

} //namespace_gslib
#endif
