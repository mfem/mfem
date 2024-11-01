// Copyright (c) 2010-2024, Lawrence Livermore National Security, LLC. Produced
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

TEST_CASE("GSLIBInterpolate", "[GSLIBInterpolate][GSLIB]")
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

   CAPTURE(space, simplex, dim, func_order, mesh_order, mesh_node_ordering,
           point_ordering, ncomp, gf_ordering, href, pref);

   if (ncomp == 1 && gf_ordering == 1)
   {
      return;
   }

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
   const int pts_cnt_1D = 5;
   int pts_cnt = pow(pts_cnt_1D, dim);
   Vector vxyz(pts_cnt * dim);
   NodalTensorFiniteElement *el = NULL;
   if (dim == 2)
   {
      el = new L2_QuadrilateralElement(pts_cnt_1D-1,BasisType::ClosedUniform);
   }
   else
   {
      el = new L2_HexahedronElement(pts_cnt_1D - 1, BasisType::ClosedUniform);
   }
   const IntegrationRule &ir = el->GetNodes();
   for (int i = 0; i < ir.GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir.IntPoint(i);
      if (point_ordering == Ordering::byNODES)
      {
         vxyz(i)           = pos_min(0) + ip.x * (pos_max(0)-pos_min(0));
         vxyz(pts_cnt + i) = pos_min(1) + ip.y * (pos_max(1)-pos_min(1));
         if (dim == 3)
         {
            vxyz(2*pts_cnt + i) = pos_min(2) + ip.z * (pos_max(2)-pos_min(2));
         }
      }
      else
      {
         vxyz(i*dim + 0) = pos_min(0) + ip.x * (pos_max(0)-pos_min(0));
         vxyz(i*dim + 1) = pos_min(1) + ip.y * (pos_max(1)-pos_min(1));
         if (dim == 3)
         {
            vxyz(i*dim + 2) = pos_min(2) + ip.z * (pos_max(2)-pos_min(2));
         }
      }
   }
   delete el;

   // Find and interpolate FE Function values
   Vector interp_vals(pts_cnt*ncomp);
   FindPointsGSLIB finder;
   finder.Setup(mesh);
   finder.SetL2AvgType(FindPointsGSLIB::NONE);
   finder.Interpolate(vxyz, field_vals, interp_vals, point_ordering);
   Array<unsigned int> code_out    = finder.GetCode();
   Vector dist_p_out = finder.GetDist();

   int not_found = 0;
   double err = 0.0, max_err = 0.0, max_dist = 0.0;
   Vector pos(dim);

   for (int i = 0; i < pts_cnt; i++)
   {
      max_dist = std::max(max_dist, dist_p_out(i));
      for (int d = 0; d < dim; d++)
      {
         pos(d) = point_ordering == Ordering::byNODES ?
                  vxyz(d*pts_cnt + i) :
                  vxyz(i*dim + d);
      }
      Vector exact_val(ncomp);
      F_exact(pos, exact_val);
      for (int j = 0; j < ncomp; j++)
      {
         if (code_out[i] < 2)
         {
            err = gf_ordering == Ordering::byNODES ?
                  fabs(exact_val(j) - interp_vals[i + j*pts_cnt]) :
                  fabs(exact_val(j) - interp_vals[i*ncomp + j]);
            max_err  = std::max(max_err, err);
         }
         else
         {
            if (j == 0) { not_found++; }
         }
      }
   }

   REQUIRE(max_err < 1e-12);
   REQUIRE(max_dist < 1e-10);
   REQUIRE(not_found == 0);

   finder.FreeData();
   delete c_fec;
}

// Generates meshes with different element types, followed by points at
// element faces and interior, and finally checks to see if these points are
// correctly detected at element boundary or not.
TEST_CASE("GSLIBFindAtElementBoundary",
          "[GSLIBFindAtElementBoundary][GSLIB]")
{
   int dim  = GENERATE(2, 3);
   CAPTURE(dim);
   int nex = 4;
   int mesh_order = 4;
   int l2_order = 4;

   int netype = dim == 2 ? 2 : 4; // 2 element types in 2D, 4 in 3D.
   int estart = dim == 2 ? 2 : 4; // starts at index 2 in 2D, 4 in 3D

   for (int et = estart; et < estart+netype; et++)
   {
      // H1 - order 1, L2 - order 0 for pyramids
      if (et == 7)
      {
         mesh_order = 1;
         l2_order = 0;
      }
      Mesh mesh;
      if (dim == 2)
      {
         mesh = Mesh::MakeCartesian2D(nex, nex, (Element::Type)et);
      }
      else
      {
         mesh = Mesh::MakeCartesian3D(nex, nex, nex, (Element::Type)et);
      }

      mesh.SetCurvature(mesh_order);
      const FiniteElementSpace *n_fespace = mesh.GetNodalFESpace();
      const GridFunction *nodes = mesh.GetNodes();

      Array<double> xyz;

      // Generate points on each element's face/edge
      for (int e = 0; e < mesh.GetNE(); e++)
      {
         Array<int> faces,ori;
         if (dim == 2)
         {
            mesh.GetElementEdges(e, faces, ori);
         }
         else
         {
            mesh.GetElementFaces(e, faces, ori);
         }

         for (int f = 0; f < faces.Size(); f++)
         {
            const FiniteElement *fe = n_fespace->GetFaceElement(faces[f]);
            const IntegrationRule ir = fe->GetNodes();

            DenseMatrix vals;
            DenseMatrix tr;
            nodes->GetFaceVectorValues(faces[f], 0, ir, vals, tr);
            xyz.Append(vals.GetData(), vals.Height()*vals.Width());
         }
      }

      int nptface = xyz.Size()/dim;

      // Generate points inside each element
      FiniteElementCollection *l2_fec = new L2_FECollection(l2_order, dim);
      FiniteElementSpace l2_fespace =
         FiniteElementSpace(&mesh, l2_fec, 1);
      DenseMatrix vals;
      DenseMatrix tr;
      for (int e = 0; e < mesh.GetNE(); e++)
      {
         const FiniteElement *fe = l2_fespace.GetFE(e);
         const IntegrationRule ir = fe->GetNodes();

         nodes->GetVectorValues(e, ir, vals, tr);
         xyz.Append(vals.GetData(), vals.Height()*vals.Width());
      }

      Vector xyzv(xyz.GetData(), xyz.Size());
      int npt = xyzv.Size()/dim;

      FindPointsGSLIB finder;
      finder.Setup(mesh);
      finder.FindPoints(xyzv, Ordering::byVDIM);
      Array<unsigned int> code_out    = finder.GetCode();
      unsigned int cmin = 5,
                   cmax = 0;
      for (int i = 0; i < nptface; i++)
      {
         cmin = std::min(code_out[i], cmin);
         cmax = std::max(code_out[i], cmax);
      }
      REQUIRE((cmin == 1 && cmax == 1)); // should be found on element boundary

      cmin = 5;
      cmax = 0;
      for (int i = nptface; i < npt; i++)
      {
         cmin = std::min(code_out[i], cmin);
         cmax = std::max(code_out[i], cmax);
      }
      REQUIRE((cmin == 0 && cmax == 0)); // should be found inside element
      delete l2_fec;
   }
}

// Generate a 4x4 Quad/Hex Mesh and interpolate point in the center of domain
// at element boundary. This tests L2 projection with and without averaging.
TEST_CASE("GSLIBInterpolateL2ElementBoundary",
          "[GSLIBInterpolateL2ElementBoundary][GSLIB]")
{
   int dim  = GENERATE(2, 3);
   CAPTURE(dim);

   int nex = 4;
   int mesh_order = 2;
   Mesh mesh;
   if (dim == 2)
   {
      mesh = Mesh::MakeCartesian2D(nex, nex, Element::QUADRILATERAL);
   }
   else
   {
      mesh = Mesh::MakeCartesian3D(nex, nex, nex, Element::HEXAHEDRON);
   }

   mesh.SetCurvature(mesh_order);

   // Set GridFunction to be interpolated
   int func_order = 3;
   FiniteElementCollection *c_fec = new L2_FECollection(func_order, dim);
   FiniteElementSpace c_fespace =
      FiniteElementSpace(&mesh, c_fec, 1);
   GridFunction field_vals(&c_fespace);
   Array<int> dofs;
   double leftval = 1.0;
   double rightval = 3.0;
   for (int e = 0; e < mesh.GetNE(); e++)
   {
      Vector center(dim);
      mesh.GetElementCenter(e, center);
      double val_to_set = center(0) < 0.5 ? leftval : rightval;
      c_fespace.GetElementDofs(e, dofs);
      Vector vals(dofs.Size());
      vals = val_to_set;
      field_vals.SetSubVector(dofs, vals);
   }

   int npt = 1;
   Vector xyz(npt*dim);
   xyz = 0.0;
   xyz(0) = 0.5;

   // Find and interpolate FE Function values
   Vector interp_vals(npt);
   FindPointsGSLIB finder;
   finder.Setup(mesh);
   finder.SetL2AvgType(FindPointsGSLIB::NONE);
   finder.Interpolate(xyz, field_vals, interp_vals, 1);
   Array<unsigned int> code_out    = finder.GetCode();

   // This point should have been found on element border. But the interpolated
   // value will come from either of the elements that share this edge/face.
   REQUIRE(code_out[0] == 1);
   REQUIRE((interp_vals(0) == MFEM_Approx(leftval) ||
            interp_vals(0) == MFEM_Approx(rightval)));

   // Interpolated value should now be average of solution coming from
   // adjacent elements.
   finder.SetL2AvgType(FindPointsGSLIB::ARITHMETIC);
   finder.Interpolate(xyz, field_vals, interp_vals, 1);
   REQUIRE(interp_vals(0) == MFEM_Approx(0.5*(leftval+rightval)));

   finder.FreeData();
   delete c_fec;
}

#ifdef MFEM_USE_MPI
// Custom interpolation procedure with gslib
TEST_CASE("GSLIBCustomInterpolation",
          "[GSLIBCustomInterpolation][Parallel][GSLIB]")
{
   int myid;
   MPI_Comm_rank(MPI_COMM_WORLD, &myid);

   int dim      = GENERATE(2, 3);
   bool simplex = GENERATE(true, false);

   CAPTURE(dim, simplex);

   int nex = 4;
   int mesh_order = 2;
   Mesh mesh;
   if (dim == 2)
   {
      Element::Type type = simplex ? Element::TRIANGLE : Element::QUADRILATERAL;
      mesh = Mesh::MakeCartesian2D(nex, nex, type);
   }
   else
   {
      Element::Type type = simplex ? Element::TETRAHEDRON : Element::HEXAHEDRON;
      mesh = Mesh::MakeCartesian3D(nex, nex, nex, type);
   }

   mesh.SetCurvature(mesh_order);
   ParMesh pmesh(MPI_COMM_WORLD, mesh);

   // f(x,y,z) = x^2 + y^2 + z^2
   auto func = [](const Vector &x)
   {
      const int dim = x.Size();
      double res = 0.0;
      for (int d = 0; d < dim; d++) { res += std::pow(x(d), 2); }
      return res;
   };

   // \nabla f(x,y,z) = [2*x,2*y,2*z]
   auto func_grad = [](const Vector &x, Vector &p)
   {
      const int dim = x.Size();
      p.SetSize(dim);
      for (int d = 0; d < dim; d++) { p(d) = 2.0*x(d); }
   };

   // Set GridFunction to be interpolated
   int func_order = 3;
   H1_FECollection c_fec(func_order, dim);
   FiniteElementSpace c_fespace(&pmesh, &c_fec, 1);
   GridFunction field_vals(&c_fespace);

   FunctionCoefficient f(func);
   field_vals.ProjectCoefficient(f);

   // Generate randomized points in [0, 1]^D. Assume ordering by VDIM.
   int npt = 101;
   Vector xyz(npt*dim);
   xyz.Randomize(myid + 1);
   if (myid == 1) // zero out # of points on rank 1
   {
      xyz.SetSize(0);
   }

   // Find points on the ParMesh
   Vector interp_vals(npt);
   FindPointsGSLIB finder;
   finder.Setup(pmesh);
   finder.FindPoints(xyz, Ordering::byVDIM);

   /** Interpolate gradient using custom interpolation procedure. */
   // We first send information to MPI ranks that own the element corresponding
   // to each point.
   Array<unsigned int> recv_elem, recv_code;
   Vector recv_rst;
   finder.DistributePointInfoToOwningMPIRanks(recv_elem, recv_rst, recv_code);
   int npt_recv = recv_elem.Size();
   // Compute gradient locally
   Vector grad(npt_recv*dim);
   for (int i = 0; i < npt_recv; i++)
   {
      const int e = recv_elem[i];

      IntegrationPoint ip;
      if (dim == 2)
      {
         ip.Set2(recv_rst(dim*i + 0),recv_rst(dim*i + 1));
      }
      else
      {
         ip.Set3(recv_rst(dim*i + 0),recv_rst(dim*i + 1),
                 recv_rst(dim*i + 2));
      }
      ElementTransformation *Tr = c_fespace.GetElementTransformation(e);
      Tr->SetIntPoint(&ip);

      Vector gradloc(grad.GetData()+i*dim,dim);
      field_vals.GetGradient(*Tr, gradloc);
   }

   // Send the computed gradient back to the ranks that requested it.
   Vector recv_grad;
   finder.DistributeInterpolatedValues(grad, dim, Ordering::byVDIM, recv_grad);

   // Check if the received gradient matched analytic gradient.
   for (int i = 0; i < npt && myid == 0; i++)
   {
      Vector x(xyz.GetData()+i*dim,dim);
      Vector grad_exact(dim);
      func_grad(x, grad_exact);

      Vector recv_grad_i(recv_grad.GetData()+i*dim,dim);

      for (int d = 0; d < dim; d++)
      {
         REQUIRE(grad_exact(d) == Approx(recv_grad(i*dim + d)));
      }
   }

   finder.FreeData();
}

TEST_CASE("GSLIBGSOP", "[GSLIBGSOP][Parallel][GSLIB]")
{
   int myid;
   MPI_Comm_rank(MPI_COMM_WORLD, &myid);

   int nlen = 5 + rand() % 1000;
   MPI_Allreduce(MPI_IN_PLACE, &nlen, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);

   Array<long long> ids(nlen);
   Vector vals(nlen);
   vals.Randomize(myid+1);

   // Force minimum values based on the identifier for deterministic behavior
   // on rank 0 and randomize the identifier on other ranks.
   if (myid == 0)
   {
      for (int i = 0; i < nlen; i++)
      {
         ids[i] = i+1;
         vals(i) = -ids[i];
      }
   }
   else
   {
      for (int i = 0; i < nlen; i++)
      {
         int num = rand() % nlen + 1;
         ids[i] = num;
      }
   }

   // Test GSOp::MIN
   GSOPGSLIB gs = GSOPGSLIB(MPI_COMM_WORLD, ids);
   gs.GS(vals, GSOPGSLIB::GSOp::MIN);

   // Check for minimum value
   for (int i = 0; i < nlen; i++)
   {
      int id = ids[i];
      REQUIRE(vals(i) == -1.0*id);
   }

   // Test GSOp::ADD
   // Set all values to 0 except on rank 0, and then add them.
   if (myid != 0) { vals = 0.0; }
   gs.GS(vals, GSOPGSLIB::GSOp::ADD);

   // Check for added value to match what was originally set on rank 0.
   for (int i = 0; i < nlen; i++)
   {
      int id = ids[i];
      REQUIRE(vals(i) == -1.0*id);
   }

   // Test GSOp::MUL
   // Randomize values on all ranks except rank 0 such that they are positive.
   if (myid != 0) { vals.Randomize(); }
   gs.GS(vals, GSOPGSLIB::GSOp::MUL);

   // Check for multipled values to be negative
   for (int i = 0; i < nlen; i++)
   {
      REQUIRE(vals(i) < 0);
   }
}
#endif // MFEM_USE_MPI

} //namespace_gslib
#endif
