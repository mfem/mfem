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

#include <algorithm>
#include <cmath>

#include "mfem.hpp"
#include "unit_tests.hpp"

using namespace mfem;

namespace
{

void StateRegressionVectorFieldCoefficient(const Vector &x, Vector &v)
{
   const real_t pi = 3.14159265358979323846;
   v.SetSize(2);
   v(0) = 0.45 + 0.42*std::sin(6.0*pi*x(0))*std::cos(4.0*pi*x(1))
          + 0.08*x(1);
   v(1) = -0.20 + 0.36*std::cos(5.0*pi*x(1))*std::sin(3.0*pi*x(0))
          + 0.06*x(0);
}

void CurveMesh(Mesh &mesh)
{
   GridFunction *nodes = mesh.GetNodes();
   Vector &x = *nodes;
   real_t *xd = x.HostReadWrite();
   const int ndofs = x.Size()/2;
   const real_t pi = 3.14159265358979323846;

   for (int i = 0; i < ndofs; i++)
   {
      const real_t xi = xd[i];
      const real_t eta = xd[ndofs + i];
      const real_t bubble = std::sin(pi*xi)*std::sin(pi*eta);
      xd[i] += 0.035*bubble;
      xd[ndofs + i] += 0.020*bubble*(0.5 - xi);
   }
   mesh.NodesUpdated();
}

void MoveNodesForStateRegression(const Vector &nodes, Vector &new_nodes)
{
   new_nodes = nodes;
   real_t *x = new_nodes.HostReadWrite();
   const int ndofs = new_nodes.Size()/2;

   for (int i = 0; i < ndofs; i++)
   {
      const real_t xi = x[i];
      const real_t eta = x[ndofs + i];
      x[i] += 0.16*xi*(1.0 - xi)*(0.25 + eta);
      x[ndofs + i] -= 0.12*eta*(1.0 - eta)*(0.35 + xi);
   }
}

bool IsFinite(const Vector &x)
{
   const real_t *xd = x.HostRead();
   for (int i = 0; i < x.Size(); i++)
   {
      if (!std::isfinite(xd[i])) { return false; }
   }
   return true;
}

void ForceHost(Vector &x)
{
   x.HostReadWrite();
   x.UseDevice(false);
}

void RunAdvectorCGOneRemap(AssemblyLevel al, bool use_device,
                           Vector &initial_field, Vector &result)
{
   const int dim = 2;
   const int mesh_order = 2;
   const int field_order = 2;

   Mesh mesh = Mesh::MakeCartesian2D(3, 2, Element::QUADRILATERAL,
                                     true, 1.0, 1.0);
   mesh.SetCurvature(mesh_order, false, dim, Ordering::byNODES);
   CurveMesh(mesh);
   if (!use_device) { ForceHost(*mesh.GetNodes()); }

   H1_FECollection fec(field_order, dim);
   FiniteElementSpace field_fes(&mesh, &fec, dim, Ordering::byNODES);

   GridFunction field_gf(&field_fes);
   VectorFunctionCoefficient field_coeff(dim,
                                         StateRegressionVectorFieldCoefficient);
   field_gf.ProjectCoefficient(field_coeff);
   if (use_device) { field_gf.UseDevice(true); }

   initial_field = field_gf;
   if (use_device) { initial_field.UseDevice(true); }
   else { ForceHost(initial_field); }

   Vector new_nodes;
   MoveNodesForStateRegression(*mesh.GetNodes(), new_nodes);
   if (use_device) { new_nodes.UseDevice(true); }
   else { ForceHost(new_nodes); }

   result = initial_field;
   if (use_device) { result.UseDevice(true); }
   else { ForceHost(result); }

   AdvectorCG advector(al, 0.20);
   advector.SetSerialMetaInfo(mesh, field_fes);
   advector.SetInitialField(*mesh.GetNodes(), initial_field);
   advector.ComputeAtNewPosition(new_nodes, result, Ordering::byNODES);

   if (!use_device)
   {
      initial_field.HostReadWrite();
      result.HostReadWrite();
      initial_field.UseDevice(false);
      result.UseDevice(false);
   }
}

} // namespace

TEST_CASE("AdvectorCG byNODES device one remap matches host", "[TMOP][GPU]")
{
   Vector gpu_initial, gpu_result;
   RunAdvectorCGOneRemap(AssemblyLevel::PARTIAL, true,
                         gpu_initial, gpu_result);

   Vector reference_initial, reference_result;
   RunAdvectorCGOneRemap(AssemblyLevel::LEGACY, false,
                         reference_initial, reference_result);

   REQUIRE(IsFinite(reference_result));
   REQUIRE(IsFinite(gpu_result));

   Vector reference_change(reference_result);
   reference_change -= reference_initial;

   reference_result.UseDevice(true);
   Vector gpu_minus_reference(reference_result.Size());
   gpu_minus_reference.UseDevice(true);
   subtract(gpu_result, reference_result, gpu_minus_reference);

   CAPTURE(reference_change.Norml2());
   CAPTURE(gpu_minus_reference.Norml2());

   const real_t tolerance =
      1.0e-8*std::max(real_t(1.0), reference_result.Norml2());
   CAPTURE(tolerance);

   REQUIRE(reference_change.Norml2() > 1.0e-8);
   REQUIRE(gpu_minus_reference.Norml2() <= tolerance);
}
