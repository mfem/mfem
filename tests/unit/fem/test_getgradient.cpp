// Copyright (c) 2010-2023, Lawrence Livermore National Security, LLC. Produced
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

namespace test_GetGradient_shared_faces
{

int dim;

double func(const Vector &coord)
{
   double x = coord(0),
          y = (dim > 1) ? coord(1) : 0.0,
          z = (dim > 2) ? coord(2) : 0.0;
   return x * x + y * y + z * z;
}

void func_grad(const Vector &coord, Vector &grad)
{
   grad.SetSize(coord.Size());
   grad(0) = 2.0 * coord(0);
   if (dim > 1) { grad(1) = 2.0 * coord(1); }
   if (dim > 2) { grad(2) = 2.0 * coord(2); }
}

void vec_func(const Vector &coord, Vector &res)
{
   double x = coord(0),
          y = (dim > 1) ? coord(1) : 0.0,
          z = (dim > 2) ? coord(2) : 0.0;

   res(0) = (x + 1.0) * (y + 1.0) * (z + 1.0);
   res(1) = x * x + y * y + z * z;
}

void vec_func_grad(const Vector &coord, DenseMatrix &grad)
{
   grad.SetSize(2, dim);
   double x = coord(0),
          y = (dim > 1) ? coord(1) : 0.0,
          z = (dim> 2) ? coord(2) : 0.0;

   grad(0, 0) = (y + 1.0) * (z + 1.0);
   grad(1, 0) = 2.0 * x;
   if (dim > 1)
   {
      grad(0, 1) = (x + 1.0) * (z + 1.0);
      grad(1, 1) = 2.0 * y;
   }
   if (dim > 2)
   {
      grad(0, 2) = (x + 1.0) * (y + 1.0);
      grad(1, 2) = 2.0 * z;
   }
}

#ifdef MFEM_USE_MPI

TEST_CASE("GetGradient Shared Faces", "[ParGridFunction][Parallel]")
{
   const int fe_order = 3;
   for (dim = 1; dim <= 3; dim++)
   {
      int num_procs;
      MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
      int myid;
      MPI_Comm_rank(MPI_COMM_WORLD, &myid);

      Mesh mesh;
      if (dim == 1)
      {
         mesh = Mesh::MakeCartesian1D(100, 1.0);
      }
      else if (dim == 2)
      {
         mesh = Mesh::LoadFromFile("../../data/star-mixed.mesh");
      }
      else
      {
         mesh = Mesh::LoadFromFile("../../data/fichera-mixed.mesh");
      }
      for (int i = 0; i < 2; i++) { mesh.UniformRefinement(); }
      ParMesh pmesh(MPI_COMM_WORLD, mesh);

      FunctionCoefficient func_coeff(func);
      VectorFunctionCoefficient vec_func_coeff(2, vec_func);
      H1_FECollection fec(fe_order, dim);
      ParFiniteElementSpace pfes(&pmesh, &fec), pfes_vec(&pmesh, &fec, 2);
      ParGridFunction pgf(&pfes), pgf_vec(&pfes_vec);
      pgf.ProjectCoefficient(func_coeff);
      pgf_vec.ProjectCoefficient(vec_func_coeff);
      pgf.ExchangeFaceNbrData();
      pgf_vec.ExchangeFaceNbrData();

      double max_error_grad = 0.0, max_error_grads = 0.0,
             max_error_vec_grad = 0.0;
      for (int f = 0; f < pmesh.GetNSharedFaces(); f++)
      {
         auto tr_f  = pmesh.GetSharedFaceTransformations(f);
         ElementTransformation &tr_e2 = tr_f->GetElement2Transformation();
         const IntegrationRule &ir_f = IntRules.Get(tr_f->GetGeometryType(),
                                                    2*fe_order + 2);
         IntegrationRule ir_e2(ir_f.GetNPoints());
         tr_f->Loc2.Transform(ir_f, ir_e2);

         Vector grad_e2, grad_exact, coord(dim);

         DenseMatrix grads_e2, vec_grad_e2, vec_grad_exact;
         pgf.GetGradients(tr_e2, ir_e2, grads_e2);

         for (int q = 0; q < ir_f.GetNPoints(); q++)
         {
            const IntegrationPoint &ip_f = ir_f.IntPoint(q);
            tr_f->SetIntPoint(&ip_f);
            tr_f->Transform(ip_f, coord);
            func_grad(coord, grad_exact);

            pgf.GetGradient(tr_e2, grad_e2);
            grad_e2 -= grad_exact;
            max_error_grad = fmax(max_error_grad, grad_e2.Norml2());

            grads_e2.GetColumn(q, grad_e2);
            grad_e2 -= grad_exact;
            max_error_grads = fmax(max_error_grads, grad_e2.Norml2());

            pgf_vec.GetVectorGradient(tr_e2, vec_grad_e2);
            vec_func_grad(coord, vec_grad_exact);
            vec_grad_e2 -= vec_grad_exact;
            max_error_vec_grad = fmax(max_error_vec_grad, vec_grad_e2.FNorm2());
         }
      }

      REQUIRE(max_error_grad == MFEM_Approx(0.0));
      REQUIRE(max_error_grads == MFEM_Approx(0.0));
      REQUIRE(max_error_vec_grad == MFEM_Approx(0.0));
   }
}

#endif

}
