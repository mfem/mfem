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

#include "fem/eltrans.hpp"
#include "fem/gslib.hpp"
#include "fem/intrules.hpp"
#include "lib/navier_solver.hpp"
#include "kernels/contact_qoi_evaluator.hpp"
#include <fstream>

using namespace mfem;
using namespace navier;

void print_matrix(mfem::DenseMatrix m)
{
   std::cout << std::scientific;
   std::cout << "{";
   for (int i = 0; i < m.NumRows(); i++)
   {
      std::cout << "{";
      for (int j = 0; j < m.NumCols(); j++)
      {
         std::cout << m(i, j);
         if (j < m.NumCols() - 1)
         {
            std::cout << ", ";
         }
      }
      if (i < m.NumRows() - 1)
      {
         std::cout << "}, ";
      }
      else
      {
         std::cout << "}";
      }
   }
   std::cout << "}\n";
   std::cout << std::fixed;
}

void print_vector(mfem::Vector v)
{
   std::cout << "{";
   for (int i = 0; i < v.Size(); i++)
   {
      std::cout << v(i);
      if (i < v.Size() - 1)
      {
         std::cout << ", ";
      }
   }
   std::cout << "}\n";
}


void analytical_velocity(const Vector &coords, Vector &u)
{
   const double x = coords(0);
   const double y = coords(1);

   u(0) = x * y;
   u(1) = 2 * x * y;
}

double analytical_pressure(const Vector &coords)
{
   const double x = coords(0);
   const double y = coords(1);

   return x+y;
}

void analytical_stress(const Vector &coords, Vector &sigma_ne)
{
   const double x = coords(0);
   const double y = coords(1);

   //   y - x  x + 2y
   //  x + 2y  3x - y

   DenseMatrix sigma(2,2);
   sigma(0, 0) = y - x;
   sigma(1, 0) = x + 2*y;
   sigma(0, 1) = x + 2*y;
   sigma(1, 1) = 3*x - y;

   Vector ne(2);
   ne(0) = 1.0;
   ne(1) = 0.0;

   sigma_ne.SetSize(2);
   sigma.Mult(ne, sigma_ne);
}

DenseMatrix* analytical_stress_mat(const Vector &coords)
{
   const double x = coords(0);
   const double y = coords(1);

   //   y - x  x + 2y
   //  x + 2y  3x - y

   auto sigma = new DenseMatrix(2, 2);
   (*sigma)(0, 0) = y - x;
   (*sigma)(1, 0) = x + 2*y;
   (*sigma)(0, 1) = x + 2*y;
   (*sigma)(1, 1) = 3*x - y;

   return sigma;
}

int main(int argc, char *argv[])
{
   Mpi::Init(argc, argv);
   int num_procs = Mpi::WorldSize();
   int myid = Mpi::WorldRank();
   Hypre::Init();

   int polynomial_order = 2;
   int refinements = 0;
   const char *mesh_file = "two_domain_test.mesh";

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh", "Mesh file to use.");
   args.AddOption(&refinements, "-r", "--ref", "");
   args.AddOption(&polynomial_order, "-o", "--order", "");
   args.ParseCheck(out);

   Mesh mesh = Mesh::LoadFromFile(mesh_file);
   mesh.EnsureNodes();

   for (int i = 0; i < refinements; i++)
   {
      mesh.UniformRefinement();
   }

   const int dim = mesh.Dimension();
   const double density = 1.0;

   out << "mesh dimension " << dim << "\n";

   Array<int> left_domain(mesh.attributes.Max());
   left_domain = 0;
   left_domain[0] = 1;

   Array<int> right_domain(mesh.attributes.Max());
   right_domain = 0;
   right_domain[1] = 1;

   ParMesh pmesh(MPI_COMM_WORLD, mesh);

   ParSubMesh fluid_mesh = ParSubMesh::CreateFromDomain(pmesh, left_domain);
   NavierSolver navier(&fluid_mesh, polynomial_order, 1.0);

   auto ugf = navier.GetCurrentVelocity();
   VectorFunctionCoefficient analytical_velocity_coeff(2, analytical_velocity);
   ugf->ProjectCoefficient(analytical_velocity_coeff);

   auto pgf = navier.GetCurrentPressure();
   FunctionCoefficient analytical_pressure_coeff(analytical_pressure);
   pgf->ProjectCoefficient(analytical_pressure_coeff);

   auto mugf = navier.GetVariableViscosity();

   ParSubMesh solid_mesh = ParSubMesh::CreateFromDomain(pmesh, right_domain);

   Array<int> interface_marker_solid(solid_mesh.bdr_attributes.Max());
   interface_marker_solid = 0;
   interface_marker_solid[2] = 1;

   auto &primary_mesh = solid_mesh;
   auto &secondary_mesh = fluid_mesh;
   auto &ir_face = navier.gll_ir_face;

   // Compute the requested QoI on the secondary mesh
   Vector qoi_mem;

   auto qoi_func = [&](ElementTransformation &tr, int pt_idx, int num_pts)
   {
      auto qoi = Reshape(qoi_mem.ReadWrite(), dim, dim, num_pts);
      auto A = Reshape(&qoi(0, 0, pt_idx), dim, dim);
      const double p = pgf->GetValue(tr);
      const double mu = mugf->GetValue(tr);
      DenseMatrix dudx(dim, dim);
      ugf->GetVectorGradient(tr, dudx);

      // (-p * I + nu (grad(u) + grad(u)^T))
      for (int i = 0; i < dim; i++)
      {
         for (int j = 0; j < dim; j++)
         {
            A(i, j) = -p * (i == j) + density * mu * (dudx(i, j) + dudx(j, i));
         }
      }

      Vector transformed_ip(2);
      tr.Transform(tr.GetIntPoint(), transformed_ip);

      out << "         el_id: " << tr.ElementNo << "\n";
      out << "transformed ip: ("
          << transformed_ip(0) << ","
          << transformed_ip(1) << ")\n";

      DenseMatrix B(Reshape(&qoi(0, 0, pt_idx), dim, dim), dim, dim);
      out << "\n";
      out << "computed\n";
      print_matrix(B);

      DenseMatrix *C = analytical_stress_mat(transformed_ip);
      out << "analytical\n";
      print_matrix(*C);
      out << "\n";
      delete C;
   };

   ContactQoiEvaluator(primary_mesh, secondary_mesh, interface_marker_solid,
                       ir_face, qoi_func, qoi_mem, dim * dim);

   char vishost[] = "localhost";
   int  visport   = 19916;
   socketstream sol_sock(vishost, visport);
   sol_sock << "parallel " << num_procs << " " << myid << "\n";
   sol_sock.precision(8);
   sol_sock << "mesh\n" << fluid_mesh << std::flush;

   return 0;
}
