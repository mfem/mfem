// Copyright (c) 2010-2021, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.
//
// Sample runs:
//   mpirun -np 4 smoother -p 0
//   mpirun -np 4 smoother -p 1
//   mpirun -np 4 smoother -p 2 -rs 3 -ds 2
//

#include "mfem.hpp"
#include "../common/mfem-common.hpp"

using namespace mfem;
using namespace std;

char vishost[] = "localhost";
int  visport   = 19916;
int  wsize     = 350;

int problem = 0;

double level_set_func(const Vector &coord)
{
   double x = coord(0), y = coord(1);

   if (problem == 0)
   {
      // Smooth circle.
      x -= 0.5; y -= 0.5;
      return 0.25 - sqrt(x*x + y*y + 1e-12);
   }
   if (problem == 1)
   {
      // Vertical line.
      return (x < 0.5) ? 0.5 - x : 5.0 - 10.0 * x;
   }
   if (problem == 2)
   {
      const int num_circ = 3;
      double rad[num_circ] = {0.3, 0.15, 0.2};
      double c[num_circ][2] = { {0.6, 0.6}, {0.3, 0.3}, {0.25, 0.75} };

      const double xc = coord(0), yc = coord(1);

      // circle 0
      double r0 = (xc-c[0][0])*(xc-c[0][0]) + (yc-c[0][1])*(yc-c[0][1]);
      r0 = (r0 > 0) ? std::sqrt(r0) : 0.0;
      if (r0 <= 0.2) { return -1.0; }

      for (int i = 0; i < num_circ; i++)
      {
         double r = (xc-c[i][0])*(xc-c[i][0]) + (yc-c[i][1])*(yc-c[i][1]);
         r = (r > 0) ? std::sqrt(r) : 0.0;
         if (r <= rad[i]) { return 1.0; }
      }

      // rectangle 1
      if (0.7 <= xc && xc <= 0.8 && 0.1 <= yc && yc <= 0.8) { return 1.0; }

      // rectangle 2
      if (0.3 <= xc && xc <= 0.8 && 0.15 <= yc && yc <= 0.2) { return 1.0; }
      return -1.0;
   }
   return 1.0;
}

void MeasureJumps(const ParGridFunction &u, ParGridFunction &jumps, bool vis);
void DiffuseField(ParGridFunction &field, int smooth_steps);

class GradientCompCoefficient : public Coefficient
{
private:
   const ParGridFunction &u;
   const int comp;

public:
   GradientCompCoefficient(const ParGridFunction &u_gf, int c)
      : u(u_gf), comp(c) { }

   virtual double Eval(ElementTransformation &T, const IntegrationPoint &ip);
};

int main (int argc, char *argv[])
{
   // 0. Initialize MPI.
   int num_procs, myid;
   MPI_Init(&argc, &argv);
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
   MPI_Comm_rank(MPI_COMM_WORLD, &myid);

   const char *mesh_file = "../../data/inline-quad.mesh";
   int rs_levels = 2;
   int order = 2;
   int diffuse_steps = 0;

   // 2. Parse command-line options.
   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&problem, "-p", "--problem",
                  "Problem setup to use.");
   args.AddOption(&rs_levels, "-rs", "--refine-serial",
                  "Number of times to refine the mesh uniformly in serial.");
   args.AddOption(&order, "-o", "--order",
                  "Polynomial degree of mesh finite element space.");
   args.AddOption(&diffuse_steps, "-ds", "--diffuse-steps",
                  "Diffusion steps.");
   args.Parse();
   if (!args.Good())
   {
      if (myid == 0) { args.PrintUsage(cout); }
      return 1;
   }
   if (myid == 0) { args.PrintOptions(cout); }

   Mesh *mesh = new Mesh(mesh_file, 1, 1, false);
   for (int lev = 0; lev < rs_levels; lev++) { mesh->UniformRefinement(); }
   ParMesh pmesh(MPI_COMM_WORLD, *mesh);
   delete mesh;
   const int dim = pmesh.Dimension();

   H1_FECollection fec_H1(order, dim);
   ParFiniteElementSpace pfes_H1(&pmesh, &fec_H1);
   ParGridFunction u(&pfes_H1);

   FunctionCoefficient ls_coeff(level_set_func);
   u.ProjectCoefficient(ls_coeff);
   DiffuseField(u, diffuse_steps);
   socketstream sock_u;
   common::VisualizeField(sock_u, vishost, visport, u, "u",
                          0, 0, wsize, wsize, "Rjmc***");

   L2_FECollection fec_L2(order, dim);
   ParFiniteElementSpace pfes_L2(&pmesh, &fec_L2);
   ParGridFunction du_comp(&pfes_L2), dudu_comp(&pfes_L2);

   H1_FECollection fec_lin_H1(1, dim);
   ParFiniteElementSpace pfes_lin_H1(&pmesh, &fec_lin_H1);
   ParGridFunction du_jumps(&pfes_lin_H1), dudu_jumps(&pfes_lin_H1),
                   comp_jumps(&pfes_lin_H1);
   du_jumps = 0.0;
   dudu_jumps = 0.0;
   for (int d = 0; d < dim; d++)
   {
      GradientCompCoefficient du_comp_coeff(u, d);
      du_comp.ProjectCoefficient(du_comp_coeff);
      for (int c = 0; c < dim; c++)
      {
         GradientCompCoefficient d2u_comp_coeff(du_comp, c);
         dudu_comp.ProjectCoefficient(d2u_comp_coeff);
         MeasureJumps(dudu_comp, comp_jumps, false);
         for (int i = 0; i < du_jumps.Size(); i++)
         {
            dudu_jumps(i) = fmax(dudu_jumps(i), comp_jumps(i));
         }
      }
      MeasureJumps(du_comp, comp_jumps, false);
      for (int i = 0; i < du_jumps.Size(); i++)
      {
         du_jumps(i) = fmax(du_jumps(i), comp_jumps(i));
      }
   }
   socketstream sock_j, sock_jj;
   common::VisualizeField(sock_j, vishost, visport, du_jumps,
                          "smoo indicator grad",
                          wsize, 0, wsize, wsize, "Rjmc***");
   common::VisualizeField(sock_jj, vishost, visport, dudu_jumps,
                          "smoo indicator hess",
                          2*wsize, 0, wsize, wsize, "Rjmc***");

   ConstantCoefficient zero(0.0);
   double norm = du_jumps.ComputeL1Error(zero);
   if (myid == 0)
   {
      std::cout << setprecision(12) << "L1 norm = " << norm << std::endl;
   }

   MPI_Finalize();
   return 0;
}

void MeasureJumps(const ParGridFunction &u, ParGridFunction &jumps, bool vis)
{
   ParFiniteElementSpace &pfes_L2 = *u.ParFESpace();
   const int dim   = pfes_L2.GetMesh()->Dimension();
   const int order = pfes_L2.FEColl()->GetOrder();

   // Normalize to [0, 1].
   ParGridFunction u_n(u);
   double u_min = u_n.Min();
   MPI_Allreduce(MPI_IN_PLACE, &u_min, 1, MPI_DOUBLE, MPI_MIN,
                 u_n.ParFESpace()->GetComm());
   u_n += fabs(u_min);
   double u_max = u_n.Max();
   MPI_Allreduce(MPI_IN_PLACE, &u_max, 1, MPI_DOUBLE, MPI_MAX,
                 u_n.ParFESpace()->GetComm());
   u_n /= u_max;

   if (vis)
   {
      socketstream sock_un;
      common::VisualizeField(sock_un, vishost, visport, u_n, "u normalized",
                             wsize, wsize, wsize, wsize, "Rjmc**");
   }

   // Form min/max at each CG dof, considering element overlaps.
   H1_FECollection fec_H1(order, dim);
   ParFiniteElementSpace pfes_H1(pfes_L2.GetParMesh(), &fec_H1);
   ParGridFunction u_min_H1(&pfes_H1), u_max_H1(&pfes_H1);
   u_min_H1 =   std::numeric_limits<double>::infinity();
   u_max_H1 = - std::numeric_limits<double>::infinity();
   const int NE = pfes_H1.GetNE();
   Array<int> dofsCG;
   const TensorBasisElement *fe_tensor =
      dynamic_cast<const TensorBasisElement *>(pfes_H1.GetFE(0));
   MFEM_VERIFY(fe_tensor, "TODO - implement for triangles");
   const Array<int> &dof_map = fe_tensor->GetDofMap();
   const int ndofs = dof_map.Size();
   for (int k = 0; k < NE; k++)
   {
      pfes_H1.GetElementDofs(k, dofsCG);
      for (int i = 0; i < ndofs; i++)
      {
         u_min_H1(dofsCG[dof_map[i]]) = fmin(u_min_H1(dofsCG[dof_map[i]]),
                                             u_n(k*ndofs + i));
         u_max_H1(dofsCG[dof_map[i]]) = fmax(u_max_H1(dofsCG[dof_map[i]]),
                                             u_n(k*ndofs + i));
      }
   }
   // MPI neighbors.
   GroupCommunicator &gcomm = pfes_H1.GroupComm();
   Array<double> minvals(u_min_H1.GetData(), u_min_H1.Size()),
                 maxvals(u_max_H1.GetData(), u_max_H1.Size());
   gcomm.Reduce<double>(minvals, GroupCommunicator::Min);
   gcomm.Bcast(minvals);
   gcomm.Reduce<double>(maxvals, GroupCommunicator::Max);
   gcomm.Bcast(maxvals);

   // Compute jumps (and reuse the min H1 function).
   ParGridFunction &u_jump_H1 = u_min_H1;
   for (int i = 0; i < u_jump_H1.Size(); i++)
   {
      u_jump_H1(i) = fabs(u_max_H1(i) - u_min_H1(i));
   }
   if (vis)
   {
      socketstream sock_j;
      common::VisualizeField(sock_j, vishost, visport, u_jump_H1, "u jumps HO",
                             2*wsize, wsize, wsize, wsize, "Rjmc**");
   }

   // Project the jumps to Q1.
   jumps.ProjectGridFunction(u_jump_H1);
   for (int i = 0; i < jumps.Size(); i++) { jumps(i) = fmax(jumps(i), 0.0); }
   Vector tv(jumps.ParFESpace()->GetTrueVSize());
   jumps.ParallelAverage(tv);
   jumps.Distribute(tv);
}

void DiffuseField(ParGridFunction &field, int smooth_steps)
{
   // Setup the Laplacian operator
   ParBilinearForm *Lap = new ParBilinearForm(field.ParFESpace());
   Lap->AddDomainIntegrator(new DiffusionIntegrator());
   Lap->Assemble();
   Lap->Finalize();
   HypreParMatrix *A = Lap->ParallelAssemble();

   HypreSmoother *S = new HypreSmoother(*A,0,smooth_steps);
   S->iterative_mode = true;

   Vector tmp(A->Width());
   field.SetTrueVector();
   Vector fieldtrue = field.GetTrueVector();
   tmp = 0.0;
   S->Mult(tmp, fieldtrue);

   field.Distribute(fieldtrue);

   delete S;
   delete Lap;
}

double GradientCompCoefficient::Eval(ElementTransformation &T,
                                     const IntegrationPoint &ip)
{
   Vector grad_u(T.GetDimension());
   u.GetGradient(T, grad_u);
   return grad_u(comp);
}
