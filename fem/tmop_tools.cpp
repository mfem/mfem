// Copyright (c) 2010-2020, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#include "tmop_tools.hpp"
#include "nonlinearform.hpp"
#include "pnonlinearform.hpp"
#include "../general/osockstream.hpp"

namespace mfem
{

using namespace mfem;

void AdvectorCG::SetInitialField(const Vector &init_nodes,
                                 const Vector &init_field)
{
   nodes0 = init_nodes;
   field0 = init_field;
}

void AdvectorCG::ComputeAtNewPosition(const Vector &new_nodes,
                                      Vector &new_field)
{
#if defined(MFEM_DEBUG) || defined(MFEM_USE_MPI)
   int myid = 0;
#endif
   Mesh *m = mesh;

#ifdef MFEM_USE_MPI
   if (pfes) { MPI_Comm_rank(pfes->GetComm(), &myid); }
   if (pmesh) { m = pmesh; }
#endif

   MFEM_VERIFY(m != NULL, "No mesh has been given to the AdaptivityEvaluator.");

   // This will be used to move the positions.
   GridFunction *mesh_nodes = m->GetNodes();
   *mesh_nodes = nodes0;
   new_field = field0;

   // Velocity of the positions.
   GridFunction u(mesh_nodes->FESpace());
   subtract(new_nodes, nodes0, u);

   TimeDependentOperator *oper = NULL;
   // This must be the fes of the ind, associated with the object's mesh.
   if (fes)  { oper = new SerialAdvectorCGOper(nodes0, u, *fes); }
#ifdef MFEM_USE_MPI
   else if (pfes) { oper = new ParAdvectorCGOper(nodes0, u, *pfes); }
#endif
   MFEM_VERIFY(oper != NULL,
               "No FE space has been given to the AdaptivityEvaluator.");
   ode_solver.Init(*oper);

   // Compute some time step [mesh_size / speed].
   double min_h = std::numeric_limits<double>::infinity();
   for (int i = 0; i < m->GetNE(); i++)
   {
      min_h = std::min(min_h, m->GetElementSize(i));
   }
   double v_max = 0.0;
   const int s = u.FESpace()->GetVSize() / 2;
   for (int i = 0; i < s; i++)
   {
      const double vel = u(i) * u(i) + u(i+s) * u(i+s);
      v_max = std::max(v_max, vel);
   }
   if (v_max == 0.0)
   {
      // No need to change the field.
      return;
   }
   v_max = std::sqrt(v_max);
   double dt = 0.5 * min_h / v_max;
   double glob_dt = dt;
#ifdef MFEM_USE_MPI
   if (pfes)
   {
      MPI_Allreduce(&dt, &glob_dt, 1, MPI_DOUBLE, MPI_MIN, pfes->GetComm());
   }
#endif

   double t = 0.0;
   bool last_step = false;
   for (int ti = 1; !last_step; ti++)
   {
      if (t + glob_dt >= 1.0)
      {
#ifdef MFEM_DEBUG
         if (myid == 0)
         {
            mfem::out << "Remap took " << ti << " steps." << std::endl;
         }
#endif
         glob_dt = 1.0 - t;
         last_step = true;
      }
      ode_solver.Step(new_field, t, glob_dt);
   }

   // Trim the overshoots and undershoots.
   const double minv = field0.Min(), maxv = field0.Max();
   for (int i = 0; i < new_field.Size(); i++)
   {
      if (new_field(i) < minv) { new_field(i) = minv; }
      if (new_field(i) > maxv) { new_field(i) = maxv; }
   }

   nodes0 = new_nodes;
   field0 = new_field;

   delete oper;
}

SerialAdvectorCGOper::SerialAdvectorCGOper(const Vector &x_start,
                                           GridFunction &vel,
                                           FiniteElementSpace &fes)
   : TimeDependentOperator(fes.GetVSize()),
     x0(x_start), x_now(*fes.GetMesh()->GetNodes()),
     u(vel), u_coeff(&u), M(&fes), K(&fes)
{
   ConvectionIntegrator *Kinteg = new ConvectionIntegrator(u_coeff);
   K.AddDomainIntegrator(Kinteg);
   K.Assemble(0);
   K.Finalize(0);

   MassIntegrator *Minteg = new MassIntegrator;
   M.AddDomainIntegrator(Minteg);
   M.Assemble();
   M.Finalize();
}

void SerialAdvectorCGOper::Mult(const Vector &ind, Vector &di_dt) const
{
   // Move the mesh.
   const double t = GetTime();
   add(x0, t, u, x_now);

   // Assemble on the new mesh.
   K.BilinearForm::operator=(0.0);
   K.Assemble();
   Vector rhs(K.Size());
   K.Mult(ind, rhs);
   M.BilinearForm::operator=(0.0);
   M.Assemble();

   di_dt = 0.0;
   CGSolver lin_solver;
   DSmoother prec;
   lin_solver.SetPreconditioner(prec);
   lin_solver.SetOperator(M.SpMat());
   lin_solver.SetRelTol(1e-12); lin_solver.SetAbsTol(0.0);
   lin_solver.SetMaxIter(100);
   lin_solver.SetPrintLevel(0);
   lin_solver.Mult(rhs, di_dt);
}

#ifdef MFEM_USE_MPI
ParAdvectorCGOper::ParAdvectorCGOper(const Vector &x_start,
                                     GridFunction &vel,
                                     ParFiniteElementSpace &pfes)
   : TimeDependentOperator(pfes.GetVSize()),
     x0(x_start), x_now(*pfes.GetMesh()->GetNodes()),
     u(vel), u_coeff(&u), M(&pfes), K(&pfes)
{
   ConvectionIntegrator *Kinteg = new ConvectionIntegrator(u_coeff);
   K.AddDomainIntegrator(Kinteg);
   K.Assemble(0);
   K.Finalize(0);

   MassIntegrator *Minteg = new MassIntegrator;
   M.AddDomainIntegrator(Minteg);
   M.Assemble();
   M.Finalize();
}

void ParAdvectorCGOper::Mult(const Vector &ind, Vector &di_dt) const
{
   // Move the mesh.
   const double t = GetTime();
   add(x0, t, u, x_now);

   // Assemble on the new mesh.
   K.BilinearForm::operator=(0.0);
   K.Assemble();
   ParGridFunction rhs(K.ParFESpace());
   K.Mult(ind, rhs);
   M.BilinearForm::operator=(0.0);
   M.Assemble();

   HypreParVector *RHS = rhs.ParallelAssemble();
   HypreParVector X(K.ParFESpace());
   X = 0.0;
   HypreParMatrix *Mh  = M.ParallelAssemble();

   CGSolver lin_solver(M.ParFESpace()->GetParMesh()->GetComm());
   HypreSmoother prec;
   prec.SetType(HypreSmoother::Jacobi, 1);
   lin_solver.SetPreconditioner(prec);
   lin_solver.SetOperator(*Mh);
   lin_solver.SetRelTol(1e-8);
   lin_solver.SetAbsTol(0.0);
   lin_solver.SetMaxIter(100);
   lin_solver.SetPrintLevel(0);
   lin_solver.Mult(*RHS, X);
   K.ParFESpace()->GetProlongationMatrix()->Mult(X, di_dt);

   delete Mh;
   delete RHS;
}
#endif

double TMOPNewtonSolver::ComputeScalingFactor(const Vector &x,
                                              const Vector &b) const
{
   const FiniteElementSpace *fes = NULL;
   double energy_in = 0.0;
#ifdef MFEM_USE_MPI
   const ParNonlinearForm *p_nlf = dynamic_cast<const ParNonlinearForm *>(oper);
   MFEM_VERIFY(!(parallel && p_nlf == NULL), "Invalid Operator subclass.");
   if (parallel)
   {
      fes = p_nlf->FESpace();
      energy_in = p_nlf->GetEnergy(x);
   }
#endif
   const bool serial = !parallel;
   const NonlinearForm *nlf = dynamic_cast<const NonlinearForm *>(oper);
   MFEM_VERIFY(!(serial && nlf == NULL), "Invalid Operator subclass.");
   if (serial)
   {
      fes = nlf->FESpace();
      energy_in = nlf->GetEnergy(x);
   }

   const bool have_b = (b.Size() == Height());

   const int NE = fes->GetMesh()->GetNE(), dim = fes->GetFE(0)->GetDim(),
             dof = fes->GetFE(0)->GetDof(), nsp = ir.GetNPoints();
   Array<int> xdofs(dof * dim);
   DenseMatrix Jpr(dim), dshape(dof, dim), pos(dof, dim);
   Vector posV(pos.Data(), dof * dim);

   Vector x_out(x.Size()), x_out_loc(fes->GetVSize());
   bool x_out_ok = false;
   double scale = 1.0, energy_out = 0.0;
   double norm0 = Norm(r);

   // Decreases the scaling of the update until the new mesh is valid.
   for (int i = 0; i < 12; i++)
   {
      add(x, -scale, c, x_out);

      if (serial)
      {
         const SparseMatrix *cP = fes->GetConformingProlongation();
         if (!cP) {x_out_loc.SetData(x_out.GetData());}
         else {cP->Mult(x_out,x_out_loc);}
         energy_out = nlf->GetGridFunctionEnergy(x_out_loc);
      }
#ifdef MFEM_USE_MPI
      else
      {
         fes->GetProlongationMatrix()->Mult(x_out, x_out_loc);
         energy_out = p_nlf->GetParGridFunctionEnergy(x_out_loc);
      }
#endif

      if (energy_out > 1.2*energy_in || std::isnan(energy_out) != 0)
      {
         if (print_level >= 0)
         { mfem::out << "Scale = " << scale << " Increasing energy.\n"; }
         scale *= 0.5; continue;
      }

      int jac_ok = 1;
      for (int i = 0; i < NE; i++)
      {
         fes->GetElementVDofs(i, xdofs);
         x_out_loc.GetSubVector(xdofs, posV);
         for (int j = 0; j < nsp; j++)
         {
            fes->GetFE(i)->CalcDShape(ir.IntPoint(j), dshape);
            MultAtB(pos, dshape, Jpr);
            if (Jpr.Det() <= 0.0) { jac_ok = 0; goto break2; }
         }
      }
   break2:
      int jac_ok_all = jac_ok;
#ifdef MFEM_USE_MPI
      if (parallel)
      {
         MPI_Allreduce(&jac_ok, &jac_ok_all, 1, MPI_INT, MPI_LAND,
                       p_nlf->ParFESpace()->GetComm());
      }
#endif

      if (jac_ok_all == 0)
      {
         if (print_level >= 0)
         { mfem::out << "Scale = " << scale << " Neg det(J) found.\n"; }
         scale *= 0.5; continue;
      }

      oper->Mult(x_out, r);
      if (have_b) { r -= b; }
      double norm = Norm(r);

      if (norm > 1.2*norm0)
      {
         if (print_level >= 0)
         { mfem::out << "Scale = " << scale << " Norm increased.\n"; }
         scale *= 0.5; continue;
      }
      else { x_out_ok = true; break; }
   }

   if (print_level >= 0)
   {
      mfem::out << "Energy decrease: "
                << (energy_in - energy_out) / energy_in * 100.0
                << "% with " << scale << " scaling.\n";
   }

   if (x_out_ok == false) { scale = 0.0; }
   return scale;
}

void TMOPNewtonSolver::ProcessNewState(const Vector &x) const
{
   if (discr_tc)
   {
      if (parallel)
      {
#ifdef MFEM_USE_MPI
         const ParNonlinearForm *nlf =
            dynamic_cast<const ParNonlinearForm *>(oper);
         Vector x_loc(nlf->ParFESpace()->GetVSize());
         nlf->ParFESpace()->GetProlongationMatrix()->Mult(x, x_loc);
         discr_tc->UpdateTargetSpecification(x_loc);
#endif
      }
      else { discr_tc->UpdateTargetSpecification(x); }
   }
}

double TMOPDescentNewtonSolver::ComputeScalingFactor(const Vector &x,
                                                     const Vector &b) const
{
   const FiniteElementSpace *fes = NULL;
   double energy_in = 0.0;
#ifdef MFEM_USE_MPI
   const ParNonlinearForm *p_nlf = dynamic_cast<const ParNonlinearForm *>(oper);
   MFEM_VERIFY(!(parallel && p_nlf == NULL), "Invalid Operator subclass.");
   if (parallel)
   {
      fes = p_nlf->FESpace();
      energy_in = p_nlf->GetEnergy(x);
   }
#endif
   const bool serial = !parallel;
   const NonlinearForm *nlf = dynamic_cast<const NonlinearForm *>(oper);
   MFEM_VERIFY(!(serial && nlf == NULL), "Invalid Operator subclass.");
   if (serial)
   {
      fes = nlf->FESpace();
      energy_in = nlf->GetEnergy(x);
   }

   const int NE = fes->GetMesh()->GetNE(), dim = fes->GetFE(0)->GetDim(),
             dof = fes->GetFE(0)->GetDof(), nsp = ir.GetNPoints();
   Array<int> xdofs(dof * dim);
   DenseMatrix Jpr(dim), dshape(dof, dim), pos(dof, dim);
   Vector posV(pos.Data(), dof * dim);
   Vector x_loc(fes->GetVSize());

   double min_detJ = infinity();
   for (int i = 0; i < NE; i++)
   {
      fes->GetElementVDofs(i, xdofs);
      x_loc.GetSubVector(xdofs, posV);

      for (int j = 0; j < nsp; j++)
      {
         fes->GetFE(i)->CalcDShape(ir.IntPoint(j), dshape);
         MultAtB(pos, dshape, Jpr);
         min_detJ = std::min(min_detJ, Jpr.Det());
      }
   }
   double min_detJ_all = min_detJ;
#ifdef MFEM_USE_MPI
   if (parallel)
   {
      MPI_Allreduce(&min_detJ, &min_detJ_all, 1, MPI_DOUBLE, MPI_MIN,
                    p_nlf->ParFESpace()->GetComm());
   }
#endif
   if (print_level >= 0)
   {
      mfem::out << "Minimum det(J) = " << min_detJ_all << '\n';
   }

   Vector x_out(x.Size());
   bool x_out_ok = false;
   double scale = 1.0, energy_out = 0.0;

   for (int i = 0; i < 7; i++)
   {
      add(x, -scale, c, x_out);
      if (serial)
      {
         const SparseMatrix *cP = fes->GetConformingProlongation();
         if (!cP) {x_loc.SetData(x_out.GetData());}
         else {cP->Mult(x_out,x_loc);}
         energy_out = nlf->GetGridFunctionEnergy(x_loc);
      }
#ifdef MFEM_USE_MPI
      else
      {
         fes->GetProlongationMatrix()->Mult(x_out, x_loc);
         energy_out = p_nlf->GetParGridFunctionEnergy(x_loc);
      }
#endif

      if (energy_out > energy_in || std::isnan(energy_out) != 0)
      {
         scale *= 0.5;
      }
      else { x_out_ok = true; break; }
   }

   if (print_level >= 0)
   {
      mfem::out << "Energy decrease: "
                << (energy_in - energy_out) / energy_in * 100.0
                << "% with " << scale << " scaling.\n";
   }

   if (x_out_ok == false) { return 0.0; }

   return scale;
}

void TMOPDescentNewtonSolver::ProcessNewState(const Vector &x) const
{
   if (discr_tc)
   {
      if (parallel)
      {
#ifdef MFEM_USE_MPI
         const ParNonlinearForm *nlf =
            dynamic_cast<const ParNonlinearForm *>(oper);
         Vector x_loc(nlf->ParFESpace()->GetVSize());
         nlf->ParFESpace()->GetProlongationMatrix()->Mult(x, x_loc);
         discr_tc->UpdateTargetSpecification(x_loc);
#endif
      }
      else { discr_tc->UpdateTargetSpecification(x); }
   }
}

#ifdef MFEM_USE_MPI
// Metric values are visualized by creating an L2 finite element functions and
// computing the metric values at the nodes.
void vis_tmop_metric_p(int order, TMOP_QualityMetric &qm,
                       const TargetConstructor &tc, ParMesh &pmesh,
                       char *title, int position)
{
   L2_FECollection fec(order, pmesh.Dimension(), BasisType::GaussLobatto);
   ParFiniteElementSpace fes(&pmesh, &fec, 1);
   ParGridFunction metric(&fes);
   InterpolateTMOP_QualityMetric(qm, tc, pmesh, metric);
   socketstream sock;
   if (pmesh.GetMyRank() == 0)
   {
      sock.open("localhost", 19916);
      sock << "solution\n";
   }
   pmesh.PrintAsOne(sock);
   metric.SaveAsOne(sock);
   if (pmesh.GetMyRank() == 0)
   {
      sock << "window_title '"<< title << "'\n"
           << "window_geometry "
           << position << " " << 0 << " " << 600 << " " << 600 << "\n"
           << "keys jRmclA\n";
   }
}
#endif

// Metric values are visualized by creating an L2 finite element functions and
// computing the metric values at the nodes.
void vis_tmop_metric_s(int order, TMOP_QualityMetric &qm,
                       const TargetConstructor &tc, Mesh &mesh,
                       char *title, int position)
{
   L2_FECollection fec(order, mesh.Dimension(), BasisType::GaussLobatto);
   FiniteElementSpace fes(&mesh, &fec, 1);
   GridFunction metric(&fes);
   InterpolateTMOP_QualityMetric(qm, tc, mesh, metric);
   osockstream sock(19916, "localhost");
   sock << "solution\n";
   mesh.Print(sock);
   metric.Save(sock);
   sock.send();
   sock << "window_title '"<< title << "'\n"
        << "window_geometry "
        << position << " " << 0 << " " << 600 << " " << 600 << "\n"
        << "keys jRmclA\n";
}

}
