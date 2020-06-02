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
   // TODO: Implement for AMR meshes.
   const int pnt_cnt = new_field.Size()/ncomp;

   new_field = field0;

   for (int i = 0; i < ncomp; i++)
   {
      Vector new_field_temp(new_field.GetData()+i*pnt_cnt, pnt_cnt);
      ComputeAtNewPositionScalar(new_nodes, new_field_temp);
   }

   field0 = new_field;
   nodes0 = new_nodes;
}

void AdvectorCG::ComputeAtNewPositionScalar(const Vector &new_nodes,
                                            Vector &new_field)
{
   Mesh *m = mesh;
#ifdef MFEM_USE_MPI
   if (pmesh) { m = pmesh; }
#endif

   MFEM_VERIFY(m != NULL, "No mesh has been given to the AdaptivityEvaluator.");

   // This will be used to move the positions.
   GridFunction *mesh_nodes = m->GetNodes();
   *mesh_nodes = nodes0;
   double minv = new_field.Min(), maxv = new_field.Max();

   // Velocity of the positions.
   GridFunction u(mesh_nodes->FESpace());
   subtract(new_nodes, nodes0, u);

   // Define a scalar FE space for the solution, and the advection operator.
   TimeDependentOperator *oper = NULL;
   FiniteElementSpace *fess = NULL;
#ifdef MFEM_USE_MPI
   ParFiniteElementSpace *pfess = NULL;
#endif
   if (fes)
   {
      fess = new FiniteElementSpace(fes->GetMesh(), fes->FEColl(), 1);
      oper = new SerialAdvectorCGOper(nodes0, u, *fess);
   }
#ifdef MFEM_USE_MPI
   else if (pfes)
   {
      pfess = new ParFiniteElementSpace(pfes->GetParMesh(), pfes->FEColl(), 1);
      oper  = new ParAdvectorCGOper(nodes0, u, *pfess);
   }
#endif
   MFEM_VERIFY(oper != NULL,
               "No FE space has been given to the AdaptivityEvaluator.");
   ode_solver.Init(*oper);

   // Compute some time step [mesh_size / speed].
   double h_min = std::numeric_limits<double>::infinity();
   for (int i = 0; i < m->GetNE(); i++)
   {
      h_min = std::min(h_min, m->GetElementSize(i));
   }
   double v_max = 0.0;
   const int s = new_field.Size();

   for (int i = 0; i < s; i++)
   {
      double vel = 0.;
      for (int j = 0; j < dim; j++)
      {
         vel += u(i+j*s)*u(i+j*s);
      }
      v_max = std::max(v_max, vel);
   }

#ifdef MFEM_USE_MPI
   if (pfes)
   {
      double v_loc = v_max, h_loc = h_min;
      MPI_Allreduce(&v_loc, &v_max, 1, MPI_DOUBLE, MPI_MAX, pfes->GetComm());
      MPI_Allreduce(&h_loc, &h_min, 1, MPI_DOUBLE, MPI_MIN, pfes->GetComm());
   }
#endif

   if (v_max == 0.0) // No need to change the field.
   {
      delete oper;
      delete fess;
#ifdef MFEM_USE_MPI
      delete pfess;
#endif
      return;
   }

   v_max = std::sqrt(v_max);
   double dt = dt_scale * h_min / v_max;

   double t = 0.0;
   bool last_step = false;
   for (int ti = 1; !last_step; ti++)
   {
      if (t + dt >= 1.0)
      {
         dt = 1.0 - t;
         last_step = true;
      }
      ode_solver.Step(new_field, t, dt);
   }

   double glob_minv = minv,
          glob_maxv = maxv;
#ifdef MFEM_USE_MPI
   if (pfes)
   {
      MPI_Allreduce(&minv, &glob_minv, 1, MPI_DOUBLE, MPI_MIN, pfes->GetComm());
      MPI_Allreduce(&maxv, &glob_maxv, 1, MPI_DOUBLE, MPI_MAX, pfes->GetComm());
   }
#endif

   // Trim the overshoots and undershoots.
   for (int i = 0; i < s; i++)
   {
      if (new_field(i) < glob_minv) { new_field(i) = glob_minv; }
      if (new_field(i) > glob_maxv) { new_field(i) = glob_maxv; }
   }

   delete oper;
   delete fess;
#ifdef MFEM_USE_MPI
   delete pfess;
#endif
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

#ifdef MFEM_USE_GSLIB
void InterpolatorFP::SetInitialField(const Vector &init_nodes,
                                     const Vector &init_field)
{
   nodes0 = init_nodes;
   Mesh *m = mesh;
#ifdef MFEM_USE_MPI
   if (pmesh) { m = pmesh; }
#endif
   m->SetNodes(nodes0);

   const double rel_bbox_el = 0.1;
   const double newton_tol  = 1.0e-12;
   const int npts_at_once   = 256;

   if (finder)
   {
      finder->FreeData();
      delete finder;
   }

   FiniteElementSpace *f = fes;
#ifdef MFEM_USE_MPI
   if (pfes)
   {
      f = pfes;
      finder = new FindPointsGSLIB(pfes->GetComm());
   }
   else { finder = new FindPointsGSLIB(); }
#else
   finder = new FindPointsGSLIB();
#endif
   finder->Setup(*m, rel_bbox_el, newton_tol, npts_at_once);

   field0_gf.SetSpace(f);
   field0_gf = init_field;

   dim = f->GetFE(0)->GetDim();
   const int pts_cnt = init_nodes.Size() / dim;
   el_id_out.SetSize(pts_cnt);
   code_out.SetSize(pts_cnt);
   task_id_out.SetSize(pts_cnt);
   pos_r_out.SetSize(pts_cnt*dim);
   dist_p_out.SetSize(pts_cnt);
}

void InterpolatorFP::ComputeAtNewPosition(const Vector &new_nodes,
                                          Vector &new_field)
{
   const int pts_cnt = new_nodes.Size() / dim;

   // The sizes may change between calls due to AMR.
   if (el_id_out.Size() != pts_cnt)
   {
      el_id_out.SetSize(pts_cnt);
      code_out.SetSize(pts_cnt);
      task_id_out.SetSize(pts_cnt);
      pos_r_out.SetSize(pts_cnt*dim);
      dist_p_out(pts_cnt);
   }

   // Interpolate FE function values on the found points.
   finder->FindPoints(new_nodes, code_out, task_id_out,
                      el_id_out, pos_r_out, dist_p_out);
   finder->Interpolate(code_out, task_id_out, el_id_out,
                       pos_r_out, field0_gf, new_field);
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
         if (!cP) { x_out_loc = x_out; }
         else     { cP->Mult(x_out, x_out_loc); }
      }
#ifdef MFEM_USE_MPI
      else
      {
         fes->GetProlongationMatrix()->Mult(x_out, x_out_loc);
      }
#endif

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

      ProcessNewState(x_out);
      if (serial)
      {
         energy_out = nlf->GetGridFunctionEnergy(x_out_loc);
      }
#ifdef MFEM_USE_MPI
      else
      {
         energy_out = p_nlf->GetParGridFunctionEnergy(x_out_loc);
      }
#endif
      if (energy_out > 1.2*energy_in || std::isnan(energy_out) != 0)
      {
         if (print_level >= 0)
         { mfem::out << "Scale = " << scale << " Increasing energy.\n"; }
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
   const NonlinearForm *nlf = dynamic_cast<const NonlinearForm *>(oper);
   const Array<NonlinearFormIntegrator*> &integs = *nlf->GetDNFI();

   // Reset the update flags of all TargetConstructors.
   // This is done to avoid repeated updates of shared TargetConstructors.
   TMOP_Integrator *ti  = NULL;
   TMOPComboIntegrator *co = NULL;
   DiscreteAdaptTC *dtc = NULL;
   for (int i = 0; i < integs.Size(); i++)
   {
      ti = dynamic_cast<TMOP_Integrator *>(integs[i]);
      if (ti)
      {
         dtc = ti->GetDiscreteAdaptTC();
         if (dtc) { dtc->ResetUpdateFlags(); }
      }
      co = dynamic_cast<TMOPComboIntegrator *>(integs[i]);
      if (co)
      {
         Array<TMOP_Integrator *> ati = co->GetTMOPIntegrators();
         for (int j = 0; j < ati.Size(); j++)
         {
            dtc = ati[j]->GetDiscreteAdaptTC();
            if (dtc) { dtc->ResetUpdateFlags(); }
         }
      }
   }

   if (parallel)
   {
#ifdef MFEM_USE_MPI
      const ParNonlinearForm *nlf =
         dynamic_cast<const ParNonlinearForm *>(oper);
      const ParFiniteElementSpace *pfesc = nlf->ParFESpace();
      Vector x_loc(pfesc->GetVSize());
      pfesc->GetProlongationMatrix()->Mult(x, x_loc);
      for (int i = 0; i < integs.Size(); i++)
      {
         ti = dynamic_cast<TMOP_Integrator *>(integs[i]);
         if (ti)
         {
            ti->UpdateAfterMeshChange(x_loc);
            ti->ComputeFDh(x_loc, *pfesc);
            UpdateDiscreteTC(*ti, x_loc);
         }
         co = dynamic_cast<TMOPComboIntegrator *>(integs[i]);
         if (co)
         {
            Array<TMOP_Integrator *> ati = co->GetTMOPIntegrators();
            for (int j = 0; j < ati.Size(); j++)
            {
               ati[j]->ComputeFDh(x_loc, *pfesc);
               UpdateDiscreteTC(*ati[j], x_loc);
            }
         }
      }
#endif
   }
   else
   {
      const FiniteElementSpace *fesc = nlf->FESpace();
      const Operator *P = nlf->GetProlongation();
      Vector x_loc;
      if (P)
      {
         x_loc.SetSize(P->Height());
         P->Mult(x,x_loc);
      }
      else
      {
         x_loc = x;
      }
      for (int i = 0; i < integs.Size(); i++)
      {
         ti = dynamic_cast<TMOP_Integrator *>(integs[i]);
         if (ti)
         {
            ti->UpdateAfterMeshChange(x_loc);
            ti->ComputeFDh(x_loc, *fesc);
            UpdateDiscreteTC(*ti, x_loc);
         }
         co = dynamic_cast<TMOPComboIntegrator *>(integs[i]);
         if (co)
         {
            Array<TMOP_Integrator *> ati = co->GetTMOPIntegrators();
            for (int j = 0; j < ati.Size(); j++)
            {
               ati[j]->ComputeFDh(x_loc, *fesc);
               UpdateDiscreteTC(*ati[j], x_loc);
            }
         }
      }
   }
}

void TMOPNewtonSolver::UpdateDiscreteTC(const TMOP_Integrator &ti,
                                        const Vector &x_new) const
{
   const bool update_flag = true;
   DiscreteAdaptTC *discrtc = ti.GetDiscreteAdaptTC();
   if (discrtc)
   {
      discrtc->UpdateTargetSpecification(x_new, update_flag);
      if (ti.GetFDFlag())
      {
         double dx = ti.GetFDh();
         discrtc->UpdateGradientTargetSpecification(x_new, dx, update_flag);
         discrtc->UpdateHessianTargetSpecification(x_new, dx, update_flag);
      }
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
      // TODO x_loc doesn't have valid values here!
      MFEM_ABORT("This function has to be fixed!");
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
         if (!cP) { x_loc = x_out; }
         else     { cP->Mult(x_out,x_loc); }
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
