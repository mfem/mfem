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
   Vector new_field_temp;
   for (int i = 0; i < ncomp; i++)
   {
      new_field_temp.MakeRef(new_field, i*pnt_cnt, pnt_cnt);
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
      oper = new SerialAdvectorCGOper(nodes0, u, *fess, al);
   }
#ifdef MFEM_USE_MPI
   else if (pfes)
   {
      pfess = new ParFiniteElementSpace(pfes->GetParMesh(), pfes->FEColl(), 1);
      oper  = new ParAdvectorCGOper(nodes0, u, *pfess, al, opt_mt);
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

   u.HostReadWrite();
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
   new_field.HostReadWrite();
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
                                           FiniteElementSpace &fes,
                                           AssemblyLevel al)
   : TimeDependentOperator(fes.GetVSize()),
     x0(x_start), x_now(*fes.GetMesh()->GetNodes()),
     u(vel), u_coeff(&u), M(&fes), K(&fes), al(al)
{
   ConvectionIntegrator *Kinteg = new ConvectionIntegrator(u_coeff);
   K.AddDomainIntegrator(Kinteg);
   K.SetAssemblyLevel(al);
   K.Assemble(0);
   K.Finalize(0);

   MassIntegrator *Minteg = new MassIntegrator;
   M.AddDomainIntegrator(Minteg);
   M.SetAssemblyLevel(al);
   M.Assemble(0);
   M.Finalize(0);
}

void SerialAdvectorCGOper::Mult(const Vector &ind, Vector &di_dt) const
{
   // Move the mesh.
   const double t = GetTime();
   add(x0, t, u, x_now);

   if (al == AssemblyLevel::PARTIAL)
   {
      K.FESpace()->GetMesh()->DeleteGeometricFactors();
   }

   // Assemble on the new mesh.
   K.BilinearForm::operator=(0.0);
   K.Assemble();
   Vector rhs(K.Size());
   K.Mult(ind, rhs);
   M.BilinearForm::operator=(0.0);
   M.Assemble();

   di_dt = 0.0;
   CGSolver lin_solver;
   Solver *prec = nullptr;
   Array<int> ess_tdof_list;
   if (al == AssemblyLevel::PARTIAL)
   {
      prec = new OperatorJacobiSmoother(M, ess_tdof_list);
      lin_solver.SetOperator(M);
   }
   else
   {
      prec = new DSmoother(M.SpMat());
      lin_solver.SetOperator(M.SpMat());
   }
   lin_solver.SetPreconditioner(*prec);
   lin_solver.SetRelTol(1e-12); lin_solver.SetAbsTol(0.0);
   lin_solver.SetMaxIter(100);
   lin_solver.SetPrintLevel(0);
   lin_solver.Mult(rhs, di_dt);

   delete prec;
}

#ifdef MFEM_USE_MPI
ParAdvectorCGOper::ParAdvectorCGOper(const Vector &x_start,
                                     GridFunction &vel,
                                     ParFiniteElementSpace &pfes,
                                     AssemblyLevel al,
                                     MemoryType mt)
   : TimeDependentOperator(pfes.GetVSize()),
     x0(x_start), x_now(*pfes.GetMesh()->GetNodes()),
     u(vel), u_coeff(&u), M(&pfes), K(&pfes), al(al)
{
   ConvectionIntegrator *Kinteg = new ConvectionIntegrator(u_coeff);
   if (al == AssemblyLevel::PARTIAL)
   {
      Kinteg->SetPAMemoryType(mt);
   }
   K.AddDomainIntegrator(Kinteg);
   K.SetAssemblyLevel(al);
   K.Assemble(0);
   K.Finalize(0);

   MassIntegrator *Minteg = new MassIntegrator;
   if (al == AssemblyLevel::PARTIAL)
   {
      Minteg->SetPAMemoryType(mt);
   }
   M.AddDomainIntegrator(Minteg);
   M.SetAssemblyLevel(al);
   M.Assemble(0);
   M.Finalize(0);
}

void ParAdvectorCGOper::Mult(const Vector &ind, Vector &di_dt) const
{
   // Move the mesh.
   const double t = GetTime();
   add(x0, t, u, x_now);

   if (al == AssemblyLevel::PARTIAL)
   {
      K.ParFESpace()->GetParMesh()->DeleteGeometricFactors();
   }

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

   OperatorHandle Mop;
   Solver *prec = nullptr;
   Array<int> ess_tdof_list;
   if (al == AssemblyLevel::PARTIAL)
   {
      M.FormSystemMatrix(ess_tdof_list, Mop);
      prec = new OperatorJacobiSmoother(M, ess_tdof_list);
   }
   else
   {
      Mop.Reset(M.ParallelAssemble());
      prec = new HypreSmoother;
      static_cast<HypreSmoother*>(prec)->SetType(HypreSmoother::Jacobi, 1);
   }

   CGSolver lin_solver(M.ParFESpace()->GetParMesh()->GetComm());
   lin_solver.SetPreconditioner(*prec);
   lin_solver.SetOperator(*Mop);
   lin_solver.SetRelTol(1e-8);
   lin_solver.SetAbsTol(0.0);
   lin_solver.SetMaxIter(100);
   lin_solver.SetPrintLevel(0);
   lin_solver.Mult(*RHS, X);
   K.ParFESpace()->GetProlongationMatrix()->Mult(X, di_dt);

   delete RHS;
   delete prec;
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
}

void InterpolatorFP::ComputeAtNewPosition(const Vector &new_nodes,
                                          Vector &new_field)
{
   finder->Interpolate(new_nodes, field0_gf, new_field);
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

   // Get the local prolongation of the solution vector.
   Vector x_out_loc(fes->GetVSize(),
                    (temp_mt == MemoryType::DEFAULT) ? Device::GetDeviceMemoryType() : temp_mt);
   if (serial)
   {
      const SparseMatrix *cP = fes->GetConformingProlongation();
      if (!cP) { x_out_loc = x; }
      else     { cP->Mult(x, x_out_loc); }
   }
#ifdef MFEM_USE_MPI
   else
   {
      fes->GetProlongationMatrix()->Mult(x, x_out_loc);
   }
#endif

   // Check if the starting mesh (given by x) is inverted. Note that x hasn't
   // been modified by the Newton update yet.
   const double min_detT_in = ComputeMinDet(x_out_loc, *fes);
   const bool untangling = (min_detT_in <= 0.0) ? true : false;
   const double untangle_factor = 1.5;
   if (untangling)
   {
      // Needed for the line search below. The untangling metrics see this
      // reference to detect deteriorations.
      *min_det_ptr = untangle_factor * min_detT_in;
   }

   const bool have_b = (b.Size() == Height());

   Vector x_out(x.Size());
   bool x_out_ok = false;
   double scale = 1.0, energy_out = 0.0, min_detT_out;
   const double norm_in = Norm(r);

   const double detJ_factor = (solver_type == 1) ? 0.25 : 0.5;

   // Perform the line search.
   for (int i = 0; i < 12; i++)
   {
      // Update the mesh and get the L-vector in x_out_loc.
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

      // Check the changes in detJ.
      min_detT_out = ComputeMinDet(x_out_loc, *fes);
      if (untangling == false && min_detT_out < 0.0)
      {
         // No untangling, and detJ got negative -- no good.
         if (print_level >= 0)
         { mfem::out << "Scale = " << scale << " Neg det(J) found.\n"; }
         scale *= detJ_factor; continue;
      }
      if (untangling == true && min_detT_out < *min_det_ptr)
      {
         // Untangling, and detJ got even more negative -- no good.
         if (print_level >= 0)
         { mfem::out << "Scale = " << scale << " Neg det(J) decreased.\n"; }
         scale *= detJ_factor; continue;
      }

      // Skip the energy and residual checks when we're untangling. The
      // untangling metrics change their denominators, which can affect the
      // energy and residual, so their increase/decrease is not relevant.
      if (untangling) { x_out_ok = true; break; }

      // Check the changes in total energy.
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
      if (energy_out > energy_in + 0.2*fabs(energy_in) ||
          std::isnan(energy_out) != 0)
      {
         if (print_level >= 0)
         {
            mfem::out << "Scale = " << scale << " Increasing energy: "
                      << energy_in << " --> " << energy_out << '\n';
         }
         scale *= 0.5; continue;
      }

      // Check the changes in the Newton residual.
      oper->Mult(x_out, r);
      if (have_b) { r -= b; }
      double norm_out = Norm(r);

      if (norm_out > 1.2*norm_in)
      {
         if (print_level >= 0)
         {
            mfem::out << "Scale = " << scale << " Norm increased: "
                      << norm_in << " --> " << norm_out << '\n';
         }
         scale *= 0.5; continue;
      }
      else { x_out_ok = true; break; }
   } // end line search

   if (untangling)
   {
      // Update the global min detJ. Untangling metrics see this min_det_ptr.
      if (min_detT_out > 0.0)
      {
         *min_det_ptr = 0.0;
         if (print_level >= 0)
         { mfem::out << "The mesh has been untangled at the used points!\n"; }
      }
      else { *min_det_ptr = untangle_factor * min_detT_out; }
   }

   if (print_level >= 0)
   {
      if (untangling)
      {
         mfem::out << "Min det(T) change: "
                   << min_detT_in << " -> " << min_detT_out
                   << " with " << scale << " scaling.\n";
      }
      else
      {
         mfem::out << "Energy decrease: "
                   << energy_in << " --> " << energy_out << " or "
                   << (energy_in - energy_out) / energy_in * 100.0
                   << "% with " << scale << " scaling.\n";
      }
   }

   if (x_out_ok == false) { scale = 0.0; }
   return scale;
}

void TMOPNewtonSolver::ProcessNewState(const Vector &x) const
{
   const NonlinearForm *nlf = dynamic_cast<const NonlinearForm *>(oper);
   const Array<NonlinearFormIntegrator*> &integs = *nlf->GetDNFI();

   // Reset the update flags of all TargetConstructors. This is done to avoid
   // repeated updates of shared TargetConstructors.
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
               ati[j]->UpdateAfterMeshChange(x_loc);
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
               ati[j]->UpdateAfterMeshChange(x_loc);
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

double TMOPNewtonSolver::ComputeMinDet(const Vector &x_loc,
                                       const FiniteElementSpace &fes) const
{
   double min_detJ = infinity();
   const int NE = fes.GetNE(), dim = fes.GetMesh()->Dimension();
   Array<int> xdofs;
   DenseMatrix Jpr(dim);
   const bool mixed_mesh = fes.GetMesh()->GetNumGeometries(dim) > 1;
   if (dim == 1 || mixed_mesh || UsesTensorBasis(fes) == false)
   {
      for (int i = 0; i < NE; i++)
      {
         const int dof = fes.GetFE(i)->GetDof();
         DenseMatrix dshape(dof, dim), pos(dof, dim);
         Vector posV(pos.Data(), dof * dim);

         fes.GetElementVDofs(i, xdofs);
         x_loc.GetSubVector(xdofs, posV);

         const IntegrationRule &irule = GetIntegrationRule(*fes.GetFE(i));
         const int nsp = irule.GetNPoints();
         for (int j = 0; j < nsp; j++)
         {
            fes.GetFE(i)->CalcDShape(irule.IntPoint(j), dshape);
            MultAtB(pos, dshape, Jpr);
            min_detJ = std::min(min_detJ, Jpr.Det());
         }
      }
   }
   else
   {
      min_detJ = dim == 2 ? MinDetJpr_2D(&fes, x_loc) :
                 dim == 3 ? MinDetJpr_3D(&fes, x_loc) : 0.0;
   }
   double min_detT_all = min_detJ;
#ifdef MFEM_USE_MPI
   if (parallel)
   {
      auto p_nlf = dynamic_cast<const ParNonlinearForm *>(oper);
      MPI_Allreduce(&min_detJ, &min_detT_all, 1, MPI_DOUBLE, MPI_MIN,
                    p_nlf->ParFESpace()->GetComm());
   }
#endif
   const DenseMatrix &Wideal =
      Geometries.GetGeomToPerfGeomJac(fes.GetFE(0)->GetGeomType());
   min_detT_all /= Wideal.Det();

   return min_detT_all;
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
