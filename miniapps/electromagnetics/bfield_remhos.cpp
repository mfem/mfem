// Copyright (c) 2017, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-734707. All Rights
// reserved. See files LICENSE and NOTICE for details.
//
// This file is part of CEED, a collection of benchmarks, miniapps, software
// libraries and APIs for efficient high-order finite element and spectral
// element discretizations for exascale applications. For more information and
// source code availability see http://github.com/ceed.
//
// The CEED research is supported by the Exascale Computing Project 17-SC-20-SC,
// a collaborative effort of two U.S. Department of Energy organizations (Office
// of Science and the National Nuclear Security Administration) responsible for
// the planning and preparation of a capable exascale ecosystem, including
// software, applications, hardware, advanced system engineering and early
// testbed platforms, in support of the nation's exascale computing imperative.

using namespace std;

#include "bfield_remhos.hpp"

namespace mfem
{

namespace electromagnetics 
{


DiscreteUpwind::DiscreteUpwind(ParFiniteElementSpace &space,
                               const SparseMatrix &adv,
                               const Array<int> &adv_smap, const Vector &Mlump,
                               Assembly &asmbly, bool updateD)
   : LOSolver(space),
     K(adv), D(), K_smap(adv_smap), M_lumped(Mlump),
     assembly(asmbly), update_D(updateD)
{
   D = K;
   ComputeDiscreteUpwindMatrix();
}

void DiscreteUpwind::ComputeDiscreteUpwindMatrix() const
{
   const int *Ip = K.HostReadI(), *Jp = K.HostReadJ(), n = K.Size();

   const double *Kp = K.HostReadData();

   double *Dp = D.HostReadWriteData();
   D.HostReadWriteI(); D.HostReadWriteJ();

   for (int i = 0, k = 0; i < n; i++)
   {
      double rowsum = 0.;
      for (int end = Ip[i+1]; k < end; k++)
      {
         int j = Jp[k];
         double kij = Kp[k];
         double kji = Kp[K_smap[k]];
         double dij = fmax(fmax(0.0,-kij),-kji);
         Dp[k] = kij + dij;
         Dp[K_smap[k]] = kji + dij;
         if (i != j) { rowsum += dij; }
      }
      D(i,i) = K(i,i) - rowsum;
   }
}

void DiscreteUpwind::CalcLOSolution(const Vector &u, Vector &du) const
{
   const int ndof = pfes.GetFE(0)->GetDof();
   Vector alpha(ndof); alpha = 0.0;

   // Recompute D due to mesh changes (K changes) in remap mode.
   if (update_D) { ComputeDiscreteUpwindMatrix(); }

   // Discretization and monotonicity terms.
   D.Mult(u, du);

   // Lump fluxes (for PDU).
   ParGridFunction u_gf(&pfes);
   u_gf = u;
   u_gf.ExchangeFaceNbrData();
   Vector &u_nd = u_gf.FaceNbrData();
   const int ne = pfes.GetNE();
   u.HostRead();
   du.HostReadWrite();
   M_lumped.HostRead();
   for (int k = 0; k < ne; k++)
   {
      // Face contributions.
      for (int f = 0; f < assembly.dofs.numBdrs; f++)
      {
         assembly.LinearFluxLumping(k, ndof, f, u, du, u_nd, alpha);
      }
   }

   const int s = du.Size();
   for (int i = 0; i < s; i++) { du(i) /= M_lumped(i); }
}

LocalInverseHOSolver::LocalInverseHOSolver(ParFiniteElementSpace &space,
                                           ParBilinearForm &Mbf,
                                           ParBilinearForm &Kbf)
   : HOSolver(space), M(Mbf), K(Kbf) { }

void LocalInverseHOSolver::CalcHOSolution(const Vector &u, Vector &du) const
{
   MFEM_VERIFY(M.GetAssemblyLevel() != AssemblyLevel::PARTIAL,
               "PA for DG is not supported for Local Inverse.");

   Vector rhs(u.Size());

   K.SpMat().HostReadWriteI();
   K.SpMat().HostReadWriteJ();
   K.SpMat().HostReadWriteData();
   HypreParMatrix *K_mat = K.ParallelAssemble(&K.SpMat());
   K_mat->Mult(u, rhs);

   const int ne = pfes.GetMesh()->GetNE();
   const int nd = pfes.GetFE(0)->GetDof();
   DenseMatrix M_loc(nd);
   DenseMatrixInverse M_loc_inv(&M_loc);
   Vector rhs_loc(nd), du_loc(nd);
   Array<int> dofs;
   for (int i = 0; i < ne; i++)
   {
      pfes.GetElementDofs(i, dofs);
      rhs.GetSubVector(dofs, rhs_loc);
      M.SpMat().GetSubMatrix(dofs, dofs, M_loc);
      M_loc_inv.Factor();
      M_loc_inv.Mult(rhs_loc, du_loc);
      du.SetSubVector(dofs, du_loc);
   }

   delete K_mat;
}


void FCTSolver::CalcCompatibleLOProduct(const ParGridFunction &us,
                                        const Vector &m, const Vector &d_us_HO,
                                        Vector &s_min, Vector &s_max,
                                        const Vector &u_new,
                                        const Array<bool> &active_el,
                                        const Array<bool> &active_dofs,
                                        Vector &d_us_LO_new)
{
   const double eps = 1e-12;
   int dof_id;

   // Compute a compatible low-order solution.
   const int NE = us.ParFESpace()->GetNE();
   const int ndofs = us.Size() / NE;

   Vector s_min_loc, s_max_loc;

   d_us_LO_new = 0.0;

   for (int k = 0; k < NE; k++)
   {
      if (active_el[k] == false) { continue; }

      double mass_us = 0.0, mass_u = 0.0;
      for (int j = 0; j < ndofs; j++)
      {
         const double us_new_HO = us(k*ndofs + j) + dt * d_us_HO(k*ndofs + j);
         mass_us += us_new_HO * m(k*ndofs + j);
         mass_u  += u_new(k*ndofs + j) * m(k*ndofs + j);
      }
      double s_avg = mass_us / mass_u;

      // Min and max of s using the full stencil of active dofs.
      s_min_loc.SetDataAndSize(s_min.GetData() + k*ndofs, ndofs);
      s_max_loc.SetDataAndSize(s_max.GetData() + k*ndofs, ndofs);
      double smin = numeric_limits<double>::infinity(),
             smax = -numeric_limits<double>::infinity();
      for (int j = 0; j < ndofs; j++)
      {
         if (active_dofs[k*ndofs + j] == false) { continue; }
         smin = min(smin, s_min_loc(j));
         smax = max(smax, s_max_loc(j));
      }

      // Fix inconsistencies due to round-off and the usage of local bounds.
      for (int j = 0; j < ndofs; j++)
      {
         if (active_dofs[k*ndofs + j] == false) { continue; }

         // Check if there's a violation, s_avg < s_min, due to round-offs that
         // are inflated by the division of a small number (the 2nd check means
         // s_avg = mass_us / mass_u > s_min up to round-off in mass_us).
         if (s_avg < smin &&
             mass_us + eps > smin * mass_u) { s_avg = smin; }
         // As above for the s_max.
         if (s_avg > smax &&
             mass_us - eps < smax * mass_u) { s_avg = smax; }

#ifdef REMHOS_FCT_PRODUCT_DEBUG
         // Check if s_avg = mass_us / mass_u is within the bounds of the full
         // stencil of active dofs.
         if (mass_us + eps < smin * mass_u ||
             mass_us - eps > smax * mass_u ||
             s_avg + eps < smin ||
             s_avg - eps > smax)
         {
            std::cout << "---\ns_avg element bounds: "
                      << smin << " " << s_avg << " " << smax << std::endl;
            std::cout << "Element " << k << std::endl;
            std::cout << "Masses " << mass_us << " " << mass_u << std::endl;
            PrintCellValues(k, NE, u_new, "u_loc: ");

            MFEM_ABORT("s_avg is not in the full stencil bounds!");
         }
#endif

         // When s_avg is not in the local bounds for some dof (it should be
         // within the full stencil of active dofs), reset the bounds to s_avg.
         if (s_avg + eps < s_min_loc(j)) { s_min_loc(j) = s_avg; }
         if (s_avg - eps > s_max_loc(j)) { s_max_loc(j) = s_avg; }
      }

      // Take into account the compatible low-order solution.
      for (int j = 0; j < ndofs; j++)
      {
         // In inactive dofs we get u_new*s_avg ~ 0, which should be fine.

         // Compatible LO solution.
         dof_id = k*ndofs + j;
         d_us_LO_new(dof_id) = (u_new(dof_id) * s_avg - us(dof_id)) / dt;
      }

#ifdef REMHOS_FCT_PRODUCT_DEBUG
      // Check the LO product solution.
      double us_min, us_max;
      for (int j = 0; j < ndofs; j++)
      {
         dof_id = k*ndofs + j;
         if (active_dofs[dof_id] == false) { continue; }

         us_min = s_min_loc(j) * u_new(dof_id);
         us_max = s_max_loc(j) * u_new(dof_id);

         if (s_avg * u_new(dof_id) + eps < us_min ||
             s_avg * u_new(dof_id) - eps > us_max)
         {
            std::cout << "---\ns_avg * u: " << k << " "
                      << us_min << " "
                      << s_avg * u_new(dof_id) << " "
                      << us_max << endl
                      << u_new(dof_id) << " " << s_avg << endl
                      << s_min_loc(j) << " " << s_max_loc(j) << "\n---\n";

            MFEM_ABORT("s_avg * u not in bounds");
         }
      }
#endif
   }
}

void FCTSolver::ScaleProductBounds(const Vector &s_min, const Vector &s_max,
                                   const Vector &u_new,
                                   const Array<bool> &active_el,
                                   const Array<bool> &active_dofs,
                                   Vector &us_min, Vector &us_max)
{
   const int NE = pfes.GetNE();
   const int ndofs = u_new.Size() / NE;
   int dof_id;
   for (int k = 0; k < NE; k++)
   {
      if (active_el[k] == false) { continue; }

      // Rescale the bounds (s_min, s_max) -> (u*s_min, u*s_max).
      for (int j = 0; j < ndofs; j++)
      {
         dof_id = k*ndofs + j;

         // For inactive dofs, s_min and s_max are undefined (inf values).
         if (active_dofs[dof_id] == false)
         {
            us_min(dof_id) = 0.0;
            us_max(dof_id) = 0.0;
            continue;
         }

         us_min(dof_id) = s_min(dof_id) * u_new(dof_id);
         us_max(dof_id) = s_max(dof_id) * u_new(dof_id);
      }
   }
}

void FluxBasedFCT::CalcFCTSolution(const ParGridFunction &u, const Vector &m,
                                   const Vector &du_ho, const Vector &du_lo,
                                   const Vector &u_min, const Vector &u_max,
                                   Vector &du) const
{
   MFEM_VERIFY(smth_indicator == NULL, "TODO: update SI bounds.");

   // Construct the flux matrix (it gets recomputed every time).
   ComputeFluxMatrix(u, du_ho, flux_ij);

   // Iterated FCT correction.
   Vector du_lo_fct(du_lo);
   for (int fct_iter = 0; fct_iter < iter_cnt; fct_iter++)
   {
      // Compute sums of incoming/outgoing fluxes at each DOF.
      AddFluxesAtDofs(flux_ij, gp, gm);

      // Compute the flux coefficients (aka alphas) into gp and gm.
      ComputeFluxCoefficients(u, du_lo_fct, m, u_min, u_max, gp, gm);

      // Apply the alpha coefficients to get the final solution.
      // Update the flux matrix for iterative FCT (when iter_cnt > 1).
      UpdateSolutionAndFlux(du_lo_fct, m, gp, gm, flux_ij, du);

      du_lo_fct = du;
   }
}

void FluxBasedFCT::CalcFCTProduct(const ParGridFunction &us, const Vector &m,
                                  const Vector &d_us_HO, const Vector &d_us_LO,
                                  Vector &s_min, Vector &s_max,
                                  const Vector &u_new,
                                  const Array<bool> &active_el,
                                  const Array<bool> &active_dofs, Vector &d_us)
{
   // Construct the flux matrix (it gets recomputed every time).
   ComputeFluxMatrix(us, d_us_HO, flux_ij);

   us.HostRead();
   d_us_LO.HostRead();
   s_min.HostReadWrite();
   s_max.HostReadWrite();
   u_new.HostRead();
   active_el.HostRead();
   active_dofs.HostRead();

   // Compute a compatible low-order solution.
   Vector dus_lo_fct(us.Size()), us_min(us.Size()), us_max(us.Size());
   CalcCompatibleLOProduct(us, m, d_us_HO, s_min, s_max, u_new,
                           active_el, active_dofs, dus_lo_fct);
   ScaleProductBounds(s_min, s_max, u_new, active_el, active_dofs,
                      us_min, us_max);

   // Update the flux matrix to a product-compatible version.
   // Compute a compatible low-order solution.
   const int NE = us.ParFESpace()->GetNE();
   const int ndofs = us.Size() / NE;
   Vector flux_el(ndofs), beta(ndofs);
   DenseMatrix fij_el(ndofs);
   fij_el = 0.0;
   Array<int> dofs;
   int dof_id;
   for (int k = 0; k < NE; k++)
   {
      if (active_el[k] == false) { continue; }

      // Take into account the compatible low-order solution.
      for (int j = 0; j < ndofs; j++)
      {
         // In inactive dofs we get u_new*s_avg ~ 0, which should be fine.

         dof_id = k*ndofs + j;
         flux_el(j) = m(dof_id) * dt * (d_us_LO(dof_id) - dus_lo_fct(dof_id));
         beta(j) = m(dof_id) * u_new(dof_id);
      }

      // Make the betas sum to 1, add the new compatible fluxes.
      beta /= beta.Sum();
      for (int j = 1; j < ndofs; j++)
      {
         for (int i = 0; i < j; i++)
         {
            fij_el(i, j) = beta(j) * flux_el(i) - beta(i) * flux_el(j);
         }
      }
      pfes.GetElementDofs(k, dofs);
      flux_ij.AddSubMatrix(dofs, dofs, fij_el);
   }

   // Iterated FCT correction.
   // To get the LO compatible product solution (with s_avg), just do
   // d_us = dus_lo_fct instead of the loop below.
   for (int fct_iter = 0; fct_iter < iter_cnt; fct_iter++)
   {
      // Compute sums of incoming fluxes at each DOF.
      AddFluxesAtDofs(flux_ij, gp, gm);

      // Compute the flux coefficients (aka alphas) into gp and gm.
      ComputeFluxCoefficients(us, dus_lo_fct, m, us_min, us_max, gp, gm);

      // Apply the alpha coefficients to get the final solution.
      // Update the fluxes for iterative FCT (when iter_cnt > 1).
      UpdateSolutionAndFlux(dus_lo_fct, m, gp, gm, flux_ij, d_us);

      ZeroOutEmptyDofs(active_el, active_dofs, d_us);

      dus_lo_fct = d_us;
   }

#ifdef REMHOS_FCT_PRODUCT_DEBUG
   // Check the bounds of the final solution.
   const double eps = 1e-12;
   Vector us_new(d_us.Size());
   add(1.0, us, dt, d_us, us_new);
   for (int k = 0; k < NE; k++)
   {
      if (active_el[k] == false) { continue; }

      for (int j = 0; j < ndofs; j++)
      {
         dof_id = k*ndofs + j;
         if (active_dofs[dof_id] == false) { continue; }

         double s = us_new(dof_id) / u_new(dof_id);
         if (s + eps < s_min(dof_id) ||
             s - eps > s_max(dof_id))
         {
            std::cout << "Final s " << j << " " << k << " "
                      << s_min(dof_id) << " "
                      << s << " "
                      << s_max(dof_id) << std::endl;
            std::cout << "---\n";
         }

         if (us_new(dof_id) + eps < us_min(dof_id) ||
             us_new(dof_id) - eps > us_max(dof_id))
         {
            std::cout << "Final us " << j << " " << k << " "
                      << us_min(dof_id) << " "
                      << us_new(dof_id) << " "
                      << us_max(dof_id) << std::endl;
            std::cout << "---\n";
         }
      }
   }
#endif
}

void FluxBasedFCT::ComputeFluxMatrix(const ParGridFunction &u,
                                     const Vector &du_ho,
                                     SparseMatrix &flux_mat) const
{
   const int s = u.Size();
   double *flux_data = flux_mat.HostReadWriteData();
   flux_mat.HostReadI(); flux_mat.HostReadJ();
   const int *K_I = K.HostReadI(), *K_J = K.HostReadJ();
   const double *K_data = K.HostReadData();
   const double *u_np = u.FaceNbrData().HostRead();
   u.HostRead();
   du_ho.HostRead();
   for (int i = 0; i < s; i++)
   {
      for (int k = K_I[i]; k < K_I[i + 1]; k++)
      {
         int j = K_J[k];
         if (j <= i) { continue; }

         double kij  = K_data[k], kji = K_data[K_smap[k]];
         double dij  = max(max(0.0, -kij), -kji);
         double u_ij = (j < s) ? u(i) - u(j)
                       : u(i) - u_np[j - s];

         flux_data[k] = dt * dij * u_ij;
      }
   }

   const int NE = pfes.GetMesh()->GetNE();
   const int ndof = s / NE;
   Array<int> dofs;
   DenseMatrix Mz(ndof);
   Vector du_z(ndof);
   for (int k = 0; k < NE; k++)
   {
      pfes.GetElementDofs(k, dofs);
      M.GetSubMatrix(dofs, dofs, Mz);
      du_ho.GetSubVector(dofs, du_z);
      for (int i = 0; i < ndof; i++)
      {
         int j = 0;
         for (; j <= i; j++) { Mz(i, j) = 0.0; }
         for (; j < ndof; j++) { Mz(i, j) *= dt * (du_z(i) - du_z(j)); }
      }
      flux_mat.AddSubMatrix(dofs, dofs, Mz, 0);
   }
}

// Compute sums of incoming fluxes for every DOF.
void FluxBasedFCT::AddFluxesAtDofs(const SparseMatrix &flux_mat,
                                   Vector &flux_pos, Vector &flux_neg) const
{
   const int s = flux_pos.Size();
   const double *flux_data = flux_mat.GetData();
   const int *flux_I = flux_mat.GetI(), *flux_J = flux_mat.GetJ();
   flux_pos = 0.0;
   flux_neg = 0.0;
   flux_pos.HostReadWrite();
   flux_neg.HostReadWrite();
   for (int i = 0; i < s; i++)
   {
      for (int k = flux_I[i]; k < flux_I[i + 1]; k++)
      {
         int j = flux_J[k];

         // The skipped fluxes will be added when the outer loop is at j as
         // the flux matrix is always symmetric.
         if (j <= i) { continue; }

         const double f_ij = flux_data[k];

         if (f_ij >= 0.0)
         {
            flux_pos(i) += f_ij;
            // Modify j if it's on the same MPI task (prevents x2 counting).
            if (j < s) { flux_neg(j) -= f_ij; }
         }
         else
         {
            flux_neg(i) += f_ij;
            // Modify j if it's on the same MPI task (prevents x2 counting).
            if (j < s) { flux_pos(j) -= f_ij; }
         }
      }
   }
}

// Compute the so-called alpha coefficients that scale the fluxes into gp, gm.
void FluxBasedFCT::
ComputeFluxCoefficients(const Vector &u, const Vector &du_lo, const Vector &m,
                        const Vector &u_min, const Vector &u_max,
                        Vector &coeff_pos, Vector &coeff_neg) const
{
   const int s = u.Size();
   for (int i = 0; i < s; i++)
   {
      const double u_lo = u(i) + dt * du_lo(i);
      const double max_pos_diff = max((u_max(i) - u_lo) * m(i), 0.0),
                   min_neg_diff = min((u_min(i) - u_lo) * m(i), 0.0);
      const double sum_pos = coeff_pos(i), sum_neg = coeff_neg(i);

      coeff_pos(i) = (sum_pos > max_pos_diff) ? max_pos_diff / sum_pos : 1.0;
      coeff_neg(i) = (sum_neg < min_neg_diff) ? min_neg_diff / sum_neg : 1.0;
   }
}

void FluxBasedFCT::
UpdateSolutionAndFlux(const Vector &du_lo, const Vector &m,
                      ParGridFunction &coeff_pos, ParGridFunction &coeff_neg,
                      SparseMatrix &flux_mat, Vector &du) const
{
   Vector &a_pos_n = coeff_pos.FaceNbrData();
   Vector &a_neg_n = coeff_neg.FaceNbrData();
   coeff_pos.ExchangeFaceNbrData();
   coeff_neg.ExchangeFaceNbrData();

   du = du_lo;

   coeff_pos.HostReadWrite();
   coeff_neg.HostReadWrite();
   du.HostReadWrite();

   double *flux_data = flux_mat.HostReadWriteData();
   const int *flux_I = flux_mat.HostReadI(), *flux_J = flux_mat.HostReadJ();
   const int s = du.Size();
   for (int i = 0; i < s; i++)
   {
      for (int k = flux_I[i]; k < flux_I[i + 1]; k++)
      {
         int j = flux_J[k];
         if (j <= i) { continue; }

         double fij = flux_data[k], a_ij;
         if (fij >= 0.0)
         {
            a_ij = (j < s) ? min(coeff_pos(i), coeff_neg(j))
                   : min(coeff_pos(i), a_neg_n(j - s));
         }
         else
         {
            a_ij = (j < s) ? min(coeff_neg(i), coeff_pos(j))
                   : min(coeff_neg(i), a_pos_n(j - s));
         }
         fij *= a_ij;

         du(i) += fij / m(i) / dt;
         if (j < s) { du(j) -= fij / m(j) / dt; }

         flux_data[k] -= fij;
      }
   }
}

void ComputeBoolIndicators(int NE, const Vector &u,
                           Array<bool> &ind_elem, Array<bool> &ind_dofs)
{
   ind_elem.SetSize(NE);
   ind_dofs.SetSize(u.Size());

   ind_elem.HostWrite();
   ind_dofs.HostWrite();
   u.HostRead();

   const int ndof = u.Size() / NE;
   int dof_id;
   for (int i = 0; i < NE; i++)
   {
      ind_elem[i] = false;
      for (int j = 0; j < ndof; j++)
      {
         dof_id = i*ndof + j;
         ind_dofs[dof_id] = (u(dof_id) > EMPTY_ZONE_TOL) ? true : false;

         if (u(dof_id) > EMPTY_ZONE_TOL) { ind_elem[i] = true; }
      }
   }
}

// This function assumes a DG space.
void ComputeRatio(int NE, const Vector &us, const Vector &u,
                  Vector &s, Array<bool> &bool_el, Array<bool> &bool_dof)
{
   ComputeBoolIndicators(NE, u, bool_el, bool_dof);

   us.HostRead();
   u.HostRead();
   s.HostWrite();
   bool_el.HostRead();
   bool_dof.HostRead();

   const int ndof = u.Size() / NE;
   for (int i = 0; i < NE; i++)
   {
      if (bool_el[i] == false)
      {
         for (int j = 0; j < ndof; j++) { s(i*ndof + j) = 0.0; }
         continue;
      }

      const double *u_el = &u(i*ndof), *us_el = &us(i*ndof);
      double *s_el = &s(i*ndof);

      // Average of the existing ratios. This does not target any kind of
      // conservation. The only goal is to have s_avg between the max and min
      // of us/u, over the active dofs.
      int n = 0;
      double sum = 0.0;
      for (int j = 0; j < ndof; j++)
      {
         if (bool_dof[i*ndof + j])
         {
            sum += us_el[j] / u_el[j];
            n++;
         }
      }
      MFEM_VERIFY(n > 0, "Major error that makes no sense");
      const double s_avg = sum / n;

      for (int j = 0; j < ndof; j++)
      {
         s_el[j] = (bool_dof[i*ndof + j]) ? us_el[j] / u_el[j] : s_avg;
      }
   }
}


DofInfo::DofInfo(ParFiniteElementSpace &pfes_sltn, int btype)
   : bounds_type(btype),
     pmesh(pfes_sltn.GetParMesh()), pfes(pfes_sltn),
     fec_bounds(pfes.GetOrder(0), pmesh->Dimension(), BasisType::GaussLobatto),
     pfes_bounds(pmesh, &fec_bounds),
     x_min(&pfes_bounds), x_max(&pfes_bounds)
{
   int n = pfes.GetVSize();
   int ne = pmesh->GetNE();

   xi_min.SetSize(n);
   xi_max.SetSize(n);
   xe_min.SetSize(ne);
   xe_max.SetSize(ne);

   ExtractBdrDofs(pfes.GetOrder(0),
                  pfes.GetFE(0)->GetGeomType(), BdrDofs);
   numFaceDofs = BdrDofs.Height();
   numBdrs = BdrDofs.Width();

   FillNeighborDofs();    // Fill NbrDof.
   FillSubcell2CellDof(); // Fill Sub2Ind.
}

void DofInfo::ComputeMatrixSparsityBounds(const Vector &el_min,
                                          const Vector &el_max,
                                          Vector &dof_min, Vector &dof_max,
                                          Array<bool> *active_el)
{
   ParMesh *pmesh = pfes.GetParMesh();
   L2_FECollection fec_bounds(0, pmesh->Dimension());
   ParFiniteElementSpace pfes_bounds(pmesh, &fec_bounds);
   ParGridFunction x_min(&pfes_bounds), x_max(&pfes_bounds);
   const int NE = pmesh->GetNE();
   const int ndofs = dof_min.Size() / NE;

   x_min.HostReadWrite();
   x_max.HostReadWrite();

   x_min = el_min;
   x_max = el_max;

   x_min.ExchangeFaceNbrData(); x_max.ExchangeFaceNbrData();
   const Vector &minv = x_min.FaceNbrData(), &maxv = x_max.FaceNbrData();
   const Table &el_to_el = pmesh->ElementToElementTable();
   Array<int> face_nbr_el;
   for (int i = 0; i < NE; i++)
   {
      double el_min = x_min(i), el_max = x_max(i);

      el_to_el.GetRow(i, face_nbr_el);
      for (int n = 0; n < face_nbr_el.Size(); n++)
      {
         if (face_nbr_el[n] < NE)
         {
            // Local neighbor.
            el_min = std::min(el_min, x_min(face_nbr_el[n]));
            el_max = std::max(el_max, x_max(face_nbr_el[n]));
         }
         else
         {
            // MPI face neighbor.
            el_min = std::min(el_min, minv(face_nbr_el[n] - NE));
            el_max = std::max(el_max, maxv(face_nbr_el[n] - NE));
         }
      }

      for (int j = 0; j < ndofs; j++)
      {
         dof_min(i*ndofs + j) = el_min;
         dof_max(i*ndofs + j) = el_max;
      }
   }
}

void DofInfo::ComputeOverlapBounds(const Vector &el_min,
                                   const Vector &el_max,
                                   Vector &dof_min, Vector &dof_max,
                                   Array<bool> *active_el)
{
   GroupCommunicator &gcomm = pfes_bounds.GroupComm();
   Array<int> dofsCG;
   const int NE = pfes.GetNE();

   // Form min/max at each CG dof, considering element overlaps.
   x_min =   std::numeric_limits<double>::infinity();
   x_max = - std::numeric_limits<double>::infinity();
   for (int i = 0; i < NE; i++)
   {
      // Inactive elements don't affect the bounds.
      if (active_el && (*active_el)[i] == false) { continue; }

      x_min.HostReadWrite();
      x_max.HostReadWrite();
      pfes_bounds.GetElementDofs(i, dofsCG);
      for (int j = 0; j < dofsCG.Size(); j++)
      {
         x_min(dofsCG[j]) = std::min(x_min(dofsCG[j]), el_min(i));
         x_max(dofsCG[j]) = std::max(x_max(dofsCG[j]), el_max(i));
      }
   }
   Array<double> minvals(x_min.GetData(), x_min.Size());
   Array<double> maxvals(x_max.GetData(), x_max.Size());
   gcomm.Reduce<double>(minvals, GroupCommunicator::Min);
   gcomm.Bcast(minvals);
   gcomm.Reduce<double>(maxvals, GroupCommunicator::Max);
   gcomm.Bcast(maxvals);

   // Use (x_min, x_max) to fill (dof_min, dof_max) for each DG dof.
   const TensorBasisElement *fe_cg =
      dynamic_cast<const TensorBasisElement *>(pfes_bounds.GetFE(0));
   const Array<int> &dof_map = fe_cg->GetDofMap();
   const int ndofs = dof_map.Size();
   for (int i = 0; i < NE; i++)
   {
      // Comment about the case when active_el != null, i.e., when this function
      // is used to compute the bounds of s:
      //
      // Note that this loop goes over all elements, including inactive ones.
      // The following happens in an inactive element:
      // - If a DOF is on the boundary with an active element, it will get the
      //   value that's propagated by the continuous functions x_min and x_max.
      // - Otherwise, the DOF would get the inf values.
      // This is the mechanism that allows new elements, that switch from
      // inactive to active, to get some valid bounds. More specifically, this
      // function is called on the old state, but the result from it is used
      // to limit the new state, which has different active elements.

      pfes_bounds.GetElementDofs(i, dofsCG);
      for (int j = 0; j < dofsCG.Size(); j++)
      {
         dof_min(i*ndofs + j) = x_min(dofsCG[dof_map[j]]);
         dof_max(i*ndofs + j) = x_max(dofsCG[dof_map[j]]);
      }
   }
}

void DofInfo::ComputeElementsMinMax(const Vector &u,
                                    Vector &u_min, Vector &u_max,
                                    Array<bool> *active_el,
                                    Array<bool> *active_dof) const
{
   const int NE = pfes.GetNE(), ndof = pfes.GetFE(0)->GetDof();
   int dof_id;
   u.HostRead(); u_min.HostReadWrite(); u_max.HostReadWrite();
   for (int k = 0; k < NE; k++)
   {
      u_min(k) = numeric_limits<double>::infinity();
      u_max(k) = -numeric_limits<double>::infinity();

      // Inactive elements don't affect the bounds.
      if (active_el && (*active_el)[k] == false) { continue; }

      for (int i = 0; i < ndof; i++)
      {
         dof_id = k*ndof + i;
         // Inactive dofs don't affect the bounds.
         if (active_dof && (*active_dof)[dof_id] == false) { continue; }

         u_min(k) = min(u_min(k), u(dof_id));
         u_max(k) = max(u_max(k), u(dof_id));
      }
   }
}

void DofInfo::FillNeighborDofs()
{
   // Use the first mesh element as indicator.
   const FiniteElement &dummy = *pfes.GetFE(0);
   const int dim = pmesh->Dimension();
   int i, j, k, ne = pmesh->GetNE();
   int nd = dummy.GetDof(), p = dummy.GetOrder();
   Array <int> bdrs, orientation;

   pmesh->ExchangeFaceNbrData();
   Table *face_to_el = pmesh->GetFaceToAllElementTable();

   NbrDof.SetSize(ne, numBdrs, numFaceDofs);

   // Permutations of BdrDofs, taking into account all possible orientations.
   // Assumes BdrDofs are ordered in xyz order, which is true for 3D hexes,
   // but it isn't true for 2D quads.
   // TODO: check other FEs, function ExtractBoundaryDofs().
   int orient_cnt = 1;
   if (dim == 2) { orient_cnt = 2; }
   if (dim == 3) { orient_cnt = 8; }
   const int dof1D_cnt = p+1;
   DenseTensor fdof_ids(numFaceDofs, numBdrs, orient_cnt);
   for (int ori = 0; ori < orient_cnt; ori++)
   {
      for (int face_id = 0; face_id < numBdrs; face_id++)
      {
         for (int fdof_id = 0; fdof_id < numFaceDofs; fdof_id++)
         {
            // Index of fdof_id in the current orientation.
            const int ori_fdof_id = GetLocalFaceDofIndex(dim, face_id, ori,
                                                         fdof_id, dof1D_cnt);
            fdof_ids(ori)(ori_fdof_id, face_id) = BdrDofs(fdof_id, face_id);
         }
      }
   }

   for (k = 0; k < ne; k++)
   {
      if (dim == 1)
      {
         pmesh->GetElementVertices(k, bdrs);

         for (i = 0; i < numBdrs; i++)
         {
            const int nbr_cnt = face_to_el->RowSize(bdrs[i]);
            if (nbr_cnt == 1)
            {
               // No neighbor element.
               NbrDof(k, i, 0) = -1;
               continue;
            }

            int el1_id, el2_id, nbr_id;
            pmesh->GetFaceElements(bdrs[i], &el1_id, &el2_id);
            if (el2_id < 0)
            {
               // This element is in a different mpi task.
               el2_id = -1 - el2_id + ne;
            }
            nbr_id = (el1_id == k) ? el2_id : el1_id;

            NbrDof(k,i,0) = nbr_id*nd + BdrDofs(0, (i+1) % 2);
         }
      }
      else if (dim==2)
      {
         pmesh->GetElementEdges(k, bdrs, orientation);

         for (i = 0; i < numBdrs; i++)
         {
            const int nbr_cnt = face_to_el->RowSize(bdrs[i]);
            if (nbr_cnt == 1)
            {
               // No neighbor element.
               for (j = 0; j < numFaceDofs; j++) { NbrDof(k, i, j) = -1; }
               continue;
            }

            int el1_id, el2_id, nbr_id;
            pmesh->GetFaceElements(bdrs[i], &el1_id, &el2_id);
            if (el2_id < 0)
            {
               // This element is in a different mpi task.
               el2_id = -1 - el2_id + ne;
            }
            nbr_id = (el1_id == k) ? el2_id : el1_id;

            int el1_info, el2_info;
            pmesh->GetFaceInfos(bdrs[i], &el1_info, &el2_info);
            const int face_id_nbr = (nbr_id == el1_id) ? el1_info / 64
                                    : el2_info / 64;
            for (j = 0; j < numFaceDofs; j++)
            {
               // Here it is utilized that the orientations of the face for
               // the two elements are opposite of each other.
               NbrDof(k,i,j) = nbr_id*nd + BdrDofs(numFaceDofs - 1 - j,
                                                   face_id_nbr);
            }
         }
      }
      else if (dim==3)
      {
         pmesh->GetElementFaces(k, bdrs, orientation);

         for (int f = 0; f < numBdrs; f++)
         {
            const int nbr_cnt = face_to_el->RowSize(bdrs[f]);
            if (nbr_cnt == 1)
            {
               // No neighbor element.
               for (j = 0; j < numFaceDofs; j++) { NbrDof(k, f, j) = -1; }
               continue;
            }

            int el1_id, el2_id, nbr_id;
            pmesh->GetFaceElements(bdrs[f], &el1_id, &el2_id);
            if (el2_id < 0)
            {
               // This element is in a different mpi task.
               el2_id = -1 - el2_id + ne;
            }
            nbr_id = (el1_id == k) ? el2_id : el1_id;

            // Local index and orientation of the face, when considered in
            // the neighbor element.
            int el1_info, el2_info;
            pmesh->GetFaceInfos(bdrs[f], &el1_info, &el2_info);
            const int face_id_nbr = (nbr_id == el1_id) ? el1_info / 64
                                    : el2_info / 64;
            const int face_or_nbr = (nbr_id == el1_id) ? el1_info % 64
                                    : el2_info % 64;
            for (j = 0; j < numFaceDofs; j++)
            {
               // What is the index of the j-th dof on the face, given its
               // orientation.
               const int loc_face_dof_id =
                  GetLocalFaceDofIndex(dim, face_id_nbr, face_or_nbr,
                                       j, dof1D_cnt);
               // What is the corresponding local dof id on the element,
               // given the face orientation.
               const int nbr_dof_id =
                  fdof_ids(face_or_nbr)(loc_face_dof_id, face_id_nbr);

               NbrDof(k, f, j) = nbr_id*nd + nbr_dof_id;
            }
         }
      }
   }
}

void DofInfo::FillSubcell2CellDof()
{
   const int dim = pmesh->Dimension(), p = pfes.GetFE(0)->GetOrder();

   if (dim==1)
   {
      numSubcells = p;
      numDofsSubcell = 2;
   }
   else if (dim==2)
   {
      numSubcells = p*p;
      numDofsSubcell = 4;
   }
   else if (dim==3)
   {
      numSubcells = p*p*p;
      numDofsSubcell = 8;
   }

   Sub2Ind.SetSize(numSubcells, numDofsSubcell);

   int aux;
   for (int m = 0; m < numSubcells; m++)
   {
      for (int j = 0; j < numDofsSubcell; j++)
      {
         if (dim == 1) { Sub2Ind(m,j) = m + j; }
         else if (dim == 2)
         {
            aux = m + m/p;
            switch (j)
            {
               case 0: Sub2Ind(m,j) =  aux; break;
               case 1: Sub2Ind(m,j) =  aux + 1; break;
               case 2: Sub2Ind(m,j) =  aux + p+1; break;
               case 3: Sub2Ind(m,j) =  aux + p+2; break;
            }
         }
         else if (dim == 3)
         {
            aux = m + m/p + (p+1)*(m/(p*p));
            switch (j)
            {
               case 0: Sub2Ind(m,j) = aux; break;
               case 1: Sub2Ind(m,j) = aux + 1; break;
               case 2: Sub2Ind(m,j) = aux + p+1; break;
               case 3: Sub2Ind(m,j) = aux + p+2; break;
               case 4: Sub2Ind(m,j) = aux + (p+1)*(p+1); break;
               case 5: Sub2Ind(m,j) = aux + (p+1)*(p+1)+1; break;
               case 6: Sub2Ind(m,j) = aux + (p+1)*(p+1)+p+1; break;
               case 7: Sub2Ind(m,j) = aux + (p+1)*(p+1)+p+2; break;
            }
         }
      }
   }
}

Assembly::Assembly(DofInfo &_dofs, LowOrderMethod &_lom,
                   const GridFunction &inflow,
                   ParFiniteElementSpace &pfes, ParMesh *submesh, int mode)
   : exec_mode(mode), inflow_gf(inflow), x_gf(&pfes),
     VolumeTerms(NULL),
     fes(&pfes), SubFes0(NULL), SubFes1(NULL),
     subcell_mesh(submesh), dofs(_dofs), lom(_lom)
{
   Mesh *mesh = fes->GetMesh();
   int k, i, m, dim = mesh->Dimension(), ne = fes->GetNE();

   Array <int> bdrs, orientation;
   FaceElementTransformations *Trans;

   bdrInt.SetSize(ne, dofs.numBdrs, dofs.numFaceDofs*dofs.numFaceDofs);
   bdrInt = 0.;

   if (lom.subcell_scheme)
   {
      VolumeTerms = lom.VolumeTerms;
      SubcellWeights.SetSize(dofs.numSubcells, dofs.numDofsSubcell, ne);

      SubFes0 = lom.SubFes0;
      SubFes1 = lom.SubFes1;
   }

   // Initialization for transport mode.
   if (exec_mode == 0)
   {
      for (k = 0; k < ne; k++)
      {
         if (dim==1)      { mesh->GetElementVertices(k, bdrs); }
         else if (dim==2) { mesh->GetElementEdges(k, bdrs, orientation); }
         else if (dim==3) { mesh->GetElementFaces(k, bdrs, orientation); }

         for (i = 0; i < dofs.numBdrs; i++)
         {
            Trans = mesh->GetFaceElementTransformations(bdrs[i]);
            ComputeFluxTerms(k, i, Trans, lom);
         }

         if (lom.subcell_scheme)
         {
            for (m = 0; m < dofs.numSubcells; m++)
            {
               ComputeSubcellWeights(k, m);
            }
         }
      }
   }
}

void Assembly::ComputeFluxTerms(const int e_id, const int BdrID,
                                FaceElementTransformations *Trans,
                                LowOrderMethod &lom)
{
   Mesh *mesh = fes->GetMesh();

   int i, j, l, dim = mesh->Dimension();
   double aux, vn;

   const FiniteElement &el = *fes->GetFE(e_id);

   Vector vval, nor(dim), shape(el.GetDof());

   for (l = 0; l < lom.irF->GetNPoints(); l++)
   {
      const IntegrationPoint &ip = lom.irF->IntPoint(l);
      IntegrationPoint eip1;
      Trans->Face->SetIntPoint(&ip);

      if (dim == 1)
      {
         Trans->Loc1.Transform(ip, eip1);
         nor(0) = 2.*eip1.x - 1.0;
      }
      else
      {
         CalcOrtho(Trans->Face->Jacobian(), nor);
      }

      if (Trans->Elem1No != e_id)
      {
         Trans->Loc2.Transform(ip, eip1);
         el.CalcShape(eip1, shape);
         Trans->Elem2->SetIntPoint(&eip1);
         lom.coef->Eval(vval, *Trans->Elem2, eip1);
         nor *= -1.;
      }
      else
      {
         Trans->Loc1.Transform(ip, eip1);
         el.CalcShape(eip1, shape);
         Trans->Elem1->SetIntPoint(&eip1);
         lom.coef->Eval(vval, *Trans->Elem1, eip1);
      }

      nor /= nor.Norml2();

      if (exec_mode == 0)
      {
         // Transport.
         vn = std::min(0., vval * nor);
      }
      else
      {
         // Remap.
         vn = std::max(0., vval * nor);
         vn *= -1.0;
      }

      const double w = ip.weight * Trans->Face->Weight();
      for (i = 0; i < dofs.numFaceDofs; i++)
      {
         aux = w * shape(dofs.BdrDofs(i,BdrID)) * vn;
         for (j = 0; j < dofs.numFaceDofs; j++)
         {
            bdrInt(e_id, BdrID, i*dofs.numFaceDofs+j) -=
               aux * shape(dofs.BdrDofs(j,BdrID));
         }
      }
   }
}

void Assembly::ComputeSubcellWeights(const int k, const int m)
{
   DenseMatrix elmat; // These are essentially the same.
   const int e_id = k*dofs.numSubcells + m;
   const FiniteElement *el0 = SubFes0->GetFE(e_id);
   const FiniteElement *el1 = SubFes1->GetFE(e_id);
   ElementTransformation *tr = subcell_mesh->GetElementTransformation(e_id);
   VolumeTerms->AssembleElementMatrix2(*el1, *el0, *tr, elmat);

   for (int j = 0; j < elmat.Width(); j++)
   {
      // Using the fact that elmat has just one row.
      SubcellWeights(k)(m,j) = elmat(0,j);
   }
}

void Assembly::LinearFluxLumping(const int k, const int nd, const int BdrID,
                                 const Vector &x, Vector &y, const Vector &x_nd,
                                 const Vector &alpha) const
{
   int i, j, dofInd;
   double xNeighbor;
   Vector xDiff(dofs.numFaceDofs);
   const int size_x = x.Size();

   for (j = 0; j < dofs.numFaceDofs; j++)
   {
      dofInd = k*nd+dofs.BdrDofs(j,BdrID);
      const int nbr_dof_id = dofs.NbrDof(k, BdrID, j);
      // Note that if the boundary is outflow, we have bdrInt = 0 by definition,
      // s.t. this value will not matter.
      if (nbr_dof_id < 0) { xNeighbor = inflow_gf(dofInd); }
      else
      {
         xNeighbor = (nbr_dof_id < size_x) ? x(nbr_dof_id)
                     : x_nd(nbr_dof_id - size_x);
      }
      xDiff(j) = xNeighbor - x(dofInd);
   }

   for (i = 0; i < dofs.numFaceDofs; i++)
   {
      dofInd = k*nd+dofs.BdrDofs(i,BdrID);
      for (j = 0; j < dofs.numFaceDofs; j++)
      {
         // alpha=0 is the low order solution, alpha=1, the Galerkin solution.
         // 0 < alpha < 1 can be used for limiting within the low order method.
         y(dofInd) += bdrInt(k, BdrID, i*dofs.numFaceDofs + j) *
                      (xDiff(i) + (xDiff(j)-xDiff(i)) *
                       alpha(dofs.BdrDofs(i,BdrID)) *
                       alpha(dofs.BdrDofs(j,BdrID)));
      }
   }
}

void Assembly::NonlinFluxLumping(const int k, const int nd,
                                 const int BdrID, const Vector &x,
                                 Vector &y, const Vector &x_nd,
                                 const Vector &alpha) const
{
   int i, j, dofInd;
   double xNeighbor, SumCorrP = 0., SumCorrN = 0., eps = 1.E-15;
   const int size_x = x.Size();
   Vector xDiff(dofs.numFaceDofs), BdrTermCorr(dofs.numFaceDofs);
   BdrTermCorr = 0.;

   for (j = 0; j < dofs.numFaceDofs; j++)
   {
      dofInd = k*nd+dofs.BdrDofs(j,BdrID);
      const int nbr_dof_id = dofs.NbrDof(k, BdrID, j);
      // Note that if the boundary is outflow, we have bdrInt = 0 by definition,
      // s.t. this value will not matter.
      if (nbr_dof_id < 0) { xNeighbor = inflow_gf(dofInd); }
      else
      {
         xNeighbor = (nbr_dof_id < size_x) ? x(nbr_dof_id)
                     : x_nd(nbr_dof_id - size_x);
      }
      xDiff(j) = xNeighbor - x(dofInd);
   }

   y.HostReadWrite();
   bdrInt.HostRead();
   xDiff.HostReadWrite();
   for (i = 0; i < dofs.numFaceDofs; i++)
   {
      dofInd = k*nd+dofs.BdrDofs(i,BdrID);
      for (j = 0; j < dofs.numFaceDofs; j++)
      {
         y(dofInd) += bdrInt(k, BdrID, i*dofs.numFaceDofs + j) * xDiff(i);
         BdrTermCorr(i) += bdrInt(k, BdrID,
                                  i*dofs.numFaceDofs + j) * (xDiff(j)-xDiff(i));
      }
      BdrTermCorr(i) *= alpha(dofs.BdrDofs(i,BdrID));
      SumCorrP += max(0., BdrTermCorr(i));
      SumCorrN += min(0., BdrTermCorr(i));
   }

   for (i = 0; i < dofs.numFaceDofs; i++)
   {
      dofInd = k*nd+dofs.BdrDofs(i,BdrID);
      if (SumCorrP + SumCorrN > eps)
      {
         BdrTermCorr(i) = min(0., BdrTermCorr(i)) -
                          max(0., BdrTermCorr(i)) * SumCorrN / SumCorrP;
      }
      else if (SumCorrP + SumCorrN < -eps)
      {
         BdrTermCorr(i) = max(0., BdrTermCorr(i)) -
                          min(0., BdrTermCorr(i)) * SumCorrP / SumCorrN;
      }
      y(dofInd) += BdrTermCorr(i);
   }
}

void ZeroOutEmptyDofs(const Array<bool> &ind_elem,
                      const Array<bool> &ind_dofs, Vector &u)
{
   ind_elem.HostRead();
   ind_dofs.HostRead();
   u.HostReadWrite();

   const int NE = ind_elem.Size();
   const int ndofs = u.Size() / NE;
   for (int k = 0; k < NE; k++)
   {
      if (ind_elem[k] == true) { continue; }

      for (int i = 0; i < ndofs; i++)
      {
         if (ind_dofs[k*ndofs + i] == false) { u(k*ndofs + i) = 0.0; }
      }
   }
}

void ComputeDiscreteUpwindingMatrix(const SparseMatrix &K,
                                    Array<int> smap, SparseMatrix& D)
{
   const int *Ip = K.GetI(), *Jp = K.GetJ(), n = K.Size();
   const double *Kp = K.GetData();

   double *Dp = D.GetData();

   for (int i = 0, k = 0; i < n; i++)
   {
      double rowsum = 0.;
      for (int end = Ip[i+1]; k < end; k++)
      {
         int j = Jp[k];
         double kij = Kp[k];
         double kji = Kp[smap[k]];
         double dij = fmax(fmax(0.0,-kij),-kji);
         Dp[k] = kij + dij;
         Dp[smap[k]] = kji + dij;
         if (i != j) { rowsum += dij; }
      }
      D(i,i) = K(i,i) -rowsum;
   }
}


int GetLocalFaceDofIndex3D(int loc_face_id, int face_orient,
                           int face_dof_id, int face_dof1D_cnt)
{
   int k1 = 0, k2 = 0;
   const int kf1 = face_dof_id % face_dof1D_cnt;
   const int kf2 = face_dof_id / face_dof1D_cnt;
   switch (loc_face_id)
   {
      case 0: // BOTTOM
         switch (face_orient)
         {
            case 0: // {0, 1, 2, 3}
               k1 = kf1;
               k2 = face_dof1D_cnt-1-kf2;
               break;
            case 1: // {0, 3, 2, 1}
               k1 = face_dof1D_cnt-1-kf2;
               k2 = kf1;
               break;
            case 2: // {1, 2, 3, 0}
               k1 = face_dof1D_cnt-1-kf2;
               k2 = face_dof1D_cnt-1-kf1;
               break;
            case 3: // {1, 0, 3, 2}
               k1 = face_dof1D_cnt-1-kf1;
               k2 = face_dof1D_cnt-1-kf2;
               break;
            case 4: // {2, 3, 0, 1}
               k1 = face_dof1D_cnt-1-kf1;
               k2 = kf2;
               break;
            case 5: // {2, 1, 0, 3}
               k1 = kf2;
               k2 = face_dof1D_cnt-1-kf1;
               break;
            case 6: // {3, 0, 1, 2}
               k1 = kf2;
               k2 = kf1;
               break;
            case 7: // {3, 2, 1, 0}
               k1 = kf1;
               k2 = kf2;
               break;
            default:
               mfem_error("This orientation does not exist in 3D");
               break;
         }
         break;
      case 1: // SOUTH
         switch (face_orient)
         {
            case 0: // {0, 1, 2, 3}
               k1 = kf1;
               k2 = kf2;
               break;
            case 1: // {0, 3, 2, 1}
               k1 = kf2;
               k2 = kf1;
               break;
            case 2: // {1, 2, 3, 0}
               k1 = kf2;
               k2 = face_dof1D_cnt-1-kf1;
               break;
            case 3: // {1, 0, 3, 2}
               k1 = face_dof1D_cnt-1-kf1;
               k2 = kf2;
               break;
            case 4: // {2, 3, 0, 1}
               k1 = face_dof1D_cnt-1-kf1;
               k2 = face_dof1D_cnt-1-kf2;
               break;
            case 5: // {2, 1, 0, 3}
               k1 = face_dof1D_cnt-1-kf2;
               k2 = face_dof1D_cnt-1-kf1;
               break;
            case 6: // {3, 0, 1, 2}
               k1 = face_dof1D_cnt-1-kf2;
               k2 = kf1;
               break;
            case 7: // {3, 2, 1, 0}
               k1 = kf1;
               k2 = face_dof1D_cnt-1-kf2;
               break;
            default:
               mfem_error("This orientation does not exist in 3D");
               break;
         }
         break;
      case 2: // EAST
         switch (face_orient)
         {
            case 0: // {0, 1, 2, 3}
               k1 = kf1;
               k2 = kf2;
               break;
            case 1: // {0, 3, 2, 1}
               k1 = kf2;
               k2 = kf1;
               break;
            case 2: // {1, 2, 3, 0}
               k1 = kf2;
               k2 = face_dof1D_cnt-1-kf1;
               break;
            case 3: // {1, 0, 3, 2}
               k1 = face_dof1D_cnt-1-kf1;
               k2 = kf2;
               break;
            case 4: // {2, 3, 0, 1}
               k1 = face_dof1D_cnt-1-kf1;
               k2 = face_dof1D_cnt-1-kf2;
               break;
            case 5: // {2, 1, 0, 3}
               k1 = face_dof1D_cnt-1-kf2;
               k2 = face_dof1D_cnt-1-kf1;
               break;
            case 6: // {3, 0, 1, 2}
               k1 = face_dof1D_cnt-1-kf2;
               k2 = kf1;
               break;
            case 7: // {3, 2, 1, 0}
               k1 = kf1;
               k2 = face_dof1D_cnt-1-kf2;
               break;
            default:
               mfem_error("This orientation does not exist in 3D");
               break;
         }
         break;
      case 3: // NORTH
         switch (face_orient)
         {
            case 0: // {0, 1, 2, 3}
               k1 = face_dof1D_cnt-1-kf1;
               k2 = kf2;
               break;
            case 1: // {0, 3, 2, 1}
               k1 = kf2;
               k2 = face_dof1D_cnt-1-kf1;
               break;
            case 2: // {1, 2, 3, 0}
               k1 = kf2;
               k2 = kf1;
               break;
            case 3: // {1, 0, 3, 2}
               k1 = kf1;
               k2 = kf2;
               break;
            case 4: // {2, 3, 0, 1}
               k1 = kf1;
               k2 = face_dof1D_cnt-1-kf2;
               break;
            case 5: // {2, 1, 0, 3}
               k1 = face_dof1D_cnt-1-kf2;
               k2 = kf1;
               break;
            case 6: // {3, 0, 1, 2}
               k1 = face_dof1D_cnt-1-kf2;
               k2 = face_dof1D_cnt-1-kf1;
               break;
            case 7: // {3, 2, 1, 0}
               k1 = face_dof1D_cnt-1-kf1;
               k2 = face_dof1D_cnt-1-kf2;
               break;
            default:
               mfem_error("This orientation does not exist in 3D");
               break;
         }
         break;
      case 4: // WEST
         switch (face_orient)
         {
            case 0: // {0, 1, 2, 3}
               k1 = face_dof1D_cnt-1-kf1;
               k2 = kf2;
               break;
            case 1: // {0, 3, 2, 1}
               k1 = kf2;
               k2 = face_dof1D_cnt-1-kf1;
               break;
            case 2: // {1, 2, 3, 0}
               k1 = kf2;
               k2 = kf1;
               break;
            case 3: // {1, 0, 3, 2}
               k1 = kf1;
               k2 = kf2;
               break;
            case 4: // {2, 3, 0, 1}
               k1 = kf1;
               k2 = face_dof1D_cnt-1-kf2;
               break;
            case 5: // {2, 1, 0, 3}
               k1 = face_dof1D_cnt-1-kf2;
               k2 = kf1;
               break;
            case 6: // {3, 0, 1, 2}
               k1 = face_dof1D_cnt-1-kf2;
               k2 = face_dof1D_cnt-1-kf1;
               break;
            case 7: // {3, 2, 1, 0}
               k1 = face_dof1D_cnt-1-kf1;
               k2 = face_dof1D_cnt-1-kf2;
               break;
            default:
               mfem_error("This orientation does not exist in 3D");
               break;
         }
         break;
      case 5: // TOP
         switch (face_orient)
         {
            case 0: // {0, 1, 2, 3}
               k1 = kf1;
               k2 = kf2;
               break;
            case 1: // {0, 3, 2, 1}
               k1 = kf2;
               k2 = kf1;
               break;
            case 2: // {1, 2, 3, 0}
               k1 = kf2;
               k2 = face_dof1D_cnt-1-kf1;
               break;
            case 3: // {1, 0, 3, 2}
               k1 = face_dof1D_cnt-1-kf1;
               k2 = kf2;
               break;
            case 4: // {2, 3, 0, 1}
               k1 = face_dof1D_cnt-1-kf1;
               k2 = face_dof1D_cnt-1-kf2;
               break;
            case 5: // {2, 1, 0, 3}
               k1 = face_dof1D_cnt-1-kf2;
               k2 = face_dof1D_cnt-1-kf1;
               break;
            case 6: // {3, 0, 1, 2}
               k1 = face_dof1D_cnt-1-kf2;
               k2 = kf1;
               break;
            case 7: // {3, 2, 1, 0}
               k1 = kf1;
               k2 = face_dof1D_cnt-1-kf2;
               break;
            default:
               mfem_error("This orientation does not exist in 3D");
               break;
         }
         break;
      default: MFEM_ABORT("This face_id does not exist in 3D");
   }
   return k1 + face_dof1D_cnt * k2;
}

int GetLocalFaceDofIndex(int dim, int loc_face_id, int face_orient,
                         int face_dof_id, int face_dof1D_cnt)
{
   switch (dim)
   {
      case 1: return face_dof_id;
      case 2:
         if (loc_face_id <= 1)
         {
            // SOUTH or EAST (canonical ordering)
            return face_dof_id;
         }
         else
         {
            // NORTH or WEST (counter-canonical ordering)
            return face_dof1D_cnt - 1 - face_dof_id;
         }
      case 3: return GetLocalFaceDofIndex3D(loc_face_id, face_orient,
                                               face_dof_id, face_dof1D_cnt);
      default: MFEM_ABORT("Dimension too high!"); return 0;
   }
}


// Assuming L2 elements.
void ExtractBdrDofs(int p, Geometry::Type gtype, DenseMatrix &dofs)
{
   switch (gtype)
   {
      case Geometry::SEGMENT:
      {
         dofs.SetSize(1, 2);
         dofs(0, 0) = 0;
         dofs(0, 1) = p;
         break;
      }
      case Geometry::SQUARE:
      {
         dofs.SetSize(p+1, 4);
         for (int i = 0; i <= p; i++)
         {
            dofs(i,0) = i;
            dofs(i,1) = i*(p+1) + p;
            dofs(i,2) = (p+1)*(p+1) - 1 - i;
            dofs(i,3) = (p-i)*(p+1);
         }
         break;
      }
      case Geometry::CUBE:
      {
         dofs.SetSize((p+1)*(p+1), 6);
         for (int bdrID = 0; bdrID < 6; bdrID++)
         {
            int o(0);
            switch (bdrID)
            {
               case 0:
                  for (int i = 0; i < (p+1)*(p+1); i++)
                  {
                     dofs(o++,bdrID) = i;
                  }
                  break;
               case 1:
                  for (int i = 0; i <= p*(p+1)*(p+1); i+=(p+1)*(p+1))
                     for (int j = 0; j < p+1; j++)
                     {
                        dofs(o++,bdrID) = i+j;
                     }
                  break;
               case 2:
                  for (int i = p; i < (p+1)*(p+1)*(p+1); i+=p+1)
                  {
                     dofs(o++,bdrID) = i;
                  }
                  break;
               case 3:
                  for (int i = 0; i <= p*(p+1)*(p+1); i+=(p+1)*(p+1))
                     for (int j = p*(p+1); j < (p+1)*(p+1); j++)
                     {
                        dofs(o++,bdrID) = i+j;
                     }
                  break;
               case 4:
                  for (int i = 0; i <= (p+1)*((p+1)*(p+1)-1); i+=p+1)
                  {
                     dofs(o++,bdrID) = i;
                  }
                  break;
               case 5:
                  for (int i = p*(p+1)*(p+1); i < (p+1)*(p+1)*(p+1); i++)
                  {
                     dofs(o++,bdrID) = i;
                  }
                  break;
            }
         }
         break;
      }
      default: MFEM_ABORT("Geometry not implemented.");
   }
}

void MixedConvectionIntegrator::AssembleElementMatrix2(
   const FiniteElement &tr_el, const FiniteElement &te_el,
   ElementTransformation &Trans, DenseMatrix &elmat)
{
   int tr_nd = tr_el.GetDof();
   int te_nd = te_el.GetDof();
   int dim = te_el.GetDim(); // Using test geometry.

#ifdef MFEM_THREAD_SAFE
   DenseMatrix dshape, adjJ, Q_ir;
   Vector shape, vec2, BdFidxT;
#endif
   elmat.SetSize(te_nd, tr_nd);
   dshape.SetSize(tr_nd,dim);
   adjJ.SetSize(dim);
   shape.SetSize(te_nd);
   vec2.SetSize(dim);
   BdFidxT.SetSize(tr_nd);

   Vector vec1;

   // Using midpoint rule and test geometry.
   const IntegrationRule *ir = &IntRules.Get(te_el.GetGeomType(), 1);

   Q.Eval(Q_ir, Trans, *ir);

   elmat = 0.0;
   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);
      tr_el.CalcDShape(ip, dshape);
      te_el.CalcShape(ip, shape);

      Trans.SetIntPoint(&ip);
      CalcAdjugate(Trans.Jacobian(), adjJ);
      Q_ir.GetColumnReference(i, vec1);
      vec1 *= alpha * ip.weight;

      adjJ.Mult(vec1, vec2);
      dshape.Mult(vec2, BdFidxT);

      AddMultVWt(shape, BdFidxT, elmat);
   }
}

void GetMinMax(const ParGridFunction &g, double &min, double &max)
{
   g.HostRead();
   double min_loc = g.Min(), max_loc = g.Max();
   MPI_Allreduce(&min_loc, &min, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
   MPI_Allreduce(&max_loc, &max, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
}

Array<int> SparseMatrix_Build_smap(const SparseMatrix &A)
{
   // Assuming that A is finalized
   const int *I = A.GetI(), *J = A.GetJ(), n = A.Size();
   Array<int> smap(I[n]);

   for (int row = 0, j = 0; row < n; row++)
   {
      for (int end = I[row+1]; j < end; j++)
      {
         int col = J[j];
         // Find the offset, _j, of the (col,row) entry and store it in smap[j].
         for (int _j = I[col], _end = I[col+1]; true; _j++)
         {
            MFEM_VERIFY(_j != _end, "Can't find the symmetric entry!");

            if (J[_j] == row) { smap[j] = _j; break; }
         }
      }
   }
   return smap;
}


}
}