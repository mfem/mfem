//                  MFEM Example 9 - Serial/Parallel Shared Code

#include "mfem.hpp"
#include <limits>

namespace mfem
{

// Function f = 1 for lumped boundary operator
real_t one(const Vector &x) { return 1.0; }

/** Abstract base class for evaluating the time-dependent operator in the ODE
    formulation. The continuous Galerkin (CG) strong form of the advection
    equation du/dt = -v.grad(u) is given by M du/dt = -K u + b, where M and K
    are the mass and advection matrices, respectively, and b represents the
    boundary flow contribution.

    The ODE can be reformulated as:
    du/dt = M_L^{-1}((-K + D) u + F^*(u) + b),
    where M_L is the lumped mass matrix, D is a low-order stabilization term,
    and F^*(u) represents the limited anti-diffusive fluxes.
    Here, F^* is a limited version of F, which recover the high-order target
    scheme. The limited anti-diffusive fluxes F^* are the sum of the limited
    element contributions of the original flux F to enforce local bounds.

    Additional to the limiter we implement the low-order scheme and
    high-order target scheme by chosing:
    - F^* = 0 for the bound-preserving low-order scheme.
    - F^* = F for the high-order target scheme which is not bound-preserving.

    This abstract class provides a framework for evaluating the right-hand side
    of the ODE and is intended to be inherited by classes that implement
    the three schemes:
    - The ClipAndScale class, which employes the limiter to enforces local
      bounds
    - The HighOrderTargetScheme class, which employs the raw anti-diffusive
      fluxes F
    - The LowOrderScheme class, which employs F = 0 and has low accuracy, but
      is bound-preserving */
class CG_FE_Evolution : public TimeDependentOperator
{
protected:
   const Vector &lumpedmassmatrix;
   FiniteElementSpace &fes;
#ifdef MFEM_USE_MPI
   ParFiniteElementSpace *pfes;
#endif
   int *I, *J;
   LinearForm b_lumped;
   GridFunction u_inflow;

   mutable DenseMatrix Ke, Me;
   mutable Vector ue, re, udote, fe, fe_star, gammae;
   mutable ConvectionIntegrator conv_int;
   mutable MassIntegrator mass_int;
   mutable Vector z;

   virtual void ComputeLOTimeDerivatives(const Vector &u, Vector &udot) const;

public:
   CG_FE_Evolution(FiniteElementSpace &fes_,
                   const Vector &lumpedmassmatrix_, FunctionCoefficient &inflow,
                   VectorFunctionCoefficient &vel, BilinearForm &M)
      : TimeDependentOperator(lumpedmassmatrix_.Size()),
        lumpedmassmatrix(lumpedmassmatrix_), fes(fes_),
        I(M.SpMat().GetI()), J(M.SpMat().GetJ()), b_lumped(&fes),
        u_inflow(&fes), conv_int(vel), mass_int()
   {
      u_inflow.ProjectCoefficient(inflow);

      // For bound preservation the boundary condition \hat{u} is enforced
      // via a lumped approximation to < (u_h - u_inflow) * min(v * n, 0 ), w >,
      // i.e., (u_i - (u_inflow)_i) * \int_F \varphi_i * min(v * n, 0).
      // The integral can be implemented as follows:
      FunctionCoefficient fc1(one);
      b_lumped.AddBdrFaceIntegrator(new BoundaryFlowIntegrator(fc1, vel, 1.0));
      b_lumped.Assemble();

      z.SetSize(lumpedmassmatrix.Size());

#ifdef MFEM_USE_MPI
      pfes = dynamic_cast<ParFiniteElementSpace *>(&fes);
      if (pfes)
      {
         // distribute the lumped mass matrix entries
         Array<real_t> lumpedmassmatrix_array(lumpedmassmatrix.GetData(),
                                              lumpedmassmatrix.Size());
         pfes->GroupComm().Reduce<real_t>(lumpedmassmatrix_array,
                                          GroupCommunicator::Sum);
         pfes->GroupComm().Bcast(lumpedmassmatrix_array);
      }
#endif
   }

   virtual void Mult(const Vector &x, Vector &y) const = 0;

   /// Estimate a CFL-like forward Euler time step for the low-order (LO)
   /// scheme, based on Lemma 4.3 in:
   ///   HIGH-ORDER MULTI-MATERIAL ALE HYDRODYNAMICS
   ///   Anderson, Dobrev, Kolev, Rieben, Tomov (2018).
   ///
   /// The LO forward Euler update can be written in the form
   ///   M_L u^{n+1} = (M_L + dt * K_LO) u^n + dt * rhs_const,
   /// where M_L is the lumped mass matrix and K_LO has nonnegative
   /// off-diagonals and nonpositive diagonal.
   /// Lemma 4.3 gives the sufficient condition for entrywise nonnegativity:
   ///   m_i + dt * (K_LO)_{ii} >= 0  for all i,
   /// i.e., dt <= min_i m_i / (-(K_LO)_{ii}) over DOFs with (K_LO)_{ii} < 0.
   ///
   /// Two estimates are returned:
   /// - dt_local: uses per-element (unassembled) matrices; ignores overlap.
   /// - dt_global: uses the globally assembled diagonal (sums element overlap).
   void ComputeLOTimeStepEstimates(real_t &dt_local, real_t &dt_global) const;

   virtual ~CG_FE_Evolution() { }
};

// High-order target scheme class
class HighOrderTargetScheme : public CG_FE_Evolution
{
private:
   mutable Vector udot;

public:
   HighOrderTargetScheme(FiniteElementSpace &fes_,
                         const Vector &lumpedmassmatrix_,
                         FunctionCoefficient &inflow,
                         VectorFunctionCoefficient &velocity, BilinearForm &M)
      : CG_FE_Evolution(fes_, lumpedmassmatrix_, inflow, velocity, M)
   {
      udot.SetSize(lumpedmassmatrix.Size());
   }

   virtual void Mult(const Vector &x, Vector &y) const override;
};

// Low-order scheme class
class LowOrderScheme : public CG_FE_Evolution
{
public:
   LowOrderScheme(FiniteElementSpace &fes_,
                  const Vector &lumpedmassmatrix_, FunctionCoefficient &inflow,
                  VectorFunctionCoefficient &velocity, BilinearForm &M)
      : CG_FE_Evolution(fes_, lumpedmassmatrix_, inflow, velocity, M) { }

   virtual void Mult(const Vector &x, Vector &y) const override
   {
      ComputeLOTimeDerivatives(x, y);
   }
};

// Clip and Scale limiter class
class ClipAndScale : public CG_FE_Evolution
{
private:
   mutable Array<real_t> umin, umax;
   mutable Vector udot;

   virtual void ComputeBounds(const Vector &u, Array<real_t> &u_min,
                              Array<real_t> &u_max) const;

public:
   ClipAndScale(FiniteElementSpace &fes_,
                const Vector &lumpedmassmatrix_, FunctionCoefficient &inflow,
                VectorFunctionCoefficient &velocity, BilinearForm &M)
      : CG_FE_Evolution(fes_, lumpedmassmatrix_, inflow, velocity, M)
   {
      umin.SetSize(lumpedmassmatrix.Size());
      umax.SetSize(lumpedmassmatrix.Size());
      udot.SetSize(lumpedmassmatrix.Size());
   }


   virtual void Mult(const Vector &x, Vector &y) const override;

   virtual ~ClipAndScale() { }
};

void CG_FE_Evolution::ComputeLOTimeDerivatives(const Vector &u,
                                               Vector &udot) const
{
   udot = 0.0;
   const int nE = fes.GetNE();
   Array<int> dofs;

   for (int e = 0; e < nE; e++)
   {
      auto element = fes.GetFE(e);
      auto eltrans = fes.GetElementTransformation(e);

      // assemble element matrix of convection operator
      conv_int.AssembleElementMatrix(*element, *eltrans, Ke);

      fes.GetElementDofs(e, dofs);
      ue.SetSize(dofs.Size());
      u.GetSubVector(dofs, ue);
      re.SetSize(dofs.Size());
      re = 0.0;

      for (int i = 0; i < dofs.Size(); i++)
      {
         for (int j = 0; j < i; j++)
         {
            // add low-order stabilization with discrete upwinding
            real_t dije = std::max(std::max(Ke(i,j), Ke(j,i)), real_t(0.0));
            real_t diffusion = dije * (ue(j) - ue(i));

            re(i) += diffusion;
            re(j) -= diffusion;
         }
      }
      // Add -K_e u_e to obtain (-K_e + D_e) u_e and add element contribution
      // to global vector
      Ke.AddMult(ue, re, -1.0);
      udot.AddElementVector(dofs, re);
   }

   // add boundary condition (u - u_inflow) * b.
   // This is under the assumption that b_lumped has been updated
   subtract(u, u_inflow, z);
   z *= b_lumped;
   udot += z;

#ifdef MFEM_USE_MPI
   if (pfes)
   {
      // Sum over the shared DOFs.
      Array<real_t> udot_array(udot.GetData(), udot.Size());
      pfes->GroupComm().Reduce<real_t>(udot_array, GroupCommunicator::Sum);
      pfes->GroupComm().Bcast(udot_array);
   }
#endif

   // apply inverse lumped mass matrix
   udot /= lumpedmassmatrix;
}

void CG_FE_Evolution::ComputeLOTimeStepEstimates(real_t &dt_local,
                                                 real_t &dt_global) const
{
   // Elementwise (unassembled) estimate.
   dt_local = std::numeric_limits<real_t>::infinity();

   // Globally assembled diagonal of K_LO (in the DOF numbering of 'fes').
   Vector kdiag(lumpedmassmatrix.Size());
   kdiag = 0.0;

   const int nE = fes.GetNE();
   Array<int> dofs;

   for (int e = 0; e < nE; e++)
   {
      auto element = fes.GetFE(e);
      auto eltrans = fes.GetElementTransformation(e);

      // Assemble element matrices for the LO operator:
      //   K_LO,e = (-K_e + D_e),
      // where D_e is the discrete upwinding diffusion constructed from K_e.
      conv_int.AssembleElementMatrix(*element, *eltrans, Ke);
      mass_int.AssembleElementMatrix(*element, *eltrans, Me);

      fes.GetElementDofs(e, dofs);
      const int nd = dofs.Size();

      Vector me(nd), kdiag_e(nd);
      for (int i = 0; i < nd; i++)
      {
         // m_i^e = sum_j (M_e)_{ij}  (row-sum lumping)
         real_t mi = 0.0;
         for (int j = 0; j < nd; j++) { mi += Me(i, j); }
         me(i) = mi;

         // Start with the diagonal from -K_e.
         kdiag_e(i) = -Ke(i, i);
      }

      // Add diagonal contributions from the diffusion D_e:
      // D_e has off-diagonal entries d_ij >= 0 and diagonal entries
      // (D_e)_{ii} = -sum_{j != i} d_ij.
      for (int i = 0; i < nd; i++)
      {
         for (int j = 0; j < i; j++)
         {
            const real_t d_ij =
               std::max(std::max(Ke(i, j), Ke(j, i)), real_t(0.0));
            kdiag_e(i) -= d_ij;
            kdiag_e(j) -= d_ij;
         }
      }

      // Per-element time step estimate (ignores overlap).
      for (int i = 0; i < nd; i++)
      {
         if (kdiag_e(i) < 0.0)
         {
            dt_local = std::min(dt_local, me(i) / (-kdiag_e(i)));
         }
      }

      // Contribute to the globally assembled diagonal.
      for (int i = 0; i < nd; i++) { kdiag(dofs[i]) += kdiag_e(i); }
   }

   // Add boundary flow contribution (diagonal) from the LO evolution operator.
   // Note: In this example set-up this is typically zero.
   kdiag += b_lumped;

#ifdef MFEM_USE_MPI
   if (pfes)
   {
      // Sum K_LO diagonal contributions over shared DOFs.
      Array<real_t> kdiag_array(kdiag.GetData(), kdiag.Size());
      pfes->GroupComm().Reduce<real_t>(kdiag_array, GroupCommunicator::Sum);
      pfes->GroupComm().Bcast(kdiag_array);
   }
#endif

   // Global (assembled) estimate.
   dt_global = std::numeric_limits<real_t>::infinity();
   for (int i = 0; i < kdiag.Size(); i++)
   {
      if (kdiag(i) < 0.0)
      {
         dt_global = std::min(dt_global, lumpedmassmatrix(i) / (-kdiag(i)));
      }
   }

#ifdef MFEM_USE_MPI
   if (pfes)
   {
      // Reduce to a global minimum over all MPI ranks.
      real_t dt_local_glob = dt_local;
      real_t dt_global_glob = dt_global;
      MPI_Allreduce(&dt_local, &dt_local_glob, 1, MPITypeMap<real_t>::mpi_type,
                    MPI_MIN, pfes->GetComm());
      MPI_Allreduce(&dt_global, &dt_global_glob, 1,
                    MPITypeMap<real_t>::mpi_type, MPI_MIN, pfes->GetComm());
      dt_local = dt_local_glob;
      dt_global = dt_global_glob;
   }
#endif
}

void HighOrderTargetScheme::Mult(const Vector &x, Vector &y) const
{
   y = 0.0;

   // compute low-order time derivative for high-order stabilization
   ComputeLOTimeDerivatives(x, udot);

   Array<int> dofs;
   for (int e = 0; e < fes.GetNE(); e++)
   {
      auto element = fes.GetFE(e);
      auto eltrans = fes.GetElementTransformation(e);

      // assemble element mass and convection matrices
      conv_int.AssembleElementMatrix(*element, *eltrans, Ke);
      mass_int.AssembleElementMatrix(*element, *eltrans, Me);

      fes.GetElementDofs(e, dofs);
      ue.SetSize(dofs.Size());
      re.SetSize(dofs.Size());
      udote.SetSize(dofs.Size());

      x.GetSubVector(dofs, ue);
      udot.GetSubVector(dofs, udote);

      re = 0.0;
      for (int i = 0; i < dofs.Size(); i++)
      {
         for (int j = 0; j < i; j++)
         {
            // add high-order stabilization without correction for low-order
            // stabilization
            real_t fije = Me(i,j) * (udote(i) - udote(j));
            re(i) += fije;
            re(j) -= fije;
         }
      }

      // add convective term and add to global vector
      Ke.AddMult(ue, re, -1.0);
      y.AddElementVector(dofs, re);
   }

   // add boundary condition (u - u_inflow) * b (u - u_inflow) * b
   subtract(x, u_inflow, z);
   z *= b_lumped;
   y += z;

#ifdef MFEM_USE_MPI
   if (pfes)
   {
      // Sum over the shared DOFs.
      Array<real_t> y_array(y.GetData(), y.Size());
      pfes->GroupComm().Reduce<real_t>(y_array, GroupCommunicator::Sum);
      pfes->GroupComm().Bcast(y_array);
   }
#endif

   // apply inverse lumped mass matrix
   y /= lumpedmassmatrix;
}

void ClipAndScale::ComputeBounds(const Vector &u,
                                 Array<real_t> &u_min,
                                 Array<real_t> &u_max) const
{
   // iterate over local number of dofs on this processor
   // and compute maximum and minimum over local stencil
   for (int i = 0; i < fes.GetVSize(); i++)
   {
      umin[i] = u(i);
      umax[i] = u(i);

      for (int k = I[i]; k < I[i+1]; k++)
      {
         int j = J[k];
         umin[i] = std::min(umin[i], u(j));
         umax[i] = std::max(umax[i], u(j));
      }
   }

#ifdef MFEM_USE_MPI
   if (pfes)
   {
      // Reduce min and max over the shared DOFs.
      pfes->GroupComm().Reduce<real_t>(umax, GroupCommunicator::Max);
      pfes->GroupComm().Bcast(umax);
      pfes->GroupComm().Reduce<real_t>(umin, GroupCommunicator::Min);
      pfes->GroupComm().Bcast(umin);
   }
#endif
}

void ClipAndScale::Mult(const Vector &x, Vector &y) const
{
   y = 0.0;

   // compute low-order time derivative for high-order
   // stabilization and local bounds
   ComputeLOTimeDerivatives(x, udot);
   ComputeBounds(x, umin, umax);

   Array<int> dofs;
   for (int e = 0; e < fes.GetNE(); e++)
   {
      auto element = fes.GetFE(e);
      auto eltrans = fes.GetElementTransformation(e);

      // assemble element mass and convection matrices
      conv_int.AssembleElementMatrix(*element, *eltrans, Ke);
      mass_int.AssembleElementMatrix(*element, *eltrans, Me);

      fes.GetElementDofs(e, dofs);
      ue.SetSize(dofs.Size());
      re.SetSize(dofs.Size());
      udote.SetSize(dofs.Size());
      fe.SetSize(dofs.Size());
      fe_star.SetSize(dofs.Size());
      gammae.SetSize(dofs.Size());

      x.GetSubVector(dofs, ue);
      udot.GetSubVector(dofs, udote);

      re = 0.0;
      fe = 0.0;
      gammae = 0.0;
      for (int i = 0; i < dofs.Size(); i++)
      {
         for (int j = 0; j < i; j++)
         {
            // add low-order diffusion
            real_t dije = std::max(std::max(Ke(i,j), Ke(j,i)), real_t(0.0));
            real_t diffusion = dije * (ue(j) - ue(i));

            re(i) += diffusion;
            re(j) -= diffusion;

            // for bounding fluxes
            gammae(i) += dije;
            gammae(j) += dije;

            // assemble raw antidifussive fluxes
            // note that fije = - fjie
            real_t fije = Me(i,j) * (udote(i) - udote(j)) - diffusion;
            fe(i) += fije;
            fe(j) -= fije;
         }
      }

      // add convective term
      Ke.AddMult(ue, re, -1.0);

      gammae *= 2.0;

      real_t P_plus = 0.0;
      real_t P_minus = 0.0;

      //Clip
      for (int i = 0; i < dofs.Size(); i++)
      {
         // bounding fluxes to enforce u_i = u_i_min implies du/dt >= 0
         // and u_i = u_i_max implies du/dt <= 0
         real_t fie_max = gammae(i) * (umax[dofs[i]] - ue(i));
         real_t fie_min = gammae(i) * (umin[dofs[i]] - ue(i));

         fe_star(i) = std::min(std::max(fie_min, fe(i)), fie_max);

         // track positive and negative contributions s
         P_plus += std::max(fe_star(i), real_t(0.0));
         P_minus += std::min(fe_star(i), real_t(0.0));
      }
      const real_t P = P_minus + P_plus;

      //and Scale for the sum of fe_star to be 0, i.e., mass conservation
      for (int i = 0; i < dofs.Size(); i++)
      {
         if (fe_star(i) > 0.0 && P > 0.0)
         {
            fe_star(i) *= - P_minus / P_plus;
         }
         else if (fe_star(i) < 0.0 && P < 0.0)
         {
            fe_star(i) *= - P_plus / P_minus;
         }
      }
      // add limited antidiffusive fluxes to element contribution
      // and add to global vector
      re += fe_star;
      y.AddElementVector(dofs, re);
   }

   // add boundary condition (u - u_inflow) * b
   subtract(x, u_inflow, z);
   z *= b_lumped;
   y += z;

#ifdef MFEM_USE_MPI
   if (pfes)
   {
      // Sum over the shared DOFs.
      Array<real_t> y_array(y.GetData(), y.Size());
      pfes->GroupComm().Reduce<real_t>(y_array, GroupCommunicator::Sum);
      pfes->GroupComm().Bcast(y_array);
   }
#endif

   // apply inverse lumped mass matrix
   y /= lumpedmassmatrix;
}

} // namespace mfem
