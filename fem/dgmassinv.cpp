// Copyright (c) 2010-2022, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#include "dgmassinv.hpp"
#include "bilinearform.hpp"
#include "../general/forall.hpp"

namespace mfem
{

DGMassInverse::DGMassInverse(FiniteElementSpace &fes_, Coefficient *coeff, int btype)
   : Solver(fes_.GetTrueVSize()),
     fes_orig(fes_),
     fec(fes_.GetMaxElementOrder(), fes_.GetMesh()->Dimension(), btype),
     fes(fes_orig.GetMesh(), &fec)
{
   MFEM_VERIFY(fes.IsDGSpace(), "Space must be DG.");
   MFEM_VERIFY(!fes.IsVariableOrder(), "Variable orders not supported.");

   if (coeff) { m = new MassIntegrator(*coeff); }
   else { m = new MassIntegrator; }

   BilinearForm M(&fes);
   M.AddDomainIntegrator(m);
   M.UseExternalIntegrators();
   M.SetAssemblyLevel(AssemblyLevel::PARTIAL);
   M.Assemble();

   diag_inv.SetSize(height);
   M.AssembleDiagonal(diag_inv);

   auto dinv = diag_inv.ReadWrite();
   MFEM_FORALL(i, height, dinv[i] = 1.0/dinv[i]; );

   r.SetSize(height);
   d.SetSize(height);
   z.SetSize(height);
}

DGMassInverse::DGMassInverse(FiniteElementSpace &fes_, int btype)
: DGMassInverse(fes_, nullptr, btype) { }

template<int T_D1D = 0, int T_Q1D = 0>
MFEM_HOST_DEVICE inline
void PAMassApply2D(const int e,
                   const int NE,
                   const double *b_,
                   const double *bt_,
                   const double *d_,
                   const double *x_,
                   double *y_,
                   const int d1d = 0,
                   const int q1d = 0)
{
   const int D1D = T_D1D ? T_D1D : d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;
   MFEM_VERIFY(D1D <= MAX_D1D, "");
   MFEM_VERIFY(Q1D <= MAX_Q1D, "");
   auto B = Reshape(b_, Q1D, D1D);
   auto Bt = Reshape(bt_, D1D, Q1D);
   auto D = Reshape(d_, Q1D, Q1D, NE);
   auto X = Reshape(x_, D1D, D1D, NE);
   auto Y = Reshape(y_, D1D, D1D, NE);

   for (int dy = 0; dy < D1D; ++dy)
   {
      for (int dx = 0; dx < D1D; ++dx)
      {
         Y(dx, dy, e) = 0.0;
      }
   }

   constexpr int max_D1D = T_D1D ? T_D1D : MAX_D1D;
   constexpr int max_Q1D = T_Q1D ? T_Q1D : MAX_Q1D;
   double sol_xy[max_Q1D][max_Q1D];
   for (int qy = 0; qy < Q1D; ++qy)
   {
      for (int qx = 0; qx < Q1D; ++qx)
      {
         sol_xy[qy][qx] = 0.0;
      }
   }
   for (int dy = 0; dy < D1D; ++dy)
   {
      double sol_x[max_Q1D];
      for (int qy = 0; qy < Q1D; ++qy)
      {
         sol_x[qy] = 0.0;
      }
      for (int dx = 0; dx < D1D; ++dx)
      {
         const double s = X(dx,dy,e);
         for (int qx = 0; qx < Q1D; ++qx)
         {
            sol_x[qx] += B(qx,dx)* s;
         }
      }
      for (int qy = 0; qy < Q1D; ++qy)
      {
         const double d2q = B(qy,dy);
         for (int qx = 0; qx < Q1D; ++qx)
         {
            sol_xy[qy][qx] += d2q * sol_x[qx];
         }
      }
   }
   for (int qy = 0; qy < Q1D; ++qy)
   {
      for (int qx = 0; qx < Q1D; ++qx)
      {
         sol_xy[qy][qx] *= D(qx,qy,e);
      }
   }
   for (int qy = 0; qy < Q1D; ++qy)
   {
      double sol_x[max_D1D];
      for (int dx = 0; dx < D1D; ++dx)
      {
         sol_x[dx] = 0.0;
      }
      for (int qx = 0; qx < Q1D; ++qx)
      {
         const double s = sol_xy[qy][qx];
         for (int dx = 0; dx < D1D; ++dx)
         {
            sol_x[dx] += Bt(dx,qx) * s;
         }
      }
      for (int dy = 0; dy < D1D; ++dy)
      {
         const double q2d = Bt(dy,qy);
         for (int dx = 0; dx < D1D; ++dx)
         {
            Y(dx,dy,e) += q2d * sol_x[dx];
         }
      }
   }
}

template<int T_D1D = 0, int T_Q1D = 0>
MFEM_HOST_DEVICE inline
void PAMassApply3D(const int e,
                   const int NE,
                   const double *b_,
                   const double *bt_,
                   const double *d_,
                   const double *x_,
                   double *y_,
                   const int d1d = 0,
                   const int q1d = 0)
{
   const int D1D = T_D1D ? T_D1D : d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;
   MFEM_VERIFY(D1D <= MAX_D1D, "");
   MFEM_VERIFY(Q1D <= MAX_Q1D, "");
   auto B = Reshape(b_, Q1D, D1D);
   auto Bt = Reshape(bt_, D1D, Q1D);
   auto D = Reshape(d_, Q1D, Q1D, Q1D, NE);
   auto X = Reshape(x_, D1D, D1D, D1D, NE);
   auto Y = Reshape(y_, D1D, D1D, D1D, NE);

   for (int dz = 0; dz < D1D; ++dz)
   {
      for (int dy = 0; dy < D1D; ++dy)
      {
         for (int dx = 0; dx < D1D; ++dx)
         {
            Y(dx, dy, dz, e) = 0.0;
         }
      }
   }

   constexpr int max_D1D = T_D1D ? T_D1D : MAX_D1D;
   constexpr int max_Q1D = T_Q1D ? T_Q1D : MAX_Q1D;
   double sol_xyz[max_Q1D][max_Q1D][max_Q1D];
   for (int qz = 0; qz < Q1D; ++qz)
   {
      for (int qy = 0; qy < Q1D; ++qy)
      {
         for (int qx = 0; qx < Q1D; ++qx)
         {
            sol_xyz[qz][qy][qx] = 0.0;
         }
      }
   }
   for (int dz = 0; dz < D1D; ++dz)
   {
      double sol_xy[max_Q1D][max_Q1D];
      for (int qy = 0; qy < Q1D; ++qy)
      {
         for (int qx = 0; qx < Q1D; ++qx)
         {
            sol_xy[qy][qx] = 0.0;
         }
      }
      for (int dy = 0; dy < D1D; ++dy)
      {
         double sol_x[max_Q1D];
         for (int qx = 0; qx < Q1D; ++qx)
         {
            sol_x[qx] = 0;
         }
         for (int dx = 0; dx < D1D; ++dx)
         {
            const double s = X(dx,dy,dz,e);
            for (int qx = 0; qx < Q1D; ++qx)
            {
               sol_x[qx] += B(qx,dx) * s;
            }
         }
         for (int qy = 0; qy < Q1D; ++qy)
         {
            const double wy = B(qy,dy);
            for (int qx = 0; qx < Q1D; ++qx)
            {
               sol_xy[qy][qx] += wy * sol_x[qx];
            }
         }
      }
      for (int qz = 0; qz < Q1D; ++qz)
      {
         const double wz = B(qz,dz);
         for (int qy = 0; qy < Q1D; ++qy)
         {
            for (int qx = 0; qx < Q1D; ++qx)
            {
               sol_xyz[qz][qy][qx] += wz * sol_xy[qy][qx];
            }
         }
      }
   }
   for (int qz = 0; qz < Q1D; ++qz)
   {
      for (int qy = 0; qy < Q1D; ++qy)
      {
         for (int qx = 0; qx < Q1D; ++qx)
         {
            sol_xyz[qz][qy][qx] *= D(qx,qy,qz,e);
         }
      }
   }
   for (int qz = 0; qz < Q1D; ++qz)
   {
      double sol_xy[max_D1D][max_D1D];
      for (int dy = 0; dy < D1D; ++dy)
      {
         for (int dx = 0; dx < D1D; ++dx)
         {
            sol_xy[dy][dx] = 0;
         }
      }
      for (int qy = 0; qy < Q1D; ++qy)
      {
         double sol_x[max_D1D];
         for (int dx = 0; dx < D1D; ++dx)
         {
            sol_x[dx] = 0;
         }
         for (int qx = 0; qx < Q1D; ++qx)
         {
            const double s = sol_xyz[qz][qy][qx];
            for (int dx = 0; dx < D1D; ++dx)
            {
               sol_x[dx] += Bt(dx,qx) * s;
            }
         }
         for (int dy = 0; dy < D1D; ++dy)
         {
            const double wy = Bt(dy,qy);
            for (int dx = 0; dx < D1D; ++dx)
            {
               sol_xy[dy][dx] += wy * sol_x[dx];
            }
         }
      }
      for (int dz = 0; dz < D1D; ++dz)
      {
         const double wz = Bt(dz,qz);
         for (int dy = 0; dy < D1D; ++dy)
         {
            for (int dx = 0; dx < D1D; ++dx)
            {
               Y(dx,dy,dz,e) += wz * sol_xy[dy][dx];
            }
         }
      }
   }
}

template <int DIM, int D1D, int Q1D>
MFEM_HOST_DEVICE inline
void DGMassApply(const int e,
                 const int NE,
                 const double *B,
                 const double *Bt,
                 const double *pa_data,
                 const double *x,
                 double *y,
                 const int d1d = 0,
                 const int q1d = 0)
{
   if (DIM == 2)
   {
      PAMassApply2D<D1D, Q1D>(e, NE, B, Bt, pa_data, x, y, d1d, q1d);
   }
   else if (DIM == 3)
   {
      PAMassApply3D<D1D, Q1D>(e, NE, B, Bt, pa_data, x, y, d1d, q1d);
   }
   else
   {
      MFEM_ABORT("Unsupported dimension.");
   }
}

MFEM_HOST_DEVICE inline
void DGMassPreconditioner(const int e,
                          const int NE,
                          const int ND,
                          const double *dinv,
                          const double *x,
                          double *y)
{
   auto X = Reshape(x, ND, NE);
   auto D = Reshape(dinv, ND, NE);
   auto Y = Reshape(y, ND, NE);
   for (int i = 0; i < ND; ++i)
   {
      Y(i, e) = D(i, e)*X(i, e);
   }
}

MFEM_HOST_DEVICE inline
void DGMassAxpy(const int e,
                const int NE,
                const int ND,
                const double a,
                const double *x,
                const double b,
                const double *y,
                double *z)
{
   auto X = Reshape(x, ND, NE);
   auto Y = Reshape(y, ND, NE);
   auto Z = Reshape(z, ND, NE);
   for (int i = 0; i < ND; ++i) { Z(i, e) = a*X(i, e) + b*Y(i, e); }
}

MFEM_HOST_DEVICE inline
double DGMassDot(const int e,
                 const int NE,
                 const int ND,
                 const double *x,
                 const double *y)
{
   auto X = Reshape(x, ND, NE);
   auto Y = Reshape(y, ND, NE);

   double dot = 0.0;
   for (int i = 0; i < ND; ++i)
   {
      dot += X(i, e)*Y(i, e);
   }
   return dot;

   //    MFEM_SHARED double s_dot[MAX_D1D*MAX_D1D*MAX_D1D];

   //     for(int t=0;t<p_blockSize;++t){
   //       dlong id = t + b*p_blockSize;
   //       s_dot[t] = 0.0;
   //       while (id<N) {
   //         s_dot[t] += x[id]*y[id];
   //         id += p_blockSize*Nblocks;
   //       }
   //     }

   //     MFEM_SYNC_THREAD;

   // #if p_blockSize>512
   //     for(int t=0;t<p_blockSize;++t) if(t<512) s_dot[t] += s_dot[t+512];
   //     MFEM_SYNC_THREAD;
   // #endif

   // #if p_blockSize>256
   //     for(int t=0;t<p_blockSize;++t) if(t<256) s_dot[t] += s_dot[t+256];
   //     MFEM_SYNC_THREAD;
   // #endif

   //     for(int t=0;t<p_blockSize;++t) if(t<128) s_dot[t] += s_dot[t+128];
   //     MFEM_SYNC_THREAD;

   //     for(int t=0;t<p_blockSize;++t) if(t< 64) s_dot[t] += s_dot[t+ 64];
   //     MFEM_SYNC_THREAD;

   //     for(int t=0;t<p_blockSize;++t) if(t< 32) s_dot[t] += s_dot[t+ 32];
   //     MFEM_SYNC_THREAD;

   //     for(int t=0;t<p_blockSize;++t) if(t< 16) s_dot[t] += s_dot[t+ 16];
   //     // MFEM_SYNC_THREAD;

   //     for(int t=0;t<p_blockSize;++t) if(t<  8) s_dot[t] += s_dot[t+  8];
   //     // MFEM_SYNC_THREAD;

   //     for(int t=0;t<p_blockSize;++t) if(t<  4) s_dot[t] += s_dot[t+  4];
   //     // MFEM_SYNC_THREAD;

   //     for(int t=0;t<p_blockSize;++t) if(t<  2) s_dot[t] += s_dot[t+  2];
   //     // MFEM_SYNC_THREAD;

   //     for(int t=0;t<p_blockSize;++t) if(t<  1) dot[b] = s_dot[0] + s_dot[1];
   //   }
}

template<int DIM, bool USE_SMEM = false, int T_D1D = 0, int T_Q1D = 0>
static void DGMassCGIteration(const int NE,
                              const Array<double> &B_,
                              const Array<double> &Bt_,
                              const Vector &pa_data_,
                              const Vector &dinv_,
                              const double rel_tol,
                              const double abs_tol,
                              const int max_iter,
                              const Vector &b_,
                              Vector &r_,
                              Vector &d_,
                              Vector &z_,
                              Vector &u_,
                              const int d1d = 0,
                              const int q1d = 0)
{
   const int ND = pow(d1d, DIM);

   const auto B = B_.Read();
   const auto Bt = Bt_.Read();
   const auto pa_data = pa_data_.Read();
   const auto dinv = dinv_.Read();
   const auto b = b_.Read();
   auto r = r_.Write();
   auto d = d_.Write();
   auto z = z_.Write();
   auto u = u_.ReadWrite();

   printf("  El.     It.    (Br,r)\n");
   printf("=============================\n");
   MFEM_FORALL(e, NE,
   {
      // int final_iter;
      // double final_norm;
      // bool converged;
      auto print_options = IterativeSolver::PrintLevel().All();

      DGMassApply<DIM, T_D1D, T_Q1D>(e, NE, B, Bt, pa_data, u, r, d1d, q1d);
      DGMassAxpy(e, NE, ND, 1.0, b, -1.0, r, r); // r = b - r

      DGMassPreconditioner(e, NE, ND, dinv, r, z);
      DGMassAxpy(e, NE, ND, 1.0, z, 0.0, z, d); // d = z

      double nom0 = DGMassDot(e, NE, ND, d, r);
      double nom = nom0;
      // MFEM_ASSERT(IsFinite(nom), "nom = " << nom);

      if (nom < 0.0)
      {
         return; // Not positive definite...
      }
      double r0 = std::max(nom*rel_tol*rel_tol, abs_tol*abs_tol);
      if (nom <= r0)
      {
         // converged = true;
         // final_iter = 0;
         // final_norm = sqrt(nom);
         return;
      }

      DGMassApply<DIM, T_D1D, T_Q1D>(e, NE, B, Bt, pa_data, d, z, d1d, q1d);
      double den = DGMassDot(e, NE, ND, z, d);
      if (den <= 0.0)
      {
         const double d2 = DGMassDot(e, NE, ND, d, d);
         if (d2 > 0.0 /* dot product */ && print_options.warnings)
         {
            mfem::out << "PCG: The operator is not positive definite. (Ad, d) = "
                        << den << '\n';
         }
         if (den == 0.0)
         {
            // converged = false;
            // final_iter = 0;
            // final_norm = sqrt(nom);
            return;
         }
      }

      // start iteration
      int i = 1;
      while (true)
      {
         const double alpha = nom/den;
         DGMassAxpy(e, NE, ND, 1.0, u, alpha, d, u); // u = u + alpha*d
         DGMassAxpy(e, NE, ND, 1.0, r, -alpha, z, r); // r = r - alpha*A*d

         DGMassPreconditioner(e, NE, ND, dinv, r, z);

         double betanom = DGMassDot(e, NE, ND, r, z);
         MFEM_ASSERT(IsFinite(betanom), "betanom = " << betanom);
         if (betanom < 0.0)
         {
            if (print_options.warnings)
            {
               mfem::out << "PCG: The preconditioner is not positive definite. (Br, r) = "
                           << betanom << '\n';
            }
            // converged = false;
            // final_iter = i;
            return;
         }

         if (print_options.iterations)
         {
            printf(" %4d    %4d    %10.6e\n", e, i, betanom);
         }

         if (betanom <= r0)
         {
            // converged = true;
            // final_iter = i;
            return;
         }

         if (++i > max_iter) { return; }

         const double beta = betanom/nom;
         DGMassAxpy(e, NE, ND, 1.0, z, beta, d, d); // d = z + beta*d
         DGMassApply<DIM, T_D1D, T_Q1D>(e, NE, B, Bt, pa_data, d, z, d1d, q1d); // z = A d
         den = DGMassDot(e, NE, ND, d, z);
         MFEM_ASSERT(IsFinite(den), "den = " << den);
         if (den <= 0.0)
         {
            const double d2 = DGMassDot(e, NE, ND, d, d);
            if (d2 && print_options.warnings)
            {
               mfem::out << "PCG: The operator is not positive definite. (Ad, d) = "
                           << den << '\n';
            }
            if (den == 0.0)
            {
               // final_iter = i;
               return;
            }
         }
         nom = betanom;
      }
   });
}

void DGMassInverse::Mult(const Vector &Mu, Vector &u) const
{
   const int dim = fes.GetMesh()->Dimension();
   const int NE = fes.GetNE();
   const int d1d = m->dofs1D;
   const int q1d = m->quad1D;
   const auto &pa_data = m->pa_data;
   const auto &B = m->maps->B;
   const auto &Bt = m->maps->Bt;

   if (dim == 2)
   {
      DGMassCGIteration<2>(NE, B, Bt, pa_data, diag_inv, rel_tol, abs_tol, max_iter,
                           Mu, r, d, z, u, d1d, q1d);
   }
   else if (dim == 3)
   {
      DGMassCGIteration<3>(NE, B, Bt, pa_data, diag_inv, rel_tol, abs_tol, max_iter,
                           Mu, r, d, z, u, d1d, q1d);
   }
}

void DGMassInverse::SetOperator(const Operator &op)
{
   MFEM_ABORT("SetOperator not supported with DGMassInverse.")
}

void DGMassInverse::SetRelTol(const double rel_tol_) { rel_tol = rel_tol_; }

void DGMassInverse::SetAbsTol(const double abs_tol_) { abs_tol = abs_tol_; }

void DGMassInverse::SetMaxIter(const double max_iter_) { max_iter = max_iter_; }

DGMassInverse::~DGMassInverse()
{
   delete m;
}

} // namespace mfem
