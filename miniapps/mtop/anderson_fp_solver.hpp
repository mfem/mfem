
#ifndef MFEM_ANDERSON_FP_SOLVER_HPP
#define MFEM_ANDERSON_FP_SOLVER_HPP
#include "mfem.hpp"

#ifdef MFEM_USE_MPI
#include <mpi.h>
#endif

#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <iomanip>

// Parallel Anderson acceleration for fixed point x = G(x)
// LS via Gram matrix: (A^T A + lambda I) gamma = A^T f
// All dot products use MFEM's device-capable inner product (x * y).
//
// Update form (difference form):
//   f_k = G(x_k) - x_k
//   A = [Δf_{k-p}, ..., Δf_{k-1}]   (n x p)
//   solve gamma = argmin || f_k - A gamma ||_2
//   x_{k+1} = x_k + beta * ( f_k - (Δg) gamma )
//
// Notes:
// - LS communication: ONE MPI_Allreduce on packed (G,b).
// - Local dots use x*y (Vector::operator*(Vector)), device-capable when vectors are UseDevice(true).

class AndersonFixedPointSolverParGramDeviceIP : public mfem::Solver
{
public:
#ifdef MFEM_USE_MPI
   AndersonFixedPointSolverParGramDeviceIP(MPI_Comm comm, int m = 5)
      : mfem::Solver(0, /*iter_mode=*/false), comm_(comm), m_(m)
   {
      MFEM_VERIFY(m_ >= 0, "Anderson: depth m must be >= 0");
      MPI_Comm_rank(comm_, &myid_);
      MPI_Comm_size(comm_, &nprocs_);
   }
#else
   AndersonFixedPointSolverParGramDeviceIP(int m = 5)
      : mfem::Solver(0, /*iter_mode=*/false), m_(m)
   {
      MFEM_VERIFY(m_ >= 0, "Anderson: depth m must be >= 0");
      myid_ = 0; nprocs_ = 1;
   }
#endif

   void SetOperator(const mfem::Operator &op) override
   {
      Gmap_ = &op;
      height = op.Height();
      width  = op.Width();
      MFEM_VERIFY(height == width, "Anderson: Operator must be square.");

      const int n = height;

      xk_.SetSize(n);
      xkp1_.SetSize(n);
      gx_.SetSize(n);
      fk_.SetSize(n);

      g_prev_.SetSize(n);
      f_prev_.SetSize(n);

      df_.SetSize(n);
      dg_.SetSize(n);
      corr_.SetSize(n);

      // Enable device semantics for all vectors that participate in dot-products / axpys.
      // (This does not force GPU; it enables MFEM Device execution where available.) :contentReference[oaicite:3]{index=3}
      EnableDeviceVectors_();

      // Ring buffers
      const int cap = std::max(1, m_);
      dF_.resize(cap);
      dG_.resize(cap);
      for (int i = 0; i < cap; ++i)
      {
         dF_[i].SetSize(n);
         dG_[i].SetSize(n);
         dF_[i].UseDevice(true);
         dG_[i].UseDevice(true);
      }

      // Small dense work (max m x m)
      Gsmall_.SetSize(cap, cap);
      Asmall_.SetSize(cap, cap);
      Vsmall_.SetSize(cap, cap);

      eval_.assign(cap, mfem::real_t(0));

      bsmall_.SetSize(cap);
      ysmall_.SetSize(cap);
      gamma_.SetSize(cap);

      ResetHistory_();
   }

   void Mult(const mfem::Vector &x0, mfem::Vector &x) const override
   {
      MFEM_VERIFY(Gmap_ != nullptr, "Anderson: call SetOperator() first.");
      MFEM_VERIFY(x0.Size() == height, "Anderson: bad input size.");

      ResetHistory_();

      if (iterative_mode) { xk_ = x; }
      else                { xk_ = x0; }

      double res0 = -1.0;
      double res  = 0.0;

      if (print_level_ > 0 && myid_ == 0)
      {
         mfem::out << "AndersonFixedPointSolverParGramDeviceIP: m=" << m_
                   << " beta=" << beta_
                   << " max_it=" << max_iter_
                   << " rtol=" << rel_tol_
                   << " atol=" << abs_tol_
                   << " reg_rel=" << reg_rel_
                   << " rcond=" << rcond_
                   << "\n";
      }

      for (int it = 0; it < max_iter_; ++it)
      {
         // gx = G(xk)
         Gmap_->Mult(xk_, gx_);

         // fk = gx - xk
         fk_ = gx_;
         fk_ -= xk_;

         // global ||fk||
         res = std::sqrt((double)DotGlobal_(fk_, fk_));
         if (it == 0) { res0 = res; }

         if (print_level_ > 0 && myid_ == 0)
         {
            mfem::out << "  it " << std::setw(4) << it
                      << "  ||G(x)-x|| = " << std::scientific << res
                      << "  depth=" << p_
                      << "\n";
         }

         const double tol = std::max(abs_tol_, rel_tol_ * res0);
         if (res <= tol)
         {
            final_iter_ = it;
            final_norm_ = res;
            x.SetSize(height, xk_); // keep memory type consistent with xk_
            x = xk_;
            return;
         }

         // No prev or m=0 -> damped fixed point
         if (!has_prev_ || m_ == 0)
         {
            xkp1_ = xk_;
            xkp1_.Add(beta_, fk_);

            g_prev_ = gx_;
            f_prev_ = fk_;
            has_prev_ = true;

            xk_ = xkp1_;
            continue;
         }

         // df = fk - f_prev, dg = gx - g_prev
         df_ = fk_; df_ -= f_prev_;
         dg_ = gx_; dg_ -= g_prev_;

         PushHistory_(df_, dg_); // updates p_ (<=m_) and ring start_

         // Solve LS via Gram system
         const bool ok = SolveLeastSquares_GramEigen_();
         if (!ok)
         {
            if (print_level_ > 0 && myid_ == 0)
            {
               mfem::out << "  LS solve failed -> restarting history, plain step.\n";
            }
            ResetHistory_();

            xkp1_ = xk_;
            xkp1_.Add(beta_, fk_);

            g_prev_ = gx_;
            f_prev_ = fk_;
            has_prev_ = true;

            xk_ = xkp1_;
            continue;
         }

         // corr = dG * gamma
         corr_ = 0.0;
         for (int j = 0; j < p_; ++j) { corr_.Add(gamma_(j), DGcol_(j)); }

         // x_{k+1} = x_k + beta * ( f_k - corr )
         xkp1_ = xk_;
         xkp1_.Add(beta_, fk_);
         xkp1_.Add(-beta_, corr_);

         // update previous
         g_prev_ = gx_;
         f_prev_ = fk_;
         xk_ = xkp1_;
      }

      final_iter_ = max_iter_;
      final_norm_ = res;
      x.SetSize(height, xk_);
      x = xk_;
   }

   // ---------------- parameters ----------------
   void SetMaxIter(int max_it) { max_iter_ = max_it; }
   void SetRelTol(double rtol) { rel_tol_ = rtol; }
   void SetAbsTol(double atol) { abs_tol_ = atol; }
   void SetBeta(double beta)   { beta_ = beta; }
   void SetPrintLevel(int pl)  { print_level_ = pl; }

   // lambda = reg_rel * trace(G)/p
   void SetRegularizationRel(double reg_rel) { reg_rel_ = reg_rel; }

   // keep eigenvalues >= rcond * lambda_max (set rcond<0 to disable)
   void SetRcond(double rcond) { rcond_ = rcond; }

   void SetDepth(int m)
   {
      MFEM_VERIFY(m >= 0, "Anderson: m must be >= 0");
      m_ = m;
      if (Gmap_ != nullptr) { SetOperator(*Gmap_); }
   }

   int    GetNumIterations() const { return final_iter_; }
   double GetFinalNorm()     const { return final_norm_; }

private:
   const mfem::Operator *Gmap_ = nullptr;

#ifdef MFEM_USE_MPI
   MPI_Comm comm_ = MPI_COMM_WORLD;
#endif
   int myid_ = 0;
   int nprocs_ = 1;

   int m_ = 5;
   int max_iter_ = 50;
   double rel_tol_ = 1e-8;
   double abs_tol_ = 0.0;
   double beta_ = 1.0;

   double reg_rel_ = 1e-12;
   double rcond_   = 1e-12;

   int print_level_ = 0;

   // History ring buffer state
   mutable bool has_prev_ = false;
   mutable int p_ = 0;
   mutable int start_ = 0;

   // Iteration vectors
   mutable mfem::Vector xk_, xkp1_, gx_, fk_;
   mutable mfem::Vector g_prev_, f_prev_;
   mutable mfem::Vector df_, dg_, corr_;

   // Stored columns Δf, Δg
   mutable std::vector<mfem::Vector> dF_, dG_;

   // Small dense work buffers (allocated as max m x m, used as p x p)
   mutable mfem::DenseMatrix Gsmall_, Asmall_, Vsmall_;
   mutable std::vector<mfem::real_t> eval_;
   mutable mfem::Vector bsmall_, ysmall_, gamma_;

   // Stats
   mutable int final_iter_ = 0;
   mutable double final_norm_ = 0.0;

private:
   void EnableDeviceVectors_() const
   {
      xk_.UseDevice(true);
      xkp1_.UseDevice(true);
      gx_.UseDevice(true);
      fk_.UseDevice(true);

      g_prev_.UseDevice(true);
      f_prev_.UseDevice(true);

      df_.UseDevice(true);
      dg_.UseDevice(true);
      corr_.UseDevice(true);
   }

   void ResetHistory_() const
   {
      has_prev_ = false;
      p_ = 0;
      start_ = 0;
   }

   int RingIndex_(int pos) const
   {
      return (start_ + pos) % std::max(1, m_);
   }

   const mfem::Vector& DFcol_(int pos) const { return dF_[RingIndex_(pos)]; }
   const mfem::Vector& DGcol_(int pos) const { return dG_[RingIndex_(pos)]; }

   void PushHistory_(const mfem::Vector &df, const mfem::Vector &dg) const
   {
      if (m_ == 0) { return; }
      const int M = m_;

      if (p_ < M)
      {
         const int idx = (start_ + p_) % M;
         dF_[idx] = df;
         dG_[idx] = dg;
         ++p_;
      }
      else
      {
         const int idx = start_;
         dF_[idx] = df;
         dG_[idx] = dg;
         start_ = (start_ + 1) % M;
         p_ = M;
      }
   }

   // --------- device-capable dot products ----------
   static mfem::real_t DotLocal_(const mfem::Vector &a,
                                                   const mfem::Vector &b)
   {
      // This is x*y (Vector::operator*(Vector)), which is the device-capable inner product. :contentReference[oaicite:4]{index=4}
      MFEM_ASSERT(a.Size() == b.Size(), "Dot: size mismatch.");
      return a * b;
   }

   mfem::real_t DotGlobal_(const mfem::Vector &a, const mfem::Vector &b) const
   {
#ifdef MFEM_USE_MPI
      // MFEM provides an inline MPI_Comm overload: loc = a*b; MPI_Allreduce(...). :contentReference[oaicite:5]{index=5}
      return mfem::InnerProduct(comm_, a, b);
#else
      return mfem::InnerProduct(a, b);
#endif
   }

   // Small symmetric Jacobi eigensolver for p x p matrix (p<=~20 typical).
   static void JacobiEigenSymmetric_(mfem::DenseMatrix &A, mfem::DenseMatrix &V,
                                    mfem::real_t *eval, int p,
                                    int max_sweeps = 50, mfem::real_t tol = 1e-14)
   {
      for (int i = 0; i < p; ++i)
      {
         for (int j = 0; j < p; ++j) { V(i,j) = (i == j) ? 1.0 : 0.0; }
      }

      auto offdiag_max = [&]() {
         mfem::real_t mx = 0.0;
         for (int i = 0; i < p; ++i)
            for (int j = i+1; j < p; ++j)
               mx = std::max(mx, (mfem::real_t)std::abs(A(i,j)));
         return mx;
      };

      for (int sweep = 0; sweep < max_sweeps; ++sweep)
      {
         const mfem::real_t mx = offdiag_max();
         if (mx < tol) { break; }

         for (int q = 1; q < p; ++q)
         {
            for (int r = 0; r < q; ++r)
            {
               const mfem::real_t a_rq = A(r,q);
               if (std::abs(a_rq) < tol) { continue; }

               const mfem::real_t a_rr = A(r,r);
               const mfem::real_t a_qq = A(q,q);

               const mfem::real_t tau = (a_qq - a_rr) / (2.0 * a_rq);
               mfem::real_t t;
               if (tau >= 0.0) { t =  1.0 / (tau + (mfem::real_t)std::sqrt(1.0 + tau*tau)); }
               else            { t = -1.0 / (-tau + (mfem::real_t)std::sqrt(1.0 + tau*tau)); }

               const mfem::real_t c = 1.0 / (mfem::real_t)std::sqrt(1.0 + t*t);
               const mfem::real_t s = t * c;

               for (int k = 0; k < p; ++k)
               {
                  if (k == r || k == q) { continue; }
                  const mfem::real_t a_kr = A(k,r);
                  const mfem::real_t a_kq = A(k,q);

                  const mfem::real_t nr = c*a_kr - s*a_kq;
                  const mfem::real_t nq = s*a_kr + c*a_kq;

                  A(k,r) = A(r,k) = nr;
                  A(k,q) = A(q,k) = nq;
               }

               const mfem::real_t a_rr_new = c*c*a_rr - 2.0*s*c*a_rq + s*s*a_qq;
               const mfem::real_t a_qq_new = s*s*a_rr + 2.0*s*c*a_rq + c*c*a_qq;

               A(r,r) = a_rr_new;
               A(q,q) = a_qq_new;
               A(r,q) = A(q,r) = 0.0;

               for (int k = 0; k < p; ++k)
               {
                  const mfem::real_t v_kr = V(k,r);
                  const mfem::real_t v_kq = V(k,q);
                  V(k,r) = c*v_kr - s*v_kq;
                  V(k,q) = s*v_kr + c*v_kq;
               }
            }
         }
      }

      for (int i = 0; i < p; ++i) { eval[i] = A(i,i); }
   }

   bool SolveLeastSquares_GramEigen_() const
   {
      const int p = p_;
      if (p <= 0) { return false; }

      // pack upper triangle of G plus b: size = p*(p+1)/2 + p
      const int tri = p*(p+1)/2;
      std::vector<mfem::real_t> pack(tri + p, mfem::real_t(0));

      // Local assembly: all dots are local, device-capable x*y (no MPI yet)
      int idx = 0;
      for (int j = 0; j < p; ++j)
      {
         const mfem::Vector &aj = DFcol_(j);
         for (int i = 0; i <= j; ++i)
         {
            pack[idx++] = DotLocal_(DFcol_(i), aj);
         }
      }
      for (int j = 0; j < p; ++j)
      {
         pack[idx++] = DotLocal_(DFcol_(j), fk_);
      }

#ifdef MFEM_USE_MPI
      // One batched allreduce
      MPI_Allreduce(MPI_IN_PLACE, pack.data(), (int)pack.size(),
                    MFEM_MPI_REAL_T, MPI_SUM, comm_);
#endif

      // Unpack into Gsmall and b
      idx = 0;
      for (int j = 0; j < p; ++j)
      {
         for (int i = 0; i <= j; ++i)
         {
            const mfem::real_t val = pack[idx++];
            Gsmall_(i,j) = val;
            Gsmall_(j,i) = val;
         }
      }
      for (int j = 0; j < p; ++j) { bsmall_(j) = pack[idx++]; }

      // lambda = reg_rel * trace(G)/p
      double trace = 0.0;
      for (int i = 0; i < p; ++i) { trace += (double)Gsmall_(i,i); }
      const double lambda = (reg_rel_ > 0.0) ? (reg_rel_ * (trace / (double)p)) : 0.0;

      // Asmall = G + lambda I
      for (int i = 0; i < p; ++i)
      {
         for (int j = 0; j < p; ++j) { Asmall_(i,j) = Gsmall_(i,j); }
      }
      if (lambda > 0.0)
      {
         for (int i = 0; i < p; ++i) { Asmall_(i,i) += (mfem::real_t)lambda; }
      }

      // Eigendecomposition (Jacobi)
      JacobiEigenSymmetric_(Asmall_, Vsmall_, eval_.data(), p);

      // Sort eigenpairs descending
      std::vector<int> perm(p);
      std::iota(perm.begin(), perm.end(), 0);
      std::stable_sort(perm.begin(), perm.end(),
                       [&](int a, int b) { return eval_[a] > eval_[b]; });

      mfem::DenseMatrix Vsorted(p, p);
      std::vector<mfem::real_t> eval_sorted(p, mfem::real_t(0));
      for (int j = 0; j < p; ++j)
      {
         const int oj = perm[j];
         eval_sorted[j] = eval_[oj];
         for (int i = 0; i < p; ++i) { Vsorted(i,j) = Vsmall_(i,oj); }
      }
      for (int j = 0; j < p; ++j)
      {
         eval_[j] = eval_sorted[j];
         for (int i = 0; i < p; ++i) { Vsmall_(i,j) = Vsorted(i,j); }
      }

      mfem::real_t lambda_max = 0.0;
      for (int i = 0; i < p; ++i) { lambda_max = std::max(lambda_max, eval_[i]); }

      if (lambda_max <= 0.0)
      {
         for (int i = 0; i < p; ++i) { gamma_(i) = 0.0; }
         return true;
      }

      const mfem::real_t cutoff = (rcond_ >= 0.0) ? (mfem::real_t)(rcond_ * (double)lambda_max) : mfem::real_t(0);

      // y = V^T b
      for (int i = 0; i < p; ++i)
      {
         mfem::real_t s = 0.0;
         for (int k = 0; k < p; ++k) { s += Vsmall_(k,i) * bsmall_(k); }
         ysmall_(i) = s;
      }

      // z_i = y_i / eval_i with truncation
      for (int i = 0; i < p; ++i)
      {
         const mfem::real_t ei = eval_[i];
         if (ei < cutoff || ei <= 0.0) { ysmall_(i) = 0.0; }
         else                          { ysmall_(i) = ysmall_(i) / ei; }
      }

      // gamma = V z
      for (int k = 0; k < p; ++k)
      {
         mfem::real_t s = 0.0;
         for (int i = 0; i < p; ++i) { s += Vsmall_(k,i) * ysmall_(i); }
         gamma_(k) = s;
      }
      for (int k = p; k < gamma_.Size(); ++k) { gamma_(k) = 0.0; }

      return true;
   }
};

#endif // MFEM_ANDERSON_FP_SOLVER_HPP