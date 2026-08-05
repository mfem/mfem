#include "MMA_Equality_MFEM.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace mfem_mma {
namespace {

/// Validate sizes for an Update()/UpdateGCMMA()/KKTresidual() call.
void CheckInput(int n, int m, const mfem::Vector &x,
                const mfem::Vector &df0dx, const mfem::Vector &hval,
                const mfem::Vector *dhdx, const mfem::Vector &xmin,
                const mfem::Vector &xmax)
{
   MFEM_VERIFY(n >= 0 && m >= 0, "negative MMA equality problem size");
   MFEM_VERIFY(x.Size() == n && df0dx.Size() == n &&
               xmin.Size() == n && xmax.Size() == n,
               "MMA equality design-vector size mismatch");
   MFEM_VERIFY(hval.Size() == m, "MMA equality value-vector size mismatch");
   MFEM_VERIFY(m == 0 || dhdx != nullptr,
               "MMA equality gradients are required");
   for (int i = 0; i < m; ++i)
   {
      MFEM_VERIFY(dhdx[i].Size() == n,
                  "MMA equality gradient size mismatch");
   }
}

/// Validate sizes for a RestoreFeasibility() call.
void CheckRestorationInput(int n, int m, const mfem::Vector &x,
                           const mfem::Vector &hval,
                           const mfem::Vector *dhdx,
                           const mfem::Vector &xmin,
                           const mfem::Vector &xmax)
{
   MFEM_VERIFY(n>=0 && m>=0, "negative MMA equality problem size");
   MFEM_VERIFY(x.Size()==n && xmin.Size()==n && xmax.Size()==n,
               "MMA equality restoration vector size mismatch");
   MFEM_VERIFY(hval.Size()==m,
               "MMA equality restoration value-vector size mismatch");
   MFEM_VERIFY(m==0 || dhdx!=nullptr,
               "MMA equality restoration gradients are required");
   for(int i=0;i<m;++i)
      MFEM_VERIFY(dhdx[i].Size()==n,
                  "MMA equality restoration gradient size mismatch");
}

/// Resize @p v to length @p n, set its device flag, and zero it.
void InitVector(mfem::Vector &v, int n, bool use_device)
{
   v.SetSize(n);
   v.UseDevice(use_device);
   v = 0.0;
}

/**
 * Update the MMA asymptotes L/U and move-limit box alpha/beta for one
 * design variable set, following the classic Svanberg oscillation rule:
 * asymptotes contract on sign changes between consecutive steps (xo1,xo2)
 * and expand on monotonic progress, clamped to [1e-4, 100] times the
 * variable's range. alpha/beta are then set to a fraction of the distance
 * to L/U, clipped to [xmin, xmax] and a 0.5*range move limit.
 */
void UpdateAsymptotes(int n, int iter, double asyminit, double asymdec,
                      double asyminc, const mfem::Vector &x,
                      const mfem::Vector &xo1, const mfem::Vector &xo2,
                      const mfem::Vector &xmin, const mfem::Vector &xmax,
                      mfem::Vector &L, mfem::Vector &U,
                      mfem::Vector &alpha, mfem::Vector &beta)
{
   const bool use_dev=x.UseDevice();
   const mfem::real_t *xp=x.Read(), *x1=xo1.Read(), *x2=xo2.Read();
   const mfem::real_t *xmn=xmin.Read(), *xmx=xmax.Read();
   mfem::real_t *lp=L.ReadWrite(), *up=U.ReadWrite();
   mfem::real_t *ap=alpha.Write(), *bp=beta.Write();

   mfem::forall_switch(use_dev,n,[=] MFEM_HOST_DEVICE (int j)
   {
      const double xj = double(xp[j]);
      const double range = fmax(double(xmx[j]) - double(xmn[j]), 1e-12);
      double lj, uj;
      if (iter < 2)
      {
         lj = xj - asyminit*range;
         uj = xj + asyminit*range;
      }
      else
      {
         const double product = (xj-double(x1[j]))*
                                (double(x1[j])-double(x2[j]));
         const double gamma = product < 0.0 ? asymdec :
                              product > 0.0 ? asyminc : 1.0;
         lj = xj - gamma*(double(x1[j])-double(lp[j]));
         uj = xj + gamma*(double(up[j])-double(x1[j]));
         lj = fmax(xj-100.0*range, fmin(lj, xj-1e-4*range));
         uj = fmin(xj+100.0*range, fmax(uj, xj+1e-4*range));
      }
      lp[j] = mfem::real_t(lj);
      up[j] = mfem::real_t(uj);
      ap[j] = mfem::real_t(fmax(double(xmn[j]),
                          fmax(lj+0.1*(xj-lj), xj-0.5*range)));
      bp[j] = mfem::real_t(fmin(double(xmx[j]),
                          fmin(uj-0.1*(uj-xj), xj+0.5*range)));
   });
}

/**
 * Build the separable MMA objective coefficients p0/q0 from the current
 * gradient df0dx and asymptotes L/U, following Svanberg's convex MMA
 * approximation. @p rho adds a curvature floor shared by both p0 and q0
 * (see ObjectiveRho()); rho=0 reduces to a small fixed regularization that
 * only guards against a zero-gradient direction having zero curvature.
 */
void BuildObjective(int n, const mfem::Vector &x,
                    const mfem::Vector &df0dx,
                    const mfem::Vector &xmin, const mfem::Vector &xmax,
                    const mfem::Vector &L, const mfem::Vector &U,
                    mfem::Vector &p0, mfem::Vector &q0,
                    double rho=0.0)
{
   const bool use_dev=x.UseDevice();
   const mfem::real_t *xp=x.Read(), *gp=df0dx.Read();
   const mfem::real_t *xmn=xmin.Read(), *xmx=xmax.Read();
   const mfem::real_t *lp=L.Read(), *up=U.Read();
   mfem::real_t *pp=p0.Write(), *qp=q0.Write();
   mfem::forall_switch(use_dev,n,[=] MFEM_HOST_DEVICE (int j)
   {
      const double g=double(gp[j]);
      const double range=fmax(double(xmx[j])-double(xmn[j]),1e-5);
      const double reg=0.001*fabs(g)+(rho+1e-5)/range;
      const double du=double(up[j])-double(xp[j]);
      const double dl=double(xp[j])-double(lp[j]);
      pp[j]=mfem::real_t(du*du*(fmax(g,0.0)+reg));
      qp[j]=mfem::real_t(dl*dl*(fmax(-g,0.0)+reg));
   });
}

/**
 * Baseline GCMMA curvature parameter: half the design-size-averaged,
 * range-weighted gradient magnitude, floored at 1e-6. Used as the initial
 * @p rho passed to BuildObjective() and grown on rejected GCMMA attempts
 * (see MMAEqualityOptimizer::UpdateGCMMA()). When @p parallel is set, the
 * weighted-gradient sum and design size are reduced across @p comm.
 */
double ObjectiveRho(int n,const mfem::Vector &df0dx,
                    const mfem::Vector &xmin,const mfem::Vector &xmax,
                    bool parallel
#ifdef MFEM_USE_MPI
                    ,MPI_Comm comm=MPI_COMM_NULL
#endif
                    )
{
   mfem::Vector work(n); work.UseDevice(df0dx.UseDevice());
   const mfem::real_t *gp=df0dx.Read();
   const mfem::real_t *xmn=xmin.Read(),*xmx=xmax.Read();
   mfem::real_t *wp=work.Write();
   mfem::forall_switch(df0dx.UseDevice(),n,[=] MFEM_HOST_DEVICE (int j)
   { wp[j]=mfem::real_t(fabs(double(gp[j]))*
                        fmax(double(xmx[j])-double(xmn[j]),1e-12)); });
   double local=double(work.Sum());
   double global=local;
   long long global_n=n;
#ifdef MFEM_USE_MPI
   if(parallel)
   {
      MPI_Allreduce(&local,&global,1,MPI_DOUBLE,MPI_SUM,comm);
      long long nl=n;
      MPI_Allreduce(&nl,&global_n,1,MPI_LONG_LONG,MPI_SUM,comm);
   }
#else
   (void)parallel;
#endif
   return std::max(1e-6,0.5*global/double(std::max<long long>(global_n,1)));
}

/**
 * Value of the separable MMA objective model P_k(candidate), normalized so
 * that P_k(xk) == f0val. When @p parallel is set the per-coordinate sum is
 * reduced across @p comm before adding f0val (added once, not per rank).
 */
double ObjectiveModelValue(int n,const mfem::Vector &xk,
                           const mfem::Vector &candidate,
                           const mfem::Vector &L,const mfem::Vector &U,
                           const mfem::Vector &p0,const mfem::Vector &q0,
                           double f0val,bool parallel
#ifdef MFEM_USE_MPI
                           ,MPI_Comm comm=MPI_COMM_NULL
#endif
                           )
{
   mfem::Vector work(n); work.UseDevice(xk.UseDevice());
   const mfem::real_t *xkp=xk.Read(),*xp=candidate.Read();
   const mfem::real_t *lp=L.Read(),*up=U.Read();
   const mfem::real_t *pp=p0.Read(),*qp=q0.Read();
   mfem::real_t *wp=work.Write();
   mfem::forall_switch(xk.UseDevice(),n,[=] MFEM_HOST_DEVICE (int j)
   {
      wp[j]=mfem::real_t(double(pp[j])/(double(up[j])-double(xp[j]))+
             double(qp[j])/(double(xp[j])-double(lp[j]))-
             double(pp[j])/(double(up[j])-double(xkp[j]))-
             double(qp[j])/(double(xkp[j])-double(lp[j])));
   });
   double local=double(work.Sum());
   double global=local;
#ifdef MFEM_USE_MPI
   if(parallel) MPI_Allreduce(&local,&global,1,MPI_DOUBLE,MPI_SUM,comm);
#else
   (void)parallel;
#endif
   return f0val+global;
}

/**
 * L1 norm of the affine equality model h(xk) + J(xk)*(candidate-xk),
 * i.e. how well the candidate satisfies the linearized equalities used by
 * the subproblem. Used as the constraint-violation term of the GCMMA merit
 * function. Gradient dot products are reduced across @p comm when
 * @p parallel is set.
 */
double AffineResidualNorm(int n,int m,const mfem::Vector &xk,
                          const mfem::Vector &candidate,
                          const mfem::Vector &hval,
                          const mfem::Vector *dhdx,bool parallel
#ifdef MFEM_USE_MPI
                          ,MPI_Comm comm=MPI_COMM_NULL
#endif
                          )
{
   mfem::Vector delta(n); delta.UseDevice(xk.UseDevice());
   const mfem::real_t *xkp=xk.Read(),*xp=candidate.Read();
   mfem::real_t *dp=delta.Write();
   mfem::forall_switch(xk.UseDevice(),n,[=] MFEM_HOST_DEVICE (int j)
   { dp[j]=xp[j]-xkp[j]; });
   std::vector<double> local(m,0.0),global(m,0.0);
   for(int i=0;i<m;++i)
      local[i]=double(mfem::InnerProduct(dhdx[i],delta));
   global=local;
#ifdef MFEM_USE_MPI
   if(parallel && m>0)
      MPI_Allreduce(local.data(),global.data(),m,MPI_DOUBLE,MPI_SUM,comm);
#else
   (void)parallel;
#endif
   double norm1=0.0;
   for(int i=0;i<m;++i) norm1+=std::abs(double(hval(i))+global[i]);
   return norm1;
}

/// L1 norm of a (small, host-resident) equality-value vector, e.g. hval.
double Norm1(const mfem::Vector &values)
{
   double result=0.0;
   for(int i=0;i<values.Size();++i) result+=std::abs(double(values(i)));
   return result;
}

/// Halve the move-limit box [alpha,beta] around xk in place, used after a
/// rejected GCMMA attempt to shrink the trust region for the next retry.
void ContractMoveLimits(int n,const mfem::Vector &xk,
                        mfem::Vector &alpha,mfem::Vector &beta)
{
   const mfem::real_t *xp=xk.Read();
   mfem::real_t *ap=alpha.ReadWrite(),*bp=beta.ReadWrite();
   mfem::forall_switch(xk.UseDevice(),n,[=] MFEM_HOST_DEVICE (int j)
   {
      ap[j]=mfem::real_t(double(xp[j])-0.5*(double(xp[j])-double(ap[j])));
      bp[j]=mfem::real_t(double(xp[j])+0.5*(double(bp[j])-double(xp[j])));
   });
}

/**
 * Squared Euclidean norm of (candidate-xk) with each coordinate scaled by
 * its move-limit range, used as the step-size denominator when growing the
 * GCMMA curvature parameter after a rejected attempt. Reduced across
 * @p comm when @p parallel is set.
 */
double ScaledStepNorm2(int n,const mfem::Vector &xk,
                       const mfem::Vector &candidate,
                       const mfem::Vector &xmin,const mfem::Vector &xmax,
                       bool parallel
#ifdef MFEM_USE_MPI
                       ,MPI_Comm comm=MPI_COMM_NULL
#endif
                       )
{
   mfem::Vector work(n); work.UseDevice(xk.UseDevice());
   const mfem::real_t *xkp=xk.Read(),*xp=candidate.Read();
   const mfem::real_t *xmn=xmin.Read(),*xmx=xmax.Read();
   mfem::real_t *wp=work.Write();
   mfem::forall_switch(xk.UseDevice(),n,[=] MFEM_HOST_DEVICE (int j)
   {
      const double range=fmax(double(xmx[j])-double(xmn[j]),1e-12);
      const double step=(double(xp[j])-double(xkp[j]))/range;
      wp[j]=mfem::real_t(step*step);
   });
   double local=double(work.Sum());
   double global=local;
#ifdef MFEM_USE_MPI
   if(parallel) MPI_Allreduce(&local,&global,1,MPI_DOUBLE,MPI_SUM,comm);
#else
   (void)parallel;
#endif
   return global;
}

/**
 * Solve the dense n-by-n system a*x = b by partial-pivoted Gaussian
 * elimination; @p a and @p b are taken by value since they are used as
 * scratch. Returns false (leaving @p x unspecified) if a pivot is smaller
 * than 1e-18, i.e. the system is numerically singular.
 */
bool SolveDense(std::vector<double> a, std::vector<double> b,
                int n, std::vector<double> &x)
{
   x.assign(n,0.0);
   for (int k=0; k<n; ++k)
   {
      int pivot=k;
      for (int i=k+1; i<n; ++i)
         if (std::abs(a[i*n+k]) > std::abs(a[pivot*n+k])) { pivot=i; }
      if (std::abs(a[pivot*n+k]) < 1e-18) { return false; }
      if (pivot != k)
      {
         for (int j=k; j<n; ++j) { std::swap(a[k*n+j],a[pivot*n+j]); }
         std::swap(b[k],b[pivot]);
      }
      for (int i=k+1; i<n; ++i)
      {
         const double factor=a[i*n+k]/a[k*n+k];
         for (int j=k+1; j<n; ++j) { a[i*n+j]-=factor*a[k*n+j]; }
         b[i]-=factor*b[k];
      }
   }
   for (int i=n-1; i>=0; --i)
   {
      double value=b[i];
      for (int j=i+1; j<n; ++j) { value-=a[i*n+j]*x[j]; }
      x[i]=value/a[i*n+i];
   }
   return true;
}

/// Immutable inputs to the separable MMA subproblem shared by Evaluate()
/// across the equality-multiplier Newton iteration in SolveSubproblem().
struct SubproblemData
{
   int n=0, m=0;
   const mfem::Vector *xk=nullptr, *h=nullptr, *dh=nullptr;
   const mfem::Vector *L=nullptr, *U=nullptr, *alpha=nullptr, *beta=nullptr;
   const mfem::Vector *p0=nullptr, *q0=nullptr;
#ifdef MFEM_USE_MPI
   MPI_Comm comm=MPI_COMM_NULL;
#endif
   bool parallel=false;
};

/// Scratch design-sized vectors reused by Evaluate() across the Newton and
/// line-search iterations of one SolveSubproblem() call, to avoid
/// reallocating device vectors on every candidate multiplier evaluation.
struct SubproblemWorkspace
{
   mfem::Vector linear, delta, inv_curvature, scaled;
   /// (Re)size all scratch vectors to length @p n with the given device flag.
   void Init(int n, bool use_device)
   {
      linear.SetSize(n); linear.UseDevice(use_device);
      delta.SetSize(n); delta.UseDevice(use_device);
      inv_curvature.SetSize(n); inv_curvature.UseDevice(use_device);
      scaled.SetSize(n); scaled.UseDevice(use_device);
   }
};

/**
 * Given fixed equality multipliers @p lambda, minimize the separable MMA
 * objective plus the linear term lambda^T*dh over the move-limit box
 * [alpha,beta] (each coordinate independently, via closed-form bound
 * checks and bisection on the convex 1-D optimality condition), writing
 * the result into @p x. Computes the resulting affine equality residual
 * r(lambda) = h + J*(x-xk) into @p residual, and, if @p matrix is
 * non-null, the reduced Hessian J*diag(1/curvature)*J^T used for the
 * multiplier Newton step in SolveSubproblem(). Local per-rank sums are
 * MPI-reduced across s.comm when s.parallel is set.
 *
 * @return Euclidean norm of the (globally reduced) residual.
 */
double Evaluate(const SubproblemData &s, const std::vector<double> &lambda,
                mfem::Vector &x, std::vector<double> &residual,
                std::vector<double> *matrix, SubproblemWorkspace &ws)
{
   const bool use_dev=s.xk->UseDevice();
   mfem::Vector &linear=ws.linear, &delta=ws.delta;
   mfem::Vector &inv_curvature=ws.inv_curvature, &scaled=ws.scaled;
   linear=0.0;
   for(int i=0;i<s.m;++i)
   {
      mfem::real_t *lin=linear.ReadWrite();
      const mfem::real_t *gp=s.dh[i].Read();
      const double li=lambda[i];
      mfem::forall_switch(use_dev,s.n,[=] MFEM_HOST_DEVICE (int j)
      { lin[j]+=mfem::real_t(li*double(gp[j])); });
   }
   const mfem::real_t *xkp=s.xk->Read(), *lp=s.L->Read();
   const mfem::real_t *up=s.U->Read(), *ap=s.alpha->Read();
   const mfem::real_t *bp=s.beta->Read(), *pp=s.p0->Read();
   const mfem::real_t *qp=s.q0->Read(), *lin=linear.Read();
   mfem::real_t *out=x.Write(), *dp=delta.Write();
   mfem::real_t *icp=inv_curvature.Write();
   const bool build_matrix=(matrix != nullptr);
   std::vector<double> local_r(s.m,0.0);
   std::vector<double> local_a(s.m*s.m,0.0);

   mfem::forall_switch(use_dev,s.n,[=] MFEM_HOST_DEVICE (int j)
   {
      double lo=double(ap[j]), hi=double(bp[j]), value;
      bool interior=true;
      double du=double(up[j])-lo, dl=lo-double(lp[j]);
      double derivative=double(pp[j])/(du*du)-double(qp[j])/(dl*dl)+double(lin[j]);
      if (derivative>=0.0) { value=lo; interior=false; }
      else
      {
         du=double(up[j])-hi; dl=hi-double(lp[j]);
         derivative=double(pp[j])/(du*du)-double(qp[j])/(dl*dl)+double(lin[j]);
         if (derivative<=0.0) { value=hi; interior=false; }
         else
         {
            for (int it=0;it<60;++it)
            {
               const double mid=0.5*(lo+hi);
               du=double(up[j])-mid; dl=mid-double(lp[j]);
               derivative=double(pp[j])/(du*du)-double(qp[j])/(dl*dl)+double(lin[j]);
               if (derivative>0.0) hi=mid; else lo=mid;
            }
            value=0.5*(lo+hi);
         }
      }
      out[j]=mfem::real_t(value);
      dp[j]=mfem::real_t(value-double(xkp[j]));
      icp[j]=0.0;
      if (build_matrix && interior)
      {
         du=double(up[j])-value;
         dl=value-double(lp[j]);
         const double curvature=2.0*double(pp[j])/(du*du*du)+
                                2.0*double(qp[j])/(dl*dl*dl);
         icp[j]=mfem::real_t(1.0/fmax(curvature,1e-30));
      }
   });

   for(int i=0;i<s.m;++i)
      local_r[i]=double(mfem::InnerProduct(s.dh[i],delta));
   if(matrix)
   {
      const mfem::real_t *inv=inv_curvature.Read();
      for(int i=0;i<s.m;++i)
      {
         const mfem::real_t *gi=s.dh[i].Read();
         mfem::real_t *sp=scaled.Write();
         mfem::forall_switch(use_dev,s.n,[=] MFEM_HOST_DEVICE (int j)
         { sp[j]=mfem::real_t(double(gi[j])*double(inv[j])); });
         for(int k=i;k<s.m;++k)
         {
            const double value=double(mfem::InnerProduct(scaled,s.dh[k]));
            local_a[i*s.m+k]=value;
            local_a[k*s.m+i]=value;
         }
      }
   }

   residual=local_r;
   if (matrix) { *matrix=local_a; }
#ifdef MFEM_USE_MPI
   if (s.parallel && s.m>0)
   {
      MPI_Allreduce(local_r.data(),residual.data(),s.m,MPI_DOUBLE,MPI_SUM,s.comm);
      if (matrix)
         MPI_Allreduce(local_a.data(),matrix->data(),s.m*s.m,MPI_DOUBLE,MPI_SUM,s.comm);
   }
#endif
   for (int i=0;i<s.m;++i) { residual[i]+=double((*s.h)(i)); }
   double norm2=0.0;
   for (double value : residual) { norm2+=value*value; }
   return std::sqrt(norm2);
}

/**
 * Solve the MMA subproblem for the current @p lambda in place: a
 * semismooth Newton iteration on the equality multipliers, each step
 * backtracked (halving theta) until the residual norm decreases, stopping
 * on convergence, a singular reduced system, or a failed line search.
 * When s.m==0 this reduces to a single unconstrained separable
 * minimization. On return @p x holds the resulting design and @p lambda
 * the multipliers that produced it.
 */
void SolveSubproblem(const SubproblemData &s, std::vector<double> &lambda,
                     mfem::Vector &x)
{
   SubproblemWorkspace ws;
   ws.Init(s.n,s.xk->UseDevice());
   if (s.m==0)
   {
      std::vector<double> residual;
      Evaluate(s,lambda,x,residual,nullptr,ws);
      return;
   }
   mfem::Vector trial(x.Size()); trial.UseDevice(x.UseDevice());
   std::vector<double> residual, matrix;
   double norm=Evaluate(s,lambda,x,residual,&matrix,ws);
   const double tolerance=1e-10*(1.0+norm);
   for (int it=0;it<80 && norm>tolerance;++it)
   {
      double scale=1.0;
      for (int i=0;i<s.m;++i) scale=std::max(scale,std::abs(matrix[i*s.m+i]));
      for (int i=0;i<s.m;++i) matrix[i*s.m+i]+=1e-12*scale;
      std::vector<double> step;
      if (!SolveDense(matrix,residual,s.m,step)) { break; }
      std::vector<double> candidate(lambda.size()), trial_residual;
      double theta=1.0, trial_norm=norm;
      for (int ls=0;ls<24;++ls)
      {
         for (int i=0;i<s.m;++i) candidate[i]=lambda[i]+theta*step[i];
         trial_norm=Evaluate(s,candidate,trial,trial_residual,nullptr,ws);
         if (trial_norm < norm) { break; }
         theta*=0.5;
      }
      if (!(trial_norm < norm)) { break; }
      lambda.swap(candidate);
      x=trial;
      norm=Evaluate(s,lambda,x,residual,&matrix,ws);
   }
}

/// Immutable inputs to the affine feasibility-restoration projection
/// shared by EvaluateRestoration() across RestoreAffine()'s Newton loop.
struct RestorationData
{
   int n=0, m=0;
   const mfem::Vector *xk=nullptr, *h=nullptr, *dh=nullptr;
   const mfem::Vector *xmin=nullptr, *xmax=nullptr;
#ifdef MFEM_USE_MPI
   MPI_Comm comm=MPI_COMM_NULL;
#endif
   bool parallel=false;
};

/// Scratch design-sized vectors reused by EvaluateRestoration() across one
/// RestoreAffine() call's Newton and line-search iterations.
struct RestorationWorkspace
{
   mfem::Vector jt_nu, delta, free_weight, scaled;
   /// (Re)size all scratch vectors to length @p n with the given device flag.
   void Init(int n, bool use_device)
   {
      jt_nu.SetSize(n); jt_nu.UseDevice(use_device);
      delta.SetSize(n); delta.UseDevice(use_device);
      free_weight.SetSize(n); free_weight.UseDevice(use_device);
      scaled.SetSize(n); scaled.UseDevice(use_device);
   }
};

/**
 * Given fixed restoration multipliers @p nu, evaluate the stationarity
 * condition of RestoreFeasibility()'s box-constrained projection
 * min 1/2 sum((x-xk)/range)^2 s.t. h(xk)+J(xk)*(x-xk)=0: each unconstrained
 * optimum x = xk - range^2*J^T*nu is clipped to [xmin,xmax] (free_weight
 * is zeroed for clipped coordinates so they drop out of the reduced
 * Hessian). Mirrors Evaluate()'s role but for the RestoreAffine() Newton
 * iteration on @p nu instead of the subproblem's @p lambda. Local per-rank
 * sums are MPI-reduced across s.comm when s.parallel is set.
 *
 * @return Euclidean norm of the (globally reduced) affine residual.
 */
double EvaluateRestoration(const RestorationData &s,
                           const std::vector<double> &nu,
                           mfem::Vector &x,
                           std::vector<double> &residual,
                           std::vector<double> *matrix,
                           RestorationWorkspace &ws)
{
   const bool use_dev=s.xk->UseDevice();
   mfem::Vector &jt_nu=ws.jt_nu, &delta=ws.delta;
   mfem::Vector &free_weight=ws.free_weight, &scaled=ws.scaled;
   jt_nu=0.0;
   for(int i=0;i<s.m;++i)
   {
      mfem::real_t *jp=jt_nu.ReadWrite();
      const mfem::real_t *gp=s.dh[i].Read();
      const double ni=nu[i];
      mfem::forall_switch(use_dev,s.n,[=] MFEM_HOST_DEVICE (int j)
      { jp[j]+=mfem::real_t(ni*double(gp[j])); });
   }
   const mfem::real_t *xkp=s.xk->Read();
   const mfem::real_t *xmn=s.xmin->Read(), *xmx=s.xmax->Read();
   const mfem::real_t *jp=jt_nu.Read();
   mfem::real_t *out=x.Write(), *dp=delta.Write(), *fwp=free_weight.Write();
   const bool build_matrix=(matrix != nullptr);
   std::vector<double> local_r(s.m,0.0), local_a(s.m*s.m,0.0);

   mfem::forall_switch(use_dev,s.n,[=] MFEM_HOST_DEVICE (int j)
   {
      const double range=fmax(double(xmx[j])-double(xmn[j]),1e-12);
      const double winv=range*range;
      const double unconstrained=double(xkp[j])-winv*double(jp[j]);
      const double value=fmax(double(xmn[j]),fmin(unconstrained,double(xmx[j])));
      const bool free=(unconstrained>double(xmn[j]) &&
                       unconstrained<double(xmx[j]));
      out[j]=mfem::real_t(value);
      dp[j]=mfem::real_t(value-double(xkp[j]));
      fwp[j]=mfem::real_t(build_matrix && free ? winv : 0.0);
   });
   for(int i=0;i<s.m;++i)
      local_r[i]=double(mfem::InnerProduct(s.dh[i],delta));
   if(matrix)
   {
      const mfem::real_t *fw=free_weight.Read();
      for(int i=0;i<s.m;++i)
      {
         const mfem::real_t *gi=s.dh[i].Read();
         mfem::real_t *sp=scaled.Write();
         mfem::forall_switch(use_dev,s.n,[=] MFEM_HOST_DEVICE (int j)
         { sp[j]=mfem::real_t(double(gi[j])*double(fw[j])); });
         for(int k=i;k<s.m;++k)
         {
            const double value=double(mfem::InnerProduct(scaled,s.dh[k]));
            local_a[i*s.m+k]=value;
            local_a[k*s.m+i]=value;
         }
      }
   }

   residual=local_r;
   if(matrix) *matrix=local_a;
#ifdef MFEM_USE_MPI
   if(s.parallel && s.m>0)
   {
      MPI_Allreduce(local_r.data(),residual.data(),s.m,MPI_DOUBLE,MPI_SUM,s.comm);
      if(matrix)
         MPI_Allreduce(local_a.data(),matrix->data(),s.m*s.m,MPI_DOUBLE,MPI_SUM,s.comm);
   }
#endif
   double norm2=0.0;
   for(int i=0;i<s.m;++i)
   {
      residual[i]+=double((*s.h)(i));
      norm2+=residual[i]*residual[i];
   }
   return std::sqrt(norm2);
}

/**
 * Newton iteration on the restoration multipliers @p nu (analogous to
 * SolveSubproblem(), but for EvaluateRestoration()) that drives @p x
 * towards satisfying the affine equality model while staying in
 * [xmin,xmax]. Returns 0 immediately, leaving @p x unchanged, when there
 * are no equalities.
 *
 * @return Euclidean norm of the remaining affine residual.
 */
double RestoreAffine(const RestorationData &s, mfem::Vector &x,
                     int max_iterations)
{
   MFEM_VERIFY(max_iterations>=0,
               "negative MMA equality restoration iteration limit");
   if(s.m==0) return 0.0;
   RestorationWorkspace ws;
   ws.Init(s.n,s.xk->UseDevice());
   std::vector<double> nu(s.m,0.0), residual, matrix;
   mfem::Vector trial(x.Size()); trial.UseDevice(x.UseDevice());
   double norm=EvaluateRestoration(s,nu,x,residual,&matrix,ws);
   const double tolerance=1e-10*(1.0+norm);
   for(int it=0;it<max_iterations && norm>tolerance;++it)
   {
      double scale=1.0;
      for(int i=0;i<s.m;++i) scale=std::max(scale,std::abs(matrix[i*s.m+i]));
      for(int i=0;i<s.m;++i) matrix[i*s.m+i]+=1e-12*scale;
      std::vector<double> step;
      if(!SolveDense(matrix,residual,s.m,step)) break;
      std::vector<double> candidate(s.m), trial_residual;
      double theta=1.0, trial_norm=norm;
      for(int ls=0;ls<24;++ls)
      {
         for(int i=0;i<s.m;++i) candidate[i]=nu[i]+theta*step[i];
         trial_norm=EvaluateRestoration(s,candidate,trial,trial_residual,nullptr,ws);
         if(trial_norm<norm) break;
         theta*=0.5;
      }
      if(!(trial_norm<norm)) break;
      nu.swap(candidate);
      x=trial;
      norm=EvaluateRestoration(s,nu,x,residual,&matrix,ws);
   }
   return norm;
}

/**
 * Bound-projected KKT stationarity measure backing
 * MMAEqualityOptimizer[Parallel]::KKTresidual(). Forms the Lagrangian
 * gradient df0dx + sum_i lambda_i*dh_i, zeroes its component pushing
 * further into an active bound (within a relative tolerance), and returns
 * the mean squared projected gradient plus the raw sum of squared equality
 * values. Reduced across @p comm when @p parallel is set.
 */
double ProjectedKKT(int n, int m, const mfem::Vector &x,
                    const mfem::Vector &df0dx, const mfem::Vector &hval,
                    const mfem::Vector *dhdx,
                    const mfem::Vector &xmin, const mfem::Vector &xmax,
                    const std::vector<double> &lambda, bool parallel
#ifdef MFEM_USE_MPI
                    , MPI_Comm comm=MPI_COMM_NULL
#endif
                    )
{
   const bool use_dev=x.UseDevice();
   mfem::Vector lagrangian(df0dx),work(n);
   lagrangian.UseDevice(use_dev); work.UseDevice(use_dev);
   for(int i=0;i<m;++i)
   {
      mfem::real_t *lp=lagrangian.ReadWrite();
      const mfem::real_t *gp=dhdx[i].Read();
      const double li=lambda[i];
      mfem::forall_switch(use_dev,n,[=] MFEM_HOST_DEVICE (int j)
      { lp[j]+=mfem::real_t(li*double(gp[j])); });
   }
   const mfem::real_t *xp=x.Read(), *gp=lagrangian.Read();
   const mfem::real_t *xmn=xmin.Read(), *xmx=xmax.Read();
   mfem::real_t *wp=work.Write();
   mfem::forall_switch(use_dev,n,[=] MFEM_HOST_DEVICE (int j)
   {
      double g=double(gp[j]);
      const double tol=1e-3*fmax(double(xmx[j])-double(xmn[j]),1e-12);
      if(double(xp[j])<=double(xmn[j])+tol) g=fmin(g,0.0);
      else if(double(xp[j])>=double(xmx[j])-tol) g=fmax(g,0.0);
      wp[j]=mfem::real_t(g*g);
   });
   double local=double(work.Sum());
   double primal=local;
   long long global_n=n;
#ifdef MFEM_USE_MPI
   if(parallel)
   {
      MPI_Allreduce(&local,&primal,1,MPI_DOUBLE,MPI_SUM,comm);
      long long nl=n;
      MPI_Allreduce(&nl,&global_n,1,MPI_LONG_LONG,MPI_SUM,comm);
   }
#else
   (void)parallel;
#endif
   for(int i=0;i<m;++i) primal+=double(hval(i))*double(hval(i));
   return primal/double(std::max<long long>(global_n,1));
}

} // namespace

// See header for the full contract of every public method below.

MMAEqualityOptimizer::MMAEqualityOptimizer(int n, int m_equalities)
   : n_(n), m_(m_equalities), lambda_(m_equalities,0.0)
{
   MFEM_VERIFY(n>=0 && m_equalities>=0, "negative MMA equality problem size");
}

void MMAEqualityOptimizer::EnsureInitialized_(const mfem::Vector &x)
{
   MFEM_VERIFY(x.Size()==n_, "MMA equality design vector has the wrong size");
   if(p0_.Size()==n_ && (n_>0 || iter_>0)) return;
   const bool use_device=x.UseDevice();
   InitVector(p0_,n_,use_device); InitVector(q0_,n_,use_device);
   InitVector(L_,n_,use_device); InitVector(U_,n_,use_device);
   InitVector(alpha_,n_,use_device); InitVector(beta_,n_,use_device);
   xo1_=x; xo2_=x;
}

void MMAEqualityOptimizer::SetAsymptotes(mfem::real_t i,mfem::real_t d,mfem::real_t inc)
{ asyminit_=double(i); asymdec_=double(d); asyminc_=double(inc); }

// Builds the asymptotes and separable objective at xk, solves for the
// equality multipliers with SolveSubproblem(), and unconditionally
// accepts the result (no globalization, unlike UpdateGCMMA()).
void MMAEqualityOptimizer::Update(mfem::Vector &x,const mfem::Vector &df0dx,
   mfem::real_t,const mfem::Vector &hval,const mfem::Vector *dhdx,
   const mfem::Vector &xmin,const mfem::Vector &xmax)
{
   CheckInput(n_,m_,x,df0dx,hval,dhdx,xmin,xmax);
   EnsureInitialized_(x);
   mfem::Vector xk(x);
   UpdateAsymptotes(n_,iter_,asyminit_,asymdec_,asyminc_,x,xo1_,xo2_,
                    xmin,xmax,L_,U_,alpha_,beta_);
   const double rho=ObjectiveRho(n_,df0dx,xmin,xmax,false);
   BuildObjective(n_,x,df0dx,xmin,xmax,L_,U_,p0_,q0_,rho);
   SubproblemData data;
   data.n=n_; data.m=m_; data.xk=&xk; data.h=&hval; data.dh=dhdx;
   data.L=&L_; data.U=&U_; data.alpha=&alpha_; data.beta=&beta_;
   data.p0=&p0_; data.q0=&q0_;
   SolveSubproblem(data,lambda_,x);
   xo2_=xo1_; xo1_=xk; ++iter_; last_step_accepted_=true;
}

// Retries the subproblem with a grown curvature parameter and a
// contracted move-limit box until the candidate is conservative and
// passes a merit/feasibility/stationarity test, or max_inner is
// exhausted; then falls back to one affine restoration attempt.
void MMAEqualityOptimizer::UpdateGCMMA(mfem::Vector &x,
   const mfem::Vector &df0dx,mfem::real_t f0val,
   const mfem::Vector &hval,const mfem::Vector *dhdx,
   const mfem::Vector &xmin,const mfem::Vector &xmax,
   GCMMAEvalCallback evaluate,int max_inner,int *inner_iterations)
{
   CheckInput(n_,m_,x,df0dx,hval,dhdx,xmin,xmax);
   MFEM_VERIFY(bool(evaluate),"MMA equality GCMMA requires an evaluator");
   MFEM_VERIFY(max_inner>0,"MMA equality GCMMA requires max_inner > 0");
   EnsureInitialized_(x);
   mfem::Vector xk(x);
   UpdateAsymptotes(n_,iter_,asyminit_,asymdec_,asyminc_,xk,xo1_,xo2_,
                    xmin,xmax,L_,U_,alpha_,beta_);
   double rho=ObjectiveRho(n_,df0dx,xmin,xmax,false);
   const std::vector<double> lambda_k=lambda_;
   const double theta0=Norm1(hval);
   bool accepted=false;
   int attempts=0;

   SubproblemData data;
   data.n=n_; data.m=m_; data.xk=&xk; data.h=&hval; data.dh=dhdx;
   data.L=&L_; data.U=&U_; data.alpha=&alpha_; data.beta=&beta_;
   data.p0=&p0_; data.q0=&q0_;

   for(;attempts<max_inner;++attempts)
   {
      BuildObjective(n_,xk,df0dx,xmin,xmax,L_,U_,p0_,q0_,rho);
      x=xk;
      SolveSubproblem(data,lambda_,x);
      mfem::Vector htrial(m_);
      mfem::real_t ftrial=0.0;
      evaluate(x,htrial,ftrial);
      MFEM_VERIFY(htrial.Size()==m_,"GCMMA evaluator returned wrong equality size");

      const double model=ObjectiveModelValue(n_,xk,x,L_,U_,p0_,q0_,
                                             double(f0val),false);
      const double affine=AffineResidualNorm(n_,m_,xk,x,hval,dhdx,false);
      const double theta_trial=Norm1(htrial);
      double sigma=1.0;
      for(double value:lambda_) sigma=std::max(sigma,1.1*std::abs(value));
      const double predicted=double(f0val)+sigma*theta0-
                             (model+sigma*affine);
      const double actual=double(f0val)+sigma*theta0-
                          (double(ftrial)+sigma*theta_trial);
      const double tolerance=1e-10*(1.0+std::abs(double(ftrial)));
      const bool conservative=double(ftrial)<=model+tolerance;
      const bool merit=predicted>1e-14 && actual>=0.1*predicted;
      const bool feasibility=theta_trial<=0.9*theta0;
      const bool stationary=std::abs(predicted)<=1e-14 &&
                            theta_trial<=std::max(theta0,1e-10) &&
                            double(ftrial)<=double(f0val)+tolerance;
      if(conservative && (merit || feasibility || stationary))
      {
         accepted=true;
         ++attempts;
         break;
      }

      const double gap=std::max(0.0,double(ftrial)-model);
      const double distance=ScaledStepNorm2(n_,xk,x,xmin,xmax,false);
      rho=std::max(2.0*rho,1.1*(rho+gap/std::max(distance,1e-12)));
      ContractMoveLimits(n_,xk,alpha_,beta_);
   }

   if(!accepted)
   {
      x=xk;
      RestorationData restoration;
      restoration.n=n_; restoration.m=m_; restoration.xk=&xk;
      restoration.h=&hval; restoration.dh=dhdx;
      restoration.xmin=&xmin; restoration.xmax=&xmax;
      RestoreAffine(restoration,x,80);
      mfem::Vector htrial(m_);
      mfem::real_t ftrial=0.0;
      evaluate(x,htrial,ftrial);
      MFEM_VERIFY(htrial.Size()==m_,
                  "GCMMA restoration evaluator returned wrong equality size");
      if(Norm1(htrial)<theta0*(1.0-1e-4))
      {
         accepted=true;
         std::fill(lambda_.begin(),lambda_.end(),0.0);
      }
      else x=xk;
   }

   if(!accepted) lambda_=lambda_k;

   if(accepted)
   {
      xo2_=xo1_; xo1_=xk; ++iter_;
   }
   last_step_accepted_=accepted;
   if(inner_iterations) *inner_iterations=attempts;
}

// Delegates the box-constrained affine projection to RestoreAffine();
// does not touch iter_, xo1_/xo2_, or lambda_.
mfem::real_t MMAEqualityOptimizer::RestoreFeasibility(
   mfem::Vector &x,const mfem::Vector &hval,const mfem::Vector *dhdx,
   const mfem::Vector &xmin,const mfem::Vector &xmax,int max_iterations)
{
   CheckRestorationInput(n_,m_,x,hval,dhdx,xmin,xmax);
   mfem::Vector xk(x);
   RestorationData data;
   data.n=n_; data.m=m_; data.xk=&xk; data.h=&hval; data.dh=dhdx;
   data.xmin=&xmin; data.xmax=&xmax;
   return mfem::real_t(RestoreAffine(data,x,max_iterations));
}

// Diagnostic only: evaluates ProjectedKKT() at the last-accepted lambda_
// rather than resolving the multipliers.
mfem::real_t MMAEqualityOptimizer::KKTresidual(const mfem::Vector &x,
   const mfem::Vector &df0dx,mfem::real_t,const mfem::Vector &hval,
   const mfem::Vector *dhdx,const mfem::Vector &xmin,
   const mfem::Vector &xmax,double *lambda_out) const
{
   CheckInput(n_,m_,x,df0dx,hval,dhdx,xmin,xmax);
   if(lambda_out) std::copy(lambda_.begin(),lambda_.end(),lambda_out);
   return mfem::real_t(ProjectedKKT(n_,m_,x,df0dx,hval,dhdx,xmin,xmax,
                                    lambda_,false));
}

#ifdef MFEM_USE_MPI
// MPI counterpart of MMAEqualityOptimizer; see header for the full
// contract of every public method below. Reduces n_local to n_global_.
MMAEqualityOptimizerParallel::MMAEqualityOptimizerParallel(
   MPI_Comm comm,int n_local,int m_equalities)
   : comm_(comm),n_local_(n_local),m_(m_equalities),lambda_(m_equalities,0.0)
{
   long long nl=n_local;
   MPI_Allreduce(&nl,&n_global_,1,MPI_LONG_LONG,MPI_SUM,comm_);
}

void MMAEqualityOptimizerParallel::EnsureInitialized_(const mfem::Vector &x)
{
   MFEM_VERIFY(x.Size()==n_local_,"local MMA equality design vector has wrong size");
   if(p0_.Size()==n_local_ && (n_local_>0 || iter_>0)) return;
   const bool use_device=x.UseDevice();
   InitVector(p0_,n_local_,use_device); InitVector(q0_,n_local_,use_device);
   InitVector(L_,n_local_,use_device); InitVector(U_,n_local_,use_device);
   InitVector(alpha_,n_local_,use_device); InitVector(beta_,n_local_,use_device);
   xo1_=x; xo2_=x;
}

void MMAEqualityOptimizerParallel::SetAsymptotes(
   mfem::real_t i,mfem::real_t d,mfem::real_t inc)
{ asyminit_=double(i); asymdec_=double(d); asyminc_=double(inc); }

// Distributed counterpart of MMAEqualityOptimizer::Update(): identical
// control flow, but ObjectiveRho()/SolveSubproblem() reduce across comm_.
void MMAEqualityOptimizerParallel::Update(mfem::Vector &x,
   const mfem::Vector &df0dx,mfem::real_t,const mfem::Vector &hval,
   const mfem::Vector *dhdx,const mfem::Vector &xmin,
   const mfem::Vector &xmax)
{
   CheckInput(n_local_,m_,x,df0dx,hval,dhdx,xmin,xmax);
   EnsureInitialized_(x);
   mfem::Vector xk(x);
   UpdateAsymptotes(n_local_,iter_,asyminit_,asymdec_,asyminc_,x,xo1_,xo2_,
                    xmin,xmax,L_,U_,alpha_,beta_);
   const double rho=ObjectiveRho(n_local_,df0dx,xmin,xmax,true,comm_);
   BuildObjective(n_local_,x,df0dx,xmin,xmax,L_,U_,p0_,q0_,rho);
   SubproblemData data;
   data.n=n_local_; data.m=m_; data.xk=&xk; data.h=&hval; data.dh=dhdx;
   data.L=&L_; data.U=&U_; data.alpha=&alpha_; data.beta=&beta_;
   data.p0=&p0_; data.q0=&q0_; data.parallel=true; data.comm=comm_;
   SolveSubproblem(data,lambda_,x);
   xo2_=xo1_; xo1_=xk; ++iter_; last_step_accepted_=true;
}

// Distributed counterpart of MMAEqualityOptimizer::UpdateGCMMA(); see
// that method's implementation comment. Every reduction-bearing helper is
// called with parallel=true and comm_ so all ranks reach the same
// accept/reject decision.
void MMAEqualityOptimizerParallel::UpdateGCMMA(mfem::Vector &x,
   const mfem::Vector &df0dx,mfem::real_t f0val,
   const mfem::Vector &hval,const mfem::Vector *dhdx,
   const mfem::Vector &xmin,const mfem::Vector &xmax,
   GCMMAEvalCallback evaluate,int max_inner,int *inner_iterations)
{
   CheckInput(n_local_,m_,x,df0dx,hval,dhdx,xmin,xmax);
   MFEM_VERIFY(bool(evaluate),"parallel MMA equality GCMMA requires an evaluator");
   MFEM_VERIFY(max_inner>0,"parallel MMA equality GCMMA requires max_inner > 0");
   EnsureInitialized_(x);
   mfem::Vector xk(x);
   UpdateAsymptotes(n_local_,iter_,asyminit_,asymdec_,asyminc_,xk,xo1_,xo2_,
                    xmin,xmax,L_,U_,alpha_,beta_);
   double rho=ObjectiveRho(n_local_,df0dx,xmin,xmax,true,comm_);
   const std::vector<double> lambda_k=lambda_;
   const double theta0=Norm1(hval);
   bool accepted=false;
   int attempts=0;

   SubproblemData data;
   data.n=n_local_; data.m=m_; data.xk=&xk; data.h=&hval; data.dh=dhdx;
   data.L=&L_; data.U=&U_; data.alpha=&alpha_; data.beta=&beta_;
   data.p0=&p0_; data.q0=&q0_; data.parallel=true; data.comm=comm_;

   for(;attempts<max_inner;++attempts)
   {
      BuildObjective(n_local_,xk,df0dx,xmin,xmax,L_,U_,p0_,q0_,rho);
      x=xk;
      SolveSubproblem(data,lambda_,x);
      mfem::Vector htrial(m_);
      mfem::real_t ftrial=0.0;
      evaluate(x,htrial,ftrial);
      MFEM_VERIFY(htrial.Size()==m_,"parallel GCMMA evaluator returned wrong equality size");

      const double model=ObjectiveModelValue(n_local_,xk,x,L_,U_,p0_,q0_,
                                             double(f0val),true,comm_);
      const double affine=AffineResidualNorm(n_local_,m_,xk,x,hval,dhdx,
                                             true,comm_);
      const double theta_trial=Norm1(htrial);
      double sigma=1.0;
      for(double value:lambda_) sigma=std::max(sigma,1.1*std::abs(value));
      const double predicted=double(f0val)+sigma*theta0-
                             (model+sigma*affine);
      const double actual=double(f0val)+sigma*theta0-
                          (double(ftrial)+sigma*theta_trial);
      const double tolerance=1e-10*(1.0+std::abs(double(ftrial)));
      const bool conservative=double(ftrial)<=model+tolerance;
      const bool merit=predicted>1e-14 && actual>=0.1*predicted;
      const bool feasibility=theta_trial<=0.9*theta0;
      const bool stationary=std::abs(predicted)<=1e-14 &&
                            theta_trial<=std::max(theta0,1e-10) &&
                            double(ftrial)<=double(f0val)+tolerance;
      if(conservative && (merit || feasibility || stationary))
      {
         accepted=true;
         ++attempts;
         break;
      }

      const double gap=std::max(0.0,double(ftrial)-model);
      const double distance=ScaledStepNorm2(n_local_,xk,x,xmin,xmax,true,comm_);
      rho=std::max(2.0*rho,1.1*(rho+gap/std::max(distance,1e-12)));
      ContractMoveLimits(n_local_,xk,alpha_,beta_);
   }

   if(!accepted)
   {
      x=xk;
      RestorationData restoration;
      restoration.n=n_local_; restoration.m=m_; restoration.xk=&xk;
      restoration.h=&hval; restoration.dh=dhdx;
      restoration.xmin=&xmin; restoration.xmax=&xmax;
      restoration.parallel=true; restoration.comm=comm_;
      RestoreAffine(restoration,x,80);
      mfem::Vector htrial(m_);
      mfem::real_t ftrial=0.0;
      evaluate(x,htrial,ftrial);
      MFEM_VERIFY(htrial.Size()==m_,
                  "parallel GCMMA restoration evaluator returned wrong equality size");
      if(Norm1(htrial)<theta0*(1.0-1e-4))
      {
         accepted=true;
         std::fill(lambda_.begin(),lambda_.end(),0.0);
      }
      else x=xk;
   }

   if(!accepted) lambda_=lambda_k;

   if(accepted)
   {
      xo2_=xo1_; xo1_=xk; ++iter_;
   }
   last_step_accepted_=accepted;
   if(inner_iterations) *inner_iterations=attempts;
}

// Distributed counterpart of MMAEqualityOptimizer::RestoreFeasibility();
// gradient dot products inside RestoreAffine() are reduced across comm_.
mfem::real_t MMAEqualityOptimizerParallel::RestoreFeasibility(
   mfem::Vector &x,const mfem::Vector &hval,const mfem::Vector *dhdx,
   const mfem::Vector &xmin,const mfem::Vector &xmax,int max_iterations)
{
   CheckRestorationInput(n_local_,m_,x,hval,dhdx,xmin,xmax);
   mfem::Vector xk(x);
   RestorationData data;
   data.n=n_local_; data.m=m_; data.xk=&xk; data.h=&hval; data.dh=dhdx;
   data.xmin=&xmin; data.xmax=&xmax; data.parallel=true; data.comm=comm_;
   return mfem::real_t(RestoreAffine(data,x,max_iterations));
}

// Diagnostic only, distributed counterpart of
// MMAEqualityOptimizer::KKTresidual(); see that method's comment.
mfem::real_t MMAEqualityOptimizerParallel::KKTresidual(
   const mfem::Vector &x,const mfem::Vector &df0dx,mfem::real_t,
   const mfem::Vector &hval,const mfem::Vector *dhdx,
   const mfem::Vector &xmin,const mfem::Vector &xmax,double *lambda_out) const
{
   CheckInput(n_local_,m_,x,df0dx,hval,dhdx,xmin,xmax);
   if(lambda_out) std::copy(lambda_.begin(),lambda_.end(),lambda_out);
   return mfem::real_t(ProjectedKKT(n_local_,m_,x,df0dx,hval,dhdx,xmin,xmax,
                                    lambda_,true,comm_));
}
#endif

} // namespace mfem_mma
