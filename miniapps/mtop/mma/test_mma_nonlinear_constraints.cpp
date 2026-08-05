/**
 * Separate nonlinear inequality and nonlinear equality tests for MMA/GCMMA.
 *
 * Inequality case: g(x)=mean(x^4)-mean(a^4)<=0, with an objective whose
 * unconstrained target is displaced along grad(g) at x=a.
 *
 * Equality case: h(x)=mean(x^2)-mean(a^2)=0, with an objective centered at
 * 1.1*a.  In both cases x=a is the analytic solution with a positive,
 * nonzero constraint multiplier.  Keeping the cases separate avoids the
 * rank-deficient coupling of an internal +/- equality pair to an inequality.
 */

#include "MMA_MFEM.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#ifdef MFEM_USE_MPI

using namespace mfem;
using namespace mfem_mma;

namespace {

int rank_id=0,rank_count=1,failures=0;

enum class Method { MMA, GCMMA, GCMMA_CALLBACK };

const char *MethodName(Method method)
{
   if(method==Method::MMA) return "MMA";
   if(method==Method::GCMMA) return "GCMMA";
   return "GCMMA callback";
}

double GlobalSum(double local)
{
   double global=0.0;
   MPI_Allreduce(&local,&global,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
   return global;
}

double GlobalMax(double local)
{
   double global=0.0;
   MPI_Allreduce(&local,&global,1,MPI_DOUBLE,MPI_MAX,MPI_COMM_WORLD);
   return global;
}

std::pair<int,int> Distribute(int n)
{
   const int base=n/rank_count,extra=n%rank_count;
   return {base+(rank_id<extra ? 1 : 0),
           rank_id*base+std::min(rank_id,extra)};
}

void Check(bool condition,const std::string &message)
{
   int local=condition ? 0 : 1,global=0;
   MPI_Allreduce(&local,&global,1,MPI_INT,MPI_MAX,MPI_COMM_WORLD);
   if(rank_id==0)
   {
      if(global==0) std::printf("  [PASS] %s\n",message.c_str());
      else { std::printf("  [FAIL] %s\n",message.c_str()); ++failures; }
   }
}

struct Values
{
   double objective=0.0,inequality=0.0,equality=0.0;
};

Values Evaluate(const Vector &x,const Vector &target,int n,
                double target_second,double target_fourth,
                Vector *df0=nullptr,Vector *dg=nullptr,Vector *dh=nullptr)
{
   double local_f=0.0,local_second=0.0,local_fourth=0.0;
   if(df0) df0->SetSize(x.Size());
   if(dg) dg->SetSize(x.Size());
   if(dh) dh->SetSize(x.Size());
   for(int j=0;j<x.Size();++j)
   {
      const double value=double(x(j));
      const double difference=value-double(target(j));
      local_f+=0.5*difference*difference/n;
      local_second+=value*value/n;
      local_fourth+=value*value*value*value/n;
      if(df0) (*df0)(j)=real_t(difference/n);
      if(dg) (*dg)(j)=real_t(4.0*value*value*value/n);
      if(dh) (*dh)(j)=real_t(2.0*value/n);
   }
   Values values;
   values.objective=GlobalSum(local_f);
   values.inequality=GlobalSum(local_fourth)-target_fourth;
   values.equality=GlobalSum(local_second)-target_second;
   return values;
}

bool RestoreQuadraticEquality(Vector &x,int n,double target_second)
{
   double local_second=0.0;
   int local_invalid=0;
   for(int j=0;j<x.Size();++j)
   {
      const double value=double(x(j));
      if(!std::isfinite(value)) local_invalid=1;
      else local_second+=value*value/n;
   }
   int global_invalid=0;
   MPI_Allreduce(&local_invalid,&global_invalid,1,MPI_INT,MPI_MAX,
                 MPI_COMM_WORLD);
   const double current_second=GlobalSum(local_second);
   if(global_invalid || !(current_second>0.0) ||
      !std::isfinite(current_second)) return false;
   x*=real_t(std::sqrt(target_second/current_second));
   return true;
}

struct Result
{
   int iterations=0,total_inner=0,multi_inner=0;
   double kkt=1.0,inequality=1.0,equality=1.0,max_error=1.0;
};

Result Run(Method method,bool equality_case)
{
   const int n=240;
   const auto distribution=Distribute(n);
   const int nl=distribution.first,offset=distribution.second;
   Vector x(nl),target(nl),objective_target(nl),xmin(nl),xmax(nl),
          df0(nl),dg(nl),dh(nl);
   xmin=0.1; xmax=0.9;
   double local_second=0.0,local_fourth=0.0;
   for(int j=0;j<nl;++j)
   {
      const int global=offset+j;
      target(j)=real_t(global%2==0 ? 0.35 : 0.55);
      const double value=double(target(j));
      // Choose the unconstrained objective target so x=target satisfies
      // stationarity with a positive analytic multiplier for the constraint
      // under test: radial for mean(x^2), cubic-normal for mean(x^4).
      objective_target(j)=real_t(equality_case ? 1.1*value :
                                 value+0.4*value*value*value);
      local_second+=value*value/n;
      local_fourth+=value*value*value*value/n;
   }
   const double target_second=GlobalSum(local_second);
   const double inequality_limit=GlobalSum(local_fourth);
   if(equality_case)
   {
      constexpr double pi=3.14159265358979323846;
      double local_initial_second=0.0;
      for(int j=0;j<nl;++j)
      {
         const double raw=1.0+0.10*std::sin(2.0*pi*(offset+j)/n);
         x(j)=real_t(raw);
         local_initial_second+=raw*raw/n;
      }
      const double initial_scale=std::sqrt(
         target_second/GlobalSum(local_initial_second));
      x*=real_t(initial_scale);
   }
   else x=0.3;

   MMAOptimizerParallel optimizer=equality_case ?
      MMAOptimizerParallel::WithEqualities(MPI_COMM_WORLD,nl,0,1) :
      MMAOptimizerParallel(MPI_COMM_WORLD,nl,1);
   optimizer.SetAsymptotes(0.15,0.7,1.2);
   Result result;
   const bool require_global_solution=
      !equality_case || method==Method::GCMMA_CALLBACK;
   for(int outer=0;outer<500 &&
       ((require_global_solution && result.max_error>=2e-2) ||
        (!require_global_solution && optimizer.NumIterations()<5) ||
        result.kkt>=1e-5 ||
        (!equality_case && result.inequality>=1e-7) ||
        (equality_case && std::abs(result.equality)>=1e-6));++outer)
   {
      Values values=Evaluate(x,objective_target,n,target_second,inequality_limit,
                             &df0,&dg,&dh);
      Vector fival(equality_case ? 2 : 1);
      std::vector<Vector> gradients(equality_case ? 2 : 1);
      if(equality_case)
      {
         fival(0)=real_t(values.equality); fival(1)=-fival(0);
         gradients[0]=dh; gradients[1]=dh; gradients[1]*=-1.0;
      }
      else { fival(0)=real_t(values.inequality); gradients[0]=dg; }
      int inner=0;
      if(method==Method::MMA)
         optimizer.Update(x,df0,real_t(values.objective),fival,
                          gradients.data(),xmin,xmax);
      else if(method==Method::GCMMA)
         optimizer.UpdateGCMMA(x,df0,real_t(values.objective),fival,
                               gradients.data(),xmin,xmax,&inner);
      else
         optimizer.UpdateGCMMA(x,df0,real_t(values.objective),fival,
                               gradients.data(),xmin,xmax,
            [&](const Vector &candidate,Vector &candidate_fi,real_t &candidate_f0)
            {
               const Values trial=Evaluate(candidate,objective_target,n,
                                           target_second,inequality_limit);
               candidate_fi.SetSize(equality_case ? 2 : 1);
               if(equality_case)
               {
                  candidate_fi(0)=real_t(trial.equality);
                  candidate_fi(1)=-candidate_fi(0);
               }
               else candidate_fi(0)=real_t(trial.inequality);
               candidate_f0=real_t(trial.objective);
            },20,&inner);
      result.total_inner+=inner;
      if(inner>1) ++result.multi_inner;

      // The reciprocal MMA models enforce only the affine equality model.
      // Restore the true homogeneous quadratic equality before evaluating
      // convergence and before constructing the next outer approximation.
      if(equality_case && !RestoreQuadraticEquality(x,n,target_second))
      {
         result.kkt=std::numeric_limits<double>::quiet_NaN();
         result.inequality=result.equality=result.kkt;
         result.max_error=std::numeric_limits<double>::infinity();
         break;
      }

      values=Evaluate(x,objective_target,n,target_second,inequality_limit,
                      &df0,&dg,&dh);
      if(equality_case)
      {
         fival(0)=real_t(values.equality); fival(1)=-fival(0);
         gradients[0]=dh; gradients[1]=dh; gradients[1]*=-1.0;
      }
      else { fival(0)=real_t(values.inequality); gradients[0]=dg; }
      result.kkt=double(optimizer.KKTresidual(x,df0,
                       real_t(values.objective),fival,
                       gradients.data(),xmin,xmax));
      result.inequality=values.inequality;
      result.equality=values.equality;
      double local_error=0.0;
      for(int j=0;j<nl;++j)
      {
         const double value=double(x(j));
         if(!std::isfinite(value))
            local_error=std::numeric_limits<double>::infinity();
         else local_error=std::max(local_error,
                                   std::abs(value-double(target(j))));
      }
      result.max_error=GlobalMax(local_error);
      if(!std::isfinite(result.kkt) ||
         !std::isfinite(result.inequality) ||
         !std::isfinite(result.equality) ||
         !std::isfinite(result.max_error))
         break;
   }
   result.iterations=optimizer.NumIterations();
   return result;
}

void Test(Method method,bool equality_case)
{
   if(rank_id==0)
      std::printf("\n--- Nonlinear %s [%s] ---\n",
                  equality_case ? "equality" : "inequality",MethodName(method));
   const Result result=Run(method,equality_case);
   if(rank_id==0)
      std::printf("  iterations=%d  total_inner=%d  multi_inner=%d"
                  "  kkt=%.3e  g=%.3e  |h|=%.3e  max_error=%.3e\n",
                  result.iterations,result.total_inner,result.multi_inner,
                  result.kkt,result.inequality,std::abs(result.equality),
                  result.max_error);
   const std::string tag=std::string("[")+
      (equality_case ? "equality, " : "inequality, ")+MethodName(method)+"] ";
   Check(std::isfinite(result.kkt),tag+"nonlinear solve remains finite");
   if(equality_case)
      Check(std::abs(result.equality)<1e-6,
            tag+"nonlinear equality is satisfied");
   else
      Check(result.inequality<1e-7,tag+"nonlinear inequality is satisfied");
   if(!equality_case || method==Method::GCMMA_CALLBACK)
      Check(result.max_error<2e-2,tag+"solution matches analytic optimum");
   else
      Check(std::isfinite(result.max_error),
            tag+"stationary design remains bounded");
   Check(result.kkt<1e-5,tag+"KKT residual < 1e-5");
   Check(result.iterations>0,tag+"performs at least one outer iteration");
   if(method==Method::GCMMA_CALLBACK)
   {
      Check(result.total_inner>=result.iterations,
            tag+"callback evaluates every accepted outer candidate");
      if(equality_case)
         Check(result.multi_inner>0,
               tag+"non-conservatism triggers callback retries");
   }
}

} // namespace

int main(int argc,char **argv)
{
   MPI_Init(&argc,&argv);
   MPI_Comm_rank(MPI_COMM_WORLD,&rank_id);
   MPI_Comm_size(MPI_COMM_WORLD,&rank_count);
   if(rank_id==0)
      std::printf("=== Nonlinear constraint MMA tests (%d ranks) ===\n",rank_count);
   for(bool equality_case:{false,true})
   {
      Test(Method::MMA,equality_case);
      Test(Method::GCMMA,equality_case);
      Test(Method::GCMMA_CALLBACK,equality_case);
   }
   if(rank_id==0)
      std::printf("\n%s\n",failures==0 ?
                  "All nonlinear constraint MMA tests PASSED." :
                  "Nonlinear constraint MMA test failures detected.");
   int global_failures=0;
   MPI_Bcast(&failures,1,MPI_INT,0,MPI_COMM_WORLD);
   global_failures=failures;
   MPI_Finalize();
   return global_failures==0 ? 0 : 1;
}

#else

int main()
{
   std::printf("MFEM was built without MPI; nonlinear MMA test skipped.\n");
   return 0;
}

#endif
