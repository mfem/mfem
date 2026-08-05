/**
 * Nonconvex density-filtered SIMP tests for equality-only MMA.
 *
 * This mirrors test_mma_nonconvex.cpp, but represents fixed volume as the
 * single true equality
 *
 *                 h(x) = mean(x) - Vfrac = 0,
 *
 * rather than as two inequalities in the generalized MMA formulation.
 * Both the ordinary equality-only MMA update and its callback-based,
 * objective-globalized GCMMA update are exercised in serial and MPI runs.
 */

#include "MMA_Equality_MFEM.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

#ifdef MFEM_USE_MPI

using namespace mfem;
using namespace mfem_mma;
using Clock=std::chrono::steady_clock;

namespace {

int rank_id=0,rank_count=1,failures=0;

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

struct Filter
{
   std::vector<std::vector<int>> indices;
   std::vector<std::vector<double>> weights;

   Filter(int n_global,int n_local,int offset,int radius)
   {
      indices.resize(n_local);
      weights.resize(n_local);
      for(int j=0;j<n_local;++j)
      {
         const int global=offset+j;
         const int first=std::max(0,global-3*radius);
         const int last=std::min(n_global-1,global+3*radius);
         double sum=0.0;
         for(int k=first;k<=last;++k)
         {
            const double distance=double(global-k);
            indices[j].push_back(k);
            weights[j].push_back(std::exp(-distance*distance/
                                          (2.0*radius*radius)));
            sum+=weights[j].back();
         }
         for(double &weight:weights[j]) weight/=sum;
      }
   }
};

struct Problem
{
   int n_global,n_local,offset;
   double volume,xmin_value;
   Filter filter;
   std::vector<double> load;

   Problem(int n,double v,int radius)
      : n_global(n),n_local(Distribute(n).first),
        offset(Distribute(n).second),volume(v),xmin_value(0.01),
        filter(n,n_local,offset,radius)
   {
      load.resize(n_local);
      constexpr double pi=3.14159265358979323846;
      for(int j=0;j<n_local;++j)
      {
         const int global=offset+j;
         const double checker=(global/radius)%2==0 ? 1.0 : -1.0;
         const double envelope=1.0+0.30*std::sin(6.0*pi*global/n_global)+
                                     0.15*std::sin(14.0*pi*global/n_global);
         load[j]=std::max(0.05,envelope*(1.0+0.40*checker));
      }
   }

   std::vector<double> Gather(const Vector &x) const
   {
      std::vector<double> full(n_global,0.0);
      for(int j=0;j<n_local;++j) full[offset+j]=double(x(j));
      MPI_Allreduce(MPI_IN_PLACE,full.data(),n_global,MPI_DOUBLE,MPI_SUM,
                    MPI_COMM_WORLD);
      return full;
   }

   double Mean(const Vector &x) const
   {
      double local=0.0;
      for(int j=0;j<n_local;++j) local+=double(x(j));
      return GlobalSum(local)/n_global;
   }

   double Evaluate(const Vector &x,double penalty,Vector *gradient=nullptr) const
   {
      const std::vector<double> full=Gather(x);
      std::vector<double> sensitivity(n_global,0.0);
      double local_objective=0.0;
      for(int j=0;j<n_local;++j)
      {
         double filtered=0.0;
         for(size_t q=0;q<filter.indices[j].size();++q)
            filtered+=filter.weights[j][q]*full[filter.indices[j][q]];
         filtered=std::max(filtered,xmin_value);
         const double power=std::pow(filtered,penalty);
         local_objective+=load[j]/power;
         if(gradient)
         {
            const double derivative=-penalty*load[j]/(power*filtered)/n_global;
            for(size_t q=0;q<filter.indices[j].size();++q)
               sensitivity[filter.indices[j][q]]+=
                  derivative*filter.weights[j][q];
         }
      }
      if(gradient)
      {
         MPI_Allreduce(MPI_IN_PLACE,sensitivity.data(),n_global,MPI_DOUBLE,
                       MPI_SUM,MPI_COMM_WORLD);
         gradient->SetSize(n_local);
         for(int j=0;j<n_local;++j)
            (*gradient)(j)=real_t(sensitivity[offset+j]);
      }
      return GlobalSum(local_objective)/n_global;
   }
};

struct Result
{
   int iterations=0,accepted=0,rejected=0,total_inner=0;
   double kkt=0.0,residual=0.0,initial_objective=0.0;
   double final_objective=0.0,max_change=0.0;
};

Result RunCase(int n,double volume,double final_penalty,bool continuation,
               int radius,bool gcmma,int max_iterations)
{
   Problem problem(n,volume,radius);
   Vector x(problem.n_local),initial(problem.n_local),xmin(problem.n_local),
          xmax(problem.n_local),gradient(problem.n_local),h(1),dh(problem.n_local);
   x=real_t(volume); initial=x; xmin=real_t(problem.xmin_value); xmax=1.0;
   dh=real_t(1.0/n);
   MMAEqualityOptimizerParallel optimizer(MPI_COMM_WORLD,problem.n_local,1);
   optimizer.SetAsymptotes(0.10,0.7,1.15);

   Result result;
   result.initial_objective=problem.Evaluate(x,final_penalty);
   const auto start=Clock::now();
   for(int outer=0;outer<max_iterations;++outer)
   {
      const double penalty=continuation ?
         1.0+(final_penalty-1.0)*std::min(outer,200)/200.0 : final_penalty;
      const double objective=problem.Evaluate(x,penalty,&gradient);
      h(0)=real_t(problem.Mean(x)-volume);
      if(gcmma)
      {
         int inner=0;
         optimizer.UpdateGCMMA(x,gradient,real_t(objective),h,&dh,xmin,xmax,
            [&](const Vector &candidate,Vector &true_h,real_t &true_objective)
            {
               true_objective=real_t(problem.Evaluate(candidate,penalty));
               true_h.SetSize(1);
               true_h(0)=real_t(problem.Mean(candidate)-volume);
            },20,&inner);
         result.total_inner+=inner;
         if(optimizer.LastStepAccepted()) ++result.accepted;
         else ++result.rejected;
      }
      else
      {
         optimizer.Update(x,gradient,real_t(objective),h,&dh,xmin,xmax);
         ++result.accepted;
      }

      const double updated_objective=problem.Evaluate(x,penalty,&gradient);
      h(0)=real_t(problem.Mean(x)-volume);
      result.kkt=double(optimizer.KKTresidual(x,gradient,
                    real_t(updated_objective),h,&dh,xmin,xmax));
      if(rank_id==0 && (outer%50==0 || outer==max_iterations-1))
         std::printf("  iter %4d: f0=%.4e  h=%+.3e  kkt=%.3e  p=%.2f\n",
                     outer,updated_objective,double(h(0)),result.kkt,penalty);
   }
   result.iterations=optimizer.NumIterations();
   result.final_objective=problem.Evaluate(x,final_penalty,&gradient);
   h(0)=real_t(problem.Mean(x)-volume);
   result.residual=std::abs(double(h(0)));
   double local_change=0.0;
   for(int j=0;j<problem.n_local;++j)
      local_change=std::max(local_change,
                            std::abs(double(x(j))-double(initial(j))));
   result.max_change=GlobalMax(local_change);
   const double milliseconds=
      std::chrono::duration<double,std::milli>(Clock::now()-start).count();
   if(rank_id==0)
      std::printf("  Final: iterations=%d accepted=%d rejected=%d inner=%d"
                  " kkt=%.3e residual=%.3e max_change=%.3e"
                  " obj=%.4e->%.4e time=%.0fms\n",
                  result.iterations,result.accepted,result.rejected,
                  result.total_inner,result.kkt,result.residual,result.max_change,
                  result.initial_objective,result.final_objective,milliseconds);
   return result;
}

void TestCase(int n,double volume,double penalty,bool continuation,int radius,
              bool gcmma,int max_iterations,const char *label)
{
   if(rank_id==0)
      std::printf("\n--- %s n=%d Vfrac=%.2f r=%d p=%s [%s] ---\n",
                  label,n,volume,radius,continuation ? "1->5" :
                  std::to_string(int(penalty)).c_str(),
                  gcmma ? "equality GCMMA" : "equality MMA");
   const Result result=RunCase(n,volume,penalty,continuation,radius,gcmma,
                               max_iterations);
   const std::string tag=std::string("[")+label+","+
                         (gcmma ? "GCMMA" : "MMA")+"] ";
   Check(std::isfinite(result.kkt) && std::isfinite(result.final_objective),
         tag+"nonconvex solve remains finite");
   Check(result.residual<1e-8,tag+"volume equality is satisfied");
   Check(result.max_change>1e-3,tag+"design is nontrivially redistributed");
   Check(result.final_objective>0.0 &&
         result.final_objective<2.0*result.initial_objective,
         tag+"final-penalty objective remains bounded");
   Check(result.iterations>0,tag+"performs accepted MMA iterations");
}

} // namespace

int main(int argc,char **argv)
{
   MPI_Init(&argc,&argv);
   MPI_Comm_rank(MPI_COMM_WORLD,&rank_id);
   MPI_Comm_size(MPI_COMM_WORLD,&rank_count);
   if(rank_id==0)
      std::printf("=== Equality-only density-filtered SIMP tests (%d ranks) ===\n",
                  rank_count);

   TestCase(1000,0.4,3.0,false,10,false,300,"p3r10");
   TestCase(1000,0.4,3.0,false,10,true, 300,"p3r10");
   TestCase(1000,0.4,5.0,false, 5,false,300,"p5r5");
   TestCase(1000,0.4,5.0,false, 5,true, 300,"p5r5");
   TestCase(1000,0.4,5.0,true, 10,false,400,"p5cont");
   TestCase(1000,0.4,5.0,true, 10,true, 400,"p5cont");

   if(rank_id==0)
      std::printf("\n%s\n",failures==0 ?
                  "All equality-only nonconvex tests PASSED." :
                  "Equality-only nonconvex test failures detected.");
   int global_failures=0;
   MPI_Bcast(&failures,1,MPI_INT,0,MPI_COMM_WORLD);
   global_failures=failures;
   MPI_Finalize();
   return global_failures==0 ? 0 : 1;
}

#else

int main()
{
   std::printf("MFEM was built without MPI; equality nonconvex test skipped.\n");
   return 0;
}

#endif
