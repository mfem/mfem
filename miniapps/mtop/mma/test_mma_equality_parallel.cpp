/** MPI examples for MMAEqualityOptimizerParallel. */

#include "MMA_Equality_MFEM.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <utility>
#include <vector>

#ifdef MFEM_USE_MPI

using namespace mfem;
using namespace mfem_mma;

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

void Check(bool condition,const char *message)
{
   int local=condition ? 0 : 1,global=0;
   MPI_Allreduce(&local,&global,1,MPI_INT,MPI_MAX,MPI_COMM_WORLD);
   if(rank_id==0)
   {
      if(global==0) std::printf("  [PASS] %s\n",message);
      else { std::printf("  [FAIL] %s\n",message); ++failures; }
   }
}

std::pair<int,int> Distribute(int n)
{
   const int base=n/rank_count,extra=n%rank_count;
   const int local=base+(rank_id<extra ? 1 : 0);
   const int offset=rank_id*base+std::min(rank_id,extra);
   return {local,offset};
}

double GlobalMean(const Vector &x,int n_global)
{
   double local=0.0;
   for(int j=0;j<x.Size();++j) local+=double(x(j));
   return GlobalSum(local)/n_global;
}

void TestAffineRestoration()
{
   if(rank_id==0) std::printf("\n--- Parallel affine restoration ---\n");
   const int n=103;
   const double volume=0.4;
   const auto distribution=Distribute(n);
   const int nl=distribution.first;
   Vector x(nl),xmin(nl),xmax(nl),h(1),dh(nl);
   x=0.8; xmin=0.0; xmax=1.0; dh=real_t(1.0/n);
   h(0)=real_t(GlobalMean(x,n)-volume);

   MMAEqualityOptimizerParallel opt(MPI_COMM_WORLD,nl,1);
   const real_t residual=opt.RestoreFeasibility(x,h,&dh,xmin,xmax);
   const double mean=GlobalMean(x,n);
   if(rank_id==0)
      std::printf("  residual=%.3e  mean=%.8f  iterations=%d\n",
                  double(residual),mean,opt.NumIterations());
   Check(double(residual)<1e-10,"distributed affine residual < 1e-10");
   Check(std::abs(mean-volume)<1e-10,"distributed restored design is feasible");
   Check(opt.NumIterations()==0,"parallel restoration does not advance MMA history");
}

void TestAffineOptimization()
{
   if(rank_id==0) std::printf("\n--- Parallel equality-constrained optimization ---\n");
   const int n=1000;
   const double volume=0.5;
   const auto distribution=Distribute(n);
   const int nl=distribution.first,offset=distribution.second;
   Vector x(nl),target(nl),exact(nl),xmin(nl),xmax(nl),df0(nl),h(1),dh(nl);
   x=volume; xmin=0.0; xmax=1.0; dh=real_t(1.0/n);
   double local_target_sum=0.0;
   for(int j=0;j<nl;++j)
   {
      target(j)=real_t(0.2+0.4*double(offset+j)/double(n-1));
      local_target_sum+=double(target(j));
   }
   const double target_mean=GlobalSum(local_target_sum)/n;
   for(int j=0;j<nl;++j)
      exact(j)=real_t(double(target(j))+volume-target_mean);

   MMAEqualityOptimizerParallel opt(MPI_COMM_WORLD,nl,1);
   opt.SetAsymptotes(0.15,0.7,1.2);
   real_t kkt=1.0;
   double analytic_error=1.0;
   int total_inner=0,outer=0,rejected=0;
   for(;outer<1000 && (opt.NumIterations()<10 || double(kkt)>1e-12 ||
                       analytic_error>1e-3);++outer)
   {
      double local_f0=0.0;
      for(int j=0;j<nl;++j)
      {
         const double difference=double(x(j))-double(target(j));
         local_f0+=0.5*difference*difference/n;
         df0(j)=real_t(difference/n);
      }
      h(0)=real_t(GlobalMean(x,n)-volume);
      int inner=0;
      opt.UpdateGCMMA(x,df0,real_t(GlobalSum(local_f0)),h,&dh,xmin,xmax,
         [&](const Vector &candidate,Vector &true_h,real_t &true_f0)
         {
            double local_value=0.0;
            for(int j=0;j<nl;++j)
            {
               const double difference=double(candidate(j))-double(target(j));
               local_value+=0.5*difference*difference/n;
            }
            true_f0=real_t(GlobalSum(local_value));
            true_h.SetSize(1);
            true_h(0)=real_t(GlobalMean(candidate,n)-volume);
         },50,&inner);
      total_inner+=inner;
      if(!opt.LastStepAccepted())
      {
         ++rejected;
         kkt=1.0;
         continue;
      }
      for(int j=0;j<nl;++j)
         df0(j)=real_t((double(x(j))-double(target(j)))/n);
      h(0)=real_t(GlobalMean(x,n)-volume);
      kkt=opt.KKTresidual(x,df0,0.0,h,&dh,xmin,xmax);
      double local_iteration_error=0.0;
      for(int j=0;j<nl;++j)
         local_iteration_error=std::max(local_iteration_error,
                                  std::abs(double(x(j))-double(exact(j))));
      analytic_error=GlobalMax(local_iteration_error);
   }

   double local_error=0.0;
   for(int j=0;j<nl;++j)
      local_error=std::max(local_error,std::abs(double(x(j))-double(exact(j))));
   const double max_error=GlobalMax(local_error);
   const double mean=GlobalMean(x,n);
   if(rank_id==0)
      std::printf("  kkt=%.3e  mean=%.8f  max_error=%.3e"
                  "  iterations=%d  outer_attempts=%d  rejected=%d"
                  "  total_inner=%d\n",
                  double(kkt),mean,max_error,opt.NumIterations(),outer,
                  rejected,total_inner);
   Check(double(kkt)<1e-10,"parallel KKT residual < 1e-10");
   Check(opt.NumIterations()>=10,
         "parallel GCMMA performs at least ten outer iterations");
   Check(outer<1000,"parallel GCMMA finishes within enlarged outer budget");
   Check(std::abs(mean-volume)<1e-8,"parallel affine equality remains satisfied");
   Check(max_error<2e-3,"parallel solution matches analytic optimum");
}

void TestNonlinearRestoration()
{
   if(rank_id==0) std::printf("\n--- Parallel nonlinear restoration ---\n");
   const int n=257;
   const double target_square=0.16;
   const int nl=Distribute(n).first;
   Vector x(nl),xmin(nl),xmax(nl),h(1),dh(nl);
   x=0.7; xmin=0.05; xmax=1.0;
   MMAEqualityOptimizerParallel opt(MPI_COMM_WORLD,nl,1);

   double true_residual=1.0;
   int restoration_steps=0;
   for(;restoration_steps<12 && true_residual>1e-10;++restoration_steps)
   {
      double local_square=0.0;
      for(int j=0;j<nl;++j)
      {
         local_square+=double(x(j))*double(x(j));
         dh(j)=real_t(2.0*double(x(j))/n);
      }
      h(0)=real_t(GlobalSum(local_square)/n-target_square);
      true_residual=std::abs(double(h(0)));
      if(true_residual<=1e-10) break;
      opt.RestoreFeasibility(x,h,&dh,xmin,xmax);
   }
   double local_square=0.0;
   for(int j=0;j<nl;++j) local_square+=double(x(j))*double(x(j));
   true_residual=std::abs(GlobalSum(local_square)/n-target_square);
   const double mean=GlobalMean(x,n);
   if(rank_id==0)
      std::printf("  true_residual=%.3e  mean=%.8f  restoration_steps=%d\n",
                  true_residual,mean,restoration_steps);
   Check(true_residual<1e-10,"parallel nonlinear equality restored");
   Check(std::abs(mean-0.4)<1e-8,"parallel restoration reaches x=0.4");
   Check(opt.NumIterations()==0,"parallel nonlinear restoration preserves history");
}

void TestGlobalizationRetries()
{
   if(rank_id==0) std::printf("\n--- Parallel GCMMA curvature retries ---\n");
   const int n=120;
   const auto distribution=Distribute(n);
   const int nl=distribution.first,offset=distribution.second;
   Vector x(nl),xk(nl),xmin(nl),xmax(nl),df0(nl),h(1),dh(nl);
   x=0.5; xk=x; xmin=0.0; xmax=1.0; dh=real_t(1.0/n);
   for(int j=0;j<nl;++j)
      df0(j)=real_t(((offset+j)%2==0 ? 1.0 : -1.0)/n);
   h(0)=0.0;
   MMAEqualityOptimizerParallel opt(MPI_COMM_WORLD,nl,1);
   int inner=0;
   opt.UpdateGCMMA(x,df0,0.0,h,&dh,xmin,xmax,
      [&](const Vector &candidate,Vector &true_h,real_t &true_f0)
      {
         double local_linear=0.0,local_square=0.0;
         for(int j=0;j<nl;++j)
         {
            const double step=double(candidate(j))-double(xk(j));
            local_linear+=double(df0(j))*step;
            local_square+=step*step/n;
         }
         true_f0=real_t(GlobalSum(local_linear+100.0*local_square));
         true_h.SetSize(1);
         true_h(0)=real_t(GlobalMean(candidate,n)-0.5);
      },15,&inner);
   const double mean=GlobalMean(x,n);
   if(rank_id==0)
      std::printf("  accepted=%d  inner=%d  mean=%.8f  iterations=%d\n",
                  opt.LastStepAccepted()?1:0,inner,mean,opt.NumIterations());
   Check(inner>1,"parallel non-conservative objective triggers retries");
   Check(opt.LastStepAccepted(),"parallel globalized candidate is accepted");
   Check(std::abs(mean-0.5)<1e-9,"parallel accepted candidate is feasible");
   Check(opt.NumIterations()==1,"parallel accepted step advances history once");
}

void TestGlobalizationRollback()
{
   if(rank_id==0) std::printf("\n--- Parallel GCMMA rejection and rollback ---\n");
   const int n=84,max_inner=2;
   const auto distribution=Distribute(n);
   const int nl=distribution.first,offset=distribution.second;
   Vector x(nl),x_before(nl),xmin(nl),xmax(nl),df0(nl),h(1),dh(nl);
   x=0.5; x_before=x; xmin=0.0; xmax=1.0; dh=real_t(1.0/n);
   for(int j=0;j<nl;++j)
      df0(j)=real_t(((offset+j)%2==0 ? 1.0 : -1.0)/n);
   h(0)=0.0;
   MMAEqualityOptimizerParallel opt(MPI_COMM_WORLD,nl,1);
   int inner=0;
   opt.UpdateGCMMA(x,df0,0.0,h,&dh,xmin,xmax,
      [&](const Vector &candidate,Vector &true_h,real_t &true_f0)
      {
         true_f0=real_t(1e20);
         true_h.SetSize(1);
         true_h(0)=real_t(GlobalMean(candidate,n)-0.5);
      },max_inner,&inner);

   double local_change=0.0;
   for(int j=0;j<nl;++j)
      local_change=std::max(local_change,
                            std::abs(double(x(j))-double(x_before(j))));
   const double max_change=GlobalMax(local_change);
   if(rank_id==0)
      std::printf("  accepted=%d  inner=%d  max_change=%.3e  iterations=%d\n",
                  opt.LastStepAccepted()?1:0,inner,max_change,opt.NumIterations());
   Check(inner==max_inner,"parallel globalization respects inner limit");
   Check(!opt.LastStepAccepted(),"parallel unacceptable candidate is rejected");
   Check(max_change==0.0,"parallel rejected step restores original design");
   Check(opt.NumIterations()==0,"parallel rejected step preserves history");
}

void TestRedundantEqualities()
{
   if(rank_id==0) std::printf("\n--- Parallel redundant equality constraints ---\n");
   const int n=600;
   const double volume=0.5;
   const auto distribution=Distribute(n);
   const int nl=distribution.first,offset=distribution.second;
   Vector x(nl),target(nl),exact(nl),xmin(nl),xmax(nl),df0(nl),h(2);
   Vector dh[2]; dh[0].SetSize(nl); dh[1].SetSize(nl);
   x=volume; xmin=0.0; xmax=1.0;
   dh[0]=real_t(1.0/n); dh[1]=real_t(2.0/n);
   double local_target_sum=0.0;
   for(int j=0;j<nl;++j)
   {
      target(j)=real_t(0.25+0.30*double(offset+j)/double(n-1));
      local_target_sum+=double(target(j));
   }
   const double target_mean=GlobalSum(local_target_sum)/n;
   for(int j=0;j<nl;++j)
      exact(j)=real_t(double(target(j))+volume-target_mean);

   MMAEqualityOptimizerParallel opt(MPI_COMM_WORLD,nl,2);
   opt.SetAsymptotes(0.15,0.7,1.2);
   real_t kkt=1.0;
   double max_error=1.0;
   for(int outer=0;outer<400 &&
       (opt.NumIterations()<10 || max_error>1e-3 || double(kkt)>1e-12);++outer)
   {
      double local_f0=0.0;
      for(int j=0;j<nl;++j)
      {
         const double difference=double(x(j))-double(target(j));
         local_f0+=0.5*difference*difference/n;
         df0(j)=real_t(difference/n);
      }
      const double residual=GlobalMean(x,n)-volume;
      h(0)=real_t(residual); h(1)=real_t(2.0*residual);
      opt.Update(x,df0,real_t(GlobalSum(local_f0)),h,dh,xmin,xmax);
      for(int j=0;j<nl;++j)
         df0(j)=real_t((double(x(j))-double(target(j)))/n);
      const double updated_residual=GlobalMean(x,n)-volume;
      h(0)=real_t(updated_residual); h(1)=real_t(2.0*updated_residual);
      kkt=opt.KKTresidual(x,df0,0.0,h,dh,xmin,xmax);
      double local_error=0.0;
      for(int j=0;j<nl;++j)
         local_error=std::max(local_error,
                              std::abs(double(x(j))-double(exact(j))));
      max_error=GlobalMax(local_error);
   }

   const auto &lambda=opt.GetLambda();
   const bool finite=std::isfinite(double(kkt)) && std::isfinite(max_error) &&
                     std::isfinite(lambda[0]) && std::isfinite(lambda[1]);
   if(rank_id==0)
      std::printf("  kkt=%.3e  h=[%.3e,%.3e]  max_error=%.3e"
                  "  lambda=[%.3e,%.3e]  iterations=%d\n",
                  double(kkt),double(h(0)),double(h(1)),max_error,
                  lambda[0],lambda[1],opt.NumIterations());
   Check(finite,"redundant parallel solve remains finite");
   Check(std::abs(double(h(0)))<1e-9 && std::abs(double(h(1)))<2e-9,
         "parallel redundant equalities are satisfied");
   Check(max_error<2e-3,"redundant parallel solution matches analytic optimum");
   Check(double(kkt)<1e-10,"redundant parallel KKT residual < 1e-10");
}

void TestManyEqualities()
{
   if(rank_id==0)
      std::printf("\n--- Parallel 100 regional equality constraints ---\n");
   const int m=100,region_size=10,n=m*region_size;
   const auto distribution=Distribute(n);
   const int nl=distribution.first,offset=distribution.second;
   Vector x(nl),target(nl),xmin(nl),xmax(nl),df0(nl),h(m);
   std::vector<Vector> dh(m);
   xmin=0.0; xmax=1.0;
   for(int i=0;i<m;++i)
   {
      dh[i].SetSize(nl);
      dh[i]=0.0;
   }
   for(int j=0;j<nl;++j)
   {
      const int global=offset+j,region=global/region_size;
      const int within=global%region_size;
      const double volume=0.35+0.30*double(region%5)/4.0;
      dh[region](j)=real_t(1.0/region_size);
      x(j)=real_t(volume);
      target(j)=real_t(volume+0.08*(double(within)-4.5)/4.5);
   }

   MMAEqualityOptimizerParallel opt(MPI_COMM_WORLD,nl,m);
   opt.SetAsymptotes(0.15,0.7,1.2);
   real_t kkt=1.0;
   double max_error=1.0,max_residual=1.0;
   std::vector<double> local_sums(m),global_sums(m);
   for(int outer=0;outer<500 &&
       (opt.NumIterations()<10 || max_error>1e-3 || double(kkt)>1e-10);++outer)
   {
      double local_f0=0.0;
      for(int j=0;j<nl;++j)
      {
         const double difference=double(x(j))-double(target(j));
         local_f0+=0.5*difference*difference/n;
         df0(j)=real_t(difference/n);
      }
      std::fill(local_sums.begin(),local_sums.end(),0.0);
      for(int j=0;j<nl;++j)
         local_sums[(offset+j)/region_size]+=double(x(j))/region_size;
      MPI_Allreduce(local_sums.data(),global_sums.data(),m,MPI_DOUBLE,MPI_SUM,
                    MPI_COMM_WORLD);
      for(int i=0;i<m;++i)
         h(i)=real_t(global_sums[i]-(0.35+0.30*double(i%5)/4.0));
      opt.Update(x,df0,real_t(GlobalSum(local_f0)),h,dh.data(),xmin,xmax);

      double local_error=0.0;
      for(int j=0;j<nl;++j)
      {
         df0(j)=real_t((double(x(j))-double(target(j)))/n);
         local_error=std::max(local_error,
                              std::abs(double(x(j))-double(target(j))));
      }
      max_error=GlobalMax(local_error);
      std::fill(local_sums.begin(),local_sums.end(),0.0);
      for(int j=0;j<nl;++j)
         local_sums[(offset+j)/region_size]+=double(x(j))/region_size;
      MPI_Allreduce(local_sums.data(),global_sums.data(),m,MPI_DOUBLE,MPI_SUM,
                    MPI_COMM_WORLD);
      max_residual=0.0;
      for(int i=0;i<m;++i)
      {
         h(i)=real_t(global_sums[i]-(0.35+0.30*double(i%5)/4.0));
         max_residual=std::max(max_residual,std::abs(double(h(i))));
      }
      kkt=opt.KKTresidual(x,df0,0.0,h,dh.data(),xmin,xmax);
   }

   if(rank_id==0)
      std::printf("  kkt=%.3e  max_residual=%.3e  max_error=%.3e"
                  "  iterations=%d\n",double(kkt),max_residual,max_error,
                  opt.NumIterations());
   Check(std::isfinite(double(kkt)),"100-equality parallel solve remains finite");
   Check(max_residual<1e-9,"all 100 parallel equalities are satisfied");
   Check(max_error<2e-3,"100-equality parallel solution matches analytic optimum");
   Check(double(kkt)<1e-8,"100-equality parallel KKT residual < 1e-8");
}

void TestZeroControlRanks()
{
   if(rank_id==0)
      std::printf("\n--- Parallel optimization with zero-control ranks ---\n");
   // With two or more ranks, keep one fewer global controls than ranks so at
   // least one process owns an empty local design vector.
   const int n=std::max(1,rank_count-1);
   const double volume=0.4;
   const auto distribution=Distribute(n);
   const int nl=distribution.first,offset=distribution.second;
   Vector x(nl),target(nl),xmin(nl),xmax(nl),df0(nl),h(1),dh(nl);
   x=volume; xmin=0.0; xmax=1.0; dh=real_t(1.0/n);
   for(int j=0;j<nl;++j)
      target(j)=real_t(n==1 ? volume :
                       0.3+0.2*double(offset+j)/double(n-1));

   const int local_zero=(nl==0 ? 1 : 0);
   int zero_ranks=0;
   MPI_Allreduce(&local_zero,&zero_ranks,1,MPI_INT,MPI_SUM,MPI_COMM_WORLD);
   const int expected_zero_ranks=std::max(0,rank_count-n);

   MMAEqualityOptimizerParallel opt(MPI_COMM_WORLD,nl,1);
   opt.SetAsymptotes(0.15,0.7,1.2);
   real_t kkt=1.0;
   double max_error=1.0;
   for(int outer=0;outer<200 &&
       (opt.NumIterations()<10 || max_error>1e-4 || double(kkt)>1e-10);++outer)
   {
      double local_f0=0.0;
      for(int j=0;j<nl;++j)
      {
         const double difference=double(x(j))-double(target(j));
         local_f0+=0.5*difference*difference/n;
         df0(j)=real_t(difference/n);
      }
      h(0)=real_t(GlobalMean(x,n)-volume);
      opt.Update(x,df0,real_t(GlobalSum(local_f0)),h,&dh,xmin,xmax);

      double local_error=0.0;
      for(int j=0;j<nl;++j)
      {
         df0(j)=real_t((double(x(j))-double(target(j)))/n);
         local_error=std::max(local_error,
                              std::abs(double(x(j))-double(target(j))));
      }
      max_error=GlobalMax(local_error);
      h(0)=real_t(GlobalMean(x,n)-volume);
      kkt=opt.KKTresidual(x,df0,0.0,h,&dh,xmin,xmax);
   }

   if(rank_id==0)
      std::printf("  ranks=%d  zero_control_ranks=%d  kkt=%.3e"
                  "  residual=%.3e  max_error=%.3e  iterations=%d\n",
                  rank_count,zero_ranks,double(kkt),std::abs(double(h(0))),
                  max_error,opt.NumIterations());
   Check(zero_ranks==expected_zero_ranks,
         "expected number of zero-control ranks participated");
   Check(std::isfinite(double(kkt)),"zero-control-rank solve remains finite");
   Check(std::abs(double(h(0)))<1e-10,
         "zero-control-rank equality is satisfied");
   Check(max_error<1e-3,
         "zero-control-rank solution matches analytic optimum");
   Check(double(kkt)<1e-8,"zero-control-rank KKT residual < 1e-8");
}

void TestMoreEqualitiesThanControls()
{
   if(rank_id==0)
      std::printf("\n--- Parallel overdetermined equalities (n=12, m=22) ---\n");
   const int n=12,m=2*(n-1);
   const auto distribution=Distribute(n);
   const int nl=distribution.first,offset=distribution.second;
   Vector x(nl),target(nl),xmin(nl),xmax(nl),df0(nl),h(m);
   std::vector<Vector> dh(m);
   xmin=0.0; xmax=1.0;
   for(int i=0;i<m;++i) { dh[i].SetSize(nl); dh[i]=0.0; }
   for(int j=0;j<nl;++j)
   {
      const int global=offset+j;
      x(j)=real_t(0.35+0.20*double(global)/double(n-1));
      target(j)=real_t(double(x(j))+0.05);
      for(int k=1;k<n;++k)
      {
         const int row=2*(k-1);
         if(global==0) { dh[row](j)=-1.0; dh[row+1](j)=-2.0; }
         if(global==k) { dh[row](j)= 1.0; dh[row+1](j)= 2.0; }
      }
   }

   MMAEqualityOptimizerParallel opt(MPI_COMM_WORLD,nl,m);
   opt.SetAsymptotes(0.15,0.7,1.2);
   real_t kkt=1.0;
   double max_error=1.0,max_residual=1.0;
   std::vector<double> local_x(n),global_x(n);
   for(int outer=0;outer<300 &&
       (opt.NumIterations()<10 || max_error>1e-4 || double(kkt)>1e-10);++outer)
   {
      double local_f0=0.0;
      for(int j=0;j<nl;++j)
      {
         const double difference=double(x(j))-double(target(j));
         local_f0+=0.5*difference*difference/n;
         df0(j)=real_t(difference/n);
      }
      std::fill(local_x.begin(),local_x.end(),0.0);
      for(int j=0;j<nl;++j) local_x[offset+j]=double(x(j));
      MPI_Allreduce(local_x.data(),global_x.data(),n,MPI_DOUBLE,MPI_SUM,
                    MPI_COMM_WORLD);
      for(int j=1;j<n;++j)
      {
         const int row=2*(j-1);
         const double prescribed=0.20*double(j)/double(n-1);
         const double residual=global_x[j]-global_x[0]-prescribed;
         h(row)=real_t(residual); h(row+1)=real_t(2.0*residual);
      }
      opt.Update(x,df0,real_t(GlobalSum(local_f0)),h,dh.data(),xmin,xmax);

      double local_error=0.0;
      for(int j=0;j<nl;++j)
      {
         df0(j)=real_t((double(x(j))-double(target(j)))/n);
         local_error=std::max(local_error,
                              std::abs(double(x(j))-double(target(j))));
      }
      max_error=GlobalMax(local_error);
      std::fill(local_x.begin(),local_x.end(),0.0);
      for(int j=0;j<nl;++j) local_x[offset+j]=double(x(j));
      MPI_Allreduce(local_x.data(),global_x.data(),n,MPI_DOUBLE,MPI_SUM,
                    MPI_COMM_WORLD);
      max_residual=0.0;
      for(int j=1;j<n;++j)
      {
         const int row=2*(j-1);
         const double prescribed=0.20*double(j)/double(n-1);
         const double residual=global_x[j]-global_x[0]-prescribed;
         h(row)=real_t(residual); h(row+1)=real_t(2.0*residual);
         max_residual=std::max(max_residual,2.0*std::abs(residual));
      }
      kkt=opt.KKTresidual(x,df0,0.0,h,dh.data(),xmin,xmax);
   }

   if(rank_id==0)
      std::printf("  kkt=%.3e  max_residual=%.3e  max_error=%.3e"
                  "  iterations=%d\n",double(kkt),max_residual,max_error,
                  opt.NumIterations());
   Check(std::isfinite(double(kkt)),"overdetermined parallel solve remains finite");
   Check(max_residual<1e-9,"all overdetermined parallel equalities are satisfied");
   Check(max_error<1e-3,"overdetermined parallel solution matches analytic optimum");
   Check(double(kkt)<1e-8,"overdetermined parallel KKT residual < 1e-8");
}

} // namespace

int main(int argc,char **argv)
{
   MPI_Init(&argc,&argv);
   MPI_Comm_rank(MPI_COMM_WORLD,&rank_id);
   MPI_Comm_size(MPI_COMM_WORLD,&rank_count);
   if(rank_id==0)
      std::printf("=== Parallel equality-only MMA examples (%d ranks) ===\n",rank_count);
   TestAffineRestoration();
   TestAffineOptimization();
   TestNonlinearRestoration();
   TestGlobalizationRetries();
   TestGlobalizationRollback();
   TestRedundantEqualities();
   TestManyEqualities();
   TestZeroControlRanks();
   TestMoreEqualitiesThanControls();
   if(rank_id==0)
      std::printf("\n%s\n",failures==0 ? "All parallel examples PASSED."
                                         : "Parallel equality-only MMA failures detected.");
   int global_failures=0;
   MPI_Bcast(&failures,1,MPI_INT,0,MPI_COMM_WORLD);
   global_failures=failures;
   MPI_Finalize();
   return global_failures==0 ? 0 : 1;
}

#else

int main()
{
   std::printf("MFEM was built without MPI; parallel equality example skipped.\n");
   return 0;
}

#endif
