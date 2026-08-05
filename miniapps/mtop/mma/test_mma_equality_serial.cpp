/**
 * Serial examples for MMAEqualityOptimizer.
 *
 * Covers:
 *   1. Affine feasibility restoration.
 *   2. Equality-constrained optimization with an analytic solution.
 *   3. Repeated restoration of a nonlinear equality.
 */

#include "MMA_Equality_MFEM.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <utility>
#include <vector>

using namespace mfem;
using namespace mfem_mma;

namespace {

int failures=0;

void Check(bool condition,const char *message)
{
   if(condition) std::printf("  [PASS] %s\n",message);
   else { std::printf("  [FAIL] %s\n",message); ++failures; }
}

double Mean(const Vector &x)
{
   double sum=0.0;
   for(int j=0;j<x.Size();++j) sum+=double(x(j));
   return sum/std::max(x.Size(),1);
}

void TestAffineRestoration()
{
   std::printf("\n--- Serial affine restoration ---\n");
   const int n=100;
   const double volume=0.4;
   Vector x(n),xmin(n),xmax(n),h(1),dh(n);
   x=0.8; xmin=0.0; xmax=1.0; dh=real_t(1.0/n);
   h(0)=real_t(Mean(x)-volume);

   MMAEqualityOptimizer opt(n,1);
   const real_t affine_residual=
      opt.RestoreFeasibility(x,h,&dh,xmin,xmax);

   std::printf("  residual=%.3e  mean=%.8f  iterations=%d\n",
               double(affine_residual),Mean(x),opt.NumIterations());
   Check(double(affine_residual)<1e-10,"affine projection residual < 1e-10");
   Check(std::abs(Mean(x)-volume)<1e-10,"restored design satisfies equality");
   Check(opt.NumIterations()==0,"restoration does not advance MMA history");
}

void TestAffineOptimization()
{
   std::printf("\n--- Serial equality-constrained optimization ---\n");
   const int n=200;
   const double volume=0.5;
   Vector x(n),target(n),exact(n),xmin(n),xmax(n),df0(n),h(1),dh(n);
   x=volume; xmin=0.0; xmax=1.0; dh=real_t(1.0/n);
   double target_mean=0.0;
   for(int j=0;j<n;++j)
   {
      target(j)=real_t(0.2+0.4*double(j)/double(n-1));
      target_mean+=double(target(j))/n;
   }
   for(int j=0;j<n;++j)
      exact(j)=real_t(double(target(j))+volume-target_mean);

   MMAEqualityOptimizer opt(n,1);
   opt.SetAsymptotes(0.15,0.7,1.2);
   real_t kkt=1.0;
   double analytic_error=1.0;
   int outer=0,total_inner=0,rejected=0;
   for(;outer<1000 && (opt.NumIterations()<10 || double(kkt)>1e-12 ||
                       analytic_error>1e-3);++outer)
   {
      double f0=0.0;
      for(int j=0;j<n;++j)
      {
         const double difference=double(x(j))-double(target(j));
         f0+=0.5*difference*difference/n;
         df0(j)=real_t(difference/n);
      }
      h(0)=real_t(Mean(x)-volume);
      int inner=0;
      opt.UpdateGCMMA(x,df0,real_t(f0),h,&dh,xmin,xmax,
         [&](const Vector &candidate,real_t &true_f0,Vector &true_h)
         {
            double value=0.0;
            for(int j=0;j<n;++j)
            {
               const double difference=double(candidate(j))-double(target(j));
               value+=0.5*difference*difference/n;
            }
            true_f0=real_t(value);
            true_h.SetSize(1);
            true_h(0)=real_t(Mean(candidate)-volume);
         },50,&inner);
      total_inner+=inner;
      if(!opt.LastStepAccepted())
      {
         ++rejected;
         kkt=1.0;
         continue;
      }

      for(int j=0;j<n;++j)
         df0(j)=real_t((double(x(j))-double(target(j)))/n);
      h(0)=real_t(Mean(x)-volume);
      kkt=opt.KKTresidual(x,df0,0.0,h,&dh,xmin,xmax);
      analytic_error=0.0;
      for(int j=0;j<n;++j)
         analytic_error=std::max(analytic_error,
                            std::abs(double(x(j))-double(exact(j))));
   }

   double max_error=0.0;
   for(int j=0;j<n;++j)
      max_error=std::max(max_error,std::abs(double(x(j))-double(exact(j))));
   std::printf("  kkt=%.3e  mean=%.8f  max_error=%.3e"
               "  iterations=%d  outer_attempts=%d  rejected=%d"
               "  total_inner=%d\n",
               double(kkt),Mean(x),max_error,opt.NumIterations(),outer,
               rejected,total_inner);
   Check(double(kkt)<1e-10,"KKT residual < 1e-10");
   Check(opt.NumIterations()>=10,"GCMMA performs at least ten outer iterations");
   Check(outer<1000,"GCMMA finishes within the enlarged outer budget");
   Check(std::abs(Mean(x)-volume)<1e-8,"affine equality remains satisfied");
   Check(max_error<2e-3,"solution matches analytic constrained optimum");
}

void TestNonlinearRestoration()
{
   std::printf("\n--- Serial nonlinear restoration ---\n");
   const int n=80;
   const double target_square=0.16;
   Vector x(n),xmin(n),xmax(n),h(1),dh(n);
   x=0.7; xmin=0.05; xmax=1.0;
   MMAEqualityOptimizer opt(n,1);

   double true_residual=1.0;
   int restoration_steps=0;
   for(;restoration_steps<12 && true_residual>1e-10;++restoration_steps)
   {
      double square_mean=0.0;
      for(int j=0;j<n;++j)
      {
         square_mean+=double(x(j))*double(x(j))/n;
         dh(j)=real_t(2.0*double(x(j))/n);
      }
      h(0)=real_t(square_mean-target_square);
      true_residual=std::abs(double(h(0)));
      if(true_residual<=1e-10) break;
      opt.RestoreFeasibility(x,h,&dh,xmin,xmax);
   }
   double square_mean=0.0;
   for(int j=0;j<n;++j) square_mean+=double(x(j))*double(x(j))/n;
   true_residual=std::abs(square_mean-target_square);

   std::printf("  true_residual=%.3e  mean=%.8f  restoration_steps=%d\n",
               true_residual,Mean(x),restoration_steps);
   Check(true_residual<1e-10,"nonlinear equality restored by reevaluation loop");
   Check(std::abs(Mean(x)-0.4)<1e-8,"restoration reaches positive root x=0.4");
   Check(opt.NumIterations()==0,"nonlinear restoration leaves MMA history untouched");
}

void TestGlobalizationRetries()
{
   std::printf("\n--- Serial GCMMA curvature retries ---\n");
   const int n=40;
   Vector x(n),xk(n),xmin(n),xmax(n),df0(n),h(1),dh(n);
   x=0.5; xk=x; xmin=0.0; xmax=1.0; dh=real_t(1.0/n);
   for(int j=0;j<n;++j) df0(j)=real_t((j%2==0 ? 1.0 : -1.0)/n);
   h(0)=0.0;
   MMAEqualityOptimizer opt(n,1);
   int inner=0;
   opt.UpdateGCMMA(x,df0,0.0,h,&dh,xmin,xmax,
      [&](const Vector &candidate,real_t &true_f0,Vector &true_h)
      {
         double linear=0.0,square=0.0;
         for(int j=0;j<n;++j)
         {
            const double step=double(candidate(j))-double(xk(j));
            linear+=double(df0(j))*step;
            square+=step*step/n;
         }
         // Matches f0 and df0 at xk but is deliberately much more curved
         // than the initial MMA objective model.
         true_f0=real_t(linear+100.0*square);
         true_h.SetSize(1);
         true_h(0)=real_t(Mean(candidate)-0.5);
      },15,&inner);

   std::printf("  accepted=%d  inner=%d  mean=%.8f  iterations=%d\n",
               opt.LastStepAccepted()?1:0,inner,Mean(x),opt.NumIterations());
   Check(inner>1,"non-conservative objective triggers curvature retries");
   Check(opt.LastStepAccepted(),"globalized candidate is eventually accepted");
   Check(std::abs(Mean(x)-0.5)<1e-9,"accepted candidate satisfies equality");
   Check(opt.NumIterations()==1,"accepted GCMMA step advances history once");
}

void TestGlobalizationRollback()
{
   std::printf("\n--- Serial GCMMA rejection and rollback ---\n");
   const int n=20,max_inner=2;
   Vector x(n),x_before(n),xmin(n),xmax(n),df0(n),h(1),dh(n);
   x=0.5; x_before=x; xmin=0.0; xmax=1.0; dh=real_t(1.0/n);
   for(int j=0;j<n;++j) df0(j)=real_t((j%2==0 ? 1.0 : -1.0)/n);
   h(0)=0.0;
   MMAEqualityOptimizer opt(n,1);
   int inner=0;
   opt.UpdateGCMMA(x,df0,0.0,h,&dh,xmin,xmax,
      [&](const Vector &candidate,real_t &true_f0,Vector &true_h)
      {
         true_f0=real_t(1e20);
         true_h.SetSize(1);
         true_h(0)=real_t(Mean(candidate)-0.5);
      },max_inner,&inner);

   double max_change=0.0;
   for(int j=0;j<n;++j)
      max_change=std::max(max_change,std::abs(double(x(j))-double(x_before(j))));
   std::printf("  accepted=%d  inner=%d  max_change=%.3e  iterations=%d\n",
               opt.LastStepAccepted()?1:0,inner,max_change,opt.NumIterations());
   Check(inner==max_inner,"globalization respects the inner-iteration limit");
   Check(!opt.LastStepAccepted(),"unacceptable candidate is rejected");
   Check(max_change==0.0,"rejected step restores the original design");
   Check(opt.NumIterations()==0,"rejected step does not advance MMA history");
}

void TestRedundantEqualities()
{
   std::printf("\n--- Serial redundant equality constraints ---\n");
   const int n=120;
   const double volume=0.5;
   Vector x(n),target(n),exact(n),xmin(n),xmax(n),df0(n),h(2);
   Vector dh[2]; dh[0].SetSize(n); dh[1].SetSize(n);
   x=volume; xmin=0.0; xmax=1.0;
   dh[0]=real_t(1.0/n); dh[1]=real_t(2.0/n);
   double target_mean=0.0;
   for(int j=0;j<n;++j)
   {
      target(j)=real_t(0.25+0.30*double(j)/double(n-1));
      target_mean+=double(target(j))/n;
   }
   for(int j=0;j<n;++j)
      exact(j)=real_t(double(target(j))+volume-target_mean);

   MMAEqualityOptimizer opt(n,2);
   opt.SetAsymptotes(0.15,0.7,1.2);
   real_t kkt=1.0;
   double max_error=1.0;
   for(int outer=0;outer<400 &&
       (opt.NumIterations()<10 || max_error>1e-3 || double(kkt)>1e-12);++outer)
   {
      double f0=0.0;
      for(int j=0;j<n;++j)
      {
         const double difference=double(x(j))-double(target(j));
         f0+=0.5*difference*difference/n;
         df0(j)=real_t(difference/n);
      }
      const double residual=Mean(x)-volume;
      h(0)=real_t(residual); h(1)=real_t(2.0*residual);
      opt.Update(x,df0,real_t(f0),h,dh,xmin,xmax);
      for(int j=0;j<n;++j)
         df0(j)=real_t((double(x(j))-double(target(j)))/n);
      const double updated_residual=Mean(x)-volume;
      h(0)=real_t(updated_residual); h(1)=real_t(2.0*updated_residual);
      kkt=opt.KKTresidual(x,df0,0.0,h,dh,xmin,xmax);
      max_error=0.0;
      for(int j=0;j<n;++j)
         max_error=std::max(max_error,
                            std::abs(double(x(j))-double(exact(j))));
   }

   const auto &lambda=opt.GetLambda();
   const bool finite=std::isfinite(double(kkt)) && std::isfinite(max_error) &&
                     std::isfinite(lambda[0]) && std::isfinite(lambda[1]);
   std::printf("  kkt=%.3e  h=[%.3e,%.3e]  max_error=%.3e"
               "  lambda=[%.3e,%.3e]  iterations=%d\n",
               double(kkt),double(h(0)),double(h(1)),max_error,
               lambda[0],lambda[1],opt.NumIterations());
   Check(finite,"redundant serial solve remains finite");
   Check(std::abs(double(h(0)))<1e-9 && std::abs(double(h(1)))<2e-9,
         "both redundant equalities are satisfied");
   Check(max_error<2e-3,"redundant serial solution matches analytic optimum");
   Check(double(kkt)<1e-10,"redundant serial KKT residual < 1e-10");
}

void TestManyEqualities()
{
   std::printf("\n--- Serial 100 regional equality constraints ---\n");
   const int m=100,region_size=10,n=m*region_size;
   Vector x(n),target(n),xmin(n),xmax(n),df0(n),h(m);
   std::vector<Vector> dh(m);
   xmin=0.0; xmax=1.0;
   for(int i=0;i<m;++i)
   {
      dh[i].SetSize(n);
      dh[i]=0.0;
      const double volume=0.35+0.30*double(i%5)/4.0;
      for(int k=0;k<region_size;++k)
      {
         const int j=i*region_size+k;
         dh[i](j)=real_t(1.0/region_size);
         x(j)=real_t(volume);
         target(j)=real_t(volume+0.08*(double(k)-4.5)/4.5);
      }
   }

   MMAEqualityOptimizer opt(n,m);
   opt.SetAsymptotes(0.15,0.7,1.2);
   real_t kkt=1.0;
   double max_error=1.0,max_residual=1.0;
   for(int outer=0;outer<500 &&
       (opt.NumIterations()<10 || max_error>1e-3 || double(kkt)>1e-10);++outer)
   {
      double f0=0.0;
      for(int j=0;j<n;++j)
      {
         const double difference=double(x(j))-double(target(j));
         f0+=0.5*difference*difference/n;
         df0(j)=real_t(difference/n);
      }
      for(int i=0;i<m;++i)
      {
         double mean=0.0;
         for(int k=0;k<region_size;++k)
            mean+=double(x(i*region_size+k))/region_size;
         h(i)=real_t(mean-(0.35+0.30*double(i%5)/4.0));
      }
      opt.Update(x,df0,real_t(f0),h,dh.data(),xmin,xmax);

      max_error=0.0;
      for(int j=0;j<n;++j)
      {
         df0(j)=real_t((double(x(j))-double(target(j)))/n);
         max_error=std::max(max_error,
                            std::abs(double(x(j))-double(target(j))));
      }
      max_residual=0.0;
      for(int i=0;i<m;++i)
      {
         double mean=0.0;
         for(int k=0;k<region_size;++k)
            mean+=double(x(i*region_size+k))/region_size;
         h(i)=real_t(mean-(0.35+0.30*double(i%5)/4.0));
         max_residual=std::max(max_residual,std::abs(double(h(i))));
      }
      kkt=opt.KKTresidual(x,df0,0.0,h,dh.data(),xmin,xmax);
   }

   std::printf("  kkt=%.3e  max_residual=%.3e  max_error=%.3e"
               "  iterations=%d\n",double(kkt),max_residual,max_error,
               opt.NumIterations());
   Check(std::isfinite(double(kkt)),"100-equality serial solve remains finite");
   Check(max_residual<1e-9,"all 100 serial equalities are satisfied");
   Check(max_error<2e-3,"100-equality serial solution matches analytic optimum");
   Check(double(kkt)<1e-8,"100-equality serial KKT residual < 1e-8");
}

void TestMoreEqualitiesThanControls()
{
   std::printf("\n--- Serial overdetermined equalities (n=12, m=22) ---\n");
   const int n=12,m=2*(n-1);
   Vector x(n),target(n),xmin(n),xmax(n),df0(n),h(m);
   std::vector<Vector> dh(m);
   xmin=0.0; xmax=1.0;
   for(int j=0;j<n;++j)
   {
      x(j)=real_t(0.35+0.20*double(j)/double(n-1));
      target(j)=real_t(double(x(j))+0.05);
   }
   for(int j=1;j<n;++j)
   {
      const int row=2*(j-1);
      dh[row].SetSize(n); dh[row]=0.0;
      dh[row+1].SetSize(n); dh[row+1]=0.0;
      dh[row](0)=-1.0; dh[row](j)=1.0;
      dh[row+1](0)=-2.0; dh[row+1](j)=2.0;
   }

   MMAEqualityOptimizer opt(n,m);
   opt.SetAsymptotes(0.15,0.7,1.2);
   real_t kkt=1.0;
   double max_error=1.0,max_residual=1.0;
   for(int outer=0;outer<300 &&
       (opt.NumIterations()<10 || max_error>1e-4 || double(kkt)>1e-10);++outer)
   {
      double f0=0.0;
      for(int j=0;j<n;++j)
      {
         const double difference=double(x(j))-double(target(j));
         f0+=0.5*difference*difference/n;
         df0(j)=real_t(difference/n);
      }
      max_residual=0.0;
      for(int j=1;j<n;++j)
      {
         const int row=2*(j-1);
         const double residual=(double(x(j))-double(x(0)))-
                               (double(target(j))-double(target(0)));
         h(row)=real_t(residual); h(row+1)=real_t(2.0*residual);
         max_residual=std::max(max_residual,2.0*std::abs(residual));
      }
      opt.Update(x,df0,real_t(f0),h,dh.data(),xmin,xmax);

      max_error=0.0;
      for(int j=0;j<n;++j)
      {
         df0(j)=real_t((double(x(j))-double(target(j)))/n);
         max_error=std::max(max_error,
                            std::abs(double(x(j))-double(target(j))));
      }
      max_residual=0.0;
      for(int j=1;j<n;++j)
      {
         const int row=2*(j-1);
         const double residual=(double(x(j))-double(x(0)))-
                               (double(target(j))-double(target(0)));
         h(row)=real_t(residual); h(row+1)=real_t(2.0*residual);
         max_residual=std::max(max_residual,2.0*std::abs(residual));
      }
      kkt=opt.KKTresidual(x,df0,0.0,h,dh.data(),xmin,xmax);
   }

   std::printf("  kkt=%.3e  max_residual=%.3e  max_error=%.3e"
               "  iterations=%d\n",double(kkt),max_residual,max_error,
               opt.NumIterations());
   Check(std::isfinite(double(kkt)),"overdetermined serial solve remains finite");
   Check(max_residual<1e-9,"all overdetermined serial equalities are satisfied");
   Check(max_error<1e-3,"overdetermined serial solution matches analytic optimum");
   Check(double(kkt)<1e-8,"overdetermined serial KKT residual < 1e-8");
}

} // namespace

int main()
{
   std::printf("=== Serial equality-only MMA examples ===\n");
   TestAffineRestoration();
   TestAffineOptimization();
   TestNonlinearRestoration();
   TestGlobalizationRetries();
   TestGlobalizationRollback();
   TestRedundantEqualities();
   TestManyEqualities();
   TestMoreEqualitiesThanControls();
   std::printf("\n%s\n",failures==0 ? "All serial examples PASSED."
                                      : "Serial equality-only MMA failures detected.");
   return failures==0 ? 0 : 1;
}
