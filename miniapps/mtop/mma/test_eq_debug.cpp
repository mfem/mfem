
#include "MMA_MFEM.hpp"
#include <mfem.hpp>
#include <mpi.h>
#include <cmath>
#include <cstdio>
using namespace mfem;
using namespace mfem_mma;

int main(int argc, char** argv){
    MPI_Init(&argc,&argv);
    const int n=10; const double Vfrac=0.4;
    Vector a(n),x(n),xmin(n),xmax(n),df0(n),dh(n);
    for(int j=0;j<n;++j) a(j)=real_t(0.3+0.4*j/n);
    x=real_t(Vfrac); xmin=real_t(0.01); xmax=real_t(1.0);
    for(int j=0;j<n;++j) dh(j)=real_t(1.0/n);

    auto opt=MMAOptimizer::WithEqualities(n,0,1,x);

    for(int it=0;it<200&&!std::isnan(double(x(0)));++it){
        for(int j=0;j<n;++j) df0(j)=real_t(2.0*(double(x(j))-double(a(j)))/n);
        double f0=0; for(int j=0;j<n;++j) f0+=std::pow(double(x(j))-double(a(j)),2)/n;
        double xm=0; for(int j=0;j<n;++j) xm+=double(x(j)); xm/=n;
        Vector h_eq(1); h_eq(0)=real_t(xm-Vfrac);
        Vector fival=PackFival(Vector(0),h_eq);
        Vector dh_arr[1]={dh};
        PackedDfidx dfidx(nullptr,0,dh_arr,1);

        opt.Update(x,df0,real_t(f0),fival,dfidx.data(),xmin,xmax);
        
        // check x after update
        bool xnan=false;
        for(int j=0;j<n;++j) if(std::isnan(double(x(j)))) xnan=true;
        // print lam values
        auto lam = opt.GetLambda();
        if(it%50==0||it==199)
          printf("  iter %3d: x[0]=%.4e x[n-1]=%.4e nan=%d  lam[0]=%.4e  kkt_skip\n",
               it,double(x(0)),double(x(n-1)),xnan?1:0, lam.empty()?0.0:lam[0]);
        if(xnan) break;
        
        for(int j=0;j<n;++j) df0(j)=real_t(2.0*(double(x(j))-double(a(j)))/n);
        xm=0; for(int j=0;j<n;++j) xm+=double(x(j)); xm/=n;
        h_eq(0)=real_t(xm-Vfrac);
        fival=PackFival(Vector(0),h_eq);
        real_t kkt=opt.KKTresidual(x,df0,real_t(f0),fival,dfidx.data(),xmin,xmax);
        printf("  kkt=%.4e  xmean=%.4e\n",double(kkt),xm);
        { int _r=0; MPI_Comm_rank(MPI_COMM_WORLD,&_r); if(_r==0&&(it%50==0||it==199)) printf("  iter %3d: kkt=%.4e  xmean=%.6f\n",it,double(kkt),xm); }
        if(std::isnan(double(kkt))) break;
    }
    MPI_Finalize(); return 0;
}
