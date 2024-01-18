#include "mtop_coefficients.hpp"
#include <vector>
#include <iostream>
#include <iterator>
#include <fstream>

int main()
{
   std::vector<double> s= {1.1,2.1,3.1,0.4,3.6,4.1};
   std::vector<double> p= {0.1,0.1,0.1,0.1,0.1,0.1};

   mfem::RiskMeasures rm(s);
   mfem::RiskMeasures ra(s,p);

   /*
   std::vector<double> ss(s.size());
   std::copy(s.begin(),s.end(),ss.begin());
   std::sort(ss.begin(),ss.end());
   std::copy(ss.begin(),ss.end(), std::ostream_iterator<double>(std::cout, " "));
   std::cout<<std::endl;*/

   for (double id=0.00; id<1.0; id=id+0.05)
   {
      std::cout<<"id="<<id<<" VaR(id)="<<rm.VaR(id)<<" CVaR(id)="<<rm.CVaR(id);
      std::cout<<" mean="<< rm.Mean()<<" STD="<<rm.STD()<<" EVaR="<<rm.EVaR(id);
      std::cout<<std::endl;
      //std::cout<<" EvaR_t="<<rm.EVaR_Find_t(id)<<std::endl;
      //std::cout<<" CVaRe t="<<rm.CVaRe_Find_t(id,1e-1)<<std::endl;
      std::cout<<"id="<<id<<" VaR(id)="<<ra.VaR(id)<<" CVaR(id)="<<ra.CVaR(id);
      std::cout<<" mean="<< ra.Mean()<<" STD="<<ra.STD()<<" EVaR="<<ra.EVaR(id);
      std::cout<<std::endl;
      //std::cout<<" EvaR_t="<<ra.EVaR_Find_t(id)<<std::endl;
      //std::cout<<" CVaRe t="<<ra.CVaRe_Find_t(id,1e-1)<<std::endl;

      std::cout<<"rm.dt="<<rm.dEVaR(id,0.5)<<" ad="<<rm.ADEVaRt(id,0.5)<<std::endl;
      std::cout<<"ra.dt="<<ra.dEVaR(id,0.5)<<" ad="<<rm.ADEVaRt(id,0.5)<<std::endl;
   }



   rm.Test_EVaR_Grad();
   ra.Test_EVaR_Grad();






}
