#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include "coefficients.hpp"


int main(int argc, char *argv[])
{

    mfem::ElementTransformation* Tr;
    mfem::IntegrationPoint* ip;


    mfem::IsoElastMat mat; mat.SetElastParam(1,0.0);
    mfem::J2YieldFunction yf(1.0,0.10,1.0);

    mfem::StressEval<mfem::IsoElastMat,mfem::J2YieldFunction> se(&mat,&yf);

    mfem::Vector ss(9); ss=0.0;
    mfem::Vector ep(9); ep=0.0;
    mfem::Vector iv(2); iv=0.0;

    mfem::Vector dee(9); dee=0.0;
    mfem::Vector ee(9); ee=0.0;
    mfem::Vector epn(9); epn=0.0;
    mfem::Vector ivn(2); ivn=0.0;

    int bb=8;

    for(int i=0;i<20;i++){


        std::cout<<i<<" " ;//std::endl;

        dee(bb)=0.11;
        ee.Add(1.0,dee);
        se.SetStrain(ee);
        se.SetPlasticStrain(epn);
        se.SetInternalParameters(ivn);
        se.Solve(ss,epn,ivn,*Tr,*ip);
        //epn=ep;
        //ivn=iv;

        std::cout<<ee[bb]<<" "<<ss[bb]<<" "<<ss[1]<<" "<<ep[bb]<<" "<<yf.Eval(ss,iv)<<std::endl;

        /*
        std::cout<<"strain="<<std::endl;
        ee.Print(std::cout,3);

        std::cout<<"stress="<<std::endl;
        std::cout.precision(6);
        ss.Print(std::cout,3);

        iv.Print(std::cout);
        */

    }

    for(int i=0;i<40;i++){


        std::cout<<i<<" " ;//std::endl;

        dee(bb)=-0.11;
        ee.Add(1.0,dee);
        se.SetStrain(ee);
        se.SetPlasticStrain(epn);
        se.SetInternalParameters(ivn);
        se.Solve(ss,ep,iv,*Tr,*ip);
        epn=ep;
        ivn=iv;

        std::cout<<ee[bb]<<" "<<ss[bb]<<" "<<ss[1]<<" "<<ep[bb]<<" "<<yf.Eval(ss,iv)<<std::endl;

        /*
        std::cout<<"strain="<<std::endl;
        ee.Print(std::cout,3);

        std::cout<<"stress="<<std::endl;
        std::cout.precision(6);
        ss.Print(std::cout,3);

        iv.Print(std::cout);
        */

    }

    for(int i=0;i<25;i++){


        std::cout<<i<<" " ;//std::endl;

        dee(bb)=0.11;
        ee.Add(1.0,dee);
        se.SetStrain(ee);
        se.SetPlasticStrain(epn);
        se.SetInternalParameters(ivn);
        se.Solve(ss,ep,iv,*Tr,*ip);
        epn=ep;
        ivn=iv;

        std::cout<<ee[bb]<<" "<<ss[bb]<<" "<<ss[1]<<" "<<ep[bb]<<" "<<yf.Eval(ss,iv)<<std::endl;

        /*
        std::cout<<"strain="<<std::endl;
        ee.Print(std::cout,3);

        std::cout<<"stress="<<std::endl;
        std::cout.precision(6);
        ss.Print(std::cout,3);

        iv.Print(std::cout);
        */

    }

    for(int i=0;i<15;i++){


        std::cout<<i<<" " ;//std::endl;

        dee(bb)=-0.11;
        ee.Add(1.0,dee);
        se.SetStrain(ee);
        se.SetPlasticStrain(epn);
        se.SetInternalParameters(ivn);
        se.Solve(ss,ep,iv,*Tr,*ip);
        epn=ep;
        ivn=iv;

        std::cout<<ee[bb]<<" "<<ss[bb]<<" "<<ss[1]<<" "<<ep[bb]<<" "<<yf.Eval(ss,iv)<<std::endl;

        /*
        std::cout<<"strain="<<std::endl;
        ee.Print(std::cout,3);

        std::cout<<"stress="<<std::endl;
        std::cout.precision(6);
        ss.Print(std::cout,3);

        iv.Print(std::cout);
        */

    }


    return 0;
}
