#include "mtop_coefficients.hpp"

namespace  mfem {


double RandShootingCoefficient::Eval(ElementTransformation& T, const IntegrationPoint& ip)
{
    tmpv.SetSize(cmin.Size());
    T.Transform(ip,tmpv);

    for(int i=0;i<tmpv.Size();i++)
    {
        tmpv[i]=tmpv[i]-center[i];
    }

    if(tmpv.Norml2()<radius)
    {
        return 0.0;
    }

    return 1.0;
}

void RandShootingCoefficient::Sample()
{
#ifdef MFEM_USE_MPI
    ParMesh* pmesh=dynamic_cast<ParMesh*>(mesh);
    if(pmesh!=nullptr){
        MPI_Comm comm=pmesh->GetComm();
        int myrank=pmesh->GetMyRank();
        if(myrank==0)
        {
            //generate the sample
            for(int i=0;i<cmin.Size();i++)
            {
                tmpv(i)=cmin(i)+udist(generator)*(cmax(i)-cmin(i)) ;
            }
        }
        //communicate the center from process zero to all others
        MPI_Scatter(tmpv.GetData(), tmpv.Size(),MPI_DOUBLE,
                center.GetData(), center.Size(), MPI_DOUBLE, 0, comm);
    }else{
        //serial mesh
        for(int i=0;i<center.Size();i++)
        {
            center(i)=cmin(i)+udist(generator)*(cmax(i)-cmin(i));
        }
    }
#else //serial version
    for(int i=0;i<center.Size();i++)
    {
        center(i)=cmin(i)+udist(generator)*(cmax(i)-cmin(i));
    }
#endif
}

}
