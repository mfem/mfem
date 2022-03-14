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

    if((tmpv*tmpv)<(radius*radius))
    {
        return 0.0;
    }

    return 1.0;

}

void RandShootingCoefficient::Sample()
{
    for(int i=0;i<center.Size();i++)
    {
        center(i)=udist(generator);
    }
    //check if the mesh is pmesh
#ifdef MFEM_USE_MPI
    ParMesh* pmesh=dynamic_cast<ParMesh*>(mesh);
    MFEM_ASSERT(pmesh==nullptr,
                "RandShootingCoefficient should be initialized with parallel mesh when MPI is enabled!");
    MPI_Comm comm=pmesh->GetComm();
    int myrank=pmesh->GetMyRank();
    if(myrank==0)
    {
        //generate the sample
        for(int i=0;i<center.Size();i++)
        {
            center(i)=cmin(i)+udist(generator)*(cmax(i)-cmin(i)) ;
        }
    }
    //communicate the center from process zero to all others
    MPI_Scatter(center.GetData(),center.Size(),MPI_DOUBLE,
                center.GetData(), center.Size(), MPI_DOUBLE, 0, comm);
#else //serial version
    for(int i=0;i<center.Size();i++)
    {
        center(i)=cmin(i)+udist(generator)*(cmax(i)-cmin(i));
    }
#endif
}

}
