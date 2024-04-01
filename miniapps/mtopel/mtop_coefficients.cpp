#include "mtop_coefficients.hpp"

namespace  mfem {

double RandDiscreteShootingCoefficient::Eval(ElementTransformation& T, const IntegrationPoint& ip)
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

void RandDiscreteShootingCoefficient::GetPosition(int gi, Vector& p)
{
   //gi=i+j*nx+k*nx*ny;
   p.SetSize(cmin.Size());

   if(gi>nx*ny*nz){
       p=0.0;
       return;
   }

   int k=gi/(nx*ny);
   int j=(gi-k*nx*ny)/nx;
   int i=gi-k*nx*ny-j*nx;
   p[0]=cmin[0]+dx[0]/2.0+dx[0]*i;
   if(cmin.Size()>1){ p[1]=cmin[1]+dx[1]/2.0+dx[1]*j;}
   if(cmin.Size()>2){ p[2]=cmin[2]+dx[2]/2.0+dx[2]*k;}
}


void RandDiscreteShootingCoefficient::Sample()
{
    if(udist==nullptr){
        udist=new std::uniform_int_distribution<int>(0,nx*ny*nz);
    }

#ifdef MFEM_USE_MPI
    ParMesh* pmesh=dynamic_cast<ParMesh*>(mesh);
    if(pmesh!=nullptr){
        MPI_Comm comm=pmesh->GetComm();
        int myrank=pmesh->GetMyRank();
        if(myrank==0)
        {
            //generate the sample
            int gi=udist->operator()(generator);
            GetPosition(gi,tmpv);
        }
        //communicate the center from process zero to all others
        MPI_Scatter(tmpv.GetData(), tmpv.Size(),MPI_DOUBLE,
                center.GetData(), center.Size(), MPI_DOUBLE, 0, comm);
    }else{
        //serial mesh
        //generate the sample
        int gi=udist->operator()(generator);
        GetPosition(gi,center);
    }
#else
    //serial version
    //generate the sample
    int gi=udist->operator()(generator);
    GetPosition(gi,center);
#endif
}

void RandDiscreteShootingCoefficient::Sample(int gi)
{
    GetPosition(gi,center);
}


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
