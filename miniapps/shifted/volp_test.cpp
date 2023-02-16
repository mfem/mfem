#include <fstream>
#include <iostream>

#include "mfem.hpp"
#include "marking.hpp"
#include "mtop_solvers.hpp"
#include "mtop_filters.hpp"
#include "shape_grad.hpp"

using namespace mfem;
using namespace std;


class GyroidCoeff:public Coefficient
{
public:
    GyroidCoeff(double cell_size=1.0){
        ll=cell_size;
    }

    virtual
    double Eval(ElementTransformation &T,
                const IntegrationPoint &ip)
    {
        //evaluate the true coordinate of the ip
        Vector xx; xx.SetSize(T.GetDimension());
        T.Transform(ip,xx);
        double x = xx[0]*ll;
        double y = xx[1]*ll;
        double z = (xx.Size()==3) ? xx[2]*ll : 0.0;

        double r=std::sin(x)*std::cos(y) +
                std::sin(y)*std::cos(z) +
               std::sin(z)*std::cos(x) ;

        return r;

        //if(r>0.0){return 1.0;}
        //return -1.0;
    }

private:
    double ll;
};


int main(int argc, char *argv[])
{
   // Initialize MPI and HYPRE.
   Mpi::Init(argc, argv);
   int myrank = Mpi::WorldRank();
   Hypre::Init();

   // Parse command-line options.
   const char *mesh_file = "../../data/inline-quad.mesh";
   int solver_type = 0;
   int rs_levels = 2;
   int order = 2;
   const char *device_config = "cpu";
   bool visualization = true;


   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");

   args.AddOption(&rs_levels, "-rs", "--refine-serial",
                  "Number of times to refine the mesh uniformly in serial.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree) or -1 for"
                  " isoparametric space.");
   args.AddOption(&device_config, "-d", "--device",
                  "Device configuration string, see Device::Configure().");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");

   args.Parse();
   if (!args.Good())
   {
      if (myrank == 0)
      {
         args.PrintUsage(cout);
      }
      return 1;
   }
   if (myrank == 0) { args.PrintOptions(cout); }

   // Enable hardware devices such as GPUs, and programming models such as CUDA,
   // OCCA, RAJA and OpenMP based on command line options.
   Device device(device_config);
   if (myrank == 0) { device.Print(); }


   // Refine the mesh.
   Mesh mesh(mesh_file, 1, 1);
   const int dim = mesh.Dimension();
   for (int lev = 0; lev < rs_levels; lev++) { mesh.UniformRefinement(); }

   // MPI distribution.
   ParMesh pmesh(MPI_COMM_WORLD, mesh);
   mesh.Clear();

   int mesh_poly_deg=2;
   FiniteElementCollection *fec = new H1_FECollection(mesh_poly_deg, dim);
   ParFiniteElementSpace *pfespace = new ParFiniteElementSpace(&pmesh, fec, dim, Ordering::byNODES);

   pmesh.SetNodalFESpace(pfespace);
   ParGridFunction x(pfespace);
   ParGridFunction dx(pfespace);
   pmesh.SetNodalGridFunction(&x);

   //gradients
   Vector npos(pfespace->GetTrueVSize());
   Vector ngrad(pfespace->GetTrueVSize());
   x.GetTrueDofs(npos);

   GyroidCoeff gyro_co(2*M_PI);
   ParFiniteElementSpace *cfespace = new ParFiniteElementSpace(&pmesh, fec, 1, Ordering::byNODES);
   ParGridFunction ggf(cfespace);
   ggf.ProjectCoefficient(gyro_co);

   ParNonlinearForm* nf=new ParNonlinearForm(pfespace);
   VolPenalIntegrator* igr=new VolPenalIntegrator(&gyro_co,0.5);
   nf->AddDomainIntegrator(igr);

   double obj=nf->GetEnergy(npos);
   std::cout<<"Obj="<<obj<<std::endl;

   nf->Mult(npos,ngrad);
   dx.SetFromTrueDofs(ngrad);

   // ParaView output.
   ParaViewDataCollection dacol("ParaViewDistance", &pmesh);
   dacol.SetLevelsOfDetail(2);
   dacol.SetHighOrderOutput(true);
   dacol.RegisterField("coo",&x);
   dacol.RegisterField("dcoo",&dx);
   dacol.RegisterField("gyro",&ggf);
   dacol.SetTime(1.0);
   dacol.SetCycle(1);
   dacol.Save();

   //FD check
   {
       mfem::Vector prtv;
       mfem::Vector tmpv;

       prtv.SetSize(npos.Size());
       tmpv.SetSize(npos.Size());

       prtv.Randomize();
       double nd=mfem::InnerProduct(pmesh.GetComm(),prtv,prtv);
       double td=mfem::InnerProduct(pmesh.GetComm(),prtv,ngrad);

       td=td/nd;
       double lsc=1.0;
       double lqoi;

       for(int l=0; l<8;l++){
           lsc/=10.0;
           prtv/=10.0;
           add(prtv,npos,tmpv);
           x.SetFromTrueDofs(tmpv);
           pmesh.DeleteGeometricFactors();
           lqoi=nf->GetEnergy(npos);
           double ld=(lqoi-obj)/lsc;
           if(myrank==0){
               std::cout<<" obj="<<obj<<" lvo="<< lqoi<<" dx="<<lsc<<" FD app="<< ld/nd<<" gr="<< td <<" err="<< std::fabs(ld/nd-td) <<std::endl;
           }
       }
   }

   {
       mfem::Vector tmpv(npos);
       tmpv.SetSize(npos.Size());
       double lqoi;
       for(int i=0;i<800;i++){
          tmpv.Add(-0.0005,ngrad);
          x.SetFromTrueDofs(tmpv);
          pmesh.DeleteGeometricFactors();
          lqoi=nf->GetEnergy(npos);
          nf->Mult(npos,ngrad);
          std::cout<<"lqoi="<<lqoi<<" nr="<<ngrad.Norml2()<<std::endl;
       }
   }

   ggf.ProjectCoefficient(gyro_co);


   dacol.SetTime(2.0);
   dacol.SetCycle(2);
   dacol.Save();




   delete nf;
   delete cfespace;
   delete pfespace;
   delete fec;

   return 0;

}
