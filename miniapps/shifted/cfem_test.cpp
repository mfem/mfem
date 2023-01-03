#include <fstream>
#include <iostream>

#include "mfem.hpp"
#include "marking.hpp"
#include "mtop_solvers.hpp"
#include "mtop_filters.hpp"


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

        if(r>0.0){return 1.0;}
        return -1.0;
    }

private:
    double ll;
};


class CheseCoeff:public Coefficient
{
public:
    CheseCoeff(double cell_size=1.0)
    {
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

        double r=std::cos(x)*std::cos(y)*std::cos(z)-0.1;

        if(r>0.0){return -1.0;}
        return 1.0;

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

   FilterSolver* filter=new FilterSolver(0.1,&pmesh);

   ParFiniteElementSpace* dfes=filter->GetDesignFES();
   ParFiniteElementSpace* ffes=filter->GetFilterFES();

   ParGridFunction desgf(dfes); desgf=0.0;
   ParGridFunction filgf(ffes); filgf=0.0;

   {// project the coefficient and filter
       GyroidCoeff gc(4.0*M_PI);
       //CheseCoeff  gc(4.0*M_PI);
       desgf.ProjectCoefficient(gc);
       Vector tdes(dfes->GetTrueVSize()); tdes=0.0;
       desgf.GetTrueDofs(tdes);
       Vector tfil(ffes->GetTrueVSize()); tfil=0.0;
       filter->Mult(tdes,tfil);
       filgf.SetFromTrueDofs(tfil);
   }

   ElementMarker* elmark=new ElementMarker(pmesh,false,true);
   elmark->SetLevelSetFunction(filgf);

   Array<int> marks;
   elmark->MarkElements(marks);

   for(int i=0;i<pmesh.GetNE();i++){
       pmesh.SetAttribute(i,marks[i]);
   }


   CFElasticitySolver* elsolv=new CFElasticitySolver(&pmesh,2);
   Vector vf(dim); vf=0.0; vf(1)=10.0;
   VectorConstantCoefficient* ff=new VectorConstantCoefficient(vf);
   LinIsoElasticityCoefficient* lec=new LinIsoElasticityCoefficient(1.0,0.3);
   elsolv->SetLinearSolver(1e-8,1e-12,100);
   elsolv->SetNewtonSolver(1e-6,1e-12,3,1);
   elsolv->AddMaterial(lec,ff,nullptr);
   elsolv->AddDispBC(2,4,0.0);
   elsolv->SetLSF(filgf,marks);
   elsolv->FSolve();
   ParGridFunction& u=elsolv->GetDisplacements();


   // ParaView output.
   ParaViewDataCollection dacol("ParaViewDistance", &pmesh);
   dacol.SetLevelsOfDetail(order);
   dacol.SetHighOrderOutput(true);
   dacol.RegisterField("design", &desgf);
   dacol.RegisterField("flter", &filgf);
   dacol.RegisterField("disp",&u);
   dacol.SetTime(1.0);
   dacol.SetCycle(1);
   dacol.Save();


   delete elsolv;
   delete elmark;
   delete filter;

   return 0;
}
