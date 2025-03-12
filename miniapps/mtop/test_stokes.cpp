#include "mfem.hpp"
#include "stokes_solver.hpp"

#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;


class GyroCoeff:public Coefficient
{
public:
    GyroCoeff():l(0.0),scale(1.0){}

    GyroCoeff(real_t l_,real_t s=1.0, real_t h=0.0)
    {
        l=l_;
        scale=s;
        shift=h;
    }

    virtual real_t Eval(ElementTransformation &T,
                    const IntegrationPoint &ip)
    {
        real_t x[3];
        Vector transip(x, 3);
        T.Transform(ip, transip);

        //real_t r=sin(l*x[0])*cos(l*x[1])+sin(l*x[1])*cos(l*x[2])+sin(l*x[2])*cos(l*x[0])-shift;
        real_t r=cos(l*x[0])*cos(l*x[1])*cos(l*x[2])-shift;

        if(r>=real_t(0.0))
        {
            if(x[0]<0.05){return 0.0;}
            if(x[0]>1.45){return 0.0;}
            if(x[1]<0.05){return 0.0;}
            if(x[1]>0.95){return 0.0;}
            return r*scale;
        }else{
            return 0.0;
        }
    }
private:
    real_t l;
    real_t scale;
    real_t shift;
};

int main(int argc, char *argv[])
{
   // 1. Initialize MPI and HYPRE.
   Mpi::Init();
   int num_procs = Mpi::WorldSize();
   int myid = Mpi::WorldRank();
   Hypre::Init();

   // 2. Parse command-line options.
   const char *mesh_file = "../../data/star.mesh";
   int order = 1;
   bool static_cond = false;
   bool pa = false;
   bool fa = false;
   const char *device_config = "cpu";
   bool visualization = true;
   bool algebraic_ceed = false;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree) or -1 for"
                  " isoparametric space.");
   args.AddOption(&static_cond, "-sc", "--static-condensation", "-no-sc",
                  "--no-static-condensation", "Enable static condensation.");
   args.AddOption(&pa, "-pa", "--partial-assembly", "-no-pa",
                  "--no-partial-assembly", "Enable Partial Assembly.");
   args.AddOption(&fa, "-fa", "--full-assembly", "-no-fa",
                  "--no-full-assembly", "Enable Full Assembly.");
   args.AddOption(&device_config, "-d", "--device",
                  "Device configuration string, see Device::Configure().");
#ifdef MFEM_USE_CEED
   args.AddOption(&algebraic_ceed, "-a", "--algebraic",
                  "-no-a", "--no-algebraic",
                  "Use algebraic Ceed solver");
#endif
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.Parse();
   if (!args.Good())
   {
      if (myid == 0)
      {
         args.PrintUsage(cout);
      }
      return 1;
   }
   if (myid == 0)
   {
      args.PrintOptions(cout);
   }

   // 3. Enable hardware devices such as GPUs, and programming models such as
   //    CUDA, OCCA, RAJA and OpenMP based on command line options.
   Device device(device_config);
   if (myid == 0) { device.Print(); }

   // 4. Read the (serial) mesh from the given mesh file on all processors.  We
   //    can handle triangular, quadrilateral, tetrahedral, hexahedral, surface
   //    and volume meshes with the same code.
   Mesh mesh(mesh_file, 1, 1);
   int dim = mesh.Dimension();

   // 5. Refine the serial mesh on all processors to increase the resolution. In
   //    this example we do 'ref_levels' of uniform refinement. We choose
   //    'ref_levels' to be the largest number that gives a final mesh with no
   //    more than 10,000 elements.
   {
      int ref_levels =
         (int)floor(log(1000./mesh.GetNE())/log(2.)/dim);
      for (int l = 0; l < ref_levels; l++)
      {
         mesh.UniformRefinement();
      }
   }

   // 6. Define a parallel mesh by a partitioning of the serial mesh. Refine
   //    this mesh further in parallel to increase the resolution. Once the
   //    parallel mesh is defined, the serial mesh can be deleted.
   ParMesh pmesh(MPI_COMM_WORLD, mesh);
   mesh.Clear();
   {
      int par_ref_levels = 1;
      for (int l = 0; l < par_ref_levels; l++)
      {
         pmesh.UniformRefinement();
      }
   }
   if(Mpi::WorldRank()==0){
       std::cout<<pmesh.GetNE()<<std::endl;
   }

   GyroCoeff gc(6.0*M_PI,1000.0,0.3);

   StokesOperator* sp=new StokesOperator(&pmesh,2);
   sp->AddVelocityBC(1,4,0.0);
   sp->AddVelocityBC(2,0,1.0);

   mfem::ConstantCoefficient mu(0.01);
   sp->SetViscosoty(mu);
   sp->SetBrinkman(gc);
   sp->Assemble();

   sp->FSolve();

   {
       ParGridFunction brink(sp->GetPressure()); brink.ProjectCoefficient(gc);
       ParaViewDataCollection paraview_dc("stokes", &pmesh);
       paraview_dc.SetPrefixPath("ParaView");
       paraview_dc.SetLevelsOfDetail(order);
       paraview_dc.SetDataFormat(VTKFormat::BINARY);
       paraview_dc.SetHighOrderOutput(true);
       paraview_dc.SetCycle(0);
       paraview_dc.SetTime(0.0);
       paraview_dc.RegisterField("velo",&(sp->GetVelocity()));
       paraview_dc.RegisterField("pres",&(sp->GetPressure()));
       paraview_dc.RegisterField("brink",&(brink));
       paraview_dc.Save();
   }

   delete sp;
   Mpi::Finalize();
   return 0;
}

