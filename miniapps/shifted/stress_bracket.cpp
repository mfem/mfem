#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include <random>
#include "MMA.hpp"

#include "mtop_solvers.hpp"
#include "mtop_filters.hpp"

class StressBracket2D
{
public:
    StressBracket2D(mfem::ParMesh* mesh_,int order_=1):E(),nu(0.2)
    {
        pmesh=mesh_;
        vorder=order_;
        esolv=new mfem::ElasticitySolver(pmesh,vorder);
        esolv->AddMaterial(new mfem::LinIsoElasticityCoefficient(E,nu));
        esolv->SetNewtonSolver(1e-8,1e-12,1,0);
        esolv->SetLinearSolver(1e-10,1e-12,400);

        dfes=nullptr;
        cobj=new mfem::ComplianceObjective();

    }

    ~StressBracket2D()
    {
        delete cobj;
        delete esolv;
    }

    void SetDesignFES(mfem::ParFiniteElementSpace* fes)
    {
        dfes=fes;
        pdens.SetSpace(dfes);
    }

    void SetDensity(mfem::Vector& vdens_,
                    double eta=0.5, double beta=8.0,double pen=3.0){
        pdens.SetFromTrueDofs(vdens_);

        E.SetDens(&pdens);
        E.SetProjParam(eta,beta);
        E.SetEMaxMin(1e-6,1.0);
        E.SetPenal(pen);

        cobj->SetE(&E);
        cobj->SetDens(pdens.GetTrueVector());
        cobj->SetDesignFES(dfes);

        // obtain the response
        esolv->DelDispBC();
        esolv->AddDispBC(1,4,0.0);
        esolv->AddSurfLoad(3,0.00,-1.00,0.0);
        esolv->FSolve();
    }

    double Compliance()
    {
        return cobj->Eval(esolv->GetDisplacements());
    }

    double Compliance(mfem::Vector& grad)
    {
        cobj->Grad(esolv->GetDisplacements(),grad);
        return cobj->Eval(esolv->GetDisplacements());
    }

    mfem::ParGridFunction& GetDisplacements()
    {
        return esolv->GetDisplacements();
    }


private:
    mfem::ParMesh* pmesh;

    mfem::ElasticitySolver* esolv;
    int vorder;

    mfem::ComplianceObjective* cobj;

    mfem::YoungModulus E;
    double nu;

    mfem::ParFiniteElementSpace* dfes;
    mfem::ParGridFunction pdens;
};

using namespace mfem;
using namespace std;

int main(int argc, char *argv[])
{
   // Initialize MPI.
   int nprocs, myrank;
   MPI_Init(&argc, &argv);
   MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
   MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

   // Parse command-line options.
   const char *mesh_file = "../../data/inline-quad.mesh";
   int solver_type = 0;
   int rs_levels = 0;
   int par_ref_levels = 0;
   int order = 2;
   int cut_int_order = order;
   const char *device_config = "cpu";
   bool visualization = true;
   double stiff_ratio=1e-6;
   const char *petscrc_file = "";

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
   args.AddOption(&stiff_ratio,"-sr", "--stiff_ratio",
                  "Stiffness ratio");
   args.AddOption(&petscrc_file, "-petscopts", "--petscopts",
                     "PetscOptions file to use.");
   args.AddOption(&par_ref_levels,
                     "-rp",
                     "--refine-parallel",
                     "Number of times to refine the mesh uniformly in parallel.");

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

   mfem::MFEMInitializePetsc(NULL,NULL,petscrc_file,NULL);

   // Refine the mesh.
   Mesh mesh(mesh_file, 1, 1);
   const int dim = mesh.SpaceDimension();
   for (int lev = 0; lev < rs_levels; lev++) { mesh.UniformRefinement(); }
   if(myrank==0){
       std::cout<<"Num elements="<<mesh.GetNE()<<std::endl;
   }

   // Define a parallel mesh by a partitioning of the serial mesh. Refine
   // this mesh further in parallel to increase the resolution. Once the
   // parallel mesh is defined, the serial mesh can be deleted.
   mfem::ParMesh pmesh(MPI_COMM_WORLD, mesh);
   mesh.Clear();
   {
       for (int l = 0; l < par_ref_levels; l++)
       {
          pmesh.UniformRefinement();
       }
   }

   if(myrank==0)
   {
       std::cout<<"num el="<<pmesh.GetNE()<<std::endl;
   }

   //allocate the filter
   mfem::FilterSolver* fsolv=new mfem::FilterSolver(0.02,&pmesh);
   fsolv->SetSolver(1e-8,1e-12,100,0);
   fsolv->AddBC(2,0.0);
   fsolv->AddBC(3,1.0);

   mfem::ParGridFunction pgdens(fsolv->GetFilterFES());
   mfem::ParGridFunction oddens(fsolv->GetDesignFES());
   mfem::Vector vdens; vdens.SetSize(fsolv->GetFilterFES()->GetTrueVSize()); vdens=0.0;
   mfem::Vector vtmpv; vtmpv.SetSize(fsolv->GetDesignFES()->GetTrueVSize()); vtmpv=0.5;
   oddens.SetFromTrueDofs(vtmpv);

   fsolv->Mult(vtmpv,vdens);
   pgdens.SetFromTrueDofs(vdens);

   StressBracket2D* brac=new StressBracket2D(&pmesh,1);
   brac->SetDesignFES(fsolv->GetFilterFES());
   brac->SetDensity(vdens,0.5,8.0,1.0);
   brac->Compliance();




   // ParaView output.
   ParaViewDataCollection dacol("ParaView", &pmesh);
   dacol.SetLevelsOfDetail(order);
   dacol.SetHighOrderOutput(true);
   dacol.RegisterField("design", &oddens);
   dacol.RegisterField("flter", &pgdens);
   dacol.RegisterField("displ", &(brac->GetDisplacements()));



   dacol.SetTime(1.0);
   dacol.SetCycle(1);
   dacol.Save();


   delete brac;
   delete fsolv;

   mfem::MFEMFinalizePetsc();
   MPI_Finalize();
   return 0;

}
