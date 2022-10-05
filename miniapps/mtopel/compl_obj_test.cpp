#include "mfem.hpp"
#include <fstream>
#include <iostream>

#include "mtop_solvers.hpp"

double InitialDens(const mfem::Vector &x)
{
    double rez=std::sin(2*M_PI*x[0])*std::sin(2*M_PI*x[1]);
    if(x.Size()==3)
    {
        rez=rez*std::sin(2*M_PI*x[2]);
    }

    return rez=0.5+0.3*(0.5+0.5*rez);
}

namespace mfem{

class VolForce:public VectorCoefficient
{
public:
    VolForce(double r_,double x,double y, double fx, double fy):VectorCoefficient(2)
    {
        r=r_;
        cntr.SetSize(2);
        frce.SetSize(2);
        cntr[0]=x;
        cntr[1]=y;
        frce[0]=fx;
        frce[1]=fy;
    }

    VolForce(double r_,double x, double y, double z, double fx, double fy, double fz):VectorCoefficient(3)
    {
        r=r_;
        cntr.SetSize(3);
        frce.SetSize(3);
        cntr[0]=x;
        cntr[1]=y;
        cntr[2]=z;
        frce[0]=fx;
        frce[1]=fy;
        frce[2]=fz;
    }

    void Eval (Vector &V, ElementTransformation &T, const IntegrationPoint &ip)
    {
        //evaluate the true coordinate of the ip
        Vector xx; xx.SetSize(T.GetDimension());
        T.Transform(ip,xx);

        for(int i=0;i<xx.Size();i++){
            xx[i]=xx[i]-cntr[i];
        }
        double cr=std::sqrt(xx*xx);

        if(cr<=r)
        {
            for(int i=0;i<T.GetDimension();i++)
            {
                V[i]=frce[i];
            }
        }else{
            V=0.0;
        }
    }



private:
    double r;
    mfem::Vector cntr;
    mfem::Vector frce;
};

}



int main(int argc, char *argv[])
{
   // Initialize MPI.
   int nprocs, myrank;
   MPI_Init(&argc, &argv);
   MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
   MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

   // Parse command-line options.
   const char *mesh_file = "../../data/star.mesh";
   int order = 1;
   bool static_cond = false;
   int ser_ref_levels = 1;
   int par_ref_levels = 1;
   double rel_tol = 1e-7;
   double abs_tol = 1e-15;
   double fradius = 0.05;
   int tot_iter = 100;
   int print_level = 1;
   bool visualization = false;

   mfem::OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&ser_ref_levels,
                  "-rs",
                  "--refine-serial",
                  "Number of times to refine the mesh uniformly in serial.");
   args.AddOption(&par_ref_levels,
                  "-rp",
                  "--refine-parallel",
                  "Number of times to refine the mesh uniformly in parallel.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree) or -1 for"
                  " isoparametric space.");
   args.AddOption(&visualization,
                  "-vis",
                  "--visualization",
                  "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&static_cond, "-sc", "--static-condensation", "-no-sc",
                  "--no-static-condensation", "Enable static condensation.");
   args.AddOption(&rel_tol,
                  "-rel",
                  "--relative-tolerance",
                  "Relative tolerance for the Newton solve.");
   args.AddOption(&abs_tol,
                  "-abs",
                  "--absolute-tolerance",
                  "Absolute tolerance for the Newton solve.");
   args.AddOption(&tot_iter,
                  "-it",
                  "--linear-iterations",
                  "Maximum iterations for the linear solve.");
   args.AddOption(&fradius,
                  "-r",
                  "--radius",
                  "Filter radius");
   args.Parse();
   if (!args.Good())
   {
      if (myrank == 0)
      {
         args.PrintUsage(std::cout);
      }
      MPI_Finalize();
      return 1;
   }

   if (myrank == 0)
   {
      args.PrintOptions(std::cout);
   }

   // Read the (serial) mesh from the given mesh file on all processors.  We
   // can handle triangular, quadrilateral, tetrahedral, hexahedral, surface
   // and volume meshes with the same code.
   mfem::Mesh mesh(mesh_file, 1, 1);
   int dim = mesh.Dimension();

   // Refine the serial mesh on all processors to increase the resolution. In
   // this example we do 'ref_levels' of uniform refinement. We choose
   // 'ref_levels' to be the largest number that gives a final mesh with no
   // more than 10,000 elements.
   {
      int ref_levels =
         (int)floor(log(1000./mesh.GetNE())/log(2.)/dim);
      for (int l = 0; l < ref_levels; l++)
      {
         mesh.UniformRefinement();
      }
   }

   // Define a parallel mesh by a partitioning of the serial mesh. Refine
   // this mesh further in parallel to increase the resolution. Once the
   // parallel mesh is defined, the serial mesh can be deleted.
   mfem::ParMesh pmesh(MPI_COMM_WORLD, mesh);
   mesh.Clear();

   mfem::FilterSolver* fsolv=new mfem::FilterSolver(0.1,&pmesh);
   fsolv->SetSolver(1e-8,1e-12,100,0);
   mfem::ParGridFunction pgdens(fsolv->GetFilterFES());
   mfem::Vector vdens; vdens.SetSize(fsolv->GetFilterFES()->GetTrueVSize());
   mfem::Vector vtmpv; vtmpv.SetSize(fsolv->GetFilterFES()->GetTrueVSize());
   {
       mfem::FunctionCoefficient fco(InitialDens);
       pgdens.ProjectCoefficient(fco);
   }
   pgdens.GetTrueDofs(vtmpv);
   fsolv->Mult(vtmpv,vdens);
   pgdens.SetFromTrueDofs(vdens);

   mfem::VolForce* volforce;
   if(dim==2){//2D force
       volforce=new mfem::VolForce(0.05,2.90,0.5,0.0,-1.0);}
   else{//3D force - the resolution should be good enough to resolve the radius
       volforce=new mfem::VolForce(0.05,0.5,0.5,2.90,0.0,1.0,0.0);
   }

   mfem::YoungModulus* E=new mfem::YoungModulus();
   E->SetDens(&pgdens);
   E->SetProjParam(0.7,8.0);
   E->SetEMaxMin(1e-6,1.0);

   mfem::ElasticitySolver* esolv=new mfem::ElasticitySolver(&pmesh,1);
   esolv->AddDispBC(2,4,0.0);
   esolv->SetVolForce(*volforce);
   esolv->AddMaterial(new mfem::LinIsoElasticityCoefficient(*E,0.2));
   esolv->FSolve();
   esolv->SetNewtonSolver(1e-7,1e-12,20,0);

   mfem::ComplianceObjective* cobj=new mfem::ComplianceObjective();
   cobj->SetElastSolver(esolv);
   cobj->SetFilter(fsolv);
   cobj->SetE(E);
   cobj->SetPoissonRatio(0.2);
   cobj->SetDens(vdens);


   double vobj=cobj->Eval();
   if(myrank==0)
   {
       std::cout<<"Objective="<<vobj<<std::endl;
   }

   mfem::Vector grad; grad.SetSize(fsolv->GetFilterFES()->GetTrueVSize());
   cobj->Grad(grad);

   {
       mfem::ParGridFunction gfemod(fsolv->GetFilterFES());
       gfemod.ProjectCoefficient(*E);

       mfem::ParGridFunction gfgrad(fsolv->GetFilterFES());
       gfgrad.SetFromTrueDofs(grad);

       mfem::ParGridFunction& disp=esolv->GetDisplacements();

       mfem::ParaViewDataCollection paraview_dc("Stokes", &pmesh);
       paraview_dc.SetPrefixPath("Filter");
       paraview_dc.SetLevelsOfDetail(order);
       paraview_dc.SetDataFormat(mfem::VTKFormat::BINARY);
       paraview_dc.SetHighOrderOutput(true);
       paraview_dc.SetCycle(0);
       paraview_dc.SetTime(0.0);
       paraview_dc.RegisterField("pgdens",&pgdens);
       paraview_dc.RegisterField("emod",&gfemod);
       paraview_dc.RegisterField("disp",&disp);
       paraview_dc.RegisterField("grad",&gfgrad);
       paraview_dc.Save();

   }

   //FD check
   {
       mfem::Vector ograd(fsolv->GetFilterFES()->GetTrueVSize());
       mfem::Vector tgrad(fsolv->GetFilterFES()->GetTrueVSize());

       //vtmpv is our original density
       fsolv->Mult(vtmpv,vdens);
       pgdens.SetFromTrueDofs(vdens);
       E->SetDens(&pgdens);
       esolv->FSolve();

       double val=cobj->Eval();
       cobj->Grad(ograd);
       fsolv->MultTranspose(ograd,tgrad);//now we have the true gradients

       mfem::Vector prtv;
       mfem::Vector tmpv;

       prtv.SetSize(tgrad.Size());
       tmpv.SetSize(tgrad.Size());

       prtv.Randomize();

       double nd=mfem::InnerProduct(pmesh.GetComm(),prtv,prtv);
       double td=mfem::InnerProduct(pmesh.GetComm(),prtv,tgrad);

       td=td/nd;
       double lsc=1.0;
       double lqoi;

       for(int l=0;l<10;l++){
           lsc/=10.0;
           prtv/=10.0;
           add(prtv,vtmpv,tmpv);
           fsolv->Mult(tmpv,vdens);//filter the design
           pgdens.SetFromTrueDofs(vdens);
           E->SetDens(&pgdens);
           esolv->FSolve();
           cobj->SetDens(vdens);
           lqoi=cobj->Eval();
           double ld=(lqoi-val)/lsc;
           if(myrank==0){
               std::cout << "dx=" << lsc <<" FD approximation=" << ld/nd
                         << " adjoint gradient=" << td
                         << " err=" << std::fabs(ld/nd-td) << std::endl;
           }
       }

   }


   delete cobj;
   delete esolv;
   delete E;
   delete volforce;
   delete fsolv;
   MPI_Finalize();
   return 0;
}
