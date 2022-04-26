#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include "MMA.hpp"

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

class TorsionForceX:public VectorCoefficient
{
public:
    TorsionForceX(double xmin, double xmax=0.95):VectorCoefficient(3)
    {
        xm=xmin;
        mx=xmax;
    }

    void Eval (Vector &V, ElementTransformation &T, const IntegrationPoint &ip)
    {
        //evaluate the true coordinate of the ip
        Vector xx; xx.SetSize(T.GetDimension());
        T.Transform(ip,xx);

        if(xx[0]<xm){V=0.0; return;}
        if(xx[0]>mx){V=0.0; return;}

        V[0]=0.0;
        V[1]=-xx[2];
        V[2]=xx[1];
    }


private:
    double xm;
    double mx;
};

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
   int par_ref_levels = 0;
   double rel_tol = 1e-7;
   double abs_tol = 1e-15;
   double fradius = 0.05;
   int tot_iter = 100;
   int max_it = 51;
   int print_level = 1;
   bool visualization = false;
   const char *petscrc_file = "";

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
   args.AddOption(&max_it,
                  "-mit",
                  "--max-optimization-iterations",
                  "Maximum iterations for the linear optimizer.");
   args.AddOption(&fradius,
                  "-r",
                  "--radius",
                  "Filter radius");
   args.AddOption(&petscrc_file, "-petscopts", "--petscopts",
                     "PetscOptions file to use.");
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
   mfem::MFEMInitializePetsc(NULL,NULL,petscrc_file,NULL);

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
         (int)floor(log(10000./mesh.GetNE())/log(2.)/dim);

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
   {
       for (int l = 0; l < par_ref_levels; l++)
       {
          pmesh.UniformRefinement();
       }
   }

   mfem::FilterSolver* fsolv=new mfem::FilterSolver(0.02,&pmesh);
   fsolv->SetSolver(1e-8,1e-12,100,0);
   fsolv->AddBC(1,1.0);
   mfem::ParGridFunction pgdens(fsolv->GetFilterFES());
   mfem::Vector vdens; vdens.SetSize(fsolv->GetFilterFES()->GetTrueVSize());
   mfem::Vector vtmpv; vtmpv.SetSize(fsolv->GetFilterFES()->GetTrueVSize());
   {
       mfem::FunctionCoefficient fco(InitialDens);
       pgdens.ProjectCoefficient(fco);
       pgdens=0.5;
   }
   pgdens.GetTrueDofs(vtmpv);
   fsolv->Mult(vtmpv,vdens);
   pgdens.SetFromTrueDofs(vdens);

   /*
   mfem::VolForce* volforce;
   if(dim==2){//2D force
       volforce=new mfem::VolForce(0.05,2.90,0.5,0.0,-1.0);}
   else{//3D force - the resolution should be good enough to resolve the radius
       volforce=new mfem::VolForce(0.05,0.5,0.5,2.90,0.0,1.0,0.0);
   }*/

   mfem::TorsionForceX* volforce;
   volforce= new mfem::TorsionForceX(0.85,0.90);

   mfem::YoungModulus* E=new mfem::YoungModulus();
   //mfem::YoungModulusSIMP* E=new mfem::YoungModulusSIMP();
   E->SetDens(&pgdens);
   E->SetProjParam(0.9,8.0);//threshold 0.7
   E->SetEMaxMin(1e-6,1.0);
   E->SetPenal(1.0);

   mfem::ElasticitySolver* esolv=new mfem::ElasticitySolver(&pmesh,2);
   esolv->AddDispBC(1,4,0.0);
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
   cobj->SetVolForce(*volforce);

   mfem::PVolumeQoI* vobj=new mfem::PVolumeQoI(fsolv->GetFilterFES());
   //mfem::VolumeQoI* vobj=new mfem::VolumeQoI(fsolv->GetFilterFES());
   vobj->SetProjection(0.1,8.0);//threshold 0.3
   //compute the total volume
   double tot_vol;
   {
       vdens=1.0;
       tot_vol=vobj->Eval(vdens);
       pgdens.GetTrueDofs(vdens);
   }
   double max_vol=0.5*tot_vol;

   //intermediate volume
   mfem::VolumeQoI* ivobj=new mfem::VolumeQoI(fsolv->GetFilterFES());
   ivobj->SetProjection(0.5,32);

   //gradients
   mfem::Vector ograd(fsolv->GetFilterFES()->GetTrueVSize()); ograd=0.0; //of the objective
   mfem::Vector vgrad(fsolv->GetFilterFES()->GetTrueVSize()); vgrad=0.0; //of the volume contr.

   mfem::NativeMMA* mma;
   {
       double a=0.0;
       double c=1000.0;
       double d=0.0;
       mma=new mfem::NativeMMA(MPI_COMM_WORLD,1, ograd,&a,&c,&d);
   }

   //the input design field and the filtered one have the same dimensionality
   mfem::Vector ogrado(fsolv->GetFilterFES()->GetTrueVSize()); ogrado=0.0;
   mfem::Vector vgrado(fsolv->GetFilterFES()->GetTrueVSize()); vgrado=0.0;

   mfem::Vector xxmax(fsolv->GetFilterFES()->GetTrueVSize()); xxmax=1.0;
   mfem::Vector xxmin(fsolv->GetFilterFES()->GetTrueVSize()); xxmin=0.0;

   double max_ch=0.1; //max design change
   double lam, lam_max, lam_min; //Lagrange multiplier

   double cpl; //compliance
   double vol; //volume
   double ivol; //intermediate volume
   double Be;
   double xx;

   mfem::Vector ndesign; ndesign.SetSize(fsolv->GetFilterFES()->GetTrueVSize());
   mfem::Vector fvector; fvector.SetSize(fsolv->GetFilterFES()->GetTrueVSize());

   {
       mfem::ParGridFunction& disp=esolv->GetDisplacements();
       mfem::ParGridFunction pdesign(fsolv->GetFilterFES());
       pdesign.ProjectCoefficient(*E);

       mfem::ParaViewDataCollection paraview_dc("Elasticity", &pmesh);
       paraview_dc.SetPrefixPath("ParaView");
       paraview_dc.SetLevelsOfDetail(order);
       paraview_dc.SetDataFormat(mfem::VTKFormat::BINARY);
       paraview_dc.SetHighOrderOutput(true);
       paraview_dc.SetCycle(0);
       paraview_dc.SetTime(0.0);

       paraview_dc.RegisterField("disp",&disp);
       paraview_dc.RegisterField("design",&pgdens);
       paraview_dc.RegisterField("pdesign",&pdesign);

       for(int i=1;i<max_it;i++){
           esolv->FSolve();

           //compute the objective and the vol constraint
           cpl=cobj->Eval();
           vol=vobj->Eval(vdens);
           ivol=ivobj->Eval(vdens);
           if(myrank==0){
               std::cout<<"it: "<<i<<" obj="<<cpl<<" vol="<<vol<<" ivol="<<ivol<<std::endl;
           }
           //compute the gradients
           cobj->Grad(ograd);
           ivobj->Grad(vdens,vgrad);
           //compute the original gradients
           fsolv->MultTranspose(ograd,ogrado);
           fsolv->MultTranspose(vgrad,vgrado);

           //OC update
           /*
           lam_min=0.0; lam_max=1e9;
           while( (lam_max-lam_min)/(lam_max+lam_min) > 1e-3 ){
               lam=0.5*(lam_max+lam_min);
               for(int j=0;j<vtmpv.Size();j++)
               {
                   Be=-ogrado[j]/(lam*vgrado[j]);
                   if(Be<0.0){Be=-1.0*Be;}//safe guard
                   xx=vtmpv[i]*sqrt(Be);
                   xx=std::min(vtmpv[j]+max_ch,xx);
                   if(xx>1.0){xx=1.0;}//project to 1.0
                   xx=std::max(vtmpv[j]-max_ch,xx);
                   if(xx<0.0){xx=0.0;}//project to 0.0
                   ndesign[j]=xx;
               }
               fsolv->Mult(ndesign,fvector);
               if(vobj->Eval(fvector) > max_vol){ lam_min=lam;}
               else{lam_max = lam;}
           }
           vtmpv=ndesign;
           */

           /*
           {
               //set xxmin and xxmax
               xxmin=vtmpv; xxmin-=max_ch;
               xxmax=vtmpv; xxmax+=max_ch;
               for(int li=0;li<xxmin.Size();li++){
                   if(xxmin[li]<0.0){xxmin[li]=0.0;}
                   if(xxmax[li]>1.0){xxmax[li]=1.0;}
               }
           }
           */
           double con=ivol-max_vol*0.8;
           mma->Update(vtmpv,ogrado,&con,&vgrado,xxmin,xxmax);
           /*
           {
               //project vtmpv
               for(int li=0;li<vtmpv.Size();li++){
                   if(vtmpv[li]>=1.0){vtmpv[li]=1.0;}
                   else if(vtmpv[li]<=0.0){vtmpv[li]=0.0;}
               }
           }
           */

           fsolv->Mult(vtmpv,vdens);
           pgdens.SetFromTrueDofs(vdens);

           //save the design and the solution
           if((i%3)==0)
           {
               pdesign.ProjectCoefficient(*E);
               paraview_dc.SetCycle(i);
               paraview_dc.SetTime(i*1.0);
               paraview_dc.Save();
           }

       }//end max_it

       for(int i=max_it;i<2*max_it;i++){
           esolv->FSolve();

           //compute the objective and the vol constraint
           cpl=cobj->Eval();
           vol=vobj->Eval(vdens);
           ivol=ivobj->Eval(vdens);
           if(myrank==0){
               std::cout<<"it: "<<i<<" obj="<<cpl<<" vol="<<vol<<" ivol="<<ivol<<std::endl;
           }
           //compute the gradients
           cobj->Grad(ograd);
           vobj->Grad(vdens,vgrad);
           //compute the original gradients
           fsolv->MultTranspose(ograd,ogrado);
           fsolv->MultTranspose(vgrad,vgrado);

           double con=vol-max_vol;
           mma->Update(vtmpv,ogrado,&con,&vgrado,xxmin,xxmax);

           fsolv->Mult(vtmpv,vdens);
           pgdens.SetFromTrueDofs(vdens);

           //save the design and the solution
           if((i%3)==0)
           {
               pdesign.ProjectCoefficient(*E);
               paraview_dc.SetCycle(i);
               paraview_dc.SetTime(i*1.0);
               paraview_dc.Save();
           }

       }//end max_it


   }

   delete mma;
   delete vobj;
   delete ivobj;
   delete cobj;
   delete esolv;
   delete E;
   delete volforce;
   delete fsolv;
   mfem::MFEMFinalizePetsc();
   MPI_Finalize();
   return 0;
}
