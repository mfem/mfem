#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include <random>
#include "MMA.hpp"

#include "mtop_solvers.hpp"



//return threshold level at the integration point
class ShotCoefficient:public mfem::Coefficient
{
public:
   ShotCoefficient(mfem::ParMesh* mesh_, double r_, double eta_=0.5):udist(0,1)
   {
      r=r_;
      eta=eta_;
      mesh=mesh_;

      dim=mesh->SpaceDimension();
      xmin.SetSize(dim);
      xmax.SetSize(dim);
      mesh->GetBoundingBox(xmin,xmax);
      ctr.SetSize(dim);

      generator.seed(100001);

   }

   virtual
   double Eval(mfem::ElementTransformation &T, const mfem::IntegrationPoint &ip)
   {

      mfem::Vector xx(dim); xx=0.0;
      T.Transform(ip,xx);
      if (xmax(1)-r<xx(1)) {return eta;}

      xx.Add(-1.0,ctr);
      double d=xx.Norml2();
      if (d<=r) { return 1.0 ;}
      return eta;
   }

   void Generate()
   {

      mfem::Vector tmpv(dim);
      MPI_Comm comm=mesh->GetComm();
      int myrank=mesh->GetMyRank();
      if (myrank==0)
      {
         //generate the sample
         for (int i=0; i<xmin.Size(); i++)
         {
            tmpv(i)=xmin(i)+udist(generator)*(xmax(i)-xmin(i)) ;
         }
      }

      //communicate the center from process zero to all others
      MPI_Scatter(tmpv.GetData(), tmpv.Size(),MPI_DOUBLE,
                  ctr.GetData(), ctr.Size(), MPI_DOUBLE, 0, comm);

   }

private:
   mfem::Vector ctr; //center
   double r;
   double eta;
   mfem::ParMesh* mesh;
   int dim;
   mfem::Vector xmin;
   mfem::Vector xmax;

   std::default_random_engine generator;
   std::uniform_real_distribution<double> udist;


};

class CoeffHoles:public mfem::Coefficient
{
public:
   CoeffHoles(double pr=0.2)
   {
      period=pr;
   }

   virtual
   double Eval(mfem::ElementTransformation &T, const mfem::IntegrationPoint &ip)
   {

      double x[3];
      mfem::Vector transip(x, T.GetSpaceDim());
      T.Transform(ip, transip);

      int nx=x[0]/period;
      int ny=x[1]/period;

      x[0]=x[0]-double(nx)*period-0.5*period;
      x[1]=x[1]-double(ny)*period-0.5*period;

      double r=sqrt(x[0]*x[0]+x[1]*x[1]);
      if (r<(0.45*period)) {return 0.2;}
      return 0.8;
   }


private:
   double period;
};

class Bridge2D
{
public:
   Bridge2D(mfem::ParMesh* pmesh_, int vorder=1):E(),nu(0.2)
   {
      esolv=new mfem::ElasticitySolver(pmesh_,vorder);
      esolv->AddMaterial(new mfem::LinIsoElasticityCoefficient(E,nu));
      esolv->SetNewtonSolver(1e-8,1e-12,1,0);
      esolv->SetLinearSolver(1e-10,1e-12,400);

      pmesh=pmesh_;
      dfes=nullptr;
      cobj=new mfem::ComplianceObjective();


   }

   void SetDesignFES(mfem::ParFiniteElementSpace* fes)
   {
      dfes=fes;
      pdens.SetSpace(dfes);
      vdens.SetSize(dfes->GetTrueVSize());
   }

   ~Bridge2D()
   {
      delete cobj;
      delete esolv;

   }

   void Solve()
   {
      //solve the problem for the base loads
      bsolx.resize(1);
      bsoly.resize(1);
      bsolz.resize(1);

      Solve(bsolx,0.7);
      Solve(bsoly,0.75);
      Solve(bsolz,0.8);

   }

   void Solve(std::vector<mfem::ParGridFunction>& bsol,double eta)
   {

      E.SetProjParam(eta,8.0);
      //set all bc
      esolv->DelDispBC();
      esolv->AddDispBC(1,4,0.0);
      esolv->AddDispBC(2,4,0.0);
      esolv->AddDispBC(3,0,0.0);
      esolv->AddSurfLoad(4,0.00,1.00,0.0);
      esolv->FSolve();
      esolv->GetSol(bsol[0]);

   }


   void SetDensity(mfem::Vector& vdens_,
                   double eta=0.5, double beta=8.0,double pen=3.0)
   {

      vdens=vdens_;
      pdens.SetFromTrueDofs(vdens);

      E.SetDens(&pdens);
      E.SetProjParam(eta,beta);
      E.SetEMaxMin(1e-6,1.0);
      E.SetPenal(pen);

      cobj->SetE(&E);
      cobj->SetDens(vdens);
      cobj->SetDesignFES(dfes);

   }

   /// Evaluates the compliance
   double Compliance(int i, double fx, double fy, double fz)
   {

      return cobj->Eval(GetSol(i,fx,fy,fz));
   }

   void GetComplianceGrad(int i, double fx, double fy, double fz,
                          mfem::Vector& grad)
   {
      if (dfes==nullptr)
      {
         mfem::mfem_error("AlcoaBracket dfes is not defined!");
      }
      cobj->Grad(GetSol(i,fx,fy,fz),grad);
   }

   double SMeanSTD(double alpha=0.0,double r=0.15)
   {
      double mean=0.0;
      double var=0.0;
      double vv;
      int num_samples=100;
      ShotCoefficient eta(pmesh,r, 0.7);
      for (int i=0; i<num_samples; i++)
      {
         eta.Generate();
         E.SetProjParam(eta,8.0);
         //E.SetProjParam(0.7,8.0);

         //solve the problem
         esolv->DelDispBC();
         esolv->AddDispBC(1,4,0.0);
         esolv->AddDispBC(2,4,0.0);
         esolv->AddDispBC(3,0,0.0);
         esolv->AddSurfLoad(4,0.00,1.00,0.0);
         esolv->FSolve();

         vv=cobj->Eval(esolv->GetDisplacements());

         mean=mean+vv;
         var=var+vv*vv;
      }

      double tprob=num_samples;
      mean=mean/tprob;
      var=var/tprob;
      var=var-mean*mean;

      double rez=alpha*mean+(1-alpha)*sqrt(var);
      return rez;
   }

   double SMeanSTD(mfem::Vector& grad, double alpha=0.0, double r=0.15)
   {
      grad=0.0;
      mfem::Vector cgrad(grad.Size()); cgrad=0.0;
      mfem::Vector lgrad(grad.Size()); lgrad=0.0;

      double mean=0.0;
      double var=0.0;
      double vv;
      int num_samples=100;
      ShotCoefficient eta(pmesh,r, 0.7);
      for (int i=0; i<num_samples; i++)
      {
         eta.Generate();

         E.SetProjParam(eta,8.0);
         //E.SetProjParam(0.7,8.0);

         //solve the problem
         esolv->DelDispBC();

         esolv->AddDispBC(1,4,0.0);
         esolv->AddDispBC(2,4,0.0);
         esolv->AddDispBC(3,0,0.0);
         esolv->AddSurfLoad(4,0.00,1.00,0.0);

         esolv->FSolve();

         esolv->GetSol(sol);

         vv=cobj->Eval(sol);


         mean=mean+vv;
         var=var+vv*vv;

         //get gradient
         cobj->Grad(sol,cgrad);
         grad.Add(1.0,cgrad);
         lgrad.Add(vv,cgrad);

      }

      double tprob=num_samples;
      mean=mean/tprob;
      var=var/tprob;
      var=var-mean*mean;

      grad/=tprob;
      lgrad/=tprob;

      cgrad=grad;

      grad*=alpha;
      grad.Add((1.0-alpha)/sqrt(var),lgrad);
      grad.Add(-(1.0-alpha)*mean/sqrt(var),cgrad);

      double rez=alpha*mean+(1-alpha)*sqrt(var);
      return rez;
   }

   double MeanSTD(double alpha=1.0)
   {

      double prob[1]= {1.0};

      double mean=0.0;
      double var=0.0;
      double tprob=3.0;

      double rez=0.0;
      double vv;
      for (int i=0; i<1; i++)
      {
         vv=Compliance(i,1.0,0.0,0.0); mean=mean+prob[i]*vv; var=var+prob[i]*vv*vv;
         vv=Compliance(i,0.0,1.0,0.0); mean=mean+prob[i]*vv; var=var+prob[i]*vv*vv;
         vv=Compliance(i,0.0,0.0,1.0); mean=mean+prob[i]*vv; var=var+prob[i]*vv*vv;
      }

      mean=mean/tprob;
      var=var/tprob;
      var=var-mean*mean;

      rez=alpha*mean+(1-alpha)*sqrt(var);
      return rez;
   }

   void MeanSTD(mfem::Vector& grad, double alpha=1.0)
   {
      grad=0.0;
      mfem::Vector cgrad(grad.Size()); cgrad=0.0;
      mfem::Vector lgrad(grad.Size()); lgrad=0.0;

      double prob[1]= {1.0};

      double tprob=3.0;

      double mean=0.0;
      double var=0.0;

      double vv;
      for (int i=0; i<1; i++)
      {
         vv=Compliance(i,1.0,0.0,0.0); mean=mean+prob[i]*vv; var=var+prob[i]*vv*vv;
         GetComplianceGrad(i,1.0,0.0,0.0,cgrad);
         grad.Add(prob[i],cgrad);
         lgrad.Add(prob[i]*vv,cgrad);


         vv=Compliance(i,0.0,1.0,0.0); mean=mean+prob[i]*vv; var=var+prob[i]*vv*vv;
         GetComplianceGrad(i,0.0,1.0,0.0,cgrad);
         grad.Add(prob[i],cgrad);
         lgrad.Add(prob[i]*vv,cgrad);

         vv=Compliance(i,0.0,0.0,1.0); mean=mean+prob[i]*vv; var=var+prob[i]*vv*vv;
         GetComplianceGrad(i,0.0,0.0,1.0,cgrad);
         grad.Add(prob[i],cgrad);
         lgrad.Add(prob[i]*vv,cgrad);
      }
      grad/=tprob;
      lgrad/=tprob;

      mean=mean/tprob;
      var=var/tprob;
      var=var-mean*mean;

      cgrad=grad;

      grad*=alpha;
      grad.Add((1.0-alpha)/sqrt(var),lgrad);
      grad.Add(-(1.0-alpha)*mean/sqrt(var),cgrad);

   }

   mfem::ParGridFunction& GetSol(int i, double fx, double fy, double fz)
   {

      sol.SetSpace((esolv->GetDisplacements()).ParFESpace()); sol=0.0;
      sol.Add(fx,bsolx[i]);
      sol.Add(fy,bsoly[i]);
      sol.Add(fz,bsolz[i]);
      return sol;
   }

   void GetSol(int i, double fx, double fy, double fz, mfem::ParGridFunction& msol)
   {
      msol.SetSpace((esolv->GetDisplacements()).ParFESpace()); msol=0.0;
      msol.Add(fx,bsolx[i]);
      msol.Add(fy,bsoly[i]);
      msol.Add(fz,bsolz[i]);
   }

private:
   mfem::YoungModulus E;
   double nu;

   mfem::ParFiniteElementSpace* dfes; //design FES
   mfem::ParGridFunction pdens;
   mfem::Vector vdens;

   mfem::ElasticitySolver* esolv;
   mfem::ComplianceObjective* cobj;

   //base solution vectors x,y,z direction loads
   std::vector<mfem::ParGridFunction> bsolx;
   std::vector<mfem::ParGridFunction> bsoly;
   std::vector<mfem::ParGridFunction> bsolz;


   mfem::ParGridFunction sol;
   mfem::ParMesh* pmesh;

};



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
   int ser_ref_levels = 0;
   int par_ref_levels = 0;
   double rel_tol = 1e-7;
   double abs_tol = 1e-15;
   double fradius = 0.05;
   int tot_iter = 100;
   int max_it = 51;
   int print_level = 1;
   bool visualization = false;
   const char *petscrc_file = "";
   int restart=0;

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
   args.AddOption(&restart,
                  "-rstr",
                  "--restart",
                  "Restart the optimization from previous design.");
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
   {
      mfem::Vector vert;
      mesh.GetVertices(vert);
      //vert*=0.01;
      mesh.SetVertices(vert);
      mfem::Vector xmin(dim), xmax(dim);
      mesh.GetBoundingBox(xmin,xmax);
      if (myrank==0)
      {
         std::cout<<"Xmin:"; xmin.Print(std::cout);
         std::cout<<"Xmax:"; xmax.Print(std::cout);
      }
   }

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

   /*
   if(restart>0)
   {
       std::cout<<"Read the mesh!"<<std::endl;
       //read the mesh and the design
       std::ostringstream oss;
       oss << std::setw(10) << std::setfill('0') << myrank;
       std::string mname="pmesh_"+oss.str()+".msh";

       std::ifstream in;
       in.open(mname.c_str(),std::ios::in);
       pmesh.Load(in,0,0);
       in.close();
   }*/



   if (myrank==0)
   {
      std::cout<<"num el="<<pmesh.GetNE()<<std::endl;
   }

   //allocate the filter
   mfem::FilterSolver* fsolv=new mfem::FilterSolver(0.04,&pmesh);
   fsolv->SetSolver(1e-8,1e-12,100,0);
   fsolv->AddBC(1,1.0);
   fsolv->AddBC(2,1.0);
   fsolv->AddBC(4,1.0);
   fsolv->AddBC(5,0.0);

   mfem::ParGridFunction pgdens(fsolv->GetFilterFES());
   mfem::ParGridFunction oddens(fsolv->GetDesignFES());
   //mfem::ParGridFunction spdegf(fsolv->GetFilterFES());
   mfem::Vector vdens; vdens.SetSize(fsolv->GetFilterFES()->GetTrueVSize());
   vdens=0.0;
   mfem::Vector vtmpv; vtmpv.SetSize(fsolv->GetDesignFES()->GetTrueVSize());
   vtmpv=0.5;

   fsolv->Mult(vtmpv,vdens);
   pgdens.SetFromTrueDofs(vdens);


   Bridge2D* alco=new Bridge2D(&pmesh,1);
   alco->SetDesignFES(pgdens.ParFESpace());
   alco->SetDensity(vdens);
   //alco->Solve();
   //mfem::ParGridFunction disp;
   //alco->GetSol(4,1,1,1,disp);

   //check gradients
   /*
   {
       mfem::Vector prtv;
       mfem::Vector tmpv;
       mfem::Vector tgrad;
       mfem::Vector fgrad;
       prtv.SetSize(vtmpv.Size());
       tmpv.SetSize(vtmpv.Size());
       tgrad.SetSize(vtmpv.Size());
       fgrad.SetSize(vdens.Size()); fgrad=0.0;
       double val=alco->SMeanSTD();
       alco->SMeanSTD(fgrad);
       fsolv->MultTranspose(fgrad,tgrad);

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
           fsolv->Mult(tmpv,vdens);
           alco->SetDensity(vdens);
           lqoi=alco->SMeanSTD();
           double ld=(lqoi-val)/lsc;
           if(myrank==0){
               std::cout << "dx=" << lsc <<" FD approximation=" << ld/nd
                         << " adjoint gradient=" << td
                         << " err=" << std::fabs(ld/nd-td) << std::endl;
           }
       }
   }*/

   mfem::PVolumeQoI* vobj=new mfem::PVolumeQoI(fsolv->GetFilterFES());
   //mfem::VolumeQoI* vobj=new mfem::VolumeQoI(fsolv->GetFilterFES());
   vobj->SetProjection(0.2,8.0);//threshold 0.2

   //compute the total volume
   double tot_vol;
   {
      vdens=1.0;
      tot_vol=vobj->Eval(vdens);
   }
   double max_vol=0.35*tot_vol;
   if (myrank==0) { std::cout<<"tot vol="<<tot_vol<<std::endl;}

   //intermediate volume
   mfem::VolumeQoI* ivobj=new mfem::VolumeQoI(fsolv->GetFilterFES());
   ivobj->SetProjection(0.5,32);

   //gradients with respect to the filtered field
   mfem::Vector ograd(fsolv->GetFilterFES()->GetTrueVSize());
   ograd=0.0; //of the objective
   mfem::Vector vgrad(fsolv->GetFilterFES()->GetTrueVSize());
   vgrad=0.0; //of the volume contr.

   //the input design field and the filtered one might not have the same dimensionality
   mfem::Vector ogrado(fsolv->GetDesignFES()->GetTrueVSize()); ogrado=0.0;
   mfem::Vector vgrado(fsolv->GetDesignFES()->GetTrueVSize()); vgrado=0.0;

   mfem::Vector xxmax(fsolv->GetDesignFES()->GetTrueVSize()); xxmax=1.0;
   mfem::Vector xxmin(fsolv->GetDesignFES()->GetTrueVSize()); xxmin=0.0;

   mfem::NativeMMA* mma;
   {
      double a=0.0;
      double c=1000.0;
      double d=0.0;
      mma=new mfem::NativeMMA(MPI_COMM_WORLD,1, ogrado,&a,&c,&d);
   }

   double max_ch=0.1; //max design change

   double cpl; //compliance
   double vol; //volume
   double ivol; //intermediate volume
   double dcpl;


   mfem::ParGridFunction solx;
   mfem::ParGridFunction soly;

   {
      mfem::ParaViewDataCollection paraview_dc("TopOpt", &pmesh);
      paraview_dc.SetPrefixPath("ParaView");
      paraview_dc.SetLevelsOfDetail(order);
      paraview_dc.SetDataFormat(mfem::VTKFormat::BINARY);
      paraview_dc.SetHighOrderOutput(true);
      paraview_dc.SetCycle(0);
      paraview_dc.SetTime(0.0);

      paraview_dc.RegisterField("design",&pgdens);

      /*
      alco->GetSol(0,1.0,0.0,0.0,solx);
      alco->GetSol(0,0.0,1.0,0.0,soly);
      paraview_dc.RegisterField("solx",&solx);
      paraview_dc.RegisterField("soly",&soly);
      */

      //spdegf.ProjectCoefficient(spderf);
      //paraview_dc.RegisterField("reta",&spdegf);

      paraview_dc.Save();

      CoeffHoles holes;
      oddens.ProjectCoefficient(holes);
      oddens.GetTrueDofs(vtmpv);
      vtmpv*=0.4;
      //vtmpv=0.3;

      if (restart>0)
      {
         std::cout<<"Read the grid function"<<std::endl;
         //read the mesh and the design
         std::ostringstream oss;
         oss << std::setw(10) << std::setfill('0') << myrank;
         std::string gname="design_"+oss.str()+".gf";
         std::ifstream in;
         in.open(gname.c_str(),std::ios::in);
         //mfem::ParGridFunction ndes(&pmesh,in);
         //in.close();
         //oddens.ProjectGridFunction(ndes);//avoids mixing the FE spaces
         //oddens.GetTrueDofs(vtmpv);

         vtmpv=0.0;

         int vs;
         in>>vs; std::cout<<"vs="<<vs<<std::endl;

         for (int i=0; i<vs; i++)
         {
            in>>vtmpv[i];
         }
         in.close();

         double nr=mfem::ParNormlp(vtmpv,2,pmesh.GetComm());
         std::cout<<"nr="<<nr<<std::endl;
      }

      fsolv->Mult(vtmpv,vdens);
      pgdens.SetFromTrueDofs(vdens);
      double nr=mfem::ParNormlp(vdens,2,pmesh.GetComm());
      std::cout<<"nr="<<nr<<std::endl;
      double meanstd;
      double std;

      for (int i=1; i<max_it; i++)
      {

         vobj->SetProjection(0.2,8.0);
         vol=vobj->Eval(vdens);
         ivol=ivobj->Eval(vdens);

         alco->SetDensity(vdens,0.5,8.0,1.0);
         meanstd=alco->SMeanSTD(ograd,0.5,0.15); //alpha=1.0, r=0.1

         if (myrank==0)
         {
            std::cout<<"it: "<<i<<" obj="<<meanstd<<" vol="<<vol<<" cvol="<<max_vol<<" ivol="<<ivol<<std::endl;
         }
         //compute the gradients
         vobj->Grad(vdens,vgrad);
         //compute the original gradients
         fsolv->MultTranspose(ograd,ogrado);
         fsolv->MultTranspose(vgrad,vgrado);

         {
            //set xxmin and xxmax
            xxmin=vtmpv; xxmin-=max_ch;
            xxmax=vtmpv; xxmax+=max_ch;
            for (int li=0; li<xxmin.Size(); li++)
            {
               if (xxmin[li]<0.0) {xxmin[li]=0.0;}
               if (xxmax[li]>1.0) {xxmax[li]=1.0;}
            }
         }

         double con=vol-max_vol;
         mma->Update(vtmpv,ogrado,&con,&vgrado,xxmin,xxmax);

         fsolv->Mult(vtmpv,vdens);
         pgdens.SetFromTrueDofs(vdens);

         //save the design
         if (i%4==0)
         {
            paraview_dc.SetCycle(i);
            paraview_dc.SetTime(i*1.0);
            paraview_dc.Save();
         }
      }
   }

   {
      //save the mesh and the design
      std::ostringstream oss;
      oss << std::setw(10) << std::setfill('0') << myrank;
      std::string mname="pmesh_"+oss.str()+".msh";
      std::string gname="design_"+oss.str()+".gf";
      std::ofstream out;
      out.open(mname.c_str(),std::ios::out);
      pmesh.ParPrint(out);
      out.close();

      //save the design
      //oddens.SetFromTrueDofs(vtmpv);
      double nr=mfem::ParNormlp(vtmpv,2,pmesh.GetComm());
      std::cout<<"nr="<<nr<<std::endl;
      nr=mfem::ParNormlp(vdens,2,pmesh.GetComm());
      std::cout<<"nr="<<nr<<std::endl;
      out.open(gname.c_str(),std::ios::out);
      out<<vtmpv.Size()<<std::endl;
      for (int i=0; i<vtmpv.Size(); i++) {out<<vtmpv[i]<<std::endl;}
      //oddens.Save(out);
      out.close();
   }

   delete mma;
   delete vobj;
   delete ivobj;
   delete alco;
   delete fsolv;

   mfem::MFEMFinalizePetsc();
   MPI_Finalize();
   return 0;
}
