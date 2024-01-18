#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include <random>
#include "MMA.hpp"

#include "mtop_solvers.hpp"

#include "mtop_coefficients.hpp"

class StripesCoefX:public mfem::Coefficient
{
public:
   StripesCoefX(double dx_=0.2)
   {
      dx=dx_;
   }

   virtual
   double Eval(mfem::ElementTransformation &T, const mfem::IntegrationPoint &ip)
   {

      double x[3];
      mfem::Vector transip(x, T.GetSpaceDim());
      T.Transform(ip, transip);

      return 0.3+0.3*sin(2.0*M_PI*x[0]/dx+M_PI/2.0);

      /*
           int nx=x[0]/dx;

           x[0]=x[0]-double(nx)*dx-0.5*dx;

           double r=sqrt(x[0]*x[0]);
           if(r<dx){return 0.1;}
           return 0.7;
      */
   }
private:
   double dx;
};

class CoeffHoles:public mfem::Coefficient
{
public:
   CoeffHoles(double pr=1.0/8.0)
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

class CutCoeff:public mfem::Coefficient
{
public:
   CutCoeff(mfem::Coefficient* co_, double tr_=0.0)
   {
      co=co_;
      tres=tr_;
      sca=1.0;
   }

   virtual
   double  Eval(mfem::ElementTransformation &T, const mfem::IntegrationPoint &ip)
   {
      double val=co->Eval(T,ip);
      if (val>tres) { return sca*(val-tres);}
      return 0.0;
   }

   void SetScale(double sca_)
   {
      sca=sca_;
   }

private:
   mfem::Coefficient *co;
   double tres;
   double sca;
};


class YoungModulusTable:public mfem::YoungModulus
{
public:
   YoungModulusTable()
   {
      dens=nullptr;
      Emax=1.0;
      Emin=1e-6;
      loc_eta=new mfem::ConstantCoefficient(0.5);
      eta=loc_eta;
      beta=8.0;
      pp=1.0;
      h=1; //default thickenss
      solid=nullptr;
      simp=true;
   }

   virtual
   ~YoungModulusTable()
   {
      delete loc_eta;
   }

   virtual
   bool GetSIMP() {return simp;}

   virtual
   void SetSIMP(bool simp_=true)
   {
      simp=simp_;
   }

   virtual
   void SetDens(mfem::ParGridFunction* dens_)
   {
      dens=dens_;
   }

   virtual
   void SetDens(mfem::Coefficient* coef_)
   {
      coef=coef_;
   }

   virtual
   void SetSolid(mfem::Coefficient* solid_)
   {
      solid=solid_;
   }

   void SetThickness(double t_)
   {
      h=t_;
   }


   virtual
   void SetProjParam(mfem::Coefficient& eta_, double beta_)
   {
      eta=&eta_;
      beta=beta_;
   }

   virtual
   void SetProjParam(double eta_, double beta_)
   {
      delete loc_eta;
      loc_eta=new mfem::ConstantCoefficient(eta_);
      eta=loc_eta;
      beta=beta_;
   }

   virtual
   void SetEMaxMin(double Emin_,double Emax_)
   {
      Emax=Emax_;
      Emin=Emin_;
   }

   virtual
   void SetPenal(double pp_)
   {
      pp=pp_;
   }

   virtual
   double  Eval(mfem::ElementTransformation &T, const mfem::IntegrationPoint &ip)
   {

      //evaluate the actual coordinates
      mfem::Vector xx(T.GetSpaceDim()); xx=0.0;
      T.Transform(ip,xx);

      if (xx(1)<1.00*h)
      {
         //evaluate density
         double dd=1.0;
         if (dens!=nullptr) {dd=dens->GetValue(T,ip);}
         else if (coef!=nullptr) {dd=coef->Eval(T,ip);}

         if (dd>1.0) {dd=1.0;}
         if (dd<0.0) {dd=0.0;}
         //eval eta
         double deta=eta->Eval(T,ip);
         //do the projection
         double pd=mfem::PointwiseTrans::HProject(dd,deta,beta);
         //evaluate the E modulus
         if (simp==false)
         {
            return Emin+(Emax-Emin)*std::pow(pd,pp);
         }
         else
         {
            return Emin+(Emax-Emin)*std::pow(dd,pp);
         }

      }
      else  //solid
      {
         if (solid==nullptr)
         {
            return Emax;
         }
         else
         {
            return Emin+(Emax-Emin)*(solid->Eval(T,ip));
         }
      }
   }

   ///returnas the pointwise gradient with respect to the density
   virtual
   double Grad(mfem::ElementTransformation &T, const mfem::IntegrationPoint &ip)
   {

      //evaluate the actual coordinates
      mfem::Vector xx(T.GetSpaceDim()); xx=0.0;
      T.Transform(ip,xx);

      if (xx(1)<1.00*h)
      {
         //evaluate density
         double dd=1.0;
         if (dens!=nullptr) {dd=dens->GetValue(T,ip);}
         else if (coef!=nullptr) {dd=coef->Eval(T,ip);}

         if (dd>1.0) {dd=1.0;}
         if (dd<0.0) {dd=0.0;}
         //eval eta
         double deta=eta->Eval(T,ip);
         //do the projection
         double pd=mfem::PointwiseTrans::HProject(dd,deta,beta);
         //evaluate hte gradient of the projection
         double pg=mfem::PointwiseTrans::HGrad(dd,deta,beta);
         //evaluate the gradient with respect to the density field
         if (simp==false)
         {
            return (Emax-Emin)*pg*pp*std::pow(pd,pp-1.0);
         }
         else
         {
            return (Emax-Emin)*std::pow(dd,pp-1.0);
         }

      }
      else
      {
         return 0.0;
      }
   }

private:
   double h; //thickness
   mfem::ParGridFunction* dens;
   mfem::Coefficient* coef;
   double Emax;
   double Emin;
   mfem::Coefficient* eta;
   mfem::Coefficient* loc_eta;
   double beta;
   double pp;

   bool simp;

   mfem::Coefficient* solid;
};


class Table
{
public:
   Table(mfem::ParMesh* pmesh_, int vorder=2):E(),nu(0.2)
   {
      esolv=new mfem::ElasticitySolver(pmesh_,vorder);
      esolv->AddMaterial(new mfem::LinIsoElasticityCoefficient(E,nu));
      esolv->SetNewtonSolver(1e-8,1e-12,1,0);
      esolv->SetLinearSolver(1e-10,1e-12,400);

      pmesh=pmesh_;
      dfes=nullptr;
      cobj=new mfem::ComplianceObjective();

      gf=new mfem::RandFieldCoefficient(pmesh_,vorder);
      gf->SetCorrelationLen(0.2);
      gf->SetMaternParameter(4.0);
      gf->SetScale(0.1);
      num_samples=100;
   }

   void SetDesignFES(mfem::ParFiniteElementSpace* fes)
   {
      dfes=fes;
      pdens.SetSpace(dfes);
      vdens.SetSize(dfes->GetTrueVSize());
   }

   ~Table()
   {
      delete gf;
      delete cobj;
      delete esolv;
   }

   void SetNumSamples(int ns)
   {
      num_samples=ns;
   }

   void SetCorrelationLen(double l)
   {
      gf->SetCorrelationLen(l);
   }

   void SetSIMP(bool simp_=false)
   {
      E.SetSIMP(simp_);
   }

   void SetDensity(mfem::Vector& vdens_,
                   double eta=0.75, double beta=8.0,double pen=1.0)
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

   mfem::Coefficient* GetE()
   {
      return &E;
   }

   /// Evaluates the compliance
   double  Compliance(mfem::Vector& grad)
   {
      //set all bc
      esolv->DelDispBC();
      esolv->AddDispBC(1,4,0.0);
      esolv->AddSurfLoad(3,0.00,1.00,0.0);

      esolv->FSolve();

      double rez=cobj->Eval(esolv->GetDisplacements());

      cobj->Grad(esolv->GetDisplacements(),grad);

      return rez;
   }

   double MeanComplCF(mfem::Vector& grad)
   {

      //set all bc
      esolv->DelDispBC();
      esolv->AddDispBC(1,4,0.0);
      mfem::Vector lgr(grad.Size());
      grad=0.0;


      mfem::VectorArrayCoefficient ff(pmesh->SpaceDimension());
      mfem::ConstantCoefficient one(1.0);
      mfem::ConstantCoefficient zero(0.0);
      mfem::ConstantCoefficient hforce(1.0);
      ff.Set(0,&hforce,false);
      ff.Set(1,&one,false);

      esolv->AddSurfLoad(3,ff);
      cobj->SetE(&E);
      cobj->SetDens(vdens);
      cobj->SetDesignFES(dfes);
      esolv->AssembleTangent();

      std::default_random_engine generator;
      std::normal_distribution<double> ndistr(0.0,0.2);


      int myrank;
      MPI_Comm_rank(pmesh->GetComm(),&myrank);


      int n=num_samples;
      double obj=0.0;
      double var=0.0;
      for (int i=0; i<n; i++)
      {


         //generate the horizontal force

         hforce.constant=ndistr(generator);
         MPI_Bcast(&hforce.constant,1,MPI_DOUBLE,
                   0, pmesh->GetComm());


         esolv->LSolve();

         double rez=cobj->Eval(esolv->GetDisplacements());
         cobj->Grad(esolv->GetDisplacements(),lgr);
         grad.Add(1.0,lgr);
         obj=obj+rez;
         var=var+rez*rez;
      }


      grad/=double(n);
      var=var/double(n);
      obj=obj/double(n);

      if (myrank==0)
      {
         std::cout<<"Var="<<var-obj*obj<<std::endl;
      }

      return obj;
   }


   double MeanCompliance(mfem::Vector& grad)
   {
      //set all bc
      esolv->DelDispBC();
      esolv->AddDispBC(1,4,0.0);
      mfem::Vector lgr(grad.Size());
      grad=0.0;

      mfem::VectorArrayCoefficient ff(pmesh->SpaceDimension());
      mfem::ConstantCoefficient one(1.0);
      mfem::ConstantCoefficient zero(0.0);
      mfem::LognormalDistributionCoefficient lf(gf,0.0,1.0);
      lf.SetScale(0.2);
      CutCoeff cut_co(gf,0.0);
      //ff.Set(1,&one,false);
      //ff.Set(0,&zero,false);
      //ff.Set(0,&cut_co,false);
      //ff.Set(0,&lf,false);
      //ff.Set(1,gf,false);
      //ff.Set(1,&cut_co,false);
      //ff.Set(1,&one,false);
      ff.Set(1,&one,false);
      ff.Set(0,gf,false);

      esolv->AddSurfLoad(3,ff);
      cobj->SetE(&E);
      cobj->SetDens(vdens);
      cobj->SetDesignFES(dfes);
      esolv->AssembleTangent();

      int n=num_samples;
      double obj=0.0;
      double var=0.0;
      for (int i=0; i<n; i++)
      {

         /*
         if((i%2)==0){ //cut_co.SetScale(1.0);
             lf.SetScale(0.2);
         }
         else
         { //cut_co.SetScale(-1.0);
             lf.SetScale(-0.2);
         }*/

         if (seeds.size()<(i+1))
         {
            int seed = static_cast<int>(std::time(nullptr));
            seeds.push_back(seed);
         }
         gf->Sample(seeds[i]);


         //gf->Sample();

         /*
         esolv->AddSurfLoad(3,ff);
         cobj->SetE(&E);
         cobj->SetDens(vdens);
         cobj->SetDesignFES(dfes);
         */

         //esolv->FSolve();

         //esolv->AssembleTangent();
         esolv->LSolve();

         double rez=cobj->Eval(esolv->GetDisplacements());
         cobj->Grad(esolv->GetDisplacements(),lgr);
         grad.Add(1.0,lgr);
         obj=obj+rez;
         var=var+rez*rez;

      }

      grad/=double(n);
      var=var/double(n);
      obj=obj/double(n);

      int myrank;
      MPI_Comm_rank(pmesh->GetComm(),&myrank);
      if (myrank==0)
      {
         std::cout<<"Var="<<var-obj*obj<<std::endl;
      }

      return obj;
   }



private:
   YoungModulusTable E;
   double nu;

   mfem::ParFiniteElementSpace* dfes; //design FES
   mfem::ParGridFunction pdens;
   mfem::Vector vdens;

   mfem::ElasticitySolver* esolv;
   mfem::ComplianceObjective* cobj;


   mfem::ParGridFunction sol;
   mfem::ParMesh* pmesh;

   mfem::RandFieldCoefficient* gf;
   int num_samples;

   std::vector<int> seeds;
};


int main(int argc, char *argv[])
{
   // Initialize MPI.
   int nprocs, myrank;
   MPI_Init(&argc, &argv);
   MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
   MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

   // Parse command-line options.
   const char *mesh_file = "foundation.msh";
   int order = 1;
   bool static_cond = false;
   int ser_ref_levels = 0;
   int par_ref_levels = 0;
   double rel_tol = 1e-7;
   double abs_tol = 1e-15;
   double fradius = 0.04;
   double corr_len = 0.2;
   int tot_iter = 100;
   int max_it = 51;
   int print_level = 1;
   int num_samples=100;
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
   args.AddOption(&num_samples,
                  "-ns",
                  "--num-samples",
                  "Number of samples.");
   args.AddOption(&fradius,
                  "-r",
                  "--radius",
                  "Filter radius");
   args.AddOption(&corr_len,
                  "-crl",
                  "--corr",
                  "Correlation length");
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

   if (myrank==0)
   {
      std::cout<<"num el="<<pmesh.GetNE()<<std::endl;
   }

   //allocate the filter
   mfem::FilterSolver* fsolv=new mfem::FilterSolver(fradius,&pmesh);
   fsolv->SetSolver(1e-8,1e-12,100,0);
   fsolv->AddBC(2,0.0);
   fsolv->AddBC(3,1.0);
   fsolv->AddBC(4,0.0);

   mfem::ParGridFunction pgdens(fsolv->GetFilterFES());
   mfem::ParGridFunction emod(fsolv->GetFilterFES()); emod=0.0;
   mfem::ParGridFunction oddens(fsolv->GetDesignFES());
   //mfem::ParGridFunction spdegf(fsolv->GetFilterFES());
   mfem::Vector vdens; vdens.SetSize(fsolv->GetFilterFES()->GetTrueVSize());
   vdens=0.0;
   mfem::Vector vtmpv; vtmpv.SetSize(fsolv->GetDesignFES()->GetTrueVSize());
   vtmpv=0.5;

   fsolv->Mult(vtmpv,vdens);
   pgdens.SetFromTrueDofs(vdens);

   Table* alco=new Table(&pmesh,1);
   alco->SetDesignFES(pgdens.ParFESpace());
   alco->SetDensity(vdens);
   alco->SetCorrelationLen(corr_len);
   alco->SetNumSamples(num_samples);

   mfem::PVolumeQoI* vobj=new mfem::PVolumeQoI(fsolv->GetFilterFES());
   //mfem::VolumeQoI* vobj=new mfem::VolumeQoI(fsolv->GetFilterFES());
   vobj->SetProjection(0.5,8.0);//threshold 0.2

   //compute the total volume
   double tot_vol;
   {
      vdens=1.0;
      tot_vol=vobj->Eval(vdens);
   }
   double max_vol=0.300*tot_vol;
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

   {
      mfem::ParaViewDataCollection paraview_dc("TopOpt", &pmesh);
      paraview_dc.SetPrefixPath("ParaView");
      paraview_dc.SetLevelsOfDetail(order);
      paraview_dc.SetDataFormat(mfem::VTKFormat::BINARY);
      paraview_dc.SetHighOrderOutput(true);
      paraview_dc.SetCycle(0);
      paraview_dc.SetTime(0.0);

      paraview_dc.RegisterField("design",&pgdens);
      paraview_dc.RegisterField("E",&emod);


      CoeffHoles holes;
      //StripesCoefX stripes(corr_len);
      StripesCoefX stripes(0.25);
      //oddens.ProjectCoefficient(holes);
      //oddens*=0.3;
      //oddens.GetTrueDofs(vtmpv);
      //oddens=0.3;
      oddens.ProjectCoefficient(stripes);
      oddens.GetTrueDofs(vtmpv);


      fsolv->Mult(vtmpv,vdens);
      pgdens.SetFromTrueDofs(vdens);
      double nr=mfem::ParNormlp(vdens,2,pmesh.GetComm());
      std::cout<<"nr="<<nr<<std::endl;

      paraview_dc.Save();

      double obj;

      for (int i=1; i<max_it; i++)
      {


         if (i<10)
         {
            vobj->SetProjection(0.5,2.0);
            alco->SetDensity(vdens,0.5,8.0,2.0);
            alco->SetSIMP(true);
         }
         else if (i<250)
         {
            vobj->SetProjection(0.5,2.0);
            alco->SetDensity(vdens,0.5,8.0,3.0);
            alco->SetSIMP(true);
         }
         else
         {
            vobj->SetProjection(0.3,8.0);
            alco->SetDensity(vdens,0.7,8.0,1.0);
            alco->SetSIMP(false);
         }

         vol=vobj->Eval(vdens);
         ivol=ivobj->Eval(vdens);

         //obj=alco->Compliance(ograd);
         obj=alco->MeanCompliance(ograd);

         if (myrank==0)
         {
            std::cout<<"it: "<<i<<" obj="<<obj<<" vol="<<vol<<" ivol="<<ivol<<std::endl;
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
            emod.ProjectCoefficient(*(alco->GetE()));
            paraview_dc.SetCycle(i);
            paraview_dc.SetTime(i*1.0);
            paraview_dc.Save();
         }
      }
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
