#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include <random>
#include "MMA.hpp"

#include "mtop_solvers.hpp"

#include "mtop_coefficients.hpp"

class YoungModulusMech:public mfem::YoungModulus
{
public:
    YoungModulusMech()
    {
        dens=nullptr;
        Emax=1.0;
        Emin=1e-6;
        loc_eta=new mfem::ConstantCoefficient(0.5);
        eta=loc_eta;
        beta=8.0;
        pp=1.0;
        hx=0.0; //below hx we have the spring for the mechanism
        solid=nullptr;
        simp=true;
    }

    virtual
    ~YoungModulusMech(){
        delete loc_eta;
    }

    virtual
    bool GetSIMP(){return simp;}

    virtual
    void SetSIMP(bool simp_=true){
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
    void SetSolid(mfem::Coefficient* solid_){
        solid=solid_;
    }

    void SetSpringProsition(double t_)
    {
        hx=t_;
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
    double 	Eval(mfem::ElementTransformation &T, const mfem::IntegrationPoint &ip)
    {

        //evaluate the actual coordinates
        mfem::Vector xx(T.GetSpaceDim()); xx=0.0;
        T.Transform(ip,xx);

        if(xx(0)>hx){
            //evaluate density
            double dd=1.0;
            if(dens!=nullptr){dd=dens->GetValue(T,ip);}
            else if(coef!=nullptr){dd=coef->Eval(T,ip);}

            if(dd>1.0){dd=1.0;}
            if(dd<0.0){dd=0.0;}
            //eval eta
            double deta=eta->Eval(T,ip);
            //do the projection
            double pd=mfem::PointwiseTrans::HProject(dd,deta,beta);
            //evaluate the E modulus
            if(simp==false){
                return Emin+(Emax-Emin)*std::pow(pd,pp);
            }else{
                return Emin+(Emax-Emin)*std::pow(dd,pp);
            }

        }else{//solid
            if(solid==nullptr)
            {
                return Emax;
            }else{
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

        if(xx(0)>hx){
        //evaluate density
        double dd=1.0;
        if(dens!=nullptr){dd=dens->GetValue(T,ip);}
        else if(coef!=nullptr){dd=coef->Eval(T,ip);}

        if(dd>1.0){dd=1.0;}
        if(dd<0.0){dd=0.0;}
        //eval eta
        double deta=eta->Eval(T,ip);
        //do the projection
        double pd=mfem::PointwiseTrans::HProject(dd,deta,beta);
        //evaluate hte gradient of the projection
        double pg=mfem::PointwiseTrans::HGrad(dd,deta,beta);
        //evaluate the gradient with respect to the density field
        if(simp==false){
            return (Emax-Emin)*pg*pp*std::pow(pd,pp-1.0);
        }else{
            return (Emax-Emin)*std::pow(dd,pp-1.0);
        }

        }else{
            return 0.0;
        }
    }

private:
    double hx; //thickness
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

class Mech
{
public:
    Mech(mfem::ParMesh* pmesh_, int vorder=2, int seed=std::numeric_limits<int>::max()):E(),nu(0.2)
    {
        esolv=new mfem::ElasticitySolver(pmesh_,vorder);
        esolv->AddMaterial(new mfem::LinIsoElasticityCoefficient(E,nu));
        esolv->SetNewtonSolver(1e-8,1e-8,1,0);
        esolv->SetLinearSolver(1e-10,1e-8,100);

        esola=new mfem::ElasticitySolver(pmesh_,vorder);
        esola->AddMaterial(new mfem::LinIsoElasticityCoefficient(E,nu));
        esola->SetNewtonSolver(1e-8,1e-8,1,0);
        esola->SetLinearSolver(1e-10,1e-8,100);

        pmesh=pmesh_;
        dfes=nullptr;
        lobj=new mfem::InvMechObjective();

        generator.seed(seed);


        gf=new mfem::RandFieldCoefficient(pmesh_,vorder);
        gf->SetCorrelationLen(0.2);
        gf->SetMaternParameter(4.0);
        gf->SetScale(0.1);
        gf->Sample(seed);

        af=new mfem::RandFieldCoefficient(pmesh_,vorder);
        af->SetCorrelationLen(0.2);
        af->SetMaternParameter(4.0);
        af->SetScale(0.1);
        af->SetZeroDirichletBC(2);
        af->Sample(seed+1347);

        num_samples=100;
    }

    void SetDesignFES(mfem::ParFiniteElementSpace* fes)
    {
        dfes=fes;
        pdens.SetSpace(dfes);
        vdens.SetSize(dfes->GetTrueVSize());
    }

    ~Mech()
    {
        delete gf;
        delete af;
        delete lobj;
        delete esolv;
        delete esola;
    }

    void GetSRand(mfem::ParGridFunction& pgf){
        pgf.ProjectCoefficient(*gf);
    }

    void GetARand(mfem::ParGridFunction& pgf){
        pgf.ProjectCoefficient(*af);
    }

    void SetNumSamples(int ns)
    {
        num_samples=ns;
    }

    void SetCorrelationLen(double l)
    {
        gf->SetCorrelationLen(l);
        af->SetCorrelationLen(l);
    }

    void SetSIMP(bool simp_=false){
        E.SetSIMP(simp_);
    }

    void SetDensity(mfem::Vector& vdens_,
                    double eta=0.75, double beta=8.0,double pen=1.0){

        vdens=vdens_;
        pdens.SetFromTrueDofs(vdens);

        E.SetDens(&pdens);
        E.SetProjParam(eta,beta);
        E.SetEMaxMin(1e-6,1.0);
        E.SetPenal(pen);

        lobj->SetE(&E);
        lobj->SetDens(vdens);
        lobj->SetDesignFES(dfes);
    }

    mfem::Coefficient* GetE(){
        return &E;
    }

    double Objective(mfem::Vector& grad)
    {
        //adjoint problem
        esolv->DelDispBC();
        esolv->AddDispBC(7,4,0.0);
        esolv->AddDispBC(5,0,0);
        esolv->AddDispBC(6,0,0);
        esolv->AddDispBC(2,1,0);
        esolv->AddSurfLoad(3,-1.00,0.00,0.00);
        esolv->FSolve();
        esolv->GetSol(adj);

        //forward solve
        //set all bc
        esolv->DelDispBC();
        esolv->AddDispBC(7,4,0.0);
        esolv->AddDispBC(5,0,0);
        esolv->AddDispBC(6,0,0);
        esolv->AddDispBC(2,1,0);
        esolv->AddSurfLoad(4,1.00,0.00,0.00);
        esolv->FSolve();

        double rez=lobj->Eval(esolv->GetDisplacements(),adj);
        lobj->Grad(esolv->GetDisplacements(),adj,grad);

        return rez;
    }

private:
    YoungModulusMech E;
    double nu;


    mfem::ParGridFunction adj;

    mfem::ParFiniteElementSpace* dfes; //design FES
    mfem::ParGridFunction pdens;
    mfem::Vector vdens;

    mfem::ElasticitySolver* esolv;
    mfem::ElasticitySolver* esola;//asymmetric solver
    mfem::InvMechObjective* lobj;


    mfem::ParGridFunction sol;
    mfem::ParMesh* pmesh;

    mfem::RandFieldCoefficient* gf;
    mfem::RandFieldCoefficient* af;//asymmetric load
    int num_samples;

    std::vector<int> seeds;
    std::vector<int> seeda;

    std::default_random_engine generator;
};


int main(int argc, char *argv[])
{
   // Initialize MPI.
   int nprocs, myrank;
   MPI_Init(&argc, &argv);
   MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
   MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

   // Parse command-line options.
   const char *mesh_file = "table.msh";
   int order = 1;
   int nsplit=4;
   bool static_cond = false;
   int ser_ref_levels = 0;
   int par_ref_levels = 1;
   double rel_tol = 1e-7;
   double abs_tol = 1e-15;
   double fradius = 0.02;
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
   args.AddOption(&nsplit,
                     "-csp",
                     "--comm_split",
                     "How many communicators should we use for the simulations.");
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

   int seed;
   {
       std::default_random_engine generator;
       std::uniform_int_distribution<int> distribution(1,std::numeric_limits<int>::max());

       for(int i=0;i<nprocs;i++){
           if(i!=0){
               if(myrank==0){
                   int rnum=distribution(generator);
                   MPI_Send(&rnum,1,MPI_INT,i,100,MPI_COMM_WORLD);}
               if(myrank==i){
                   MPI_Recv(&seed,1,MPI_INT,0,100,MPI_COMM_WORLD,MPI_STATUS_IGNORE);}
           }else{
               seed=distribution(generator);
           }
       }
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
       if(myrank==0){
           std::cout<<"Xmin:";xmin.Print(std::cout);
           std::cout<<"Xmax:";xmax.Print(std::cout);
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
   //mfem::ParMesh pmesh(MPI_COMM_WORLD, mesh);
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
   mfem::FilterSolver* fsolv=new mfem::FilterSolver(fradius,&pmesh);
   fsolv->SetSolver(1e-8,1e-12,100,0);
   fsolv->AddBC(1,0.0);
   fsolv->AddBC(4,1.0);
   fsolv->AddBC(3,1.0);

   mfem::ParGridFunction pgdens(fsolv->GetFilterFES());
   mfem::ParGridFunction emod(fsolv->GetFilterFES()); emod=0.0;
   mfem::ParGridFunction oddens(fsolv->GetDesignFES());
   //mfem::ParGridFunction spdegf(fsolv->GetFilterFES());
   mfem::Vector vdens; vdens.SetSize(fsolv->GetFilterFES()->GetTrueVSize()); vdens=0.0;
   mfem::Vector vtmpv; vtmpv.SetSize(fsolv->GetDesignFES()->GetTrueVSize()); vtmpv=0.5;

   fsolv->Mult(vtmpv,vdens);
   pgdens.SetFromTrueDofs(vdens);

   Mech* alco=new Mech(&pmesh,1,seed);
   alco->SetDesignFES(pgdens.ParFESpace());
   alco->SetDensity(vdens);
   alco->SetCorrelationLen(corr_len);
   alco->SetNumSamples(num_samples);

   delete alco;
   delete fsolv;


   mfem::MFEMFinalizePetsc();
   MPI_Finalize();
   return 0;
}
