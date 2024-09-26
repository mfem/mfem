#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include <random>
#include "MMA.hpp"

#include "mtop_coefficients.hpp"
#include "mtop_solvers.hpp"

#include <bitset>

namespace adsampl {

double inv_sigmoid(double x, double p=1.0, double a=0.0)
{
    double tol = 1e-12;
    double c=p/(1.0-a);
    x = std::min(std::max(tol,x), c-tol);
    return std::log(x/(c-x));
}

/// @brief Sigmoid function
double sigmoid(double x,double p=1.0, double a=0.0)
{
    double s=p/(1.0-a);
    if (x >= 0)
    {
        return s/(1.0+std::exp(-x));
    }
    else
    {
        return s*std::exp(x)/(1.0+std::exp(x));
    }
}

/// @brief Derivative of sigmoid function
double der_sigmoid(double x,double p=1.0, double a=0.0)
{
    double s=p/(1.0-a);
    double tmp = sigmoid(-x);
    return s*(tmp - std::pow(tmp,2));
}


double Find_t(std::vector<double>& p, std::vector<double>& q,
              double alpha, double gamma,
              std::vector<double>& f,
              std::vector<int>& ind,
              double tol=1e-12, int max_it=100
              )
{

    double cval=-1.0;
    {
        std::vector<bool> pv; pv.resize(p.size());
        for(size_t i=0;i<p.size();i++){
            pv[i]=true;
        }
        for(size_t i=0;i<ind.size();i++){
            pv[ind[i]]=false;
        }
        for(size_t i=0;i<p.size();i++){
            if(pv[i]){	cval=cval+q[i]; }
        }
    }

    std::vector<double> g; g.resize(f.size());
    {
        for(size_t i=0;i<ind.size();i++){
            g[i]=inv_sigmoid(q[ind[i]],p[ind[i]],alpha)+gamma*f[i];
        }
    }

    bool flag=false; //iteration flag
    int iter=0;

    double ff;
    double df;
    double tt=0.0;
    double dc=0.0;

    for(int k=0;k<max_it;k++)
    {
        iter++;

        ff=cval;
        df=0.0;

        for(size_t i=0;i<ind.size();i++){
            ff=ff+sigmoid(g[i]-tt,p[ind[i]],alpha);
            df=df-der_sigmoid(g[i]-tt,p[ind[i]],alpha);
        }

        if(fabs(df)<tol){break;}
        dc=-ff/df;
        tt=tt-dc;
        if(fabs(dc)<tol){flag=true; break;}
    }

    if(!flag){
        mfem::mfem_warning("Projection reached maximum iteration without converging. "
                     "Result may not be accurate.");
    }

    return tt;
}





}

class CoeffHoles:public mfem::Coefficient
{
public:
    CoeffHoles(double pr=0.5)
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
        if(r<(0.45*period)){return 0.2;}
        return 0.8;
    }


private:
    double period;
};

class AlcoaBracket
{
public:
    AlcoaBracket(mfem::ParMesh* pmesh, int vorder=1,int seed=std::numeric_limits<int>::max()):E(),nu(0.2)
    {
        esolv=new mfem::ElasticitySolver(pmesh,vorder);
        esolv->AddMaterial(new mfem::LinIsoElasticityCoefficient(E,nu));
        esolv->SetNewtonSolver(1e-8,1e-12,1,0);
        esolv->SetLinearSolver(1e-10,1e-12,400);

        dfes=nullptr;
        cobj=new mfem::ComplianceObjective();

        generator.seed(seed);

        ppmesh=pmesh;

    }

    void SetDesignFES(mfem::ParFiniteElementSpace* fes)
    {
        dfes=fes;
        pdens.SetSpace(dfes);
        vdens.SetSize(dfes->GetTrueVSize());
    }

    ~AlcoaBracket()
    {
        delete cobj;
        delete esolv;

    }

    double Compliance(std::bitset<6>& supp, double eta, mfem::Vector& grad)
    {
        E.SetProjParam(eta,8.0);
        //set all bc
        esolv->DelDispBC();
        for(int j=0;j<6;j++){
            if(supp[j]==true){esolv->AddDispBC(2+j,4,0.0);}
        }
        esolv->AddSurfLoad(1,0.00,1.00,0.0);
        esolv->FSolve();
        esolv->GetSol(sol);

        cobj->Grad(sol,grad);
        return cobj->Eval(sol);
    }

    double Compliance(mfem::Vector& grad, double eta=0.5)
    {
        std::bitset<6> supp;
        for(int i=0;i<6;i++){supp[i]=true;}
        return Compliance(supp,eta,grad);
    }


    double MeanCompl(mfem::Vector& grad)
    {
        grad=0.0;
        mfem::Vector cgrad(grad.Size()); cgrad=0.0;

        double rez=0.0;
        for(size_t i=0;i<primp.size();i++){
            rez=rez+primp[i]*Compliance(vsupp[i],thresholds[i],cgrad);
            grad.Add(primp[i],cgrad);
        }
        return rez;
    }


    void SetDensity(mfem::Vector& vdens_,
                    double eta=0.5, double beta=8.0,double pen=3.0){

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

    std::vector<double>& GetDualProb(){	return dualq;}
    std::vector<double>& GetThesholds(){ return thresholds;}
    std::vector<std::bitset<6>>& GetSupp(){ return vsupp;}


    //clears all simulation cases and the associated probabilities
    void ClearCases()
    {
        vsupp.clear();
        dualq.clear();
        thresholds.clear();
    }

    //sets the simulation cases and the associated probabilities
    void SetCases(double eta=0.5)
    {
       std::bitset<6> bset;
       for(int j=0;j<6;j++){bset[j]=true;}
       vsupp.push_back(bset);
       thresholds.push_back(eta);
       dualq.push_back(0.1);

       for(int i=0;i<6;i++){
           for(int j=0;j<6;j++){
               if(j!=i){ bset[j]=true;}
               else{bset[j]=false;}
           }
           vsupp.push_back(bset);
           thresholds.push_back(eta);
           dualq.push_back(0.1);
       }

       for(int i=0;i<5;i++){
           for(int j=0;j<6;j++){
               if((j!=i)&&(j!=(i+1))){bset[j]=true;}
               else{bset[j]=false;}
           }
           vsupp.push_back(bset);
           thresholds.push_back(eta);
           dualq.push_back(0.01);
       }

       for(int i=0;i<4;i++){
           for(int j=0;j<6;j++){
               if((j!=i)&&(j!=(i+1))&&(j!=(i+2))){bset[j]=true;}
               else{bset[j]=false;}
           }
           vsupp.push_back(bset);
           thresholds.push_back(eta);
           dualq.push_back(0.01);
       }

       primp.resize(dualq.size());
       //normalize
       double sum=0.0;
       for(size_t i=0;i<dualq.size();i++){sum=sum+dualq[i];}
       for(size_t i=0;i<dualq.size();i++){
           dualq[i]=dualq[i]/sum;
           primp[i]=dualq[i];
       }
    }

    double EGDUpdate(mfem::Vector& grad, double eta_rate)
    {
        int myrank=ppmesh->GetMyRank();
        //compute the objective and the gradients
        std::vector<double> vals; vals.resize(dualq.size());

        grad=0.0;
        mfem::Vector cgrad(grad.Size()); cgrad=0.0;

        double rez=0.0;
        double nfa=0.0;
        for(size_t i=0;i<dualq.size();i++){
            vals[i]=Compliance(vsupp[i],thresholds[i],cgrad);
            rez=rez+vals[i]*dualq[i];
            grad.Add(dualq[i],cgrad);
            nfa=nfa+dualq[i];
        }
        grad/=nfa;
        rez/=nfa;

        std::vector<double> w; w.resize(dualq.size());
        for(size_t i=0;i<dualq.size();i++){ w[i]=dualq[i]*exp(eta_rate*vals[i]); }

        nfa=0.0;
        for(size_t i=0;i<w.size();i++){nfa=nfa+w[i];}
        for(size_t i=0;i<w.size();i++){
            w[i]/=nfa;
            dualq[i]=dualq[i]+w[i];

            if(myrank==0){std::cout<<w[i]<<" ";}
        }
        if(myrank==0){std::cout<<std::endl;}

        for(size_t i=0;i<w.size();i++){
            if(myrank==0){std::cout<<dualq[i]<<" ";}
        }
        if(myrank==0){std::cout<<std::endl;}


        return rez;
    }


    double EGDUpdate(mfem::Vector& grad, double eta_rate, int nsampl)
    {
        //construct discrete distribution
        std::discrete_distribution<int> d(dualq.begin(),dualq.end());


        int* tmpv=new int[nsampl];

        //generate nsampl
        MPI_Comm comm=ppmesh->GetComm();
        int myrank=ppmesh->GetMyRank();
        if(myrank==0){
            for(int i=0;i<nsampl;i++){
                tmpv[i]=d(generator);
                std::cout<<tmpv[i]<<" ";
            }
            std::cout<<std::endl;
        }
        //communicate the center from process zero to all others
        MPI_Bcast(tmpv, nsampl, MPI_INT, 0, comm);

        //compute the objective and the gradients
        std::vector<double> vals; vals.resize(nsampl);

        grad=0.0;
        mfem::Vector cgrad(grad.Size()); cgrad=0.0;

        double rez=0.0;
        double nfa=double(nsampl);
        for(int i=0;i<nsampl;i++){
            vals[i]=Compliance(vsupp[tmpv[i]],thresholds[tmpv[i]],cgrad);
            rez=rez+vals[i];
            grad.Add(1.0,cgrad);
        }
        grad/=nfa;
        rez/=nfa;

        std::vector<double> w; w=dualq;
        for(int i=0;i<nsampl;i++){ w[tmpv[i]]=0.0;}

        //update the dual probabilities
        for(int i=0;i<nsampl;i++){
            w[tmpv[i]]=w[tmpv[i]]+dualq[tmpv[i]]*exp(eta_rate*vals[i]);
        }
        //normalize the dual probabilities
        nfa=0.0;
        for(size_t i=0;i<w.size();i++){nfa=nfa+w[i];}
        for(size_t i=0;i<w.size();i++){
            w[i]/=nfa;
            dualq[i]=dualq[i]+w[i];
        }


        delete [] tmpv;
        //return the objective
        return rez;
    }



private:
    mfem::YoungModulus E;
    double nu;

    mfem::ParFiniteElementSpace* dfes; //design FES
    mfem::ParGridFunction pdens;
    mfem::Vector vdens;

    mfem::ElasticitySolver* esolv;
    mfem::ComplianceObjective* cobj;

    mfem::ParGridFunction sol;


    //the following three vectors should have the same size
    std::vector<double> dualq;
    std::vector<double> primp;
    std::vector<std::bitset<6>> vsupp;
    std::vector<double> thresholds;

    std::default_random_engine generator;
    mfem::ParMesh* ppmesh;
};



int main(int argc, char *argv[])
{
   // Initialize MPI.
   int nprocs, myrank;
   MPI_Init(&argc, &argv);
   MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
   MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

   // Parse command-line options.
   const char *mesh_file = "./canti_2D_6.msh";
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
       vert*=0.01;
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
   mfem::FilterSolver* fsolv=new mfem::FilterSolver(0.07,&pmesh);
   fsolv->SetSolver(1e-8,1e-12,100,0);
   fsolv->AddBC(1,1.0);
   fsolv->AddBC(2,1.0);
   fsolv->AddBC(3,1.0);
   fsolv->AddBC(4,1.0);
   fsolv->AddBC(5,1.0);
   fsolv->AddBC(6,1.0);
   fsolv->AddBC(7,1.0);
   fsolv->AddBC(8,0.0);

   mfem::ParGridFunction pgdens(fsolv->GetFilterFES());
   mfem::ParGridFunction oddens(fsolv->GetDesignFES());
   mfem::ParGridFunction spdegf(fsolv->GetFilterFES());
   mfem::Vector vdens; vdens.SetSize(fsolv->GetFilterFES()->GetTrueVSize()); vdens=0.0;
   mfem::Vector vtmpv; vtmpv.SetSize(fsolv->GetDesignFES()->GetTrueVSize()); vtmpv=0.5;

   fsolv->Mult(vtmpv,vdens);
   pgdens.SetFromTrueDofs(vdens);


   AlcoaBracket* alco=new AlcoaBracket(&pmesh,1);
   alco->SetDesignFES(pgdens.ParFESpace());
   alco->SetDensity(vdens);
   alco->SetCases(0.7);

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
       double val=alco->MeanCompliance();
       alco->MeanCompliance(fgrad);
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
           alco->Solve();
           lqoi=alco->MeanCompliance();
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
   vobj->SetProjection(0.5,8.0);//threshold 0.2

   //compute the total volume
   double tot_vol;
   {
       vdens=1.0;
       tot_vol=vobj->Eval(vdens);
   }
   double max_vol=0.4*tot_vol;
   if(myrank==0){ std::cout<<"tot vol="<<tot_vol<<std::endl;}

   //intermediate volume
   mfem::VolumeQoI* ivobj=new mfem::VolumeQoI(fsolv->GetFilterFES());
   ivobj->SetProjection(0.5,32);

   //gradients with respect to the filtered field
   mfem::Vector ograd(fsolv->GetFilterFES()->GetTrueVSize()); ograd=0.0; //of the objective
   mfem::Vector vgrad(fsolv->GetFilterFES()->GetTrueVSize()); vgrad=0.0; //of the volume contr.

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

      //alco->GetSol(6,0.0,1.0,0.0,solx);
      //alco->GetSol(1,0.0,1.0,0.0,soly);
      //paraview_dc.RegisterField("solx",&solx);
      //paraview_dc.RegisterField("soly",&soly);

      //spdegf.ProjectCoefficient(spderf);
      //paraview_dc.RegisterField("reta",&spdegf);

      paraview_dc.Save();

      CoeffHoles holes;
      oddens.ProjectCoefficient(holes);
      oddens.GetTrueDofs(vtmpv);
      vtmpv=0.3;
      fsolv->Mult(vtmpv,vdens);
      pgdens.SetFromTrueDofs(vdens);


      for(int i=1;i<max_it;i++){

          vobj->SetProjection(0.3,8.0);
          alco->SetDensity(vdens,0.7,8.0,1.0);

          //cpl=alco->Compliance(ograd);
          //cpl=alco->MeanCompl(ograd);
          cpl=alco->EGDUpdate(ograd,0.001);
          vol=vobj->Eval(vdens);
          ivol=ivobj->Eval(vdens);


          if(myrank==0){
              std::cout<<"it: "<<i<<" obj="<<cpl<<" vol="<<vol<<" cvol="<<max_vol<<" ivol="<<ivol
                      <<std::endl;
          }
          vobj->Grad(vdens,vgrad);
          //compute the original gradients
          fsolv->MultTranspose(ograd,ogrado);
          fsolv->MultTranspose(vgrad,vgrado);

          {
              //set xxmin and xxmax
              xxmin=vtmpv; xxmin-=max_ch;
              xxmax=vtmpv; xxmax+=max_ch;
              for(int li=0;li<xxmin.Size();li++){
                  if(xxmin[li]<0.0){xxmin[li]=0.0;}
                  if(xxmax[li]>1.0){xxmax[li]=1.0;}
              }
          }

          double con=vol-max_vol;
          mma->Update(vtmpv,ogrado,&con,&vgrado,xxmin,xxmax);

          fsolv->Mult(vtmpv,vdens);
          pgdens.SetFromTrueDofs(vdens);

          //alco->GetSol(1,0.0,1.0,0.0,solx);
          //alco->GetSol(6,0.0,1.0,0.0,soly);

          //paraview_dc.RegisterField("solx",&solx);
          //paraview_dc.RegisterField("soly",&soly);

          //save the design
          if(i%4==0)
          {
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
