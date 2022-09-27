#include "mfem.hpp"
#include <fstream>
#include <iostream>

#include "elasticity.hpp"
#include "../shifted/marking.hpp"


double Circ(const mfem::Vector &x)
{
    double r=0.5;
    //double nr=x.Norml2();
    double nr=std::sqrt(x*x);
    return r-nr;
}

void CircGrad(const mfem::Vector &x, mfem::Vector & res)
{
    res=x;
    //double nr=x.Norml2();
    double nr=std::sqrt(x*x);
    if(nr>(10.0*std::numeric_limits<double>::epsilon())){res/=nr;}
    else{res=0.0;}
}


double CplxCompO(const mfem::Vector &x, const mfem::Vector &ct)
{
    double d=0.0;
    mfem::Vector rx; rx=x;
    for(int i=0;i<rx.Size();i++){
        rx(i)=x(i)-ct(i);
    }

    //sphere/ball
    {
        double r=0.5;
        double nr=std::sqrt(rx*rx);
        d=r-nr;
    }

    //cyl z
    {
        double r=0.3;
        double nr=std::sqrt(rx(0)*rx(0)+rx(1)*rx(1));
        double d1=nr-r;
        if(d1<d){
            d=d1;
        }
    }

    if(x.Size()==3)
    {
        //cyl x
        {
            double r=0.3;
            double nr=std::sqrt(rx(2)*rx(2)+rx(1)*rx(1));
            double d1=nr-r;
            if(d1<d){
                d=d1;
            }
        }
        //cyl y
        {
            double r=0.3;
            double nr=std::sqrt(rx(2)*rx(2)+rx(0)*rx(0));
            double d1=nr-r;
            if(d1<d){
                d=d1;
            }
        }
    }

    return d;
}

void CplxCompGradO(const mfem::Vector &x,  const mfem::Vector &ct, mfem::Vector & res)
{

    mfem::Vector rx; rx=x;
    for(int i=0;i<rx.Size();i++){
        rx(i)=x(i)-ct(i);
    }

    res=rx;
    double d=0.0;
    //sphere/ball
    {

        double r=0.5;
        double nr=std::sqrt(rx*rx);
        if(nr>(10.0*std::numeric_limits<double>::epsilon())){res/=nr;}
        else{res=0.0;}
        d=r-nr;
    }

    //cyl x
    {
        double r=0.3;
        double nr=std::sqrt(rx(0)*rx(0)+rx(1)*rx(1));
        double d1=nr-r;
        if(d1<d){
            d=d1;
            res=0.0;
            if(nr>(10.0*std::numeric_limits<double>::epsilon()))
            {
                res(0)=-rx(0);
                res(1)=-rx(1);
                res/=nr;
            }

        }
    }

    if(x.Size()==3){
        //cyl x
        {
            double r=0.3;
            double nr=std::sqrt(rx(2)*rx(2)+rx(1)*rx(1));
            double d1=nr-r;
            if(d1<d){
                d=d1;
                res=0.0;
                if(nr>(10.0*std::numeric_limits<double>::epsilon()))
                {
                    res(0)=0.0;
                    res(1)=-rx(1);
                    res(2)=-rx(2);
                    res/=nr;
                }
            }
        }
        //cyl y
        {
            double r=0.3;
            double nr=std::sqrt(rx(2)*rx(2)+rx(0)*rx(0));
            double d1=nr-r;
            if(d1<d){
                d=d1;
                res=0.0;
                if(nr>(10.0*std::numeric_limits<double>::epsilon()))
                {
                    res(0)=-rx(0);
                    res(1)=0.0;
                    res(2)=-rx(2);
                    res/=nr;
                }
            }
        }
    }
}

double CplxComp(const mfem::Vector &x)
{
    double d,d1;
    mfem::Vector ct; ct.SetSize(x.Size());
    ct=0.0;
    d=CplxCompO(x,ct);

    ct(0)=0.25;
    ct(1)=0.25;
    d1=CplxCompO(x,ct);
    if(d1>d){d=d1;}

    ct(0)=-0.25;
    ct(1)=-0.25;
    d1=CplxCompO(x,ct);
    if(d1>d){d=d1;}

    ct(0)=-0.25;
    ct(1)=0.5;
    d1=CplxCompO(x,ct);
    if(d1>d){d=d1;}

    ct(0)=0.25;
    ct(1)=-0.5;
    d1=CplxCompO(x,ct);
    if(d1>d){d=d1;}

    return d;
}


void CplxCompGrad(const mfem::Vector &x, mfem::Vector &res)
{
    mfem::Vector ct; ct.SetSize(x.Size());
    double d,d1;
    ct=0.0;
    ct(0)=0.2;
    d=CplxCompO(x,ct);
    CplxCompGradO(x,ct,res);

    ct(0)=0.25;
    ct(1)=0.25;
    d1=CplxCompO(x,ct);
    if(d1>d){
        d=d1;
        CplxCompGradO(x,ct,res);
    }

    ct(0)=-0.25;
    ct(1)=-0.25;
    d1=CplxCompO(x,ct);
    if(d1>d){
        d=d1;
        CplxCompGradO(x,ct,res);
    }

    ct(0)=-0.25;
    ct(1)=0.5;
    d1=CplxCompO(x,ct);
    if(d1>d){
        d=d1;
        CplxCompGradO(x,ct,res);
    }

    ct(0)=0.25;
    ct(1)=-0.5;
    d1=CplxCompO(x,ct);
    if(d1>d){
        d=d1;
        CplxCompGradO(x,ct,res);
    }

}


double TestDistFunc2D(const mfem::Vector &x)
{
    double rs=0.35;
    double rb=0.5;
    double x0=0.5;
    double y0=0.5;
    double dx=0.6;

    double d[5];
    double dd;
    for(int i=0;i<5;i++)
    {
        x0=0.5+i*dx;
        dd=(x[0]-x0)*(x[0]-x0)+(x[1]-y0)*(x[1]-y0);
        dd=std::sqrt(dd);
        if(dd<0.5*(rs+rb)){d[i]=dd-rs;}
        else{d[i]=rb-dd;}
    }

    for(int i=0;i<5;i++)
    {
        if(d[0]<d[i]){d[0]=d[i];}
    }

    return d[0];
}

double TestDistFunc3D(const mfem::Vector &x)
{
    double rs=0.35;
    double rb=0.5;
    double x0=0.5;
    double y0=0.5;
    double z0=0.5;
    double dz=0.6;

    double d[5];
    double dd;
    for(int i=0;i<5;i++)
    {
        z0=0.5+i*dz;
        dd=(x[0]-x0)*(x[0]-x0)+(x[1]-y0)*(x[1]-y0)+(x[2]-z0)*(x[2]-z0);
        dd=std::sqrt(dd);
        if(dd<0.5*(rs+rb)){d[i]=dd-rs;}
        else{d[i]=rb-dd;}
    }

    for(int i=0;i<5;i++)
    {
        if(d[0]<d[i]){d[0]=d[i];}
    }

    return d[0];
}


void DispFunction(const mfem::Vector &xx, mfem::Vector &uu)
{
    double a=3.0;
    double b=2.0;
    double c=3.0;
    double d=2.0;

    if(xx.Size()==2)
    {
        uu[0] = sin(a*xx[0])+cos(b*xx[1]);
        uu[1] = cos(c*xx[0]+d*xx[1]);
    }else{//size==3
        uu[0] = sin(a*xx[0])+cos(b*xx[1]);
        uu[1] = sin(a*xx[1])+cos(b*xx[2]);
        uu[2] = sin(c*xx[2])+cos(d*xx[0]);
    }

}

void TractionFunction(const mfem::Vector &xx, mfem::Vector &tr)
{

    tr.SetSize(xx.Size());
    double a=3.0;
    double b=2.0;
    double c=3.0;
    double d=2.0;

    double E=1.0;
    double nnu=0.3;

    //the center is at 0
    mfem::Vector nn; nn=xx;
    double nr=xx.Norml2();
    if(nr>5*std::numeric_limits<double>::epsilon()){
        nn/=nr;
    }else{nn=0.0;}

    if(xx.Size()==2)
    {
        tr[0] = (2.0*E/(2.0+2.0*nnu)*a*cos(a*xx[0])+E*nnu/(1.0+nnu)/(1.0-2.0*nnu)
        *(a*cos(a*xx[0])-d*sin(c*xx[0]+d*xx[1])))*nn[0]-E*(b*sin(b*xx[1])+c*sin(c*xx[0]
        +d*xx[1]))/(2.0+2.0*nnu)*nn[1];
        tr[1] = -E*(b*sin(b*xx[1])+c*sin(c*xx[0]+d*xx[1]))/(2.0+2.0*nnu)*nn[0]+(
        -2.0*E/(2.0+2.0*nnu)*d*sin(c*xx[0]+d*xx[1])+E*nnu/(1.0+nnu)/(1.0-2.0*nnu)*(a*
        cos(a*xx[0])-d*sin(c*xx[0]+d*xx[1])))*nn[1];

    }else
    {
        tr[0] = (2.0*E/(2.0+2.0*nnu)*a*cos(a*xx[0])+E*nnu/(1.0+nnu)/(1.0-2.0*nnu)
        *(a*cos(a*xx[0])+a*cos(a*xx[1])+c*cos(c*xx[2])))*nn[0]-E/(2.0+2.0*nnu)*b*sin(b*
        xx[1])*nn[1]-E/(2.0+2.0*nnu)*d*sin(d*xx[0])*nn[2];
        tr[1] = -E/(2.0+2.0*nnu)*b*sin(b*xx[1])*nn[0]+(2.0*E/(2.0+2.0*nnu)*a*cos(
        a*xx[1])+E*nnu/(1.0+nnu)/(1.0-2.0*nnu)*(a*cos(a*xx[0])+a*cos(a*xx[1])+c*cos(c*
        xx[2])))*nn[1]-E/(2.0+2.0*nnu)*b*sin(b*xx[2])*nn[2];
        tr[2] = -E/(2.0+2.0*nnu)*d*sin(d*xx[0])*nn[0]-E/(2.0+2.0*nnu)*b*sin(b*xx
        [2])*nn[1]+(2.0*E/(2.0+2.0*nnu)*c*cos(c*xx[2])+E*nnu/(1.0+nnu)/(1.0-2.0*nnu)*(a
        *cos(a*xx[0])+a*cos(a*xx[1])+c*cos(c*xx[2])))*nn[2];
    }
}

void VolForceFunction(const mfem::Vector &xx, mfem::Vector &ff)
{
    double a=3.0;
    double b=2.0;
    double c=3.0;
    double d=2.0;

    double E=1.0;
    double nnu=0.3;

    if(xx.Size()==2){
        ff[0] = 2.0*E/(2.0+2.0*nnu)*a*a*sin(a*xx[0])
                -E*nnu/(1.0+nnu)/(1.0-2.0*nnu)*(-a*a*sin(a*xx[0])
                -d*c*cos(c*xx[0]+d*xx[1]))
                +E*(b*b*cos(b*xx[1])+d*c*cos(c*xx[0]+d*xx[1]))/(2.0+2.0*nnu);
        ff[1] = E*c*c*cos(c*xx[0]+d*xx[1])/(2.0+2.0*nnu)
                +2.0*E/(2.0+2.0*nnu)*d*d*cos(c*xx[0]+d*xx[1])
                +E*nnu/(1.0+nnu)/(1.0-2.0*nnu)*d*d*cos(c*xx[0]+d*xx[1]);
    }else{//Size()=3
        ff[0] = 2.0*E/(2.0+2.0*nnu)*a*a*sin(a*xx[0])
                +E*nnu/(1.0+nnu)/(1.0-2.0*nnu)*a*a*sin(a*xx[0])
                +E/(2.0+2.0*nnu)*b*b*cos(b*xx[1]);
        ff[1] = 2.0*E/(2.0+2.0*nnu)*a*a*sin(a*xx[1])
                +E*nnu/(1.0+nnu)/(1.0-2.0*nnu)*a*a*sin(a*xx[1])
                +E/(2.0+2.0*nnu)*b*b*cos(b*xx[2]);
        ff[2] = E/(2.0+2.0*nnu)*d*d*cos(d*xx[0])
                +2.0*E/(2.0+2.0*nnu)*c*c*sin(c*xx[2])
                +E*nnu/(1.0+nnu)/(1.0-2.0*nnu)*c*c*sin(c*xx[2]);
    }

    //ff*=-1.0;
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
   /*
   {
      int ref_levels =
         (int)floor(log(1000./mesh.GetNE())/log(2.)/dim);
      for (int l = 0; l < ref_levels; l++)
      {
         mesh.UniformRefinement();
      }
   }*/


   // Define a parallel mesh by a partitioning of the serial mesh. Refine
   // this mesh further in parallel to increase the resolution. Once the
   // parallel mesh is defined, the serial mesh can be deleted.
   mfem::ParMesh pmesh(MPI_COMM_WORLD, mesh);
   mesh.Clear();
   {
       for(int l=0; l < par_ref_levels; l++){
           pmesh.UniformRefinement();
       }
   }
   pmesh.ExchangeFaceNbrData();
   pmesh.ExchangeFaceNbrNodes();

   mfem::FunctionCoefficient* distco;
   mfem::VectorFunctionCoefficient* gradco;
   /*
   if(dim==2){
       distco=new mfem::FunctionCoefficient(TestDistFunc2D);
   }else{
       distco=new mfem::FunctionCoefficient(TestDistFunc3D);
   }
   */
   distco=new mfem::FunctionCoefficient(Circ);
   gradco=new mfem::VectorFunctionCoefficient(dim,CircGrad);

   //distco=new mfem::FunctionCoefficient(CplxComp);
   //gradco=new mfem::VectorFunctionCoefficient(dim,CplxCompGrad);


   mfem::H1_FECollection fec(order,dim);
   mfem::ParFiniteElementSpace des(&pmesh,&fec,1); des.ExchangeFaceNbrData();
   mfem::ParGridFunction dist(&des);
   dist.ProjectCoefficient(*distco);
   dist.ExchangeFaceNbrData();


   mfem::ParFiniteElementSpace fes(&pmesh,&fec,dim,mfem::Ordering::byVDIM); fes.ExchangeFaceNbrData();
   mfem::ParNonlinearForm* nf=new mfem::ParNonlinearForm(&fes);

   mfem::ParGridFunction dispgf(&fes);
   mfem::VectorFunctionCoefficient dispc(dim, DispFunction);
   mfem::ShiftedVectorFunctionCoefficient shdispc(dim,DispFunction);
   dispgf.ProjectCoefficient(dispc);
   dispgf.ExchangeFaceNbrData();

   mfem::Vector force(dim); force=0.0; force(1)=1.0;
   //mfem::VectorConstantCoefficient volf(force);
   mfem::VectorFunctionCoefficient volf(dim,VolForceFunction);

   mfem::LinIsoElasticityCoefficient ecoef(1,0.3);


   mfem::Array<int> elm_markers;
   mfem::Array<int> fct_markers;
   mfem::Array<int> ess_tdof_list;

   {
       mfem::ElementMarker smarker(pmesh,false);
       //mfem::ElementMarker smarker(pmesh,true);
       smarker.SetLevelSetFunction(dist);
       smarker.MarkElements(elm_markers);
       smarker.MarkFaces(fct_markers);
       smarker.ListEssentialTDofs(elm_markers,fes,ess_tdof_list);

       //std::cout<<pmesh.GetNE()<<" "<<elm_markers.Size()<<std::endl;
       for(int i=0;i<pmesh.GetNE();i++){
           pmesh.SetAttribute(i,elm_markers[i]+1);
       }
       pmesh.ExchangeFaceNbrData();
   }

   //nf->AddDomainIntegrator(new mfem::MXElasticityIntegrator(&ecoef));

   {
       mfem::SBM2ELIntegrator* tin=new mfem::SBM2ELIntegrator(&pmesh,&fes,&ecoef,elm_markers,fct_markers);
       tin->SetDistance(distco,gradco);
       //tin->SetShiftOrder(order+1);
       tin->SetShiftOrder(order);
       //tin->SetShiftOrder(1);
       tin->SetElasticityCoefficient(ecoef);
       tin->SetBdrCoefficient(&shdispc);

       //std::cout<<"order="<<order<<std::endl;
       double Ci=order*order;
       double eta=1.0/2;
       double EE=1.0;
       double nnu=0.3;
       double mmu=EE/(1.0+nnu);
       double kappa=EE/(3.0*(1.0-2.0*nnu));
       double lam_max=2*mmu;
       if(lam_max<3.0*kappa){lam_max=3.0*kappa;}

       tin->SetPenalization(2.0*Ci*eta*lam_max);
       //tin->SetPenalization(100.0);
       nf->AddDomainIntegrator(tin);
       //mfem::NLElasticityIntegrator* ein=new mfem::NLElasticityIntegrator(ecoef);
       //nf->AddDomainIntegrator(ein);
       mfem::NLVolForceIntegrator* fin=new mfem::NLVolForceIntegrator(volf);
       nf->AddDomainIntegrator(fin);
   }

   //nf->AddDomainIntegrator(new mfem::NLElasticityIntegrator(&ecoef));


   mfem::Vector sol; sol.SetSize(fes.GetTrueVSize()); sol=0.0;
   //dispgf.GetTrueDofs(sol);

   nf->SetGradientType(mfem::Operator::Type::Hypre_ParCSR);
   //nf->SetGradientType(mfem::Operator::Type::PETSC_MATAIJ);

   mfem::Operator& A=nf->GetGradient(sol);


   /*
   {
       std::fstream out("full.mat",std::ios::out);
       A.PrintMatlab(out);
       out.close();
   }
   */



   mfem::Vector rhs; rhs.SetSize(fes.GetTrueVSize()); rhs=0.0;
   nf->Mult(sol,rhs);
   rhs*=-1.0;

   //check markers
   int mbla=0;
   for(int i=0;i<elm_markers.Size();i++){
       if(elm_markers[i]==mfem::ElementMarker::SBElementType::INSIDE)
       {
           mbla++;
       }
   }
   int rbla=0;
   MPI_Allreduce(&mbla,&rbla,1,MPI_INT,MPI_SUM,MPI_COMM_WORLD);
   if(myrank==0){
       std::cout<<"rlar="<<rbla<<std::endl;
   }


   //check dofs
   int bla=ess_tdof_list.Size();
   int blar=0;
   MPI_Allreduce(&bla,&blar,1,MPI_INT,MPI_SUM,MPI_COMM_WORLD);
   if(myrank==0){
       std::cout<<"blar="<<blar<<std::endl;
   }


   //ess_tdof_list.Print(std::cout);

   mfem::HypreParMatrix* M=static_cast<mfem::HypreParMatrix*>(&A);
   mfem::HypreParMatrix* Ae=M->EliminateRowsCols(ess_tdof_list);
   //M->EliminateBC(*Ae,ess_tdof_list, sol, rhs);
   M->EliminateZeroRows();
   delete Ae;


   /*
   {
       std::fstream out("elm.mat",std::ios::out);
       M->PrintMatlab(out);
       out.close();
   }
   */


   /*
   mfem::GMRESSolver *ls=new mfem::GMRESSolver(pmesh.GetComm());
   ls->SetPrintLevel(print_level);
   ls->SetAbsTol(1e-12);
   ls->SetRelTol(1e-7);
   ls->SetMaxIter(1000);
   ls->SetOperator(A);
   ls->Mult(rhs,sol);
   */



   mfem::HypreBoomerAMG *prec= new mfem::HypreBoomerAMG();
   //mfem::CGSolver *ls=new mfem::CGSolver(pmesh.GetComm());
   mfem::GMRESSolver *ls=new mfem::GMRESSolver(pmesh.GetComm());
   prec->SetSystemsOptions(pmesh.Dimension());
   prec->SetElasticityOptions(&fes);
   ls->SetPrintLevel(print_level);
   ls->SetAbsTol(1e-12);
   ls->SetRelTol(1e-12);
   ls->SetMaxIter(10000);
   ls->SetPreconditioner(*prec);
   prec->SetPrintLevel(print_level);
   ls->SetOperator(*M);
   ls->Mult(rhs,sol);
   delete ls;
   delete prec;

   {
       double nrsol=mfem::ParNormlp(sol,2,MPI_COMM_WORLD);
       double nrrhs=mfem::ParNormlp(rhs,2,MPI_COMM_WORLD);
       if(myrank==0){
           std::cout<<"|sol|="<<nrsol<<std::endl;
           std::cout<<"|res|="<<nrrhs<<std::endl;
       }
   }


   //mfem::PetscLinearSolve *ls=new mfem::PetscLinearSolver(pmesh->GetComm());
   //ls->SetOperator(A);











   {
       mfem::ParGridFunction dispsol(&fes);
       dispsol.SetFromTrueDofs(sol);

       mfem::ParGridFunction proc(&des);
       mfem::ConstantCoefficient ProcCoeff((double)(myrank));
       proc.ProjectCoefficient(ProcCoeff);

       mfem::Array<int> activel;activel.SetSize(elm_markers.Size()); activel=0;
       for(int i=0;i<elm_markers.Size();i++){
           if(elm_markers[i]==mfem::ElementMarker::SBElementType::INSIDE){activel[i]=1;}
       }

       double l2err=dispsol.ComputeL2Error(dispc,nullptr,&activel);
       if(myrank==0){
        std::cout<<"L2err="<<l2err<<std::endl;
       }


       //std::cout<<"L1:"<<dispsol.ComputeL1Error(dispc)<<std::endl;

       mfem::ParaViewDataCollection paraview_dc("Elast", &pmesh);
       paraview_dc.SetPrefixPath("ParaView");
       paraview_dc.SetLevelsOfDetail(order+4);
       paraview_dc.SetDataFormat(mfem::VTKFormat::BINARY);
       paraview_dc.SetHighOrderOutput(true);
       paraview_dc.SetCycle(0);
       paraview_dc.SetTime(0.0);
       paraview_dc.RegisterField("dist",&dist);
       paraview_dc.RegisterField("disp",&dispgf);
       paraview_dc.RegisterField("solu",&dispsol);
       paraview_dc.RegisterField("proc",&proc);
       paraview_dc.Save();
   }


   delete nf;
   delete gradco;
   delete distco;
   mfem::MFEMFinalizePetsc();
   MPI_Finalize();
   return 0;

}
