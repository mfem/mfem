#include "mfem.hpp"
#include <fstream>
#include <iostream>

#include "elasticity.hpp"
#include "../shifted/marking.hpp"


double Circ(const mfem::Vector &x)
{
    double r=0.9;
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
   /*
   {
      int ref_levels =
         (int)floor(log(10000./mesh.GetNE())/log(2.)/dim);
      for (int l = 0; l < ref_levels; l++)
      {
         mesh.UniformRefinement();
      }
   }
   */

   // Define a parallel mesh by a partitioning of the serial mesh. Refine
   // this mesh further in parallel to increase the resolution. Once the
   // parallel mesh is defined, the serial mesh can be deleted.
   mfem::ParMesh pmesh(MPI_COMM_WORLD, mesh);
   mesh.Clear();

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

   mfem::H1_FECollection fec(order,dim);
   mfem::ParFiniteElementSpace des(&pmesh,&fec,1);
   mfem::ParGridFunction dist(&des);
   dist.ProjectCoefficient(*distco);

   mfem::Array<int> elm_markers;
   mfem::Array<int> fct_markers;

   {
       mfem::ShiftedFaceMarker smarker(pmesh,des,false);
       smarker.MarkElements(dist,elm_markers);
       smarker.MarkFaces(elm_markers,fct_markers);
   }

   mfem::ParFiniteElementSpace fes(&pmesh,&fec,dim,mfem::Ordering::byVDIM);
   mfem::ParNonlinearForm* nf=new mfem::ParNonlinearForm(&fes);

   mfem::LinIsoElasticityCoefficient ecoef(1,0.3);

   //nf->AddDomainIntegrator(new mfem::MXElasticityIntegrator(&ecoef));

   {
       mfem::SBM2ELIntegrator* tin=new mfem::SBM2ELIntegrator(&pmesh,&fes,&ecoef,elm_markers,fct_markers);
       //mfem::FCElasticityIntegrator* tin=new mfem::FCElasticityIntegrator(&pmesh,&fes,&ecoef,elm_markers,fct_markers);
       tin->SetDistance(distco,gradco);
       tin->SetShiftOrder(2);
       tin->SetElasticityCoefficient(ecoef);
       nf->AddDomainIntegrator(tin);
   }

   //nf->AddDomainIntegrator(new mfem::NLElasticityIntegrator(&ecoef));

   mfem::Vector sol; sol.SetSize(fes.GetTrueVSize()); sol=0.0;

   nf->SetGradientType(mfem::Operator::Type::Hypre_ParCSR);


   mfem::Operator& A=nf->GetGradient(sol);
   {
       std::fstream out("full.mat",std::ios::out);
       A.PrintMatlab(out);
       out.close();
   }

   {
       mfem::ParaViewDataCollection paraview_dc("Elast", &pmesh);
       paraview_dc.SetPrefixPath("ParaView");
       paraview_dc.SetLevelsOfDetail(order);
       paraview_dc.SetDataFormat(mfem::VTKFormat::BINARY);
       paraview_dc.SetHighOrderOutput(true);
       paraview_dc.SetCycle(0);
       paraview_dc.SetTime(0.0);
       paraview_dc.RegisterField("dist",&dist);
       paraview_dc.Save();
   }


   delete nf;
   delete gradco;
   delete distco;

   MPI_Finalize();
   return 0;

}
