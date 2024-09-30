#include "mfem.hpp"
#include <fstream>
#include <iostream>


class MyCoeff:public mfem::Coefficient
{
public:
    MyCoeff(double period_=M_PI)
    {
        period=period_;
    }

    virtual
    double Eval(mfem::ElementTransformation &T, const mfem::IntegrationPoint &ip)
    {

        double x[3]={0.0,0.0,0.0};
        mfem::Vector transip(x, T.GetSpaceDim());
        T.Transform(ip, transip);
        return sin(x[0]*period)*cos(x[1]*period/2.0)+sin(x[1]*period)*cos(x[2]*period/2.0);

    }
private:
    double period;
};




int main(int argc, char *argv[])
{
    // 1. Initialize MPI and HYPRE.
    int nprocs, myid;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);


    mfem::Hypre::Init();

    // 2. Parse command-line options.
    const char *mesh_file = "../data/star.mesh";
    int order = 1;
    bool visualization = true;
    int ser_ref_levels = 0;
    int par_ref_levels = 1;
    double period=M_PI;

    mfem::OptionsParser args(argc, argv);
    args.AddOption(&mesh_file, "-m", "--mesh",
                   "Mesh file to use.");
    args.AddOption(&order, "-o", "--order",
                   "Finite element order (polynomial degree) or -1 for"
                   " isoparametric space.");
    args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                   "--no-visualization",
                   "Enable or disable GLVis visualization.");
    args.AddOption(&ser_ref_levels,
                   "-rs",
                   "--refine-serial",
                   "Number of times to refine the mesh uniformly in serial.");
    args.AddOption(&par_ref_levels,
                   "-rp",
                   "--refine-parallel",
                   "Number of times to refine the mesh uniformly in parallel.");
    args.AddOption(&period,
                   "-pp",
                   "--period",
                   "Period of the coefficient");
    args.Parse();
    if (!args.Good())
    {
       if (myid == 0)
       {
          args.PrintUsage(std::cout);
       }
       return 1;
    }
    if (myid == 0)
    {
       args.PrintOptions(std::cout);
    }

    mfem::Mesh mesh(mesh_file, 1, 1);
    int dim = mesh.Dimension();

    {
       for (int l = 0; l < ser_ref_levels; l++)
       {
          mesh.UniformRefinement();
       }
    }

    mfem::ParMesh pmesh(MPI_COMM_WORLD, mesh);
    mesh.Clear();
    {
       for (int l = 0; l < par_ref_levels; l++)
       {
          pmesh.UniformRefinement();
       }
    }

   mfem::FiniteElementCollection *fec;
   fec = new mfem::H1_FECollection(order, dim);

   mfem::ParFiniteElementSpace fespace(&pmesh, fec);
   HYPRE_BigInt size = fespace.GlobalTrueVSize();
   if (myid == 0)
   {
      std::cout << "Number of finite element unknowns: " << size << std::endl;
   }

   mfem::Array<int> ess_tdof_list;

   mfem::ParLinearForm b(&fespace);
   mfem::ConstantCoefficient one(1.0);
   MyCoeff ff(period);
   b.AddDomainIntegrator(new mfem::DomainLFIntegrator(ff));
   b.Assemble();

   mfem::ParGridFunction x(&fespace);
   x = 0.0;

   mfem::ParGridFunction o(&fespace);
   o.ProjectCoefficient(ff);


   mfem::ParBilinearForm a(&fespace);
   a.AddDomainIntegrator(new mfem::MassIntegrator(one));
   a.Assemble();

   mfem::OperatorPtr A;
   mfem::Vector B, X;
   a.FormLinearSystem(ess_tdof_list, x, b, A, X, B);



   mfem::Solver *prec = new mfem::HypreBoomerAMG();
   mfem::CGSolver cg(MPI_COMM_WORLD);
   cg.SetRelTol(1e-12);
   cg.SetMaxIter(2000);
   cg.SetPrintLevel(1);
   cg.SetPreconditioner(*prec);
   cg.SetOperator(*A);
   cg.Mult(B, X);
   delete prec;

   a.RecoverFEMSolution(X, b, x);

   {
       mfem::ParaViewDataCollection paraview_dc("L2_proj",&pmesh);
       paraview_dc.SetPrefixPath("ParaView");
       paraview_dc.SetLevelsOfDetail(order);
       paraview_dc.SetDataFormat(mfem::VTKFormat::BINARY);
       paraview_dc.SetHighOrderOutput(true);
       paraview_dc.SetCycle(0);
       paraview_dc.SetTime(0.0);
       paraview_dc.RegisterField("proj",&x);
       paraview_dc.RegisterField("orig",&o);
       paraview_dc.Save();
   }



   double err=x.ComputeL2Error(ff);
   if(myid==0){
       std::cout<<"err="<<err<<std::endl;
   }

   delete fec;


   MPI_Finalize();
   return 0;
}



