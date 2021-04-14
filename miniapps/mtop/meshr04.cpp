#include "mfem.hpp"
#include <fstream>
#include <iostream>

#include "mtop_integrators.hpp"

class ThresholdCoefficient : public mfem::Coefficient
{
private:
    double eta;
    double beta;
    const mfem::GridFunction &u;

public:

    ThresholdCoefficient(const mfem::GridFunction &u_gf, double etac, double betac):u(u_gf)
    {
        eta=etac;
        beta=betac;
    }

    virtual
    double Eval(mfem::ElementTransformation &T, const mfem::IntegrationPoint &ip) override
    {
        T.SetIntPoint(&ip);
        double val=u.GetValue(T,ip);
        return mfem::PointHeavisideProj::Project(val,eta,beta);
    }

};

class DiffusionCoefficient : public mfem::Coefficient
{
private:
    double eta;
    double beta;
    double rho_min;
    const mfem::GridFunction* u;

public:

    DiffusionCoefficient(const mfem::GridFunction &u_gf, double etac, double betac, double rh_min):u(&u_gf)
    {
        eta=etac;
        beta=betac;
        rho_min=rh_min;
    }

    virtual
    double Eval(mfem::ElementTransformation &T, const mfem::IntegrationPoint &ip) override
    {
        T.SetIntPoint(&ip);
        double val=u->GetValue(T,ip);
        double pr=mfem::PointHeavisideProj::Project(val,eta,beta);
        double rez=(1.0-rho_min)*std::pow(pr,1.0)+rho_min;
        //if(pr<0.0){std::cout<<"pr="<<pr<<std::endl;}
        return rez;
    }

    void SetGridFunction (const mfem::GridFunction& u_gf)
    {
        u=&u_gf;
    }

};



//Gyroid function

double Gyroid(const mfem::Vector &xx)
{
    const double period = 8.0 * M_PI;
    double x=xx[0]*period;
    double y=xx[1]*period;
    double z=0.0;
    if(xx.Size()==3)
    {
       z=xx[2]*period;
    }
    double val=std::sin(x)*std::cos(y) +
           std::sin(y)*std::cos(z) +
           std::sin(z)*std::cos(x);

    z=z+0.5*M_PI;
    double va1=std::sin(x)*std::cos(y) +
           std::sin(y)*std::cos(z) +
           std::sin(z)*std::cos(x);

    double rez=0.0;
    if(fabs(val)<0.2)
    {
        rez=1.0;
    }

    if(fabs(va1)<0.2)
    {
        rez=1.0;
    }

    return rez;
}


int main(int argc, char *argv[])
{
   // 1. Initialize MPI.
   int num_procs, myid;
   MPI_Init(&argc, &argv);
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
   MPI_Comm_rank(MPI_COMM_WORLD, &myid);

   // 2. Parse command-line options.
   const char *mesh_file = "../data/star.mesh";
   int order = 1;
   bool pa = false;
   const char *device_config = "cpu";
   bool nc_simplices = false;
   int max_dofs = 200000;
   bool restart = false;
   bool visualization = true;

   double eta=0.5;
   double beta=4.0;
   double rho_min=1e-3;
   double filter_r=0.01;
   int filter_o=2;
   int objective=1;

   double newton_rel_tol = 1e-7;
   double newton_abs_tol = 1e-12;

   mfem::OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree).");
   args.AddOption(&pa, "-pa", "--partial-assembly", "-no-pa",
                  "--no-partial-assembly", "Enable Partial Assembly.");
   args.AddOption(&device_config, "-d", "--device",
                  "Device configuration string, see Device::Configure().");
   args.AddOption(&nc_simplices, "-ns", "--nonconforming-simplices",
                  "-cs", "--conforming-simplices",
                  "For simplicial meshes, enable/disable nonconforming"
                  " refinement");
   args.AddOption(&max_dofs, "-md", "--max-dofs",
                  "Stop after reaching this many degrees of freedom.");
   args.AddOption(&restart, "-res", "--restart", "-no-res", "--no-restart",
                  "Restart computation from the last checkpoint.");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");

   args.AddOption(&filter_o, "-of", "--order-filter",
                  "Finite element order for the PDE filter (>=1).");
   args.AddOption(&rho_min, "-rmin", "--rho-min",
                  "Minimal value of rho.");
   args.AddOption(&eta, "-eta", "--eta",
                  "Threshold value.");
   args.AddOption(&beta, "-beta", "--beta",
                  "Projection value.");
   args.AddOption(&objective, "-obj", "--obj",
                  "Objective:1) compliance; 2) p-norm(p=8).");
   args.AddOption(&filter_r, "-rf", "--radius-filter",
                  "PDE filter radius.");


   args.Parse();
   if (!args.Good())
   {
      if (myid == 0)
      {
         args.PrintUsage(std::cout);
      }
      MPI_Finalize();
      return 1;
   }

   if (myid == 0)
   {
      args.PrintOptions(std::cout);
   }

   mfem::ParMesh *pmesh;

   mfem::Mesh mesh(mesh_file, 1, 1);
   mesh.EnsureNCMesh();
   int dim = mesh.Dimension();

   //    Refine the serial mesh on all processors to increase the resolution. In
   //    this example we do 'ref_levels' of uniform refinement. We choose
   //    'ref_levels' to be the largest number that gives a final mesh with no
   //    more than 10,000 elements.
   {
      int ref_levels =
         (int)floor(log(100000./mesh.GetNE())/log(2.)/dim);
      for (int l = 0; l < ref_levels; l++)
      {
         mesh.UniformRefinement();
      }
   }

   pmesh = new mfem::ParMesh(MPI_COMM_WORLD, mesh);
   //pmesh->UniformRefinement();
   //pmesh->UniformRefinement();
   MFEM_VERIFY(pmesh->bdr_attributes.Size() > 0,
               "Boundary attributes required in the mesh.");
   mfem::Array<int> ess_bdr(pmesh->bdr_attributes.Max());
   ess_bdr = 1;

   // 8. Define a finite element space on the mesh. The polynomial order is
   //    one (linear) by default, but this can be changed on the command line.
   mfem::H1_FECollection fec(order, dim);
   mfem::ParFiniteElementSpace fespace(pmesh, &fec);

   mfem::H1_FECollection pfec(filter_o, dim);
   mfem::ParFiniteElementSpace pfes(pmesh, &pfec);

   mfem::ParBilinearForm a(&fespace);
   mfem::ParLinearForm b(&fespace);
   mfem::ParGridFunction gfdiff(&pfes);
   gfdiff=1.0;
   mfem::ConstantCoefficient one(1.0);
   DiffusionCoefficient rmc(gfdiff,eta,beta,rho_min);
   mfem::Coefficient* densco=new mfem::FunctionCoefficient(Gyroid);

   // Define the Heat source
   //mfem::ConstantCoefficient* loadco=new mfem::ConstantCoefficient(1.0);
   mfem::Coefficient* loadco=new mfem::FunctionCoefficient(Gyroid);

   mfem::BilinearFormIntegrator *integ = new mfem::DiffusionIntegrator(rmc);

   a.AddDomainIntegrator(integ);
   //integ = new mfem::DiffusionIntegrator(dfcdiff);
   //a.AddDomainIntegrator(integ);
   b.AddDomainIntegrator(new mfem::DomainLFIntegrator(*loadco));

   // 10. The solution vector x and the associated finite element grid function
   //     will be maintained over the AMR iterations. We initialize it to zero.
   mfem::ParGridFunction x(&fespace);
   x = 0;

   mfem::Array<mfem::ParFiniteElementSpace*> asfes;
   asfes.Append(&fespace);

   // 12. Set up an error estimator. Here we use the Zienkiewicz-Zhu estimator
   //     with L2 projection in the smoothing step to better handle hanging
   //     nodes and parallel partitioning. We need to supply a space for the
   //     discontinuous flux (L2) and a space for the smoothed flux (H(div) is
   //     used here).
   mfem::L2_FECollection flux_fec(order, dim);
   mfem::ParFiniteElementSpace flux_fes(pmesh, &flux_fec, dim);
   mfem::RT_FECollection smooth_flux_fec(order-1, dim);
   mfem::ParFiniteElementSpace smooth_flux_fes(pmesh, &smooth_flux_fec);
   // Another possible option for the smoothed flux space:
   // H1_FECollection smooth_flux_fec(order, dim);
   // ParFiniteElementSpace smooth_flux_fes(pmesh, &smooth_flux_fec, dim);
   mfem::L2ZienkiewiczZhuEstimator estimator(*integ, x, flux_fes, smooth_flux_fes);

   // 13. A refiner selects and refines elements based on a refinement strategy.
   //     The strategy here is to refine elements with errors larger than a
   //     fraction of the maximum element error. Other strategies are possible.
   //     The refiner will call the given error estimator.
   mfem::ThresholdRefiner refiner(estimator);
   refiner.SetTotalErrorFraction(0.5);

   mfem::ParaViewDataCollection *dacol=new mfem::ParaViewDataCollection("ParHeat",pmesh);
   dacol->SetLevelsOfDetail(order);

   dacol->RegisterField("sol", &x);
   dacol->RegisterField("dens", &gfdiff);
   dacol->SetTime(0.0);
   dacol->SetCycle(0);
   dacol->Save();
   // 14. The main AMR loop. In each iteration we solve the problem on the
   //     current mesh, visualize the solution, and refine the mesh.
   for (int it = 0; ; it++)
   {
      mfem::PDEFilterTO filter(*pmesh,filter_r,filter_o);
      filter.Filter(*densco, gfdiff); //gfdiff will automaticaly update the diffusion coefficient rmc



      HYPRE_Int global_dofs = fespace.GlobalTrueVSize();
      /*
      if (myid == 0)
      {
         std::cout << "\nAMR iteration " << it << std::endl;
         std::cout << "Number of unknowns: " << global_dofs << std::endl;
      }
      */

      // 15. Assemble the right-hand side and determine the list of true
      //     (i.e. parallel conforming) essential boundary dofs.
      mfem::Array<int> ess_tdof_list;
      fespace.GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
      b.Assemble();

      // 16. Assemble the stiffness matrix. Note that MFEM doesn't care at this
      //     point that the mesh is nonconforming and parallel.  The FE space is
      //     considered 'cut' along hanging edges/faces, and also across
      //     processor boundaries.
      a.Assemble();

      // 17. Create the parallel linear system: eliminate boundary conditions.
      //     The system will be solved for true (unconstrained/unique) DOFs only.
      mfem::OperatorPtr A;
      mfem::Vector B, X;

      const int copy_interior = 1;
      a.FormLinearSystem(ess_tdof_list, x, b, A, X, B, copy_interior);


      // 18. Solve the linear system A X = B.
      //     * With full assembly, use the BoomerAMG preconditioner from hypre.
      //     * With partial assembly, use a diagonal preconditioner.
      mfem::Solver *M = NULL;
      {
        mfem::HypreBoomerAMG *amg = new mfem::HypreBoomerAMG;
        amg->SetPrintLevel(0);
        M = amg;
      }

      mfem::CGSolver cg(MPI_COMM_WORLD);
      cg.SetRelTol(newton_rel_tol/10);
      cg.SetAbsTol(newton_abs_tol/10);
      cg.SetMaxIter(2000);
      cg.SetPrintLevel(0); // print the first and the last iterations only
      cg.SetPreconditioner(*M);
      cg.SetOperator(*A);
      cg.Mult(B, X);
      delete M;

      // 19. Switch back to the host and extract the parallel grid function
      //     corresponding to the finite element approximation X. This is the
      //     local solution on each processor.
      a.RecoverFEMSolution(X, b, x);

      dacol->SetTime(it+1);
      dacol->SetCycle(it+1);
      dacol->Save();

      mfem::ParBlockNonlinearForm* ob=new mfem::ParBlockNonlinearForm(asfes);
      if(objective==1){
          ob->AddDomainIntegrator(new mfem::ThermalComplianceIntegrator(*loadco));
      }else
      {
          ob->AddDomainIntegrator(new mfem::DiffusionObjIntegrator(4));
      }
      // Compute the objective
      double obj=ob->GetEnergy(x);
      if(myid==0){
          //std::cout<<"it= "<<it<<" total_dofs= "<<fespace.GlobalTrueVSize()<<" Objective= "<<obj<<std::endl;
          std::cout<<fespace.GlobalTrueVSize()<<" "<<obj<<std::endl;
      }
      delete ob;


      if (global_dofs >= max_dofs)
      {
         if (myid == 0)
         {
            std::cout << "Reached the maximum number of dofs. Stop." << std::endl;
         }
         break;
      }

      // 21. Call the refiner to modify the mesh. The refiner calls the error
      //     estimator to obtain element errors, then it selects elements to be
      //     refined and finally it modifies the mesh. The Stop() method can be
      //     used to determine if a stopping criterion was met.

      pmesh->UniformRefinement();
      // 22. Update the finite element space (recalculate the number of DOFs,
      //     etc.) and create a grid function update matrix. Apply the matrix
      //     to any GridFunctions over the space. In this case, the update
      //     matrix is an interpolation matrix so the updated GridFunction will
      //     still represent the same function as before refinement.
      fespace.Update();
      x.Update();
      pfes.Update();
      gfdiff.Update();

      // 23. Load balance the mesh, and update the space and solution. Currently
      //     available only for nonconforming meshes.
      if (pmesh->Nonconforming())
      {
         pmesh->Rebalance();

         // Update the space and the GridFunction. This time the update matrix
         // redistributes the GridFunction among the processors.
         fespace.Update();
         x.Update();
         pfes.Update();
         gfdiff.Update();
      }

      // 24. Inform also the bilinear and linear forms that the space has
      //     changed.
      a.Update();
      b.Update();

   }
   delete dacol;

   delete densco;
   delete loadco;
   delete pmesh;
   MPI_Finalize();
   return 0;
}


