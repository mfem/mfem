//                       MFEM Example 19 - Parallel Version
//
// Compile with: make ex19p
//
// Sample runs:
//    mpirun -np 2 ex19p -m ../data/beam-quad.mesh
//    mpirun -np 2 ex19p -m ../data/beam-tri.mesh
//    mpirun -np 2 ex19p -m ../data/beam-hex.mesh
//    mpirun -np 2 ex19p -m ../data/beam-tet.mesh
//    mpirun -np 2 ex19p -m ../data/beam-wedge.mesh
//
// Description:  This examples solves a quasi-static incompressible nonlinear
//               elasticity problem of the form 0 = H(x), where H is an
//               incompressible hyperelastic model and x is a block state vector
//               containing displacement and pressure variables. The geometry of
//               the domain is assumed to be as follows:
//
//                                 +---------------------+
//                    boundary --->|                     |<--- boundary
//                    attribute 1  |                     |     attribute 2
//                    (fixed)      +---------------------+     (fixed, nonzero)
//
//               The example demonstrates the use of block nonlinear operators
//               (the class RubberOperator defining H(x)) as well as a nonlinear
//               Newton solver for the quasi-static problem. Each Newton step
//               requires the inversion of a Jacobian matrix, which is done
//               through a (preconditioned) inner solver. The specialized block
//               preconditioner is implemented as a user-defined solver.
//
//               We recommend viewing examples 2, 5, and 10 before viewing this
//               example.

#include "ex71.hpp"

int main(int argc, char *argv[])
{
   // 1. Parse command-line options
   const char *mesh_file = "../data/beam-tet.mesh";
   int ser_ref_levels = 0;
   int order = 1;
   bool visualization = true;
   double newton_rel_tol = 1e-4;
   double newton_abs_tol = 1e-6;
   int newton_iter = 500;
   int print_level = 0;
   double pp = 2.0;
   int integrator=0;
   mfem::StopWatch* timer=new mfem::StopWatch();

   mfem::OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&ser_ref_levels, "-rs", "--refine-serial",
                  "Number of times to refine the mesh uniformly in serial.");
   args.AddOption(&order, "-o", "--order",
                  "Order (degree) of the finite elements.");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&newton_rel_tol, "-rel", "--relative-tolerance",
                  "Relative tolerance for the Newton solve.");
   args.AddOption(&newton_abs_tol, "-abs", "--absolute-tolerance",
                  "Absolute tolerance for the Newton solve.");
   args.AddOption(&newton_iter, "-it", "--newton-iterations",
                  "Maximum iterations for the Newton solve.");
   args.AddOption(&pp, "-pp", "--power-parameter",
                  "Power parameter (>=2.0) for the p-Laplacian.");
   args.AddOption((&print_level),"-prt","--print-level",
                  "Print level.");
   args.AddOption(&integrator, "-int","--integrator",
                  "Integrator 0: standard; 1: AD uaing energy; 2: AD using gradients");

   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(std::cout);
      return 1;
   }
   args.PrintOptions(std::cout);


   // 2. Read the (serial) mesh from the given mesh file.
   mfem::Mesh *mesh = new mfem::Mesh(mesh_file, 1, 1);
   int dim = mesh->Dimension();

   // 3. Refine the mesh in serial to increase the resolution. In this example
   //    we do 'ser_ref_levels' of uniform refinement, where 'ser_ref_levels' is
   //    a command-line parameter.
   for (int lev = 0; lev < ser_ref_levels; lev++)
   {
      mesh->UniformRefinement();
   }

   // 4. Define the power parameter for the p-Laplacian and all other
   //    coefficients
   mfem::ConstantCoefficient c_pp(pp);
   mfem::ConstantCoefficient load(1.000000000);
   mfem::ConstantCoefficient c_ee(0.000000001);

   // 5. Define the finite element spaces for the solution
   mfem::H1_FECollection fec(order,dim);
   mfem::FiniteElementSpace fespace(mesh,&fec,1,mfem::Ordering::byVDIM);
   int glob_size=fespace.GetTrueVSize();
   std::cout << "Number of finite element unknowns: " << glob_size << std::endl;

   // 6. Define the Dirichlet conditions
   mfem::Array<int> ess_bdr(mesh->bdr_attributes.Max());
   ess_bdr = 1;

   // 7. Define the nonlinear form
   mfem::NonlinearForm* nf=new mfem::NonlinearForm(&fespace);

   // 8. Define the solution vector x
   mfem::GridFunction x(&fespace);
   x = 0.0;
   mfem::Vector tv(fespace.GetTrueVSize());
   mfem::Vector sv(fespace.GetTrueVSize());
   tv=0.0;
   sv=0.0;

   // 9. Define ParaView DataCollection
   mfem::ParaViewDataCollection *dacol=new mfem::ParaViewDataCollection("pLap",mesh);
   dacol->SetLevelsOfDetail(order);
   dacol->RegisterField("sol",&x);


   // 11. Set domain integrators - start with linear diffusion
   {
       // the default power coefficient is 2.0
       mfem::ConstantCoefficient lpp(2.0);
       if(integrator==0)
       {
            nf->AddDomainIntegrator(new mfem::pLaplace(lpp,c_ee,load));
       }else
       if(integrator==1)
       {
           nf->AddDomainIntegrator(new mfem::pLaplaceAD(lpp,c_ee,load));
       }
       nf->SetEssentialBC(ess_bdr);
       // compute the energy
       double energy=nf->GetEnergy(tv);
       std::cout<<"[2] The total energy of the system is E="<<energy<<std::endl;
       // time the assembly
       timer->Clear();
       timer->Start();
       mfem::Operator &op=nf->GetGradient(sv);
       timer->Stop();
       std::cout<<"[2] The assembly time is: "<<timer->RealTime()<<std::endl;
       mfem::Solver *prec;
#ifdef MFEM_USE_SUITESPARSE
       prec=new mfem::UMFPackSolver();
#else
       prec=new mfem::GSSmoother();
#endif
       mfem::GMRESSolver *j_gmres = new mfem::GMRESSolver();
       j_gmres->SetRelTol(1e-7);
       j_gmres->SetAbsTol(1e-15);
       j_gmres->SetMaxIter(500);
       j_gmres->SetPrintLevel(print_level);
       j_gmres->SetPreconditioner(*prec);

       mfem::NewtonSolver* ns;
       ns=new mfem::NewtonSolver();
       ns->iterative_mode = true;
       ns->SetSolver(*j_gmres);
       ns->SetOperator(*nf);
       ns->SetPrintLevel(print_level);
       ns->SetRelTol(1e-6);
       ns->SetAbsTol(1e-12);
       ns->SetMaxIter(10);
       //solve the problem
       timer->Clear();
       timer->Start();
       ns->Mult(tv, sv);
       timer->Stop();
       std::cout<<"Time for the NewtonSolver: "<<timer->RealTime()<<std::endl;

       energy=nf->GetEnergy(sv);
       std::cout<<"[pp=2] The total energy of the system is E="<<energy<<std::endl;

       delete ns;
       delete j_gmres;
       delete prec;

       x.SetFromTrueDofs(sv);
       dacol->SetTime(2.0);
       dacol->SetCycle(2);
       dacol->Save();
   }

   // 12. Continue with powers higher than 2
   for(int i=3;i<pp;i++)
   {
       delete nf;
       nf=new mfem::NonlinearForm(&fespace);
       mfem::ConstantCoefficient lpp((double)i);
       if(integrator==0)
       {
            nf->AddDomainIntegrator(new mfem::pLaplace(lpp,c_ee,load));
       }else
       if(integrator==1)
       {
           nf->AddDomainIntegrator(new mfem::pLaplaceAD(lpp,c_ee,load));
       }
       nf->SetEssentialBC(ess_bdr);
       // compute the energy
       double energy=nf->GetEnergy(sv);
       std::cout<<"[pp="<<i<<"] The total energy of the system is E="<<energy<<std::endl;
       // time the assembly
       timer->Clear();
       timer->Start();
       mfem::Operator &op=nf->GetGradient(sv);
       timer->Stop();
       std::cout<<"[pp="<<i<<"] The assembly time is: "<<timer->RealTime()<<std::endl;
       mfem::Solver *prec;
#ifdef MFEM_USE_SUITESPARSE
       prec=new mfem::UMFPackSolver();
#else
       prec=new mfem::GSSmoother();
#endif
       mfem::GMRESSolver *j_gmres = new mfem::GMRESSolver();
       j_gmres->SetRelTol(1e-7);
       j_gmres->SetAbsTol(1e-15);
       j_gmres->SetMaxIter(500);
       j_gmres->SetPrintLevel(print_level);
       j_gmres->SetPreconditioner(*prec);

       mfem::NewtonSolver* ns;
       ns=new mfem::NewtonSolver();
       ns->iterative_mode = true;
       ns->SetSolver(*j_gmres);
       ns->SetOperator(*nf);
       ns->SetPrintLevel(print_level);
       ns->SetRelTol(1e-6);
       ns->SetAbsTol(1e-12);
       ns->SetMaxIter(10);
       //solve the problem
       timer->Clear();
       timer->Start();
       ns->Mult(tv, sv);
       timer->Stop();
       std::cout<<"Time for the NewtonSolver: "<<timer->RealTime()<<std::endl;

       energy=nf->GetEnergy(sv);
       std::cout<<"[pp="<<i<<"] The total energy of the system is E="<<energy<<std::endl;

       delete ns;
       delete j_gmres;
       delete prec;

       x.SetFromTrueDofs(sv);
       dacol->SetTime(i);
       dacol->SetCycle(i);
       dacol->Save();
   }

   // 13. Continue with the final power
   if( std::abs(pp-2.0) > std::numeric_limits<double>::epsilon())
   {
       delete nf;
       nf=new mfem::NonlinearForm(&fespace);
       if(integrator==0)
       {
            nf->AddDomainIntegrator(new mfem::pLaplace(c_pp,c_ee,load));
       }else
       if(integrator==1)
       {
           nf->AddDomainIntegrator(new mfem::pLaplaceAD(c_pp,c_ee,load));
       }
       nf->SetEssentialBC(ess_bdr);
       // compute the energy
       double energy=nf->GetEnergy(sv);
       std::cout<<"[pp="<<pp<<"] The total energy of the system is E="<<energy<<std::endl;
       // time the assembly
       timer->Clear();
       timer->Start();
       mfem::Operator &op=nf->GetGradient(sv);
       timer->Stop();
       std::cout<<"[pp="<<pp<<"] The assembly time is: "<<timer->RealTime()<<std::endl;
       mfem::Solver *prec;
#ifdef MFEM_USE_SUITESPARSE
       prec=new mfem::UMFPackSolver();
#else
       prec=new mfem::GSSmoother();
#endif
       mfem::GMRESSolver *j_gmres = new mfem::GMRESSolver();
       j_gmres->SetRelTol(1e-7);
       j_gmres->SetAbsTol(1e-15);
       j_gmres->SetMaxIter(500);
       j_gmres->SetPrintLevel(print_level);
       j_gmres->SetPreconditioner(*prec);

       mfem::NewtonSolver* ns;
       ns=new mfem::NewtonSolver();
       ns->iterative_mode = true;
       ns->SetSolver(*j_gmres);
       ns->SetOperator(*nf);
       ns->SetPrintLevel(print_level);
       ns->SetRelTol(1e-6);
       ns->SetAbsTol(1e-12);
       ns->SetMaxIter(10);
       //solve the problem
       timer->Clear();
       timer->Start();
       ns->Mult(tv, sv);
       timer->Stop();
       std::cout<<"Time for the NewtonSolver: "<<timer->RealTime()<<std::endl;

       energy=nf->GetEnergy(sv);
       std::cout<<"[pp="<<pp<<"] The total energy of the system is E="<<energy<<std::endl;

       delete ns;
       delete j_gmres;
       delete prec;

       x.SetFromTrueDofs(sv);
       dacol->SetTime(pp);
       if(pp<2.0)
       {
           dacol->SetCycle(std::floor(pp));
       }
       else
       {
           dacol->SetCycle(std::ceil(pp));
       }
       dacol->Save();
   }



   // 19. Free the used memory
   delete dacol;
   delete nf;
   delete mesh;
   delete timer;


   return 0;
}


