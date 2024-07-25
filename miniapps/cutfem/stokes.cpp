//         first test on Stokes



#include "mfem.hpp"
#include <fstream>
#include <iostream>



using namespace std;
using namespace mfem;


void f_rhs(const Vector &x,Vector &y);
void u_ex(const Vector &x,Vector &y);
real_t p_ex(const Vector &x);




int main(int argc, char *argv[])
{
   StopWatch chrono;

   // 1. Parse command line options.
   string mesh_file = "../../data/ref-square.mesh";
   int order = 1;
   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh", "Mesh file to use.");
   args.AddOption(&order, "-o", "--order", "Finite element polynomial degree");
   args.ParseCheck();

   Mesh mesh(mesh_file);
   mesh.UniformRefinement();
	mesh.UniformRefinement();
	mesh.UniformRefinement();
   mesh.UniformRefinement();
   // mesh.UniformRefinement(); //TODO: how to do this without repeating

   H1_FECollection fec(order+1, mesh.Dimension());
   FiniteElementSpace Vspace(&mesh, &fec,mesh.Dimension(),Ordering::byVDIM);
   H1_FECollection fec2(order, mesh.Dimension());
   FiniteElementSpace Qspace(&mesh, &fec2);
   cout << "Dimension: " << mesh.Dimension() << endl;

 
   int dim = mesh.Dimension();

   const char *device_config = "cpu";
   Device device(device_config);
   device.Print();

   Array<int> block_offsets(3); // number of variables + 1
   block_offsets[0] = 0;
   block_offsets[1] = Vspace.GetVSize();
   block_offsets[2] = Qspace.GetVSize();
   block_offsets.PartialSum();


   //    (u,p) and the linear forms (fform, gform).
   MemoryType mt = device.GetMemoryType();
   BlockVector x(block_offsets, mt), rhs(block_offsets, mt);
   // 4. Extract the list of all the boundary DOFs. 
   Array<int> boundary_dofs;
   Vspace.GetBoundaryTrueDofs(boundary_dofs);

   GridFunction v(&Vspace);
   VectorFunctionCoefficient v_coeff(dim, u_ex);
   v.ProjectCoefficient(v_coeff);
   Array<int> ess_bdr(mesh.bdr_attributes.Max());
   ess_bdr = 1;

   // 6. Set up the linear form b(.) corresponding to the right-hand side.
   VectorFunctionCoefficient f (dim,f_rhs);
   // LinearForm b(&fespace);

   LinearForm *fform(new LinearForm);
   fform->Update(&Vspace, rhs.GetBlock(0), 0);
   fform->AddDomainIntegrator(new VectorDomainLFIntegrator(f));
   fform->Assemble();
   fform->SyncAliasMemory(rhs);

   // if rhs on second block: 
   // LinearForm *gform(new LinearForm);
   // gform->Update(&Qspace, rhs.GetBlock(1), 0);
   // gform->AddDomainIntegrator(new DomainLFIntegrator(zero));
   // gform->Assemble();
   // gform->SyncAliasMemory(rhs);


   // Bilinear forms: 
   BilinearForm *mVarf(new BilinearForm(&Vspace));
   MixedBilinearForm *bVarf(new MixedBilinearForm(&Vspace, &Qspace));

   ConstantCoefficient lambda(0.0);
   ConstantCoefficient mu(1.0);

   mVarf->AddDomainIntegrator(new VectorDiffusionIntegrator);
   mVarf->Assemble();
   mVarf->EliminateEssentialBC(ess_bdr, v, rhs.GetBlock(0));
   mVarf->Finalize(); 

   bVarf->AddDomainIntegrator(new VectorDivergenceIntegrator);
   bVarf->Assemble();
   bVarf->EliminateTrialDofs(ess_bdr, v, rhs.GetBlock(1));
   bVarf->Finalize();


   BlockOperator StokesOp(block_offsets);

   TransposeOperator *Bt = NULL;
   SparseMatrix &Ms(mVarf->SpMat());
   SparseMatrix &Bs(bVarf->SpMat());
   // Bs *= -1.;
   Bt = new TransposeOperator(&Bs);

   StokesOp.SetBlock(0,0, &Ms);
   StokesOp.SetBlock(0,1, Bt);
   StokesOp.SetBlock(1,0, &Bs);


   // for preconditioner 
   BilinearForm *cVarf(new BilinearForm(&Qspace));
   ConstantCoefficient one(1.0);
   cVarf->AddDomainIntegrator(new MassIntegrator(one));
   cVarf->Assemble();


   BlockDiagonalPreconditioner stokesPrec(block_offsets);
   Solver *solM, *solI;


   SparseMatrix &M(mVarf->SpMat());
   SparseMatrix &I(cVarf->SpMat());


   solM = new DSmoother(M);
   solI = new GSSmoother(I);

   solM->iterative_mode = false;
   solI->iterative_mode = false;

   stokesPrec.SetDiagonalBlock(0, solM);
   stokesPrec.SetDiagonalBlock(1, solI);

   // 11. Solve the linear system with MINRES.
   int maxIter(1000);
   real_t rtol(1.e-6);
   real_t atol(1.e-10);


   chrono.Clear();
   chrono.Start();
   MINRESSolver solver;
   solver.SetAbsTol(atol);
   solver.SetRelTol(rtol);
   solver.SetMaxIter(maxIter);
   solver.SetOperator(StokesOp);
   solver.SetPreconditioner(stokesPrec);
   solver.SetPrintLevel(1);
   x = 0.0;
   solver.Mult(rhs, x);
   if (device.IsEnabled()) { x.HostRead(); }
   chrono.Stop();


   if (solver.GetConverged())
   {
      std::cout << "MINRES converged in " << solver.GetNumIterations()
                << " iterations with a residual norm of "
                << solver.GetFinalNorm() << ".\n";
   }
   else
   {
      std::cout << "MINRES did not converge in " << solver.GetNumIterations()
                << " iterations. Residual norm is " << solver.GetFinalNorm()
                << ".\n";
   }
   std::cout << "MINRES solver took " << chrono.RealTime() << "s.\n";


   GridFunction u(&Vspace);
   GridFunction p(&Qspace);
   u.MakeTRef(&Vspace, x.GetBlock(0), 0);
   p.MakeRef(&Qspace, x.GetBlock(1), 0);


   int order_quad = max(2, 2*order+1);
   const IntegrationRule *irs[Geometry::NumGeom];
   for (int i=0; i < Geometry::NumGeom; ++i)
   {
      irs[i] = &(IntRules.Get(i, order_quad));
   }
   FunctionCoefficient press (p_ex);

   cout << "\n|| u_h - u ||_{L^2} = " << u.ComputeL2Error(v_coeff,irs) << '\n' << endl;
   cout << "\n|| p_h - p ||_{L^2} = " << p.ComputeL2Error(press,irs) << '\n' << endl;


   ParaViewDataCollection paraview_dc("stokes", &mesh);
   paraview_dc.SetPrefixPath("ParaView");
   paraview_dc.SetLevelsOfDetail(order);
   paraview_dc.SetCycle(0);
   paraview_dc.SetDataFormat(VTKFormat::BINARY);
   paraview_dc.SetHighOrderOutput(true);
   paraview_dc.SetTime(0.0); // set the time
   paraview_dc.RegisterField("velocity",&u);
   paraview_dc.RegisterField("pressure",&p);
   paraview_dc.Save();

   return 0;
}

void f_rhs(const Vector &x,Vector &y)
{
   y(0) = sin(x(1)) + 3*x(0) * x(0), y(1) = sin(x(0)) + 3*x(1) * x(1);
}

void u_ex(const Vector &x,Vector &y)
{
   y(0)= sin(x(1)),y(1)=sin(x(0));
}

real_t p_ex(const Vector &x)
{
   return - x(0) * x(0)*x(0)  - x(1) * x(1)*x(1) + 0.5;
}

