//         first test on Stokes



#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include "my_integrators.hpp"



using namespace std;
using namespace mfem;


void f_rhs(const Vector &x,Vector &y);
void u_ex(const Vector &x,Vector &y);
real_t p_ex(const Vector &x);
real_t ellipsoide_func(const Vector &x);
void g_neumann(const Vector &x,Vector &z );



int main(int argc, char *argv[])
{
   StopWatch chrono;

   // 1. Parse command line options.
   int n = 20;
   int order = 1;
   OptionsParser args(argc, argv);
   args.AddOption(&order, "-o", "--order", "Finite element polynomial degree");
      args.AddOption(&n, "-n", "--n", "n");

   args.ParseCheck();
   Mesh mesh = Mesh::MakeCartesian2D( n*2, n , mfem::Element::Type::QUADRILATERAL, true, 1, 0.5);

   double h_min, h_max, kappa_min, kappa_max;
   mesh.GetCharacteristics(h_min, h_max, kappa_min, kappa_max);

   // mesh.UniformRefinement(); //TODO: how to do this without repeating

   H1_FECollection fec(order+1, mesh.Dimension());
   FiniteElementSpace Vspace(&mesh, &fec,mesh.Dimension(),Ordering::byVDIM);
   H1_FECollection fec2(order, mesh.Dimension());
   FiniteElementSpace Qspace(&mesh, &fec2);
   cout << "Dimension: " << mesh.Dimension() << endl;

   GridFunction cgf(&Qspace);
   FunctionCoefficient circle(ellipsoide_func);
   cgf.ProjectCoefficient(circle);
 



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

   GridFunction v(&Vspace);
   VectorFunctionCoefficient v_coeff(dim, u_ex);
   v.ProjectCoefficient(v_coeff);


   VectorFunctionCoefficient neumann(dim,g_neumann);
   Array<int> boundary_dofs(mesh.bdr_attributes.Max());
   boundary_dofs = 1;


   Array<int> outside_dofs_v;
   Array<int> outside_dofs_p;
   Array<int> marks;
   Array<int> face_marks;
   {
       ElementMarker* elmark=new ElementMarker(mesh,true,true); 
       elmark->SetLevelSetFunction(cgf);
       elmark->MarkElements(marks);
       elmark->ListEssentialTDofs(marks,Vspace,outside_dofs_v);
       elmark->ListEssentialTDofs(marks,Qspace,outside_dofs_p);
      elmark->MarkGhostPenaltyFaces(face_marks);
       delete elmark;
   }
   //  outside_dofs.Append(boundary_dofs);
   //  outside_dofs.Sort();
   //  outside_dofs.Unique();

   int otherorder = 2;
   int aorder = 8; // Algoim integration points
   AlgoimIntegrationRules* air=new AlgoimIntegrationRules(aorder,circle,otherorder);
   real_t gp1 = 0.1/(h_min*h_min);
   real_t gp2 = -0.1;

   // 6. Set up the linear form b(.) corresponding to the right-hand side.
   VectorFunctionCoefficient f (dim,f_rhs);
   // LinearForm b(&fespace);

   LinearForm *fform(new LinearForm);
   fform->Update(&Vspace, rhs.GetBlock(0), 0);
   // fform->AddDomainIntegrator(new VectorDomainLFIntegrator(f));
   fform->AddDomainIntegrator(new CutVectorDomainLFIntegrator(f,&marks,air));
   fform->AddDomainIntegrator(new CutUnfittedVectorBoundaryLFIntegrator(neumann,&marks,air));

   fform->Assemble();
   fform->SyncAliasMemory(rhs);

   ConstantCoefficient zero(0.0);

   // if rhs on second block: 
   // LinearForm *gform(new LinearForm);
   // gform->Update(&Qspace, rhs.GetBlock(1), 0);
   // gform->AddDomainIntegrator(new CutDomainLFIntegrator(zero,&marks,air));
   // gform->Assemble();
   // gform->SyncAliasMemory(rhs);
   
   ConstantCoefficient one(1.0);


   // Bilinear forms: 
   BilinearForm *mVarf(new BilinearForm(&Vspace));
   MixedBilinearForm *bVarf(new MixedBilinearForm(&Vspace, &Qspace));
   BilinearForm *cVarf(new BilinearForm(&Qspace));

   mVarf->SetDiagonalPolicy(Operator::DiagonalPolicy::DIAG_KEEP);
   cVarf->SetDiagonalPolicy(Operator::DiagonalPolicy::DIAG_KEEP);


   ConstantCoefficient lambda(0.0);
   ConstantCoefficient mu(1.0);

   // mVarf->AddDomainIntegrator(new VectorDiffusionIntegrator);
   mVarf->AddDomainIntegrator(new CutVectorDiffusionIntegrator(one,&marks,air));
   mVarf->AddInteriorFaceIntegrator(new CutGhostPenaltyVectorIntegrator(gp1,&face_marks));
   mVarf->Assemble(0);
   mVarf->EliminateEssentialBC(boundary_dofs, v, rhs.GetBlock(0));
   // smVarf->EliminateVDofs(outside_dofs_v);
   mVarf->Finalize(0);



   bVarf->AddDomainIntegrator(new CutVectorDivergenceIntegrator(one,&marks,air));
   bVarf->Assemble(0);
   bVarf->EliminateTrialDofs(boundary_dofs, v, rhs.GetBlock(1));
   bVarf->Finalize(0);

   cVarf->AddInteriorFaceIntegrator(new CutGhostPenaltyIntegrator(gp2,&face_marks));
   cVarf->Assemble(0);
   cVarf->Finalize(0);


   BlockOperator StokesOp(block_offsets);

   TransposeOperator *Bt = NULL;
   SparseMatrix &Ms(mVarf->SpMat());
   SparseMatrix &Bs(bVarf->SpMat());
   SparseMatrix &C(cVarf->SpMat());


   cout <<  outside_dofs_p.Size()<<": "<<  outside_dofs_v.Size()  << endl;

   Vector before;
   C.GetDiag(before);
   cout <<"before"<<endl;
   //before.Print();

   for(int i=0;i<outside_dofs_v.Size();i++)
   {
      // cout<<outside_dofs_v[i]<<endl;
      Ms.EliminateRowColDiag(outside_dofs_v[i],1.0);
   }

   //Ms.SetDiagIdentity();



   for(int i=0;i<outside_dofs_p.Size();i++)
   {
      C.EliminateRowCol(outside_dofs_p[i]);
   }


   // Bs *= -1.;
   Bt = new TransposeOperator(&Bs);

   StokesOp.SetBlock(0,0, &Ms);
   StokesOp.SetBlock(0,1, Bt);
   StokesOp.SetBlock(1,0, &Bs);
   StokesOp.SetBlock(1,1, &C);

   // rhs.Print();
   // C.Print();
   // Bs.Print();

   // for preconditioner 
   BilinearForm *mpre(new BilinearForm(&Vspace));
   BilinearForm *cpre(new BilinearForm(&Qspace));
   cpre->AddDomainIntegrator(new MassIntegrator);
   cpre->Assemble();
   cpre->Finalize(); 

   BlockDiagonalPreconditioner stokesPrec(block_offsets);
   Solver *solM, *solI;

   mpre->AddDomainIntegrator(new VectorDiffusionIntegrator);
   mpre->Assemble();
   mpre->Finalize(); 


   SparseMatrix &M(mpre->SpMat());
   SparseMatrix &I(cpre->SpMat());


   solM = new GSSmoother(Ms);
   solI = new GSSmoother(I);


  Vector id_vec(bVarf->Height());
   id_vec = 1.0;
   SparseMatrix id_mat(id_vec);



   stokesPrec.SetDiagonalBlock(0, solM);
   stokesPrec.SetDiagonalBlock(1, &id_mat);

   // 11. Solve the linear system with MINRES.
   int maxIter(30000);
   real_t rtol(1.e-15);
   real_t atol(1.e-10);


   chrono.Clear();
   chrono.Start();
   MINRESSolver solver;
   solver.SetAbsTol(atol);
   solver.SetRelTol(rtol);
   solver.SetMaxIter(maxIter);
   solver.SetOperator(StokesOp);
   solver.SetPreconditioner(stokesPrec);
   solver.SetPrintLevel(2);
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


   int order_quad = max(5, 2*order+1);
   const IntegrationRule *irs[Geometry::NumGeom];
   for (int i=0; i < Geometry::NumGeom; ++i)
   {
      irs[i] = &(IntRules.Get(i, order_quad));
   }
   FunctionCoefficient press (p_ex);

      //compute the error
   {
       NonlinearForm* nf=new NonlinearForm(&Vspace);
       nf->AddDomainIntegrator(new CutVectorErrorIntegrator(v_coeff,&marks,air));
        real_t error_squared = nf->GetEnergy(u.GetTrueVector());
       cout << "\n|| u_h - u ||_{L^2} = " << sqrt(error_squared)<< std::endl;

         NonlinearForm* nf2=new NonlinearForm(&Qspace);
       nf2->AddDomainIntegrator(new CutScalarErrorIntegrator(press,&marks,air));
        real_t error_squared_p = nf2->GetEnergy(p.GetTrueVector());
      cout << "\n|| p_h - p ||_{L^2} = " << sqrt(error_squared_p) << '\n' << endl;

       delete nf;
              delete nf2;
   }

   // cout << "\n|| u_h - u ||_{L^2} = " << u.ComputeL2Error(v_coeff,irs) << '\n' << endl;
   // cout << "\n|| p_h - p ||_{L^2} = " << p.ComputeL2Error(press,irs) << '\n' << endl;

   // to visualize level set and markings
   L2_FECollection* l2fec= new L2_FECollection(0,mesh.Dimension());
   FiniteElementSpace* l2fes= new FiniteElementSpace(&mesh,l2fec,1);
   GridFunction mgf(l2fes);
   for(int i=0;i<marks.Size();i++){
      mgf[i]=marks[i];
   }

   GridFunction exact_p(&Qspace);
   exact_p.ProjectCoefficient(press);

   GridFunction error_u(&Vspace);
   error_u = u;
   error_u -= v;

   GridFunction error_p(&Qspace);
   error_p = p;
   error_p -= exact_p;

   ParaViewDataCollection paraview_dc("stokes_cut", &mesh);
   paraview_dc.SetPrefixPath("ParaView");
   paraview_dc.SetLevelsOfDetail(order);
   paraview_dc.SetCycle(0);
   paraview_dc.SetDataFormat(VTKFormat::BINARY);
   paraview_dc.SetHighOrderOutput(true);
   paraview_dc.SetTime(0.0); // set the time
   paraview_dc.RegisterField("velocity",&u);
   paraview_dc.RegisterField("pressure",&p);
   paraview_dc.RegisterField("marks", &mgf);
   paraview_dc.RegisterField("exact_v", &v);
      paraview_dc.RegisterField("exact_p", &exact_p);

      paraview_dc.RegisterField("level_set",&cgf);
            paraview_dc.RegisterField("error_u", &error_u);

      paraview_dc.RegisterField("error_p",&error_p);

   paraview_dc.Save();

   return 0;
}

void f_rhs(const Vector &x,Vector &y)
{
   y(0) = sin(x(1)) + 3*x(0) * x(0), y(1) = sin(x(0)) + 3*x(1) * x(1);
   // y(0) = sin(x(1)), y(1) = sin(x(0)) ;
}

void u_ex(const Vector &x,Vector &y)
{
   y(0)= sin(x(1)),y(1)=sin(x(0));
}
void g_neumann(const Vector &x,Vector &z )
{
   real_t a = 1.5;
    real_t b = 0.5;
    real_t x0 = 0.5;
    real_t y0 = 0.25;
    real_t xx = x(0)-x0;
    real_t y = x(1)-y0;
    real_t normalize = sqrt((xx*xx)/(a*a*a*a) + y*y/(b*b*b*b));
   z(0) =cos(x(1))*y/(b*b*normalize)   + (- x(0) * x(0)*x(0)  - x(1) * x(1)*x(1) + 0.5)*xx/(a*a*normalize); 
   z(1) =cos(x(0))*xx/(a*a*normalize)  + (-x(0) * x(0)*x(0)  - x(1) * x(1)*x(1) + 0.5)*y/(b*b*normalize);
}

real_t p_ex(const Vector &x)
{
   return - x(0) * x(0)*x(0)  - x(1) * x(1)*x(1) + 0.5;
}

real_t circle_func(const Vector &x)
{
    real_t x0 = 0.5;
    real_t y0 = 0.5;
    real_t r = 0.4;
    return -(x(0)-x0)*(x(0)-x0) - (x(1)-y0)*(x(1)-y0) + r*r;  
}

real_t ellipsoide_func(const Vector &x)
{
    real_t x0 = 0.5;
    real_t y0 = 0.25;
    real_t r = 0.35;
    real_t xx = x(0)-x0;
    real_t y = x(1)-y0;
    return -(xx)*(xx)/(1.5*1.5) - (y)*(y)/(0.5*0.5)+ r*r; // + 0.25*cos(atan2(x(1)-y0,x(0)-x0))*cos(atan2(x(1)-y0,x(0)-x0));  
}
