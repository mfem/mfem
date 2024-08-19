
#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include "my_integrators.hpp"


using namespace std;
using namespace mfem;


real_t f_rhs(const Vector &x);
real_t u_ex(const Vector &x);
real_t g_neuman(const Vector &x);
void u_grad_exact(const Vector &x, Vector &u);
real_t circle_func(const Vector &x);
real_t ellipsoide_func(const Vector &x);
real_t neg_circle_func(const Vector &x);


int main(int argc, char *argv[])
{

   int n = 40;
   Mesh mesh = Mesh::MakeCartesian2D( n*2, n , mfem::Element::Type::QUADRILATERAL, true, 1.0, 0.5);

   int order = 1;

   OptionsParser args(argc, argv);
   args.AddOption(&order, "-o", "--order", "Finite element polynomial degree");
   args.ParseCheck();


   double h_min, h_max, kappa_min, kappa_max;

   mesh.GetCharacteristics(h_min, h_max, kappa_min, kappa_max);

   H1_FECollection fec(order, mesh.Dimension());
   FiniteElementSpace fespace(&mesh, &fec);
   cout << "Number of unknowns: " << fespace.GetTrueVSize() << endl;


   ConstantCoefficient one(1.0);
   ConstantCoefficient zero(0.0);


   MyGridFunction x(&fespace);
   FunctionCoefficient bc (u_ex);
   FunctionCoefficient f (f_rhs);
   x.ProjectCoefficient(bc);

   // leve set function 
   GridFunction cgf(&fespace);
   FunctionCoefficient circle(circle_func);
   cgf.ProjectCoefficient(circle);

   // mark elements and outside DOFs
   Array<int> boundary_dofs;
   fespace.GetBoundaryTrueDofs(boundary_dofs);
   Array<int> outside_dofs;
   Array<int> marks;
   {
       Array<int> outside_dofs;
       ElementMarker* elmark=new ElementMarker(mesh,true,true); 
       elmark->SetLevelSetFunction(cgf);
       elmark->MarkElements(marks);
       elmark->ListEssentialTDofs(marks,fespace,outside_dofs);
       delete elmark;
   }
   FunctionCoefficient neumann(g_neuman);

   outside_dofs.Append(boundary_dofs);
   outside_dofs.Sort();
   outside_dofs.Unique();

   int otherorder = 2;
   int aorder = 2; // Algoim integration points
   AlgoimIntegrationRules* air=new AlgoimIntegrationRules(aorder,circle,otherorder);
   real_t gp = 1/(h_min*h_min);

   // 6. Set up the linear form b(.) corresponding to the right-hand side.

   LinearForm b(&fespace);

   b.AddDomainIntegrator(new CutDomainLFIntegrator(f,&marks,air));
   b.AddDomainIntegrator(new CutUnfittedBoundaryLFIntegrator(neumann,&marks,air));
   b.Assemble();

   // 7. Set up the bilinear form a(.,.) corresponding to the -Delta operator.
   BilinearForm a(&fespace);
   a.AddDomainIntegrator(new CutDiffusionIntegrator(one,&marks,air));
   a.AddInteriorFaceIntegrator(new CutGhostPenaltyIntegrator(gp,&marks));
   a.Assemble();


   SparseMatrix A;
   Vector B, X;
   a.FormLinearSystem(boundary_dofs, x, b, A, X, B);

   // 9. Solve the system using PCG with symmetric Gauss-Seidel preconditioner.
   GSSmoother M(A);
   PCG(A,M, B, X, 1, 1000, 1e-16, 0.0);

   // 10. Recover the solution x as a grid function and save to file
   a.RecoverFEMSolution(X, b, x);

   
   GridFunction cgf_opposite(&fespace);
   FunctionCoefficient circle_opposite(neg_circle_func);
   cgf_opposite.ProjectCoefficient(circle_opposite);

   // error function has opposite definition of element marks 
   Array<int> marks_error;
   ElementMarker* elmark=new ElementMarker(mesh,true,true); 
   elmark->SetLevelSetFunction(cgf_opposite);
   elmark->MarkElements(marks_error);
   delete elmark;


   const IntegrationRule *irs = NULL;
   // computing only on the inner domains
   cout << "\n|| u_h - u ||_{L^2} = " << x.myComputeL2Error(bc,&marks_error) << '\n' << endl;

   //compute the error
   {
       NonlinearForm* nf=new NonlinearForm(&fespace);
       nf->AddDomainIntegrator(new CutScalarErrorIntegrator(bc,&marks,air));

       cout << "\n|| u_h - u ||_{L^2} = " << nf->GetEnergy(x.GetTrueVector())<< std::endl;


       delete nf;
   }

   // to visualize level set and markings
   L2_FECollection* l2fec= new L2_FECollection(0,mesh.Dimension());
   FiniteElementSpace* l2fes= new FiniteElementSpace(&mesh,l2fec,1);
   GridFunction mgf(l2fes);
   for(int i=0;i<marks.Size();i++){
      mgf[i]=marks[i];
   }
   GridFunction exact_sol(&fespace);
   exact_sol.ProjectCoefficient(bc);

   GridFunction mgf_error(l2fes);

   for(int i=0;i<marks_error.Size();i++){
      mgf_error[i]=marks_error[i];
   }
   GridFunction error(&fespace);
   error = x;
   error -= exact_sol;

   char vishost[] = "localhost";
   int  visport   = 19916;
   socketstream sol_sock(vishost, visport);
   sol_sock.precision(8);
   sol_sock << "solution\n" << mesh << x << flush;
   //    // save solution with paraview 
   ParaViewDataCollection paraview_dc("diffusion_cut", &mesh);
   paraview_dc.SetPrefixPath("ParaView");
   paraview_dc.SetLevelsOfDetail(order);
   paraview_dc.SetCycle(0);
   paraview_dc.SetDataFormat(VTKFormat::BINARY);
   paraview_dc.SetHighOrderOutput(true);
   paraview_dc.SetTime(0.0); // set the time
   paraview_dc.RegisterField("velocity",&x);
   paraview_dc.RegisterField("marks", &mgf);
      paraview_dc.RegisterField("marks_errer", &mgf_error);

   paraview_dc.RegisterField("level_set",&cgf);
      paraview_dc.RegisterField("exact_sol",&exact_sol);
            paraview_dc.RegisterField("error",&error);

   paraview_dc.Save();


   delete l2fec;
   delete l2fes;
   delete air;

   cout << "h " << h_min<<endl;

   cout << "finished " << endl;
   return 0;
}


real_t f_rhs(const Vector &x)
{
    real_t x0 = 0.5;
    real_t y0 = 0.5;
   return -8*M_PI*cos(2*M_PI*((x(0)-x0)*(x(0)-x0)+(x(1)-y0)*(x(1)-y0))) + 16*M_PI*M_PI*((x(0)-x0)*(x(0)-x0)+(x(1)-y0)*(x(1)-y0))*sin(2*M_PI*((x(0)-x0)*(x(0)-x0)+(x(1)-y0)*(x(1)-y0)));
}

real_t g_neuman(const Vector &x)
{
    real_t x0 = 0.5;
    real_t y0 = 0.5;
    real_t r = 0.4;
   return 4*M_PI*r*cos(2*M_PI*r*r);
}

real_t u_ex(const Vector &x)
{
       real_t x0 = 0.5;
    real_t y0 = 0.5;
    return sin(2*M_PI*((x(0)-x0)*(x(0)-x0)+(x(1)-y0)*(x(1)-y0)));
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
    real_t y0 = 0.5;
    real_t r = 0.35;
    return -(x(0)-x0)*(x(0)-x0)/(1.5*1.5) - (x(1)-y0)*(x(1)-y0)/(0.5*0.5)+ r*r;  
}


real_t neg_circle_func(const Vector &x)
{
    real_t x0 = 0.5;
    real_t y0 = 0.5;
    real_t r = 0.4;
    return (x(0)-x0)*(x(0)-x0) + (x(1)-y0)*(x(1)-y0) - r*r;  
}
