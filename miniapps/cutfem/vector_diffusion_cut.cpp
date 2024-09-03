//       Solving the vectorized version of the Laplace  problem -Delta u = 1 with 
//       Dirichlet boundary conditions


#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include "my_integrators.hpp"



using namespace std;
using namespace mfem;


void f_rhs(const Vector &x,Vector &y);
void u_ex(const Vector &x,Vector &y);
void g_neumann(const Vector &x,Vector &z);
real_t circle_func(const Vector &x);
real_t ellipsoide_func(const Vector &x);



int main(int argc, char *argv[])
{
   


   int n =30;
   Mesh mesh = Mesh::MakeCartesian2D( n*2, n , mfem::Element::Type::QUADRILATERAL, true, 1, 0.5);
   int dim = mesh.Dimension();

   int order = 1;

   OptionsParser args(argc, argv);
   args.AddOption(&order, "-o", "--order", "Finite element polynomial degree");
   args.ParseCheck();


   double h_min, h_max, kappa_min, kappa_max;
   mesh.GetCharacteristics(h_min, h_max, kappa_min, kappa_max);

   H1_FECollection fec(order, mesh.Dimension());
   FiniteElementSpace fespace(&mesh, &fec,mesh.Dimension(),Ordering::byNODES);

   FiniteElementSpace fespacescalar(&mesh, &fec);

   // cout << "Number of unknowns: " << fespace.GetTrueVSize() << endl;

   ConstantCoefficient one(1.0);
   ConstantCoefficient zero(1.0);


   GridFunction x(&fespace);
   VectorFunctionCoefficient bc (dim,u_ex);
   VectorFunctionCoefficient f (dim,f_rhs);
   x.ProjectCoefficient(bc);
   VectorFunctionCoefficient neumann(dim,g_neumann);


   // level set function 
   GridFunction cgf(&fespacescalar);
   FunctionCoefficient circle(ellipsoide_func);
   cgf.ProjectCoefficient(circle);

   // mark elements and outside DOFs
   Array<int> boundary_dofs;
   fespace.GetBoundaryTrueDofs(boundary_dofs);
   // Array<int> outside_dofs;
   Array<int> marks;
   {
      //  Array<int> outside_dofs;
       ElementMarker* elmark=new ElementMarker(mesh,true,true); 
       elmark->SetLevelSetFunction(cgf);
       elmark->MarkElements(marks);
      //  elmark->ListEssentialTDofs(marks,fespace,outside_dofs);
       delete elmark;
   }
   //  outside_dofs.Append(boundary_dofs);
   //  outside_dofs.Sort();
   //  outside_dofs.Unique();

   int otherorder = 2;
   int aorder = 2; // Algoim integration points
   AlgoimIntegrationRules* air=new AlgoimIntegrationRules(aorder,circle,otherorder);
   real_t gp = 0.1/(h_min*h_min);

   // 6. Set up the linear form b(.) corresponding to the right-hand side.

   LinearForm b(&fespace);

   b.AddDomainIntegrator(new CutVectorDomainLFIntegrator(f,&marks,air));
   b.AddDomainIntegrator(new CutUnfittedVectorBoundaryLFIntegrator(neumann,&marks,air));
   b.Assemble();

   // 7. Set up the bilinear form a(.,.) corresponding to the -Delta operator.
   BilinearForm a(&fespace);
   a.AddDomainIntegrator(new CutVectorDiffusionIntegrator(one,&marks,air));
   a.AddInteriorFaceIntegrator(new CutGhostPenaltyVectorIntegrator(gp,&marks));
   a.Assemble();

   
   SparseMatrix A;
   Vector B, X;
   a.FormLinearSystem(boundary_dofs, x, b, A, X, B);

   // A.Print();
   // 9. Solve the system using PCG with symmetric Gauss-Seidel preconditioner.
   GSSmoother M(A);
   PCG(A,M, B, X, 2, 2000, 1e-20, 0.0);

   // 10. Recover the solution x as a grid function and save to file
   a.RecoverFEMSolution(X, b, x);


   //compute the error
   // {
   //     NonlinearForm* nf=new NonlinearForm(&fespace);
   //     nf->AddDomainIntegrator(new CutScalarErrorIntegrator(bc,&marks,air));

   //     cout << "\n|| u_h - u ||_{L^2} = " << nf->GetEnergy(x.GetTrueVector())<< std::endl;

   //     delete nf;
   // }

   // to visualize level set and markings
   L2_FECollection* l2fec= new L2_FECollection(0,mesh.Dimension());
   FiniteElementSpace* l2fes= new FiniteElementSpace(&mesh,l2fec,1);
   GridFunction mgf(l2fes);
   for(int i=0;i<marks.Size();i++){
      mgf[i]=marks[i];
   }
   GridFunction exact_sol(&fespace);
   exact_sol.ProjectCoefficient(bc);


   GridFunction error(&fespace);
   error = x;
   error -= exact_sol;

   //  // GLVis 
   // char vishost[] = "localhost";
   // int  visport   = 19916;
   // socketstream sol_sock(vishost, visport);
   // sol_sock.precision(8);
   // sol_sock << "solution\n" << mesh << x << flush;
   // //    // save solution with paraview 
   ParaViewDataCollection paraview_dc("vector_cut", &mesh);
   paraview_dc.SetPrefixPath("ParaView");
   paraview_dc.SetLevelsOfDetail(order);
   paraview_dc.SetCycle(0);
   paraview_dc.SetDataFormat(VTKFormat::BINARY);
   paraview_dc.SetHighOrderOutput(true);
   paraview_dc.SetTime(0.0); // set the time
   paraview_dc.RegisterField("solution",&x);
   paraview_dc.RegisterField("marks", &mgf);
   paraview_dc.RegisterField("level_set",&cgf);
   paraview_dc.RegisterField("exact_sol",&exact_sol);
   paraview_dc.RegisterField("error",&error);
   paraview_dc.Save();


   delete l2fec;
   delete l2fes;
   delete air;

   cout << "h:" << h_min<<endl;
   return 0;  
}
//for gradient case
void f_rhs(const Vector &x,Vector &y)
{
   y(0)= sin(x(0)),y(1)=sin(x(0));//sin(x(0));
}

void u_ex(const Vector &x,Vector &y)
{
   y(0)= sin(x(0)),y(1)=sin(x(0));
}

void g_neumann(const Vector &x, Vector &z)
{
    real_t a = 1.5;
    real_t b = 0.5;
    real_t x0 = 0.5;
    real_t y0 = 0.25;
    real_t xx = x(0)-x0;
    real_t y = x(1)-y0;
    real_t normalize = sqrt((xx*xx)/(a*a*a*a) + y*y/(b*b*b*b));
   //  z(0) = cos(x(1))*y/(b*b*normalize), z(1)=cos(x(0))*xx/(a*a*normalize);
   // z(0) =cos(x(1))*y/(b*b*normalize), z(1)=cos(x(0))*xx/(a*a*normalize);
   z(0) =cos(x(0))*xx/(a*a*normalize), z(1)=cos(x(0))*xx/(a*a*normalize);
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
// // for symmetric gradient
// void f_rhs(const Vector &x,Vector &y)
// {
//    y(0) = 3*cos(x(0))*sin(x(1)) - cos(x(0))*cos(x(1)), y(1)=3*sin(x(0))*sin(x(1))+sin(x(0))*cos(x(1));
// }

// void u_ex(const Vector &x,Vector &y)
// {
//    y(0)= cos(x(0))*sin(x(1)),y(1)=sin(x(0))*sin(x(1));
// }



