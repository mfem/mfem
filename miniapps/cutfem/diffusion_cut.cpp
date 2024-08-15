
#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include "my_integrators.hpp"


using namespace std;
using namespace mfem;


real_t f_rhs(const Vector &x);
real_t u_ex(const Vector &x);
void u_grad_exact(const Vector &x, Vector &u);
real_t circle_func(const Vector &x);
real_t ellipsoide_func(const Vector &x);

int main(int argc, char *argv[])
{
   string mesh_file = "../../data/inline-quad.mesh";;

   int order = 2;
   int rs_levels= 5;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh", "Mesh file to use.");
   args.AddOption(&order, "-o", "--order", "Finite element polynomial degree");
   args.AddOption(&rs_levels, "-rs", "--refine-serial",
                  "Number of times to refine the mesh uniformly in serial.");
   args.ParseCheck();

   Mesh mesh(mesh_file);
   for (int lev = 0; lev < rs_levels; lev++) { mesh.UniformRefinement(); }

   double h_min, h_max, kappa_min, kappa_max;

   mesh.GetCharacteristics(h_min, h_max, kappa_min, kappa_max);

   H1_FECollection fec(order, mesh.Dimension());
   FiniteElementSpace fespace(&mesh, &fec);
   cout << "Number of unknowns: " << fespace.GetTrueVSize() << endl;


   ConstantCoefficient one(1.0);
   ConstantCoefficient zero(0.0);


   GridFunction x(&fespace);
   FunctionCoefficient bc (u_ex);
   FunctionCoefficient f (f_rhs);
   x.ProjectCoefficient(zero);

   // leve set function 
   GridFunction cgf(&fespace);
   FunctionCoefficient circle(ellipsoide_func);
   cgf.ProjectCoefficient(circle);

   // mark elements and outside DOFs
   Array<int> boundary_dofs;
   fespace.GetBoundaryTrueDofs(boundary_dofs);
   Array<int> outside_dofs;
   Array<int> marks;
   {
       Array<int> outside_dofs;
       ElementMarker* elmark=new ElementMarker(mesh,true,true); // should the last argument be true or false
       elmark->SetLevelSetFunction(cgf);
       elmark->MarkElements(marks);
       elmark->ListEssentialTDofs(marks,fespace,outside_dofs);
       delete elmark;
   }


   outside_dofs.Append(boundary_dofs);
   outside_dofs.Sort();
   outside_dofs.Unique();

   int otherorder = 2;
   int aorder = 2; // Algoim integration points
   AlgoimIntegrationRules* air=new AlgoimIntegrationRules(aorder,circle,otherorder);
   real_t gp = 0.1/(h_min*h_min);

   // 6. Set up the linear form b(.) corresponding to the right-hand side.

   LinearForm b(&fespace);

   b.AddDomainIntegrator(new CutDomainLFIntegrator(f,&marks,air));
   b.Assemble();

   // 7. Set up the bilinear form a(.,.) corresponding to the -Delta operator.
   BilinearForm a(&fespace);
   a.AddDomainIntegrator(new CutDiffusionIntegrator(one,&marks,air));
   // a.AddInteriorFaceIntegrator(new CutGhostPenaltyIntegrator(gp,&marks));
   a.Assemble();


   SparseMatrix A;
   Vector B, X;
   a.FormLinearSystem(boundary_dofs, x, b, A, X, B);

   // 9. Solve the system using PCG with symmetric Gauss-Seidel preconditioner.
   GSSmoother M(A);
   CG(A, B, X, 1, 500, 1e-12, 0.0);

   // 10. Recover the solution x as a grid function and save to file
   a.RecoverFEMSolution(X, b, x);

   // to visualize level set and markings
   L2_FECollection* l2fec= new L2_FECollection(0,mesh.Dimension());
   FiniteElementSpace* l2fes= new FiniteElementSpace(&mesh,l2fec,1);
   GridFunction mgf(l2fes);
   for(int i=0;i<marks.Size();i++){
      mgf[i]=marks[i];
   }


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
   paraview_dc.RegisterField("level_set",&cgf);
   paraview_dc.Save();

   cout << "finished " << endl;
   return 0;
}


real_t f_rhs(const Vector &x)
{
    real_t x0 = 0.5;
    real_t y0 = 0.5;
   return -8*M_PI*cos(2*M_PI*((x(0)-x0)*(x(0)-x0)+(x(1)-y0)*(x(1)-y0))) + 16*M_PI*M_PI*((x(0)-x0)*(x(0)-x0)+(x(1)-y0)*(x(1)-y0))*sin(2*M_PI*((x(0)-x0)*(x(0)-x0)+(x(1)-y0)*(x(1)-y0)));
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
    real_t r = 0.35;
    return -(x(0)-x0)*(x(0)-x0) - (x(1)-y0)*(x(1)-y0) + r*r;  
}

real_t ellipsoide_func(const Vector &x)
{
    real_t x0 = 0.5;
    real_t y0 = 0.5;
    real_t r = 0.35;
    return -(x(0)-x0)*(x(0)-x0)/(1.5*1.5) - (x(1)-y0)*(x(1)-y0)/(0.5*0.5)+ r*r;  
}
