//
// Compile with: make helmholtz
//
// Sample runs:  helmholtz -m ../data/one-hex.mesh
//               helmholtz -m ../data/fichera.mesh
//               helmholtz -m ../data/fichera-mixed.mesh
//
// Description:  This example code demonstrates the use of MFEM to define a
//               simple finite element discretization of the Helmholtz problem
//               -Delta p - omega^2 u = 1 with impedance boundary conditiones.
//
#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

// H1                               
double u_exact(const Vector &x);
void gradu_exact(const Vector &x, Vector &gradu);

// Vector FE
void U_exact(const Vector &x, Vector & U);
// H(curl)
void curlU_exact(const Vector &x, Vector &curlU);
double curlU2D_exact(const Vector &x);
// H(div)
double divU_exact(const Vector &x);

int dim;
int prob=0;
Vector alpha;

int main(int argc, char *argv[])
{
   // geometry file
   const char *mesh_file = "../data/inline-quad.mesh";
   // finite element order of approximation
   int order = 1;
   // static condensation flag
   bool visualization = 1;
   // number of initial ref
   int ref = 1;

   // optional command line inputs
   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree) or -1 for"
                  " isoparametric space.");
   args.AddOption(&prob, "-prob", "--problem",
                  "Problem kind: 0: H1, 1: H(curl), 2: H(div)");               
   args.AddOption(&ref, "-ref", "--ref",
                  "Number of refinements.");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.Parse();
   // check if the inputs are correct
   if (!args.Good())
   {
      args.PrintUsage(cout);
      return 1;
   }
   args.PrintOptions(cout);

   // 3. Read the mesh from the given mesh file.
   Mesh *mesh = new Mesh(mesh_file, 1, 1);
   // Mesh *mesh = new Mesh(1, 1, Element::QUADRILATERAL, true, 1.0, 1.0, false);
   dim = mesh->Dimension();

   alpha.SetSize(dim); 
   for (int i=0; i<dim; i++) alpha(i) = 2.0*M_PI*(double)(i+1);
   // 3. Executing uniform h-refinement
   for (int i = 0; i < ref; i++ )
   {
      mesh->UniformRefinement();
   }

   // 6. Define a finite element space on the mesh.
   FiniteElementCollection *fec=nullptr;
   switch (prob)
   {
      case 0: fec = new H1_FECollection(order,dim);   break;
      case 1: fec = new ND_FECollection(order,dim);   break;
      case 2: fec = new RT_FECollection(order-1,dim); break;
   default: break;
   }
   FiniteElementSpace *fespace = new FiniteElementSpace(mesh, fec);
   cout << "Number of finite element unknowns: " << fespace->GetTrueVSize()
        << endl;
   
   Array<int> ess_tdof_list;
   if (mesh->bdr_attributes.Size())
   {
      Array<int> ess_bdr(mesh->bdr_attributes.Max());
      ess_bdr = 0;
      fespace->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
   }

   GridFunction u_gf(fespace);
   FunctionCoefficient * u, *divU,  *curlU2D; 
   VectorFunctionCoefficient * U, *gradu, *curlU; 

   FunctionCoefficient *u_ex=nullptr;
   VectorFunctionCoefficient *U_ex=nullptr;

   ConstantCoefficient one(1.0);

   // Calculate H1 projection
   LinearForm b(fespace);
   BilinearForm a(fespace);

   switch (prob)
   {
   case 0: //(grad u_ex, grad v) + (u_ex,v)
      u_ex = new FunctionCoefficient(u_exact);

      u = new FunctionCoefficient(u_exact);
      gradu = new VectorFunctionCoefficient(dim,gradu_exact);
      b.AddDomainIntegrator(new DomainLFGradIntegrator(*gradu)); 
      b.AddDomainIntegrator(new DomainLFIntegrator(*u));

      // (grad u, grad v) + (u,v)
      a.AddDomainIntegrator(new DiffusionIntegrator(one));
      a.AddDomainIntegrator(new MassIntegrator(one));

      break;
   case 1: //(curl u_ex, curl v + (u_ex,v)
      U_ex = new VectorFunctionCoefficient(dim,U_exact);

      U = new VectorFunctionCoefficient(dim,U_exact);
      if (dim == 3)
      {
         curlU = new VectorFunctionCoefficient(dim,curlU_exact);
         b.AddDomainIntegrator(new VectorFEDomainLFCurlIntegrator(*curlU));
      }
      else if (dim == 2)
      {
         curlU2D = new FunctionCoefficient(curlU2D_exact);
         b.AddDomainIntegrator(new VectorFEDomainLFCurlIntegrator(*curlU2D));
      }
      b.AddDomainIntegrator(new VectorFEDomainLFIntegrator(*U));
      // (curl u, curl v) + (u,v)
      a.AddDomainIntegrator(new CurlCurlIntegrator(one));
      a.AddDomainIntegrator(new VectorFEMassIntegrator(one));
      break;

   case 2: //(div u_ex, div v) + (u_ex,v)   
      U_ex = new VectorFunctionCoefficient(dim,U_exact);
      U = new VectorFunctionCoefficient(dim,U_exact);
      divU = new FunctionCoefficient(divU_exact);
      b.AddDomainIntegrator(new VectorFEDomainLFDivIntegrator(*divU));
      b.AddDomainIntegrator(new VectorFEDomainLFIntegrator(*U));

      // (div u, div v) + (u,v)
      a.AddDomainIntegrator(new DivDivIntegrator(one));
      a.AddDomainIntegrator(new VectorFEMassIntegrator(one));
      break;

   default:
      break;
   }
   b.Assemble();
   a.Assemble();
   OperatorPtr A;
   Vector X, B;
   a.FormLinearSystem(ess_tdof_list, u_gf, b, A, X,B);

   UMFPackSolver umf_solver;
   umf_solver.Control[UMFPACK_ORDERING] = UMFPACK_ORDERING_METIS;
   umf_solver.SetOperator(*A);
   umf_solver.Mult(B, X);

   a.RecoverFEMSolution(X,B,u_gf);

   int order_quad = max(2, 2*order+1);
   const IntegrationRule *irs[Geometry::NumGeom];
   for (int i=0; i < Geometry::NumGeom; ++i)
   {
      irs[i] = &(IntRules.Get(i, order_quad));
   }

   double L2err = 0.0;
   switch (prob)
   {
   case 0:
      L2err = u_gf.ComputeL2Error(*u_ex);
      break;
      case 1:
      case 2:
      L2err = u_gf.ComputeL2Error(*U_ex);
      break;   
   default:
      break;
   }

   cout << " || u_h - u ||_{L^2} = " << L2err <<  endl;

   if (visualization)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      string keys;
      if (dim ==2 )
      {
         // keys = "keys mrRljc\n";
         keys = "keys \n";
      }
      else
      {
         keys = "keys mc\n";
      }
      socketstream sol_sock_re(vishost, visport);
      sol_sock_re.precision(8);
      sol_sock_re << "solution\n" << *mesh << u_gf <<
                  "window_title 'Numerical Pressure (real part)' "
                  << keys << flush;
   }

   delete fespace;
   delete fec;
   delete mesh;
   return 0;
}

double u_exact(const Vector &x)
{
   double u;
   double y=0;
   for (int i=0; i<dim; i++)
   {
      y+= alpha(i) * x(i);
   }
   u = cos(y);
   return u;
}

void gradu_exact(const Vector &x, Vector &du)
{
   double s=0.0;
   for (int i=0; i<dim; i++)
   {
      s+= alpha(i) * x(i);
   }
   for (int i=0; i<dim; i++)
   {
      du[i] = -alpha(i) * sin(s);
   }
}

void U_exact(const Vector &x, Vector & U)
{
   double s = x.Sum();
   for (int i=0; i<dim; i++)
   {
      U[i] = cos(alpha(i) * s);
   }
}
// H(curl)
void curlU_exact(const Vector &x, Vector &curlU)
{
   MFEM_VERIFY(dim == 3, "This should be called only for 3D cases");

   double s = x.Sum();
   curlU[0] = -alpha(2)*sin(alpha(2) * s) + alpha(1)*sin(alpha(1) * s);
   curlU[1] = -alpha(0)*sin(alpha(0) * s) + alpha(2)*sin(alpha(2) * s);
   curlU[2] = -alpha(1)*sin(alpha(1) * s) + alpha(0)*sin(alpha(0) * s);
}

double curlU2D_exact(const Vector &x) 
{
   MFEM_VERIFY(dim == 2, "This should be called only for 2D cases");
   double s = x(0) + x(1);
   return -alpha(1)*sin(alpha(1) * s) + alpha(0)*sin(alpha(0) * s);
}

// H(div)
double divU_exact(const Vector &x)
{
   double divu = 0.0;
   double s = x.Sum();

   for (int i = 0; i<dim; i++)
   {
      divu += -alpha(i) * sin(alpha(i) * s);
   }
   return divu;
}


