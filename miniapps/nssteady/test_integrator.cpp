#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include <random>
#include "snavier_cg.hpp"

using namespace mfem;

// Forward declarations
void vFun_ex1(const Vector & x, Vector & v);
void vFun_ex2(const Vector & x, Vector & v);
void vFun_ex3(const Vector & x, Vector & v);
void vFun_ex4(const Vector & x, Vector & v);

// Test
int main(int argc, char *argv[])
{
   //
   /// 1. Define parameters and Parse command-line options. 
   //
  // const char *mesh_file = "../../data/beam-quad.mesh";
   int vorder = 2;

   int fun = 1;

   int n = 10;                // mesh
   int dim = 2;
   int elem = 0;
   int ser_ref_levels = 0;

   double tol = 1e-10;           // tol to check difference of matrices

   // TODO: check parsing and assign variables
   mfem::OptionsParser args(argc, argv);
   args.AddOption(&dim,
                     "-d",
                     "--dimension",
                     "Dimension of the problem (2 = 2d, 3 = 3d)");
   args.AddOption(&elem,
                     "-e",
                     "--element-type",
                     "Type of elements used (0: Quad/Hex, 1: Tri/Tet)");
   args.AddOption(&n,
                     "-n",
                     "--num-elements",
                     "Number of elements in uniform mesh.");
   args.AddOption(&ser_ref_levels,
                     "-rs",
                     "--refine-serial",
                     "Number of times to refine the mesh uniformly in serial.");
   args.AddOption(&vorder, "-ov", "--order_vel",
                     "Finite element order for velocity (polynomial degree) or -1 for"
                     " isoparametric space.");
   args.AddOption(&fun, "-f", "--test-function",
                     "Analytic function to test");
   args.AddOption(&tol,
                     "-t",
                     "--tolerance",
                     "Tolerance for checking difference in Frobenious norm of assembled matrices.");

   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(std::cout);
      return 1;
   }
   args.PrintOptions(std::cout);

   //
   /// 2. Read the (serial) mesh from the given mesh file on all processors.
   //

   //Mesh mesh(mesh_file, 1, 1);
   
   Element::Type type;
   switch (elem)
   {
      case 0: // quad
         type = (dim == 2) ? Element::QUADRILATERAL: Element::HEXAHEDRON;
         break;
      case 1: // tri
         type = (dim == 2) ? Element::TRIANGLE: Element::TETRAHEDRON;
         break;
   }
   Mesh mesh;
   switch (dim)
   {
      case 2: // 2d
         mesh = Mesh::MakeCartesian2D(n,n,type,true);	
         break;
      case 3: // 3d
         mesh = Mesh::MakeCartesian3D(n,n,n,type,true);	
         break;
   }   
   
   dim = mesh.Dimension();
   for (int l = 0; l < ser_ref_levels; l++)
   {
      mesh.UniformRefinement();
   }


   //
   /// 3. Define FE collection and spaces for velocity and pressure
   //
   dim=mesh.Dimension();

   H1_FECollection    vfec(vorder,dim);
   FiniteElementSpace vfes(&mesh,&vfec,dim);
   

   //
   /// 4. Grid functions and vectors
   //

   // initialize GridFunctions
   GridFunction v_gf(&vfes);

   VectorFunctionCoefficient *vcoeff;
   switch (fun)
   {
   case 1:
      {vcoeff = new VectorFunctionCoefficient(dim, vFun_ex1); break;}
   case 2:
      {vcoeff = new VectorFunctionCoefficient(dim, vFun_ex2); break;}
   case 3:
      {vcoeff = new VectorFunctionCoefficient(dim, vFun_ex3); break;}
   case 4:
      {vcoeff = new VectorFunctionCoefficient(dim, vFun_ex4); break;}
   default:
      break;
   }
   v_gf.ProjectCoefficient(*vcoeff);

   // initialize vectors
   Vector v;
   v_gf.GetTrueDofs(v);

   // setup GridFunctionCoefficients
   VectorGridFunctionCoefficient v_vc(&v_gf);
   

   //
   /// 5. Define Bilinear and NonlinearForms for Convective term
   //
   Array<int> empty;
   SparseMatrix Cbl;
   bool SkewSym = false;

   BilinearForm C_blForm(&vfes);
   C_blForm.AddDomainIntegrator(new VectorConvectionIntegrator(v_vc, 1.0, SkewSym)); 
   C_blForm.Assemble();
   C_blForm.FormSystemMatrix(empty,Cbl);

   NonlinearForm C_nlForm(&vfes);
   ConstantCoefficient one(1.0);
   C_nlForm.AddDomainIntegrator(new ConvectiveVectorConvectionNLFIntegrator(one));
   SparseMatrix& Cnl = dynamic_cast<SparseMatrix&>(C_nlForm.GetGradient(v));

   //
   /// 6. Check if the assembled matrices are the same
   //
   DenseMatrix Cbl_d, Cnl_d;
   Cbl.ToDenseMatrix(Cbl_d);
   Cnl.ToDenseMatrix(Cnl_d);
   Cbl_d.Add(-1,Cnl_d);
   double norm = Cbl_d.FNorm();

   bool matricesAreSame = norm < tol;

   // Print the result
   if (matricesAreSame)
   {
      std::cout << "The assembled matrices are the same." << std::endl;
   }
   else
   {
      std::cout << "The assembled matrices are different. Tol: " << norm << std::endl;
   }

   return 0;
}



void vFun_ex1(const Vector & x, Vector & v)
{
   double xi(x(0));
   double yi(x(1));
   double zi(0.0);
   if (x.Size() == 3)
   {
      zi = x(2);
   }

   v(0) = - exp(xi)*sin(yi)*cos(zi);
   v(1) = - exp(xi)*cos(yi)*cos(zi);

   if (x.Size() == 3)
   {
      v(2) = exp(xi)*sin(yi)*sin(zi);
   }
}


void vFun_ex2(const Vector & x, Vector & v)
{
   double xi(x(0));
   double yi(x(1));
   double zi(0.0);
   if (x.Size() == 3)
   {
      zi = x(2);
   }

   v(0) = 1;
   v(1) = 1;

   if (x.Size() == 3)
   {
      v(2) = 0.5;
   }
}

void vFun_ex3(const Vector &x, Vector &v)
{
   const int dim = x.Size();
   const double s = 0.1/64.;

   v = 0.0;
   v(dim-1) = s*x(0)*x(0)*(8.0-x(0));
   v(0) = -s*x(0)*x(0);
}

void vFun_ex4(const Vector &X, Vector &v)
{
   const int dim = X.Size();

   double x = X[0];
   double y = X[1];
   if( dim == 3) {
      double z = X[2];
   }

   v = 0.0;

   v(0) = -cos(y)*sin(x);
   v(1) = cos(x)*sin(y);
   if( dim == 3) { v(2) = 0; }
}
     