#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include <random>
#include "snavier_cg.hpp"

using namespace mfem;

// Forward declarations
void vFun_ex1(const Vector & x, Vector & v);
void vFun_ex2(const Vector & x, Vector & v);

// Test
int main(int argc, char *argv[])
{
   //
   /// 1. Define parameters.
   //
   const char *mesh_file = "../../data/star.mesh";
   int porder = 1;
   int vorder = 2;
   int ser_ref_levels = 2;


   //
   /// 2. Read the (serial) mesh from the given mesh file on all processors.
   //
   Mesh mesh(mesh_file, 1, 1);
   int dim = mesh.Dimension();
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
   VectorFunctionCoefficient vcoeff(dim, vFun_ex2);
   v_gf.ProjectCoefficient(vcoeff);

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

   BilinearForm C_blForm(&vfes);
   C_blForm.AddDomainIntegrator(new VectorConvectionIntegrator(v_vc)); 
   C_blForm.Assemble();
   C_blForm.FormSystemMatrix(empty,Cbl);

   NonlinearForm C_nlForm(&vfes);
   ConstantCoefficient one(1.0);
   C_nlForm.AddDomainIntegrator(new ConvectiveVectorConvectionNLFIntegrator(one));
   SparseMatrix& Cnl = dynamic_cast<SparseMatrix&>(C_nlForm.GetGradient(v));

   //
   /// 6. Check if the assembled matrices are the same
   //
   double tol = 1e-10;

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
      v(2) = 0;
   }
}