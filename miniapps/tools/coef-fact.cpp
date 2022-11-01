// Copyright (c) 2010-2020, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.
//
//    -------------------------------------------------------------------
//    Coef Fact Miniapp:  Visualize Coefficient fields
//    -------------------------------------------------------------------
//
//  ./coef-fact
//  ./coef-fact -c coef-fact.inp
//
#include "mfem.hpp"
#include "../common/fem_extras.hpp"
#include "../../general/text.hpp"

using namespace std;
using namespace mfem;
using namespace mfem::common;

double MyScalarFunc(const Vector &x)
{
   return x * x;
}

void MyVectorFunc(const Vector &x, Vector &v)
{
   v.SetSize(x.Size());
   v.Set(-2.0, x);
}

class MyCoefficient : public Coefficient
{
private:
   Vector k;
   mutable Vector x;

public:
   MyCoefficient(const Vector & _k) : k(_k), x(_k.Size()) {}

   double Eval(ElementTransformation &T, const IntegrationPoint &ip)
   { T.Transform(ip, x); return sin(k * x); }
};

class MyVectorCoefficient : public VectorCoefficient
{
private:
   double a;
   Vector b;
   mutable Vector x;

public:
   MyVectorCoefficient(double _a, const Vector & _b)
      : VectorCoefficient(_b.Size()), a(_a), b(_b), x(_b.Size()) {}

   void Eval(Vector & v, ElementTransformation &T, const IntegrationPoint &ip)
   { T.Transform(ip, x); v = b; v.Add(a, x); }
};

class MyCoefFactory : public CoefFactory
{
public:
   MyCoefFactory() {}

   using CoefFactory::GetScalarCoef;
   using CoefFactory::GetVectorCoef;
   using CoefFactory::GetMatrixCoef;

   Coefficient * GetScalarCoef(string &name, istream &input)
   {
      int c = -1;
      if (name == "MyCoefficient")
      {
         int dim;
         input >> dim;
         MFEM_VERIFY(dim >=1 && dim <= 3,
                     "Invalid dimension for MyCoefficient "
                     "read by MyCoefFactory");
         Vector val(dim);
         for (int i=0; i<dim; i++) { input >> val[i]; }
         c = sCoefs.Append(new MyCoefficient(val));
      }
      else
      {
         return CoefFactory::GetScalarCoef(name, input);
      }
      return sCoefs[--c];
   }

   VectorCoefficient * GetVectorCoef(string &name, istream &input)
   {
      int c = -1;
      if (name == "MyVectorCoefficient")
      {
         int dim;
         input >> dim;
         MFEM_VERIFY(dim >=1 && dim <= 3,
                     "Invalid dimension for MyVectorCoefficient "
                     "read by MyCoefFactory");
         double a;
         input >> a;
         Vector val(dim);
         for (int i=0; i<dim; i++) { input >> val[i]; }
         c = vCoefs.Append(new MyVectorCoefficient(a, val));
      }
      else
      {
         return CoefFactory::GetVectorCoef(name, input);
      }
      return vCoefs[--c];
   }
};

const char coef_str[] =
   "scalar_coef\nConstantCoefficient\n3.14\nvector_coef\nVectorConstantCoefficient\n2 2.0 1.0\nscalar_coef\nFunctionCoefficient\n0 0\nvector_coef\nVectorFunctionCoefficient\n2 0 0\nscalar_coef\nMyCoefficient\n2 2.0 1.0\nvector_coef\nMyVectorCoefficient\n2 3.0 2.0 1.0\n";

int main(int argc, char *argv[])
{
#ifdef MFEM_USE_MPI
   MPI_Session mpi;
   if (!mpi.Root()) { mfem::out.Disable(); mfem::err.Disable(); }
#endif

   // Parse command-line options.
   const char *mesh_file = "../../data/star.mesh";
   const char *coef_file = "";
   int order = 1;
   bool visualization = true;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&coef_file, "-c", "--coef-file",
                  "Set the coefficient file name.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree).");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(mfem::out);
      return 1;
   }
   args.PrintOptions(mfem::out);

   // 4. Read the (serial) mesh from the given mesh file on all processors.  We
   //    can handle triangular, quadrilateral, tetrahedral, hexahedral, surface
   //    and volume meshes with the same code.
   Mesh mesh(mesh_file, 1, 1);
   int dim = mesh.Dimension();

   // 5. Refine the serial mesh on all processors to increase the resolution. In
   //    this example we do 'ref_levels' of uniform refinement. We choose
   //    'ref_levels' to be the largest number that gives a final mesh with no
   //    more than 10,000 elements.
   {
      int ref_levels =
         (int)floor(log(10000./mesh.GetNE())/log(2.)/dim);
      for (int l = 0; l < ref_levels; l++)
      {
         mesh.UniformRefinement();
      }
   }

   H1_FESpace fespace_h1(&mesh, order, mesh.Dimension());
   ND_FESpace fespace_nd(&mesh, order, mesh.Dimension());

   GridFunction sgf(&fespace_h1);
   GridFunction vgf(&fespace_nd);

   MyCoefFactory coefFact;
   coefFact.AddExternalFunction(MyScalarFunc);
   coefFact.AddExternalFunction(MyVectorFunc);

   istream * iss = NULL;
   if (strncmp(coef_file,"",1) != 0)
   {
      iss = new ifstream(coef_file);
   }
   else
   {
      iss = new istringstream(coef_str);
   }
   skip_comment_lines(*iss, '#');

   char vishost[] = "localhost";
   int  visport   = 19916;
   socketstream s_sock, v_sock;

   string buff;
   while (*iss >> buff)
   {
      if (buff == "scalar_coef")
      {
         Coefficient * sc = coefFact.GetScalarCoef(*iss);
         sgf.ProjectCoefficient(*sc);
         if (visualization)
         {
            VisualizeField(s_sock, vishost, visport, sgf, "Scalar Coef",
                           0, 0, 275, 250);
         }
      }
      else if (buff == "vector_coef")
      {
         VectorCoefficient * vc = coefFact.GetVectorCoef(*iss);
         vgf.ProjectCoefficient(*vc);
         if (visualization)
         {
            VisualizeField(v_sock, vishost, visport, vgf, "Vector Coef",
                           275 + 3, 0, 275, 250);
         }
      }
      skip_comment_lines(*iss, '#');

      char c;
      cout << "press (q)uit or (c)ontinue --> " << flush;
      cin >> c;

      if (c != 'c')
      {
         break;
      }
   }

   return 0;
}
