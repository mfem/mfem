// Copyright (c) 2010-2021, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#include "mfem.hpp"
#include "unit_tests.hpp"

using namespace mfem;

namespace project_bdr
{

double func_1D_lin(const Vector &x)
{
   return x[0] + 1.0;
}

double func_2D_lin(const Vector &x)
{
   return x[0] + 2.0 * x[1];
}

double func_3D_lin(const Vector &x)
{
   return x[0] + 2.0 * x[1] + 3.0 * x[2];
}

void Func_1D_lin(const Vector &x, Vector &v)
{
   v.SetSize(1);
   v[0] =  1.234 * x[0] - 2.357;
}

void Func_2D_lin(const Vector &x, Vector &v)
{
   v.SetSize(2);
   v[0] =  1.234 * x[0] - 2.357 * x[1];
   v[1] =  2.537 * x[0] + 4.321 * x[1];
}

void Func_3D_lin(const Vector &x, Vector &v)
{
   v.SetSize(3);
   v[0] =  1.234 * x[0] - 2.357 * x[1] + 3.572 * x[2];
   v[1] =  2.537 * x[0] + 4.321 * x[1] - 1.234 * x[2];
   v[2] = -2.572 * x[0] + 1.321 * x[1] + 3.234 * x[2];
}

TEST_CASE("ProjectBdrCoefficient",
          "[GridFunction]"
          "[VectorGridFunctionCoefficient]")
{
   int log = 1;
   int n = 1;
   int order = 1;
   int npts = 0;

   double tol = 1e-6;

   for (int type = (int)Element::SEGMENT;
        type <= (int)Element::HEXAHEDRON; type++)
   {
     int dim = (type == (int)Element::SEGMENT) ? 1 :
       ((type < (int)Element::TETRAHEDRON) ? 2 : 3);

     Mesh mesh = (dim == 1) ?
       Mesh::MakeCartesian1D(n, 2.0) :
       ((dim == 2) ?
	Mesh::MakeCartesian2D(n, n, (Element::Type)type, true, 2.0, 3.0) :
	Mesh::MakeCartesian3D(n, n, n, (Element::Type)type, 2.0, 3.0, 5.0));

     FunctionCoefficient funcCoef((dim == 1) ?
				  func_1D_lin :
				  ((dim == 2) ?
				   func_2D_lin :
				   func_3D_lin));

     VectorFunctionCoefficient FuncCoef(dim, (dim == 1) ?
					Func_1D_lin :
					((dim == 2) ?
					 Func_2D_lin :
					 Func_3D_lin));
     
      SECTION("ProjectBdrCoefficient tests for element type " +
              std::to_string(type))
      {
         H1_FECollection h1_fec(order, dim);
         FiniteElementSpace h1_fespace(&mesh,  &h1_fec);
         FiniteElementSpace h1v_fespace(&mesh,  &h1_fec, dim);
         GridFunction h1_x( &h1_fespace);
         GridFunction h1v_x( &h1v_fespace);

         GridFunctionCoefficient h1_xCoef( &h1_x);
         VectorGridFunctionCoefficient h1v_xCoef( &h1v_x);

         Array<int> bdr_marker(6);

	 double f_val = 0.0;
	 double h1_val = 0.0;

	 Vector F_val(dim);
	 Vector h1v_val(dim);	 
	 
         for (int b = 1; b<=6; b++)
         {
            bdr_marker = 0;
            bdr_marker[b-1] = 1;

            h1_x = 0.0;
            h1_x.ProjectBdrCoefficient(funcCoef, bdr_marker);

            h1v_x = 0.0;
            h1v_x.ProjectBdrCoefficient(FuncCoef, bdr_marker);

            for (int be = 0; be < mesh.GetNBE(); be++)
            {
               Element *e = mesh.GetBdrElement(be);
               if (e->GetAttribute() != b) { continue; }

               ElementTransformation *T = mesh.GetBdrElementTransformation(be);
               const FiniteElement   *fe = h1_fespace.GetBE(be);
               const IntegrationRule &ir = IntRules.Get(fe->GetGeomType(),
                                                        2*order + 2);

               double h1_err = 0.0;
               double h1v_err = 0.0;

               for (int j=0; j<ir.GetNPoints(); j++)
               {
                  npts++;
                  const IntegrationPoint &ip = ir.IntPoint(j);
                  T->SetIntPoint(&ip);

                  f_val = funcCoef.Eval(*T, ip);
                  h1_val = h1_xCoef.Eval(*T, ip);

		  FuncCoef.Eval(F_val, *T, ip);
		  h1v_xCoef.Eval(h1v_val, *T, ip);
		  
                  double h1_dist = h1_val - f_val;
                  double h1v_dist = Distance(F_val, h1v_val, dim);

                  h1_err += h1_dist;
                  h1v_err += h1v_dist;

                  if (log > 0 && h1_dist > tol)
                  {
                     std::cout << be << ":" << j << " h1 "
                               << f_val << " vs. "
                               << h1_val << ", " << h1_dist << std::endl;
                  }
                  if (log > 0 && h1v_dist > tol)
                  {
                     std::cout << be << ":" << j << " h1v ("
                               << F_val[0];
		     if (dim > 1) std::cout << "," << F_val[1];
		     if (dim > 2) std::cout << "," << F_val[2];
		     std::cout << ") vs. ("
                               << h1v_val[0];
		     if (dim > 1) std::cout << "," << h1v_val[1];
		     if (dim > 2) std::cout << "," << h1v_val[2];
		     std::cout << "), " << h1v_dist << std::endl;
                  }
               }
               h1_err  /= ir.GetNPoints();
               h1v_err  /= ir.GetNPoints();

               REQUIRE( h1_err == MFEM_Approx(0.0));
               REQUIRE( h1v_err == MFEM_Approx(0.0));
            }
         }
      }
   }
}

TEST_CASE("ProjectBdrCoefficientTangent",
          "[GridFunction]"
          "[VectorGridFunctionCoefficient]")
{
   int log = 1;
   int n = 1;
   int order = 1;
   int npts = 0;

   double tol = 1e-6;

   for (int type = (int)Element::TRIANGLE;
        type <= (int)Element::HEXAHEDRON; type++)
   {
      int dim = (type < (int)Element::TETRAHEDRON) ? 2 : 3;
      
      Mesh mesh = (dim == 2) ?
	Mesh::MakeCartesian2D(n, n, (Element::Type)type, true, 2.0, 3.0) :
	Mesh::MakeCartesian3D(n, n, n, (Element::Type)type, 2.0, 3.0, 5.0);

      VectorFunctionCoefficient funcCoef(dim, (dim == 2) ?
					 Func_2D_lin : Func_3D_lin);

      SECTION("ProjectBdrCoefficientTangent tests for element type " +
              std::to_string(type))
      {
         ND_FECollection  nd_fec(order+1, dim);

         FiniteElementSpace  nd_fespace(&mesh,  &nd_fec);

         GridFunction  nd_x( &nd_fespace);

         VectorGridFunctionCoefficient  nd_xCoef( &nd_x);

         Array<int> bdr_marker(2*dim);

         Vector normal(dim);
         Vector  f_val(dim);
         Vector nd_val(dim);
         Vector    nxd(3);

         for (int b = 1; b<=2*dim; b++)
         {
            bdr_marker = 0;
            bdr_marker[b-1] = 1;

            nd_x = 0.0;
            nd_x.ProjectBdrCoefficientTangent(funcCoef, bdr_marker);

            for (int be = 0; be < mesh.GetNBE(); be++)
            {
               Element *e = mesh.GetBdrElement(be);
               if (e->GetAttribute() != b) { continue; }

               ElementTransformation *T = mesh.GetBdrElementTransformation(be);
               const FiniteElement   *fe = nd_fespace.GetBE(be);
               const IntegrationRule &ir = IntRules.Get(fe->GetGeomType(),
                                                        2*order + 2);

               double nd_err = 0.0;

               for (int j=0; j<ir.GetNPoints(); j++)
               {
                  npts++;
                  const IntegrationPoint &ip = ir.IntPoint(j);
                  T->SetIntPoint(&ip);

                  CalcOrtho(T->Jacobian(), normal);

                  funcCoef.Eval(f_val, *T, ip);
                  nd_xCoef.Eval(nd_val, *T, ip);

                  nd_val -= f_val;

		  if (dim == 2)
		    {
		      nxd[0] = 0.0;
		      nxd[1] = 0.0;
		    }
		  else
		    {
		      nxd[0] = normal[1] * nd_val[2] - normal[2] * nd_val[1];
		      nxd[1] = normal[2] * nd_val[0] - normal[0] * nd_val[2];
		    }
		  nxd[2] = normal[0] * nd_val[1] - normal[1] * nd_val[0];

                  double nd_dist = nxd.Norml2();

                  nd_err += nd_dist;

                  if (log > 0 && nd_dist > tol)
                  {
                     std::cout << be << ":" << j << " nd ("
                               << f_val[0] << "," << f_val[1];
		     if (dim > 2) std::cout << "," << f_val[2];
		     std::cout << ") vs. ("
                               << nd_val[0] << "," << nd_val[1];
		     if (dim > 2) std::cout << "," << nd_val[2];
		     std::cout << ") " << nd_dist << std::endl;
                  }
               }
               nd_err  /= ir.GetNPoints();

               REQUIRE( nd_err == MFEM_Approx(0.0));
            }
         }
      }
   }
}

TEST_CASE("ProjectBdrCoefficientNormal",
          "[GridFunction]"
          "[VectorGridFunctionCoefficient]")
{
   int log = 1;
   int n = 1;
   int order = 1;
   int npts = 0;

   double tol = 1e-6;

   for (int type = (int)Element::TRIANGLE;
        type <= (int)Element::HEXAHEDRON; type++)
   {
      int dim = (type < (int)Element::TETRAHEDRON) ? 2 : 3;
      
      Mesh mesh = (dim == 2) ?
	Mesh::MakeCartesian2D(n, n, (Element::Type)type, true, 2.0, 3.0) :
	Mesh::MakeCartesian3D(n, n, n, (Element::Type)type, 2.0, 3.0, 5.0);

      VectorFunctionCoefficient funcCoef(dim, (dim == 2) ?
					 Func_2D_lin : Func_3D_lin);

      SECTION("ProjectBdrCoefficientNormal tests for element type " +
              std::to_string(type))
      {
         H1_FECollection h1_fec(order, dim);
         RT_FECollection rt_fec(order+1, dim);

         FiniteElementSpace h1v_fespace(&mesh,  &h1_fec, dim);
         FiniteElementSpace  rt_fespace(&mesh,  &rt_fec);
	 
         GridFunction h1v_x( &h1v_fespace);
         GridFunction  rt_x( &rt_fespace);

         VectorGridFunctionCoefficient  h1v_xCoef( &h1v_x);
         VectorGridFunctionCoefficient  rt_xCoef( &rt_x);

         Array<int> bdr_marker(6);

         Vector normal(dim);
         Vector   f_val(dim);
         Vector h1v_val(dim);
         Vector  rt_val(dim);

         for (int b = 1; b<=6; b++)
         {
            bdr_marker = 0;
            bdr_marker[b-1] = 1;

            h1v_x = 0.0;
            h1v_x.ProjectBdrCoefficientNormal(funcCoef, bdr_marker);

            rt_x = 0.0;
            rt_x.ProjectBdrCoefficientNormal(funcCoef, bdr_marker);

            for (int be = 0; be < mesh.GetNBE(); be++)
            {
               Element *e = mesh.GetBdrElement(be);
               if (e->GetAttribute() != b) { continue; }

               ElementTransformation *T = mesh.GetBdrElementTransformation(be);
               const FiniteElement   *fe = rt_fespace.GetBE(be);
               const IntegrationRule &ir = IntRules.Get(fe->GetGeomType(),
                                                        2*order + 2);

               double h1v_err = 0.0;
               double  rt_err = 0.0;

               for (int j=0; j<ir.GetNPoints(); j++)
               {
                  npts++;
                  const IntegrationPoint &ip = ir.IntPoint(j);
                  T->SetIntPoint(&ip);

                  CalcOrtho(T->Jacobian(), normal);

                  funcCoef.Eval(f_val, *T, ip);
                  h1v_xCoef.Eval(h1v_val, *T, ip);
                  rt_xCoef.Eval(rt_val, *T, ip);

                  h1v_val -= f_val;
                  rt_val -= f_val;

                  double h1v_dist = h1v_val * normal;
                  double  rt_dist =  rt_val * normal;

                  h1v_err += h1v_dist;
                  rt_err += rt_dist;

                  if (log > 0 && h1v_dist > tol)
                  {
                     std::cout << be << ":" << j << " h1v ("
                               << f_val[0] << "," << f_val[1];
		     if (dim > 2) std::cout << "," << f_val[2];
		     std::cout << ") vs. ("
                               << h1v_val[0] << "," << h1v_val[1];
		     if (dim > 2) std::cout << "," << h1v_val[2];
		     std::cout << ") " << h1v_dist << std::endl;
                  }
                  if (log > 0 && rt_dist > tol)
                  {
                     std::cout << be << ":" << j << " rt ("
                               << f_val[0] << "," << f_val[1];
		     if (dim > 2) std::cout << "," << f_val[2];
		     std::cout << ") vs. ("
                               << rt_val[0] << "," << rt_val[1];
		     if (dim > 2) std::cout << "," << rt_val[2];
		     std::cout << ") " << rt_dist << std::endl;
                  }
               }
               h1v_err  /= ir.GetNPoints();
               rt_err  /= ir.GetNPoints();

               REQUIRE( h1v_err == MFEM_Approx(0.0));
               REQUIRE(  rt_err == MFEM_Approx(0.0));
            }
         }
      }
   }
}

} // namespace project_bdr
