#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

real_t sin_func(const Vector &x)
{
   real_t val = sin(M_PI * x.Sum());
   return val;
}

void sin_vfunc(const Vector &x, Vector &y)
{
   y.SetSize(x.Size());
   for (int i = 0; i < y.Size(); i++)
   {
      y(i) = sin(M_PI * x[i]);
   }
}

int main(int argc, char *argv[])
{
   // 1. Parse command line options.
   string mesh_file = "../data/star.mesh";
   int order = 1;


   char vishost[] = "localhost";
   int  visport   = 19916;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh", "Mesh file to use.");
   args.AddOption(&order, "-o", "--order", "Finite element polynomial degree");
   args.ParseCheck();

   Mesh mesh(mesh_file);
   int dim = mesh.Dimension();

   H1_FECollection H1fec(order, mesh.Dimension());
   RT_FECollection RTfec(order, mesh.Dimension());
   ND_FECollection NDfec(order, mesh.Dimension());

   FiniteElementSpace H1fes(&mesh, &H1fec);
   FiniteElementSpace H1vecfes(&mesh, &H1fec, dim, Ordering::byVDIM);
   FiniteElementSpace RTfes(&mesh, &RTfec);
   FiniteElementSpace NDfes(&mesh, &NDfec);


   H1_Trace_FECollection H1trace_fec(order, mesh.Dimension());
   RT_Trace_FECollection RTtrace_fec(order, mesh.Dimension());
   ND_Trace_FECollection NDtrace_fec(order, mesh.Dimension());

   FiniteElementSpace H1trace_fes(&mesh, &H1trace_fec);
   FiniteElementSpace H1trace_vecfes(&mesh, &H1trace_fec, dim, Ordering::byVDIM);
   FiniteElementSpace RTtrace_fes(&mesh, &RTtrace_fec);
   FiniteElementSpace NDtrace_fes(&mesh, &NDtrace_fec);

   FunctionCoefficient cf(sin_func);
   VectorFunctionCoefficient vec_cf(dim, sin_vfunc);

   GridFunction x_H1(&H1fes); x_H1 = 0.0;
   GridFunction x_vecH1(&H1vecfes); x_vecH1 = 0.0;
   GridFunction x_RT(&RTfes); x_RT = 0.0;
   GridFunction x_ND(&NDfes); x_ND = 0.0;

   x_H1.ProjectCoefficient(cf);
   x_vecH1.ProjectCoefficient(vec_cf);
   x_RT.ProjectCoefficient(vec_cf);
   x_ND.ProjectCoefficient(vec_cf);

   GridFunction trace_x_H1(&H1trace_fes); trace_x_H1 = 0.0;
   GridFunction trace_x_vecH1(&H1trace_vecfes); trace_x_vecH1 = 0.0;
   GridFunction trace_x_RT(&RTtrace_fes); trace_x_RT = 0.0;
   GridFunction trace_x_ND(&NDtrace_fes); trace_x_ND = 0.0;

   trace_x_H1.ProjectFaceCoefficient(cf);
   trace_x_vecH1.ProjectFaceCoefficient(vec_cf);
   trace_x_RT.ProjectFaceCoefficientNormal(vec_cf);
   trace_x_ND.ProjectFaceCoefficientTangent(vec_cf);


   Array<int> vdofs, trace_vdofs;
   GridFunction trace_x_H1_mapped(&H1fes); trace_x_H1_mapped = 0.0;
   GridFunction trace_x_vecH1_mapped(&H1vecfes); trace_x_vecH1_mapped = 0.0;
   GridFunction trace_x_RT_mapped(&RTfes); trace_x_RT_mapped = 0.0;
   GridFunction trace_x_ND_mapped(&NDfes); trace_x_ND_mapped = 0.0;
   Vector values;
   for (int i = 0; i<mesh.GetNE(); i++)
   {
      H1fes.GetElementInteriorVDofs(i, vdofs);
      x_H1.SetSubVector(vdofs, 0.0);
      H1trace_fes.GetElementVDofs(i, trace_vdofs);
      trace_x_H1.GetSubVector(trace_vdofs, values);
      trace_x_H1_mapped.SetSubVector(trace_vdofs,values);

      H1vecfes.GetElementInteriorVDofs(i, vdofs);
      x_vecH1.SetSubVector(vdofs, 0.0);
      H1trace_vecfes.GetElementVDofs(i, trace_vdofs);
      trace_x_vecH1.GetSubVector(trace_vdofs, values);
      trace_x_vecH1_mapped.SetSubVector(trace_vdofs,values);

      RTfes.GetElementInteriorVDofs(i, vdofs);
      x_RT.SetSubVector(vdofs, 0.0);
      RTtrace_fes.GetElementVDofs(i, trace_vdofs);
      trace_x_RT.GetSubVector(trace_vdofs, values);
      trace_x_RT_mapped.SetSubVector(trace_vdofs,values);

      NDfes.GetElementInteriorVDofs(i, vdofs);
      x_ND.SetSubVector(vdofs, 0.0);
      NDtrace_fes.GetElementVDofs(i, trace_vdofs);
      trace_x_ND.GetSubVector(trace_vdofs, values);
      trace_x_ND_mapped.SetSubVector(trace_vdofs,values);
   }

   socketstream sol_H1_sock(vishost, visport);
   sol_H1_sock.precision(8);
   sol_H1_sock << "solution\n" << mesh << x_H1
               << "window_title 'H1 Field'" << flush;

   socketstream sol_H1_tr_sock(vishost, visport);
   sol_H1_tr_sock.precision(8);
   sol_H1_tr_sock << "solution\n" << mesh << trace_x_H1_mapped
                  << "window_title 'H1 Trace'" << flush;

   socketstream sol_vecH1_sock(vishost, visport);
   sol_vecH1_sock.precision(8);
   sol_vecH1_sock << "solution\n" << mesh << x_vecH1
                  << "window_title 'Vector H1 Field'" << flush;

   socketstream sol_vecH1_tr_sock(vishost, visport);
   sol_vecH1_tr_sock.precision(8);
   sol_vecH1_tr_sock << "solution\n" << mesh << trace_x_vecH1_mapped
                     << "window_title 'Vector H1 Trace'" << flush;

   socketstream sol_RT_sock(vishost, visport);
   sol_RT_sock.precision(8);
   sol_RT_sock << "solution\n" << mesh << x_RT
               << "window_title 'RT Field'" << flush;

   socketstream sol_RT_tr_sock(vishost, visport);
   sol_RT_tr_sock.precision(8);
   sol_RT_tr_sock << "solution\n" << mesh << trace_x_RT_mapped
                  << "window_title 'RT Trace'" << flush;

   socketstream sol_ND_sock(vishost, visport);
   sol_ND_sock.precision(8);
   sol_ND_sock << "solution\n" << mesh << x_ND
               << "window_title 'ND Field'" << flush;

   socketstream sol_ND_tr_sock(vishost, visport);
   sol_ND_tr_sock.precision(8);
   sol_ND_tr_sock << "solution\n" << mesh << trace_x_ND_mapped
                  << "window_title 'ND Trace'" << flush;

   trace_x_H1_mapped -= x_H1;
   cout << "||x_H1_trace - x_H1||₂ : " << trace_x_H1_mapped.Norml2() << endl;

   trace_x_vecH1_mapped -= x_vecH1;
   cout << "||x_vecH1_trace - x_vecH1||₂ : " << trace_x_vecH1_mapped.Norml2() <<
        endl;

   trace_x_RT_mapped -= x_RT;
   cout << "||x_RT_trace - x_RT||₂ : " << trace_x_RT_mapped.Norml2() << endl;


   trace_x_ND_mapped -= x_ND;
   cout << "||x_ND_trace - x_ND||₂ : " << trace_x_ND_mapped.Norml2() << endl;


   return 0;
}