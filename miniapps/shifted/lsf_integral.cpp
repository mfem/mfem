// Copyright (c) 2010-2024, Lawrence Livermore National Security, LLC. Produced
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
//     ---------------------------------------------------------------------
//     Miniapp: Integral of implicit domains defined by a level set function
//     ---------------------------------------------------------------------
//
//  The miniapp demonstrates an interface to the Algoim library for computing
//  volumetric and surface integrals over domains and surfaces defined
//  implicitly by a level set function. The miniapp requires MFEM to be built
//  with Blitz and Algoim libraries (see INSTALL).
//
//  Compile with: make lsf_integral
//
//  Sample runs:
//
//  Evaluates surface and volumetric integral for a circle with radius 1
//    lsf_integral -m ../../data/star-q3.mesh
//
//  Evaluates surface and volumetric integral for a level set defined
//  by y=0.5-(0.1*sin(3*pi*x+pi/2))
//    lsf_integral -ls 2 -m ../../data/inline-quad.mesh -rs 2 -o 3 -ao 3

#include "mfem.hpp"

using namespace mfem;
using namespace std;

// Level set function for sphere in 3D and circle in 2D
real_t sphere_ls(const Vector &x)
{
   real_t r2= x*x;
   return -sqrt(r2)+1.0;//the radius is 1.0
}

// Level set function for a sinusoidal wave.
// Resulting zero isocontour is at y=0.5-(0.1*sin(3*pi*x+pi/2))
real_t sinusoidal_ls(const Vector &x)
{
   real_t a1 = 20., a2 = 2., a3 = 3.;
   return tanh(a1*(x(1)-0.5) + a2*sin(a3*(x(0)-0.5)*M_PI));
}

int main(int argc, char *argv[])
{
   // Parse command-line options
   const char *mesh_file = "../../data/star-q3.mesh";
   int ser_ref_levels = 1;
   int order = 2;
   int iorder = 2; // MFEM integration points
   int aorder = 2; // Algoim integration points
   bool visualization = true;
   int print_level = 0;
   int ls_type = 1;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh", "Mesh file to use.");
   args.AddOption(&ser_ref_levels,
                  "-rs",
                  "--refine-serial",
                  "Number of times to refine the mesh uniformly in serial.");
   args.AddOption(&order,
                  "-o",
                  "--order",
                  "Order (degree) of the finite elements.");
   args.AddOption(&iorder,
                  "-io",
                  "--iorder",
                  "MFEM Integration order.");
   args.AddOption(&aorder,
                  "-ao",
                  "--aorder",
                  "Algoim Integration order.");
   args.AddOption(&ls_type,
                  "-ls",
                  "--ls-type",
                  "Level set type: 1: circle, 2 sinusoidal wave");
   args.AddOption(&visualization,
                  "-vis",
                  "--visualization",
                  "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption((&print_level), "-prt", "--print-level", "Print level.");
   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(std::cout);
      return 1;
   }
   args.PrintOptions(std::cout);

   // Read the (serial) mesh from the given mesh file.
   Mesh *mesh = new Mesh(mesh_file, 1, 1);
   int dim = mesh->Dimension();
   // Refine the mesh in serial to increase the resolution. In this example
   // we do 'ser_ref_levels' of uniform refinement, where 'ser_ref_levels' is
   // a command-line parameter.
   for (int lev = 0; lev < ser_ref_levels; lev++)
   {
      mesh->UniformRefinement();
   }

   // Define the finite element space for the level-set function.
   H1_FECollection fec(order, dim);
   FiniteElementSpace fespace(mesh, &fec, 1, Ordering::byVDIM);
   int glob_size = fespace.GetTrueVSize();
   std::cout << "Number of finite element unknowns: " << glob_size << std::endl;

   // Define the level set grid function
   GridFunction x(&fespace);

   // Define the level set coefficient
   Coefficient *ls_coeff = nullptr;
   if (ls_type == 1)
   {
      ls_coeff=new FunctionCoefficient(sphere_ls);
   }
   else if (ls_type == 2)
   {
      ls_coeff=new FunctionCoefficient(sinusoidal_ls);
   }
   else
   {
      MFEM_ABORT("Level set coefficient not defined");
   }

   // Project the coefficient onto the LS grid function
   x.ProjectCoefficient(*ls_coeff);

   if (visualization)
   {
      osockstream sock(19916, "localhost");
      sock << "solution\n";
      mesh->Print(sock);
      x.Save(sock);
      sock.send();
      sock << "window_title 'Level set'\n"
           << "window_geometry "
           << 1200 << " " << 0 << " " << 600 << " " << 600 << "\n"
           << "keys jRmmclA" << endl;
   }

   // Exact volume and area
   real_t exact_volume = -10, exact_area = -10;
   if (ls_type == 1)
   {
      if (strncmp(mesh_file,"../../data/star-q3.mesh",100) == 0)
      {
         exact_volume = M_PI;
         exact_area   = M_PI*2;
      }
      else if (strncmp(mesh_file, "../../data/inline-quad.mesh",100) == 0)
      {
         exact_volume = M_PI/4;
         exact_area   = M_PI/2;
      }
   }
   else if (ls_type == 2)
   {
      if (strncmp(mesh_file, "../../data/inline-quad.mesh",100) == 0)
      {
         exact_volume = 0.5;
         exact_area   = 1.194452300992437;
      }
   }
   (void)(&exact_area); // suppress a warning

   ElementTransformation *trans;
   const IntegrationRule* ir=nullptr;

   // Integration with Algoim
   real_t vol=0.0;

#ifdef MFEM_USE_ALGOIM
   real_t area=0.0;

   AlgoimIntegrationRules* air=new AlgoimIntegrationRules(aorder,*ls_coeff,order);

   IntegrationRule eir;
   Vector sweights;

   for (int i=0; i<fespace.GetNE(); i++)
   {
      // get the element transformation
      trans = fespace.GetElementTransformation(i);

      air->GetVolumeIntegrationRule(*trans,eir);
      for (int j = 0; j < eir.GetNPoints(); j++)
      {
         const IntegrationPoint &ip = eir.IntPoint(j);
         trans->SetIntPoint(&ip);
         vol += ip.weight * trans->Weight();
      }

      // compute the perimeter/area contribution from the element
      air->GetSurfaceIntegrationRule(*trans,eir);
      air->GetSurfaceWeights(*trans,eir,sweights);
      for (int j = 0; j < eir.GetNPoints(); j++)
      {
         const IntegrationPoint &ip = eir.IntPoint(j);
         trans->SetIntPoint(&ip);
         area += ip.weight * sweights(j) * trans->Weight();
      }
   }

   delete air;

   if (exact_volume > 0)
   {
      std::cout<<"Algoim Volume="<<vol<<" Error="<<vol-exact_volume<<std::endl;
      std::cout<<"Algoim Area="<<area<<" Error="<<area-exact_area<<std::endl;
   }
   else
   {
      std::cout<<"Algoim Volume="<<vol<<std::endl;
      std::cout<<"Algoim Area="<<area<<std::endl;
   }
#endif

   // Perform standard MFEM integration
   vol=0.0;
   for (int i=0; i<fespace.GetNE(); i++)
   {
      const FiniteElement* el=fespace.GetFE(i);
      // get the element transformation
      trans = fespace.GetElementTransformation(i);

      // compute the volume contribution from the element
      ir=&IntRules.Get(el->GetGeomType(), iorder);
      for (int j = 0; j < ir->GetNPoints(); j++)
      {
         const IntegrationPoint &ip = ir->IntPoint(j);
         trans->SetIntPoint(&ip);
         real_t vlsf=x.GetValue(*trans,ip);
         if (vlsf>=0.0)
         {
            vol += ip.weight * trans->Weight();
         }
      }
   }

   if (exact_volume > 0.0)
   {
      std::cout<<"MFEM Volume="<<vol<<" Error="<<vol-exact_volume<<std::endl;
   }
   else
   {
      std::cout<<"MFEM Volume="<<vol<<std::endl;
   }

   ParaViewDataCollection dacol("ParaViewLSF", mesh);
   dacol.SetLevelsOfDetail(order);
   dacol.RegisterField("LSF",&x);
   dacol.SetTime(1.0);
   dacol.SetCycle(1);
   dacol.Save();

   delete ls_coeff;
   delete mesh;

   return 0;
}
