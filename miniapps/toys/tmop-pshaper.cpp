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
//           ------------------------------------------------------------------
//           TMOP-Shaper Miniapp: Convert an image to a mesh to match the image
//           ------------------------------------------------------------------
//
// This miniapp is a specialized version of the TMOP miniapp that converts an
// input image to a mesh optimized to capture the image.It allows the fast
// approximate meshing of any domain for which there is an image.

// The input to image should be in 8-bit grayscale PGM format. You can use a
// number of image manipulation tools, such as GIMP (gimp.org) and ImageMagick's
// convert utility (imagemagick.org/script/convert.php) to convert your image to
// this format as a pre-processing step, e.g.:
//
//   /usr/bin/convert australia.svg -compress none -depth 8 australia.pgm
//
// Compile with: make tmop-pshaper
//
// Sample runs:  mpirun -np 4 tmop-pshaper -i australia.pgm -nc 3 -rs 4



#include "mfem.hpp"
#include "../common/mfem-common.hpp"
#include <iostream>
#include <fstream>
#include "../meshing/mesh-optimizer.hpp"
#include "mondrian.hpp"

using namespace mfem;
using namespace std;

// Get a GridFunction for material indicator based on nodal positions.
void GetMaterialIndicator(GridFunction &ind, GridFunction &x,
                          const ParsePGM &pgm,
                          int NC, Vector &xmin, Vector &xmax);

// Get Size targets based on the material indicator
void GetSize(GridFunction &size, GridFunction &ind, bool flip,
             const int size_type);


int main (int argc, char *argv[])
{
   // 0. Initialize MPI.
   int num_procs, myid;
   MPI_Init(&argc, &argv);
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
   MPI_Comm_rank(MPI_COMM_WORLD, &myid);

   // 1. Set the method's default parameters.
   const char *mesh_file = "../meshing/square01.mesh";
   const char *img_file = "australia.pgm";
   int ncolors           = 3;
   int mesh_poly_deg     = 2;
   int rs_levels         = 0;
   int rp_levels         = 0;
   int metric_id         = -1;
   int size_type         = 0;
   int quad_type         = 1;
   int quad_order        = 4;
   int solver_iter       = 100;
   double solver_rtol    = 1e-10;
   int max_lin_iter      = 100;
   bool move_bnd         = false;
   int combomet          = 0;
   bool normalization    = false;
   bool visualization    = true;
   int verbosity_level   = 0;
   bool fdscheme         = false;
   int adapt_eval        = 0;

   // 2. Parse command-line options.
   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&img_file, "-i", "--img",
                  "Input image.");
   args.AddOption(&mesh_poly_deg, "-o", "--order",
                  "Polynomial degree of mesh finite element space.");
   args.AddOption(&ncolors, "-nc", "--num-colors",
                  "Number of colors considered (1-256, based on binning).");
   args.AddOption(&size_type, "-sz", "--size-type",
                  "Set size based on 0: material color, 1: derivative of material color.");
   args.AddOption(&rs_levels, "-rs", "--refine-serial",
                  "Number of times to refine the mesh uniformly in serial.");
   args.AddOption(&rp_levels, "-rp", "--refine-parallel",
                  "Number of times to refine the mesh uniformly in parallel.");
   args.AddOption(&metric_id, "-mid", "--metric-id",
                  "Mesh optimization metric:\n\t"
                  "1  : |T|^2                          -- 2D shape\n\t"
                  "2  : 0.5|T|^2/tau-1                 -- 2D shape (condition number)\n\t"
                  "7  : |T-T^-t|^2                     -- 2D shape+size\n\t"
                  "9  : tau*|T-T^-t|^2                 -- 2D shape+size\n\t"
                  "50 : 0.5|T^tT|^2/tau^2-1            -- 2D shape\n\t"
                  "55 : (tau-1)^2                      -- 2D size\n\t"
                  "56 : 0.5(sqrt(tau)-1/sqrt(tau))^2   -- 2D size\n\t"
                  "58 : |T^tT|^2/(tau^2)-2*|T|^2/tau+2 -- 2D shape\n\t"
                  "77 : 0.5(tau-1/tau)^2               -- 2D size\n\t");
   args.AddOption(&quad_order, "-qo", "--quad_order",
                  "Order of the quadrature rule.");
   args.AddOption(&solver_iter, "-ni", "--newton-iters",
                  "Maximum number of Newton iterations.");
   args.AddOption(&solver_rtol, "-rtol", "--newton-rel-tolerance",
                  "Relative tolerance for the Newton solver.");
   args.AddOption(&max_lin_iter, "-li", "--lin-iter",
                  "Maximum number of iterations in the linear solve.");
   args.AddOption(&move_bnd, "-bnd", "--move-boundary", "-fix-bnd",
                  "--fix-boundary",
                  "Enable motion along horizontal and vertical boundaries.");
   args.AddOption(&combomet, "-cmb", "--combo-type",
                  "Combination of metrics options:"
                  "0: Use single metric\n\t"
                  "1: Shape + adapted size given discretely; shared target");
   args.AddOption(&normalization, "-nor", "--normalization", "-no-nor",
                  "--no-normalization",
                  "Make all terms in the optimization functional unitless.");
   args.AddOption(&fdscheme, "-fd", "--fd_approximation",
                  "-no-fd", "--no-fd-approx",
                  "Enable finite difference based derivative computations.");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&verbosity_level, "-vl", "--verbosity-level",
                  "Set the verbosity level - 0, 1, or 2.");
   args.Parse();
   if (!args.Good())
   {
      if (myid == 0) { args.PrintUsage(cout); }
      return 1;
   }
   if (myid == 0) { args.PrintOptions(cout); }

   // Read the image
   ParsePGM pgm(img_file);

   // 3. Initialize and refine the starting mesh.
   Mesh *mesh = new Mesh(mesh_file, 1, 1, false);
   for (int lev = 0; lev < rs_levels; lev++)
   {
      mesh->UniformRefinement();
   }
   const int dim = mesh->Dimension();
   if (myid == 0)
   {
      cout << "Mesh curvature: ";
      if (mesh->GetNodes()) { cout << mesh->GetNodes()->OwnFEC()->Name(); }
      else { cout << "(NONE)"; }
      cout << endl;
   }

   Vector xmin, xmax;
   mesh->GetBoundingBox(xmin, xmax);

   ParMesh *pmesh = new ParMesh(MPI_COMM_WORLD, *mesh);

   delete mesh;
   for (int lev = 0; lev < rp_levels; lev++)
   {
      pmesh->UniformRefinement();
   }

   // 4. Define a finite element space on the mesh. Here we use vector finite
   //    elements which are tensor products of quadratic finite elements. The
   //    number of components in the vector finite element space is specified by
   //    the last parameter of the FiniteElementSpace constructor.
   FiniteElementCollection *fec;
   if (mesh_poly_deg <= 0)
   {
      fec = new QuadraticPosFECollection;
      mesh_poly_deg = 2;
   }
   else { fec = new H1_FECollection(mesh_poly_deg, dim); }
   ParFiniteElementSpace *pfespace = new ParFiniteElementSpace(pmesh, fec, dim);

   // 5. Make the mesh curved based on the above finite element space. This
   //    means that we define the mesh elements through a fespace-based
   //    transformation of the reference element.
   pmesh->SetNodalFESpace(pfespace);

   // 6. Set up an empty right-hand side vector b, which is equivalent to b=0.
   Vector b(0);

   // 7. Get the mesh nodes (vertices and other degrees of freedom in the finite
   //    element space) as a finite element grid function in fespace. Note that
   //    changing x automatically changes the shapes of the mesh elements.
   ParGridFunction x(pfespace);
   pmesh->SetNodalGridFunction(&x);
   x.SetTrueVector();
   x.SetFromTrueVector();

   // 10. Save the starting (prior to the optimization) mesh to a file. This
   //     output can be viewed later using GLVis: "glvis -m perturbed -np
   //     num_mpi_tasks".
   {
      ostringstream mesh_name;
      mesh_name << "perturbed.mesh";
      ofstream mesh_ofs(mesh_name.str().c_str());
      mesh_ofs.precision(8);
      pmesh->PrintAsOne(mesh_ofs);
   }

   // 11. Store the starting (prior to the optimization) positions.
   ParGridFunction x0(pfespace);
   x0 = x;

   // 12. Form the integrator that uses the chosen metric and target.
   double tauval = -0.1;
   TMOP_QualityMetric *metric = NULL;
   if ( metric_id == -1 )
   {
      if ( combomet == 0)
      {
         metric_id = 7;
      }
      else
      {
         metric_id = 77;
      }
   }
   switch (metric_id)
   {
      case 1: metric = new TMOP_Metric_001; break;
      case 2: metric = new TMOP_Metric_002; break;
      case 7: metric = new TMOP_Metric_007; break;
      case 9: metric = new TMOP_Metric_009; break;
      case 14: metric = new TMOP_Metric_SSA2D; break;
      case 50: metric = new TMOP_Metric_050; break;
      case 55: metric = new TMOP_Metric_055; break;
      case 56: metric = new TMOP_Metric_056; break;
      case 58: metric = new TMOP_Metric_058; break;
      case 77: metric = new TMOP_Metric_077; break;
      case 85: metric = new TMOP_Metric_085; break;
      default:
         if (myid == 0) { cout << "Unknown metric_id: " << metric_id << endl; }
         return 3;
   }
   TargetConstructor::TargetType target_t;
   TargetConstructor *target_c = NULL;
   HessianCoefficient *adapt_coeff = NULL;
   H1_FECollection ind_fec(mesh_poly_deg, dim);
   ParFiniteElementSpace ind_fes(pmesh, &ind_fec);
   ParGridFunction size(&ind_fes), ind(&ind_fes);

   // Get material indicator GridFunction
   GetMaterialIndicator(ind, x, pgm, ncolors, xmin, xmax);
   bool flip = strcmp(img_file,"australia.pgm") == 0 ? 1 : 0;
   GetSize(size, ind, flip, size_type );

   if (visualization)
   {
      socketstream sock;
      if (myid == 0)
      {
         sock.open("localhost", 19916);
         sock << "solution\n";
      }
      pmesh->PrintAsOne(sock);
      ind.SaveAsOne(sock);
      if (myid == 0)
      {
         sock << "window_title 'Material indicator'\n"
              << "window_geometry "
              <<  0 << " " << 0 << " " << 600 << " " << 600 << "\n"
              << "keys jRmclA" << endl;
      }
   }

   target_t = TargetConstructor::GIVEN_SHAPE_AND_SIZE;
   DiscreteAdaptTC *tc = new DiscreteAdaptTC(target_t);
   tc->SetAdaptivityEvaluator(new AdvectorCG);
   tc->SetParDiscreteTargetSize(size);
   target_c = tc;

   if (target_c == NULL)
   {
      target_c = new TargetConstructor(target_t, MPI_COMM_WORLD);
   }
   target_c->SetNodes(x0);
   TMOP_Integrator *he_nlf_integ= new TMOP_Integrator(metric, target_c);
   if (fdscheme) { he_nlf_integ->EnableFiniteDifferences(x); }

   // 13. Setup the quadrature rule for the non-linear form integrator.
   const IntegrationRule *ir = NULL;
   const int geom_type = pfespace->GetFE(0)->GetGeomType();
   ir = &IntRulesLo.Get(geom_type, quad_order);
   if (myid == 0)
   { cout << "Quadrature points per cell: " << ir->GetNPoints() << endl; }
   he_nlf_integ->SetIntegrationRule(*ir);

   if (normalization) { he_nlf_integ->ParEnableNormalization(x0); }

   // 15. Setup the final NonlinearForm (which defines the integral of interest,
   //     its first and second derivatives). Here we can use a combination of
   //     metrics, i.e., optimize the sum of two integrals, where both are
   //     scaled by used-defined space-dependent weights.  Note that there are
   //     no command-line options for the weights and the type of the second
   //     metric; one should update those in the code.
   ParNonlinearForm a(pfespace);
   ConstantCoefficient *coeff1 = NULL;
   TMOP_QualityMetric *metric2 = NULL;
   TargetConstructor *target_c2 = NULL;
   ConstantCoefficient *coeff2 = NULL;
   double w1 = 0.5;

   if (combomet > 0)
   {
      // First metric.
      coeff1 = new ConstantCoefficient(w1);
      he_nlf_integ->SetCoefficient(*coeff1);

      // Second metric.
      metric2 = new TMOP_Metric_002;
      TMOP_Integrator *he_nlf_integ2 = NULL;
      he_nlf_integ2 = new TMOP_Integrator(metric2, target_c);
      he_nlf_integ2->SetIntegrationRule(*ir);
      if (fdscheme) { he_nlf_integ2->EnableFiniteDifferences(x); }
      coeff2 = new ConstantCoefficient(1-w1);
      he_nlf_integ2->SetCoefficient(*coeff2);

      TMOPComboIntegrator *combo = new TMOPComboIntegrator;
      combo->AddTMOPIntegrator(he_nlf_integ);
      combo->AddTMOPIntegrator(he_nlf_integ2);
      if (normalization) { combo->ParEnableNormalization(x0); }

      a.AddDomainIntegrator(combo);
   }
   else { a.AddDomainIntegrator(he_nlf_integ); }

   const double init_energy = a.GetParGridFunctionEnergy(x);

   // 17. Fix all boundary nodes, or fix only a given component depending on the
   //     boundary attributes of the given mesh.  Attributes 1/2/3 correspond to
   //     fixed x/y/z components of the node.  Attribute 4 corresponds to an
   //     entirely fixed node.  Other boundary attributes do not affect the node
   //     movement boundary conditions.
   if (move_bnd == false)
   {
      Array<int> ess_bdr(pmesh->bdr_attributes.Max());
      ess_bdr = 1;
      a.SetEssentialBC(ess_bdr);
   }
   else
   {
      const int nd  = pfespace->GetBE(0)->GetDof();
      int n = 0;
      for (int i = 0; i < pmesh->GetNBE(); i++)
      {
         const int attr = pmesh->GetBdrElement(i)->GetAttribute();
         MFEM_VERIFY(!(dim == 2 && attr == 3),
                     "Boundary attribute 3 must be used only for 3D meshes. "
                     "Adjust the attributes (1/2/3/4 for fixed x/y/z/all "
                     "components, rest for free nodes), or use -fix-bnd.");
         if (attr == 1 || attr == 2 || attr == 3) { n += nd; }
         if (attr == 4) { n += nd * dim; }
      }
      Array<int> ess_vdofs(n), vdofs;
      n = 0;
      for (int i = 0; i < pmesh->GetNBE(); i++)
      {
         const int attr = pmesh->GetBdrElement(i)->GetAttribute();
         pfespace->GetBdrElementVDofs(i, vdofs);
         if (attr == 1) // Fix x components.
         {
            for (int j = 0; j < nd; j++)
            { ess_vdofs[n++] = vdofs[j]; }
         }
         else if (attr == 2) // Fix y components.
         {
            for (int j = 0; j < nd; j++)
            { ess_vdofs[n++] = vdofs[j+nd]; }
         }
         else if (attr == 3) // Fix z components.
         {
            for (int j = 0; j < nd; j++)
            { ess_vdofs[n++] = vdofs[j+2*nd]; }
         }
         else if (attr == 4) // Fix all components.
         {
            for (int j = 0; j < vdofs.Size(); j++)
            { ess_vdofs[n++] = vdofs[j]; }
         }
      }
      a.SetEssentialVDofs(ess_vdofs);
   }

   // 18. As we use the Newton method to solve the resulting nonlinear system,
   //     here we setup the linear solver for the system's Jacobian.
   Solver *S = NULL;
   const double linsol_rtol = 1e-12;
   MINRESSolver *minres = new MINRESSolver(MPI_COMM_WORLD);
   minres->SetMaxIter(max_lin_iter);
   minres->SetRelTol(linsol_rtol);
   minres->SetAbsTol(0.0);
   minres->SetPrintLevel(verbosity_level >= 2 ? 3 : -1);
   S = minres;

   // Perform the nonlinear optimization.
   TMOPNewtonSolver solver(pfespace->GetComm(), *ir, 0);
   solver.SetPreconditioner(*S);
   solver.SetMaxIter(solver_iter);
   solver.SetRelTol(solver_rtol);
   solver.SetAbsTol(0.0);
   solver.SetPrintLevel(verbosity_level >= 1 ? 1 : -1);
   solver.SetOperator(a);
   solver.Mult(b, x.GetTrueVector());
   x.SetFromTrueVector();
   if (myid == 0 && solver.GetConverged() == false)
   {
      cout << "Nonlinear solver: rtol = " << solver_rtol << " not achieved.\n";
   }

   // 21. Save the optimized mesh to a file. This output can be viewed later
   //     using GLVis: "glvis -m optimized -np num_mpi_tasks".
   // Save the final mesh
   {
      int sav_len = strlen(img_file)-4;
      char sav_file[1]; // enough to hold all numbers up to 64-bits
      char sav_format[1];
      sprintf(sav_format, "%s%d%s", "%.",sav_len,"s%s");
      sprintf(sav_file, sav_format, img_file,".mesh");

      ofstream mesh_ofs(sav_file);
      mesh_ofs.precision(8);
      pmesh->PrintAsOne(mesh_ofs);
   }

   // 22. Compute the amount of energy decrease.
   const double fin_energy = a.GetParGridFunctionEnergy(x);
   double metric_part = fin_energy;
   if (myid == 0)
   {
      cout << "Initial strain energy: " << init_energy
           << " = metrics: " << init_energy
           << " + limiting term: " << 0.0 << endl;
      cout << "  Final strain energy: " << fin_energy
           << " = metrics: " << metric_part
           << " + limiting term: " << fin_energy - metric_part << endl;
      cout << "The strain energy decreased by: " << setprecision(12)
           << (init_energy - fin_energy) * 100.0 / init_energy << " %." << endl;
   }

   // 23. Visualize the mesh displacement.
   if (visualization)
   {
      x0 -= x;
      socketstream sock;
      if (myid == 0)
      {
         sock.open("localhost", 19916);
         sock << "solution\n";
      }
      pmesh->PrintAsOne(sock);
      x0.SaveAsOne(sock);
      if (myid == 0)
      {
         sock << "window_title 'Displacements'\n"
              << "window_geometry "
              << 1200 << " " << 0 << " " << 600 << " " << 600 << "\n"
              << "keys jRmclA" << endl;
      }
   }

   // 24. Free the used memory.
   delete S;
   delete target_c2;
   delete metric2;
   delete coeff1;
   delete target_c;
   delete adapt_coeff;
   delete metric;
   delete pfespace;
   delete fec;
   delete pmesh;

   MPI_Finalize();
   return 0;
}

void GetMaterialIndicator(GridFunction &ind, GridFunction &x,
                          const ParsePGM &pgm,
                          int NC, Vector &xmin, Vector &xmax)
{
   const int ndofs = ind.Size(),
             dim   = x.FESpace()->GetFE(0)->GetDim();
   // Get material values
   for (int i = 0; i < ndofs; i++)
   {
      Vector pt(dim);
      for (int j = 0; j < dim; j++)
      {
         pt(j) = x(i + j*ndofs);
      }
      int m = material(pgm, 256/NC, pt, xmin, xmax);
      ind(i) = (double)m;
   }
   const double loc_min = ind.Min();
   double min = loc_min;
   MPI_Allreduce(&loc_min, &min, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
   ind -= min;
   const double loc_max = ind.Max();
   double max = loc_max;
   MPI_Allreduce(&loc_max, &max, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
   ind /= max;

}

void GetSize(GridFunction &size, GridFunction &ind, bool flip,
             const int size_type)
{
   const int ndofs = size.Size();
   switch (size_type)
   {
      case 0: // Size based on material color
      {
         const double small = 0.001, big = 0.01;
         for (int i = 0; i < ndofs; i++)
         {
            double val = 1-ind(i);
            if (flip) { val = ind(i);}
            size(i) = val * small + (1.0 - val) * big;
         }
         break;
      }
      default: MFEM_ABORT(" Unknown size_type ");
   }
}
