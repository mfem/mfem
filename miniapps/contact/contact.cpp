// Copyright (c) 2010-2025, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

//                       MFEM Contact Miniapp

// Compile with: make contact

// Sample runs
// Problem 0: two-block (linear elasticity)
// mpirun -np 4 ./contact -prob 0 -sr 0 -pr 0 -tr 2 -nsteps 4  -msteps 0 -amgf
// mpirun -np 4 ./contact -prob 0 -sr 0 -pr 0 -tr 2 -nsteps 4  -msteps 0 -no-amgf

// Problem 1: ironing (linear elasticity)
// mpirun -np 4 ./contact -prob 1 -sr 0 -pr 0 -tr 2 -nsteps 4  -msteps 6 -amgf
// mpirun -np 4 ./contact -prob 1 -sr 0 -pr 0 -tr 2 -nsteps 4  -msteps 6 -no-amgf

// Problem 2: beam–sphere (linear or non-linear elasticity)
// mpirun -np 4 ./contact -prob 2 -sr 0 -pr 0 -tr 2 -nsteps 6 -msteps 0 -lin -amgf
// mpirun -np 4 ./contact -prob 2 -sr 0 -pr 0 -tr 2 -nsteps 6 -msteps 0 -lin -no-amgf
// mpirun -np 4 ./contact -prob 2 -sr 0 -pr 0 -tr 2 -nsteps 6 -msteps 0 -nonlin -amgf
// mpirun -np 4 ./contact -prob 2 -sr 0 -pr 0 -tr 2 -nsteps 6 -msteps 0 -nonlin -no-amgf

// Description:
// This miniapp solves benchmark frictionless contact problems using a
// self-contained Interior Point (IP) optimizer. Contact constraints
// are supplied by Tribol. The linear systems inside the IP iterations
// are solved with PCG, preconditioned by either standard HypreBoomerAMG
// or AMG with Filtering (AMGF). AMGF enhances AMG by applying an additional
// subspace-correction step on a small subspace associated with the contact interface.

// Notes:
//  1. Non-linear elasticity is supported only for -prob 2. If -nonlin is
//     requested for -prob 0/1, the app falls back to the linear model.
//  2. The required meshes for -prob 0,1 are generated on the fly. For -prob 2,
//     the mesh is constructed from the file ./meshes/beam-sphere.mesh.
//  3. AMGF requires a parallel direct solver for the filtered subspace; build
//     MFEM with MUMPS or MKL CPardiso. If unavailable, requesting -amgf aborts
//     with an error.
//  4. This miniapp requires MFEM build with Tribol, an open source contact mechanics library
//     available at https://github.com/LLNL/Tribol.

#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include "ip.hpp"

using namespace std;
using namespace mfem;

enum problem_name
{
   twoblock,
   ironing,
   beamsphere
};

Mesh * GetProblemMesh(problem_name prob_name);

int main(int argc, char *argv[])
{
#ifdef MFEM_USE_SINGLE
   cout << "Contact miniapp is not supported in single precision.\n\n";
   return MFEM_SKIP_RETURN_VALUE;
#endif
   // 0. Initialize MPI and HYPRE.
   Mpi::Init();
   int myid = Mpi::WorldRank();
   int num_procs = Mpi::WorldSize();
   Hypre::Init();

   problem_name prob_name;

   // 1. Command-line options.
   // Number of serial uniform refinements
   int sref = 1;
   // Number of parallel uniform refinements
   int pref = 0;
   // Enable/disable GLVis visualization
   bool visualization = true;
   // Enable/disable ParaView output
   bool paraview = false;
   // Problem choice (0,1,2)
   int prob_no = 0;
   // number of times steps in the z-direction (problem 0)
   int nsteps = 1;
   // number of times steps in the x-direction (problem 1)
   int msteps = 0;
   // Use non-linear elasticity (only for problem 2)
   bool nonlinear = false;
   // Tribol search proximity ratio
   real_t tribol_ratio = 8.0;
   // Enable/disable AMGF preconditioner (default is AMG)
   bool amgf = false;

   // 1. Parse command-line options.
   OptionsParser args(argc, argv);
   args.AddOption(&prob_no, "-prob", "--problem-number",
                  "Choice of problem:"
                  "0: two-block problem"
                  "1: ironing problem"
                  "2: beam-sphere problem");
   args.AddOption(&nonlinear, "-nonlin", "--nonlinear", "-lin",
                  "--linear", "Choice between linear and non-linear Elasticiy model.");
   args.AddOption(&sref, "-sr", "--serial-refinements",
                  "Number of uniform refinements.");
   args.AddOption(&nsteps, "-nsteps", "--nsteps",
                  "Number of steps.");
   args.AddOption(&msteps, "-msteps", "--msteps",
                  "Number of extra steps.");
   args.AddOption(&pref, "-pr", "--parallel-refinements",
                  "Number of uniform refinements.");
   args.AddOption(&amgf, "-amgf", "--amgf", "-no-amgf",
                  "--no-amgf",
                  "Enable or disable AMGF with Filtering solver.");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&paraview, "-paraview", "--paraview", "-no-paraview",
                  "--no-paraview",
                  "Enable or disable ParaView visualization.");
   args.AddOption(&tribol_ratio, "-tr", "--tribol-proximity-parameter",
                  "Tribol-proximity-parameter.");

   args.Parse();
   if (!args.Good())
   {
      if (myid == 0)
      {
         args.PrintUsage(cout);
      }
      return 1;
   }
   if (myid == 0)
   {
      args.PrintOptions(cout);
   }

   // Validate selection and convert to enum.
   MFEM_VERIFY(prob_no >= 0 &&
               prob_no <= 2, "Unknown test problem number: " << prob_no);

   prob_name = (problem_name)prob_no;

   // Only the beam–sphere supports non-linear elasticity; fall back if needed.
   if (nonlinear && prob_name!=problem_name::beamsphere)
   {
      if (myid == 0)
      {
         cout << "Non-linear elasticity not supported for the two-block and ironing problems"
              << endl;
         cout << "Switching to the linear model ..." << endl;
      }
      nonlinear = false;
   }

   // Bound constraints are used only for the non-linear case
   // (activated later after a few load steps).
   bool bound_constraints = (nonlinear) ? true : false;


   // 2. Get the problem mesh.
   Mesh * mesh = GetProblemMesh(prob_name);

   // 3. Refine the mesh serially.
   for (int i = 0; i<sref; i++)
   {
      mesh->UniformRefinement();
   }

   // 4. Convert to ParMesh and refine in parallel.
   ParMesh pmesh(MPI_COMM_WORLD,*mesh);
   delete mesh;
   for (int i = 0; i<pref; i++)
   {
      pmesh.UniformRefinement();
   }

   // Material parameters per attribute (Young’s modulus and Poisson ratio).
   Vector E(pmesh.attributes.Max());
   Vector nu(pmesh.attributes.Max());

   // Essential boundary specification:
   // - ess_bdr_attr: boundary attribute ids to constrain
   // - ess_bdr_attr_comp: component to constrain (0=x,1=y,2=z, -1=all)
   Array<int> ess_bdr_attr;
   Array<int> ess_bdr_attr_comp;

   // 5. Define problem-dependent BCs and material properties.
   switch (prob_name)
   {
      // case twoblock:
      //    ess_bdr_attr.Append(1); ess_bdr_attr_comp.Append(-1);
      //    ess_bdr_attr.Append(10); ess_bdr_attr_comp.Append(-1);
      //    E[0] = 1.0;  E[1] = 1e3;
      //    nu[0] = 0.499;  nu[1] = 0.0;
      //    break;
      case twoblock:
      case ironing:
         ess_bdr_attr.Append(2); ess_bdr_attr_comp.Append(-1);
         ess_bdr_attr.Append(6); ess_bdr_attr_comp.Append(-1);
         E[0] = 1.0;  E[1] = 1e3;
         nu[0] = 0.499;  nu[1] = 0.0;
         break;
      case beamsphere:
         ess_bdr_attr.Append(1); ess_bdr_attr_comp.Append(0);
         ess_bdr_attr.Append(2); ess_bdr_attr_comp.Append(1);
         ess_bdr_attr.Append(4); ess_bdr_attr_comp.Append(2);
         ess_bdr_attr.Append(5); ess_bdr_attr_comp.Append(-1);
         E = 1.e3;
         nu = 0.4;
         break;
      default:
         MFEM_ABORT("Should be unreachable");
         break;
   }

   // 6. Build the elasticity problem object (linear or non-linear).
   // It owns the FE space and will assemble the stiffness and RHS.
   ElasticityOperator prob(&pmesh, ess_bdr_attr,ess_bdr_attr_comp, E, nu,
                           nonlinear);


   ParFiniteElementSpace * fes = prob.GetFESpace();
   HYPRE_BigInt gndofs = fes->GlobalTrueVSize();
   if (myid == 0)
   {
      mfem::out << "\n --------------------------------------" << endl;
      mfem::out << "  Global number of dofs = " << gndofs << endl;
      mfem::out << " --------------------------------------" << endl;
   }

   // 7. Set up the contact interface for Tribol.
   std::set<int> mortar_attr;
   std::set<int> nonmortar_attr;
   switch (prob_name)
   {
      case twoblock:
      case  ironing:
         mortar_attr.insert(3);
         nonmortar_attr.insert(4);
         break;
      case beamsphere:
         mortar_attr.insert(6);
         mortar_attr.insert(9);
         nonmortar_attr.insert(7);
         nonmortar_attr.insert(8);
         break;
      default:
         MFEM_ABORT("Should be unreachable");
         break;
   }



   // Displacement field (current step), and a copy for visualization.
   ParGridFunction x_gf(fes); x_gf = 0.0;
   ParMesh pmesh_copy(pmesh);
   ParFiniteElementSpace fes_copy(*fes,pmesh_copy);
   ParGridFunction xcopy_gf(&fes_copy); xcopy_gf = x_gf;

   // Optional ParaView data collection
   ParaViewDataCollection * paraview_dc = nullptr;
   if (paraview)
   {
      std::ostringstream paraview_file_name;
      paraview_file_name << "contact-problem_" << prob_no
                         << "_par_ref_" << pref
                         << "_ser_ref_" << sref
                         << "_nonlinear_" << nonlinear;
      paraview_dc = new ParaViewDataCollection(paraview_file_name.str(), &pmesh_copy);
      paraview_dc->SetPrefixPath("ParaView");
      paraview_dc->SetLevelsOfDetail(2);
      paraview_dc->SetDataFormat(VTKFormat::BINARY);
      paraview_dc->SetHighOrderOutput(true);
      paraview_dc->RegisterField("u", &xcopy_gf);
      paraview_dc->SetCycle(0);
      paraview_dc->SetTime(0.0);
      paraview_dc->Save();
   }

   // Optional GLVis connection.
   socketstream sol_sock;
   if (visualization)
   {
      char vishost[] = "localhost";
      int visport = 19916;
      sol_sock.open(vishost, visport);
      sol_sock.precision(8);
   }

   // Reference vs current coordinates (used to visualize total deformation).
   ParGridFunction ref_coords(prob.GetFESpace());
   ParGridFunction new_coords(prob.GetFESpace());
   pmesh.GetNodes(new_coords);
   pmesh.GetNodes(ref_coords);

   // Load magnitude for problem 2 (beam-sphere).
   real_t p = 40.0;
   ConstantCoefficient f(p);

   // 8. Construct the contact optimization problem wrapper. It sets up the
   // contact system using Tribol and provides operators for objective evaluation,
   // gradients, Hessians, and constraints. It also provides the filtered subspace
   // transfer operator used by AMGF
   OptContactProblem contact(&prob, mortar_attr, nonmortar_attr,
                             tribol_ratio, bound_constraints);


   int dim = pmesh.Dimension();
   Vector ess_values(dim); ess_values = 0.0;
   Array<int> ess_bdr;
   if (pmesh.bdr_attributes.Size())
   {
      ess_bdr.SetSize(pmesh.bdr_attributes.Max());
   }

   // 9. Time/load stepping loop.
   int total_steps = nsteps + msteps;
   for (int i = 0; i<total_steps; i++)
   {
      if (Mpi::Root())
      {
         std::ostringstream oss;
         oss << "\n Solving optimization problem for time step: " << i;
         mfem::out << oss.str() << endl;
         mfem::out << " " <<std::string(oss.str().size()-2, '-') << endl;
      }
      ess_values = 0.0;
      // 9(a). Apply problem-specific boundary conditions / loads for this step.
      switch (prob_name)
      {
         case twoblock:
         case ironing:
            if (ess_bdr.Size())
            {
               ess_bdr = 0; ess_bdr[5] = 1;
            }
            if (i < nsteps)
            {
               ess_values[2] = -1.0/1.4*(i+1)/nsteps;
            }
            else
            {
               ess_values[0] = 3.0/1.4*(i+1-nsteps)/msteps;
               ess_values[2] = -1.0/1.4;
            }
            prob.SetDisplacementDirichletData(ess_values, ess_bdr);
            break;
         case beamsphere:
            if (ess_bdr.Size())
            {
               ess_bdr = 0; ess_bdr[2] = 1;
            }
            f.constant = -p*(i+1)/nsteps;
            prob.SetNeumanPressureData(f,ess_bdr);
            break;
         default:
            MFEM_ABORT("Should be unreachable");
            break;
      }

      // 9(b). Assemble the elasticity system with current BCs.
      prob.FormLinearSystem();

      // Store the current (reference) displacement in true dofs.
      Vector xref;
      x_gf.GetTrueDofs(xref);

      // 9(c). Form the contact system. This also builds the filtered subspace
      // transfer used by AMGF.
      contact.FormContactSystem(&new_coords, xref);

      // 9(d). Activate bound constraints after a few steps in the non-linear case.
      int activation_step = 2;
      if (bound_constraints && i>activation_step)
      {
         contact.ActivateBoundConstraints();
      }

      // 9(e). Choose preconditioner: AMGF (with direct solver for subspace) or AMG.
      Solver * prec = nullptr;
      Solver * subspacesolver = nullptr;
      if (amgf)
      {
#ifdef MFEM_USE_MUMPS
         subspacesolver = new MUMPSSolver(MPI_COMM_WORLD);
         dynamic_cast<MUMPSSolver*>(subspacesolver)->SetPrintLevel(0);
#else
#ifdef MFEM_USE_MKL_CPARDISO
         subspacesolver = new CPardisoSolver(MPI_COMM_WORLD);
#else
         MFEM_ABORT("MFEM must be built with MUMPS or MKL_CPARDISO in order to use AMGF");
#endif
#endif
         prec = new AMGFSolver();
         auto * amgfprec = dynamic_cast<AMGFSolver *>(prec);
         amgfprec->AMG().SetSystemsOptions(3);
         amgfprec->AMG().SetPrintLevel(0);
         amgfprec->AMG().SetRelaxType(88);
         amgfprec->SetFilteredSubspaceSolver(*subspacesolver);
         amgfprec->SetFilteredSubspaceTransferOperator(
            *contact.GetContactSubspaceTransferOperator());
      }
      else
      {
         prec = new HypreBoomerAMG();
         auto * amgprec = dynamic_cast<HypreBoomerAMG *>(prec);
         amgprec->SetSystemsOptions(3);
         amgprec->SetPrintLevel(0);
         amgprec->SetRelaxType(88);
      }

      // 9(f). Linear solver used inside the IP optimizer.
      CGSolver cgsolver(MPI_COMM_WORLD);
      cgsolver.SetPrintLevel(0);
      cgsolver.SetRelTol(1e-10);
      cgsolver.SetMaxIter(10000);
      cgsolver.SetPreconditioner(*prec);

      // 9(g). Interior-Point optimizer driving contact resolution.
      IPSolver optimizer(&contact);
      optimizer.SetTol(1e-6);
      optimizer.SetMaxIter(100);
      optimizer.SetLinearSolver(&cgsolver);
      optimizer.SetPrintLevel(0);

      // Initial guess = previous reference configuration.
      x_gf.SetTrueVector();
      int ndofs = prob.GetFESpace()->GetTrueVSize();
      Vector x0(ndofs); x0 = 0.0;
      x0.Set(1.0, xref);

      // Solve for the next step displacement (xf).
      Vector xf(ndofs); xf = 0.0;
      optimizer.Mult(x0, xf);

      delete prec;
      if (subspacesolver) { delete subspacesolver; }

      // Update internal state for next step (for bound constraints).
      Vector dx(xf); dx -= x0;
      if (bound_constraints)
      {
         contact.SetDisplacement(dx, (i>=activation_step));
      }

      // 9(h). Report objective values, dofs, IP and CG iteration counts.
      int eval_err;
      real_t Einitial = contact.E(x0, eval_err);
      real_t Efinal = contact.E(xf, eval_err);
      Array<int> & PCGiterations = optimizer.GetLinearSolverIterations();

      if (Mpi::Root())
      {
         mfem::out << " Initial Energy objective        = " << Einitial << endl;
         mfem::out << " Final Energy objective          = " << Efinal << endl;
         mfem::out << " Optimizer number of iterations  = " <<
                   optimizer.GetNumIterations() << endl;
         mfem::out << " PCG number of iterations        = " ;
         for (int i = 0; i < PCGiterations.Size(); ++i)
         {
            std::cout << PCGiterations[i] << " ";
         }
         mfem::out << "\n";
      }

      // 9(i). Visualization outputs.
      x_gf.SetFromTrueDofs(xf);
      add(ref_coords,x_gf,new_coords);
      pmesh_copy.SetNodes(new_coords);
      xcopy_gf = x_gf;
      xcopy_gf.SetTrueVector();
      if (paraview)
      {
         paraview_dc->SetCycle(i+1);
         paraview_dc->SetTime(real_t(i+1));
         paraview_dc->Save();
      }

      if (visualization)
      {
         sol_sock << "parallel " << num_procs << " " << myid << "\n"
                  << "solution\n" << pmesh_copy << x_gf << flush;

         if (i == total_steps - 1)
         {
            pmesh.MoveNodes(x_gf);
            char vishost[] = "localhost";
            int  visport   = 19916;
            socketstream sol_sock_final(vishost, visport);
            sol_sock_final << "parallel " << num_procs << " " << myid << "\n";
            sol_sock_final.precision(8);
            sol_sock_final << "solution\n" << pmesh << x_gf << flush;
         }
      }

      // 9(j). Update RHS/load for the next step (if any).
      if (i == total_steps-1) { break; }
      prob.UpdateRHS();
   }

   // 10. Cleanup.
   if (paraview_dc) { delete paraview_dc; }
   return 0;
}


Mesh * GetProblemMesh(problem_name prob_name)
{

   if (prob_name == problem_name::beamsphere)
   {
      return new Mesh("meshes/beam-sphere.mesh",1);
   }
   else // construct the two-block or ironing mesh
   {
      Mesh * combined_mesh = nullptr;
      constexpr real_t scale = 9.0/3.6;
      constexpr real_t lx0 = 3.6*scale, ly0 = 1.6*scale, lz0 = 1.0*scale;
      constexpr int nx0 = 24, ny0 = 12, nz0 = 6;
      Mesh mesh0 = Mesh::MakeCartesian3D(nx0, ny0, nz0, Element::HEXAHEDRON,
                                         lx0, ly0, lz0);

      // adjust boundary attributes
      for (int i = 0; i<mesh0.GetNBE(); i++)
      {
         int battr = mesh0.GetBdrElement(i)->GetAttribute();
         int new_battr;
         switch (battr)
         {
            case 1: new_battr = 2; break;
            case 6: new_battr = 3; break;
            default: new_battr = 1; break;
         }
         mesh0.SetBdrAttribute(i, new_battr);
      }
      mesh0.SetAttributes();
      if (prob_name == problem_name::twoblock)
      {
         constexpr real_t lx = 1.0, ly = 1.0, lz = 1.0;
         constexpr int nx = 5, ny = 5, nz = 5;
         Mesh mesh = Mesh::MakeCartesian3D(nx, ny, nz, Element::HEXAHEDRON, lx, ly, lz);

         // adjust element attributes
         for (int i = 0; i<mesh.GetNE(); i++)  { mesh.GetElement(i)->SetAttribute(2); }
         mesh.SetAttributes();

         // adjust boundary attributes
         for (int i = 0; i<mesh.GetNBE(); i++)
         {
            int battr = mesh.GetBdrElement(i)->GetAttribute();
            int new_battr;
            switch (battr)
            {
               case 1: new_battr = 4; break;
               case 6: new_battr = 6; break;
               default: new_battr = 5; break;
            }
            mesh.SetBdrAttribute(i, new_battr);
         }
         mesh.SetAttributes();

         Vector shift(3);
         shift(0) = (ly0-ly)/3;
         shift(1) = (ly0-ly)/2;
         shift(2) = lz0;

         auto shift_map = [&](const Vector &x, Vector &y)
         {
            y.SetSize(3);
            y = x;
            y+=shift;
         };

         mesh.Transform(shift_map);

         Mesh *mesh_array2[] = {&mesh, &mesh0};
         combined_mesh = new Mesh(mesh_array2, 2);


      }
      else if (prob_name == problem_name::ironing)
      {
         constexpr real_t lx = 1.6*scale, ly = 2.0*scale, lz = 0.2*scale;
         constexpr int nx = 16, ny = 20, nz = 2;
         Mesh mesh = Mesh::MakeCartesian3D(nx, ny, nz, Element::HEXAHEDRON, lx, ly, lz);

         // adjust element attributes
         for (int i = 0; i<mesh.GetNE(); i++) { mesh.GetElement(i)->SetAttribute(2); }
         mesh.SetAttributes();

         // adjust boundary attributes
         for (int i = 0; i<mesh.GetNBE(); i++)
         {
            int battr = mesh.GetBdrElement(i)->GetAttribute();
            int new_battr;
            switch (battr)
            {
               case 1: new_battr = 6; break;
               case 6: new_battr = 4; break;
               default: new_battr = 5; break;
            }
            mesh.SetBdrAttribute(i, new_battr);
         }
         mesh.SetAttributes();

         constexpr real_t theta_arc = M_PI/2;
         constexpr real_t rin = lx / theta_arc - lz;
         constexpr real_t rout = lx / theta_arc + lz;

         constexpr real_t xshift = lx0/4;
         constexpr real_t yshift = 0.5*(ly0-ly);
         constexpr real_t zshift = rout + lz0;

         auto bend_map = [&](const Vector &x, Vector &y)
         {
            const real_t theta = -((x(0) / lx + 0.5) * theta_arc);
            const real_t r = rin + (x(2) + lz);
            y.SetSize(3);
            y(0) = xshift + r * std::cos(theta);
            y(1) = yshift + x(1);
            y(2) = zshift + r * std::sin(theta);
         };

         mesh.Transform(bend_map);
         Mesh *mesh_array2[] = {&mesh, &mesh0};
         combined_mesh = new Mesh(mesh_array2, 2);
      }

      return combined_mesh;
   }
}
