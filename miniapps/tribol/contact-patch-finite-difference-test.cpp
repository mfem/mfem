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
//                -----------------------------------------
//                Tribol Miniapp: Mortar contact patch test
//                -----------------------------------------
//
//
// Command line options:
//  - -r, --refine: number of uniform refinements of the mesh (default: 2)
//

#include "mfem.hpp"

#include "axom/slic.hpp"

#include "tribol/interface/tribol.hpp"
#include "tribol/interface/mfem_tribol.hpp"

// Define MPI_REAL_T
#if defined(MFEM_USE_DOUBLE)
#define MPI_REAL_T MPI_DOUBLE
#else
#error "Tribol requires MFEM built with double precision!"
#endif


using namespace mfem;

class ContactObj
{
protected:
   HypreParMatrix * Jacobian = nullptr;
   mfem::Vector gap;
   std::unique_ptr<mfem::BlockOperator> A_blk;
   ParMesh * mesh = nullptr;
   ParGridFunction * coords = nullptr;
   std::set<int> mortar_attrs;
   std::set<int> nonmortar_attrs;
public:
   ContactObj(ParMesh * mesh_, 
	      const std::set<int> & mortar_attrs_,
	      const std::set<int> & nonmortar_attrs_,
	      ParGridFunction * coords_);
   void GetGap(mfem::Vector & g) const;
   mfem::HypreParMatrix * GetJacobian() const;
   virtual ~ContactObj();
};


int main(int argc, char *argv[])
{
   // Initialize MPI
   mfem::Mpi::Init();

   // Initialize logging with axom::slic
   axom::slic::SimpleLogger logger;
   axom::slic::setIsRoot(mfem::Mpi::Root());

   // Define command line options
   int ref_levels = 2;   // number of times to uniformly refine the serial mesh
   double x0shift = 0.0;
   // Parse command line options
   mfem::OptionsParser args(argc, argv);
   args.AddOption(&ref_levels, "-r", "--refine",
                  "Number of times to refine the mesh uniformly.");
   args.AddOption(&x0shift, "-x0shift", "--x0shift", "magnitude (inf norm) of random displacement where finite difference test is evaluated"); 
   args.Parse();
   if (!args.Good())
   {
      if (mfem::Mpi::Root())
      {
         args.PrintUsage(std::cout);
      }
      return EXIT_FAILURE;
   }
   if (mfem::Mpi::Root())
   {
      args.PrintOptions(std::cout);
   }

   // Fixed options
   // two block mesh; bottom block = [0,1]^3 and top block = [0,1]x[0,1]x[0.99,1.99]
   std::string mesh_file = "modified-two-hex.mesh";
   //std::string mesh_file = "Test4.mesh"; 
   // Problem dimension (NOTE: Tribol's mortar only works in 3D)
   constexpr int dim = 3;
   // FE polynomial degree (NOTE: only 1 works for now)
   constexpr int order = 1;
   // z=1 plane of bottom block (contact plane)
   //std::set<int> mortar_attrs({3});
   std::set<int> mortar_attrs({4});
   // z=0.99 plane of top block (contact plane)
   //std::set<int> nonmortar_attrs({4});
   std::set<int> nonmortar_attrs({5});
   // per-dimension sets of boundary attributes with homogeneous Dirichlet BCs.
   // allows transverse deformation of the blocks while precluding rigid body
   // rotations/translations.
   std::vector<std::set<int>> fixed_attrs(dim);
   fixed_attrs[0] = {1}; // x=0 plane of both blocks
   fixed_attrs[1] = {2}; // y=0 plane of both blocks
   fixed_attrs[2] = {3, 6}; // 3: z=0 plane of bottom block; 6: z=1.99 plane of top block

   // Read the mesh, refine, and create a mfem::ParMesh
   mfem::Mesh serial_mesh(mesh_file);
   for (int i = 0; i < ref_levels; ++i)
   {
      serial_mesh.UniformRefinement();
   }
   mfem::ParMesh mesh(MPI_COMM_WORLD, serial_mesh);
   mfem::ParMesh mesh_copy(mesh);
   
   serial_mesh.Clear();

   MFEM_ASSERT(dim == mesh.Dimension(),
               "This miniapp must be run with the supplied two-hex.mesh file.");

   // Create an H1 finite element space on the mesh for displacements/forces
   mfem::H1_FECollection fec(order, dim);
   mfem::ParFiniteElementSpace fespace(&mesh, &fec, dim);
   auto n_displacement_dofs = fespace.GlobalTrueVSize();
   if (mfem::Mpi::Root())
   {
      std::cout << "Number of displacement unknowns: " << n_displacement_dofs <<
                std::endl;
   }

   // Create coordinate and displacement grid functions
   mfem::ParGridFunction coords(&fespace);
   mesh.SetNodalGridFunction(&coords);
   mfem::ParGridFunction displacement(&fespace);
   displacement = 0.0;

   // Find true dofs with homogeneous Dirichlet BCs
   mfem::Array<int> ess_tdof_list;
   {
      mfem::Array<int> ess_vdof_marker(fespace.GetVSize());
      ess_vdof_marker = 0;
      for (int i = 0; i < dim; ++i)
      {
         mfem::Array<int> ess_bdr(mesh.bdr_attributes.Max());
         ess_bdr = 0;
         for (auto xfixed_attr : fixed_attrs[i])
         {
            ess_bdr[xfixed_attr-1] = 1;
         }
         mfem::Array<int> new_ess_vdof_marker;
         fespace.GetEssentialVDofs(ess_bdr, new_ess_vdof_marker, i);
         for (int j = 0; j < new_ess_vdof_marker.Size(); ++j)
         {
            ess_vdof_marker[j] = ess_vdof_marker[j] || new_ess_vdof_marker[j];
         }
      }
      mfem::Array<int> ess_tdof_marker;
      fespace.GetRestrictionMatrix()->BooleanMult(ess_vdof_marker, ess_tdof_marker);
      mfem::FiniteElementSpace::MarkerToList(ess_tdof_marker, ess_tdof_list);
   }

   // #1: Initialize Tribol contact library
   tribol::initialize(dim, MPI_COMM_WORLD);

   
   /* Begin Tucker addition
    * finite difference check of the gap function Jacobian at u = u0
    * we evaluate the norm of the finite difference residual
    * err(eps) = || (g(u0 + eps * udir) - g(u0)) / eps - J(u0) * udir ||_2
    * which in the absence of finite-precision
    * err(eps) = O(eps) when the gap is not linear
    * err(eps) = 0, when the gap is linear
    */
   int dimU = fespace.GetTrueVSize(); 
   Vector u0(dimU); u0 = 0.0; 
   Vector u1(dimU); u1 = 0.0;
   Vector udir(dimU); udir = 0.0; udir.Randomize(); udir *= 1.e-2;

   Array<int> vdofs;
   for (int i = 0; i < mesh.GetNBE(); i++)
   {
      const int attr = (mesh.GetBdrElement(i))->GetAttribute();
      if (attr == 4)
      {
         fespace.GetBdrElementVDofs(i, vdofs);
         for (int j = 0; j < vdofs.Size(); j++)
         {
             if (j / 4 == 2)
             {      
                u0(vdofs[j]) = -1.0 * x0shift;
             }
         }
      }
   }
   
   
   ParGridFunction new_coords(&fespace);
   mesh.GetNodes(new_coords);
   
   // evaluate the gap and gap Jacobian at u = u0   
   u1.Set(1.0, u0);
   displacement.SetFromTrueDofs(u1);
   add(coords, displacement, new_coords);

   ContactObj contact0(&mesh, mortar_attrs, nonmortar_attrs, &new_coords);
   HypreParMatrix * J0 = contact0.GetJacobian();
   int dimG = J0->Height();
   
   Vector g0(dimG); g0 = 0.0; contact0.GetGap(g0);
   Vector g1(dimG); g1 = 0.0;

   // finite difference residual
   Vector fdres(dimG); fdres = 0.0;

   // J0udir = J(u0) * udir
   Vector J0udir(dimG); J0->Mult(udir, J0udir);

   // output various configurations
   // to visualize u = u0, u = u0 + eps * udir
   // use linear adjustment for eps here  
   std::ostringstream paraview_file_name;
   paraview_file_name << "BlockConfigurations_ref_" << ref_levels << "shift" << x0shift;
   ParaViewDataCollection * paraview_dc = new ParaViewDataCollection(paraview_file_name.str(), &mesh_copy);
   paraview_dc->SetPrefixPath("ParaView");
   paraview_dc->SetLevelsOfDetail(1);
   paraview_dc->SetDataFormat(VTKFormat::BINARY);
   paraview_dc->SetHighOrderOutput(true);
   paraview_dc->SetCycle(0);
   paraview_dc->SetTime(double(0));
   paraview_dc->Save();
   
   std::ofstream fdepsStream;
   std::ostringstream fdeps_file_name;
   fdeps_file_name << "data/fdeps.dat";

   std::ofstream fderrStream;
   std::ostringstream fderr_file_name;
   fderr_file_name << "data/fderr.dat";
   
   // write new configuration (reference coordinates + displacement u0) to file 
   u1.Set(1.0, u0);
   displacement.SetFromTrueDofs(u1);
   add(coords, displacement, new_coords);
   
   Vector config(u0.Size()); config = 0.0;
   new_coords.GetTrueDofs(config);
   

   std::ofstream configStream;
   std::ostringstream config_file_name;
   config_file_name << "data/configuration.dat";

   std::ofstream gap0Stream;
   std::ostringstream gap0_file_name;
   gap0_file_name << "data/gap.dat";
   if (mfem::Mpi::Root())
   {
      fdepsStream.open(fdeps_file_name.str(), std::ios::out | std::ios::trunc);
      fderrStream.open(fderr_file_name.str(), std::ios::out | std::ios::trunc);
      configStream.open(config_file_name.str(), std::ios::out | std::ios::trunc);
      for (int i = 0; i < config.Size(); i++)
      {
         configStream << config(i) << std::endl;
      }
      configStream.close();
      gap0Stream.open(gap0_file_name.str(), std::ios::out | std::ios::trunc);
      for (int i = 0; i < g0.Size(); i++)
      {
         gap0Stream << g0(i) << std::endl;
      }
      gap0Stream.close();
   }

   double eps = 1.0;
   for (int i = 0; i < 30; i++)
   {
      u1.Set(1.0, u0);
      u1.Add(eps, udir);
      displacement.SetFromTrueDofs(u1);
      add(coords, displacement, new_coords);
      ContactObj contact1(&mesh, mortar_attrs, nonmortar_attrs, &new_coords);
      contact1.GetGap(g1);
      fdres.Set(1. / eps, g1);
      fdres.Add(-1. / eps, g0);
      std::cout << "------------ -------------------------------\n\n";
      double fd_J0udirTJ0udir = InnerProduct(MPI_COMM_WORLD, fdres, J0udir);
      double fd_J0udir_l2norm = GlobalLpNorm(2, fdres.Norml2(), MPI_COMM_WORLD);
      double J0udir_l2norm = GlobalLpNorm(2, J0udir.Norml2(), MPI_COMM_WORLD);
      fdres.Add(-1, J0udir);
      double fderr_l2norm = GlobalLpNorm(2, fdres.Norml2(), MPI_COMM_WORLD);
      double udir_l2norm = GlobalLpNorm(2, udir.Norml2(), MPI_COMM_WORLD);

      if (mfem::Mpi::Root())
      {
         //std::cout << "angle between J0duir and fd approx = " << acos(fd_J0udirTJ0udir / (fd_J0udir_l2norm * J0udir_l2norm)) * 180.0 / 3.14159265359 << " (degrees)" << std::endl;
         //std::cout << "|| (g(u0 + eps * udir) - g(u0)) / eps ||_2 / || J(u0) * udir||_2 = " << fd_J0udir_l2norm / J0udir_l2norm << std::endl;
         std::cout << "||(g(u0 + eps * udir) - g(u0)) / eps - J(u0) * udir|| = " << fderr_l2norm << ", eps = " << eps << "\n\n";
	 //std::cout << "||(g(u0 + eps * udir) - g(u0)) / eps - J(u0) * udir||_2 / ||udir||_2 = " << fderr_l2norm / udir_l2norm << std::endl;
	 fdepsStream << eps << std::endl;
	 fderrStream << fderr_l2norm << std::endl;
      }
      eps /= 2.0;
   }
   if (mfem::Mpi::Root())
   {
      fdepsStream.close();
      fderrStream.close();
   }

   eps  = 1.0;
   int neps = (int) 1.e2;
   double deps =  eps / ((double) neps);
   std::ofstream epsStream;
   std::ostringstream eps_file_name;
   eps_file_name << "data/eps_ref_" << ref_levels << ".dat";
   
   std::ofstream gapStream;
   std::ostringstream gap_file_name;
   gap_file_name << "data/gap_ref_" << ref_levels << ".dat";
   
   if (mfem::Mpi::Root())
   {
      epsStream.open(eps_file_name.str(), std::ios::out | std::ios::trunc);
      gapStream.open(gap_file_name.str(), std::ios::out | std::ios::trunc);
   }




   for (int i = 0; i < neps; i++)
   {
      u1.Set(1.0, u0);
      u1.Add(eps, udir);
      displacement.SetFromTrueDofs(u1);
      add(coords, displacement, new_coords);
      mesh_copy.SetNodes(new_coords);
      paraview_dc->SetCycle(i+1) ;
      paraview_dc->SetTime((double) (i+1));
      paraview_dc->Save();
      ContactObj contact1(&mesh, mortar_attrs, nonmortar_attrs, &new_coords);
      contact1.GetGap(g1);
      epsStream << eps << std::endl;
      gapStream << g1.Norml2() << std::endl;
      eps -= deps;
   }

   if (mfem::Mpi::Root())
   {
      epsStream.close();
      gapStream.close();
   }

   // #7: Tribol cleanup: deletes coupling schemes and clears associated memory
   tribol::finalize();

   return 0;
}

ContactObj::ContactObj(ParMesh * mesh_, const std::set<int> & mortar_attrs_,
	      const std::set<int> & nonmortar_attrs_,
	      ParGridFunction * coords_) : 
	mesh(mesh_), mortar_attrs(mortar_attrs_), 
	nonmortar_attrs(nonmortar_attrs_),
	coords(coords_)
{
   // #2: Create a Tribol coupling scheme: defines contact surfaces and enforcement
   int coupling_scheme_id = 0;
   // NOTE: While there is a single mfem ParMesh for this problem, Tribol
   // defines a mortar and a nonmortar contact mesh, each with a unique mesh ID.
   // The Tribol mesh IDs for each contact surface are defined here.
   int mesh1_id = 0;
   int mesh2_id = 1;
   tribol::registerMfemCouplingScheme(
      coupling_scheme_id, mesh1_id, mesh2_id,
      *mesh, *coords, mortar_attrs, nonmortar_attrs,
      tribol::SURFACE_TO_SURFACE,
      tribol::NO_CASE,
      tribol::SINGLE_MORTAR,
      tribol::FRICTIONLESS,
      tribol::LAGRANGE_MULTIPLIER,
      tribol::BINNING_GRID
   );
   
   // #3: Set additional options/access pressure grid function on contact surfaces
   // Access Tribol's pressure grid function (on the contact surface). The
   // pressure ParGridFunction is created upon calling
   // registerMfemCouplingScheme(). It's lifetime coincides with the lifetime of
   // the coupling scheme, so the host code can reference and update it as
   // needed.
   auto& pressure = tribol::getMfemPressure(coupling_scheme_id);

   // Set Tribol options for Lagrange multiplier enforcement
   tribol::setLagrangeMultiplierOptions(
      coupling_scheme_id,
      tribol::ImplicitEvalMode::MORTAR_RESIDUAL_JACOBIAN
   );

   // #4: Update contact mesh decomposition so the on-rank Tribol meshes
   // coincide with the current configuration of the mesh. This must be called
   // before tribol::update().
   tribol::updateMfemParallelDecomposition();

   // #5: Update contact gaps, forces, and tangent stiffness contributions
   int cycle = 1;   // pseudo cycle
   mfem::real_t t = 1.0;  // pseudo time
   mfem::real_t dt = 1.0; // pseudo dt
   tribol::update(cycle, t, dt);

   
   // #6a: Return contact contribution to the tangent stiffness matrix as a
   // block operator. See documentation for getMfemBlockJacobian() for block
   // definitions.
   //auto A_blk = tribol::getMfemBlockJacobian(coupling_scheme_id);
   A_blk = tribol::getMfemBlockJacobian(coupling_scheme_id);
   Jacobian = (HypreParMatrix *)(& A_blk->GetBlock(1, 0));

   
    mfem::BlockVector B_blk(A_blk->RowOffsets());
    B_blk = 0.0;

   // Fill with initial nodal gaps.
   // Note forces from contact are currently zero since pressure is zero prior
   // to first solve.
   mfem::Vector gap_temp;
   // #6b: Return computed gap constraints on the contact surfaces
   tribol::getMfemGap(coupling_scheme_id, gap_temp); // gap on ldofs
   auto& P_submesh = *pressure.ParFESpace()->GetProlongationMatrix();
   //auto& gap_true = B_blk.GetBlock(1); // gap tdof vectorParFESpace()
   // gap is a dual vector, so (gap tdof vector) = P^T * (gap ldof vector)
   gap.SetSize(P_submesh.Width()); gap = 0.0;
   
   P_submesh.MultTranspose(gap_temp, gap);
}

void ContactObj::GetGap(mfem::Vector & g) const
{
   g.SetSize(gap.Size());
   g.Set(1.0, gap);
}


mfem::HypreParMatrix * ContactObj::GetJacobian() const
{
   return Jacobian;
}

ContactObj::~ContactObj()
{
}
