// Copyright (c) 2010-2023, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.


#include "mfem.hpp"

#include "axom/slic.hpp"

#include "tribol/interface/tribol.hpp"
#include "tribol/interface/mfem_tribol.hpp"

int main(int argc, char *argv[])
{
   // Initialize MPI
   mfem::Mpi::Init();
   int num_ranks = mfem::Mpi::WorldSize();
   int my_rank = mfem::Mpi::WorldRank();

   // Initialize logging with axom::slic
   axom::slic::SimpleLogger logger;
   axom::slic::setIsRoot(mfem::Mpi::Root());
   
   // Define command line options
   int ref_levels = 2;   // number of times to uniformly refine the serial mesh before constructing the parallel mesh
   double lambda = 50.0; // Lame parameter lambda
   double mu = 50.0;     // Lame parameter mu (shear modulus)

   // Parse command line options
   mfem::OptionsParser args(argc, argv);
   args.AddOption(&ref_levels, "-r", "--refine",
                  "Number of times to refine the mesh uniformly.");
   args.AddOption(&lambda, "-l", "--lambda",
                  "Lame parameter lambda.");
   args.AddOption(&mu, "-m", "--mu",
                  "Lame parameter mu (shear modulus).");
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
   std::string mesh_file = "two-hex.mesh"; // two block mesh; bottom block = [0,1]^3 and
                                           // top block = [0,1]x[0,1]x[0.99,1.99]
   int order = 1; // FE polynomial degree (NOTE: only 1 works for now)
   std::set<int> mortar_attrs({4}); // z=1 plane of bottom block
   std::set<int> nonmortar_attrs({5}); // z=0.99 plane of top block
   std::vector<std::set<int>> fixed_attrs(3); // per-dimension sets of boundary attributes with
                                              // homogeneous Dirichlet BCs
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
   serial_mesh.Clear();

   // Create an H1 finite element space on the mesh for displacements/forces
   mfem::H1_FECollection fec(order, 3);
   mfem::ParFiniteElementSpace fespace(&mesh, &fec);
   if (mfem::Mpi::Root())
   {
      std::cout << "Number of displacement unknowns: " << fespace.GlobalTrueVSize() << std::endl;
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
      for (int i = 0; i < 3; ++i)
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

   // Create small deformation linear elastic bilinear form
   mfem::ParBilinearForm a(&fespace);
   mfem::ConstantCoefficient lambda_coeff(lambda);
   mfem::ConstantCoefficient mu_coeff(mu);
   a.AddDomainIntegrator(new mfem::ElasticityIntegrator(lambda_coeff, mu_coeff));

   // Compute elasticity contribution to tangent stiffness matrix
   a.Assemble();
   std::unique_ptr<mfem::HypreParMatrix> A;
   a.FormSystemMatrix(ess_tdof_list, *A);

   // Initialize Tribol contact library
   tribol::initialize(3, MPI_COMM_WORLD);

   // Create a Tribol coupling scheme: defines contact surfaces and enforcement
   int coupling_scheme_id = 0;
   int mesh1_id = 0;
   int mesh2_id = 1;
   tribol::registerMfemCouplingScheme(
      coupling_scheme_id, mesh1_id, mesh2_id,
      mesh, coords, mortar_attrs, nonmortar_attrs,
      tribol::SURFACE_TO_SURFACE,
      tribol::NO_SLIDING,
      tribol::SINGLE_MORTAR,
      tribol::FRICTIONLESS,
      tribol::LAGRANGE_MULTIPLIER,
      tribol::BINNING_GRID
   );

   // Set Tribol options for Lagrange multiplier enforcement
   tribol::setLagrangeMultiplierOptions(
      coupling_scheme_id,
      tribol::ImplicitEvalMode::MORTAR_RESIDUAL_JACOBIAN
   );

   // Update contact mesh decomposition
   tribol::updateMfemParallelDecomposition();

   // Update contact gaps, forces, and tangent stiffness
   int cycle = 1;   // pseudo cycle
   double t = 1.0;  // pseudo time
   double dt = 1.0; // pseudo dt
   tribol::update(cycle, t, dt);

   // Return contact contribution to the tangent stiffness matrix
   auto A_blk = tribol::getMfemBlockJacobian(coupling_scheme_id);
   A_blk->SetBlock(0, 0, A.release());  // add elasticity contribution to the block operator

   // Convert block tangent stiffness to a single HypreParMatrix
   mfem::Array2D<mfem::HypreParMatrix*> hypre_blocks(2, 2);
   for (int i{0}; i < 2; ++i)
   {
      for (int j{0}; j < 2; ++j)
      {
         if (A_blk->GetBlock(i, j).Height() != 0 && A_blk->GetBlock(i, j).Width() != 0)
         {
         hypre_blocks(i, j) = dynamic_cast<mfem::HypreParMatrix*>(&A_blk->GetBlock(i, j));
         }
         else
         {
         hypre_blocks(i, j) = nullptr;
         }
      }
   }
   auto A_merged = std::unique_ptr<mfem::HypreParMatrix>(
      mfem::HypreParMatrixFromBlocks(hypre_blocks)
   );

   // Create block RHS vector holding forces and gaps at tdofs
   mfem::BlockVector B_blk(A_blk->RowOffsets());
   B_blk = 0.0;

   // Fill with initial nodal gaps.
   // Note forces from contact are currently zero since pressure is currently zero.
   mfem::Vector g;
   tribol::getMfemGap(coupling_scheme_id, g); // gap on ldofs
   {
      auto& G = B_blk.GetBlock(1); // gap tdof vector
      auto& P_submesh = *tribol::getMfemPressure(coupling_scheme_id).ParFESpace()->GetProlongationMatrix();
      P_submesh.MultTranspose(g, G); // gap is a dual vector, so
                                     // gap tdof vector = P^T * (gap ldof vector)
   }

   // Create block solution vector holding displacements and pressures at tdofs
   mfem::BlockVector X_blk(A_blk->ColOffsets());
   X_blk = 0.0;

   // Solve for displacements and pressures
   mfem::CGSolver solver(MPI_COMM_WORLD);
   solver.SetRelTol(1.0e-12);
   solver.SetMaxIter(2000);
   solver.SetPrintLevel(1);
   solver.SetOperator(*A_merged);
   solver.Mult(B_blk, X_blk);

   return 0;
}