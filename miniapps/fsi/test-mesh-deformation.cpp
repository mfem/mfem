// =====================================================================================
//          2D Mesh Deformation Example (Turek CSM + Mesh Deformation)
// -------------------------------------------------------------------------------------
// Sample runs:
//
// Each run checks, at every step, that the deformed fluid mesh stays valid (positive Jacobian) and
// conforming with the deformed solid (fluid area + deformed solid area conserved), and reports
// PASSED/FAILED at the end.
//
// 0. Default: linear elasticity with distance-based local stiffening at the beam tip
//    (-mdt 2 -es), the configuration used by the FSI drivers. Stays valid through the peak
//    beam deflection.
// mpirun -np 4 ./test-mesh-deformation -tf 0.6 -eo 2 -rs 1 -dt 1e-3 -of ./Output/MeshDeformation
//
// NOTE: the Laplacian variants (1-4 below), and linear elasticity without stiffening (-no-es), invert
// elements near the beam tip before the peak deflection (with -rs 1 -eo 2 -dt 1e-3: first inverted
// step 121-347 depending on the variant), so these runs FAIL the mesh validity check.
//
// 1. Total deformation (non-incremental MeshLaplacian, -mdt = 0), deform with solid diplacement
// mpirun -np 4 ./test-mesh-deformation -tf 1.0 -dwd -eo 2 -rs 1 -dt 1e-3 -of /home/shared/Output/MeshDeformation-Laplacian-displacement
//
// 2. Total deformation (non-incremental MeshLaplacian, -mdt = 0), deform with solid velocity
// mpirun -np 4 ./test-mesh-deformation -tf 1.0 -dwv -eo 2 -rs 1 -dt 1e-3 -of /home/shared/Output/MeshDeformation-Laplacian-velocity
//
// 3. Relative deformation (incremental MeshLaplacian, -mdt = 1), deform with solid diplacement
// mpirun -np 4 ./test-mesh-deformation -tf 1.0 -mdt 1 -dwd -eo 2 -rs 1 -dt 1e-3 -of /home/shared/Output/MeshDeformation-IncrementalLaplacian-displacement
//
// 4. Total deformation (incremental MeshLaplacian, -mdt = 1), deform with solid velocity
// mpirun -np 4 ./test-mesh-deformation -tf 1.0 -mdt 1 -dwv -eo 2 -rs 1 -dt 1e-3 -of /home/shared/Output/MeshDeformation-IncrementalLaplacian-velocity
//
// 5. Total deformation (MeshLinearElasticity, -mdt = 2), deform with solid displacement
// mpirun -np 4 ./test-mesh-deformation -tf 1.0 -mdt 2 -eo 2 -rs 1 -dt 1e-3 -of /home/shared/Output/MeshDeformation-LinearElasticity-displacement
//
//                    + ----------------------------------------+
//                    |       ___                               |
//                    |      /   \----------|                   |
//                    |      \___/----------|                   |
//                    |                                         |
//                    + ----------------------------------------+
//

// Solid mechanics module
#include "structure/elasticity_solver.hpp"

// Mesh deformation module
#include "ale/mesh_laplacian.hpp"
#include "ale/mesh_linear_elasticity.hpp"

// Interface transfer/FSI
#include "interface_transfer.hpp"

// Mesh utilities
#include "fsi_utils.hpp"

// Output
#include <fstream>
#include "FilesystemHelper.hpp"

using namespace mfem;
using namespace elasticity;
using namespace common_fsi_utils;

using TransferBackend = InterfaceTransfer::Backend;

// Forward declarations of functions
void gravitational_force(const Vector &x, real_t time, Vector &u);

/// Minimum Jacobian determinant and total area/volume of the (deformed) mesh.
void MeshQuality(ParMesh &mesh, real_t &min_detJ, real_t &volume);

/// Volume of the deformed solid, int det(I + grad u) dX, for the displacement
/// @a u defined on the (undeformed) reference solid mesh.
real_t DeformedSolidVolume(ParGridFunction &u);


struct s_ElasticityContext // elasticity params
{
   int order = 2;        // Polynomial order of the elasticity space
   real_t density = 1e3; // Density
   real_t nu = 0.4;      // Poisson's ratio
   real_t E = 2.0e6;     // Young's modulus
   real_t mu;            // G, Shear modulus, Lame's second parameter
   real_t lambda;        // Lame's first parameter
   // const real_t g = 9.80665; // m/s² (standard gravity)
   real_t g = 2.0;
} Elasticity_ctx;

struct s_MeshContext // mesh
{
   real_t D = 0.41;  // Channel diameter
   real_t L = 2.2;   // Channel length
   real_t BL = 0.35; // Elastic beam length
   real_t BT = 0.02; // Elastic beam thickness
   int serial_ref_levels = 0;
   int parallel_ref_levels = 0;
   int order = -1; // polynomial order of the mesh nodes
   real_t E = 1e6; // Young's modulus for the mesh deformation
   real_t nu = 0.35; // Poisson's ratio for the mesh deformation
   int stiffening_type = 1; // 0: Jacobian-based, 1: Distance-based
   bool enable_stiffening = true;
   int distance_solver_type = 0; // 0: Heat, 1: PLaplacian, 2: Normalization
   real_t stiffening_degree = 1.0; // Default stiffening degree
} Mesh_ctx;


const char *outfolder = "./Output/Test-MeshDeformation";
int num_procs;

real_t dt;

int main(int argc, char *argv[])
{

   ///////////////////////////////////////////////////////////////////////////////////////////////
   /// 1. Initialize MPI and Hypre
   ///////////////////////////////////////////////////////////////////////////////////////////////

   Mpi::Init(argc, argv);
   Hypre::Init();

   num_procs = Mpi::WorldSize();
   int myid = Mpi::WorldRank();

   ///////////////////////////////////////////////////////////////////////////////////////////////
   /// 2. Parse command-line options.
   ///////////////////////////////////////////////////////////////////////////////////////////////

   // Mesh file
   const char *mesh_file = "../../data/turek-fsi-new.e";

   // Timestepping parameters
   real_t current_time = 0.0;
   real_t t_final = 2.0;
   dt = 1e-5;

   // If true, the mesh will be deformed with the displacement field, otherwise it will be deformed with the mesh velocity field
   int mesh_deformation_type =
      2; // 0 - Laplacian, 1 - IncrementalLaplacian, 2 - Elasticity
   bool deform_with_displacement = true;
   bool pa = false;
   real_t alpha = 1.0; // Diffusion coefficient for the mesh Laplacian
   int prec_type =
      1; // Type of preconditioner to use for the mesh Laplacian (0 - Jacobi, 1 - Hypre BoomerAMG)

   // Output parameters
   int save_freq = 1;

   int current_step = 0;
   bool last_step = false;

   bool verbose = false; // verbose output during time integration

   OptionsParser args(argc, argv);
   // FE
   args.AddOption(&Elasticity_ctx.order, "-eo", "--elasticity-order",
                  "polynomial_order for solid elasticity space");
   args.AddOption(&Elasticity_ctx.E, "-E", "--young-modulus",
                  "Young's modulus (default: 2.0e6).");
   args.AddOption(&Elasticity_ctx.nu, "-nu", "--poisson-ratio",
                  "Poisson's ratio (default: 0.4).");
   args.AddOption(&Elasticity_ctx.density, "-rho-s", "--rho-solid",
                  "Density of the solid (default: 1e3).");
   args.AddOption(&Elasticity_ctx.g, "-g", "--gravity",
                  "Gravity acceleration (default: 2.0).");
   // Time integration parameters
   args.AddOption(&t_final, "-tf", "--t_final", "t_final");
   args.AddOption(&dt, "-dt", "--dt", "dt.");
   // Mesh parameters
   args.AddOption(&Mesh_ctx.serial_ref_levels, "-rs", "--rs", "serial_ref_levels");
   args.AddOption(&Mesh_ctx.parallel_ref_levels, "-pr", "--pr",
                  "parallel_ref_levels");
   args.AddOption(&Mesh_ctx.order, "-mo", "--mesh-order",
                  "Polynomial order of the mesh nodes (default: 2).");
   args.AddOption(&deform_with_displacement, "-dwd", "--deform-with-displacement",
                  "-dwv", "--deform-with-velocity",
                  "Deform the mesh with the displacement field (default: true). "
                  "If false, the mesh will be deformed with the mesh velocity field.");
   args.AddOption(&mesh_deformation_type, "-mdt", "--mesh-deformation-type",
                  "Type of mesh deformation to use (0 - Laplacian, 1 - IncrementalLaplacian, 2 - Elasticity). "
                  "Default is 2 (Elasticity).");
   args.AddOption(&Mesh_ctx.E, "-mE", "--mesh-young-modulus",
                  "Young's modulus for the mesh deformation (default: 1e6).");
   args.AddOption(&Mesh_ctx.nu, "-mnu", "--mesh-poisson-ratio",
                  "Poisson's ratio for the mesh deformation (default: 0.4).");
   args.AddOption(&Mesh_ctx.enable_stiffening, "-es", "--enable-stiffening",
                  "-no-es", "--no-enable-stiffening",
                  "Enable mesh local stiffening (default: true).");
   args.AddOption(&Mesh_ctx.stiffening_type, "-st", "--stiffening-type",
                  "Type of mesh stiffening to use (0 - Jacobian-based, 1 - Distance-based). "
                  "Default is 1 (Jacobian-based).");
   args.AddOption(&Mesh_ctx.distance_solver_type, "-dst", "--distance-solver-type",
                  "Type of solver to use for the distance function in distance-based stiffening (0 - Heat, 1 - PLaplacian, 2 - Normalization). "
                  "Default is 0 (Heat).");
   args.AddOption(&Mesh_ctx.stiffening_degree, "-msd", "--mesh-stiffening-degree",
                  "Degree of mesh stiffening (default: 1.0). This parameter controls how strongly the mesh stiffening is applied, with higher values leading to stronger stiffening effects.");
   args.AddOption(&pa, "-pa", "--partial-assembly", "-no-pa",
                  "--no-partial-assembly",
                  "Enable partial assembly (default: false). ");
   args.AddOption(&alpha, "-alpha", "--diffusion-coefficient",
                  "Diffusion coefficient for the mesh Laplacian (default: 1.0).");
   args.AddOption(&prec_type, "-prec", "--preconditioner-type",
                  "Type of preconditioner to use for the mesh Laplacian (default: 1). "
                  "0 - Jacobi, 1 - Hypre BoomerAMG.");
   // Output parameters
   args.AddOption(&save_freq, "-sf", "--save-freq",
                  "Save frequency.");
   args.AddOption(&outfolder, "-of", "--output-folder",
                  "Output folder.");
   args.AddOption(&verbose, "-v", "--verbose", "-no-v", "--no-verbose",
                  "Enable verbose output during time integration (default: false).");
   args.ParseCheck();

   // Compute Lame's parameters
   Elasticity_ctx.mu = Elasticity_ctx.E / (2.0 * (1.0 +
                                                  Elasticity_ctx.nu));                                                     // G, Shear modulus, Lame's second parameter
   Elasticity_ctx.lambda = Elasticity_ctx.E * Elasticity_ctx.nu / ((
                                                                      1.0 + Elasticity_ctx.nu) * (1.0 - 2.0 *
                                                                            Elasticity_ctx.nu)); // Lame's first parameter

   if (Mesh_ctx.order < 0)
   {
      Mesh_ctx.order = std::min(Elasticity_ctx.order,
                                2); // mesh order should be at least 2
   }

   // If using the incremental laplacian inform the code that we are using the incremental mesh deformation
   bool incremental_mesh_deformation = mesh_deformation_type == 1;

   ///////////////////////////////////////////////////////////////////////////////////////////////
   /// 3. Create serial Mesh, parallel and fluid/solid sub-meshes
   ///////////////////////////////////////////////////////////////////////////////////////////////

   if (Mpi::Root())
   {
      mfem::out << "\nLoading mesh and refining in serial...";
   }

   Mesh *parent_mesh = new Mesh(mesh_file);
   const int dim = parent_mesh->Dimension();

   if (Mpi::Root())
   {
      mfem::out << "Refining serial mesh...";
   }

   // 1. Refine serial mesh
   for (int i = 0; i < Mesh_ctx.serial_ref_levels; i++)
   {
      parent_mesh->UniformRefinement();
   }

   // Create the sub-domains for the beam and fluid domains
   AttributeSets &attr_sets = parent_mesh->attribute_sets;
   AttributeSets &bdr_attr_sets = parent_mesh->bdr_attribute_sets;

   if (Mpi::Root())
   {
      mfem::out << " done." << std::endl;
      mfem::out << "Partitioning and creating parallel mesh..." << std::endl;
   }

   // Generate mesh partitioning and create a parallel mesh.
   int partition_type = 1;
   int np = num_procs;
   int *partitioning = parent_mesh->GeneratePartitioning(np, partition_type);

   // Create parallel mesh
   ParMesh pmesh(MPI_COMM_WORLD, *parent_mesh, partitioning, partition_type);
   ExportMeshwithPartitioning(outfolder, *parent_mesh, partitioning,
                              "Partitioning"); // filename not defined
   delete[] partitioning;

   for (int l = 0; l < Mesh_ctx.parallel_ref_levels; l++)
   {
      pmesh.UniformRefinement();
   }

   pmesh.SetAttributes();
   pmesh.EnsureNodes();

   if (Mpi::Root())
   {
      mfem::out << "done." << std::endl;
      mfem::out << "Creating sub-meshes...";
   }

   // Extract parallel submeshes
   Array<int> solid_domain_attribute;
   Array<int> fluid_domain_attribute;
   solid_domain_attribute = attr_sets.GetAttributeSet("beam");
   fluid_domain_attribute = attr_sets.GetAttributeSet("fluid");

   auto solid_pmesh = ParSubMesh::CreateFromDomain(pmesh, solid_domain_attribute);
   auto fluid_pmesh = ParSubMesh::CreateFromDomain(pmesh, fluid_domain_attribute);
   fluid_pmesh.SetCurvature(Mesh_ctx.order);

   //////////////////
   // SET MARKERS
   //////////////////

   // Setup boundary markers
   // If we use the parent mesh size we're sure that these markers will be compatible with the different submeshes
   Array<int> fluid_inlet = bdr_attr_sets.GetAttributeSet("inlet");
   Array<int> fluid_outlet = bdr_attr_sets.GetAttributeSet("outlet");
   Array<int> fluid_top_wall = bdr_attr_sets.GetAttributeSet("top_wall");
   Array<int> fluid_bottom_wall = bdr_attr_sets.GetAttributeSet("bottom_wall");
   Array<int> fluid_cylinder = bdr_attr_sets.GetAttributeSet("cylinder");

   // Interface attributes for the solid-fluid interface in parent mesh, used in  BidirectionalInterfaceTransfer
   // @NOTE: if we don't use marker from the parent mesh there might be compatibility issues with the markers depending on the direction of the transfer
   Array<int> beam_cylinder_curve =
      bdr_attr_sets.GetAttributeSet("beam_cylinder_curve");
   Array<int> beam_sides = bdr_attr_sets.GetAttributeSet("beam_sides");
   Array<int> beam_tip = bdr_attr_sets.GetAttributeSet("beam_tip");

   // Combine sides and tip to have a single marker for the whole beam boundary
   Array<int> fluid_solid_interface(2);
   fluid_solid_interface[0] = beam_sides[0];
   fluid_solid_interface[1] = beam_tip[0];
   Array<int> fluid_solid_interface_marker = bdr_attr_sets.AttrToMarker(
                                                parent_mesh->bdr_attributes.Max(), fluid_solid_interface);
   Array<int> fluid_solid_interface_tip_marker = bdr_attr_sets.AttrToMarker(
                                                    parent_mesh->bdr_attributes.Max(), beam_tip);

   delete parent_mesh; // we don't need the parent mesh anymore

   if (Mpi::Root())
   {
      mfem::out << "done." << std::endl;
   }

   ///////////////////////////////////////////////////////////////////////////////////////////////
   /// 4. Create Flow, Solid, and Mesh Deformation Solvers
   ///////////////////////////////////////////////////////////////////////////////////////////////

   /////////////
   // SOLID
   /////////////

   if (Mpi::Root())
   {
      mfem::out << "\nCreating Elasticity solver..." << std::endl;
   }

   // Print mechanical properties
   if (Mpi::Root())
   {
      mfem::out << "Elasticity properties:" << std::endl;
      mfem::out << "  - Young's modulus: " << std::scientific << Elasticity_ctx.E <<
                " Pa" << std::endl;
      mfem::out << "  - Poisson's ratio: " << Elasticity_ctx.nu << " (dimensionless)"
                << std::endl;
      mfem::out << "  - Density: " << std::scientific << Elasticity_ctx.density <<
                " kg/m³" << std::endl;
      mfem::out << "  - Shear modulus (mu): " << std::scientific << Elasticity_ctx.mu
                << " Pa" << std::endl;
      mfem::out << "  - Lame's first parameter (lambda): " << std::scientific <<
                Elasticity_ctx.lambda << " Pa" << std::endl;
   }

   // Create the elasticity solver
   int solver_type = 15; // HHTAlphaSolver
   ElasticitySolver elasticity(solver_type, &solid_pmesh, Elasticity_ctx.order,
                               Elasticity_ctx.density, Elasticity_ctx.lambda, Elasticity_ctx.mu);

   // Grid functions and vectors for the solid
   Vector solid_displacement, solid_displacement_tn;
   Vector solid_vel, solid_vel_tn;

   auto elasticity_fes = elasticity.GetFESpace();

   ParGridFunction solid_displacement_gf(elasticity_fes);
   solid_displacement_gf = 0.0;
   solid_displacement_gf.GetTrueDofs(solid_displacement);
   solid_displacement_gf.GetTrueDofs(solid_displacement_tn);

   ParGridFunction *solid_relative_displacement_gf =
      nullptr; // Relative solid displacement field
   if (incremental_mesh_deformation)
   {
      solid_relative_displacement_gf = new ParGridFunction(
         elasticity_fes); // Relative solid displacement field
      *solid_relative_displacement_gf = 0.0;
   }

   ParGridFunction solid_vel_gf(elasticity_fes);
   solid_vel_gf = 0.0;
   solid_vel_gf.GetTrueDofs(solid_vel);
   solid_vel_gf.GetTrueDofs(solid_vel_tn);

   ParGridFunction mesh_velocity_onSolid_gf(elasticity_fes);
   mesh_velocity_onSolid_gf = 0.0;

   /////////////////////////
   // MESH DEFORMATION
   /////////////////////////

   if (Mpi::Root())
   {
      mfem::out << "Creating mesh deformation solver...";
   }

   auto vfec = H1_FECollection(2, fluid_pmesh.Dimension());
   auto vfes = ParFiniteElementSpace(&fluid_pmesh, &vfec, fluid_pmesh.Dimension());

   MeshDeformation *fluid_mesh_deformation = nullptr;

   switch (mesh_deformation_type)
   {
      case 0: // Laplacian
         fluid_mesh_deformation = new MeshLaplacian(&fluid_pmesh);
         break;
      case 1: // Incremental Laplacian
         fluid_mesh_deformation = new MeshLaplacian(&fluid_pmesh, true);
         break;
      case 2: // Elasticity
         fluid_mesh_deformation = new MeshLinearElasticity(&fluid_pmesh);
         break;
      default:
         MFEM_ABORT("Invalid mesh deformation type.");
   }

   ParFiniteElementSpace *mesh_nodes_fes = fluid_mesh_deformation->GetFESpace();

   ParGridFunction *mesh_velocity_gf = new ParGridFunction(
      mesh_nodes_fes); // Mesh velocity field

   ParGridFunction *mesh_displacement_gf = nullptr;
   VectorGridFunctionCoefficient *mesh_displacement_coeff = nullptr;
   if (deform_with_displacement)
   {
      mesh_displacement_gf = new ParGridFunction(mesh_nodes_fes);
      mesh_displacement_coeff = new VectorGridFunctionCoefficient(
         &solid_displacement_gf);
   }
   else
   {
      mesh_displacement_gf = new ParGridFunction(elasticity_fes);
      mesh_displacement_coeff = new VectorGridFunctionCoefficient(
         mesh_displacement_gf); // Holds the mesh displacement field
   }
   *mesh_displacement_gf = 0.0;

   if (Mpi::Root())
   {
      mfem::out << "done." << std::endl;
   }

   ///////////////////////////////////////////////////////////////////////////////////////////////
   /// 5. Add boundary conditions and initial conditions
   ///////////////////////////////////////////////////////////////////////////////////////////////

   if (Mpi::Root())
   {
      mfem::out << "\nSetting up boundary conditions and initial conditions..." <<
                std::endl;
   }

   ////////////
   // SOLID
   ////////////
   // - Fully Constrained at the cylinder base Γ = 7
   // - Body force (gravity) on the beam

   elasticity.AddFixedConstraint(beam_cylinder_curve[0]);
   elasticity.AddBodyForce(gravitational_force, solid_domain_attribute[0]);

   elasticity.Setup();

   ////////////////////
   // MESH DEFORMATION
   ////////////////////

   Array<int> fluid_mesh_deformation_ess_bdr(fluid_pmesh.bdr_attributes.Max());
   fluid_mesh_deformation_ess_bdr = 1;

   fluid_mesh_deformation->SetEssentialBoundary(fluid_mesh_deformation_ess_bdr);

   if (Mesh_ctx.enable_stiffening)
   {
      fluid_mesh_deformation->EnableLocalStiffening(fluid_solid_interface_tip_marker,
                                                    Mesh_ctx.stiffening_type, Mesh_ctx.distance_solver_type,
                                                    Mesh_ctx.stiffening_degree);
   }

   switch (mesh_deformation_type)
   {
      case 0: // Laplacian
         static_cast<MeshLaplacian *>(fluid_mesh_deformation)->Setup(alpha, pa,
                                                                     prec_type);
         break;
      case 1: // Incremental Laplacian
         static_cast<MeshLaplacian *>(fluid_mesh_deformation)->Setup(alpha, pa,
                                                                     prec_type);
         break;
      case 2: // Elasticity
      {
         static_cast<MeshLinearElasticity *>(fluid_mesh_deformation)->Setup(Mesh_ctx.E,
                                                                            Mesh_ctx.nu, pa, prec_type);
         break;
      }
      default:
         MFEM_ABORT("Invalid mesh deformation type.");
   }

   ///////////////////////////////////////////////////////////////////////////////////////////////
   /// 6. Setup interface transfer
   ///////////////////////////////////////////////////////////////////////////////////////////////

   if (Mpi::Root())
   {
      mfem::out << "Setting up interface transfer for FSI problem... " << std::endl;
   }

   // For the SOLID-->FLUID use the undeformed coordinates (as SOLID is in the undeformed configuration, set by true)
   // For the FLUID-->SOLID use the deformed coordinates (as FLUID is in the deformed configuration set by false). We'll provide the offset as a coefficient in the Update()

   BidirectionalInterfaceTransfer finder_solid_to_mesh(solid_vel_gf.ParFESpace(),
                                                       mesh_nodes_fes, fluid_solid_interface_marker, TransferBackend::GSLIB,
                                                       pmesh.GetComm(), true, !incremental_mesh_deformation);

   // Tolerance for the boundary distance: points for which GSLIB finds distance > tol will be considered as not found.
   // Useful when the offset function used to update the finders is not exact.
   real_t boundary_distance_tolerance = 1e-2;
   finder_solid_to_mesh.SetBoundaryDistanceTolerance(boundary_distance_tolerance,
                                                     boundary_distance_tolerance);

   // Define lambdas to compute QoI for the interface transfer
   int sdim = fluid_pmesh.Dimension();


   ///////////////////////////////////////////////////////////////////////////////////////////////
   /// 7. Set up output
   ///////////////////////////////////////////////////////////////////////////////////////////////

   ParaViewDataCollection pvdc_fluid("Fluid", &fluid_pmesh);
   pvdc_fluid.SetPrefixPath(outfolder);
   pvdc_fluid.SetDataFormat(VTKFormat::BINARY);
   pvdc_fluid.SetHighOrderOutput(true);
   pvdc_fluid.SetCompression(true);
   pvdc_fluid.SetCompressionLevel(9);
   pvdc_fluid.SetLevelsOfDetail(Mesh_ctx.order);
   pvdc_fluid.SetCycle(current_step);
   pvdc_fluid.SetTime(0.0);
   pvdc_fluid.RegisterField("mesh_velocity",
                            mesh_velocity_gf); // CHECK --> Can we store the mesh velocity even tho it has a different fes?
   if (deform_with_displacement)
   {
      pvdc_fluid.RegisterField("mesh_displacement", mesh_displacement_gf);
   }

   ParGridFunction *distance_gf = nullptr;
   ParGridFunction *coeff_gf = nullptr;
   if (Mesh_ctx.enable_stiffening)
   {
      auto fes_stiffening = fluid_mesh_deformation->GetStiffeningFESpace();
      coeff_gf = new ParGridFunction(fes_stiffening);
      *coeff_gf = 0.0;
      auto mesh_deformation_coeff = fluid_mesh_deformation->GetCoefficient();
      coeff_gf->ProjectCoefficient(*mesh_deformation_coeff);
      pvdc_fluid.RegisterField("coeff", coeff_gf);

      if (Mesh_ctx.stiffening_type == 1)
      {
         auto stiffening_coeff = fluid_mesh_deformation->GetLocalStiffeningCoefficient();
         distance_gf = static_cast<DistanceBasedStiffeningCoefficient *>
                       (stiffening_coeff)->GetDistanceGf();
         pvdc_fluid.RegisterField("distance", distance_gf);
      }
   }

   pvdc_fluid.Save();

   ParaViewDataCollection pvdc_solid("Solid", &solid_pmesh);
   pvdc_solid.SetPrefixPath(outfolder);
   pvdc_solid.SetDataFormat(VTKFormat::BINARY);
   pvdc_solid.SetHighOrderOutput(true);
   pvdc_solid.SetCompression(true);
   pvdc_solid.SetCompressionLevel(9);
   pvdc_solid.SetLevelsOfDetail(Elasticity_ctx.order);
   pvdc_solid.SetCycle(current_step);
   pvdc_solid.SetTime(0.0);
   pvdc_solid.RegisterField("displacement", &solid_displacement_gf);
   pvdc_solid.RegisterField("velocity", &solid_vel_gf);
   pvdc_solid.RegisterField("mesh_velocity_onSolid", &mesh_velocity_onSolid_gf);
   pvdc_solid.Save();

   ///////////////////////////////////////////////////////////////////////////////////////////////
   /// 8. Solve Mesh  Deformation problem
   ///////////////////////////////////////////////////////////////////////////////////////////////

   // Nothing for this test case, but might be useful for other cases
   // Reference values for the mesh deformation checks (undeformed meshes)
   real_t min_detJ_0, fluid_volume_0;
   MeshQuality(fluid_pmesh, min_detJ_0, fluid_volume_0);
   const real_t total_volume_0 = fluid_volume_0 + DeformedSolidVolume(
                                    solid_displacement_gf);
   real_t min_detJ_ratio = 1.0, max_volume_defect = 0.0;
   int first_invalid_step = -1; // first step with a non-positive Jacobian

   int paraview_step = 0; // ParaView step counter


   if (Mpi::Root())
   {
      out << "-----------------------------------------------------------------------------------------------------------------------------------------------------------------------"
          << std::endl;
      out << std::left
          << std::setw(8) << "Step"
          << std::setw(16) << "Time"
          << std::setw(16) << "dt"
          << std::setw(16) << "disp_x"
          << std::setw(16) << "disp_y"
          << std::endl;
      out << "-----------------------------------------------------------------------------------------------------------------------------------------------------------------------"
          << std::endl;
   }


   for (int step = 0; !last_step; ++step) // TIME-INTEGRATION LOOP
   {
      if (current_time + dt >= t_final - dt / 2)
      {
         last_step = true;
      }

      // Store the previous solid displacement
      solid_displacement_gf.GetTrueDofs(solid_displacement_tn);

      // --------------------------------------------------------------------------------- //
      //// ----- 1.. Solve Elasticity on SOLID -----
      // --------------------------------------------------------------------------------- //

      if (Mpi::Root() && verbose)
      {
         mfem::out << "Solving Elasticity on SOLID...";
      }

      // Solve
      elasticity.Step(solid_displacement, solid_vel, current_time, dt);
      solid_vel_gf.SetFromTrueDofs(solid_vel);
      solid_displacement_gf.SetFromTrueDofs(solid_displacement);


      if (Mpi::Root() && verbose)
      {
         mfem::out << " done." << std::endl;
      }

      // --------------------------------------------------------------------------------- //

      // --------------------------------------------------------------------------------- //
      //// ----- 2.. Solve Mesh Deformation -----
      // --------------------------------------------------------------------------------- //

      if (Mpi::Root() && verbose)
      {
         mfem::out << "Solve Mesh Deformation...";
      }


      if (Mpi::Root() && verbose)
      {
         mfem::out << "\nDeforming mesh...";
      }

      real_t fluid_mesh_displacement_norm = 0.0;
      if (deform_with_displacement)
      {
         Vector velocity;

         if (incremental_mesh_deformation)
         {
            // Compute relative displacement
            Vector rel_displacement;
            solid_displacement_gf.GetTrueDofs(rel_displacement);
            rel_displacement -=
               solid_displacement_tn;                         // Compute relative displacement
            solid_relative_displacement_gf->SetFromTrueDofs(
               rel_displacement); // Set the relative displacement field
            *mesh_displacement_gf = 0.0;
            finder_solid_to_mesh.InterpolateForward(*solid_relative_displacement_gf,
                                                    *mesh_displacement_gf);
            fluid_mesh_deformation->Solve(*mesh_displacement_gf,
                                          true); // Solve the mesh laplacian for the relative displacement
            mesh_displacement_gf->GetTrueDofs(velocity); // v = d - d_prev
         }
         else
         {
            Vector displacement_prev;
            mesh_displacement_gf->GetTrueDofs(displacement_prev);
            *mesh_displacement_gf = 0.0;
            finder_solid_to_mesh.InterpolateForward(solid_displacement_gf,
                                                    *mesh_displacement_gf);
            fluid_mesh_deformation->Solve(*mesh_displacement_gf,
                                          true); // Solve the mesh laplacian for the total displacement
            mesh_displacement_gf->GetTrueDofs(velocity); // v = d
            velocity -= displacement_prev;               // v = d - d_prev
         }

         velocity *= (1.0 / dt);
         mesh_velocity_gf->SetFromTrueDofs(velocity);

         fluid_mesh_displacement_norm = mesh_displacement_gf->Norml2();

      }
      else
      {
         *mesh_velocity_gf = 0.0;
         finder_solid_to_mesh.InterpolateForward(solid_vel_gf, *mesh_velocity_gf);
         fluid_mesh_deformation->Solve(*mesh_velocity_gf, true,
                                       dt); // We solve the mesh laplacian and deform the mesh with the mesh velocity
         fluid_mesh_displacement_norm = mesh_velocity_gf->Norml2();
         fluid_mesh_displacement_norm *= dt;

         mesh_displacement_gf->Add(dt, solid_vel_gf);
      }

      if (Mpi::Root() && verbose)
         mfem::out << " Moving fluid mesh |dx|_l2 = " << std::scientific <<
                   std::setprecision(3)
                   << fluid_mesh_displacement_norm << "" << std::endl;

      if (Mpi::Root() && verbose)
      {
         mfem::out << "Updating MeshDeformation solver with new mesh deformation...";
      }

      fluid_mesh_deformation->Update();
      //elasticity.Update(); --> Not needed, as we don't deform the solid mesh

      if (Mpi::Root() && verbose)
      {
         mfem::out << "Updating InterfaceTransfer with new mesh deformation...";
      }

      // the backward transfer (MESH-->SOLID) needs deformed coordinates on source, so we pass the mesh displacement coefficient
      finder_solid_to_mesh.Update(true, true, nullptr, mesh_displacement_coeff);

      // Mesh deformation checks:
      // 1. the deformed fluid mesh must stay valid (positive Jacobian);
      // 2. the deformed fluid mesh must stay conforming with the deformed solid:
      //    since the outer boundary is fixed, the fluid area plus the deformed
      //    solid area is conserved, and any gap or overlap at the interface
      //    shows up as an area defect.
      {
         real_t min_detJ, fluid_volume;
         MeshQuality(fluid_pmesh, min_detJ, fluid_volume);
         const real_t solid_volume = DeformedSolidVolume(solid_displacement_gf);
         const real_t detJ_ratio = min_detJ / min_detJ_0;
         const real_t volume_defect =
            std::abs(fluid_volume + solid_volume - total_volume_0) / total_volume_0;
         min_detJ_ratio = std::min(min_detJ_ratio, detJ_ratio);
         max_volume_defect = std::max(max_volume_defect, volume_defect);
         if (detJ_ratio <= 0.0 && first_invalid_step < 0)
         {
            first_invalid_step = step;
         }
         if (Mpi::Root() && verbose)
         {
            mfem::out << "Mesh checks: min detJ / min detJ_0 = " << std::scientific
                      << std::setprecision(4) << detJ_ratio
                      << " | relative area defect = " << volume_defect << std::endl;
         }
      }

      if (Mpi::Root() && verbose)
      {
         mfem::out << " done." << std::endl;
      }

      // --------------------------------------------------------------------------------- //

      // --------------------------------------------------------------------------------- //
      //// ----- 3.. Interpolate MESH --> SOLID -----
      // --------------------------------------------------------------------------------- //

      // Interpolate some quantity just to check the transfer MESH --> SOLID is working
      mesh_velocity_onSolid_gf = 0.0;
      finder_solid_to_mesh.InterpolateBackward(*mesh_velocity_gf,
                                               mesh_velocity_onSolid_gf);

      // --------------------------------------------------------------------------------- //

      // Output
      paraview_step++;
      if ((step + 1) % save_freq == 0 || last_step)
      {
         if (Mpi::Root() && verbose)
         {
            mfem::out << "\nWriting output to ParaView...";
         }
         pvdc_fluid.SetCycle(paraview_step);
         pvdc_fluid.SetTime(current_time);
         pvdc_fluid.Save();
         pvdc_solid.SetCycle(paraview_step);
         pvdc_solid.SetTime(current_time);
         pvdc_solid.Save();
         if (Mpi::Root() && verbose)
         {
            mfem::out << " done." << std::endl;
         }
      }

      // Output some information
      // Output some information
      bool compute_output = true;
      real_t disp_x = 0.0, disp_y = 0.0;
      if (compute_output)
      {
         Vector point1(2), point2(2);
         point1(0) = -1.1; // Cylinder left vertex (fixed)
         point1(1) = 0.0;
         point2(0) = -0.65; // Beam right vertex (free)
         point2(1) = 0.01;

         Array<int> elem_ids;
         Array<IntegrationPoint> ips;

         DenseMatrix points_solid(dim, 1);
         points_solid.SetCol(0, point2);
         solid_pmesh.FindPoints(points_solid, elem_ids, ips);

         // Initialize displacement values
         disp_x = 0.0;
         disp_y = 0.0;

         // Only evaluate grid functions if the point was found on this rank
         Vector val(2);
         if (elem_ids[0] >= 0)
         {
            solid_displacement_gf.GetVectorValue(elem_ids[0], ips[0], val);
            disp_x = val(0);
            disp_y = val(1);
         }

         // Reduce displacement values across all ranks
         real_t global_disp_vals[2] = {disp_x, disp_y};
         real_t reduced_disp_vals[2];
         MPI_Allreduce(global_disp_vals, reduced_disp_vals, 2, MPI_DOUBLE, MPI_SUM,
                       MPI_COMM_WORLD);

         disp_x = reduced_disp_vals[0];
         disp_y = reduced_disp_vals[1];

      }

      if (Mpi::Root())
      {
         out << std::left
             << std::setw(8) << step
             << std::setw(16) << std::scientific << std::setprecision(8) << current_time
             << std::setw(16) << std::scientific << std::setprecision(8) << dt
             << std::setw(16) << std::scientific << std::setprecision(8) << disp_x
             << std::setw(16) << std::scientific << std::setprecision(8) << disp_y
             << std::endl;
      }


   }

   ///////////////////////////////////////////////////////////////////////////////////////////////
   /// 9. Cleanup
   ///////////////////////////////////////////////////////////////////////////////////////////////

   delete mesh_velocity_gf;

   // Summary of the mesh deformation checks
   {
      const real_t volume_defect_tol = 1e-6;
      const bool pass = (min_detJ_ratio > 0.0) &&
                        (max_volume_defect < volume_defect_tol);
      if (Mpi::Root())
      {
         mfem::out << "\nMesh deformation checks: min detJ / min detJ_0 = "
                   << std::scientific << std::setprecision(4) << min_detJ_ratio
                   << " | max relative area defect (fluid + deformed solid) = "
                   << max_volume_defect << " (tol " << volume_defect_tol << ") --> "
                   << (pass ? "PASSED" : "FAILED") << std::endl;
         if (first_invalid_step >= 0)
         {
            mfem::out << "First step with an invalid (inverted) fluid mesh: "
                      << first_invalid_step << std::endl;
         }
      }
      MFEM_VERIFY(pass, "Mesh deformation checks failed.");
   }

   delete fluid_mesh_deformation; // Mesh deformation solver

   if (incremental_mesh_deformation)
   {
      delete solid_relative_displacement_gf; // Relative solid displacement field
   }

   delete coeff_gf;

   return 0;
}

void gravitational_force(const Vector &x, real_t time, Vector &u)
{
   u(1) = -Elasticity_ctx.g * Elasticity_ctx.density;
   u(0) = 0.0;
};

void MeshQuality(ParMesh &mesh, real_t &min_detJ, real_t &volume)
{
   const int order = mesh.GetNodes() ?
                     mesh.GetNodes()->FESpace()->GetMaxElementOrder() : 1;
   real_t loc_min = infinity(), loc_vol = 0.0;
   IntegrationRules gll_rules(0, Quadrature1D::GaussLobatto);
   for (int e = 0; e < mesh.GetNE(); e++)
   {
      ElementTransformation *T = mesh.GetElementTransformation(e);
      // Closed (Gauss-Lobatto) rule, so that the element vertices are sampled
      const IntegrationRule &ir_min =
         gll_rules.Get(T->GetGeometryType(), 2 * order + 1);
      for (int i = 0; i < ir_min.GetNPoints(); i++)
      {
         T->SetIntPoint(&ir_min.IntPoint(i));
         loc_min = std::min(loc_min, T->Jacobian().Det());
      }
      const IntegrationRule &ir = IntRules.Get(T->GetGeometryType(), 2 * order + 2);
      for (int i = 0; i < ir.GetNPoints(); i++)
      {
         const IntegrationPoint &ip = ir.IntPoint(i);
         T->SetIntPoint(&ip);
         loc_vol += ip.weight * T->Weight();
      }
   }
   MPI_Allreduce(&loc_min, &min_detJ, 1, MPITypeMap<real_t>::mpi_type, MPI_MIN,
                 mesh.GetComm());
   MPI_Allreduce(&loc_vol, &volume, 1, MPITypeMap<real_t>::mpi_type, MPI_SUM,
                 mesh.GetComm());
}

real_t DeformedSolidVolume(ParGridFunction &u)
{
   ParMesh &mesh = *u.ParFESpace()->GetParMesh();
   const int dim = mesh.Dimension();
   const int order = u.ParFESpace()->GetMaxElementOrder();
   DenseMatrix F(dim);
   real_t loc_vol = 0.0, volume = 0.0;
   for (int e = 0; e < mesh.GetNE(); e++)
   {
      ElementTransformation *T = mesh.GetElementTransformation(e);
      const IntegrationRule &ir = IntRules.Get(T->GetGeometryType(),
                                               2 * dim * order + 2);
      for (int i = 0; i < ir.GetNPoints(); i++)
      {
         const IntegrationPoint &ip = ir.IntPoint(i);
         T->SetIntPoint(&ip);
         u.GetVectorGradient(*T, F);
         for (int d = 0; d < dim; d++) { F(d, d) += 1.0; }
         loc_vol += ip.weight * T->Weight() * F.Det();
      }
   }
   MPI_Allreduce(&loc_vol, &volume, 1, MPITypeMap<real_t>::mpi_type, MPI_SUM,
                 mesh.GetComm());
   return volume;
}
