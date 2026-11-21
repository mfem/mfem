// =====================================================================================
//          Example for Local Stiffening Coefficient
// -------------------------------------------------------------------------------------
// Sample runs:
//
// mpirun -np 4 ./test-local-stiffening
//
//                    + ----------------------------------------+
//                    |       ___                               |
//                    |      /   \----------|                   |
//                    |      \___/----------|                   |
//                    |                                         |
//                    + ----------------------------------------+
//


// Mesh deformation module
#include "ale/local_stiffening_coefficient.hpp"

// Mesh utilities
#include "fsi_utils.hpp"

// Output
#include <fstream>
#include "FilesystemHelper.hpp"

using namespace mfem;

struct s_MeshContext // mesh
{
   real_t D = 0.41;  // Channel diameter
   real_t L = 2.2;   // Channel length
   real_t BL = 0.35; // Elastic beam length
   real_t BT = 0.02; // Elastic beam thickness
   int serial_ref_levels = 0;
   int parallel_ref_levels = 0;
   int order = 2; // polynomial order of the mesh nodes
   real_t E = 1e6; // Young's modulus for the mesh deformation
   real_t nu = 0.4; // Poisson's ratio for the mesh deformation
   real_t mu;      // G, Shear modulus, Lame's second parameter
   real_t lambda;  // Lame's first parameter
   int stiffening_type = 0; // 0: Jacobian-based, 1: Distance-based
   real_t stiffening_degree = 1.0; // Default stiffening degree
   int distance_solver_type = 0; // 0: Heat, 1: PLaplacian, 2: Normalization
   bool filter_ls = false; // Whether to apply filtering to the level set function
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

   OptionsParser args(argc, argv);

   // Mesh parameters
   args.AddOption(&Mesh_ctx.serial_ref_levels, "-rs", "--rs", "serial_ref_levels");
   args.AddOption(&Mesh_ctx.parallel_ref_levels, "-pr", "--pr",
                  "parallel_ref_levels");
   args.AddOption(&Mesh_ctx.order, "-mo", "--mesh-order",
                  "Polynomial order of the mesh nodes (default: 2).");
   args.AddOption(&Mesh_ctx.E, "-mE", "--mesh-young-modulus",
                  "Young's modulus for the mesh deformation (default: 1e6).");
   args.AddOption(&Mesh_ctx.nu, "-mnu", "--mesh-poisson-ratio",
                  "Poisson's ratio for the mesh deformation (default: 0.4).");
   args.AddOption(&Mesh_ctx.stiffening_type, "-mst", "--mesh-stiffening-type",
                  "Stiffening type for the mesh deformation (default: 0).\n"
                  "0: Jacobian-based, 1: Distance-based");
   args.AddOption(&Mesh_ctx.stiffening_degree, "-msd", "--mesh-stiffening-degree",
                  "Stiffening degree for the jacobian-based mesh local stiffening (default: 1.0).");
   args.AddOption(&Mesh_ctx.distance_solver_type, "-mdt",
                  "--mesh-distance-solver-type",
                  "Distance solver type for the mesh deformation (default: 0).\n"
                  "0: Heat, 1: PLaplacian, 2: Normalization");
   args.AddOption(&Mesh_ctx.filter_ls, "-fls", "--filter-level-set", "-no-fls",
                  "--no-filter-level-set",
                  "Whether to apply filtering to the level set function (default: false).");

   // Output parameters
   args.AddOption(&outfolder, "-of", "--output-folder",
                  "Output folder.");

   args.ParseCheck();

   // Compute Lame's parameters
   Mesh_ctx.mu = Mesh_ctx.E / (2.0 * (1.0 +
                                      Mesh_ctx.nu));                                                     // G, Shear modulus, Lame's second parameter
   Mesh_ctx.lambda = Mesh_ctx.E * Mesh_ctx.nu / ((1.0 + Mesh_ctx.nu) *
                                                 (1.0 - 2.0 * Mesh_ctx.nu)); // Lame's first parameter


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
   common_fsi_utils::ExportMeshwithPartitioning(outfolder, *parent_mesh,
                                                partitioning, "Partitioning"); // filename not defined
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
   /// 4. Create Coefficients and setup output
   ///////////////////////////////////////////////////////////////////////////////////////////////

   H1_FECollection fec(Mesh_ctx.order, fluid_pmesh.Dimension());
   ParFiniteElementSpace fespace(&fluid_pmesh, &fec);

   // 7. Define the solution x as a finite element grid function in fespace. Set
   //    the initial guess to zero, which also sets the boundary conditions.
   ParGridFunction x(&fespace);
   x = 0.0;

   ParGridFunction nu_gf(&fespace); nu_gf = Mesh_ctx.nu; // Poisson's ratio
   ParGridFunction mu_gf(&fespace); mu_gf = Mesh_ctx.mu; // Shear modulus
   ParGridFunction mu_stiff_gf(&fespace);

   LocalStiffeningCoefficient *stiff_coeff = nullptr;

   switch (Mesh_ctx.stiffening_type)
   {
      case 0: // Jacobian-Based
      {
         stiff_coeff = new JacobianBasedStiffeningCoefficient(&fespace,
                                                              fluid_solid_interface_tip_marker, Mesh_ctx.stiffening_degree);
         break;
      }
      case 1: // Distance-Based
      {
         stiff_coeff = new DistanceBasedStiffeningCoefficient(&fespace,
                                                              fluid_solid_interface_tip_marker, Mesh_ctx.distance_solver_type,
                                                              Mesh_ctx.filter_ls);
         real_t Rmin =
            0.01; // Minimum distance to wall boundaries (start of transition zone)
         real_t Rmax =
            0.05;  // Maximum distance to wall boundaries (end of transition zone)
         real_t alpha = 10.0; // Stiffening exponent
         real_t Amax = 1e2;  // Maximum stiffening coefficient
         real_t Amin = 1e-2;  // Minimum stiffening coefficient
         static_cast<DistanceBasedStiffeningCoefficient*>(stiff_coeff)->SetParameters(
            Rmin, Rmax, alpha, Amax, Amin, Mesh_ctx.stiffening_degree);
         break;
      }
      default:
         mfem_error("Unknown stiffening type");
         break;
   }

   ProductCoefficient mu_stiff_coeff(Mesh_ctx.mu, *stiff_coeff);
   mu_stiff_gf.ProjectCoefficient(mu_stiff_coeff);


   // Checks
   bool pass = true;
   {
      // The stiffened modulus must be positive and finite everywhere.
      real_t loc_min = infinity(), loc_max = -infinity(), gmin, gmax;
      for (int i = 0; i < mu_stiff_gf.Size(); i++)
      {
         loc_min = std::min(loc_min, mu_stiff_gf(i));
         loc_max = std::max(loc_max, mu_stiff_gf(i));
      }
      MPI_Allreduce(&loc_min, &gmin, 1, MFEM_MPI_REAL_T, MPI_MIN,
                    fluid_pmesh.GetComm());
      MPI_Allreduce(&loc_max, &gmax, 1, MFEM_MPI_REAL_T, MPI_MAX,
                    fluid_pmesh.GetComm());
      const bool ok = gmin > 0.0 && std::isfinite(gmax);
      pass = pass && ok;
      if (Mpi::Root())
      {
         mfem::out << "Stiffened modulus range: [" << std::scientific << gmin << ", " <<
                   gmax
                   << "] --> " << (ok ? "ok" : "NOT positive/finite") << std::endl;
      }
   }
   if (Mesh_ctx.stiffening_type == 1)
   {
      // Compare the computed distance with the exact Euclidean distance to the
      // beam tip (a vertical segment), in the stiffening zone d <= Rmax where it
      // controls the coefficient. The beam tip is taken from the boundary
      // elements marked as tip in the fluid mesh.
      real_t loc_bb[4] = {infinity(), infinity(), infinity(), infinity()}, bb[4];
      for (int be = 0; be < fluid_pmesh.GetNBE(); be++)
      {
         if (!fluid_solid_interface_tip_marker[fluid_pmesh.GetBdrAttribute(be) - 1]) { continue; }
         Array<int> verts;
         fluid_pmesh.GetBdrElementVertices(be, verts);
         for (int v : verts)
         {
            const real_t *X = fluid_pmesh.GetVertex(v);
            loc_bb[0] = std::min(loc_bb[0], X[0]);
            loc_bb[1] = std::min(loc_bb[1], X[1]);
            loc_bb[2] = std::min(loc_bb[2], -X[0]);
            loc_bb[3] = std::min(loc_bb[3], -X[1]);
         }
      }
      MPI_Allreduce(loc_bb, bb, 4, MFEM_MPI_REAL_T, MPI_MIN, fluid_pmesh.GetComm());
      const real_t tx0 = bb[0], ty0 = bb[1], tx1 = -bb[2], ty1 = -bb[3];
      auto exact_dist = [=](const Vector &X)
      {
         const real_t px = std::min(std::max(X(0), tx0), tx1);
         const real_t py = std::min(std::max(X(1), ty0), ty1);
         return std::hypot(X(0) - px, X(1) - py);
      };
      FunctionCoefficient exact_dist_coeff(exact_dist);
      ParGridFunction exact_dist_gf(&fespace);
      exact_dist_gf.ProjectCoefficient(exact_dist_coeff);
      ParGridFunction &dist_gf =
         *static_cast<DistanceBasedStiffeningCoefficient*>(stiff_coeff)->GetDistanceGf();
      const real_t Rmax = 0.05, transition_width = 0.05 - 0.01;
      real_t loc_err = 0.0, err;
      for (int i = 0; i < dist_gf.Size(); i++)
      {
         if (exact_dist_gf(i) <= Rmax)
         {
            loc_err = std::max(loc_err, std::abs(dist_gf(i) - exact_dist_gf(i)));
         }
      }
      MPI_Allreduce(&loc_err, &err, 1, MFEM_MPI_REAL_T, MPI_MAX,
                    fluid_pmesh.GetComm());
      // The distance error shifts the stiffening transition zone: require it
      // to be well below the zone width.
      const bool ok = err < 0.25 * transition_width;
      pass = pass && ok;
      if (Mpi::Root())
      {
         mfem::out << "Beam tip segment: x in [" << tx0 << ", " << tx1 << "], y in ["
                   << ty0 << ", " << ty1 << "]" << std::endl;
         mfem::out << "Distance to the beam tip, max error for d <= Rmax: " <<
                   std::scientific
                   << err << " (tol " << 0.25 * transition_width << ") --> "
                   << (ok ? "ok" : "TOO LARGE") << std::endl;
      }
   }
   if (Mpi::Root())
   {
      mfem::out << "Local stiffening checks " << (pass ? "PASSED" : "FAILED") <<
                std::endl;
   }

   ParaViewDataCollection pvdc_fluid("Fluid", &fluid_pmesh);
   pvdc_fluid.SetPrefixPath(outfolder);
   pvdc_fluid.SetDataFormat(VTKFormat::BINARY);
   pvdc_fluid.SetHighOrderOutput(true);
   pvdc_fluid.SetCompression(true);
   pvdc_fluid.SetCompressionLevel(9);
   pvdc_fluid.SetLevelsOfDetail(Mesh_ctx.order);
   pvdc_fluid.RegisterField("poisson", &nu_gf);
   pvdc_fluid.RegisterField("mu", &mu_gf);
   pvdc_fluid.RegisterField("mu_stiff", &mu_stiff_gf);
   if (Mesh_ctx.stiffening_type == 1)
   {
      pvdc_fluid.RegisterField("distance",
                               static_cast<DistanceBasedStiffeningCoefficient*>(stiff_coeff)->GetDistanceGf());
      pvdc_fluid.RegisterField("level_set",
                               static_cast<DistanceBasedStiffeningCoefficient*>(stiff_coeff)->GetLevelSetGf());
   }

   pvdc_fluid.Save();


   ///////////////////////////////////////////////////////////////////////////////////////////////
   /// 5. Cleanup
   ///////////////////////////////////////////////////////////////////////////////////////////////

   delete stiff_coeff;

   return pass ? 0 : 1;
}


