// =====================================================================================
//          2D Interface Transfer Example using InterfaceTransfer Class
// -------------------------------------------------------------------------------------
// This example demonstrates the transfer of field data and QoI (e.g., temperature and
// heat flux) across an interface on a 2D mesh.
// It sets up a parallel mesh with fluid and solid subdomains, projects analytic fields,
// and performs data transfer between subdomains using both a native SubMesh transfer
// map and GSLIB interpolation. The results are exported for visualization in ParaView.
//
// The following mesh is used:
//
//               7
//         -------------
//        |             |
//    5   |    Fluid    | 6
//        |   Domain    |
//        |     (2)     |
//        |             |
//         --4---3---4--  Interface
//        |             |
//        |    Solid    |
//     2  |   Domain    | 2
//        |     (1)     |
//        |             |
//         -------------
//               1
//
// We define an analytical temperature field on the fluid domain and transfer
// temperature and heat flux (k ∇T) from FLUID --> SOLID domain.
//
// Requirements:
//   - MFEM built with NetCDF and GSLIB support
//   - Input mesh files: multidomain2d-quad.e or multidomain2d-tri.e
//
// Sample runs:
//
//   1. Same order for source and destination (gradient normal to interface):
//      mpirun -np 2 ./transfer2d -os 2 -od 2 -of ./Output/Transfer2D
//
//   2. Same order for source and destination (gradient tangent to interface):
//      mpirun -np 2 ./transfer2d -os 2 -od 2 -f 1 -of ./Output/Transfer2D
//
//   3. Different orders for source and destination (gradient normal to interface):
//      mpirun -np 2 ./transfer2d -os 1 -od 2 -of ./Output/Transfer2D
//
//   4. Same order for source and destination, but src is byNODES and dst is byVDIM:
//      mpirun -np 2 ./transfer2d -os 2 -od 2 --by-nodes-src --by-vdim-dst -f 1 -of ./Output/Transfer2D
//
// =====================================================================================

#include "mfem.hpp"
#include "../fsi_utils.hpp"

// Interface transfer
#include "../interface_transfer.hpp"

#include <fstream>
#include <sstream>
#include <iostream>
#include <memory>
#include "../FilesystemHelper.hpp"

#include <unistd.h>

using namespace mfem;

using InterfaceTransfer = common_fsi_utils::InterfaceTransfer;
using BidirectionalInterfaceTransfer =
   common_fsi_utils::BidirectionalInterfaceTransfer;
using TransferBackend = InterfaceTransfer::Backend;

int main(int argc, char *argv[])
{

   /*{ // Uncomment this block to ease MPI debuggings
      int i = 0;
      while (0 == i)
         sleep(5);
   } */

   ///////////////////////////////////////////////////////////////////////////////////////////////
   /// 1. Initialize MPI and Hypre
   ///////////////////////////////////////////////////////////////////////////////////////////////

   Mpi::Init();
   Hypre::Init();
   int num_procs = Mpi::WorldSize();
   int myid = Mpi::WorldRank();

   ///////////////////////////////////////////////////////////////////////////////////////////////
   /// 2. Parse command-line options.
   ///////////////////////////////////////////////////////////////////////////////////////////////

   // Function
   int function = 0; // 0: normal, 1: tangent to interface
   // FE
   int order_src = 1;
   int order_dst = 1;
   bool reorder_space_src = true; // byNODES (default) or byVDIM
   bool reorder_space_dst = true; // byNODES (default) or byVDIM
   // Mesh
   int serial_ref_levels = 0;
   int parallel_ref_levels = 0;
   bool hex = true;
   // Postprocessing
   bool paraview = true;
   const char *outfolder = "";

   OptionsParser args(argc, argv);
   args.AddOption(&function, "-f", "--function",
                  "Function to use for the transfer (0: normal, 1: tangent to interface).");
   // FE
   args.AddOption(&order_src, "-os", "--order-src",
                  "Finite element order (polynomial degree) for source.");
   args.AddOption(&order_dst, "-od", "--order-dst",
                  "Finite element order (polynomial degree) for destination.");
   args.AddOption(&reorder_space_src, "-nodes-src", "--by-nodes-src", "-vdim-src",
                  "--by-vdim-src",
                  "Use byNODES ordering of vector space instead of byVDIM");
   args.AddOption(&reorder_space_dst, "-nodes-dst", "--by-nodes-dst", "-vdim-dst",
                  "--by-vdim-dst",
                  "Use byNODES ordering of vector space instead of byVDIM");
   // Mesh
   args.AddOption(&hex, "-hex", "--hex-mesh", "-tet", "--tet-mesh",
                  "Use hexahedral mesh.");
   args.AddOption(&serial_ref_levels, "-rs", "--serial-ref-levels",
                  "Number of serial refinement levels.");
   args.AddOption(&parallel_ref_levels, "-rp", "--parallel-ref-levels",
                  "Number of parallel refinement levels.");
   // Postprocessing
   args.AddOption(&paraview, "--paraview", "-paraview", "-no-paraview",
                  "--no-paraview",
                  "Enable or disable Paraview visualization.");
   args.AddOption(&outfolder, "-of", "--out-folder",
                  "Output folder.");

   args.ParseCheck();


   // Disable native backend if the order is not the same for source and destination
   bool enable_native_backend = (order_src == order_dst);
#ifndef MFEM_USE_GSLIB
   // Without GSLIB only the Native transfer exists, which needs the same FE space on both sides
   MFEM_VERIFY(enable_native_backend,
               "MFEM built without GSLIB: only the Native transfer is available, use -os == -od.");
#endif

   // Determine ordering
   auto ordering_src = reorder_space_src ? mfem::Ordering::byNODES :
                       mfem::Ordering::byVDIM;
   auto ordering_dst = reorder_space_dst ? mfem::Ordering::byNODES :
                       mfem::Ordering::byVDIM;

   ///////////////////////////////////////////////////////////////////////////////////////////////
   /// 3. Create serial Mesh and parallel
   ///////////////////////////////////////////////////////////////////////////////////////////////

   if (Mpi::Root())
   {
      mfem::out << "\nLoading mesh... ";
   }

   // Load serial mesh
   Mesh *serial_mesh = nullptr;
#ifdef MFEM_USE_NETCDF
   if (hex)
      // Load mesh (NETCDF required)
   {
      serial_mesh = new Mesh("../../data/multidomain2d-quad.e");
   }
   else
   {
      serial_mesh = new Mesh("../../data/multidomain2d-tri.e");
   }
#else
   MFEM_ABORT("MFEM is not built with NetCDF support!");
#endif

   int sdim = serial_mesh->SpaceDimension();
   serial_mesh->EnsureNodes();

   if (Mpi::Root())
   {
      mfem::out << "done." << std::endl;
   }

   // Generate mesh partitioning

   if (Mpi::Root())
   {
      mfem::out << "Generating partitioning and creating parallel mesh... ";
   }

   // Partition type:
   // 0) METIS_PartGraphRecursive (sorted neighbor lists)
   // 1) METIS_PartGraphKway      (sorted neighbor lists) (default)
   // 2) METIS_PartGraphVKway     (sorted neighbor lists)
   // 3) METIS_PartGraphRecursive
   // 4) METIS_PartGraphKway
   // 5) METIS_PartGraphVKway
   int partition_type = 1;
   int np = num_procs;
   int *partitioning = serial_mesh->GeneratePartitioning(np, partition_type);

   // Create parallel mesh
   ParMesh parent_mesh = ParMesh(MPI_COMM_WORLD, *serial_mesh, partitioning,
                                 partition_type);
   common_fsi_utils::ExportMeshwithPartitioning(outfolder, *serial_mesh,
                                                partitioning);
   delete[] partitioning;
   delete serial_mesh;

   for (int l = 0; l < parallel_ref_levels; l++)
   {
      parent_mesh.UniformRefinement();
   }

   if (Mpi::Root())
   {
      mfem::out << "done." << std::endl;
      mfem::out << "Creating sub-meshes... ";
   }

   // Create the sub-domains for the solid, solid and fluid domains
   AttributeSets &attr_sets = parent_mesh.attribute_sets;
   AttributeSets &bdr_attr_sets = parent_mesh.bdr_attribute_sets;

   Array<int> solid_domain_attribute;
   Array<int> fluid_domain_attribute;

   solid_domain_attribute = attr_sets.GetAttributeSet("Solid");
   fluid_domain_attribute = attr_sets.GetAttributeSet("Fluid");

   auto solid_submesh =
      std::make_shared<ParSubMesh>(ParSubMesh::CreateFromDomain(parent_mesh,
                                                                solid_domain_attribute));

   auto fluid_submesh =
      std::make_shared<ParSubMesh>(ParSubMesh::CreateFromDomain(parent_mesh,
                                                                fluid_domain_attribute));

   // Define interface markers
   Array<int> fluid_solid_interface;
   Array<int> solid_cylinder_interface;

   Array<int> fluid_lateral_left_attr;
   Array<int> fluid_lateral_right_attr;
   Array<int> fluid_top_attr;
   Array<int> solid_lateral_attr;
   Array<int> solid_bottom_attr;

   Array<int> fluid_solid_interface_marker;
   Array<int> solid_cylinder_interface_marker;

   // Extract boundary attributes
   fluid_solid_interface = bdr_attr_sets.GetAttributeSet("Fluid-Solid");
   solid_cylinder_interface = bdr_attr_sets.GetAttributeSet("Cylinder-Solid");

   fluid_lateral_left_attr = bdr_attr_sets.GetAttributeSet("Fluid-Lateral-Left");
   fluid_lateral_right_attr = bdr_attr_sets.GetAttributeSet("Fluid-Lateral-Right");
   fluid_top_attr = bdr_attr_sets.GetAttributeSet("Fluid-Top");
   solid_lateral_attr = bdr_attr_sets.GetAttributeSet("Solid-Lateral");
   solid_bottom_attr = bdr_attr_sets.GetAttributeSet("Solid-Bottom");

   // Extract boundary attributes markers on parent mesh (needed for GSLIB interpolation)
   fluid_solid_interface_marker =
      bdr_attr_sets.GetAttributeSetMarker("Fluid-Solid");
   solid_cylinder_interface_marker =
      bdr_attr_sets.GetAttributeSetMarker("Cylinder-Solid");

   Array<int> solid_domain_attributes = AttributeSets::AttrToMarker(
                                           solid_submesh->attributes.Max(), solid_domain_attribute);
   Array<int> fluid_domain_attributes = AttributeSets::AttrToMarker(
                                           fluid_submesh->attributes.Max(), fluid_domain_attribute);

   if (Mpi::Root())
   {
      mfem::out << "done." << std::endl;
   }

   ///////////////////////////////////////////////////////////////////////////////////////////////
   /// 4. Create GridFunctions and Coefficients
   ///////////////////////////////////////////////////////////////////////////////////////////////

   if (Mpi::Root())
   {
      mfem::out << "\nCreating FE space and GridFunctions... ";
   }

   // Create the Finite Element spaces and grid functions
   H1_FECollection fec_src(order_src, parent_mesh.Dimension());
   H1_FECollection fec_dst(order_dst, parent_mesh.Dimension());

   // SOLID is destination, FLUID is source
   ParFiniteElementSpace fes_solid(solid_submesh.get(), &fec_dst, 1, ordering_dst);
   ParFiniteElementSpace fes_solid_grad(solid_submesh.get(), &fec_dst, sdim,
                                        ordering_dst);

   ParFiniteElementSpace fes_fluid(fluid_submesh.get(), &fec_src, 1, ordering_src);
   ParFiniteElementSpace fes_fluid_grad(fluid_submesh.get(), &fec_src, sdim,
                                        ordering_src);

   ParGridFunction temperature_solid_gf(&fes_solid);
   temperature_solid_gf = 0.0;
   ParGridFunction heatFlux_norm_solid_gf(&fes_solid);
   heatFlux_norm_solid_gf = 0.0;
   ParGridFunction heatFlux_solid_gf(&fes_solid_grad);
   heatFlux_solid_gf = 0.0;
   ParGridFunction heatFlux_exact_transfered_gf(&fes_solid_grad);
   heatFlux_exact_transfered_gf = 0.0;
   ParGridFunction temperature_solid_gf_copy(&fes_solid);
   temperature_solid_gf_copy = 0.0;

   ParGridFunction temperature_fluid_gf(&fes_fluid);
   temperature_fluid_gf = 0.0;
   ParGridFunction heatFlux_fluid_exact_gf(&fes_fluid_grad);
   heatFlux_fluid_exact_gf = 0.0;

   // Create Coefficients for conductivity
   ConstantCoefficient fluid_conductivity(1.0);
   ConstantCoefficient solid_conductivity(1.0);

   if (Mpi::Root())
   {
      mfem::out << "done." << std::endl;
   }

   ///////////////////////////////////////////////////////////////////////////////////////////////
   /// 5. Setup interface transfer
   ///////////////////////////////////////////////////////////////////////////////////////////////

   // Create the submesh transfer map
   if (Mpi::Root())
   {
      mfem::out << "Creating InterfaceTransfer (Hybrid)...";
   }

   InterfaceTransfer *finder_fluid_to_solid_hybrid = nullptr;
   InterfaceTransfer *finder_fluid_to_solid_gslib = nullptr;

   if (enable_native_backend)
#ifdef MFEM_USE_GSLIB
      finder_fluid_to_solid_hybrid = new InterfaceTransfer(&fes_fluid, &fes_solid,
                                                           fluid_solid_interface_marker, TransferBackend::Hybrid, parent_mesh.GetComm());
#else
      finder_fluid_to_solid_hybrid = new InterfaceTransfer(&fes_fluid, &fes_solid,
                                                           fluid_solid_interface_marker, TransferBackend::Native, parent_mesh.GetComm());
#endif

   if (Mpi::Root())
   {
      mfem::out << "done." << std::endl;
   }

   // Setup GSLIB interpolation
   if (Mpi::Root())
   {
      mfem::out << "Creating InterfaceTransfer (GSLIB)...";
   }

#ifdef MFEM_USE_GSLIB
   finder_fluid_to_solid_gslib = new InterfaceTransfer(&fes_fluid, &fes_solid,
                                                       fluid_solid_interface_marker, TransferBackend::GSLIB, parent_mesh.GetComm());
#endif

   if (Mpi::Root())
   {
      mfem::out << "done." << std::endl;
   }

   // Define function to compute gradient on source mesh (fluid)
   int vdim = fes_solid_grad.GetVDim();

   auto fluid_heatFlux_func = [&](ElementTransformation &Tr,
                                  const IntegrationPoint &ip, Vector &qoi_loc)
   {
      real_t conductivity_val = fluid_conductivity.Eval(Tr, ip);
      temperature_fluid_gf.GetGradient(Tr, qoi_loc);
      qoi_loc *= conductivity_val;
   };

   auto fluid_heatFlux_norm_func = [&](ElementTransformation &Tr,
                                       const IntegrationPoint &ip, Vector &qoi_loc)
   {
      // Compute normal n
      Vector nor(sdim);
      Vector grad(sdim); grad = 0.0;

      // Compute the unit normal vector
      CalcOrtho(Tr.Jacobian(), nor);
      const real_t scale = nor.Norml2();
      nor /= scale;

      // Get the conductivity value and gradient at the integration point
      real_t conductivity_val = fluid_conductivity.Eval(Tr, ip);
      temperature_fluid_gf.GetGradient(Tr, grad);
      grad *= conductivity_val;

      // Compute the dot product with normal
      qoi_loc = grad * nor;
   };


   if (Mpi::Root())
   {
      mfem::out << "Setting up DataCollection...";
   }

   // Setup ouput
   ParaViewDataCollection paraview_dc_solid("Solid", solid_submesh.get());
   ParaViewDataCollection paraview_dc_fluid("Fluid", fluid_submesh.get());
   if (paraview)
   {
      paraview_dc_fluid.SetPrefixPath(outfolder);
      paraview_dc_fluid.SetDataFormat(VTKFormat::BINARY);
      if (order_src > 1)
      {
         paraview_dc_fluid.SetHighOrderOutput(true);
         paraview_dc_fluid.SetLevelsOfDetail(order_src);
      }
      paraview_dc_fluid.RegisterField("Temperature", &temperature_fluid_gf);
      paraview_dc_fluid.RegisterField("Gradient-exact", &heatFlux_fluid_exact_gf);

      paraview_dc_solid.SetPrefixPath(outfolder);
      paraview_dc_solid.SetDataFormat(VTKFormat::BINARY);
      if (order_dst > 1)
      {
         paraview_dc_solid.SetHighOrderOutput(true);
         paraview_dc_solid.SetLevelsOfDetail(order_dst);
      }

      paraview_dc_solid.RegisterField("Temperature-Submesh", &temperature_solid_gf);
      paraview_dc_solid.RegisterField("Temperature-GSLIB",
                                      &temperature_solid_gf_copy);
      paraview_dc_solid.RegisterField("Gradient", &heatFlux_solid_gf);
      paraview_dc_solid.RegisterField("Gradiente-Exact-Transfered",
                                      &heatFlux_exact_transfered_gf);
      paraview_dc_solid.RegisterField("Gradient-Normal", &heatFlux_norm_solid_gf);
   }

   if (Mpi::Root())
   {
      mfem::out << "done." << std::endl;
   }

   ///////////////////////////////////////////////////////////////////////////////////////////////
   /// 6. Transfer
   ///////////////////////////////////////////////////////////////////////////////////////////////

   MPI_Barrier(parent_mesh.GetComm());

   Array<int> fs_solid_element_idx;
   finder_fluid_to_solid_hybrid ? finder_fluid_to_solid_hybrid->GetElementIdx(
      fs_solid_element_idx)
   : finder_fluid_to_solid_gslib->GetElementIdx(fs_solid_element_idx);

#ifdef MFEM_USE_GSLIB
   Vector bdr_element_coords = finder_fluid_to_solid_gslib->GetBdrElementCoords();
   int npts_solid = bdr_element_coords.Size() / sdim;
#endif

   // Define Temperature analytic field in the fluid domain
   // f(x,y) = x^2
   auto func2 = [](const Vector &x)
   {
      const int dim = x.Size();
      double f = 0.0;
      f = x(0) * x(0);
      return f;
   };

   // Define Gradient of Temperature analytic field
   // ∇f(x,y) = [2*x, 0]
   auto func_grad2 = [](const Vector &x, Vector &p)
   {
      const int dim = x.Size();
      p.SetSize(dim);
      p = 0.0;
      p(0) = 2.0 * x(0);
      p(1) = 0.0;
   };

   // Define Temperature analytic field in the fluid domain
   // f(x,y) = y + y^2
   auto func1 = [](const Vector &x)
   {
      const int dim = x.Size();
      double f = 0.0;
      f = 3.0 - 2*x(1) - x(1) * x(1);
      return f;
   };

   // Define Gradient of Temperature analytic field
   // ∇f(x,y) = [0, 1 + 2*y]
   auto func_grad1 = [](const Vector &x, Vector &p)
   {
      const int dim = x.Size();
      p.SetSize(dim);
      p = 0.0;
      p(0) = 0.0;
      p(1) = -2.0 - 2.0 * x(1);
   };

   if (Mpi::Root())
   {
      mfem::out << "\nTransfer solution from fluid --> solid domain... " <<
                std::endl;
   }

   // Initial conditions on fluid
   FunctionCoefficient *Tfunc = nullptr;
   VectorFunctionCoefficient *grad_Tfunc = nullptr;

   switch (function)
   {
      case 0: // Normal to interface
         Tfunc = new FunctionCoefficient(func1);
         grad_Tfunc = new VectorFunctionCoefficient(sdim, func_grad1);
         break;
      case 1: // Tangent to interface
         Tfunc = new FunctionCoefficient(func2);
         grad_Tfunc = new VectorFunctionCoefficient(sdim, func_grad2);
         break;
      default:
         MFEM_ABORT("Invalid function type selected.");
   }

   ConstantCoefficient zero_coeff(0.0);
   temperature_solid_gf.ProjectCoefficient(zero_coeff);

   temperature_fluid_gf.ProjectCoefficient(*Tfunc);
   heatFlux_fluid_exact_gf.ProjectCoefficient(*grad_Tfunc);

   // Export before transfer
   if (paraview)
   {
      paraview_dc_solid.SetCycle(0);
      paraview_dc_solid.SetTime(0.0);
      paraview_dc_solid.Save();
      paraview_dc_fluid.SetCycle(0);
      paraview_dc_fluid.SetTime(0.0);
      paraview_dc_fluid.Save();
   }

   StopWatch chrono;
   real_t t_native, t_gslib, t_gslib_grad, t_gslib_grad_norm;

   // Transfer with SubMesh Transfer Map

   if (enable_native_backend)
   {
      if (Mpi::Root())
      {
         mfem::out << "Using SubMesh Transfer Map...";
      }

      chrono.Start();
      finder_fluid_to_solid_hybrid->Interpolate(temperature_fluid_gf,
                                                temperature_solid_gf);
      chrono.Stop();
      t_native = chrono.RealTime();

      if (Mpi::Root())
      {
         mfem::out << "done." << std::endl;
      }
   }

#ifdef MFEM_USE_GSLIB
   // Transfer with GSLIB
   if (Mpi::Root())
   {
      mfem::out << "Using GSLIB...";
   }

   chrono.Clear();
   chrono.Start();
   finder_fluid_to_solid_gslib->Interpolate(temperature_fluid_gf,
                                            temperature_solid_gf_copy);
   chrono.Stop();
   t_gslib = chrono.RealTime();

   finder_fluid_to_solid_gslib->Interpolate(heatFlux_fluid_exact_gf,
                                            heatFlux_exact_transfered_gf);

   if (Mpi::Root())
      mfem::out
            << "done." << std::endl;

   // Transfer gradient with GSLIB
   if (Mpi::Root())
   {
      mfem::out << "Using GSLIB for heat flux...";
   }

   chrono.Clear();
   chrono.Start();
   finder_fluid_to_solid_gslib->InterpolateQoI(fluid_heatFlux_func,
                                               heatFlux_solid_gf);
   chrono.Stop();
   t_gslib_grad = chrono.RealTime();

   if (Mpi::Root())
   {
      mfem::out << "done." << std::endl;
   }

   // Transfer normal gradient with GSLIB

   if (Mpi::Root())
   {
      mfem::out << "Using GSLIB for normal heat flux...";
   }

   chrono.Clear();
   chrono.Start();
   finder_fluid_to_solid_gslib->InterpolateQoI(fluid_heatFlux_norm_func,
                                               heatFlux_norm_solid_gf, true);
   chrono.Stop();
   t_gslib_grad_norm = chrono.RealTime();

   if (Mpi::Root())
   {
      mfem::out << "done." << std::endl;
   }
#endif // MFEM_USE_GSLIB

   // Accuracy check: max nodal error on the interface dofs of the destination (solid) w.r.t. the exact field
   {
      Array<int> iface_marker(solid_submesh->bdr_attributes.Max());
      iface_marker = 0;
      for (int a : fluid_solid_interface) { if (a <= iface_marker.Size()) { iface_marker[a - 1] = 1; } }
      auto iface_err = [&](ParGridFunction &gf, Coefficient *sc,
                           VectorCoefficient *vc) -> real_t
      {
         ParGridFunction ex(gf.ParFESpace());
         ex = 0.0;
         if (sc) { ex.ProjectBdrCoefficient(*sc, iface_marker); }
         else { ex.ProjectBdrCoefficient(*vc, iface_marker); }
         Array<int> tdofs;
         gf.ParFESpace()->GetEssentialTrueDofs(iface_marker, tdofs);
         Vector a, b;
         gf.GetTrueDofs(a);
         ex.GetTrueDofs(b);
         real_t err = 0.0;
         for (int i : tdofs) { err = std::max(err, std::abs(a(i) - b(i))); }
         MPI_Allreduce(MPI_IN_PLACE, &err, 1, MFEM_MPI_REAL_T, MPI_MAX, parent_mesh.GetComm());
         return err;
      };
      // -1: not run (Native needs -os == -od; GSLIB needs MFEM_USE_GSLIB)
      real_t e_nat = -1.0, e_nat_vec = -1.0, e_nat_qoi_el = -1.0,
             e_nat_qoi_bdr = -1.0;
      real_t e_gs = -1.0, e_gs_vec = -1.0, e_gs_qoi_bdr = -1.0;
      if (enable_native_backend)
      {
         e_nat = iface_err(temperature_solid_gf, Tfunc, nullptr);
         InterfaceTransfer native_vec(&fes_fluid_grad, &fes_solid_grad,
                                      fluid_solid_interface_marker,
                                      TransferBackend::Native, parent_mesh.GetComm());
         ParGridFunction nat_vec_gf(&fes_solid_grad);
         nat_vec_gf = 0.0;
         native_vec.Interpolate(heatFlux_fluid_exact_gf, nat_vec_gf);
         e_nat_vec = iface_err(nat_vec_gf, nullptr, grad_Tfunc);
         // Vector QoI, element version (adjacent element transformation) and boundary version (InterpolateQoIv2)
         nat_vec_gf = 0.0;
         native_vec.InterpolateQoI(fluid_heatFlux_func, nat_vec_gf, false);
         e_nat_qoi_el = iface_err(nat_vec_gf, nullptr, grad_Tfunc);
         nat_vec_gf = 0.0;
         native_vec.InterpolateQoI(fluid_heatFlux_func, nat_vec_gf, true);
         e_nat_qoi_bdr = iface_err(nat_vec_gf, nullptr, grad_Tfunc);
      }
#ifdef MFEM_USE_GSLIB
      e_gs = iface_err(temperature_solid_gf_copy, Tfunc, nullptr);
      e_gs_vec = iface_err(heatFlux_exact_transfered_gf, nullptr, grad_Tfunc);
      // Vector QoI through the boundary version (InterpolateQoIv2): exercises the vdof layout for vdim > 1
      ParGridFunction qoi_v2_gf(&fes_solid_grad);
      qoi_v2_gf = 0.0;
      InterfaceTransfer gslib_vec(&fes_fluid_grad, &fes_solid_grad,
                                  fluid_solid_interface_marker,
                                  TransferBackend::GSLIB, parent_mesh.GetComm());
      gslib_vec.InterpolateQoI(fluid_heatFlux_func, qoi_v2_gf, true);
      e_gs_qoi_bdr = iface_err(qoi_v2_gf, nullptr, grad_Tfunc);
#endif
      if (Mpi::Root())
      {
         mfem::out << std::scientific << "Interface max error (Native): scalar " << e_nat
                   << " | vector " << e_nat_vec
                   << " | vector QoI (element) " << e_nat_qoi_el << " | vector QoI (bdr) " <<
                   e_nat_qoi_bdr << std::endl;
         mfem::out << std::scientific << "Interface max error (GSLIB):  scalar " << e_gs
                   << " | vector " << e_gs_vec
                   << " | vector QoI (bdr) " << e_gs_qoi_bdr << std::endl;
      }
   }

   // Export after transfer
   if (paraview)
   {
      paraview_dc_solid.SetCycle(1);
      paraview_dc_solid.SetTime(0.1);
      paraview_dc_solid.Save();
      paraview_dc_fluid.SetCycle(1);
      paraview_dc_fluid.SetTime(0.1);
      paraview_dc_fluid.Save();
   }

   if (Mpi::Root())
   {
      if (enable_native_backend)
      {
         mfem::out << "Transfer time (Native): " << t_native << " s" << std::endl;
      }

#ifdef MFEM_USE_GSLIB
      mfem::out << "Transfer time (GSLIB): " << t_gslib << " s" << std::endl;
      mfem::out << "Transfer time (GSLIB Gradient): " << t_gslib_grad << " s" <<
                std::endl;
      mfem::out << "Transfer time (GSLIB Normal Gradient): " << t_gslib_grad_norm <<
                " s" << std::endl;
#endif
   }

   ///////////////////////////////////////////////////////////////////////////////////////////////
   /// 8. Cleanup
   ///////////////////////////////////////////////////////////////////////////////////////////////

   delete Tfunc;
   delete grad_Tfunc;

   delete finder_fluid_to_solid_hybrid;
   delete finder_fluid_to_solid_gslib;

   return 0;
}