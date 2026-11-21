// =====================================================================================
//          3D Interface Transfer Example using InterfaceTransfer Class
// -------------------------------------------------------------------------------------
// This example demonstrates the transfer of field data and QoI (e.g., temperature and
// heat flux) across an interface on a 2D mesh.
// It sets up a parallel mesh with cylinder and box subdomains, projects analytic fields,
// and performs data transfer between subdomains using both a native SubMesh transfer
// map and GSLIB interpolation. The results are exported for visualization in ParaView.
//
// The mesh used is the same as in the multidomain miniapp (cylinder inside a box).
//
// We define an analytical temperature field on the cylinder domain and transfer
// temperature and heat flux (k ∇T) from cylinder --> box domain.
//
// Requirements:
//   - MFEM built with GSLIB support
//   - Input mesh files: multidomain-hex.mesh
//
// Sample runs:
//
//   1. Same order for source and destination (gradient normal to interface):
//      mpirun -np 2 ./transfer3d -rs 1 -os 2 -od 2 -of ./Output/Transfer3D
//
//   2. Same order for source and destination (gradient tangent to interface):
//      mpirun -np 2 ./transfer3d -rs 1 -os 2 -od 2 -f 1 -of ./Output/Transfer3D
//
//   3. Different orders for source and destination (gradient normal to interface):
//      mpirun -np 2 ./transfer3d -rs 1 -os 2 -od 4 -of ./Output/Transfer3D
//
//   4. Same order for source and destination, but src is byNODES and dst is byVDIM:
//      mpirun -np 2 ./transfer3d -rs 1 -os 2 -od 2 --by-nodes-src --by-vdim-dst -f 1 -of ./Output/Transfer3D
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

   /* { // Uncomment this block to ease MPI debuggings
      int i = 0;
      while (0 == i)
         sleep(5);
   }  */

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
   int fun = 0; // 0: normal to the interface, 1: tangential to the interface
   // FE
   int order_src = 1; // Source mesh polynomial order
   int order_dst = 1; // Destination mesh polynomial order
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
   args.AddOption(&fun, "-f", "--function",
                  "Function to transfer: 0 - normal to the interface, "
                  "1 - tangential to the interface.");
   // FE
   args.AddOption(&order_src, "-os", "--order-src",
                  "Source finite element order (polynomial degree).");
   args.AddOption(&order_dst, "-od", "--order-dst",
                  "Destination finite element order (polynomial degree).");
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

   // Enable native backend if the order is the same for both source and destination meshes
   bool enable_native_backend = false;
   enable_native_backend = (order_src == order_dst);
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
   Mesh *serial_mesh = new Mesh("../multidomain/multidomain-hex.mesh");
   int sdim = serial_mesh->SpaceDimension();
   serial_mesh->EnsureNodes();

   for (int l = 0; l < serial_ref_levels; l++)
   {
      serial_mesh->UniformRefinement();
   }

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

   // Create the sub-domains for the box, box and cylinder domains
   Array<int> cylinder_domain_attribute(1);
   Array<int> box_domain_attribute(1);
   cylinder_domain_attribute[0] = 1;
   box_domain_attribute[0] = 2;

   auto box_submesh =
      std::make_shared<ParSubMesh>(ParSubMesh::CreateFromDomain(parent_mesh,
                                                                box_domain_attribute));

   auto cylinder_submesh =
      std::make_shared<ParSubMesh>(ParSubMesh::CreateFromDomain(parent_mesh,
                                                                cylinder_domain_attribute));

   // Define interface markers
   Array<int> box_cylinder_interface_attr(1);
   box_cylinder_interface_attr[0] = 9;
   Array<int> box_cylinder_interface_marker = AttributeSets::AttrToMarker(
                                                 box_submesh->bdr_attributes.Max(), box_cylinder_interface_attr);

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

   // BOX is destination, CYLINDER is source
   ParFiniteElementSpace fes_box(box_submesh.get(), &fec_dst, 1, ordering_dst);
   ParFiniteElementSpace fes_box_grad(box_submesh.get(), &fec_dst, sdim,
                                      ordering_dst);

   ParFiniteElementSpace fes_cylinder(cylinder_submesh.get(), &fec_src, 1,
                                      ordering_src);
   ParFiniteElementSpace fes_cylinder_grad(cylinder_submesh.get(), &fec_src, sdim,
                                           ordering_src);

   ParGridFunction temperature_box_gf(&fes_box);
   temperature_box_gf = 0.0;
   ParGridFunction heatFlux_box_normal_gf(&fes_box);
   heatFlux_box_normal_gf = 0.0;
   ParGridFunction heatFlux_box_gf(&fes_box_grad);
   heatFlux_box_gf = 0.0;
   ParGridFunction heatFlux_exact_transfered_gf(&fes_box_grad);
   heatFlux_exact_transfered_gf = 0.0;
   ParGridFunction temperature_box_gf_copy(&fes_box);
   temperature_box_gf_copy = 0.0;

   ParGridFunction temperature_cylinder_gf(&fes_cylinder);
   temperature_cylinder_gf = 0.0;
   ParGridFunction heatFlux_cylinder_exact_gf(&fes_cylinder_grad);
   heatFlux_cylinder_exact_gf = 0.0;

   // Create Coefficients for conductivity
   ConstantCoefficient cylinder_conductivity(1.0);
   ConstantCoefficient box_conductivity(1.0);

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

   InterfaceTransfer *finder_cylinder_to_box_hybrid = nullptr;
   InterfaceTransfer *finder_cylinder_to_box_gslib = nullptr;

   if (enable_native_backend)
#ifdef MFEM_USE_GSLIB
      finder_cylinder_to_box_hybrid = new InterfaceTransfer(&fes_cylinder, &fes_box,
                                                            box_cylinder_interface_marker, TransferBackend::Hybrid, parent_mesh.GetComm());
#else
      finder_cylinder_to_box_hybrid = new InterfaceTransfer(&fes_cylinder, &fes_box,
                                                            box_cylinder_interface_marker, TransferBackend::Native, parent_mesh.GetComm());
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
   finder_cylinder_to_box_gslib = new InterfaceTransfer(&fes_cylinder, &fes_box,
                                                        box_cylinder_interface_marker, TransferBackend::GSLIB, parent_mesh.GetComm());
#endif

   if (Mpi::Root())
   {
      mfem::out << "done." << std::endl;
   }

   // Define function to compute gradient on source mesh (cylinder)
   int vdim = fes_box_grad.GetVDim();

   // Vector qoi_src, qoi_dst; // QoI vector, used to store qoi_src in func and in call to GSLIB interpolator
   auto cylinder_heatFlux_func = [&](ElementTransformation &Tr,
                                     const IntegrationPoint &ip, Vector &qoi_loc)
   {
      real_t conductivity_val = cylinder_conductivity.Eval(Tr, ip);
      temperature_cylinder_gf.GetGradient(Tr, qoi_loc);
      qoi_loc *= conductivity_val;

      // In case conductivity is a tensor, we would need to do something like:
      // DenseMatrix conductivity_tensor;
      // Vector grad_temp(vdim);
      // cylinder_conductivity.Eval(conductivity_tensor, Tr, ip);
      // temperature_cylinder_gf.GetGradient(Tr, grad_temp);
      // conductivity_tensor.Mult(grad_temp, qoi_loc);
   };

   auto cylinder_heatFlux_norm_func = [&](ElementTransformation &Tr,
                                          const IntegrationPoint &ip, Vector &qoi_loc)
   {
      // Compute normal n
      Vector nor(sdim);
      Vector grad(sdim); grad = 0.0;

      // Compute the unit normal vector
      CalcOrtho(Tr.Jacobian(), nor);
      const real_t scale = nor.Norml2();
      nor /= scale;

      // Compute the gradient of temperature
      real_t conductivity_val = cylinder_conductivity.Eval(Tr, ip);
      temperature_cylinder_gf.GetGradient(Tr, grad);
      grad *= conductivity_val;
      // Compute the normal component of the gradient
      qoi_loc = grad * nor;
   };

   if (Mpi::Root())
   {
      mfem::out << "Setting up DataCollection...";
   }

   // Setup ouput
   ParaViewDataCollection paraview_dc_box("Solid", box_submesh.get());
   ParaViewDataCollection paraview_dc_cylinder("Cylinder", cylinder_submesh.get());
   if (paraview)
   {
      paraview_dc_cylinder.SetPrefixPath(outfolder);
      paraview_dc_cylinder.SetDataFormat(VTKFormat::BINARY);
      if (order_src > 1)
      {
         paraview_dc_cylinder.SetHighOrderOutput(true);
         paraview_dc_cylinder.SetLevelsOfDetail(order_src);
      }
      paraview_dc_cylinder.RegisterField("Temperature", &temperature_cylinder_gf);
      paraview_dc_cylinder.RegisterField("Gradient-exact",
                                         &heatFlux_cylinder_exact_gf);

      paraview_dc_box.SetPrefixPath(outfolder);
      paraview_dc_box.SetDataFormat(VTKFormat::BINARY);
      if (order_dst > 1)
      {
         paraview_dc_box.SetHighOrderOutput(true);
         paraview_dc_box.SetLevelsOfDetail(order_dst);
      }

      paraview_dc_box.RegisterField("Temperature-Submesh", &temperature_box_gf);
      paraview_dc_box.RegisterField("Temperature-GSLIB", &temperature_box_gf_copy);
      paraview_dc_box.RegisterField("Gradient", &heatFlux_box_gf);
      paraview_dc_box.RegisterField("Gradient-Exact-Transferred",
                                    &heatFlux_exact_transfered_gf);
      paraview_dc_box.RegisterField("Gradient-Normal", &heatFlux_box_normal_gf);
   }

   if (Mpi::Root())
   {
      mfem::out << "done." << std::endl;
   }

   ///////////////////////////////////////////////////////////////////////////////////////////////
   /// 6. Transfer
   ///////////////////////////////////////////////////////////////////////////////////////////////

   MPI_Barrier(parent_mesh.GetComm());

   Array<int> fs_box_element_idx;
   finder_cylinder_to_box_hybrid ? finder_cylinder_to_box_hybrid->GetElementIdx(
      fs_box_element_idx)
   : finder_cylinder_to_box_gslib->GetElementIdx(fs_box_element_idx);

#ifdef MFEM_USE_GSLIB
   Vector bdr_element_coords = finder_cylinder_to_box_gslib->GetBdrElementCoords();
   int npts_box = bdr_element_coords.Size() / sdim;
#endif

   // Define Temperature analytic field in the cylinder domain
   // f(x,y) = x^2 + y^2
   auto func1 = [](const Vector &x)
   {
      const int dim = x.Size();
      double f = 0.0;
      f = x(0) * x(0) + x(1) * x(1);
      return f;
   };

   // Define Gradient of Temperature analytic field
   // ∇f(x,y) = [2*x, 2*y]
   auto func_grad1 = [](const Vector &x, Vector &p)
   {
      const int dim = x.Size();
      p.SetSize(dim);
      p = 0.0;
      p(0) = 2.0 * x(0);
      p(1) = 2.0 * x(1);
   };


   // Define Temperature analytic field in the cylinder domain
   // f(z) = z^2
   auto func2 = [](const Vector &x)
   {
      return x(2) * x(2);
   };

   // Define Gradient of Temperature analytic field
   // ∇f(z) = [0, 0, 2*z]
   auto func_grad2 = [](const Vector &x, Vector &p)
   {
      const int dim = x.Size();
      p.SetSize(dim);
      p = 0.0;
      p(2) = 2.0 * x(2);
   };


   if (Mpi::Root())
   {
      mfem::out << "\nTransfer solution from cylinder --> box domain... " <<
                std::endl;
   }

   // Initial conditions on cylinder

   FunctionCoefficient *Tfunc = nullptr;
   VectorFunctionCoefficient *grad_Tfunc = nullptr;

   switch (fun)
   {
      case 0: // Normal to the interface
         Tfunc = new FunctionCoefficient(func1);
         grad_Tfunc = new VectorFunctionCoefficient(3, func_grad1);
         break;
      case 1: // Tangential to the interface
         Tfunc = new FunctionCoefficient(func2);
         grad_Tfunc = new VectorFunctionCoefficient(3, func_grad2);
         break;
      default:
         if (Mpi::Root())
         {
            mfem::out << "Invalid function choice. Exiting." << std::endl;
         }
         return 1;
   }

   ConstantCoefficient zero_coeff(0.0);
   temperature_box_gf.ProjectCoefficient(zero_coeff);

   temperature_cylinder_gf.ProjectCoefficient(*Tfunc);
   heatFlux_cylinder_exact_gf.ProjectCoefficient(*grad_Tfunc);

   // Export before transfer
   if (paraview)
   {
      paraview_dc_box.SetCycle(0);
      paraview_dc_box.SetTime(0.0);
      paraview_dc_box.Save();
      paraview_dc_cylinder.SetCycle(0);
      paraview_dc_cylinder.SetTime(0.0);
      paraview_dc_cylinder.Save();
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
      finder_cylinder_to_box_hybrid->Interpolate(temperature_cylinder_gf,
                                                 temperature_box_gf);
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
   finder_cylinder_to_box_gslib->Interpolate(temperature_cylinder_gf,
                                             temperature_box_gf_copy);
   chrono.Stop();
   t_gslib = chrono.RealTime();

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
   finder_cylinder_to_box_gslib->InterpolateQoI(cylinder_heatFlux_func,
                                                heatFlux_box_gf);
   chrono.Stop();
   t_gslib_grad = chrono.RealTime();

   finder_cylinder_to_box_gslib->Interpolate(heatFlux_cylinder_exact_gf,
                                             heatFlux_exact_transfered_gf);


   if (Mpi::Root())
   {
      mfem::out << "done." << std::endl;
   }


   // Transfer normal component of heat flux with GSLIB
   if (Mpi::Root())
   {
      mfem::out << "Using GSLIB for normal heat flux...";
   }

   chrono.Clear();
   chrono.Start();
   finder_cylinder_to_box_gslib->InterpolateQoI(cylinder_heatFlux_norm_func,
                                                heatFlux_box_normal_gf, true);
   chrono.Stop();
   t_gslib_grad_norm = chrono.RealTime();

   if (Mpi::Root())
   {
      mfem::out << "done." << std::endl;
   }
#endif // MFEM_USE_GSLIB

   // Max nodal error on the interface against the exact field (true dofs on
   // the box side of the interface), as in transfer2d.
   auto iface_err = [&](ParGridFunction &gf, Coefficient *sc,
                        VectorCoefficient *vc)
   {
      ParGridFunction ex(gf.ParFESpace());
      ex = 0.0;
      if (sc) { ex.ProjectBdrCoefficient(*sc, box_cylinder_interface_marker); }
      else { ex.ProjectBdrCoefficient(*vc, box_cylinder_interface_marker); }
      Array<int> tdofs;
      gf.ParFESpace()->GetEssentialTrueDofs(box_cylinder_interface_marker, tdofs);
      Vector a, b;
      gf.GetTrueDofs(a);
      ex.GetTrueDofs(b);
      real_t err = 0.0;
      for (int i : tdofs) { err = std::max(err, std::abs(a(i) - b(i))); }
      MPI_Allreduce(MPI_IN_PLACE, &err, 1, MFEM_MPI_REAL_T, MPI_MAX,
                    parent_mesh.GetComm());
      return err;
   };
   // -1: not run (Native needs -os == -od; GSLIB needs MFEM_USE_GSLIB)
   real_t e_nat = -1.0, e_gs = -1.0, e_gs_vec = -1.0, e_gs_qoi = -1.0;
   if (enable_native_backend) { e_nat = iface_err(temperature_box_gf, Tfunc, nullptr); }
#ifdef MFEM_USE_GSLIB
   e_gs = iface_err(temperature_box_gf_copy, Tfunc, nullptr);
   e_gs_vec = iface_err(heatFlux_exact_transfered_gf, nullptr, grad_Tfunc);
   e_gs_qoi = iface_err(heatFlux_box_gf, nullptr, grad_Tfunc);
#endif
   if (Mpi::Root())
   {
      mfem::out << std::scientific << "Interface max error: Native scalar " << e_nat
                << " | GSLIB scalar " << e_gs << " | GSLIB vector " << e_gs_vec
                << " | GSLIB vector QoI " << e_gs_qoi << std::endl;
   }

   // Export after transfer
   if (paraview)
   {
      paraview_dc_box.SetCycle(1);
      paraview_dc_box.SetTime(0.1);
      paraview_dc_box.Save();
      paraview_dc_cylinder.SetCycle(1);
      paraview_dc_cylinder.SetTime(0.1);
      paraview_dc_cylinder.Save();
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
      mfem::out << "Transfer time (GSLIB Gradient Normal): " << t_gslib_grad_norm <<
                " s" << std::endl;
#endif
   }

   ///////////////////////////////////////////////////////////////////////////////////////////////
   /// 8. Cleanup
   ///////////////////////////////////////////////////////////////////////////////////////////////

   delete Tfunc;
   delete grad_Tfunc;

   delete finder_cylinder_to_box_hybrid;
   delete finder_cylinder_to_box_gslib;

   return 0;
}