// Copyright (c) 2010-2023, Lawrence Livermore National Security, LLC. Produced
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
//    ----------------------------------------------------------------------
//    Parallel Find Points Miniapp: Evaluate grid function in physical space
//    ----------------------------------------------------------------------
//
// This miniapp demonstrates the interpolation of a high-order grid function on
// a set of points in physical-space. The miniapp is based on GSLIB-FindPoints,
// which provides two key functionalities. First, for a given set of points in
// the physical-space, it determines the computational coordinates (element
// number, reference-space coordinates inside the element, and processor number
// [in parallel]) for each point. Second, based on computational coordinates, it
// interpolates a grid function in the given points. Inside GSLIB, computation
// of the coordinates requires use of a Hash Table to identify the candidate
// processor and element for each point, followed by the Newton's method to
// determine the reference-space coordinates inside the candidate element.
//
// Compile with: make pfindpts
//
// Sample runs:
//    mpirun -np 2 pfindpts -m ../../data/rt-2d-p4-tri.mesh -o 4
//    mpirun -np 2 pfindpts -m ../../data/inline-tri.mesh -o 3
//    mpirun -np 2 pfindpts -m ../../data/inline-quad.mesh -o 3
//    mpirun -np 2 pfindpts -m ../../data/inline-quad.mesh -o 3 -po 1
//    mpirun -np 2 pfindpts -m ../../data/inline-quad.mesh -o 3 -po 1 -gfo 1 -nc 2
//    mpirun -np 2 pfindpts -m ../../data/inline-quad.mesh -o 3 -hr
//    mpirun -np 2 pfindpts -m ../../data/inline-tet.mesh -o 3
//    mpirun -np 2 pfindpts -m ../../data/inline-hex.mesh -o 3
//    mpirun -np 2 pfindpts -m ../../data/inline-wedge.mesh -o 3
//    mpirun -np 2 pfindpts -m ../../data/amr-quad.mesh -o 2
//    mpirun -np 2 pfindpts -m ../../data/rt-2d-q3.mesh -o 3 -mo 4 -ft 2
//    mpirun -np 2 pfindpts -m ../../data/inline-quad.mesh -ft 1 -no-vis -sr0
//    mpirun -np 2 pfindpts -m ../../data/square-mixed.mesh -o 2 -mo 2
//    mpirun -np 2 pfindpts -m ../../data/square-mixed.mesh -o 2 -mo 2 -hr
//    mpirun -np 2 pfindpts -m ../../data/square-mixed.mesh -o 2 -mo 3 -ft 2
//    mpirun -np 2 pfindpts -m ../../data/fichera-mixed.mesh -o 3 -mo 2
//    mpirun -np 2 pfindpts -m ../../data/inline-pyramid.mesh -o 1 -mo 1
//    mpirun -np 2 pfindpts -m ../../data/tinyzoo-3d.mesh -o 1 -mo 1

// make pfindpts -j && mpirun -np 5 pfindpts -m ../../data/inline-hex.mesh -d debug -rs 2 -mo 3 -o 3 -ji 0.01 -nc 2 -po 1 -gfo 1 -eo 1
#include "mfem.hpp"
#include "general/forall.hpp"
#include "../common/mfem-common.hpp"

using namespace mfem;
using namespace std;

double func_order;

// Scalar function to project
double field_func(const Vector &x)
{
   const int dim = x.Size();
   double res = 0.0;
   for (int d = 0; d < dim; d++) { res += std::pow(x(d), func_order); }
   return res;
}

void F_exact(const Vector &p, Vector &F)
{
   F(0) = field_func(p);
   for (int i = 1; i < F.Size(); i++) { F(i) = (i+1)*F(0); }
}

int main (int argc, char *argv[])
{
   // Initialize MPI and HYPRE.
   Mpi::Init(argc, argv);
   int num_procs = Mpi::WorldSize();
   int myid = Mpi::WorldRank();
   Hypre::Init();

   // Set the method's default parameters.
   const char *mesh_file = "../../data/rt-2d-q3.mesh";
   int order             = 3;
   int mesh_poly_deg     = 3;
   int rs_levels         = 0;
   int rp_levels         = 0;
   bool visualization    = false;
   int fieldtype         = 0;
   int ncomp             = 1;
   bool search_on_rank_0 = false;
   bool hrefinement      = false;
   int point_ordering    = 0;
   int gf_ordering       = 0;
   const char *devopt    = "cpu";
   double jitter         = 0.0;
   int exact_sol_order   = 1;
   int smooth            = 0; //kershaw transformation parameter
   int jobid             = 0;
   int npt               = 100; //points per proc
   int nx                = 6; //points per proc

   // Parse command-line options.
   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree).");
   args.AddOption(&mesh_poly_deg, "-mo", "--mesh-order",
                  "Polynomial degree of mesh finite element space.");
   args.AddOption(&rs_levels, "-rs", "--refine-serial",
                  "Number of times to refine the mesh uniformly in serial.");
   args.AddOption(&rp_levels, "-rp", "--refine-parallel",
                  "Number of times to refine the mesh uniformly in parallel.");
   args.AddOption(&fieldtype, "-ft", "--field-type",
                  "Field type: 0 - H1, 1 - L2, 2 - H(div), 3 - H(curl).");
   args.AddOption(&ncomp, "-nc", "--ncomp",
                  "Number of components for H1 or L2 GridFunctions");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&search_on_rank_0, "-sr0", "--search-on-r0", "-no-sr0",
                  "--no-search-on-r0",
                  "Enable search only on rank 0 (disable to search points on all tasks).");
   args.AddOption(&hrefinement, "-hr", "--h-refinement", "-no-hr",
                  "--no-h-refinement",
                  "Do random h refinements to mesh (does not work for pyramids).");
   args.AddOption(&point_ordering, "-po", "--point-ordering",
                  "Ordering of points to be found."
                  "0 (default): byNodes, 1: byVDIM");
   args.AddOption(&gf_ordering, "-gfo", "--gridfunc-ordering",
                  "Ordering of fespace that will be used for gridfunction to be interpolated."
                  "0 (default): byNodes, 1: byVDIM");
   args.AddOption(&devopt, "-d", "--device",
                  "Device configuration string, see Device::Configure().");
   args.AddOption(&jitter, "-ji", "--jitter",
                  "Random perturbation scaling factor.");
   args.AddOption(&exact_sol_order, "-eo", "--exact-sol-order",
                  "Order for analytic solution.");
   args.AddOption(&smooth, "-smooth", "--smooth",
                  "smooth parameter of kershaw");
   args.AddOption(&jobid, "-jid", "--jid",
                  "job id used for visit  save files");
   args.AddOption(&npt, "-npt", "--npt",
                  "# points per proc");
   args.AddOption(&nx, "-nx", "--nx",
                  "# of elements in x(is multipled by rs)");
   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(cout);
      return 1;
   }
   if (myid == 0) { args.PrintOptions(cout); }
   bool cpu_mode = strcmp(devopt,"cpu")==0;

   if (hrefinement)
   {
      MFEM_VERIFY(strcmp(devopt,"cpu")==0, "HR-adaptivity is currently only"
                  " supported on cpus.");
   }
   Device device(devopt);
   if (myid == 0) { device.Print();}

   func_order = std::min(exact_sol_order, 2);

   // Initialize and refine the starting mesh.
   //    Mesh *mesh = new Mesh(mesh_file, 1, 1, false);
   int nex = nx*std::pow(2, rs_levels);
   const int etype = 0;
   Mesh *mesh = new Mesh(Mesh::MakeCartesian3D(nex, nex, nex, etype == 0 ?
                                               Element::HEXAHEDRON :
                                               Element::TETRAHEDRON));

   //    for (int lev = 0; lev < rs_levels; lev++) { mesh->UniformRefinement(); }
   const int dim = mesh->Dimension();
   if (myid == 0)
   {
      cout << "Mesh curvature of the original mesh: ";
      if (mesh->GetNodes()) { cout << mesh->GetNodes()->OwnFEC()->Name(); }
      else { cout << "(NONE)"; }
      cout << endl;
   }

   // Mesh bounding box (for the full serial mesh).
   Vector pos_min, pos_max;
   MFEM_VERIFY(mesh_poly_deg > 0, "The order of the mesh must be positive.");
   mesh->GetBoundingBox(pos_min, pos_max, mesh_poly_deg);
   if (myid == 0)
   {
      cout << "--- Generating equidistant point for:\n"
           << "x in [" << pos_min(0) << ", " << pos_max(0) << "]\n"
           << "y in [" << pos_min(1) << ", " << pos_max(1) << "]\n";
      if (dim == 3)
      {
         cout << "z in [" << pos_min(2) << ", " << pos_max(2) << "]\n";
      }
   }

   // Distribute the mesh.
   if (hrefinement) { mesh->EnsureNCMesh(); }
   ParMesh pmesh(MPI_COMM_WORLD, *mesh);
   delete mesh;
   for (int lev = 0; lev < rp_levels; lev++) { pmesh.UniformRefinement(); }

   // Random h-refinements to mesh
   if (hrefinement) { pmesh.RandomRefinement(0.5); }

   // Curve the mesh based on the chosen polynomial degree.
   H1_FECollection fecm(mesh_poly_deg, dim);
   ParFiniteElementSpace pfespace(&pmesh, &fecm, dim);
   pmesh.SetNodalFESpace(&pfespace);
   ParGridFunction x(&pfespace);
   pmesh.SetNodalGridFunction(&x);
   if (myid == 0)
   {
      cout << "Mesh curvature of the curved mesh: " << fecm.Name() << endl;
   }

   int nelemglob = pmesh.GetGlobalNE();
   if (myid == 0)
   {
      cout << "Number of elements: " << nelemglob << endl;
   }

   // Kershaw transformation
   if (smooth > 0)
   {
      // 1 leads to a linear transformation, 2 cubic, and 3 5th order.
      common::KershawTransformation kershawT(pmesh.Dimension(), 0.3, 0.3, smooth);
      pmesh.Transform(kershawT);
   }
   if (myid == 0) { cout << "Kershaw transformation done." << endl; }

   Vector h0(pfespace.GetNDofs());
   h0 = infinity();
   double vol_loc = 0.0;
   Array<int> dofs;
   for (int i = 0; i < pmesh.GetNE(); i++)
   {
      // Get the local scalar element degrees of freedom in dofs.
      pfespace.GetElementDofs(i, dofs);
      // Adjust the value of h0 in dofs based on the local mesh size.
      const double hi = pmesh.GetElementSize(i);
      for (int j = 0; j < dofs.Size(); j++)
      {
         h0(dofs[j]) = min(h0(dofs[j]), hi);
      }
      vol_loc += pmesh.GetElementVolume(i);
   }

   ParGridFunction rdm(&pfespace);
   rdm.Randomize(myid+1);
   rdm -= 0.25; // Shift to random values in [-0.5,0.5].
   rdm *= jitter;
   rdm.HostReadWrite();
   // Scale the random values to be of order of the local mesh size.
   for (int i = 0; i < pfespace.GetNDofs(); i++)
   {
      for (int d = 0; d < dim; d++)
      {
         rdm(pfespace.DofToVDof(i,d)) *= h0(i);
      }
   }
   Array<int> vdofs;
   for (int i = 0; i < pfespace.GetNBE(); i++)
   {
      // Get the vector degrees of freedom in the boundary element.
      pfespace.GetBdrElementVDofs(i, vdofs);
      // Set the boundary values to zero.
      for (int j = 0; j < vdofs.Size(); j++) { rdm(vdofs[j]) = 0.0; }
   }
   rdm.SetTrueVector();
   rdm.SetFromTrueVector();
   x -= rdm;
   // Set the perturbation of all nodes from the true nodes.
   x.SetTrueVector();
   x.SetFromTrueVector();

   pmesh.DeleteGeometricFactors();
   double vol = 0;
   for (int e = 0; e < pmesh.GetNE(); e++)
   {
      vol += pmesh.GetElementVolume(e);
   }
   MPI_Allreduce(MPI_IN_PLACE, &vol, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

   MFEM_VERIFY(ncomp > 0, "Invalid number of components.");
   int vec_dim = ncomp;
   FiniteElementCollection *fec = NULL;
   if (fieldtype == 0)
   {
      fec = new H1_FECollection(order, dim);
      if (myid == 0) { cout << "H1-GridFunction\n"; }
   }
   else if (fieldtype == 1)
   {
      fec = new L2_FECollection(order, dim);
      if (myid == 0) { cout << "L2-GridFunction\n"; }
   }
   else if (fieldtype == 2)
   {
      fec = new RT_FECollection(order, dim);
      ncomp = 1;
      vec_dim = dim;
      if (myid == 0) { cout << "H(div)-GridFunction\n"; }
   }
   else if (fieldtype == 3)
   {
      fec = new ND_FECollection(order, dim);
      ncomp = 1;
      vec_dim = dim;
      if (myid == 0) { cout << "H(curl)-GridFunction\n"; }
   }
   else
   {
      if (myid == 0) { MFEM_ABORT("Invalid FECollection type."); }
   }
   ParFiniteElementSpace sc_fes(&pmesh, fec, ncomp, gf_ordering);
   ParGridFunction field_vals(&sc_fes);

   // Project the GridFunction using VectorFunctionCoefficient.
   VectorFunctionCoefficient F(vec_dim, F_exact);
   field_vals.ProjectCoefficient(F);

   // Display the mesh and the field through glvis.
   if (visualization)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      socketstream sout;
      sout.open(vishost, visport);
      if (!sout)
      {
         if (myid == 0)
         {
            cout << "Unable to connect to GLVis server at "
                 << vishost << ':' << visport << endl;
         }
      }
      else
      {
         sout << "parallel " << num_procs << " " << myid << "\n";
         sout.precision(8);
         sout << "solution\n" << pmesh << field_vals;
         if (dim == 2) { sout << "keys RmjA*****\n"; }
         if (dim == 3) { sout << "keys mA\n"; }
         sout << flush;
      }
   }

   // Generate equidistant points in physical coordinates over the whole mesh.
   // Note that some points might be outside, if the mesh is not a box. Note
   // also that all tasks search the same points (not mandatory).
   int pts_cnt = npt;
   Vector vxyz(pts_cnt * dim);
   vxyz.UseDevice(!cpu_mode);
   vxyz.Randomize(myid+1);
   //    vxyz = 0.1;

   if ( (myid != 0) && (search_on_rank_0) )
   {
      pts_cnt = 0;
      vxyz.Destroy();
   }

   // Find and Interpolate FE function values on the desired points.
   Vector interp_vals(pts_cnt*vec_dim);

   FindPointsGSLIB finder(MPI_COMM_WORLD);
   finder.Setup(pmesh);
   finder.FindPoints(vxyz, point_ordering);
   MPI_Barrier(MPI_COMM_WORLD);
   //    std::cout << myid << " K10 done findpoints" << std::endl;

   Array<unsigned int> code_out1    = finder.GetCode();
   Array<unsigned int> el_out1    = finder.GetGSLIBElem();
   Vector ref_rst1    = finder.GetGSLIBReferencePosition();
   Vector ref_rst0   = finder.GetReferencePosition();
   Vector dist1    = finder.GetDist();
   Array<unsigned int> proc_out1    = finder.GetProc();
   vxyz.HostReadWrite();

   int notfound = 0;
   for (int i = 0; i < code_out1.Size(); i++)
   {
      int c1 = code_out1[i];
      int e1 = el_out1[i];
      Vector ref1(ref_rst1.GetData()+i*dim, dim);
      Vector dref = ref1;
      if (c1 == 2 || (std::fabs(dist1(i)) > 1e-10 && myid == 0))
      {
         notfound++;
         if (point_ordering == 0)
         {
            std::cout << "Pt xyz: " << vxyz(i) << " " << vxyz(i + pts_cnt) <<  " " <<
                      vxyz(i+2*pts_cnt) << " k10\n";
         }
         else
         {
            std::cout << "Pt xyz: " << vxyz(i*dim+0) << " " << vxyz(i*dim+1) <<  " " <<
                      vxyz(i*dim+2) << " k10\n";
         }
         std::cout << "FPT DEV: " << c1 << " " << e1 << " " << dist1(i) << " " <<
                   ref1(0) << " " << ref1(1) << " " << ref1(2) << " " <<
                   proc_out1[i] << " k10\n";
      }
   }

   MPI_Barrier(MPI_COMM_WORLD);
   Vector info0    = finder.GetInfo();
   //    info0.Print();
   //    vxyz.Print();
   //    ref_rst0.Print();
   //    dist1.Print();

   finder.Interpolate(field_vals, interp_vals);
   Vector info1    = finder.GetInfo();
   if (interp_vals.UseDevice())
   {
      interp_vals.HostReadWrite();
   }
   vxyz.HostReadWrite();
   //    info1.Print();
   //    interp_vals.Print();

   Array<unsigned int> code_out    = finder.GetCode();
   Array<unsigned int> task_id_out = finder.GetProc();
   Vector dist_p_out = finder.GetDist();
   Vector rst = finder.GetReferencePosition();

   int face_pts = 0, not_found = 0, found_loc = 0, found_away = 0;
   double err = 0.0, max_err = 0.0, max_dist = 0.0;

   Vector pos(dim);
   for (int j = 0; j < vec_dim; j++)
   {
      for (int i = 0; i < pts_cnt; i++)
      {
         if (j == 0)
         {
            (task_id_out[i] == (unsigned)myid) ? found_loc++ : found_away++;
         }

         if (code_out[i] < 2)
         {
            for (int d = 0; d < dim; d++)
            {
               pos(d) = point_ordering == Ordering::byNODES ?
                        vxyz(d*pts_cnt + i) :
                        vxyz(i*dim + d);
            }
            Vector exact_val(vec_dim);
            F_exact(pos, exact_val);
            err = gf_ordering == Ordering::byNODES ?
                  fabs(exact_val(j) - interp_vals[i + j*pts_cnt]) :
                  fabs(exact_val(j) - interp_vals[i*vec_dim + j]);
            max_err  = std::max(max_err, err);
            max_dist = std::max(max_dist, dist_p_out(i));
            if (code_out[i] == 1 && j == 0) { face_pts++; }
         }
         else { if (j == 0) { not_found++; } }
      }
   }

   MPI_Allreduce(MPI_IN_PLACE, &found_loc, 1, MPI_INT, MPI_SUM,
                 pfespace.GetComm());
   MPI_Allreduce(MPI_IN_PLACE, &found_away, 1, MPI_INT, MPI_SUM,
                 pfespace.GetComm());
   MPI_Allreduce(MPI_IN_PLACE, &face_pts, 1, MPI_INT, MPI_SUM, pfespace.GetComm());
   MPI_Allreduce(MPI_IN_PLACE, &not_found, 1, MPI_INT, MPI_SUM,
                 pfespace.GetComm());
   MPI_Allreduce(MPI_IN_PLACE, &max_err, 1, MPI_DOUBLE, MPI_MAX,
                 pfespace.GetComm());
   MPI_Allreduce(MPI_IN_PLACE, &max_dist, 1, MPI_DOUBLE, MPI_MAX,
                 pfespace.GetComm());
   MPI_Allreduce(MPI_IN_PLACE, &err, 1, MPI_DOUBLE, MPI_SUM, pfespace.GetComm());
   //         MPI_Allreduce(MPI_IN_PLACE, &npt_on_faces, 1, MPI_INT, MPI_SUM,
   //                       pfespace.GetComm());


   if (myid == 0)
   {
      cout << setprecision(16)
           << "Total number of elements: " << nelemglob
           << "\nTotal number of procs: " << num_procs
           << "\nSearched total points: " << pts_cnt*num_procs
           << "\nFound locally on ranks:  " << found_loc
           << "\nFound on other tasks: " << found_away
           << "\nPoints not found:     " << not_found
           << "\nPoints on faces:      " << face_pts
           //                    << "\nPoints put on faces:  " << npt_on_faces
           << "\nMax interp error:     " << max_err
           << "\nMax dist (of found):  " << max_dist
           //                    << "\nTotal Time:  " << FindPointsSW.RealTime()
           << endl;
   }

   if (myid == 0)
   {
      cout << "FindPointsGSLIB-Timing-info " <<
           "jobid,ne,np,dim,meshorder,solorder,funcorder,fieldtype,smooth,npts,nptt,"
           <<
           "foundloc,foundaway,notfound,foundface,maxerr,maxdist,"<<
           "setup_split,setup_nodalmapping,setup_setup,findpts_findpts,findpts_device_setup,findpts_mapelemrst,"
           <<
           "interpolate_h1,interpolate_general,interpolate_l2_pass2 " <<
           jobid << "," <<
           nelemglob << "," <<
           num_procs << "," <<
           dim << "," <<
           mesh_poly_deg << "," << order << "," <<
           func_order << "," << fieldtype << "," <<
           smooth << "," <<
           pts_cnt << "," <<
           pts_cnt*num_procs << "," <<
           found_loc << "," <<
           found_away << "," <<
           not_found << "," <<
           face_pts << "," <<
           //           npt_on_faces << "," <<
           max_err << "," <<
           max_dist << "," <<
           finder.setup_split_time << "," <<
           finder.setup_nodalmapping_time << "," <<
           finder.setup_findpts_setup_time << "," <<
           finder.findpts_findpts_time << "," <<
           finder.findpts_setup_device_arrays_time << "," <<
           finder.findpts_mapelemrst_time << "," <<
           finder.interpolate_h1_time << "," <<
           finder.interpolate_general_time << "," <<
           finder.interpolate_l2_pass2_time << "," <<
           std::endl;
   }

   // Free the internal gslib data.
   finder.FreeData();

   delete fec;

   return 0;
}
