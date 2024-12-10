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
//    mpirun -np 2 pfindpts -m ../../data/rt-2d-p4-tri.mesh -o 8 -mo 4
//    mpirun -np 2 pfindpts -m ../../data/inline-tri.mesh -o 3
//    mpirun -np 2 pfindpts -m ../../data/inline-quad.mesh -o 3
//    mpirun -np 2 pfindpts -m ../../data/inline-quad.mesh -o 3 -po 1
//    mpirun -np 2 pfindpts -m ../../data/inline-quad.mesh -o 3 -po 1 -gfo 1 -nc 2
//    mpirun -np 2 pfindpts -m ../../data/inline-quad.mesh -o 3 -hr
//    mpirun -np 2 pfindpts -m ../../data/inline-tet.mesh -o 3
//    mpirun -np 2 pfindpts -m ../../data/inline-hex.mesh -o 3
//    mpirun -np 2 pfindpts -m ../../data/inline-wedge.mesh -o 3
//    mpirun -np 2 pfindpts -m ../../data/amr-quad.mesh -o 2
//    mpirun -np 2 pfindpts -m ../../data/rt-2d-q3.mesh -o 8 -mo 4 -ft 2
//    mpirun -np 2 pfindpts -m ../../data/inline-quad.mesh -ft 1 -sr0
//    mpirun -np 2 pfindpts -m ../../data/square-mixed.mesh -o 2 -mo 2
//    mpirun -np 2 pfindpts -m ../../data/square-mixed.mesh -o 2 -mo 2 -hr
//    mpirun -np 2 pfindpts -m ../../data/square-mixed.mesh -o 2 -mo 3 -ft 2
//    mpirun -np 2 pfindpts -m ../../data/fichera-mixed.mesh -o 3 -mo 2
//    mpirun -np 2 pfindpts -m ../../data/inline-pyramid.mesh -o 1 -mo 1
//    mpirun -np 2 pfindpts -m ../../data/tinyzoo-3d.mesh -o 1 -mo 1
// Device runs:
//    mpirun -np 2 pfindpts -m ../../data/inline-quad.mesh -o 3 -mo 2 -random 1 -d debug
//    mpirun -np 2 pfindpts -m ../../data/amr-quad.mesh -rs 1 -o 4 -mo 2 -random 1 -npt 100 -d debug
//    mpirun -np 2 pfindpts -m ../../data/inline-hex.mesh -o 3 -mo 2 -random 1 -d debug


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
   int randomization     = 0;
   int npt               = 100; //points per proc
   int visport           = 19916;

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
   args.AddOption(&randomization, "-random", "--random",
                  "0: generate points randomly in the bounding box of domain,"
                  "1: generate points randomly inside each element in mesh.");
   args.AddOption(&npt, "-npt", "--npt",
                  "# points per proc or element");
   args.AddOption(&visport, "-p", "--send-port", "Socket for GLVis.");

   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(cout);
      return 1;
   }
   if (myid == 0) { args.PrintOptions(cout); }

   bool cpu_mode = strcmp(devopt,"cpu")==0;
   Device device(devopt);
   if (myid == 0) { device.Print();}

   func_order = std::min(order, 2);

   // Initialize and refine the starting mesh.
   Mesh *mesh = new Mesh(mesh_file, 1, 1, false);
   for (int lev = 0; lev < rs_levels; lev++) { mesh->UniformRefinement(); }
   const int dim = mesh->Dimension();

   if (mesh->GetNumGeometries(dim) != 1 ||
       (mesh->GetElementType(0)!=Element::QUADRILATERAL &&
        mesh->GetElementType(0) != Element::HEXAHEDRON))
   {
      randomization = 0;
   }

   Vector xmin, xmax;
   mesh->GetBoundingBox(xmin, xmax);

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
           << "y in [" << pos_min(1) << ", " << pos_max(1) << "]" << std::endl;
      if (dim == 3)
      {
         cout << "z in [" << pos_min(2) << ", " << pos_max(2) << "]" << std::endl;
      }
   }

   // Distribute the mesh.
   if (hrefinement) { mesh->EnsureNCMesh(); }
   ParMesh pmesh(MPI_COMM_WORLD, *mesh);
   if (randomization == 0) { delete mesh; }
   else
   {
      // we will need mesh nodal space later
      if (mesh->GetNodes() == NULL) { mesh->SetCurvature(1); }
   }
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

   MFEM_VERIFY(ncomp > 0, "Invalid number of components.");
   int vec_dim = ncomp;
   FiniteElementCollection *fec = NULL;
   if (fieldtype == 0)
   {
      fec = new H1_FECollection(order, dim);
      if (myid == 0) { cout << "H1-GridFunction" << std::endl; }
   }
   else if (fieldtype == 1)
   {
      fec = new L2_FECollection(order, dim);
      if (myid == 0) { cout << "L2-GridFunction" << std::endl; }
   }
   else if (fieldtype == 2)
   {
      fec = new RT_FECollection(order, dim);
      ncomp = 1;
      vec_dim = dim;
      if (myid == 0) { cout << "H(div)-GridFunction" << std::endl; }
   }
   else if (fieldtype == 3)
   {
      fec = new ND_FECollection(order, dim);
      ncomp = 1;
      vec_dim = dim;
      if (myid == 0) { cout << "H(curl)-GridFunction" << std::endl; }
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
   Vector vxyz;
   vxyz.UseDevice(!cpu_mode);
   int npt_face_per_elem = 4; // number of pts on el faces for randomization != 0
   if (randomization == 0)
   {
      vxyz.SetSize(pts_cnt * dim);
      vxyz.Randomize(myid+1);

      // Scale based on min/max dimensions
      for (int i = 0; i < pts_cnt; i++)
      {
         for (int d = 0; d < dim; d++)
         {
            if (point_ordering == Ordering::byNODES)
            {
               vxyz(i + d*pts_cnt) =
                  pos_min(d) + vxyz(i + d*pts_cnt) * (pos_max(d) - pos_min(d));
            }
            else
            {
               vxyz(i*dim + d) =
                  pos_min(d) + vxyz(i*dim + d) * (pos_max(d) - pos_min(d));
            }
         }
      }
   }
   else // randomization == 1
   {
      pts_cnt = npt*nelemglob;
      vxyz.SetSize(pts_cnt * dim);
      for (int i=0; i<mesh->GetNE(); i++)
      {
         const FiniteElementSpace *s_fespace = mesh->GetNodalFESpace();
         ElementTransformation *transf = s_fespace->GetElementTransformation(i);

         Vector pos_ref1(npt*dim);
         pos_ref1.Randomize((myid+1)*17.0);
         for (int j=0; j<npt; j++)
         {
            IntegrationPoint ip;
            ip.x = pos_ref1(j*dim + 0);
            ip.y = pos_ref1(j*dim + 1);
            if (dim == 3)
            {
               ip.z = pos_ref1(j*dim + 2);
            }
            if (j < npt_face_per_elem)
            {
               ip.x = 0.0; // force point to be on the face
            }
            Vector pos_i(dim);
            transf->Transform(ip, pos_i);
            for (int d=0; d<dim; d++)
            {
               if (point_ordering == Ordering::byNODES)
               {
                  vxyz(j + npt*i + d*npt*nelemglob) = pos_i(d);
               }
               else
               {
                  vxyz((j + npt*i)*dim + d) = pos_i(d);
               }
            }
         }
      }
   }

   if ( (myid != 0) && (search_on_rank_0) )
   {
      pts_cnt = 0;
      vxyz.Destroy();
   }

   // Find and Interpolate FE function values on the desired points.
   Vector interp_vals(pts_cnt*vec_dim);
   FindPointsGSLIB finder(MPI_COMM_WORLD);
   finder.Setup(pmesh);
   finder.SetDistanceToleranceForPointsFoundOnBoundary(10);
   // Enable GPU to CPU fallback for GPUData only if you must use an older
   // version of GSLIB.
   // finder.SetGPUtoCPUFallback(true);
   finder.FindPoints(vxyz, point_ordering);

   Array<unsigned int> code_out1    = finder.GetCode();
   Array<unsigned int> el_out1    = finder.GetGSLIBElem();
   Vector ref_rst1    = finder.GetGSLIBReferencePosition();
   Vector ref_rst0   = finder.GetReferencePosition();
   Vector dist1    = finder.GetDist();
   Array<unsigned int> proc_out1    = finder.GetProc();

   finder.Interpolate(field_vals, interp_vals);
   if (interp_vals.UseDevice())
   {
      interp_vals.HostReadWrite();
   }
   vxyz.HostReadWrite();

   Array<unsigned int> code_out    = finder.GetCode();
   Array<unsigned int> task_id_out = finder.GetProc();
   Vector dist_p_out = finder.GetDist();
   Vector rst = finder.GetReferencePosition();

   int face_pts = 0, not_found = 0, found_loc = 0, found_away = 0;
   double error = 0.0, max_error = 0.0, max_dist = 0.0;

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
            error = gf_ordering == Ordering::byNODES ?
                    fabs(exact_val(j) - interp_vals[i + j*pts_cnt]) :
                    fabs(exact_val(j) - interp_vals[i*vec_dim + j]);
            max_error  = std::max(max_error, error);
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
   MPI_Allreduce(MPI_IN_PLACE, &max_error, 1, MPI_DOUBLE, MPI_MAX,
                 pfespace.GetComm());
   MPI_Allreduce(MPI_IN_PLACE, &max_dist, 1, MPI_DOUBLE, MPI_MAX,
                 pfespace.GetComm());
   MPI_Allreduce(MPI_IN_PLACE, &error, 1, MPI_DOUBLE, MPI_SUM, pfespace.GetComm());


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
           << "\nMax interp error:     " << max_error
           << "\nMax dist (of found):  " << max_dist
           << endl;
   }

   // Free the internal gslib data.
   finder.FreeData();

   delete fec;
   if (randomization != 0) { delete mesh; }

   return 0;
}
