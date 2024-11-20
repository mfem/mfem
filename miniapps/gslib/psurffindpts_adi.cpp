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
//    make psurffindpts_adi -j && mpirun -np 1 psurffindpts_adi -mo 4 -nx 1 --dim 2 -o 1 -visit
//    make psurffindpts_adi -j && mpirun -np 2 psurffindpts_adi -mo 2 -dim 3 -nx 2 -o 1 -visit

#include "mfem.hpp"
#include "general/forall.hpp"
#include "../common/mfem-common.hpp"

using namespace mfem;
using namespace std;

double func_order;  // order of the user-defined field function

// Scalar function to project
double field_func(const Vector &x)
{
   const int dim = x.Size();
   double res = 0.0;
   for (int d=0; d<dim; d++)
   {
      res += std::pow(x(d), func_order);
   }
   return res;
}

void F_exact(const Vector &p, Vector &F)
{
   F(0) = field_func(p);
   for (int i = 1; i<F.Size(); i++)
   {
      F(i) = (i+1)*F(0);
   }
}

double ComputeMeshArea(Mesh *mesh)
{
   double area = 0.0;
   for (int i = 0; i < mesh->GetNE(); i++)
   {
      area += mesh->GetElementVolume(i);
   }
   // MPI_Allreduce(MPI_IN_PLACE, &area, 1, MPI_DOUBLE, MPI_SUM, pmesh->GetComm());
   return area;
}

int main (int argc, char *argv[])
{
   std::remove("out");

   // Initialize MPI and HYPRE.
   Mpi::Init(argc, argv);
   Hypre::Init();

   int num_procs = Mpi::WorldSize();
   int myid      = Mpi::WorldRank();

   ofstream ofile("out", std::ios::app);

   // Set the method's default parameters.
   const char *mesh_file = "NULL";
   int order             = 3;
   int mesh_poly_deg     = 3;
   int rs_levels         = 0;
   int rp_levels         = 0;
   bool visualization    = false;
   int fieldtype          = 0;
   int ncomp             = 1;
   bool search_on_rank_0 = false;
   bool hrefinement       = false;
   int point_ordering    = 0;
   int gf_ordering       = 0;
   const char *devopt    = "cpu";
   double jitter         = 0.0;
   int exact_sol_order   = 1;
   int smooth            = 0; //kershaw transformation parameter
   int jobid             = 0;
   int npt               = 100; //points per proc
   int nx                = 6; //points per proc
   int dim               = 3;
   int vdim              = 3;
   int etype             = 0;
   bool visit            = false;
   int gpucode           = 0;

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
   args.AddOption(&dim, "-dim", "--dim",
                  "Dimension");
   args.AddOption(&etype, "-et", "--et",
                  "element type: 0 - quad/hex, 1 - triangle/tetrahedron");
   args.AddOption(&visit, "-visit", "--visit", "-no-visit",
                  "--no-visit",
                  "Enable or disable VISIT output");
   args.AddOption(&gpucode, "-gpucode", "--gpucode",
                  "code for custom gpu kernels");
   args.Parse();

   if (!args.Good())
   {
      args.PrintUsage(cout);
      return 1;
   }
   if (myid==0)
   {
      args.PrintOptions(cout);
   }
   bool cpu_mode = strcmp(devopt,"cpu")==0;

   if (hrefinement)
   {
      MFEM_VERIFY(strcmp(devopt,"cpu")==0, "HR-adaptivity is currently only"
                  " supported on cpus.");
   }
   Device device(devopt);
   if (myid == 0)
   {
      device.Print();
   }

   // restricting max user-defined field function order
   func_order = std::min(exact_sol_order, 4);

   // Initialize and refine the starting mesh.
   int nex = nx*std::pow(2, rs_levels); // mesh size after refinement
   Mesh *mesh = NULL;
   if (strcmp(mesh_file,"NULL")!=0)
   {
      mesh = new Mesh(mesh_file, 1, 1, false);
      for (int lev=0; lev<rs_levels+rp_levels; lev++)
      {
         mesh->UniformRefinement();
      }
      dim  = mesh->Dimension();  // Ref. space dim (might be different from SpaceDim)
   }
   else
   {
      // It seems the mesh generated by code below has unit side length
      if (dim==2)
      {
         mesh = new Mesh(Mesh::MakeCartesian2D(nex, nex, etype == 0 ?
                                               Element::QUADRILATERAL :
                                               Element::TRIANGLE));

      }
      else if (dim==3)
      {
         mesh = new Mesh(Mesh::MakeCartesian3D(nex, nex, nex, etype == 0 ?
                                               Element::HEXAHEDRON :
                                               Element::TETRAHEDRON));
      }
      else
      {
         MFEM_ABORT("Only 2D and 3D supported at the moment.");
      }
   }
   mesh->EnsureNodes();  // Ensure gridfunction object exists in mesh object
   vdim = mesh->SpaceDimension();


   if (visualization && myid == 0)
   {
      osockstream sock(19916, "localhost");
      sock << "mesh\n";
      mesh->Print(sock);
      sock.send();
      sock << "window_title 'Displacements'\n"
           << "window_geometry "
           << 00 << " " << 0 << " " << 400 << " " << 400 << "\n"
           << "keys RmclA" << endl;
   }

   if (myid==0)
   {
      cout << "Mesh curvature of the original mesh: ";
      if (mesh->GetNodes())
      {
         cout << mesh->GetNodes()->OwnFEC()->Name() << endl;
      }
      else
      {
         cout << "(NONE)" << endl;
      }

      if (visit)
      {
         VisItDataCollection dc("inputmesh", mesh);
         dc.SetFormat(DataCollection::SERIAL_FORMAT);
         dc.Save();
      }
   }
   MPI_Barrier(MPI_COMM_WORLD);

   MFEM_VERIFY(mesh_poly_deg>0, "The order of the mesh must be positive.");

   // H1_2D_P<mesh_poly_deg> for false, L2_T1_2D_P<mesh_poly_deg> for true
   mesh->SetCurvature(mesh_poly_deg, false, -1, 0);
   if (myid==0)
   {
      cout << "Mesh curvature after SetCurvature: "
           << mesh->GetNodes()->OwnFEC()->Name()
           << endl;
   }


   Mesh *submesh = NULL;
   if (dim==vdim)
   {
      int nattr = mesh->bdr_attributes.Max();
      // mesh->bdr_attributes.Print();
      Array<int> subdomain_attributes(nattr);
      for (int i=0; i<nattr; i++)
      {
         subdomain_attributes[i] = i+1;
      }
      submesh = new Mesh( SubMesh::CreateFromBoundary(*mesh, subdomain_attributes) );
   }
   else
   {
      submesh = mesh;
   }
   const FiniteElementSpace *sm_fes = NULL;
   const GridFunction *sm_gf = NULL;
   dim  = submesh->Dimension();
   vdim = submesh->SpaceDimension();
   sm_gf = submesh->GetNodes();
   sm_fes = sm_gf->FESpace();

   ParMesh psubmesh(MPI_COMM_WORLD, *submesh);

   MFEM_VERIFY(ncomp>0, "Invalid number of components.");
   FiniteElementCollection *fec = new H1_FECollection(order, dim);
   if (myid==0)
   {
      cout << "H1-GridFunction\n";
   }
   ParFiniteElementSpace sc_fes(&psubmesh, fec, ncomp, gf_ordering);
   ParGridFunction field_vals(&sc_fes);
   VectorFunctionCoefficient F(ncomp, F_exact);
   field_vals.ProjectCoefficient(F);

   if (visit)
   {
      VisItDataCollection dc("psubmesh", &psubmesh);
      dc.RegisterField("soln", &field_vals);
      dc.SetFormat(DataCollection::PARALLEL_FORMAT);
      dc.Save();
   }
   // Display the mesh and the field through glvis.
   if (visualization)
   {
      socketstream sock;
      if (myid == 0)
      {
         sock.open("localhost", 19916);
         sock << "solution\n";
      }
      psubmesh.PrintAsOne(sock);
      field_vals.SaveAsOne(sock);
      if (myid == 0)
      {
         sock << "window_title 'Solution'\n"
              << "window_geometry "
              << 400 << " " << 0 << " " << 400 << " " << 400 << "\n"
              << "keys RmjApp" << endl;
      }
   }
   MPI_Barrier(MPI_COMM_WORLD);

   // -----------FindPointsSetup----------------
   FindPointsGSLIB finder(MPI_COMM_WORLD);
   finder.SetupSurf(psubmesh);
   finder.SetDistanceToleranceForPointsFoundOnBoundary(10);

   double meshvol = ComputeMeshArea(&psubmesh);
   MPI_Allreduce(MPI_IN_PLACE, &meshvol, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
   double meshabbvol = 0.0;
   double meshobbvol = 0.0;

   Mesh *mesh_abb, *mesh_obb, *mesh_lhbb, *mesh_ghbb;
   if (visit)
   {
      mesh_abb  = finder.GetBoundingBoxMeshSurf(0);  // Axis aligned bounding box
      mesh_obb  = finder.GetBoundingBoxMeshSurf(1);  // Oriented bounding box
      mesh_lhbb = finder.GetBoundingBoxMeshSurf(2);  // Local Hash bounding box
      mesh_ghbb = finder.GetBoundingBoxMeshSurf(3);  // Global Hash bounding box
      if (mesh_abb && myid == 0) {
         meshabbvol = ComputeMeshArea(mesh_abb);
         meshobbvol = ComputeMeshArea(mesh_obb);
      }
      if (myid==0)
      {
         VisItDataCollection dc0("findersurfabb", mesh_abb);
         dc0.SetFormat(DataCollection::SERIAL_FORMAT);
         dc0.Save();

         {
            Array<int> attrlist(1);
            for (int i = 0; i < mesh_abb->GetNE(); i++)
            {
               attrlist[0] = i+1;
               auto mesh_abbt = SubMesh::CreateFromDomain(*mesh_abb, attrlist);
               VisItDataCollection dct("findersurfabbt", &mesh_abbt);
               dct.SetFormat(DataCollection::SERIAL_FORMAT);
               dct.SetCycle(i);
               dct.SetTime(i*1.0);
               dct.Save();
            }
         }

         VisItDataCollection dc1("findersurfobb", mesh_obb);
         dc1.SetFormat(DataCollection::SERIAL_FORMAT);
         dc1.Save();
         {
            {
               Array<int> attrlist(1);
               for (int i = 0; i < mesh_obb->GetNE(); i++)
               {
                  attrlist[0] = i+1;
                  auto mesh_obbt = SubMesh::CreateFromDomain(*mesh_obb, attrlist);
                  VisItDataCollection dct("findersurfobbt", &mesh_obbt);
                  dct.SetFormat(DataCollection::SERIAL_FORMAT);
                  dct.SetCycle(i);
                  dct.SetTime(i*1.0);
                  dct.Save();
               }
            }
         }

         VisItDataCollection dc2("findersurflhbb", mesh_lhbb);
         dc2.SetFormat(DataCollection::SERIAL_FORMAT);
         dc2.Save();

         VisItDataCollection dc3("findersurfghbb", mesh_ghbb);
         dc3.SetFormat(DataCollection::SERIAL_FORMAT);
         dc3.Save();
      }
   }
   MPI_Barrier(MPI_COMM_WORLD);

   if (myid == 0)
   {
      std::cout << "Mesh, AABB, OBB areas: " << meshvol << " " <<
      meshabbvol << " " << meshobbvol << std::endl;
   }
   // -----------FindPointsSetup----------------

   // const int vdim   = sm_fes->GetVDim();
   int nel_sm = submesh->GetNE();
   // int nel_sm = 1000;
   Vector point_pos(npt*vdim*nel_sm);
   Vector field_exact(nel_sm*npt*ncomp);
   int npt_per_proc = point_pos.Size()/vdim;
   if (myid==0)
   {
      cout << "points_per_proc: " << npt_per_proc << ", nel_sm: " << nel_sm << endl;
   }
   // Get element coordinates
   for (int i=0; i<nel_sm; i++)
   {
      const FiniteElement *fe = sm_fes->GetFE(i);
      const Geometry::Type gt = fe->GetGeomType();
      const int rdim          = fe->GetDim();

      ElementTransformation *transf = sm_fes->GetElementTransformation(i);

      Vector pos_ref1(npt);
      Vector pos_ref2(npt);
      pos_ref1.Randomize(0.0);
      pos_ref2.Randomize(0.0);
      for (int j=0; j<npt; j++)
      {
         IntegrationPoint ip;
         if (j<4)
         {
            ip.x = (j%2==0) ? 0.0 : 1.0;
            if (rdim==2)
            {
               ip.y = j/2==0 ? 0.0 : 1.0;
            }
         }
         else
         {
            ip.x = pos_ref1(j);
            if (rdim==2)
            {
               ip.y = pos_ref2(j);
            }
         }
         Vector pos_i(vdim);
         transf->SetIntPoint(&ip);
         transf->Transform(ip, pos_i);
         for (int d=0; d<vdim; d++)
         {
            point_pos(nel_sm*npt*d + i*npt + j) = pos_i(d);
         }
         // point_pos(0) = 5.07731534;// 0.289066938 -0.934150366
         // point_pos(1) = 0.0;
         // point_pos(2) = 0.0;
         // point_pos(0) = 0.451419;
         // point_pos(1) = 2.21173;
         // point_pos(2) =  0;
         // point_pos ordering is bynodes so be careful
      }
   }

   finder.FindPointsSurf(point_pos);

   MPI_Barrier(MPI_COMM_WORLD);

   Array<unsigned int> code_out1 = finder.GetCode();
   Array<unsigned int> el_out1   = finder.GetGSLIBElem();
   Vector ref_rst1               = finder.GetGSLIBReferencePosition();
   Vector dist1                  = finder.GetDist();
   Array<unsigned int> proc_out1 = finder.GetProc();
   point_pos.HostReadWrite();

   int notfound = 0;
   for (int i=0; i<code_out1.Size(); i++)
   {
      int c1 = code_out1[i];
      int e1 = el_out1[i];
      Vector ref1(ref_rst1.GetData()+i*dim, dim);
      Vector dref = ref1;
      if ( c1==2 || (std::fabs(dist1(i))>1e-10 && myid==0) )
      {
         notfound++;
         if (point_ordering==0)
         {
            std::cout << "Pt xyz: " << point_pos(i) << " "
                      << point_pos(i + npt_per_proc) <<  " "
                      << (vdim==3 ? point_pos(i+2*npt_per_proc) : 0)
                      << " adi\n";
         }
         else
         {
            std::cout << "Pt xyz: " << point_pos(i*vdim+0) << " "
                      << point_pos(i*vdim+1) <<  " "
                      << (vdim==3 ? point_pos(i*vdim+2) : 0)
                      << " adi\n";
         }
         std::cout << "FPT DEV (c1,e1,dist1,r,s,t,proc): "
                   << c1                    << " "
                   << e1                    << " "
                   << dist1(i)              << " "
                   << ref1(0)               << " "
                   << (dim==2 ? ref1(1):0 ) << " "
                   << proc_out1[i]          << " adi\n";
      }
   }

   MPI_Barrier(MPI_COMM_WORLD);
   Array<int> newton_out = finder.GetNewtonIters();
   int newton_min = newton_out.Min();
   int newton_max = newton_out.Max();
   int newton_mean = newton_out.Sum()/newton_out.Size();
   if (myid==0)
   {
      std::cout << "Newton iteration min/max/mean: "
                << newton_min  << " "
                << newton_max  << " "
                << newton_mean << endl;
   }

   MPI_Barrier(MPI_COMM_WORLD);

   // Interpolate the field at the points
   Vector field_out(npt_per_proc*ncomp);
   field_out.UseDevice(true);

   finder.InterpolateSurf(field_vals, field_out);

   Vector info1 = finder.GetInfo();
   if (field_out.UseDevice())
   {
      field_out.HostReadWrite();
   }
   point_pos.HostReadWrite();

   Array<unsigned int> code_out    = finder.GetCode();
   Array<unsigned int> task_id_out = finder.GetProc();
   Vector dist_p_out               = finder.GetDist();
   Vector rst                      = finder.GetReferencePosition();
   int face_pts = 0, not_found = 0, found_loc = 0, found_away = 0;
   double err = 0.0, max_err = 0.0, max_dist = 0.0;
   Vector pos(vdim);
   for (int j=0; j<ncomp; j++)
   {
      for (int i=0; i<npt_per_proc; i++)
      {
         if (j==0)
         {
            (task_id_out[i]==(unsigned)myid) ? found_loc++ : found_away++;
         }

         if (code_out[i]<2)
         {
            for (int d=0; d<vdim; d++)
            {
               pos(d) = point_ordering==Ordering::byNODES ?
                        point_pos(d*npt_per_proc + i) :
                        point_pos(i*vdim + d);
            }

            Vector exact_val(ncomp);
            F_exact(pos,exact_val);
            err = gf_ordering == Ordering::byNODES ?
                  fabs(exact_val(j) - field_out[i + j*npt_per_proc]) :
                  fabs(exact_val(j) - field_out[i*ncomp + j]);
            // if (err > 1e-10)
            // {
            //    std::cout << std::setprecision(10) << pos(0) << " " << pos(1) << " "
            //              << exact_val(0) << " " << " "
            //              << field_out[i + j*npt_per_proc] << " " <<
            //              err << " k10info\n";

            // }
            max_err  = std::max(max_err, err);
            max_dist = std::max(max_dist, dist_p_out(i));

            if (code_out[i]==1 && j==0)
            {
               face_pts++;
            }
         }
         else
         {
            if (j==0)
            {
               not_found++;
            }
         }
      }
   }
   MPI_Allreduce(MPI_IN_PLACE, &found_loc, 1, MPI_INT, MPI_SUM, sc_fes.GetComm());
   MPI_Allreduce(MPI_IN_PLACE, &found_away, 1, MPI_INT, MPI_SUM, sc_fes.GetComm());
   MPI_Allreduce(MPI_IN_PLACE, &face_pts, 1, MPI_INT, MPI_SUM, sc_fes.GetComm());
   MPI_Allreduce(MPI_IN_PLACE, &not_found, 1, MPI_INT, MPI_SUM, sc_fes.GetComm());
   MPI_Allreduce(MPI_IN_PLACE, &max_err, 1, MPI_DOUBLE, MPI_MAX, sc_fes.GetComm());
   MPI_Allreduce(MPI_IN_PLACE, &max_dist, 1, MPI_DOUBLE, MPI_MAX,
                 sc_fes.GetComm());
   MPI_Allreduce(MPI_IN_PLACE, &err, 1, MPI_DOUBLE, MPI_SUM, sc_fes.GetComm());

   if (myid==0)
   {
      cout << setprecision(16)
           <<   "Total number of elements: " << nel_sm
           << "\nTotal number of procs:    " << num_procs
           << "\nSearched total points:    " << npt_per_proc*num_procs
           << "\nFound locally on ranks:   " << found_loc
           << "\nFound on other tasks:     " << found_away
           << "\nPoints not found:         " << not_found
           << "\nPoints on faces:          " << face_pts
           << "\nMax interp error:         " << max_err
           << "\nMax dist (of found):      " << max_dist
           //                    << "\nTotal Time:  " << FindPointsSW.RealTime()
           << endl;
   }

   if (myid==0)
   {
      cout << "FindPointsGSLIB-Timing-info "
           << "jobid,devid,gpucode,ne,np,dim,spacedim,meshorder,solorder,funcorder,fieldtype,smooth,npts,nptt,"
           << "foundloc,foundaway,notfound,foundface,maxerr,maxdist,"
           << "setup_split,setup_nodalmapping,setup_setup,findpts_findpts,findpts_device_setup,findpts_mapelemrst,"
           << "interpolate_h1,interpolate_general,interpolate_l2_pass2 "
           << jobid << ","
           << device.GetId() << ","
           << gpucode << ","
           << nel_sm << ","
           << num_procs << ","
           << dim << ","
           << vdim << ","
           << mesh_poly_deg << "," << order << ","
           << func_order << "," << fieldtype << ","
           << smooth << ","
           << npt_per_proc << ","
           << npt_per_proc*num_procs << ","
           << found_loc << ","
           << found_away << ","
           << not_found << ","
           << face_pts << ","
           << max_err << ","
           << max_dist << ","
           << finder.setup_split_time << ","
           << finder.setup_nodalmapping_time << ","
           << finder.setup_findpts_setup_time << ","
           << finder.findpts_findpts_time << ","
           << finder.findpts_setup_device_arrays_time << ","
           << finder.findpts_mapelemrst_time << ","
           << finder.interpolate_h1_time << ","
           << finder.interpolate_general_time << ","
           << finder.interpolate_l2_pass2_time << ","
           << std::endl;
   }

   int mesh_size = finder.GetGLLMesh().Size();

   if (myid == 0)
   {
      cout << "FindPointsGSLIB-KernelTiming-info "
           << "jobid,devid,gpucode,ne,np,dim,spacedim,meshorder,solorder,funcorder,fieldtype,smooth,npts,nptt,gllsize,"
           << "mintime,measuredmintime,actualkerneltime,fastkerneltime "
           << jobid                              << ","
           << device.GetId()                     << ","
           << gpucode                            << ","
           << nel_sm                             << ","
           << num_procs                          << ","
           << dim                                << ","
           << vdim                               << ","
           << mesh_poly_deg                      << ","
           << order                              << ","
           << func_order                         << ","
           << fieldtype                           << ","
           << smooth                             << ","
           << npt_per_proc                       << ","
           << npt_per_proc*num_procs             << ","
           << mesh_size                          << ","
           << finder.min_fpt_kernel_time          << ","
           << finder.measured_min_fpt_kernel_time << ","
           << finder.fpt_kernel_time              << ","
           << finder.fast_fpt_kernel_time         << ","
           << std::endl;
   }



   // finder.FreeData();
   delete submesh, mesh;
   // // delete fec;

   // // cout << "Just before FreeData" << endl;
   // // finder.FreeData();


   return 0;
}
