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


#include "mfem.hpp"
#include "general/forall.hpp"
#include "../common/mfem-common.hpp"
namespace gslib
{
   #include "gslib.h"
}

using namespace mfem;
using namespace std;

template <std::size_t D, std::size_t... Ns>
struct out_pt
{
   unsigned int index, proc;
   std::array<double,  D + (Ns + ...)> data;

   auto get_v_size()
   {
      return (Ns + ...);
   }

   auto get_n_scalars()
   {
      return D;
   }
};

// template <std::size_t... Ns>
// struct sum;

// template <>
// struct sum<> {
//     static constexpr std::size_t value = 0;
// };

// template <std::size_t First, std::size_t... Rest>
// struct sum<First, Rest...> {
//     static constexpr std::size_t value = First + sum<Rest...>::value;
// };


// template <std::size_t D, std::size_t... Ns>
// struct out_pt
// {
//     unsigned int index, proc;
//     std::array<double, D + sum<Ns...>::value> data;

//     auto get_v_size()
//     {
//        return sum<Ns...>::value;
//     }

//     auto get_n_scalars()
//     {
//        return D;
//     }
// };

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
                  "Enable search only on rank 0 (disable to search points on all tasks). "
                  "All points added by other procs are ignored.");
   args.AddOption(&hrefinement, "-hr", "--h-refinement", "-no-hr",
                  "--no-h-refinement",
                  "Do random h refinements to mesh (does not work for pyramids).");
   args.AddOption(&point_ordering, "-po", "--point-ordering",
                  "Ordering of points to be found."
                  "0 (default): byNodes, 1: byVDIM");
   args.AddOption(&gf_ordering, "-gfo", "--gridfunc-ordering",
                  "Ordering of fespace that will be used for grid function to be interpolated. "
                  "0 (default): byNodes, 1: byVDIM");
   args.AddOption(&devopt, "-d", "--device",
                  "Device configuration string, see Device::Configure().");
   args.AddOption(&randomization, "-random", "--random",
                  "0: generate points randomly in the bounding box of domain, "
                  "1: generate points randomly inside each element in mesh.");
   args.AddOption(&npt, "-npt", "--npt",
                  "# points / rank initialized on entire mesh (random = 0) or every element (random = 1).");
   args.AddOption(&visport, "-p", "--send-port", "Socket for GLVis.");

   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(cout);
      return 1;
   }
   if (myid == 0) { args.PrintOptions(cout); }

   int npart = 10;
   constexpr int dim = 3;

   using outPt_t = out_pt<dim, 2, 2, 3>;
   outPt_t *opt;
   struct gslib::array out_pt_pt;
   // out_pt_pt->n = npart;
   // out_pt_pt->max = npart;
   // out_pt_pt->ptr = data.GetData();
   array_init(outPt_t, &out_pt_pt, npart); // array of struct initialized
   Array<unsigned int> proc_info(npart);

   opt = (outPt_t *)out_pt_pt.ptr;
   for (int i = 0; i < npart; i++)
   {
      opt[i].index = myid;
      opt[i].proc = myid % 2 == 0 ? 0 : (myid + 1) % num_procs;
      proc_info[i] = opt[i].proc;
      for (int d = 0;
               d < (opt->get_v_size()+opt->get_n_scalars());
               d++)
      {
         opt[i].data[d] = 0.1 * i + 0.01 * d + myid;
      }
   }
   out_pt_pt.n = npart;

   struct gslib::comm *gsl_comm = new gslib::comm;
   struct gslib::crystal *cr      = new gslib::crystal;
   comm_init(gsl_comm, MPI_COMM_WORLD);
   crystal_init(cr, gsl_comm);

   // use this if proc is a member of the struct
   sarray_transfer(outPt_t, &out_pt_pt, proc, 1, cr);

   // use below if you want to specify proc as a separate list
   // sarray_transfer_ext(outPt_t, &out_pt_pt, proc_info.GetData(), 1, cr);

   int npart_recv = out_pt_pt.n;
   opt = (outPt_t *)out_pt_pt.ptr;
   for (int i = 0; i < npart_recv; i++)
   {
      if (myid == 0)
      {
         std::cout << i << " " << opt[i].index << " "
                   << opt[i].proc << " k10info\n";
         for (int d = 0; d < (opt->get_v_size()+opt->get_n_scalars()); d++)
         {
            std::cout << opt[i].data[d] << " ";
         }
         std::cout << " k10info\n";
      }
   }

   MPI_Barrier(MPI_COMM_WORLD);
   outPt_t dummy = opt[0];
   std::cout << myid <<   " " << npart_recv << " "
             << dummy.proc << " k10info\n";

   array_free(&out_pt_pt);



   return 0;
}
