//               Generate a NURBS mesh and its LOR version from scratch
//
// Compile with: make nurbs_lor_cartesian
//
// Sample runs:  nurbs_lor_cartesian -d 2 -n 4 -o 3 -interp 3
//               nurbs_lor_cartesian -d 3 -n 5 -o 2 -nel 10 -m 2
//               nurbs_lor_cartesian -d 2 -n 1 -o 2 -nel 50 -a 0.3
//               nurbs_lor_cartesian -d 2 -n 2 -o 2 -nel 0 -a 1 -m 1
//
// Description:  This example code generates a NURBS mesh "from scratch"
//               by building up a patch topology mesh and patches. A LOR
//               version of the mesh is then generated using an interpolant
//               defined by interp_rule. The domain of the mesh is always
//               [0, np] in each dimension, where np is the number of patches
//
//               Interpolation rules (-interp):
//                 - 0: Greville points (default)
//                 - 1: Botella points
//                 - 2: Demko points
//                 - 3: Uniform points

#include "mfem.hpp"
#include <iostream>

using namespace std;
using namespace mfem;

int main(int argc, char *argv[])
{
   // 1. Parse command-line options.
   int dim = 2;
   bool flatdim = false; // flattens other dimensions
   int np = 1;
   int order = 1;
   int mult = 1;
   int interp_rule_ = 0;
   int nel_per_patch = 0;
   real_t alpha = 1.0;

   OptionsParser args(argc, argv);
   args.AddOption(&dim, "-d", "--dim",
                  "Dimension of the mesh (1, 2, or 3).");
   args.AddOption(&flatdim, "-fdim", "--flat-dim", "-no-fdim",
                  "--no-flat-dim", "Flatten dimensions except the first (n=1).");
   args.AddOption(&np, "-n", "--num-patches",
                  "Number of patches in the mesh per dimension.");
   args.AddOption(&order, "-o", "--order",
                  "Order of nurbs bases.");
   args.AddOption(&mult, "-m", "--mult",
                  "Multiplicity of interior knots; must be [1, p+1].");
   args.AddOption(&interp_rule_, "-interp", "--interpolation-rule",
                  "Interpolation Rule: 0 - Greville, 1 - Botella, 2 - Demko, 3 - Uniform");
   args.AddOption(&nel_per_patch, "-nel", "--nelements-per-patch",
                  "Number of elements per patch per dimension. "
                  "Default (0) is the patch index + 1.");
   args.AddOption(&alpha, "-a", "--alpha",
                  "Knotvectors are distributed in [0,1]. alpha is the ratio of the largest "
                  "vs smallest interval. 1 = uniform. If negative, will use reciprocal.");
   args.Parse();

   // Print & verify options
   args.PrintOptions(cout);
   MFEM_VERIFY(dim >= 1 && dim <= 3, "Invalid dimension");
   MFEM_VERIFY(np >= 1, "Must have at least one patch");
   MFEM_VERIFY(order >= 1, "Order of nurbs bases must be at least 1");
   MFEM_VERIFY(mult >= 1 && mult <= order+1, "Multiplicity must be in [1, p+1]");
   MFEM_VERIFY(nel_per_patch >= 0, "Invalid elements per patch");
   MFEM_VERIFY(alpha != 0, "Invalid knot distribution parameter");
   alpha = (alpha > 0) ? alpha : -1.0/alpha;
   NURBSInterpolationRule interp_rule = static_cast<NURBSInterpolationRule>(interp_rule_);

   // 1. Parameters
   const int nx = (dim >= 1) ? np : 1; // Number of patches in each dimension
   const int ny = (flatdim) ? 1 : ((dim >= 2) ? np : 1);
   const int nz = (flatdim) ? 1 : ((dim == 3) ? np : 1);
   const int NP = nx*ny*nz;            // Total number of patches in the mesh
   const int pdim = dim + 1;           // Projective/homogeneous dimension

   // 2. Create the patch-topology mesh
   //    Default ordering is space-filling-curve, set to false to get Cartesian ordering
   Mesh patchTopo;
   if (dim == 1)
   {
      // patchTopo = new Mesh::MakeCartesian1D(NP, L);
      patchTopo = Mesh::MakeCartesian1D(nx, (real_t)nx);
   }
   else if (dim == 2)
   {
      patchTopo = Mesh::MakeCartesian2D
      (
         nx, ny, Element::QUADRILATERAL, true,
         (real_t)nx, (real_t)ny, false
      );
   }
   else if (dim == 3)
   {
      patchTopo = Mesh::MakeCartesian3D
      (
         nx, ny, nz, Element::HEXAHEDRON,
         (real_t)nx, (real_t)ny, (real_t)nz, false
      );
   }
   else
   {
      MFEM_ABORT("Invalid dimension");
   }

   // 3. Create the reference knotvectors and control points
   //    for each patch (same in all dimensions)
   Array<const KnotVector*> kv_ref(np);
   std::vector<Array<real_t>> cpts_ref(np);
   Vector intervals; // intervals between knots
   Array<int> cont;  // continuity at each knot
   Vector x;         // physical coordinates to interpolate
   int nel;          // Number of elements
   int nknot;        // Number of knots
   int ncp;          // Number of control points

   for (int I = 0; I < np; I++)
   {
      if (nel_per_patch == 0)
      {
         // Default - each patch has (I+1) knot spans/elements in each dimension.
         nel = I + 1;
      }
      else
      {
         // Constant number of elements for each patch
         nel = nel_per_patch;
      }

      // Ends always have C^{-1} continuity
      nknot = 2*(order+1) + mult*(nel-1);
      ncp = nknot - order - 1;

      // Define knot vectors by intervals and continuity
      intervals.SetSize(nel);
      if (nel == 1)
      {
         intervals[0] = 1.0;
      }
      else
      {
         for (int i = 0; i < nel; i++)
         {
            intervals[i] = 1.0 + (alpha - 1.0) * i / (nel - 1.0);
         }
         intervals /= intervals.Sum(); // normalize
      }

      cont.SetSize(nel+1);
      cont[0] = cont[nel] = -1;
      for (int i = 1; i < nel; i++)
      {
         cont[i] = order-mult;
      }
      kv_ref[I] = new KnotVector(order, intervals, cont);

      // Knots to interpolate at
      Vector u = kv_ref[I]->GetInterpolationPoints(NURBSInterpolationRule::Botella);
      // Coordinates to interpolate (linearly increasing from I to I+1)
      x.SetSize(ncp);
      for (int i = 0; i < ncp; i++)
      {
         x[i] = u[i]/u[ncp-1] + I;
      }
      // Find control points that interpolate the coordinates
      cpts_ref[I].SetSize(ncp);
      Vector cpts(ncp);
      kv_ref[I]->GetInterpolant(x, u, cpts);
      cpts_ref[I].CopyFrom(cpts.GetData());
   }
   Array<real_t> cpts_flat({0.0, 1.0}); // control points for flattened dimensions

   // 4. Create the patches
   Array<NURBSPatch*> patches(NP);
   int I,J,K; // patch indices
   int i,j,k; // dof indices
   int dofidx;
   constexpr int maxdim = 3;
   Array<int> NCP(maxdim); // number of control points per dim
   NCP = 1; // init

   for (int p = 0; p < NP; p++)
   {
      Array<const KnotVector*> kvs(dim);
      I = p % nx;
      J = (p / nx) % ny;
      K = p / (ny * nx);
      int IJK[maxdim] = {I,J,K};

      // Collect the knot vectors for this patch
      for (int d = 0; d < dim; d++)
      {
         if (d!=0 && flatdim)
         {
            // If flattened, make a basic 1st order knot vector
            kvs[d] = new KnotVector(1, Vector({0.0, 1.0}));
         }
         else
         {
            kvs[d] = new KnotVector(*kv_ref[IJK[d]]);
         }
         NCP[d] = kvs[d]->GetNCP();
      }

      // Define the control points for this patch
      // The domain for each patch in physical space is [I, I+1] x [J, J+1] x [K, K+1]
      Array<real_t> control_points(pdim * NCP.Product());
      for (int k = 0; k < NCP[2]; k++)
      {
         for (int j = 0; j < NCP[1]; j++)
         {
            for (int i = 0; i < NCP[0]; i++)
            {
               dofidx = i + j*NCP[0] + k*NCP[0]*NCP[1];
               int ijk[maxdim] = {i,j,k};

               // Set the control points (+ weight) for the LO mesh
               for (int d = 0; d < dim; d++)
               {
                  real_t c = (d!=0 && flatdim) ? cpts_flat[ijk[d]] : cpts_ref[IJK[d]][ijk[d]];
                  control_points[pdim*dofidx + d] = c;
               }
               control_points[pdim*dofidx + dim] = 1.0; // weight
            }
         }
      }

      // Create patch
      patches[p] = new NURBSPatch(kvs, pdim, control_points.GetData());
   }

   // 5. Create the mesh
   NURBSExtension ext(&patchTopo, patches);
   Mesh mesh = Mesh(ext);

   // 6. Create the LOR mesh
   // const int Ndof = mesh.NURBSext->GetNDof();
   // SparseMatrix* R = new SparseMatrix(Ndof, Ndof);
   Mesh lo_mesh = mesh.GetLowOrderNURBSMesh(interp_rule);

   // 7. Write meshes to file
   // High-order mesh
   ofstream orig_ofs("ho_mesh.mesh");
   orig_ofs.precision(8);
   mesh.Print(orig_ofs);
   cout << "High-Order mesh written to ho_mesh.mesh" << endl;

   // Low-order mesh
   ofstream ofs("lo_mesh.mesh");
   ofs.precision(8);
   lo_mesh.Print(ofs);
   cout << "Low-Order mesh written to lo_mesh.mesh" << endl;

   // Patch topology mesh
   // ofstream topo_ofs("topo_mesh.mesh");
   // topo_ofs.precision(8);
   // patchTopo.Print(topo_ofs);
   // cout << "Patch topology mesh written to topo_mesh.mesh" << endl;

   // Free memory
   // delete R;

   return 0;
}
