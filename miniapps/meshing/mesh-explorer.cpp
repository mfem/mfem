// Copyright (c) 2010-2020, Lawrence Livermore National Security, LLC. Produced
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
//            -----------------------------------------------------
//            Mesh Explorer Miniapp:  Explore and manipulate meshes
//            -----------------------------------------------------
//
// This miniapp is a handy tool to examine, visualize and manipulate a given
// mesh. Some of its features are:
//
//    - visualizing of mesh materials and individual mesh elements
//    - mesh scaling, randomization, and general transformation
//    - manipulation of the mesh curvature
//    - the ability to simulate parallel partitioning
//    - quantitative and visual reports of mesh quality
//
// Compile with: make mesh-explorer
//
// Sample runs:  mesh-explorer
//               mesh-explorer -m ../../data/beam-tri.mesh
//               mesh-explorer -m ../../data/star-q2.mesh
//               mesh-explorer -m ../../data/disc-nurbs.mesh
//               mesh-explorer -m ../../data/escher-p3.mesh
//               mesh-explorer -m ../../data/mobius-strip.mesh

#include "mfem.hpp"
#include <fstream>
#include <limits>
#include <cstdlib>

using namespace mfem;
using namespace std;

// This tranformation can be applied to a mesh with the 't' menu option.
void transformation(const Vector &p, Vector &v)
{
   // simple shear transformation
   double s = 0.1;

   if (p.Size() == 3)
   {
      v(0) = p(0) + s*p(1) + s*p(2);
      v(1) = p(1) + s*p(2) + s*p(0);
      v(2) = p(2);
   }
   else if (p.Size() == 2)
   {
      v(0) = p(0) + s*p(1);
      v(1) = p(1) + s*p(0);
   }
   else
   {
      v = p;
   }
}

// This function is used with the 'r' menu option, sub-option 'l' to refine a
// mesh locally in a region, defined by return values <= region_eps.
double region_eps = 1e-8;
double region(const Vector &p)
{
   const double x = p(0), y = p(1);
   // here we describe the region: (x <= 1/4) && (y >= 0) && (y <= 1)
   return std::max(std::max(x - 0.25, -y), y - 1.0);
}

Mesh *read_par_mesh(int np, const char *mesh_prefix)
{
   Mesh *mesh;
   Array<Mesh *> mesh_array;

   mesh_array.SetSize(np);
   for (int p = 0; p < np; p++)
   {
      ostringstream fname;
      fname << mesh_prefix << '.' << setfill('0') << setw(6) << p;
      ifgzstream meshin(fname.str().c_str());
      if (!meshin)
      {
         cerr << "Can not open mesh file: " << fname.str().c_str()
              << '!' << endl;
         for (p--; p >= 0; p--)
         {
            delete mesh_array[p];
         }
         return NULL;
      }
      mesh_array[p] = new Mesh(meshin, 1, 0);
      // set element and boundary attributes to be the processor number + 1
      if (1)
      {
         for (int i = 0; i < mesh_array[p]->GetNE(); i++)
         {
            mesh_array[p]->GetElement(i)->SetAttribute(p+1);
         }
         for (int i = 0; i < mesh_array[p]->GetNBE(); i++)
         {
            mesh_array[p]->GetBdrElement(i)->SetAttribute(p+1);
         }
      }
   }
   mesh = new Mesh(mesh_array, np);

   for (int p = 0; p < np; p++)
   {
      delete mesh_array[np-1-p];
   }
   mesh_array.DeleteAll();

   return mesh;
}

// Given a 3D mesh, produce a 2D mesh consisting of its boundary elements.
Mesh *skin_mesh(Mesh *mesh)
{
   // Determine mapping from vertex to boundary vertex
   Array<int> v2v(mesh->GetNV());
   v2v = -1;
   for (int i = 0; i < mesh->GetNBE(); i++)
   {
      Element *el = mesh->GetBdrElement(i);
      int *v = el->GetVertices();
      int nv = el->GetNVertices();
      for (int j = 0; j < nv; j++)
      {
         v2v[v[j]] = 0;
      }
   }
   int nbvt = 0;
   for (int i = 0; i < v2v.Size(); i++)
   {
      if (v2v[i] == 0)
      {
         v2v[i] = nbvt++;
      }
   }

   // Create a new mesh for the boundary
   Mesh * bmesh = new Mesh(mesh->Dimension() - 1, nbvt, mesh->GetNBE(),
                           0, mesh->SpaceDimension());

   // Copy vertices to the boundary mesh
   nbvt = 0;
   for (int i = 0; i < v2v.Size(); i++)
   {
      if (v2v[i] >= 0)
      {
         double *c = mesh->GetVertex(i);
         bmesh->AddVertex(c);
         nbvt++;
      }
   }

   // Copy elements to the boundary mesh
   int bv[4];
   for (int i = 0; i < mesh->GetNBE(); i++)
   {
      Element *el = mesh->GetBdrElement(i);
      int *v = el->GetVertices();
      int nv = el->GetNVertices();

      for (int j = 0; j < nv; j++)
      {
         bv[j] = v2v[v[j]];
      }

      switch (el->GetGeometryType())
      {
         case Geometry::SEGMENT:
            bmesh->AddSegment(bv, el->GetAttribute());
            break;
         case Geometry::TRIANGLE:
            bmesh->AddTriangle(bv, el->GetAttribute());
            break;
         case Geometry::SQUARE:
            bmesh->AddQuad(bv, el->GetAttribute());
            break;
         default:
            break; /// This should not happen
      }

   }
   bmesh->FinalizeTopology();

   // Copy GridFunction describing nodes if present
   if (mesh->GetNodes())
   {
      FiniteElementSpace *fes = mesh->GetNodes()->FESpace();
      const FiniteElementCollection *fec = fes->FEColl();
      if (dynamic_cast<const H1_FECollection*>(fec))
      {
         FiniteElementCollection *fec_copy =
            FiniteElementCollection::New(fec->Name());
         FiniteElementSpace *fes_copy =
            new FiniteElementSpace(*fes, bmesh, fec_copy);
         GridFunction *bdr_nodes = new GridFunction(fes_copy);
         bdr_nodes->MakeOwner(fec_copy);

         bmesh->NewNodes(*bdr_nodes, true);

         Array<int> vdofs;
         Array<int> bvdofs;
         Vector v;
         for (int i=0; i<mesh->GetNBE(); i++)
         {
            fes->GetBdrElementVDofs(i, vdofs);
            mesh->GetNodes()->GetSubVector(vdofs, v);

            fes_copy->GetElementVDofs(i, bvdofs);
            bdr_nodes->SetSubVector(bvdofs, v);
         }
      }
      else
      {
         cout << "\nDiscontinuous nodes not yet supported" << endl;
      }
   }

   return bmesh;
}

int main (int argc, char *argv[])
{
   int np = 0;
   const char *mesh_file = "../../data/beam-hex.mesh";
   bool refine = true;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to visualize.");
   args.AddOption(&np, "-np", "--num-proc",
                  "Load mesh from multiple processors.");
   args.AddOption(&refine, "-ref", "--refinement", "-no-ref", "--no-refinement",
                  "Prepare the mesh for refinement or not.");
   args.Parse();
   if (!args.Good())
   {
      if (!args.Help())
      {
         args.PrintError(cout);
         cout << endl;
      }
      cout << "Visualize and manipulate a serial mesh:\n"
           << "   mesh-explorer -m <mesh_file>\n"
           << "Visualize and manipulate a parallel mesh:\n"
           << "   mesh-explorer -np <#proc> -m <mesh_prefix>\n" << endl
           << "All Options:\n";
      args.PrintHelp(cout);
      return 1;
   }
   args.PrintOptions(cout);

   Mesh *mesh;
   Mesh *bdr_mesh = NULL;
   if (np <= 0)
   {
      mesh = new Mesh(mesh_file, 1, refine);
   }
   else
   {
      mesh = read_par_mesh(np, mesh_file);
      if (mesh == NULL)
      {
         return 3;
      }
   }
   int dim  = mesh->Dimension();
   int sdim = mesh->SpaceDimension();

   FiniteElementCollection *bdr_attr_fec = NULL;
   FiniteElementCollection *attr_fec;
   if (dim == 2)
   {
      attr_fec = new Const2DFECollection;
   }
   else
   {
      bdr_attr_fec = new Const2DFECollection;
      attr_fec = new Const3DFECollection;
   }

   int print_char = 1;
   while (1)
   {
      if (print_char)
      {
         cout << endl;
         mesh->PrintCharacteristics();
         cout << "boundary attribs   :";
         for (int i = 0; i < mesh->bdr_attributes.Size(); i++)
         {
            cout << ' ' << mesh->bdr_attributes[i];
         }
         cout << '\n' << "material attribs   :";
         for (int i = 0; i < mesh->attributes.Size(); i++)
         {
            cout << ' ' << mesh->attributes[i];
         }
         cout << endl;
         cout << "mesh curvature     : ";
         if (mesh->GetNodalFESpace() != NULL)
         {
            cout << mesh->GetNodalFESpace()->FEColl()->Name() << endl;
         }
         else
         {
            cout << "NONE" << endl;
         }
      }
      print_char = 0;
      cout << endl;
      cout << "What would you like to do?\n"
           "r) Refine\n"
           "c) Change curvature\n"
           "s) Scale\n"
           "t) Transform\n"
           "j) Jitter\n"
           "v) View\n"
           "m) View materials\n"
           "b) View boundary\n"
           "e) View elements\n"
           "h) View element sizes, h\n"
           "k) View element ratios, kappa\n"
           "x) Print sub-element stats\n"
           "f) Find physical point in reference space\n"
           "p) Generate a partitioning\n"
           "o) Reorder elements\n"
           "S) Save in MFEM format\n"
           "V) Save in VTK format (only linear and quadratic meshes)\n"
           "q) Quit\n"
#ifdef MFEM_USE_ZLIB
           "Z) Save in MFEM format with compression\n"
#endif
           "--> " << flush;
      char mk;
      cin >> mk;
      if (!cin) { break; }

      if (mk == 'q')
      {
         break;
      }

      if (mk == 'r')
      {
         cout <<
              "Choose type of refinement:\n"
              "s) standard refinement with Mesh::UniformRefinement()\n"
              "b) Mesh::UniformRefinement() (bisection for tet meshes)\n"
              "u) uniform refinement with a factor\n"
              "g) non-uniform refinement (Gauss-Lobatto) with a factor\n"
              "l) refine locally using the region() function\n"
              "--> " << flush;
         char sk;
         cin >> sk;
         switch (sk)
         {
            case 's':
               mesh->UniformRefinement();
               // Make sure tet-only meshes are marked for local refinement.
               mesh->Finalize(true);
               break;
            case 'b':
               mesh->UniformRefinement(1); // ref_algo = 1
               break;
            case 'u':
            case 'g':
            {
               cout << "enter refinement factor --> " << flush;
               int ref_factor;
               cin >> ref_factor;
               if (ref_factor <= 1 || ref_factor > 32) { break; }
               int ref_type = (sk == 'u') ? BasisType::ClosedUniform :
                              BasisType::GaussLobatto;
               Mesh *rmesh = new Mesh(mesh, ref_factor, ref_type);
               delete mesh;
               mesh = rmesh;
               break;
            }
            case 'l':
            {
               Vector pt;
               Array<int> marked_elements;
               for (int i = 0; i < mesh->GetNE(); i++)
               {
                  // check all nodes of the element
                  IsoparametricTransformation T;
                  mesh->GetElementTransformation(i, &T);
                  for (int j = 0; j < T.GetPointMat().Width(); j++)
                  {
                     T.GetPointMat().GetColumnReference(j, pt);
                     if (region(pt) <= region_eps)
                     {
                        marked_elements.Append(i);
                        break;
                     }
                  }
               }
               mesh->GeneralRefinement(marked_elements);
               break;
            }
         }
         print_char = 1;
      }

      if (mk == 'c')
      {
         int p;
         cout << "enter new order for mesh curvature --> " << flush;
         cin >> p;
         mesh->SetCurvature(p > 0 ? p : -p, p <= 0);
         print_char = 1;
      }

      if (mk == 's')
      {
         double factor;
         cout << "scaling factor ---> " << flush;
         cin >> factor;

         GridFunction *nodes = mesh->GetNodes();
         if (nodes == NULL)
         {
            for (int i = 0; i < mesh->GetNV(); i++)
            {
               double *v = mesh->GetVertex(i);
               v[0] *= factor;
               v[1] *= factor;
               if (dim == 3)
               {
                  v[2] *= factor;
               }
            }
         }
         else
         {
            *nodes *= factor;
         }

         print_char = 1;
      }

      if (mk == 't')
      {
         mesh->Transform(transformation);
         print_char = 1;
      }

      if (mk == 'j')
      {
         double jitter;
         cout << "jitter factor ---> " << flush;
         cin >> jitter;

         GridFunction *nodes = mesh->GetNodes();

         if (nodes == NULL)
         {
            cerr << "The mesh should have nodes, introduce curvature first!\n";
         }
         else
         {
            FiniteElementSpace *fespace = nodes->FESpace();

            GridFunction rdm(fespace);
            rdm.Randomize();
            rdm -= 0.5; // shift to random values in [-0.5,0.5]
            rdm *= jitter;

            // compute minimal local mesh size
            Vector h0(fespace->GetNDofs());
            h0 = infinity();
            {
               Array<int> dofs;
               for (int i = 0; i < fespace->GetNE(); i++)
               {
                  fespace->GetElementDofs(i, dofs);
                  for (int j = 0; j < dofs.Size(); j++)
                  {
                     h0(dofs[j]) = std::min(h0(dofs[j]), mesh->GetElementSize(i));
                  }
               }
            }

            // scale the random values to be of order of the local mesh size
            for (int i = 0; i < fespace->GetNDofs(); i++)
            {
               for (int d = 0; d < dim; d++)
               {
                  rdm(fespace->DofToVDof(i,d)) *= h0(i);
               }
            }

            char move_bdr = 'n';
            cout << "move boundary nodes? [y/n] ---> " << flush;
            cin >> move_bdr;

            // don't perturb the boundary
            if (move_bdr == 'n')
            {
               Array<int> vdofs;
               for (int i = 0; i < fespace->GetNBE(); i++)
               {
                  fespace->GetBdrElementVDofs(i, vdofs);
                  for (int j = 0; j < vdofs.Size(); j++)
                  {
                     rdm(vdofs[j]) = 0.0;
                  }
               }
            }

            *nodes += rdm;
         }

         print_char = 1;
      }

      if (mk == 'x')
      {
         int sd, nz = 0;
         DenseMatrix J(dim);
         double min_det_J, max_det_J, min_det_J_z, max_det_J_z;
         double min_kappa, max_kappa, max_ratio_det_J_z;
         min_det_J = min_kappa = infinity();
         max_det_J = max_kappa = max_ratio_det_J_z = -infinity();
         cout << "subdivision factor ---> " << flush;
         cin >> sd;
         Array<int> bad_elems_by_geom(Geometry::NumGeom);
         bad_elems_by_geom = 0;
         for (int i = 0; i < mesh->GetNE(); i++)
         {
            Geometry::Type geom = mesh->GetElementBaseGeometry(i);
            ElementTransformation *T = mesh->GetElementTransformation(i);

            RefinedGeometry *RefG = GlobGeometryRefiner.Refine(geom, sd, 1);
            IntegrationRule &ir = RefG->RefPts;

            min_det_J_z = infinity();
            max_det_J_z = -infinity();
            for (int j = 0; j < ir.GetNPoints(); j++)
            {
               T->SetIntPoint(&ir.IntPoint(j));
               Geometries.JacToPerfJac(geom, T->Jacobian(), J);

               double det_J = J.Det();
               double kappa =
                  J.CalcSingularvalue(0) / J.CalcSingularvalue(dim-1);

               min_det_J_z = fmin(min_det_J_z, det_J);
               max_det_J_z = fmax(max_det_J_z, det_J);

               min_kappa = fmin(min_kappa, kappa);
               max_kappa = fmax(max_kappa, kappa);
            }
            max_ratio_det_J_z =
               fmax(max_ratio_det_J_z, max_det_J_z/min_det_J_z);
            min_det_J = fmin(min_det_J, min_det_J_z);
            max_det_J = fmax(max_det_J, max_det_J_z);
            if (min_det_J_z <= 0.0)
            {
               nz++;
               bad_elems_by_geom[geom]++;
            }
         }
         cout << "\nbad elements = " << nz;
         if (nz)
         {
            cout << "  --  ";
            Mesh::PrintElementsByGeometry(dim, bad_elems_by_geom, cout);
         }
         cout << "\nmin det(J)   = " << min_det_J
              << "\nmax det(J)   = " << max_det_J
              << "\nglobal ratio = " << max_det_J/min_det_J
              << "\nmax el ratio = " << max_ratio_det_J_z
              << "\nmin kappa    = " << min_kappa
              << "\nmax kappa    = " << max_kappa << endl;
      }

      if (mk == 'f')
      {
         DenseMatrix point_mat(sdim,1);
         cout << "\npoint in physical space ---> " << flush;
         for (int i = 0; i < sdim; i++)
         {
            cin >> point_mat(i,0);
         }
         Array<int> elem_ids;
         Array<IntegrationPoint> ips;

         // physical -> reference space
         mesh->FindPoints(point_mat, elem_ids, ips);

         cout << "point in reference space:";
         if (elem_ids[0] == -1)
         {
            cout << " NOT FOUND!\n";
         }
         else
         {
            cout << " element " << elem_ids[0] << ", ip =";
            cout << " " << ips[0].x;
            if (sdim > 1)
            {
               cout << " " << ips[0].y;
               if (sdim > 2)
               {
                  cout << " " << ips[0].z;
               }
            }
            cout << endl;
         }
      }

      if (mk == 'o')
      {
         cout << "What type of reordering?\n"
              "g) Gecko edge-product minimization\n"
              "h) Hilbert spatial sort\n"
              "--> " << flush;
         char rk;
         cin >> rk;

         Array<int> ordering, tentative;
         if (rk == 'h')
         {
            mesh->GetHilbertElementOrdering(ordering);
            mesh->ReorderElements(ordering);
         }
         else if (rk == 'g')
         {
            int outer, inner, window, period;
            cout << "Enter number of outer iterations (default 5): " << flush;
            cin >> outer;
            cout << "Enter number of inner iterations (default 4): " << flush;
            cin >> inner;
            cout << "Enter window size (default 4, beware of exponential cost): "
                 << flush;
            cin >> window;
            cout << "Enter period for window size increment (default 2): "
                 << flush;
            cin >> period;

            double best_cost = infinity();
            for (int i = 0; i < outer; i++)
            {
               int seed = i+1;
               double cost = mesh->GetGeckoElementOrdering(
                                tentative, inner, window, period, seed, true);

               if (cost < best_cost)
               {
                  ordering = tentative;
                  best_cost = cost;
               }
            }
            cout << "Final cost: " << best_cost << endl;

            mesh->ReorderElements(ordering);
         }
      }

      // These are the cases that open a new GLVis window
      if (mk == 'm' || mk == 'b' || mk == 'e' || mk == 'v' || mk == 'h' ||
          mk == 'k' || mk == 'p')
      {
         Array<int> bdr_part;
         Array<int> part(mesh->GetNE());
         FiniteElementSpace *bdr_attr_fespace = NULL;
         FiniteElementSpace *attr_fespace =
            new FiniteElementSpace(mesh, attr_fec);
         GridFunction bdr_attr;
         GridFunction attr(attr_fespace);

         if (mk == 'm')
         {
            for (int i = 0; i < mesh->GetNE(); i++)
            {
               part[i] = (attr(i) = mesh->GetAttribute(i)) - 1;
            }
         }

         if (mk == 'b')
         {
            if (dim == 3)
            {
               delete bdr_mesh;
               bdr_mesh = skin_mesh(mesh);
               bdr_attr_fespace =
                  new FiniteElementSpace(bdr_mesh, bdr_attr_fec);
               bdr_part.SetSize(bdr_mesh->GetNE());
               bdr_attr.SetSpace(bdr_attr_fespace);
               for (int i = 0; i < bdr_mesh->GetNE(); i++)
               {
                  bdr_part[i] = (bdr_attr(i) = bdr_mesh->GetAttribute(i)) - 1;
               }
            }
            else
            {
               attr = 1.0;
            }
         }

         if (mk == 'v')
         {
            attr = 1.0;
         }

         if (mk == 'e')
         {
            Array<int> coloring;
            srand(time(0));
            double a = double(rand()) / (double(RAND_MAX) + 1.);
            int el0 = (int)floor(a * mesh->GetNE());
            cout << "Generating coloring starting with element " << el0+1
                 << " / " << mesh->GetNE() << endl;
            mesh->GetElementColoring(coloring, el0);
            for (int i = 0; i < coloring.Size(); i++)
            {
               attr(i) = coloring[i];
            }
            cout << "Number of colors: " << attr.Max() + 1 << endl;
            for (int i = 0; i < mesh->GetNE(); i++)
            {
               // part[i] = i; // checkerboard element coloring
               attr(i) = part[i] = i; // coloring by element number
            }
         }

         if (mk == 'h')
         {
            DenseMatrix J(dim);
            double h_min, h_max;
            h_min = infinity();
            h_max = -h_min;
            for (int i = 0; i < mesh->GetNE(); i++)
            {
               int geom = mesh->GetElementBaseGeometry(i);
               ElementTransformation *T = mesh->GetElementTransformation(i);
               T->SetIntPoint(&Geometries.GetCenter(geom));
               Geometries.JacToPerfJac(geom, T->Jacobian(), J);

               attr(i) = J.Det();
               if (attr(i) < 0.0)
               {
                  attr(i) = -pow(-attr(i), 1.0/double(dim));
               }
               else
               {
                  attr(i) = pow(attr(i), 1.0/double(dim));
               }
               h_min = min(h_min, attr(i));
               h_max = max(h_max, attr(i));
            }
            cout << "h_min = " << h_min << ", h_max = " << h_max << endl;
         }

         if (mk == 'k')
         {
            DenseMatrix J(dim);
            for (int i = 0; i < mesh->GetNE(); i++)
            {
               int geom = mesh->GetElementBaseGeometry(i);
               ElementTransformation *T = mesh->GetElementTransformation(i);
               T->SetIntPoint(&Geometries.GetCenter(geom));
               Geometries.JacToPerfJac(geom, T->Jacobian(), J);
               attr(i) = J.CalcSingularvalue(0) / J.CalcSingularvalue(dim-1);
            }
         }

         if (mk == 'p')
         {
            int *partitioning = NULL, np;
            cout << "What type of partitioning?\n"
                 "c) Cartesian\n"
                 "s) Simple 1D split of the element sequence\n"
                 "0) METIS_PartGraphRecursive (sorted neighbor lists)\n"
                 "1) METIS_PartGraphKway      (sorted neighbor lists)"
                 " (default)\n"
                 "2) METIS_PartGraphVKway     (sorted neighbor lists)\n"
                 "3) METIS_PartGraphRecursive\n"
                 "4) METIS_PartGraphKway\n"
                 "5) METIS_PartGraphVKway\n"
                 "--> " << flush;
            char pk;
            cin >> pk;
            if (pk == 'c')
            {
               int nxyz[3];
               cout << "Enter nx: " << flush;
               cin >> nxyz[0]; np = nxyz[0];
               if (mesh->Dimension() > 1)
               {
                  cout << "Enter ny: " << flush;
                  cin >> nxyz[1]; np *= nxyz[1];
                  if (mesh->Dimension() > 2)
                  {
                     cout << "Enter nz: " << flush;
                     cin >> nxyz[2]; np *= nxyz[2];
                  }
               }
               partitioning = mesh->CartesianPartitioning(nxyz);
            }
            else if (pk == 's')
            {
               cout << "Enter number of processors: " << flush;
               cin >> np;

               partitioning = new int[mesh->GetNE()];
               for (int i = 0; i < mesh->GetNE(); i++)
               {
                  partitioning[i] = i * np / mesh->GetNE();
               }
            }
            else
            {
               int part_method = pk - '0';
               if (part_method < 0 || part_method > 5)
               {
                  continue;
               }
               cout << "Enter number of processors: " << flush;
               cin >> np;
               partitioning = mesh->GeneratePartitioning(np, part_method);
            }
            if (partitioning)
            {
               const char part_file[] = "partitioning.txt";
               ofstream opart(part_file);
               opart << "number_of_elements " << mesh->GetNE() << '\n'
                     << "number_of_processors " << np << '\n';
               for (int i = 0; i < mesh->GetNE(); i++)
               {
                  opart << partitioning[i] << '\n';
               }
               cout << "Partitioning file: " << part_file << endl;

               Array<int> proc_el(np);
               proc_el = 0;
               for (int i = 0; i < mesh->GetNE(); i++)
               {
                  proc_el[partitioning[i]]++;
               }
               int min_el = proc_el[0], max_el = proc_el[0];
               for (int i = 1; i < np; i++)
               {
                  if (min_el > proc_el[i])
                  {
                     min_el = proc_el[i];
                  }
                  if (max_el < proc_el[i])
                  {
                     max_el = proc_el[i];
                  }
               }
               cout << "Partitioning stats:\n"
                    << "           "
                    << setw(12) << "minimum"
                    << setw(12) << "average"
                    << setw(12) << "maximum"
                    << setw(12) << "total" << '\n';
               cout << " elements  "
                    << setw(12) << min_el
                    << setw(12) << double(mesh->GetNE())/np
                    << setw(12) << max_el
                    << setw(12) << mesh->GetNE() << endl;
            }
            else
            {
               continue;
            }

            for (int i = 0; i < mesh->GetNE(); i++)
            {
               attr(i) = part[i] = partitioning[i];
            }
            delete [] partitioning;
         }

         char vishost[] = "localhost";
         int  visport   = 19916;
         socketstream sol_sock(vishost, visport);
         if (sol_sock.is_open())
         {
            sol_sock.precision(14);
            if (sdim == 2)
            {
               sol_sock << "fem2d_gf_data_keys\n";
               if (mk != 'p')
               {
                  mesh->Print(sol_sock);
               }
               else
               {
                  // NURBS meshes do not support PrintWithPartitioning
                  if (mesh->NURBSext)
                  {
                     mesh->Print(sol_sock);
                     for (int i = 0; i < mesh->GetNE(); i++)
                     {
                        attr(i) = part[i];
                     }
                  }
                  else
                  {
                     mesh->PrintWithPartitioning(part, sol_sock, 1);
                  }
               }
               attr.Save(sol_sock);
               sol_sock << "RjlmAb***********";
               if (mk == 'v')
               {
                  sol_sock << "e";
               }
               else
               {
                  sol_sock << "\n";
               }
            }
            else
            {
               sol_sock << "fem3d_gf_data_keys\n";
               if (mk == 'v' || mk == 'h' || mk == 'k')
               {
                  mesh->Print(sol_sock);
               }
               else if (mk == 'b')
               {
                  bdr_mesh->Print(sol_sock);
                  bdr_attr.Save(sol_sock);
                  sol_sock << "mcaaA";
                  // Switch to a discrete color scale
                  sol_sock << "pppppp" << "pppppp" << "pppppp";
               }
               else
               {
                  // NURBS meshes do not support PrintWithPartitioning
                  if (mesh->NURBSext)
                  {
                     mesh->Print(sol_sock);
                     for (int i = 0; i < mesh->GetNE(); i++)
                     {
                        attr(i) = part[i];
                     }
                  }
                  else
                  {
                     mesh->PrintWithPartitioning(part, sol_sock);
                  }
               }
               if (mk != 'b')
               {
                  attr.Save(sol_sock);
                  sol_sock << "maaA";
                  if (mk == 'v')
                  {
                     sol_sock << "aa";
                  }
                  else
                  {
                     sol_sock << "\n";
                  }
               }
            }
            sol_sock << flush;
         }
         else
         {
            cout << "Unable to connect to "
                 << vishost << ':' << visport << endl;
         }
         delete attr_fespace;
         delete bdr_attr_fespace;
      }

      if (mk == 'S')
      {
         const char mesh_file[] = "mesh-explorer.mesh";
         ofstream omesh(mesh_file);
         omesh.precision(14);
         mesh->Print(omesh);
         cout << "New mesh file: " << mesh_file << endl;
      }

      if (mk == 'V')
      {
         const char mesh_file[] = "mesh-explorer.vtk";
         ofstream omesh(mesh_file);
         omesh.precision(14);
         mesh->PrintVTK(omesh);
         cout << "New VTK mesh file: " << mesh_file << endl;
      }

#ifdef MFEM_USE_ZLIB
      if (mk == 'Z')
      {
         const char mesh_file[] = "mesh-explorer.mesh.gz";
         ofgzstream omesh(mesh_file, "zwb9");
         omesh.precision(14);
         mesh->Print(omesh);
         cout << "New mesh file: " << mesh_file << endl;
      }
#endif

   }

   delete bdr_attr_fec;
   delete attr_fec;
   delete bdr_mesh;
   delete mesh;
   return 0;
}
