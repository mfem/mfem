//                                Contact example
//
// Compile with: make contact
//
// Sample runs:  ./contact -m1 block1.mesh -m2 block2.mesh -at "5 6 7 8"
// Sample runs:  ./contact -m1 block1_d.mesh -m2 block2_d.mesh -at "5 6 7 8"

#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include "problems/problems.hpp"
#include "util/util.hpp"

using namespace std;
using namespace mfem;

int main(int argc, char *argv[])
{
   // 1. Parse command-line options.
   const char *mesh_file1 = "meshes/block1.mesh";
   const char *mesh_file2 = "meshes/block2.mesh";
   int order = 1;
   Array<int> attr;
   Array<int> m_attr;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file1, "-m1", "--mesh1",
                  "First mesh file to use.");
   args.AddOption(&mesh_file2, "-m2", "--mesh2",
                  "Second mesh file to use.");
   args.AddOption(&attr, "-at", "--attributes-surf",
                  "Attributes of boundary faces on contact surface for mesh 2.");
   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(cout);
      return 1;
   }
   args.PrintOptions(cout);



   ElasticityProblem prob1(mesh_file1,order);
   ElasticityProblem prob2(mesh_file2,order);

   Mesh * mesh1= prob1.GetMesh();
   Mesh * mesh2= prob2.GetMesh();

   const int dim = prob1.GetMesh()->Dimension();
   MFEM_VERIFY(dim == prob2.GetMesh()->Dimension(), "");


   // boundary attribute 2 is the potential contact surface of nodes
   attr.Append(3);
   // boundary attribute 2 is the potential contact surface for master surface
   m_attr.Append(3);

   cout << "Number of finite element unknowns for mesh1: "
        << prob1.GetFESpace()->GetTrueVSize() << endl;

   GridFunction nodes0 = *mesh1->GetNodes(); // undeformed mesh1 nodal grid function
   GridFunction *nodes1 = mesh1->GetNodes();

   cout << "Number of finite element unknowns for mesh2: "
        << prob2.GetFESpace()->GetTrueVSize() << endl;

   // degrees of freedom of both meshes
   int ndof_1 = prob1.GetNumDofs();
   int ndof_2 = prob2.GetNumDofs();
   int ndofs = ndof_1 + ndof_2;
   // number of nodes for each mesh
   int nnd_1 = mesh1->GetNV();
   int nnd_2 = mesh2->GetNV();
   int nnd = nnd_1 + nnd_2;

   SparseMatrix A1 = prob1.GetOperator();
   SparseMatrix A2 = prob2.GetOperator();
   SparseMatrix K(ndofs,ndofs);
   for (int i=0; i<A1.Height(); i++)
   {
      Array<int> col_tmp;
      Vector v_tmp;
      col_tmp = 0;
      v_tmp = 0.0;
      A1.GetRow(i, col_tmp, v_tmp);
      K.SetRow(i, col_tmp, v_tmp);
   }
   for (int i=0; i<A2.Height(); i++)
   {
      Array<int> col_tmp;
      Vector v_tmp;
      col_tmp = 0;
      v_tmp = 0.0;
      A2.GetRow(i, col_tmp, v_tmp);
      for (int j=0; j<col_tmp.Size(); j++)
      {
         col_tmp[j] += ndof_1;
      }
      K.SetRow(i+ndof_1, col_tmp, v_tmp);  // mesh1 top left corner
   }

   // Construct node to segment contact constraint.

   attr.Sort();
   cout << "Boundary attributes for contact surface faces in mesh 2" << endl;
   for (auto a : attr) { cout << a << endl; }

   Array<int> bdryFaces2;  // TODO: remove this?

   std::set<int> bdryVerts2;
   for (int b=0; b<mesh2->GetNBE(); ++b)
   {
      if (attr.FindSorted(mesh2->GetBdrAttribute(b)) >= 0)
      {
         bdryFaces2.Append(b);
         Array<int> vert;
         mesh2->GetBdrElementVertices(b, vert);
         for (auto v : vert)
         {
            bdryVerts2.insert(v);
         }
      }
   }

   int npoints = bdryVerts2.size();
   Array<int> s_conn(npoints); // connectivity of the second/slave mesh
   Vector xyz(dim * npoints);
   xyz = 0.0;

   cout << "Boundary vertices for contact surface vertices in mesh 2" << endl;

   // construct the nodal coordinates on mesh2 to be projected, including displacement
   int count = 0;
   GridFunction x2 = prob2.GetGridFunction();
   for (auto v : bdryVerts2)
   {
      for (int i=0; i<dim; ++i)
      {
         xyz[count + (i * npoints)] = mesh2->GetVertex(v)[i] + x2[v*dim+i];
      }

      s_conn[count] = v + nnd_1; // dof1 is the master
      count++;
   }

   MFEM_VERIFY(count == npoints, "");

   // gap function
   Vector g(npoints*dim);
   g = -1.0;
   // segment reference coordinates of the closest point
   Vector m_xi(npoints*(dim-1));
   m_xi = -1.0;
   Vector xs(dim*npoints);
   xs = 0.0;
   for (int i=0; i<npoints; i++)
   {
      for (int j=0; j<dim; j++)
      {
         xs[i*dim+j] = xyz[i + (j*npoints)];
      }
   }

   Array<int> m_conn(
      npoints*4); // only works for linear elements that have 4 vertices!
   DenseMatrix coordsm(npoints*4, dim);

   // adding displacement to mesh1 using a fixed grid function from mesh1
   GridFunction x1 = prob1.GetGridFunction();
   x1 = 1e-4; // x1 order: [xyz xyz... xyz]
   add(nodes0, x1, *nodes1);

   FindPointsInMesh(*mesh1, xyz, m_conn, m_xi);

   for (int i=0; i<npoints; i++)
   {
      for (int j=0; j<4; j++)
      {
         for (int k=0; k<dim; k++)
         {
            coordsm(i*4+j,k) = mesh1->GetVertex(m_conn[i*4+j])[k]+x1[dim*m_conn[i*4+j]+k];
         }
      }
   }
   SparseMatrix M(nnd,ndofs);
   Array<SparseMatrix *> dM(npoints);
   for (int i = 0; i<npoints; i++)
   {
      dM[i] = new SparseMatrix(ndofs,ndofs);
   }
   Assemble_Contact(nnd, xs, m_xi, coordsm,
                    s_conn, m_conn, g, M, dM);

   std::set<int> dirbdryv2;
   for (int b=0; b<mesh2->GetNBE(); ++b)
   {
      if (mesh2->GetBdrAttribute(b) == 1)
      {
         Array<int> vert;
         mesh2->GetBdrElementVertices(b, vert);
         for (auto v : vert)
         {
            dirbdryv2.insert(v);
         }
      }
   }
   std::set<int> dirbdryv1;
   for (int b=0; b<mesh1->GetNBE(); ++b)
   {
      if (mesh1->GetBdrAttribute(b) == 1)
      {
         Array<int> vert;
         mesh1->GetBdrElementVertices(b, vert);
         for (auto v : vert)
         {
            dirbdryv1.insert(v);
         }
      }
   }

   Array<int> Dirichlet_dof;
   Array<double> Dirichlet_val;

   for (auto v : dirbdryv2)
   {
      for (int i=0; i<dim; ++i)
      {
         Dirichlet_dof.Append(v*dim + i + ndof_1);
         Dirichlet_val.Append(0.);
      }
   }
   double delta = 0.1;
   for (auto v : dirbdryv1)
   {
      Dirichlet_dof.Append(v*dim + 0);
      Dirichlet_val.Append(delta);
      Dirichlet_dof.Append(v*dim + 1);
      Dirichlet_val.Append(0.);
      Dirichlet_dof.Append(v*dim + 2);
      Dirichlet_val.Append(0.);
   }
   return 0;
}
