//                                Contact example
//
// Compile with: make contact
//
// Sample runs:  ./contact -m1 block1.mesh -m2 block2.mesh -at "5 6 7 8"
// Sample runs:  ./contact -m1 block1_d.mesh -m2 block2_d.mesh -at "5 6 7 8"

#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include "util/util.hpp"
#include "util/contact_util.hpp"

using namespace std;
using namespace mfem;


// Coordinates in xyz are assumed to be ordered as [X, Y, Z]
// where X is the list of x-coordinates for all points and so on.
// conn: connectivity of the target surface elements
// xi: surface reference cooridnates for the cloest point, involves a linear transformation from [0,1] to [-1,1]
void FindPointsInMesh(Mesh & mesh, Vector const& xyz, Array<int>& conn, Vector& xi)
{
   const int dim = mesh.Dimension();
   const int np = xyz.Size() / dim;

   MFEM_VERIFY(np * dim == xyz.Size(), "");

   mesh.EnsureNodes();

   //FindPointsGSLIB finder(MPI_COMM_WORLD);
   FindPointsGSLIB finder;

   finder.SetDistanceToleranceForPointsFoundOnBoundary(0.5);

   const double bb_t = 0.5;
   finder.Setup(mesh, bb_t);

   finder.FindPoints(xyz);

   /// Return code for each point searched by FindPoints: inside element (0), on
   /// element boundary (1), or not found (2).
   Array<unsigned int> codes = finder.GetCode();

   /// Return element number for each point found by FindPoints.
   Array<unsigned int> elems = finder.GetElem();

   /// Return reference coordinates for each point found by FindPoints.
   Vector refcrd = finder.GetReferencePosition();

   /// Return distance between the sought and the found point in physical space,
   /// for each point found by FindPoints.
   Vector dist = finder.GetDist();

   MFEM_VERIFY(dist.Size() == np, "");
   MFEM_VERIFY(refcrd.Size() == np * dim, "");
   MFEM_VERIFY(elems.Size() == np, "");
   MFEM_VERIFY(codes.Size() == np, "");

   bool allfound = true;
   for (auto code : codes)
      if (code == 2) { allfound = false; }

   MFEM_VERIFY(allfound, "A point was not found");

   cout << "Maximum distance of projected points: " << dist.Max() << endl;

   // extract information
   for (int i=0; i<np; ++i)
   {
      int refFace, refNormal, refNormalSide;
      bool is_interior = -1;
      Vector normal = GetNormalVector(mesh, elems[i], refcrd.GetData() + (i*dim),
                                      refFace, refNormal, is_interior);

      int phyFace;
      if (is_interior)
      {
         phyFace = -1; // the id of the face that has the closest point
         FindSurfaceToProject(mesh, elems[i], phyFace);

         Array<int> cbdrVert;
         mesh.GetFaceVertices(phyFace, cbdrVert);
         Vector xs(dim);
         xs[0] = xyz[i + 0*np];
         xs[1] = xyz[i + 1*np];
         xs[2] = xyz[i + 2*np];

         Vector xi_tmp(dim-1);
         // get nodes!
         GridFunction *nodes = mesh.GetNodes();
         DenseMatrix coords(4,3);
         for (int i=0; i<4; i++)
         {
            for (int j=0; j<3; j++)
            {
               coords(i,j) = (*nodes)[cbdrVert[i]*3+j];
            }
         }
         SlaveToMaster(coords, xs, xi_tmp);

         for (int j=0; j<dim-1; ++j)
         {
            xi[i*(dim-1)+j] = xi_tmp[j];
         }
         // now get get the projection to the surface
      }
      else
      {
         Vector faceRefCrd(dim-1);
         {
            int fd = 0;
            for (int j=0; j<dim; ++j)
            {
               if (j == refNormal)
               {
                  refNormalSide = (refcrd[(i*dim) + j] > 0.5);
               }
               else
               {
                  faceRefCrd[fd] = refcrd[(i*dim) + j];
                  fd++;
               }
            }
            MFEM_VERIFY(fd == dim-1, "");
         }

         for (int j=0; j<dim-1; ++j)
         {
            xi[i*(dim-1)+j] = faceRefCrd[j]*2.0 - 1.0;
         }
      }

      // Get the element face
      Array<int> faces;
      Array<int> ori;
      int face;

      if (is_interior)
      {
         face = phyFace;
      }
      else
      {
         mesh.GetElementFaces(elems[i], faces, ori);
         face = faces[refFace];
      }

      Array<int> faceVert;
      mesh.GetFaceVertices(face, faceVert);

      for (int p=0; p<4; p++)
      {
         conn[4*i+p] = faceVert[p];
      }
   }

   int sz = xi.Size()/2;
   for (int i = 0; i<sz; i++)
   {
      mfem::out << "\ni = " << i << ", ξᵢ = ("<<xi[i*(dim-1)]<<","<<xi[i*(dim-1)+1]<<"): -> \n";
      for (int j = 0; j<4; j++)
      {
         PrintVertex(&mesh,conn[4*i+j]);
      }
      mfem::out << endl;
   }
}

int main(int argc, char *argv[])
{
   // 1. Parse command-line options.
   const char *mesh_file1 = "meshes/block1.mesh";
   const char *mesh_file2 = "meshes/block2.mesh";

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

   Mesh mesh1(mesh_file1, 1, 1);
   Mesh mesh2(mesh_file2, 1, 1);

   const int dim = mesh1.Dimension();
   MFEM_VERIFY(dim == mesh2.Dimension(), "");

   // boundary attribute 2 is the potential contact surface of nodes
   attr.Append(2);
   // boundary attribute 2 is the potential contact surface for master surface
   m_attr.Append(2);

   //  Define a finite element space on the mesh. Here we use vector finite
   //  elements, i.e. dim copies of a scalar finite element space. The vector
   //  dimension is specified by the last argument of the FiniteElementSpace
   //  constructor.
   FiniteElementCollection *fec1;
   FiniteElementSpace *fespace1;
   fec1 = new H1_FECollection(1, dim);
   fespace1 = new FiniteElementSpace(&mesh1, fec1, dim, Ordering::byVDIM);
   cout << "Number of finite element unknowns for mesh1: "
        << fespace1->GetTrueVSize() << endl;
   mesh1.SetNodalFESpace(fespace1);
   GridFunction nodes0 = *mesh1.GetNodes(); // undeformed mesh1 nodal grid function
   GridFunction *nodes1 = mesh1.GetNodes();

   FiniteElementCollection *fec2 = new H1_FECollection(1, dim);
   FiniteElementSpace *fespace2 = new FiniteElementSpace(&mesh2, fec2, dim,
                                                         Ordering::byVDIM);
   cout << "Number of finite element unknowns for mesh2: "
        << fespace2->GetTrueVSize() << endl;

   // degrees of freedom of both meshes
   int ndof_1 = fespace1->GetTrueVSize();
   int ndof_2 = fespace2->GetTrueVSize();
   int ndofs = ndof_1 + ndof_2;
   // number of nodes for each mesh
   int nnd_1 = mesh1.GetNV();
   int nnd_2 = mesh2.GetNV();
   int nnd = nnd_1 + nnd_2;
   // Determine the list of true (i.e. conforming) essential boundary dofs.
   // In this example, the boundary conditions are defined by marking only
   // boundary attribute 1 from the mesh as essential and converting it to a
   // list of true dofs.
   Array<int> ess_tdof_list1, ess_bdr1(mesh1.bdr_attributes.Max());
   ess_bdr1 = 0;
   //ess_bdr1[0] = 1;
   // Not ready to be passed on yet
   // fespace->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);

   Array<int> ess_tdof_list2, ess_bdr2(mesh2.bdr_attributes.Max());
   ess_bdr2 = 0;
   //ess_bdr2[0] = 1;

   // Define the displacement vector x as a finite element grid function
   // corresponding to fespace. GridFunction is a derived class of Vector.
   GridFunction x1(fespace1);
   x1 = 0.0;
   GridFunction x2(fespace2);
   x2 = 0.0;

   // Generate force
   LinearForm *b1 = new LinearForm(fespace1);
   b1->Assemble();

   LinearForm *b2 = new LinearForm(fespace2);
   b2->Assemble();

   // Set up the bilinear form a(.,.) on the finite element space
   //  corresponding to the linear elasticity integrator with piece-wise
   //  constants coefficient lambda and mu.
   Vector lambda1(mesh1.attributes.Max());
   lambda1 = 57.6923076923;
   PWConstCoefficient lambda1_func(lambda1);
   Vector mu1(mesh1.attributes.Max());
   mu1 = 38.4615384615;
   PWConstCoefficient mu1_func(mu1);

   BilinearForm *a1 = new BilinearForm(fespace1);
   a1->AddDomainIntegrator(new ElasticityIntegrator(lambda1_func,mu1_func));

   Vector lambda2(mesh2.attributes.Max());
   lambda2 = 57.6923076923;
   PWConstCoefficient lambda2_func(lambda2);
   Vector mu2(mesh2.attributes.Max());
   mu2 = 38.4615384615;
   PWConstCoefficient mu2_func(mu2);

   BilinearForm *a2 = new BilinearForm(fespace2);
   a2->AddDomainIntegrator(new ElasticityIntegrator(lambda2_func,mu2_func));

   a1->Assemble();
   SparseMatrix A1;
   Vector B1, X1;
   a1->FormLinearSystem(ess_tdof_list1, x1, *b1, A1, X1, B1);

   a2->Assemble();
   SparseMatrix A2;
   Vector B2, X2;

   a2->FormLinearSystem(ess_tdof_list2, x2, *b2, A2, X2, B2);
   // Combine elasticity operator for two meshes into one.
   // Block Matrix
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
   for (int b=0; b<mesh2.GetNBE(); ++b)
   {
      if (attr.FindSorted(mesh2.GetBdrAttribute(b)) >= 0)
      {
         bdryFaces2.Append(b);
         Array<int> vert;
         mesh2.GetBdrElementVertices(b, vert);
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
   for (auto v : bdryVerts2)
   {
      // cout << v << ": " << mesh2.GetVertex(v)[0] << ", "
      //      << mesh2.GetVertex(v)[1] << ", "
      //      << mesh2.GetVertex(v)[2] << endl;

      for (int i=0; i<dim; ++i)
      {
         xyz[count + (i * npoints)] = mesh2.GetVertex(v)[i] + x2[v*dim+i];
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
   x1 = 0.0; // x1 order: [xyz xyz... xyz]
   add(nodes0, x1, *nodes1);

   FindPointsInMesh(mesh1, xyz, m_conn, m_xi);

   for (int i=0; i<npoints; i++)
   {
      for (int j=0; j<4; j++)
      {
         for (int k=0; k<dim; k++)
         {
            coordsm(i*4+j,k) = mesh1.GetVertex(m_conn[i*4+j])[k]+x1[dim*m_conn[i*4+j]+k];
         }
      }
   }

   SparseMatrix M(nnd,ndofs);
   std::vector<SparseMatrix> dM(nnd, SparseMatrix(ndofs,ndofs));

   Assemble_Contact(nnd, npoints, xs, m_xi, coordsm,
                    s_conn, m_conn, g, M, dM);

   std::set<int> dirbdryv2;
   for (int b=0; b<mesh2.GetNBE(); ++b)
   {
      if (mesh2.GetBdrAttribute(b) == 1)
      {
         Array<int> vert;
         mesh2.GetBdrElementVertices(b, vert);
         for (auto v : vert)
         {
            dirbdryv2.insert(v);
         }
      }
   }
   std::set<int> dirbdryv1;
   for (int b=0; b<mesh1.GetNBE(); ++b)
   {
      if (mesh1.GetBdrAttribute(b) == 1)
      {
         Array<int> vert;
         mesh1.GetBdrElementVertices(b, vert);
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
   //M.Print();
   /*Vector eps(ndofs);
   Vector sol(ndofs); sol = 0.;
   for(int i=0;i<ndofs;i++) eps[i] = 1e-5 * i ;
   for(int i=0;i<9;i++)
   {
     cout<<i<<endl;
     dM[s_conn[i]].Mult(eps,sol);
     sol.Print();
   }
   */
   return 0;
}
