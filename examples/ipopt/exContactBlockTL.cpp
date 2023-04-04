//                                Contact example
//
// Compile with: make contact
//
// Sample runs:  ./contact -m1 block1.mesh -m2 block2.mesh -at "5 6 7 8"
// Sample runs:  ./contact -m1 block1_d.mesh -m2 block2_d.mesh -at "5 6 7 8"

#include <fstream>
#include <iostream>
#include <array>

#include "mfem.hpp"
#include "nodepair.hpp"

#include "IpIpoptApplication.hpp"
#include "IpSolveStatistics.hpp"
#include "exContactBlockTL.hpp"

using namespace std;
using namespace mfem;

bool ifequalarray(const Array<int> a1, const Array<int> a2)
{
   if (a1.Size()!=a2.Size())
   {
      return false;
   }
   for (int i=0; i<a1.Size(); i++)
   {
      if (a1[i] != a2[i])
      {
         return false;
      }
   }
   return true;
}

void FindSurfaceToProject(Mesh& mesh, const int elem, int& cbdrface)
{
   Array<int> attr;
   attr.Append(2);
   Array<int> faces;
   Array<int> ori;
   std::vector<Array<int> > facesVertices;
   std::vector<int > faceid;
   mesh.GetElementFaces(elem, faces, ori);
   int face = -1;
   for (int i=0; i<faces.Size(); i++)
   {
      face = faces[i];
      Array<int> faceVert;
      if (!mesh.FaceIsInterior(face)) // if on the boundary
      {
         mesh.GetFaceVertices(face, faceVert);
         faceVert.Sort();
         facesVertices.push_back(faceVert);
         faceid.push_back(face);
      }
   }
   int bdrface = facesVertices.size();

   Array<int> bdryFaces;
   // This shoulnd't need to be rebuilt
   std::vector<Array<int> > bdryVerts;
   for (int b=0; b<mesh.GetNBE(); ++b)
   {
      if (attr.FindSorted(mesh.GetBdrAttribute(b)) >= 0)  // found the contact surface
      {
         bdryFaces.Append(b);
         Array<int> vert;
         mesh.GetBdrElementVertices(b, vert);
         vert.Sort();
         bdryVerts.push_back(vert);
      }
   }

   int bdrvert = bdryVerts.size();
   cbdrface = -1;  // the face number of the contact surface element
   int count_cbdrface = 0;  // the number of matching surfaces, used for checks

   for (int i=0; i<bdrface; i++)
   {
      for (int j=0; j<bdrvert; j++)
      {
         if (ifequalarray(facesVertices[i], bdryVerts[j]))
         {
            cbdrface = faceid[i];
            count_cbdrface += 1;
         }
      }
   }
   MFEM_VERIFY(count_cbdrface == 1,"projection surface not found");

};

mfem::Vector GetNormalVector(Mesh & mesh, const int elem, const double *ref,
                             int & refFace, int & refNormal, bool & interior)
{
   ElementTransformation *trans = mesh.GetElementTransformation(elem);
   const int dim = mesh.Dimension();
   const int spaceDim = trans->GetSpaceDim();

   MFEM_VERIFY(spaceDim == 3, "");

   mfem::Vector n(spaceDim);

   IntegrationPoint ip;
   ip.Set(ref, dim);

   trans->SetIntPoint(&ip);
   //CalcOrtho(trans->Jacobian(), n);  // Works only for face transformations
   const DenseMatrix jac = trans->Jacobian();

   int dimNormal = -1;
   int normalSide = -1;

   const double tol = 1.0e-8;
   for (int i=0; i<dim; ++i)
   {
      const double d0 = std::abs(ref[i]);
      const double d1 = std::abs(ref[i] - 1.0);

      const double d = std::min(d0, d1);
      // TODO: this works only for hexahedral meshes!

      if (d < tol)
      {
         MFEM_VERIFY(dimNormal == -1, "");
         dimNormal = i;

         if (d0 < tol)
         {
            normalSide = 0;
         }
         else
         {
            normalSide = 1;
         }
      }
   }
   // closest point on the boundary
   if (dimNormal < 0 || normalSide < 0) // node is inside the element
   {
      interior = 1;
      mfem::Vector n(3);
      n = 0.0;
      return n;
   }

   MFEM_VERIFY(dimNormal >= 0 && normalSide >= 0, "");
   refNormal = dimNormal;

   MFEM_VERIFY(dim == 3, "");

   {
      // Find the reference face
      if (dimNormal == 0)
      {
         refFace = (normalSide == 1) ? 2 : 4;
      }
      else if (dimNormal == 1)
      {
         refFace = (normalSide == 1) ? 3 : 1;
      }
      else
      {
         refFace = (normalSide == 1) ? 5 : 0;
      }
   }

   std::vector<mfem::
   Vector> tang(2);

   int tangDir[2] = {-1, -1};
   {
      int t = 0;
      for (int i=0; i<dim; ++i)
      {
         if (i != dimNormal)
         {
            tangDir[t] = i;
            t++;
         }
      }

      MFEM_VERIFY(t == 2, "");
   }

   for (int i=0; i<2; ++i)
   {
      tang[i].SetSize(3);

      mfem::Vector tangRef(3);
      tangRef = 0.0;
      tangRef[tangDir[i]] = 1.0;

      jac.Mult(tangRef, tang[i]);
   }

   mfem::Vector c(3);  // Cross product

   c[0] = (tang[0][1] * tang[1][2]) - (tang[0][2] * tang[1][1]);
   c[1] = (tang[0][2] * tang[1][0]) - (tang[0][0] * tang[1][2]);
   c[2] = (tang[0][0] * tang[1][1]) - (tang[0][1] * tang[1][0]);

   c /= c.Norml2();

   mfem::Vector nref(3);
   nref = 0.0;
   nref[dimNormal] = 1.0;

   mfem::Vector ndir(3);
   jac.Mult(nref, ndir);

   ndir /= ndir.Norml2();

   const double dp = ndir * c;

   // TODO: eliminate c?
   n = c;
   if (dp < 0.0)
   {
      n *= -1.0;
   }
   interior = 0;
   return n;
}

// WARNING: global variable, just for this little example.
std::array<std::array<int, 3>, 8> HEX_VERT =
{
   {  {0,0,0},
      {1,0,0},
      {1,1,0},
      {0,1,0},
      {0,0,1},
      {1,0,1},
      {1,1,1},
      {0,1,1}
   }
};

int GetHexVertex(int cdim, int c, int fa, int fb, mfem::Vector & refCrd)
{
   int ref[3];
   ref[cdim] = c;
   ref[cdim == 0 ? 1 : 0] = fa;
   ref[cdim == 2 ? 1 : 2] = fb;

   for (int i=0; i<3; ++i) { refCrd[i] = ref[i]; }

   int refv = -1;

   for (int i=0; i<8; ++i)
   {
      bool match = true;
      for (int j=0; j<3; ++j)
      {
         if (ref[j] != HEX_VERT[i][j]) { match = false; }
      }

      if (match) { refv = i; }
   }

   MFEM_VERIFY(refv >= 0, "");

   return refv;
}

// Coordinates in xyz are assumed to be ordered as [X, Y, Z]
// where X is the list of x-coordinates for all points and so on.
// conn: connectivity of the target surface elements
// xi: surface reference cooridnates for the cloest point, involves a linear transformation from [0,1] to [-1,1]
void FindPointsInMesh(Mesh & mesh, mfem::Vector const& xyz, Array<int>& conn,
                      mfem::Vector& xi)
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
   mfem::Vector refcrd = finder.GetReferencePosition();

   /// Return distance between the sought and the found point in physical space,
   /// for each point found by FindPoints.
   mfem::Vector dist = finder.GetDist();

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
      mfem::Vector normal = GetNormalVector(mesh, elems[i],
                                            refcrd.GetData() + (i*dim),
                                            refFace, refNormal, is_interior);
      int phyFace;
      if (is_interior)
      {
         phyFace = -1; // the id of the face that has the closest point
         FindSurfaceToProject(mesh, elems[i], phyFace);

         Array<int> cbdrVert;
         mesh.GetFaceVertices(phyFace, cbdrVert);
         mfem::Vector xs(dim);
         xs[0] = xyz[i + 0*np];
         xs[1] = xyz[i + 1*np];
         xs[2] = xyz[i + 2*np];
         mfem::Vector xi_tmp(dim-1);
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
         mfem::Vector faceRefCrd(dim-1);
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
}


/* Constructor. */
ExContactBlockTL::ExContactBlockTL(int argc, char *argv[])
   :
   mesh1{nullptr},
   mesh2{nullptr},
   fec1{nullptr},
   fec2{nullptr},
   fespace1{nullptr},
   fespace2{nullptr},
   nodes0{nullptr},
   nodes1{nullptr},
   nodes2{nullptr},
   x1{nullptr},
   x2{nullptr},
   b1{nullptr},
   b2{nullptr},
   lambda1_func{nullptr},
   lambda2_func{nullptr},
   mu1_func{nullptr},
   mu2_func{nullptr},
   a1{nullptr},
   a2{nullptr},
   K{nullptr},
   coordsm{nullptr},
   M{nullptr},
   dM{nullptr}
{
   // 1. Parse command-line options.
   mesh_file1 = "block1.mesh";
   mesh_file2 = "block2.mesh";
   const char *mf1 = mesh_file1.c_str();
   const char *mf2 = mesh_file2.c_str();

   OptionsParser args(argc, argv);
   args.AddOption(&mf1, "-m1", "--mesh1",
                  "First mesh file to use.");
   args.AddOption(&mf2, "-m2", "--mesh2",
                  "Second mesh file to use.");
   args.AddOption(&attr, "-at", "--attributes-surf",
                  "Attributes of boundary faces on contact surface for mesh 2.");
   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(cout);
      assert(0 && "Error in ExContactBlockTL::ExContactBlockTL.");
   }
   args.PrintOptions(cout);

   mesh1 = new Mesh(mf1, 1, 1);
   mesh2 = new Mesh(mf2, 1, 1);

   dim = mesh1->Dimension();
   MFEM_VERIFY(dim == mesh2->Dimension(), "");

   // boundary attribute 2 is the potential contact surface of nodes
   attr.Append(2);
   // boundary attribute 2 is the potential contact surface for master surface
   m_attr.Append(2);

   fec1 = new H1_FECollection(1, dim);
   fespace1 = new FiniteElementSpace(mesh1, fec1, dim, Ordering::byVDIM);
   cout << "Number of finite element unknowns for mesh1: "
        << fespace1->GetTrueVSize() << endl;
   mesh1->SetNodalFESpace(fespace1);

   fec2 = new H1_FECollection(1, dim);
   fespace2 = new FiniteElementSpace(mesh2, fec2, dim, Ordering::byVDIM);
   cout << "Number of finite element unknowns for mesh2: "
        << fespace2->GetTrueVSize() << endl;

   nodes0 = mesh1->GetNodes();
   nodes1 = mesh1->GetNodes();

   // degrees of freedom of both meshes
   ndof_1 = fespace1->GetTrueVSize();
   ndof_2 = fespace2->GetTrueVSize();
   ndofs = ndof_1 + ndof_2;
   // number of nodes for each mesh
   nnd_1 = mesh1->GetNV();
   nnd_2 = mesh2->GetNV();
   nnd = nnd_1 + nnd_2;

   Array<int> ess_bdr1(mesh1->bdr_attributes.Max());
   cout<<mesh1->bdr_attributes.Max()<<endl;
   ess_bdr1 = 0;
   Array<int> ess_bdr2(mesh2->bdr_attributes.Max());
   ess_bdr2 = 0;

   x1 = new GridFunction(fespace1);
   x2 = new GridFunction(fespace2);
   (*x1) = 0.0;
   (*x2) = 0.0;

   b1 = new LinearForm(fespace1);
   b2 = new LinearForm(fespace2);
   b1->Assemble();
   b2->Assemble();

   lambda1.SetSize(mesh1->attributes.Max());
   mu1.SetSize(mesh1->attributes.Max());
   lambda1 = 57.6923076923;
   mu1 = 38.4615384615;
   lambda1_func = new PWConstCoefficient(lambda1);
   mu1_func = new PWConstCoefficient(mu1);

   lambda2.SetSize(mesh2->attributes.Max());
   mu2.SetSize(mesh2->attributes.Max());
   lambda2 = 57.6923076923;
   mu2 = 38.4615384615;
   lambda2_func = new PWConstCoefficient(lambda2);
   mu2_func = new PWConstCoefficient(mu2);

   a1 = new BilinearForm(fespace1);
   a1->AddDomainIntegrator(new ElasticityIntegrator(*lambda1_func, *mu1_func));

   a2 = new BilinearForm(fespace2);
   a2->AddDomainIntegrator(new ElasticityIntegrator(*lambda2_func, *mu2_func));

   a1->Assemble();
   a1->FormLinearSystem(ess_tdof_list1, *x1, *b1, A1, X1, B1);

   a2->Assemble();
   a2->FormLinearSystem(ess_tdof_list2, *x2, *b2, A2, X2, B2);
   A1.Print();
   K = new SparseMatrix(ndofs, ndofs);
   for (int i=0; i<A1.Height(); i++)
   {
      Array<int> col_tmp;
      mfem::Vector v_tmp;
      col_tmp = 0;
      v_tmp = 0.0;
      A1.GetRow(i, col_tmp, v_tmp);
      K->SetRow(i, col_tmp, v_tmp);
   }
   for (int i=0; i<A2.Height(); i++)
   {
      Array<int> col_tmp;
      mfem::Vector v_tmp;
      col_tmp = 0;
      v_tmp = 0.0;
      A2.GetRow(i, col_tmp, v_tmp);
      for (int j=0; j<col_tmp.Size(); j++)
      {
         col_tmp[j] += ndof_1;
      }
      K->SetRow(i+ndof_1, col_tmp, v_tmp);  // mesh1 top left corner
      for (int j=0; j<col_tmp.Size(); j++)
      {
         col_tmp[j] -= ndof_1;
      }
   }
   K->Finalize(1,false);
   //   K->Threshold(1e-19, false);
   // Construct node to segment contact constraint.
   attr.Sort();

   for (int b=0; b<mesh2->GetNBE(); ++b)
   {
      if (attr.FindSorted(mesh2->GetBdrAttribute(b)) >= 0)
      {
         Array<int> vert;
         mesh2->GetBdrElementVertices(b, vert);
         for (auto v : vert)
         {
            bdryVerts2.insert(v);
         }
      }
   }

   npoints = bdryVerts2.size();
   s_conn.SetSize(npoints);
   xyz.SetSize(dim * npoints);
   xyz = 0.0;

   cout << "Boundary vertices for contact surface vertices in mesh 2" << endl;

   // construct the nodal coordinates on mesh2 to be projected, including displacement
   int count = 0;
   for (auto v : bdryVerts2)
   {
      for (int i=0; i<dim; ++i)
      {
         xyz[count + (i * npoints)] = mesh2->GetVertex(v)[i] + (*x2)[v*dim+i];
      }

      s_conn[count] = v + nnd_1; // dof1 is the master
      count++;
   }

   MFEM_VERIFY(count == npoints, "");

   // gap function
   g.SetSize(npoints*dim);
   g = -1.0;
   // segment reference coordinates of the closest point
   m_xi.SetSize(npoints*(dim-1));
   m_xi = -1.0;
   xs.SetSize(dim*npoints);
   xs = 0.0;
   for (int i=0; i<npoints; i++)
   {
      for (int j=0; j<dim; j++)
      {
         xs[i*dim+j] = xyz[i + (j*npoints)];
      }
   }

   m_conn.SetSize(
      npoints*4); // only works for linear elements that have 4 vertices!
   coordsm = new DenseMatrix(npoints*4, dim);

   // adding displacement to mesh1 using a fixed grid function from mesh1
   (*x1) = 0.0; // x1 order: [xyz xyz... xyz]
   add(*nodes0, *x1, *nodes1);

   FindPointsInMesh(*mesh1, xyz, m_conn, m_xi);

   for (int i=0; i<npoints; i++)
   {
      for (int j=0; j<4; j++)
      {
         for (int k=0; k<dim; k++)
         {
            (*coordsm)(i*4+j,k) = mesh1->GetVertex(m_conn[i*4+j])[k]+
                                  (*x1)[dim*m_conn[i*4+j]+k];
         }
      }
   }

   //coordsm.Print();
   M = new SparseMatrix(nnd,ndofs);
   dM = new std::vector<SparseMatrix>(nnd, SparseMatrix(ndofs,ndofs));

   Assemble_Contact(nnd, npoints, ndofs, xs, m_xi, *coordsm, s_conn, m_conn, g, *M,
                    *dM);
   M->Finalize(1,false);
   //   M->Threshold(1e-19, false);

   for (int i=0; i<nnd; i++)
   {
      (*dM)[i].Finalize(1,false);
      //      (*dM)[i].Threshold(1e-19, false);
   }
   assert(M);


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
}

ExContactBlockTL::~ExContactBlockTL()
{
   delete mesh1;
   delete mesh2;
   delete fec1;
   delete fec2;
   delete fespace1;
   delete fespace2;
   delete x1;
   delete x2;
   delete b1;
   delete b2;
   delete lambda1_func;
   delete lambda2_func;
   delete mu1_func;
   delete mu2_func;
   delete a1;
   delete a2;
   delete K;
   delete coordsm;
   delete M;
   delete dM;
}

void ExContactBlockTL::update_g()
{
   int count = 0;
   for (auto v : bdryVerts2)
   {
      for (int i=0; i<dim; ++i)
      {
         xyz[count + (i * npoints)] = mesh2->GetVertex(v)[i] + (*x2)[v*dim+i];
      }
      count++;
   }
   MFEM_VERIFY(count == npoints, "");

   xs = 0.0;
   for (int i=0; i<npoints; i++)
   {
      for (int j=0; j<dim; j++)
      {
         xs[i*dim+j] = xyz[i + (j*npoints)];
      }
   }

   add(*nodes0, *x1, *nodes1);
   FindPointsInMesh(*mesh1, xyz, m_conn, m_xi);

   for (int i=0; i<npoints; i++)
   {
      for (int j=0; j<4; j++)
      {
         for (int k=0; k<dim; k++)
         {
            (*coordsm)(i*4+j,k) = mesh1->GetVertex(m_conn[i*4+j])[k]+
                                  (*x1)[dim*m_conn[i*4+j]+k];
         }
      }
   }
   M->Clear();
   delete M;
   M = nullptr;
   M = new SparseMatrix(nnd,ndofs);
   //   M->OverrideSize(nnd,ndofs);
   for (int i=0; i<nnd; i++)
   {
      (*dM)[i].Clear();
      //      (*dM)[i].OverrideSize(ndofs,ndofs);
   }
   delete dM;
   dM = new std::vector<SparseMatrix>(nnd, SparseMatrix(ndofs,ndofs));

   Assemble_Contact(nnd, npoints, ndofs, xs, m_xi, *coordsm, s_conn, m_conn, g, *M,
                    *dM);
   M->Finalize(1,false);
   for (int i=0; i<nnd; i++)
   {
      (*dM)[i].Finalize(1,false);
   }
}

void ExContactBlockTL::update_jac()
{
   update_g();
}

void ExContactBlockTL::update_hess()
{
   update_g();
}

bool ExContactBlockTL::get_nlp_info(
   Index&          n,
   Index&          m,
   Index&          nnz_jac_g,
   Index&          nnz_h_lag,
   IndexStyleEnum& index_style
)
{
   // The problem described in ExContactBlockTL.hpp has 2 variables, x1, & x2,
   n = ndofs;

   // one equality constraint,
   m = nnd;

   nnz_jac_g = M->NumNonZeroElems();
   nnz_jac_g = nnd*ndofs; // treat it as a dense matrix for now

   // treat it as a dense matrix for now. only need lower-triangular part

   nnz_h_lag = (ndofs*ndofs + ndofs)/2;

   // We use the standard fortran index style for row/col entries
   index_style = C_STYLE;

   return true;
}

bool ExContactBlockTL::get_bounds_info(
   Index   n,
   Number* x_l,
   Number* x_u,
   Index   m,
   Number* g_l,
   Number* g_u
)
{
   assert(n == ndofs);
   assert(m == nnd);

   for (auto i=0; i<n; i++)
   {
      x_l[i] = -1.0e20;
      x_u[i] = +1.0e20;
   }

   for (auto i=0; i<Dirichlet_dof.Size(); i++)
   {
      x_l[Dirichlet_dof[i]] = Dirichlet_val[i];
      x_u[Dirichlet_dof[i]] = Dirichlet_val[i];
   }

   // we only have equality constraints
   for (auto i=0; i<m; i++)
   {
      g_l[i] = 0.0;
      g_u[i] = +1.0e20;
   }

   return true;
}

bool ExContactBlockTL::get_starting_point(
   Index   n,
   bool    init_x,
   Number* x,
   bool    init_z,
   Number* z_L,
   Number* z_U,
   Index   m,
   bool    init_lambda,
   Number* lambda
)
{
   assert(init_x == true);
   assert(init_z == false);
   assert(init_lambda == false);

   for (auto i=0; i<n; i++)
   {
      x[i] = 0;
   }
   for (auto i=0; i<ndof_1; i++)
   {
      x[i] = (*x1)[i];
   }
   for (auto i=ndof_1; i<n; i++)
   {
      x[i] = (*x2)[i-ndof_1] ;
   }

   return true;
}

bool ExContactBlockTL::eval_f(
   Index         n,
   const Number* x,
   bool          new_x,
   Number&       obj_value
)
{
   //   if(new_x)
   {
      for (auto i=0; i<ndof_1; i++)
      {
         (*x1)[i] = x[i];
      }
      for (auto i=ndof_1; i<n; i++)
      {
         (*x2)[i-ndof_1] = x[i];
      }
   }

   obj_value = 0;
   obj_value += A1.InnerProduct(*x1, *x1);
   obj_value += A2.InnerProduct(*x2, *x2);
   obj_value *= 0.5;

   return true;
}

bool ExContactBlockTL::eval_grad_f(
   Index         n,
   const Number* x,
   bool          new_x,
   Number*       grad_f
)
{
   //   if(new_x)
   {
      for (auto i=0; i<ndof_1; i++)
      {
         (*x1)[i] = x[i];
      }
      for (auto i=ndof_1; i<n; i++)
      {
         (*x2)[i-ndof_1] = x[i];
      }
   }

   // return the gradient of the objective function grad_{x} f(x)
   mfem::Vector temp1(ndof_1);
   mfem::Vector temp2(ndof_2);

   A1.Mult(*x1, temp1);
   A2.Mult(*x2, temp2);

   for (auto i=0; i<ndof_1; i++)
   {
      grad_f[i] = temp1[i];
   }
   for (auto i=0; i<ndof_2; i++)
   {
      grad_f[i+ndof_1] = temp2[i];
   }

   return true;
}

bool ExContactBlockTL::eval_g(
   Index         n,
   const Number* x,
   bool          new_x,
   Index         m,
   Number*       cons
)
{
   assert(n == ndofs);
   assert(m == nnd);

   //   if(new_x)
   {
      for (auto i=0; i<ndof_1; i++)
      {
         (*x1)[i] = x[i];
      }
      for (auto i=ndof_1; i<n; i++)
      {
         (*x2)[i-ndof_1] = x[i];
      }
      update_g();
   }

   for (auto i=0; i<m; i++)
   {
      cons[i] = g[i];
   }

   return true;
}

bool ExContactBlockTL::eval_jac_g(
   Index         n,
   const Number* x,
   bool          new_x,
   Index         m,
   Index         nele_jac,
   Index*        iRow,
   Index*        jCol,
   Number*       values
)
{
   assert(n == ndofs);
   assert(m == nnd);
   assert(n*m == nele_jac); // TODO: dense matrix for now
   if (new_x)
   {
      for (auto i=0; i<ndof_1; i++)
      {
         (*x1)[i] = x[i];
      }
      for (auto i=ndof_1; i<n; i++)
      {
         (*x2)[i-ndof_1] = x[i];
      }
      // TODO: do something here to update jac
   }

   // TODO: we use dense Jac for now
   if ( values == nullptr )
   {
      // return the structure of the jacobian of the constraints
      for (auto i=0; i<m; i++)
      {
         for (auto j=0; j<n; j++)
         {
            iRow[i*n+j] = i;
            jCol[i*n+j] = j;
         }
      }
   }
   else
   {
      const int *M_i = M->GetI();
      const int *M_j = M->GetJ();
      const double *M_data = M->GetData();

      for (auto i=0; i<nele_jac; i++)
      {
         values[i] = 1e-20;
      }
      // TODO: M_i and M_j can be nullptr ???
      for (auto i=0; i<m; i++)
      {
         for (auto k=M_i[i]; k<M_i[i+1]; k++)
         {
            values[i*n+M_j[k]] = M_data[k];
         }
      }
   }

   return true;
}

bool ExContactBlockTL::eval_h(
   Index         n,
   const Number* x,
   bool          new_x,
   Number        obj_factor,
   Index         m,
   const Number* lambda,
   bool          new_lambda,
   Index         nele_hess,
   Index*        iRow,
   Index*        jCol,
   Number*       values
)
{
   assert(n == ndofs);
   assert(m == nnd);
   assert((n*n+n)/2 == nele_hess); // TODO: dense matrix for now

   if (new_x)
   {
      for (auto i=0; i<ndof_1; i++)
      {
         (*x1)[i] = x[i];
      }
      for (auto i=ndof_1; i<n; i++)
      {
         (*x2)[i-ndof_1] = x[i];
      }
      // TODO: do something here to update hes
   }

   // TODO: we use dense Hes for now
   if ( values == nullptr )
   {
      // return the structure. This is a symmetric matrix, fill the lower left triangle only.
      int k = 0;
      for (auto i=0; i<n; i++)
      {
         for (auto j=0; j<=i; j++)
         {
            iRow[k] = i;
            jCol[k] = j;
            k++;
         }
      }
   }
   else
   {
      // return the values
      for (auto k=0; k<nele_hess; k++)
      {
         values[k] = 1e-20;
      }

      const int *K_i = K->GetI();
      const int *K_j = K->GetJ();
      const double *K_data = K->GetData();
      for (auto i=0; i<n; i++)
      {
         for (auto k=K_i[i]; k<K_i[i+1]; k++)
         {
            if (K_j[k]<=i)
            {
               values[(i*i+i)/2+K_j[k]] += K_data[k] * obj_factor;
            }
         }
      }

      for (auto con_idx=0; con_idx<m; con_idx++)
      {
         const int *dM_i = dM->at(con_idx).GetI();
         const int *dM_j = dM->at(con_idx).GetJ();
         const double *dM_data = dM->at(con_idx).GetData();
         for (auto i=0; i<n; i++)
         {
            for (auto k=dM_i[i]; k<dM_i[i+1]; k++)
            {
               if (dM_j[k]<=i)
               {
                  values[(i*i+i)/2+dM_j[k]] += dM_data[k] * lambda[con_idx];
               }
            }
         }
      }
   }

   return true;
}

void ExContactBlockTL::finalize_solution(
   SolverReturn               status,
   Index                      n,
   const Number*              x,
   const Number*              z_L,
   const Number*              z_U,
   Index                      m,
   const Number*              g,
   const Number*              lambda,
   Number                     obj_value,
   const IpoptData*           ip_data,
   IpoptCalculatedQuantities* ip_cq
)
{}

int main(int argc, char *argv[])
{
   // Create an instance of your nlp...
   SmartPtr<TNLP> mynlp = new ExContactBlockTL(argc, argv);

   // Create an instance of the IpoptApplication
   //
   // We are using the factory, since this allows us to compile this
   // example with an Ipopt Windows DLL
   SmartPtr<IpoptApplication> app = IpoptApplicationFactory();

   // Initialize the IpoptApplication and process the options
   ApplicationReturnStatus status;
   status = app->Initialize();
   if ( status != Solve_Succeeded )
   {
      std::cout << std::endl << std::endl << "*** Error during initialization!" <<
                std::endl;
      return (int) status;
   }

   status = app->OptimizeTNLP(mynlp);

   if ( status == Solve_Succeeded )
   {
      // Retrieve some statistics about the solve
      Index iter_count = app->Statistics()->IterationCount();
      std::cout << std::endl << std::endl << "*** The problem solved in " <<
                iter_count << " iterations!" << std::endl;

      Number final_obj = app->Statistics()->FinalObjective();
      std::cout << std::endl << std::endl <<
                "*** The final value of the objective function is " << final_obj << '.'
                << std::endl;
   }

   return (int) status;
}
