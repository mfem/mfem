//                                Contact example
//
// Compile with: make contact
//
// Sample runs:  ./contact -m1 block1.mesh -m2 block2.mesh -at "5 6 7 8"
// Sample runs:  ./contact -m1 block1_d.mesh -m2 block2_d.mesh -at "5 6 7 8"

#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include "nodepair.hpp"

using namespace std;
using namespace mfem;

Vector GetNormalVector(Mesh & mesh, const int elem, const double *ref,
                       int & refFace, int & refNormal)
{
   ElementTransformation *trans = mesh.GetElementTransformation(elem);
   const int dim = mesh.Dimension();
   const int spaceDim = trans->GetSpaceDim();

   MFEM_VERIFY(spaceDim == 3, "");

   Vector n(spaceDim);

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

   std::vector<Vector> tang(2);

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

      Vector tangRef(3);
      tangRef = 0.0;
      tangRef[tangDir[i]] = 1.0;

      jac.Mult(tangRef, tang[i]);
   }

   Vector c(3);  // Cross product

   c[0] = (tang[0][1] * tang[1][2]) - (tang[0][2] * tang[1][1]);
   c[1] = (tang[0][2] * tang[1][0]) - (tang[0][0] * tang[1][2]);
   c[2] = (tang[0][0] * tang[1][1]) - (tang[0][1] * tang[1][0]);

   c /= c.Norml2();

   Vector nref(3);
   nref = 0.0;
   nref[dimNormal] = 1.0;

   Vector ndir(3);
   jac.Mult(nref, ndir);

   ndir /= ndir.Norml2();

   const double dp = ndir * c;

   // TODO: eliminate c?
   n = c;
   if (dp < 0.0)
   {
      n *= -1.0;
   }

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

int GetHexVertex(int cdim, int c, int fa, int fb, Vector & refCrd)
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

   for (int i=0; i<np; ++i)
   {
      cout << "Point " << i << ": (";
      for (int j=0; j<dim; ++j)
      {
         cout << xyz[i + (j*np)];
         if (j == dim-1)
         {
            cout << ")" << endl;
         }
         else
         {
            cout << ", ";
         }
      }

      cout << "  element: " << elems[i] << endl;
      cout << "  element " << elems[i] << " vertices:" << endl;
      Array<int> vert;
      mesh.GetElementVertices(elems[i], vert);
      for (auto v : vert)
      {
         cout << "    " << v << endl;
      }

      cout << "  reference coordinates: (";
      for (int j=0; j<dim; ++j)
      {
         cout << refcrd[(i*dim) + j];
         if (j == dim-1)
         {
            cout << ")" << endl;
         }
         else
         {
            cout << ", ";
         }
      }

      int refFace, refNormal, refNormalSide;
      Vector normal = GetNormalVector(mesh, elems[i], refcrd.GetData() + (i*dim),
                                      refFace, refNormal);

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
      cout << "  face reference coordinates: (";
      for (int j=0; j<dim-1; ++j)
      {
         cout << faceRefCrd[j];
         if (j == dim-2)
         {
            cout << ")" << endl;
         }
         else
         {
            cout << ", ";
         }
      }

      cout << "  normal vector: ";
      normal.Print();

      IntegrationPoint ip;
      ip.Set(refcrd.GetData() + (i*dim), dim);
      ElementTransformation *trans = mesh.GetElementTransformation(elems[i]);
      Vector phys(trans->GetSpaceDim());
      trans->Transform(ip, phys);
      cout << "  physical coordinates: ";
      phys.Print();

      // Get the element face
      Array<int> faces;
      Array<int> ori;
      mesh.GetElementFaces(elems[i], faces, ori);

      const int face = faces[refFace];

      Array<int> faceVert;
      mesh.GetFaceVertices(face, faceVert);

      cout << "  face " << face << " vertices:" << endl;
      for (auto v : faceVert)
      {
         cout << "    " << v << endl;
      }
    
      for (int p=0; p<4; p++)
      {
        conn[4*i+p] = faceVert[p];
      }

      Vector ref(dim);
      for (int p=0; p<2; ++p)
         for (int q=0; q<2; ++q)
         {
            const int refv = GetHexVertex(refNormal, refNormalSide, p, q, ref);
            cout << "  face reference vertex (" << p << "," << q
                 << ") is global vertex " << vert[refv] << endl;

            {
               // Sanity check
               ip.Set(ref.GetData(), dim);
               trans->Transform(ip, phys);
               for (int j=0; j<dim; ++j)
               {
                  phys[j] -= mesh.GetVertex(vert[refv])[j];
               }
               cout << phys.Norml2()<<endl;
               MFEM_VERIFY(phys.Norml2() < 1.0e-12, "Sanity check failed");
            }
         }
   }
}

int main(int argc, char *argv[])
{
   // 1. Parse command-line options.
   const char *mesh_file1 = "block1.mesh";
   const char *mesh_file2 = "block2.mesh";

   Array<int> attr;

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

   FiniteElementCollection *fec2 = new H1_FECollection(1, dim);
   FiniteElementSpace *fespace2 = new FiniteElementSpace(&mesh2, fec2, dim, Ordering::byVDIM);
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

   // Generate elastic internal energy for both meshes
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
   for (auto a : attr)
   {
      cout << a << endl;
   }

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
      cout << v << ": " << mesh2.GetVertex(v)[0] << ", "
           << mesh2.GetVertex(v)[1] << ", "
           << mesh2.GetVertex(v)[2] << endl;

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
   
   Array<int> m_conn(npoints*4); // only works for linear elements that have 4 vertices!
   DenseMatrix coordsm(npoints*4, dim);
 
   // adding displacement to mesh1
   // use a separate mesh?
   x1 = -1e-4; 
   mesh1.SetNodalFESpace(fespace1);
   GridFunction *nodes1 = mesh1.GetNodes();
   *nodes1 += x1;
   nodes1->Print();
 
   FindPointsInMesh(mesh1, xyz, m_conn, m_xi);
   for (int i=0; i<npoints; i++)
   {
     for (int j=0; j<4; j++)
     {
	 for (int k=0; k<dim; k++)
	 {
           coordsm(i*4+j,k) = mesh1.GetVertex(m_conn[i*4+j])[k];     
	 }
     }
   }
   
   SparseMatrix M(nnd,ndofs);
   std::vector<SparseMatrix> dM(nnd, SparseMatrix(ndofs,ndofs));
   
   Assemble_Contact(nnd, npoints, ndofs, xs, m_xi, coordsm, s_conn, m_conn, 
                    g, M, dM);
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
