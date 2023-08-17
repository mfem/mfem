//                             Problem classes, which contain
//                             needed functionality for an
//                             interior-point filter-line search solver   
//                               
//
//

#include <fstream>
#include <iostream>
#include <array>

#include "mfem.hpp"
#include "problems.hpp"
#include "nodepair.hpp"

using namespace std;
using namespace mfem;


OptProblem::OptProblem() {}

void OptProblem::CalcObjectiveGrad(const BlockVector &x, BlockVector &y) const
{
  Duf(x, y.GetBlock(0));
  Dmf(x, y.GetBlock(1));
}

OptProblem::~OptProblem()
{
  block_offsetsx.DeleteAll();
}


// min E(d) s.t. g(d) >= 0
// min_(d,s) E(d) s.t. c(d,s) := g(d) - s = 0, s >= 0

/*ContactProblem::ContactProblem(int dimd, int dimg) : OptProblem(), dimD(dimd), dimS(dimg), block_offsetsx(3)
{
  dimU = dimD;
  dimM = dimS;
  dimC = dimS;
  block_offsetsx[0] = 0;
  block_offsetsx[1] = dimU;
  block_offsetsx[2] = dimM;
  block_offsetsx.PartialSum();
  ml.SetSize(dimM); ml = 0.0;
}*/

ContactProblem::ContactProblem() : OptProblem(), block_offsetsx(3)
{
  /*dimU = dimD;
  dimM = dimS;
  dimC = dimS;
  block_offsetsx[0] = 0;
  block_offsetsx[1] = dimU;
  block_offsetsx[2] = dimM;
  block_offsetsx.PartialSum();
  ml.SetSize(dimM); ml = 0.0;*/
}

void ContactProblem::InitializeParentData(int dimd, int dims)
{
  dimU = dimd;
  dimM = dims;
  dimC = dims;
  block_offsetsx[0] = 0;
  block_offsetsx[1] = dimU;
  block_offsetsx[2] = dimM;
  block_offsetsx.PartialSum();
  ml.SetSize(dimM); ml = 0.0;
}

double ContactProblem::CalcObjective(const BlockVector &x) const { return E(x.GetBlock(0)); }

void ContactProblem::Duf(const BlockVector &x, Vector &y) const { DdE(x.GetBlock(0), y); }

void ContactProblem::Dmf(const BlockVector &x, Vector &y) const { y = 0.0; }

SparseMatrix* ContactProblem::Duuf(const BlockVector &x) { return DddE(x.GetBlock(0)); }

SparseMatrix* ContactProblem::Dumf(const BlockVector &x) { return nullptr; }

SparseMatrix* ContactProblem::Dmuf(const BlockVector &x) { return nullptr; }

SparseMatrix* ContactProblem::Dmmf(const BlockVector &x) { return nullptr; }

void ContactProblem::c(const BlockVector &x, Vector &y) const // c(u,m) = g(u) - m 
{
  g(x.GetBlock(0), y);
  y.Add(-1.0, x.GetBlock(1));  
}

SparseMatrix* ContactProblem::Duc(const BlockVector &x) { return Ddg(x.GetBlock(0)); }

SparseMatrix* ContactProblem::Dmc(const BlockVector &x) 
{ 
  Vector negIdentDiag(dimM);
  negIdentDiag = -1.0;
  return new SparseMatrix(negIdentDiag);
} 

ContactProblem::~ContactProblem() {}




//ObstacleProblem::ObstacleProblem(FiniteElementSpace *fes, double (*fSource)(const Vector &)) : ContactProblem(fes->GetTrueVSize(), fes->GetTrueVSize()), Vh(fes), f(dimD)
ObstacleProblem::ObstacleProblem(FiniteElementSpace *fes, double (*fSource)(const Vector &)) : ContactProblem()
{
  Vh = fes;
  dimD = fes->GetTrueVSize();
  dimS = fes->GetTrueVSize();
  InitializeParentData(dimD, dimS);
  Kform = new BilinearForm(Vh);
  Kform->AddDomainIntegrator(new MassIntegrator);
  Kform->AddDomainIntegrator(new DiffusionIntegrator);
  Kform->Assemble();
  Kform->Finalize();
  Kform->FormSystemMatrix(empty_tdof_list, K);

  FunctionCoefficient fcoeff(fSource);
  fform = new LinearForm(Vh);
  fform->AddDomainIntegrator(new DomainLFIntegrator(fcoeff));
  fform->Assemble();
  f.SetSize(dimD);
  f.Set(1.0, *fform);

  Vector iDiag(dimD); iDiag = 1.0;
  J = new SparseMatrix(iDiag);

}

double ObstacleProblem::E(const Vector &d) const
{
  Vector Kd(dimD); Kd = 0.0;
  K.Mult(d, Kd);
  return 0.5 * InnerProduct(d, Kd) - InnerProduct(f, d);
}

void ObstacleProblem::DdE(const Vector &d, Vector &gradE) const
{
  K.Mult(d, gradE);
  gradE.Add(-1.0, f);
}

SparseMatrix* ObstacleProblem::DddE(const Vector &d)
{
  return new SparseMatrix(K); 
}

// g(d) = d >= 0
void ObstacleProblem::g(const Vector &d, Vector &gd) const
{
  gd.Set(1.0, d);
}

SparseMatrix* ObstacleProblem::Ddg(const Vector &d)
{
  return new SparseMatrix(*J);
}

ObstacleProblem::~ObstacleProblem()
{
  delete Kform;
  delete fform;
  delete J;
}

//-------------------
DirichletObstacleProblem::DirichletObstacleProblem(FiniteElementSpace *fes, Vector &x0DC, double (*fSource)(const Vector &), 
		double (*obstacleSource)(const Vector &),
		Array<int> tdof_list, bool reduced = true) : ContactProblem()
{
  Vh = fes;
  ess_tdof_list = tdof_list;
  dimD = fes->GetTrueVSize();
  if(reduced)
  {
    dimS = dimD - ess_tdof_list.Size(); 
  }
  else
  {
    dimS = dimD;
  } 
  InitializeParentData(dimD, dimS);

  xDC.SetSize(dimD);
  xDC.Set(1.0, x0DC);
  
  // define Hessian of energy objective
  // K = [[ \hat{K}   0]
  //      [ 0         I]]
  // where \hat{K} acts on dofs not constrained by the Dirichlet condition
  // I acts on dofs constrained by the Dirichlet condition
  Kform = new BilinearForm(Vh);
  Kform->AddDomainIntegrator(new DiffusionIntegrator);
  Kform->Assemble();
  Kform->EliminateVDofs(ess_tdof_list);
  Kform->Finalize();
  K = new SparseMatrix(Kform->SpMat());
  
  // define right hand side dual-vector
  // f_i = int fSource(x) \phi_i(x) dx, where {\phi_i}_i is the FE basis
  // f = f - K1 xDC, where K1 contains the eliminated part of K
  FunctionCoefficient fcoeff(fSource);
  fform = new LinearForm(Vh);
  fform->AddDomainIntegrator(new DomainLFIntegrator(fcoeff));
  fform->Assemble();
  f.SetSize(dimD);
  f.Set(1.0, *fform);
  Kform->EliminateVDofsInRHS(ess_tdof_list, xDC, f);

  // define obstacle function
  FunctionCoefficient psicoeff(obstacleSource);
  GridFunction psi_gf(Vh);
  psi_gf.ProjectCoefficient(psicoeff);
  psi.SetSize(dimS); psi = 0.0;
  //psi.Set(1.0, psi_gf);
  
  // ------ construct dimS x dimD Jacobian with zero columns correspodning to Dirichlet dofs
  J = new SparseMatrix(dimS, dimD);
  bool freeDof;
  int rowCount = 0;
  for(int j = 0; j < dimD; j++)
  {
    freeDof = true;
    for(int i = 0; i < ess_tdof_list.Size(); i++)
    {
      if( j == ess_tdof_list[i])
      {
        freeDof = false;
      }
    }
    
    Array<int> col_tmp; mfem::Vector v_tmp;
    col_tmp.SetSize(1); v_tmp.SetSize(1);
    col_tmp[0] = j;
    if (freeDof)
    {
      v_tmp(0) = 1.0;
      J->SetRow(rowCount, col_tmp, v_tmp);
      psi(rowCount) = psi_gf(j);
      rowCount += 1;
    }
    else if(dimD == dimS)
    {
      v_tmp(0) = 0.0;
      J->SetRow(rowCount, col_tmp, v_tmp);
      psi(rowCount) = psi_gf(j)-0.01;
      rowCount += 1;
    }
  }
  J->Finalize();
}

double DirichletObstacleProblem::E(const Vector &d) const
{
  Vector Kd(dimD); Kd = 0.0;
  K->Mult(d, Kd);
  return 0.5 * InnerProduct(d, Kd) - InnerProduct(f, d);
}

void DirichletObstacleProblem::DdE(const Vector &d, Vector &gradE) const
{
  K->Mult(d, gradE);
  gradE.Add(-1.0, f);
}

SparseMatrix* DirichletObstacleProblem::DddE(const Vector &d)
{
  return new SparseMatrix(*K); 
}

// g(d) = d - \psi >= 0
//        d - \psi - s  = 0
//                   s >= 0
void DirichletObstacleProblem::g(const Vector &d, Vector &gd) const
{
  J->Mult(d, gd);
  gd.Add(-1., psi);
}

SparseMatrix* DirichletObstacleProblem::Ddg(const Vector &d)
{
  return new SparseMatrix(*J);
}

DirichletObstacleProblem::~DirichletObstacleProblem()
{
  delete J;
  delete K;
  delete Kform;
  delete fform;
}


ReducedContactProblem::ReducedContactProblem(ContactProblem * contactin, Array<int> activeConstraintsin, Array<int> fixedDofsin) : ContactProblem()
{
  contact = contactin;
  activeConstraints = activeConstraintsin;
  fixedDofs = fixedDofsin;
  dimD = contact->GetDimD();
  dimS = activeConstraints.Size();
  InitializeParentData(dimD, dimS);
  dimSin = contact->GetDimS();
}

ReducedContactProblem::~ReducedContactProblem() {}

double ReducedContactProblem::E(const Vector &d) const
{
  return contact->E(d);
}

void ReducedContactProblem::DdE(const Vector &d, Vector &gradE) const
{
  contact->DdE(d, gradE);
}

SparseMatrix * ReducedContactProblem::DddE(const Vector &d)
{
  return contact->DddE(d);
}

void ReducedContactProblem::g(const Vector &d, Vector &gd) const
{
  Vector gdin(dimSin); gdin = 0.0;
  contact->g(d, gdin);
  for(int i = 0; i < dimS; i++)
  {
    gd(i) = gdin(activeConstraints[i]);
  } 
}

SparseMatrix * ReducedContactProblem::Ddg(const Vector &d)
{
  SparseMatrix * Jin;
  Jin = contact->Ddg(d);
  SparseMatrix * J;
  J = new SparseMatrix(dimS, dimD);
  for(int i = 0; i < dimS; i++)
  {
    Array<int> col_tmp;
    mfem::Vector v_tmp;
    col_tmp = 0;
    v_tmp   = 0.0;
    Jin->GetRow(activeConstraints[i], col_tmp, v_tmp);
    bool freeDof;
    for(int j = 0; j < v_tmp.Size(); j++)
    {
      freeDof = true;
      for(int k = 0; k < fixedDofs.Size(); k++)
      {
        if(col_tmp[j] == fixedDofs[k])
	{
	  freeDof = false;
	}
      }
      if(!freeDof)
      {
        v_tmp(j) = 0.0;
      }
    }
    J->SetRow(i, col_tmp, v_tmp);
  }
  J->Finalize();
  J->Print();
  delete Jin;
  return J;
}


// quadratic approximation of the objective
// linear approximation of the gap function constraint
// E(d) = 1 / 2 d^T K d + f^T d
// g(d) = J d + g0
QPContactProblem::QPContactProblem(const SparseMatrix Kin, const SparseMatrix Jin, const Vector fin, const Vector g0in) : ContactProblem()
{
  K = new SparseMatrix(Kin);
  J = new SparseMatrix(Jin);
  K->Finalize();
  J->Finalize();
  f.SetSize(fin.Size()); f = fin;
  g0.SetSize(g0in.Size()); g0 = g0in;
  MFEM_VERIFY(K->Width() == J->Width(), "K and J do not have the same number of columns.");
  MFEM_VERIFY(J->Height() == g0.Size(), "Number of rows of J is not equal to length of g0 vector.");
  MFEM_VERIFY(K->Height() == f.Size(), "Number of rows of K is not equal to length of f vector.");
  
  dimD = K->Height();
  dimS = J->Height();
  InitializeParentData(dimD, dimS);
}

// E(d) = 1 / 2 d^T K d + f^T d
double QPContactProblem::E(const Vector &d) const
{
  Vector Kd(dimD); Kd = 0.0;
  K->Mult(d, Kd);
  return 0.5 * InnerProduct(d, Kd) + InnerProduct(f, d);
}


// gradient(E) = K d + f
void QPContactProblem::DdE(const Vector &d, Vector &gradE) const
{
  K->Mult(d, gradE);
  gradE.Add(1.0, f);
}

// Hessian(E) = K
SparseMatrix* QPContactProblem::DddE(const Vector &d)
{
  return new SparseMatrix(*K); 
}

// g(d) = J * d + g0 >= 0
void QPContactProblem::g(const Vector &d, Vector &gd) const
{
  J->Mult(d, gd);
  gd.Add(1.0, g0);
}

// Jacobian(g) = J
SparseMatrix* QPContactProblem::Ddg(const Vector &d)
{
  return new SparseMatrix(*J);
}

QPContactProblem::~QPContactProblem()
{
  delete K;
  delete J;
}





// -------------- utility functions --------------------

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
   attr.Append(3);
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
   //cout << "-----------------\n";
   /*for(int i = 0; i < dim; i++)
   {
     cout << "ref_" << i << " = " << ref[i] << endl;
   }*/
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

   FindPointsGSLIB finder;

   finder.SetDistanceToleranceForPointsFoundOnBoundary(0.5);

   const double bb_t = 0.5;
   finder.Setup(mesh, bb_t);

   finder.FindPoints(xyz,mfem::Ordering::byVDIM);

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
   if(!allfound)
   {
     for(int i = 0; i < dist.Size(); i++)
     {
       cout << "distance between saught and found points = " << dist(i) << endl;
     }
   }
   MFEM_VERIFY(allfound, "A point was not found");

   cout << "Maximum distance of projected points: " << dist.Max() << endl;

   // extract information
   for (int i=0; i<np; ++i)
   {
      int refFace, refNormal;
      // int refNormalSide;
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
         xs[0] = xyz[i*dim];
         xs[1] = xyz[i*dim + 1];
         xs[2] = xyz[i*dim + 2];
         
         mfem::Vector xi_tmp(dim-1);
         // get nodes!

         GridFunction *nodes = mesh.GetNodes();
         DenseMatrix coords(4,3);
         for (int k=0; k<4; k++)
         {
            for (int j=0; j<3; j++)
            {
               coords(k,j) = (*nodes)[cbdrVert[k]*3+j];
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
                  // refNormalSide = (refcrd[(i*dim) + j] > 0.5);
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
ExContactBlockTL::ExContactBlockTL(int ref_levels)
   : 
   ContactProblem(),
   mesh1{nullptr},
   mesh2{nullptr},
   fec1{nullptr},
   fec2{nullptr},
   fespace1{nullptr},
   fespace2{nullptr},
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
   mesh_file1 = "meshes/block1.mesh";
   mesh_file2 = "meshes/rotatedblock2.mesh";
   const char *mf1 = mesh_file1.c_str();
   const char *mf2 = mesh_file2.c_str();

   mesh1 = new Mesh(mf1, 1, 1);
   mesh2 = new Mesh(mf2, 1, 1);
   {
     for(int l = 0; l < ref_levels; l++)
     {
       mesh1->UniformRefinement();
       mesh2->UniformRefinement();
     }
   }

   dim = mesh1->Dimension();
   MFEM_VERIFY(dim == mesh2->Dimension(), "");

   // boundary attribute 2 is the potential contact surface of nodes
   attr.Append(3);
   // boundary attribute 2 is the potential contact surface for master surface
   m_attr.Append(3);

   fec1     = new H1_FECollection(1, dim);
   fespace1 = new FiniteElementSpace(mesh1, fec1, dim, Ordering::byVDIM);
   ndof_1   = fespace1->GetTrueVSize();
   cout << "Number of finite element unknowns for mesh1: " << ndof_1 << endl;
   mesh1->SetNodalFESpace(fespace1);

   fec2 = new H1_FECollection(1, dim);
   fespace2 = new FiniteElementSpace(mesh2, fec2, dim, Ordering::byVDIM);
   ndof_2   = fespace2->GetTrueVSize();
   cout << "Number of finite element unknowns for mesh2: " << ndof_2 << endl;
   mesh2->SetNodalFESpace(fespace2);
   
   // degrees of freedom of both meshes
   ndofs = ndof_1 + ndof_2;
   
   // number of nodes for each mesh
   nnd_1 = mesh1->GetNV();
   nnd_2 = mesh2->GetNV();
   nnd = nnd_1 + nnd_2;

   nodes0.SetSpace(mesh1->GetNodes()->FESpace());
   nodes0 = *mesh1->GetNodes();
   nodes1 = mesh1->GetNodes();


   Array<int> ess_bdr1(mesh1->bdr_attributes.Max());
   ess_bdr1 = 0;
   Array<int> ess_bdr2(mesh2->bdr_attributes.Max());
   ess_bdr2 = 0;
   
   for (int b=0; b<mesh2->GetNBE(); ++b)
   {
      if (mesh2->GetBdrAttribute(b)== 3)
      {
         Array<int> vert;
         mesh2->GetBdrElementVertices(b, vert);
         for (auto v : vert)
         {
            bdryVerts2.insert(v);
         }
      }
   }
   
   std::set<int> dirbdryv2;
   for (int b=0; b<mesh2->GetNBE(); ++b)
   {
      if (mesh2->GetBdrAttribute(b) == 2)
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
      if (mesh1->GetBdrAttribute(b) == 2)
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

   x1 = new GridFunction(fespace1);
   x2 = new GridFunction(fespace2);
   (*x1) = 0.0;
   (*x2) = 0.0;
   for(int i = 0; i < Dirichlet_dof.Size(); i++)
   {
     if(Dirichlet_dof[i] >= ndof_1)
     {
       (*x2)(Dirichlet_dof[i] - ndof_1) = Dirichlet_val[i];
       ess_tdof_list2.Append(Dirichlet_dof[i] - ndof_1);
     }
     else
     {
       (*x1)(Dirichlet_dof[i]) = Dirichlet_val[i];
       ess_tdof_list1.Append(Dirichlet_dof[i]);
     }
   }

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
   a1->SetDiagonalPolicy(Operator::DIAG_ONE);
   a1->AddDomainIntegrator(new ElasticityIntegrator(*lambda1_func, *mu1_func));
   a1->Assemble();
   // a1->EliminateVDofs(ess_tdof_list1);
   // a1->Finalize();
   // A1 = a1->SpMat();

   // B1.SetSize(ndof_1); B1 = 0.0;
   // a1->EliminateVDofsInRHS(ess_tdof_list1, (*x1), B1);
   a1->FormLinearSystem(ess_tdof_list1,*x1,*b1,A1,X1,B1);

   a2 = new BilinearForm(fespace2);
   a2->SetDiagonalPolicy(Operator::DIAG_ONE);
   a2->AddDomainIntegrator(new ElasticityIntegrator(*lambda2_func, *mu2_func));
   a2->Assemble();
   // a2->EliminateVDofs(ess_tdof_list2);
   // a2->Finalize();
   // A2 = a2->SpMat();
   
   // B2.SetSize(ndof_2); B2 = 0.0;
   // a2->EliminateVDofsInRHS(ess_tdof_list2, (*x2), B2);

   a2->FormLinearSystem(ess_tdof_list2,*x2,*b2,A2,X2,B2);

   Array<int> offs(3);
   offs[0] = 0;
   offs[1] = ndof_1;
   offs[2] = ndof_2;
   offs.PartialSum();
   BlockMatrix Kblock(offs);
   Kblock.SetBlock(0,0,&A1);
   Kblock.SetBlock(1,1,&A2);
   K = Kblock.CreateMonolithic();

   // K = new SparseMatrix(ndofs, ndofs);
   // for (int i=0; i<A1.Height(); i++) // 1,1 block
   // {
   //    Array<int> col_tmp;
   //    mfem::Vector v_tmp;
   //    // col_tmp = 0;
   //    // v_tmp = 0.0;
   //    A1.GetRow(i, col_tmp, v_tmp);
   //    K->SetRow(i, col_tmp, v_tmp);
   // }
   // for (int i=0; i<A2.Height(); i++) // 2, 2 block
   // {
   //    Array<int> col_tmp;
   //    mfem::Vector v_tmp;
   //    // col_tmp = 0;
   //    // v_tmp = 0.0;
   //    A2.GetRow(i, col_tmp, v_tmp);
   //    for (int j=0; j<col_tmp.Size(); j++)
   //    {
   //       col_tmp[j] += ndof_1;
   //    }
   //    K->SetRow(i+ndof_1, col_tmp, v_tmp);  // mesh1 top left corner
   // }
   // K->Finalize(1,false);
   K->Threshold(0.0);
   K->SortColumnIndices();
   // Construct node to segment contact constraint.
   attr.Sort();

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
         xyz(count*dim + i) = mesh2->GetVertex(v)[i] + (*x2)[v*dim+i];
      }
      s_conn[count] = v + nnd_1; // dof1 is the master
      count++;
   }

   MFEM_VERIFY(count == npoints, "");

   // segment reference coordinates of the closest point
   m_xi.SetSize(npoints*(dim-1));
   m_xi = -1.0;
   // xs.SetSize(dim*npoints);
   // xs = 0.0;
   // for (int i=0; i<npoints; i++)
   // {
   //    for (int j=0; j<dim; j++)
   //    {
   //       xs[i*dim+j] = xyz[i + (j*npoints)];
   //    }
   // }

   m_conn.SetSize(4*npoints); // only works for linear elements that have 4 vertices!
   coordsm = new DenseMatrix(4*npoints, dim);

   // adding displacement to mesh1 using a fixed grid function from mesh1
   // Tucker modification, removing the nullifying of x1...
   (*x1) = 0.0; // x1 order: [xyz xyz... xyz]
   add(nodes0, *x1, *nodes1); // issues with moving the mesh nodes?

   FindPointsInMesh(*mesh1, xyz, m_conn, m_xi); // memory is leaked when this function is called

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
   
   // --- enforcing compatibility with the contactproblem structure
   dimD = ndofs;
   dimS = nnd;
   InitializeParentData(dimD, dimS);
   // ---
   
   M = new SparseMatrix(nnd,ndofs);
   dM = new std::vector<SparseMatrix>(nnd, SparseMatrix(ndofs,ndofs));

   // Assemble_Contact(nnd, npoints, ndofs, xs, m_xi, *coordsm, s_conn, m_conn, gapv, *M,
   //                  *dM);
   Assemble_Contact(nnd, npoints, ndofs, xyz, m_xi, *coordsm, s_conn, m_conn, gapv, *M,
                    *dM);                    
   assert(M);
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

FiniteElementSpace ExContactBlockTL::GetVh1()
{
  return *fespace1;
}

FiniteElementSpace ExContactBlockTL::GetVh2()
{
  return *fespace2;
}


// update gap function based on a current configuration
// how is that data fed here
void ExContactBlockTL::update_g() const
{
   int count = 0;
   for (auto v : bdryVerts2)
   {
      for (int i=0; i<dim; ++i)
      {
         // xyz[count + (i * npoints)] = mesh2->GetVertex(v)[i] + (*x2)[v*dim+i];
         xyz(count*dim + i) = mesh2->GetVertex(v)[i] + (*x2)[v*dim+i];
      }
      count++;
   }
   MFEM_VERIFY(count == npoints, "");

   // xs = 0.0;
   // for (int i=0; i<npoints; i++)
   // {
   //    for (int j=0; j<dim; j++)
   //    {
   //       xs[i*dim+j] = xyz[i + (j*npoints)];
   //    }
   // }

   add(nodes0, *x1, *nodes1);
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
   for (int i=0; i<nnd; i++)
   {
      (*dM)[i].Clear();
   }
   delete dM;
   dM = new std::vector<SparseMatrix>(nnd, SparseMatrix(ndofs,ndofs));

   Assemble_Contact(nnd, npoints, ndofs, xyz, m_xi, *coordsm, s_conn, m_conn, gapv, *M,
                    *dM);
}

void ExContactBlockTL::update_jac()
{
   assert(0 && "cannot reach here");
   update_g();
}

void ExContactBlockTL::update_hess()
{
   assert(0 && "cannot reach here");
   update_g();
}



double ExContactBlockTL::E(const Vector &d) const
{
  double obj_val = 0.0;
  Number * x = new Number[ndofs];
  
  for(int i = 0; i < ndofs; i++)
  {
    x[i] = d(i);
  }
  
//   bool boolreturned;
//   boolreturned = eval_f(0, x, true, obj_val);
  eval_f(0, x, true, obj_val);
  delete [] x;  
  return obj_val;
}

void ExContactBlockTL::DdE(const Vector &d, Vector &gradE) const
{
  Number * x = new Number[ndofs];
  Number * grad_obj = new Number[ndofs];
  
  for(int i = 0; i < ndofs; i++)
  {
    x[i] = d(i);
  }
  
//   bool boolreturned;
//   boolreturned = eval_grad_f(0, x, true, grad_obj);
  
  eval_grad_f(0, x, true, grad_obj);

  for(int i = 0; i < ndofs; i++)
  {
    gradE(i) = grad_obj[i];
  }
  
  delete [] x;
  delete [] grad_obj;
}

SparseMatrix* ExContactBlockTL::DddE(const Vector &d)
{
  return new SparseMatrix(*K); 
}


   
// g(d) = d >= 0
void ExContactBlockTL::g(const Vector &d, Vector &gd) const
{
  Number * x    = new Number[ndofs];
  Number * gapx = new Number[nnd];
  
  for(int i = 0; i < ndofs; i++)
  {
    x[i] = d(i);
  }
  
//   bool boolreturned;
//   boolreturned = eval_g(ndofs, x, true, nnd, gapx);
  eval_g(ndofs, x, true, nnd, gapx);
  for(int i = 0; i < nnd; i++)
  {
    gd(i) = gapx[i];
  }
  
  delete [] x;
  delete [] gapx;
}

SparseMatrix* ExContactBlockTL::Ddg(const Vector &d)
{
  // !!!!!!!!!!!!!!!!TO DO: call eval_jac_g ....
  // only do so after eval_jac_g has been updated in order 
  // that the gap function Jacobian data that is stored in 
  // the SparseMatrix member data M is updated
  for(int i = 0; i < ndof_1; i++)
  {
    (*x1)(i) = d(i);
  }
  for(int i = ndof_1; i < ndofs; i++)
  {
    (*x2)(i-ndof_1) = d(i);
  }
  //update_g();
  return new SparseMatrix(*M);
}
















//bool ExContactBlockTL::get_nlp_info(
//   Index&          n,
//   Index&          m,
//   Index&          nnz_jac_g,
//   Index&          nnz_h_lag,
//   IndexStyleEnum& index_style
//)
//{
//   // The problem described in ExContactBlockTL.hpp has 2 variables, x1, & x2,
//   n = ndofs;
//
//   // one equality constraint,
//   m = nnd;
//
//   nnz_jac_g = nnd*ndofs; // treat it as a dense matrix for now
//
//   // treat it as a dense matrix for now. only need lower-triangular part
//   nnz_h_lag = (ndofs*ndofs + ndofs)/2;
//
//   // We use the standard fortran index style for row/col entries
//   index_style = C_STYLE;
//
//   return true;
//}
//
//bool ExContactBlockTL::get_bounds_info(
//   Index   n,
//   Number* x_l,
//   Number* x_u,
//   Index   m,
//   Number* g_l,
//   Number* g_u
//)
//{
//   assert(n == ndofs);
//   assert(m == nnd);
//
//   for (auto i=0; i<n; i++)
//   {
//      x_l[i] = -1.0e20;
//      x_u[i] = +1.0e20;
//   }
//   for (auto i=0; i<Dirichlet_dof.Size(); i++)
//   {
//      x_l[Dirichlet_dof[i]] = Dirichlet_val[i];
//      x_u[Dirichlet_dof[i]] = Dirichlet_val[i];
//   }
//
//   // we only have equality constraints
//   for (auto i=0; i<m; i++)
//   {
//      g_l[i] = 0.0;
//      g_u[i] = +1.0e20;
//   }
//
//   return true;
//}
//
//bool ExContactBlockTL::get_starting_point(
//   Index   n,
//   bool    init_x,
//   Number* x,
//   bool    init_z,
//   Number* z_L,
//   Number* z_U,
//   Index   m,
//   bool    init_lambda,
//   Number* lambda
//)
//{
//   assert(init_x == true);
//   assert(init_z == false);
//   assert(init_lambda == false);
//
//   for (auto i=0; i<n; i++)
//   {
//      x[i] = 0;
//   }
//   for (auto i=0; i<ndof_1; i++)
//   {
//      x[i] = (*x1)[i];
//   }
//   for (auto i=ndof_1; i<n; i++)
//   {
//      x[i] = (*x2)[i-ndof_1] ;
//   }
//
//   return true;
//}
//
bool ExContactBlockTL::eval_f(
   Index         n,
   const Number* x,
   bool          new_x,
   Number&       obj_value
) const
{
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

   obj_value  = 0;
   obj_value += A1.InnerProduct(*x1, *x1);
   obj_value += A2.InnerProduct(*x2, *x2);
   obj_value *= 0.5;
   
   // --- addition
   obj_value -= InnerProduct(*x1, B1);
   obj_value -= InnerProduct(*x2, B2);
   // ---
   
   return true;
}

bool ExContactBlockTL::eval_grad_f(
   Index         n,
   const Number* x,
   bool          new_x,
   Number*       grad_f
) const
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

   // --- addition
   temp1.Add(-1.0, B1);
   temp2.Add(-1.0, B2);
   // ---

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
) const
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
      cons[i] = gapv[i];
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
) const
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
         values[i] = 0.0;
      }
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
         values[k] = 0.0;
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



//
//void ExContactBlockTL::finalize_solution(
//   SolverReturn               status,
//   Index                      n,
//   const Number*              x,
//   const Number*              z_L,
//   const Number*              z_U,
//   Index                      m,
//   const Number*              g,
//   const Number*              lambda,
//   Number                     obj_value,
//   const IpoptData*           ip_data,
//   IpoptCalculatedQuantities* ip_cq
//)
//{}
