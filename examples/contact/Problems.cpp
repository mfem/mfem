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
#include "Problems.hpp"
#include "nodepair.hpp"


using namespace std;
using namespace mfem;


GeneralOptProblem::GeneralOptProblem() {}

void GeneralOptProblem::CalcObjectiveGrad(const BlockVector &x, BlockVector &y) const
{
  Duf(x, y.GetBlock(0));
  Dmf(x, y.GetBlock(1));
}

GeneralOptProblem::~GeneralOptProblem()
{
  block_offsetsx.DeleteAll();
}


// min E(d) s.t. g(d) >= 0
// min_(d,s) E(d) s.t. c(d,s) := g(d) - s = 0, s >= 0

OptProblem::OptProblem() : GeneralOptProblem(), block_offsetsx(3)
{
}

void OptProblem::InitializeParentData(int dimd, int dims)
{
  dimU = dimd;
  dimM = dims;
  dimC = dims;
  block_offsetsx[0] = 0;
  block_offsetsx[1] = dimU;
  block_offsetsx[2] = dimM;
  block_offsetsx.PartialSum();
  ml.SetSize(dimM); ml = 0.0;
  Vector negIdentVec(dimM);
  negIdentVec = -1.0;
  negIdentity  = new SparseMatrix(negIdentVec);
  zeroMatum = nullptr;
  zeroMatmu = nullptr;
  zeroMatmm = nullptr;
}

double OptProblem::CalcObjective(const BlockVector &x) const { return E(x.GetBlock(0)); }

void OptProblem::Duf(const BlockVector &x, Vector &y) const { DdE(x.GetBlock(0), y); }

void OptProblem::Dmf(const BlockVector &x, Vector &y) const { y = 0.0; }

SparseMatrix* OptProblem::Duuf(const BlockVector &x) { return DddE(x.GetBlock(0)); }

SparseMatrix* OptProblem::Dumf(const BlockVector &x) { return nullptr; }

SparseMatrix* OptProblem::Dmuf(const BlockVector &x) { return nullptr; }

SparseMatrix* OptProblem::Dmmf(const BlockVector &x) { return nullptr; }

void OptProblem::c(const BlockVector &x, Vector &y) const // c(u,m) = g(u) - m 
{
  g(x.GetBlock(0), y);
  y.Add(-1.0, x.GetBlock(1));  
}

SparseMatrix* OptProblem::Duc(const BlockVector &x) { return Ddg(x.GetBlock(0)); }

SparseMatrix* OptProblem::Dmc(const BlockVector &x) 
{ 
  return negIdentity;
} 

SparseMatrix* OptProblem::lDuuc(const BlockVector &x, const Vector &l)
{
  return lDddg(x.GetBlock(0), l);
}

SparseMatrix* OptProblem::lDumc(const BlockVector &x, const Vector &l)
{
  return zeroMatum;
}

SparseMatrix* OptProblem::lDmuc(const BlockVector &x, const Vector &l)
{
  return zeroMatmu;
}

SparseMatrix* OptProblem::lDmmc(const BlockVector &x, const Vector &l)
{
  return zeroMatmm;
}

OptProblem::~OptProblem() 
{
  delete negIdentity;
}




//-------------------
ObstacleProblem::ObstacleProblem(FiniteElementSpace *fes, double (*fSource)(const Vector &), 
		double (*obstacleSource)(const Vector &)) : OptProblem(), Vh(fes), twoBounds(false), Hcl(nullptr)
{
  dimD = fes->GetTrueVSize();
  dimS = dimD;
  InitializeParentData(dimD, dimS);

  
  Kform = new BilinearForm(Vh);
  Kform->AddDomainIntegrator(new DiffusionIntegrator);
  Kform->AddDomainIntegrator(new MassIntegrator);
  Kform->Assemble();
  Kform->Finalize();
  K = new SparseMatrix(Kform->SpMat());

  FunctionCoefficient fcoeff(fSource);
  fform = new LinearForm(Vh);
  fform->AddDomainIntegrator(new DomainLFIntegrator(fcoeff));
  fform->Assemble();
  f.SetSize(dimD);
  f.Set(1.0, *fform);
    
  // define obstacle function
  FunctionCoefficient psicoeff(obstacleSource);
  GridFunction psi_gf(Vh);
  psi_gf.ProjectCoefficient(psicoeff);
  psil.SetSize(dimS); psil = 0.0;
  psil.Set(1.0, psi_gf);
  
  // ------ construct dimS x dimD Jacobian with zero columns correspodning to Dirichlet dofs
  Vector one(dimD); one = 1.0;
  J = new SparseMatrix(one);

  // constant term in energy function
  Ce = 0.0;
}


ObstacleProblem::ObstacleProblem(FiniteElementSpace *fes, Vector &x0DC, double (*fSource)(const Vector &), 
		double (*obstacleSource)(const Vector &),
		Array<int> tdof_list) : OptProblem(), Vh(fes), ess_tdof_list(tdof_list), twoBounds(false), Hcl(nullptr)
{
  dimD = fes->GetTrueVSize();
  dimS = dimD;
  InitializeParentData(dimD, dimS);

  xDC.SetSize(dimD);
  xDC.Set(1.0, x0DC);
  // constant term in energy function to ensure
  // that the energy at the optimal 
  // point converges under mesh refinement
  Vector local_DC;
  xDC.GetSubVector(ess_tdof_list, local_DC);
  Ce = pow(local_DC.Norml2(), 2) / 2.;
  
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
  psil.SetSize(dimS); psil = 0.0;
  psil.Set(1.0, psi_gf);
  
  // ------ construct dimS x dimD Jacobian with zero columns correspodning to Dirichlet dofs
  Vector temp(dimS); temp = 1.0;
  temp.SetSubVector(ess_tdof_list, 0.0);
  J = new SparseMatrix(temp);
}

ObstacleProblem::ObstacleProblem(FiniteElementSpace *fes, Vector &x0DC, double (*fSource)(const Vector &), 
		double (*obstacleSourcel)(const Vector &),
		double (*obstacleSourceu)(const Vector &),
		Array<int> tdof_list) : OptProblem(), Vh(fes), ess_tdof_list(tdof_list), twoBounds(true), J(nullptr), Hcl(nullptr)
{
  dimD = fes->GetTrueVSize();
  dimS = dimD;// - ess_tdof_list.Size();
  InitializeParentData(dimD, dimS);

  xDC.SetSize(dimD);
  xDC.Set(1.0, x0DC);
  // constant term in energy function to ensure
  // that the energy at the optimal 
  // point converges under mesh refinement
  Vector local_DC;
  xDC.GetSubVector(ess_tdof_list, local_DC);
  Ce = pow(local_DC.Norml2(), 2) / 2.;
  
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

  // define lower and upper obstacle functions
  FunctionCoefficient psilcoeff(obstacleSourcel);
  GridFunction psil_gf(Vh);
  psil_gf.ProjectCoefficient(psilcoeff);
  psil.SetSize(dimS); psil = 0.0;
  psil.Set(1.0, psil_gf);

  FunctionCoefficient psiucoeff(obstacleSourceu);
  GridFunction psiu_gf(Vh);
  psiu_gf.ProjectCoefficient(psiucoeff);
  psiu.SetSize(dimS); psiu = 0.0;
  psiu.Set(1.0, psiu_gf);
}





double ObstacleProblem::E(const Vector &d) const
{
  Vector Kd(dimD); Kd = 0.0;
  K->Mult(d, Kd);
  return 0.5 * InnerProduct(d, Kd) - InnerProduct(f, d) + Ce;
}

void ObstacleProblem::DdE(const Vector &d, Vector &gradE) const
{
  K->Mult(d, gradE);
  gradE.Add(-1.0, f);
}

SparseMatrix* ObstacleProblem::DddE(const Vector &d)
{
  return K; 
}

// one bound:
// g(d) = d - \psi_l >= 0
//        d - \psi_l - s  = 0
//                    s >= 0
// two bounds:
// g(d) = 1 / 2 (d - \psi_l) (\psi_u - d) >= 0
// when \psi_l < \psi_u, g(d) >= 0 <==> \psi_l <= d <= \psi_u
void ObstacleProblem::g(const Vector &d, Vector &gd) const
{
  if(!twoBounds)
  {
    J->Mult(d, gd);
    gd.Add(-1., psil);
  }
  else
  {
    Vector temp(dimS);
    temp.Set(1.0, d);
    temp.Add(-1.0, psil);
    gd.Set(0.5, temp);

    temp.Set(1.0, psiu);
    temp.Add(-1.0, d);
    gd *= temp;
  }
  gd.SetSubVector(ess_tdof_list, 1.e-8);
}

SparseMatrix* ObstacleProblem::Ddg(const Vector &d)
{
  if(!twoBounds)
  {
    return J;
  }
  else
  {
    if(J != nullptr)
    {
      delete J; J = nullptr;
    }
    Vector temp(dimS);
    temp.Set(-1.0, d);
    temp.Add(0.5, psil);
    temp.Add(0.5, psiu);
    temp.SetSubVector(ess_tdof_list, 0.0);
    J = new SparseMatrix(temp);
    return J;
  }
}

SparseMatrix* ObstacleProblem::lDddg(const Vector &d, const Vector &l)
{
  if(!twoBounds)
  {
    return Hcl;
  }
  else
  {
    if(Hcl != nullptr)
    {
      delete Hcl; Hcl = nullptr;
    }
    Vector temp(dimD);
    temp.Set(-1.0, l);
    Hcl = new SparseMatrix(temp);
    return Hcl;
  }
}

ObstacleProblem::~ObstacleProblem()
{
  delete Kform;
  delete fform;
  delete J;
  delete K;
  if(Hcl != nullptr)
  {
    delete Hcl;
  }
}


// quadratic approximation of the objective
// linear approximation of the gap function constraint
// E(d) = 1 / 2 d^T K d + f^T d
// g(d) = J d + g0
QPOptProblem::QPOptProblem(const SparseMatrix Kin, const SparseMatrix Jin, const Vector fin, const Vector g0in) : OptProblem()
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
  
  zeroMatdd = nullptr;
}

// E(d) = 1 / 2 d^T K d + f^T d
double QPOptProblem::E(const Vector &d) const
{
  Vector Kd(dimD); Kd = 0.0;
  K->Mult(d, Kd);
  return 0.5 * InnerProduct(d, Kd) + InnerProduct(f, d);
}


// gradient(E) = K d + f
void QPOptProblem::DdE(const Vector &d, Vector &gradE) const
{
  K->Mult(d, gradE);
  gradE.Add(1.0, f);
}

// Hessian(E) = K
SparseMatrix* QPOptProblem::DddE(const Vector &d)
{
  return K; 
}

// g(d) = J * d + g0 >= 0
void QPOptProblem::g(const Vector &d, Vector &gd) const
{
  J->Mult(d, gd);
  gd.Add(1.0, g0);
}

// Jacobian(g) = J
SparseMatrix* QPOptProblem::Ddg(const Vector &d)
{
  return J;
}

SparseMatrix* QPOptProblem::lDddg(const Vector &d, const Vector &l)
{
  return zeroMatdd;
}

QPOptProblem::~QPOptProblem()
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

   std::vector<mfem::Vector> tang(2);

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
                  // refNormalSide = (refcrd[(i*dim) + j] > 0.5); // not used
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
   finder.FreeData();
}


/* Constructor. */
ExContactBlockTL::ExContactBlockTL(Mesh * mesh1in, Mesh * mesh2in, int FEorder)
   : 
   OptProblem(),
   mesh1(mesh1in),
   mesh2(mesh2in),
   fec1{nullptr},
   fec2{nullptr},
   fespace1{nullptr},
   fespace2{nullptr},
   nodes1{nullptr},
   nodes2{nullptr},
   x1{nullptr},
   x2{nullptr},
   lambda1_func{nullptr},
   lambda2_func{nullptr},
   mu1_func{nullptr},
   mu2_func{nullptr},
   a1{nullptr},
   a2{nullptr},
   K{nullptr},
   coordsm{nullptr},
   M{nullptr},
   dM{nullptr},
   block_offsets(3)
{

   dim = mesh1->Dimension();
   MFEM_VERIFY(dim == mesh2->Dimension(), "");

   // boundary attribute 2 is the potential contact surface of nodes
   attr.Append(3);
   // boundary attribute 2 is the potential contact surface for master surface
   m_attr.Append(3);

   fec1     = new H1_FECollection(1, dim);
   fespace1 = new FiniteElementSpace(mesh1, fec1, dim, Ordering::byVDIM);
   ndof_1   = fespace1->GetTrueVSize();
   mesh1->SetNodalFESpace(fespace1);

   fec2 = new H1_FECollection(1, dim);
   fespace2 = new FiniteElementSpace(mesh2, fec2, dim, Ordering::byVDIM);
   ndof_2   = fespace2->GetTrueVSize();
   
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
   ess_bdr1 = 0; ess_bdr1[1] = 1;
   Array<int> ess_bdr2(mesh2->bdr_attributes.Max());
   ess_bdr2 = 0; ess_bdr2[1] = 1;
   
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
   
   x1 = new GridFunction(fespace1);
   x2 = new GridFunction(fespace2);
   (*x1) = 0.0;
   (*x2) = 0.0;
  
   Vector delta_vec(dim); delta_vec = 0.0; delta_vec[0] = 0.1;
   VectorConstantCoefficient delta_cf(delta_vec);
   x1->ProjectBdrCoefficient(delta_cf,ess_bdr1);
   fespace1->GetEssentialTrueDofs(ess_bdr1,ess_tdof_list1);
   fespace2->GetEssentialTrueDofs(ess_bdr2,ess_tdof_list2);


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
   a1->Assemble();
   a1->EliminateVDofs(ess_tdof_list1);
   a1->Finalize();
   A1 = (a1->SpMat());
   
   B1.SetSize(ndof_1); B1 = 0.0;
   a1->EliminateVDofsInRHS(ess_tdof_list1, (*x1), B1);

   a2 = new BilinearForm(fespace2);
   a2->AddDomainIntegrator(new ElasticityIntegrator(*lambda2_func, *mu2_func));
   a2->Assemble();
   a2->EliminateVDofs(ess_tdof_list2);
   a2->Finalize();
   A2 = (a2->SpMat());
   
   B2.SetSize(ndof_2); B2 = 0.0;
   a2->EliminateVDofsInRHS(ess_tdof_list2, (*x2), B2);
   
   //block_offsets.SetSize(3);
   block_offsets[0] = 0;
   block_offsets[1] = ndof_1;
   block_offsets[2] = ndof_2;
   block_offsets.PartialSum();
   BlockMatrix ABlock(block_offsets, block_offsets);
   ABlock.SetBlock(0, 0, &A1);
   ABlock.SetBlock(1, 1, &A2);
   K = ABlock.CreateMonolithic();
 
   B = new BlockVector(block_offsets);
   B->GetBlock(0).Set(1.0, B1);
   B->GetBlock(1).Set(1.0, B2);
   // Construct node to segment contact constraint.
   attr.Sort();


   npoints = bdryVerts2.size();
   s_conn.SetSize(npoints);
   xyz.SetSize(dim * npoints);
   xyz = 0.0;


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

   m_conn.SetSize(4*npoints); // only works for linear elements that have 4 vertices!
   coordsm = new DenseMatrix(4*npoints, dim);

   // adding displacement to mesh1 using a fixed grid function from mesh1
   Vector displ(x1->Size()); displ = 0.0;
   add(nodes0, displ, *nodes1); // issues with moving the mesh nodes?

   FindPointsInMesh(*mesh1, xyz, m_conn, m_xi); 

   for (int i=0; i<npoints; i++)
   {
      for (int j=0; j<4; j++)
      {
         for (int k=0; k<dim; k++)
         {
            (*coordsm)(i*4+j,k) = mesh1->GetVertex(m_conn[i*4+j])[k]+
                                  displ[dim*m_conn[i*4+j]+k];
         }
      }
   }
   
   // --- enforcing compatibility with the contactproblem structure
   dimD = ndofs;
   dimS = nnd;
   InitializeParentData(dimD, dimS);
   zeroMatdd = nullptr;
   // ---
   
   M  = new SparseMatrix(nnd,ndofs);
   dM = new std::vector<SparseMatrix>(nnd, SparseMatrix(ndofs,ndofs));

   Assemble_Contact(nnd, npoints, ndofs, xs, m_xi, *coordsm, s_conn, m_conn, gapv, *M, *dM);
   M->Finalize(1,false);

   for (int i=0; i<nnd; i++)
   {
      (*dM)[i].Finalize(1,false);
   }
   assert(M);
}



ExContactBlockTL::~ExContactBlockTL()
{
   delete fec1;
   delete fec2;
   delete fespace1;
   delete fespace2;
   delete x1;
   delete x2;
   delete lambda1_func;
   delete lambda2_func;
   delete mu1_func;
   delete mu2_func;
   delete a1;
   delete a2;
   delete K;
   delete coordsm;
   delete M;
   for (int i=0; i<nnd; i++)
   {
      (*dM)[i].Clear();
   }
   delete dM;
   delete B;
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
   delete M;
   M = new SparseMatrix(nnd,ndofs);
   for (int i=0; i<nnd; i++)
   {
      (*dM)[i].Clear();
   }
   delete dM;
   dM = new std::vector<SparseMatrix>(nnd, SparseMatrix(ndofs,ndofs));

   Assemble_Contact(nnd, npoints, ndofs, xs, m_xi, *coordsm, s_conn, m_conn, gapv, *M,
                    *dM);
   M->Finalize(1,false);
   for (int i=0; i<nnd; i++)
   {
      (*dM)[i].Finalize(1,false);
   }
}


double ExContactBlockTL::E(const Vector &d) const
{
  return (0.5 * K->InnerProduct(d, d) - InnerProduct(d, *B));
}

void ExContactBlockTL::DdE(const Vector &d, Vector &gradE) const
{
  gradE = 0.0;
  K->Mult(d, gradE);
  gradE.Add(-1.0, *B); 
}

SparseMatrix* ExContactBlockTL::DddE(const Vector &d) // second argument for SparseMatrix
{
  return K; 
}


   
// g(d) = d >= 0
void ExContactBlockTL::g(const Vector &d, Vector &gd) const
{
   for (auto i=0; i<ndof_1; i++)
   {
      (*x1)[i] = d(i);
   }
   for (auto i=ndof_1; i<ndofs; i++)
   {
      (*x2)[i-ndof_1] = d(i);
   }
   update_g();
   gd.Set(1.0, gapv);
}

SparseMatrix* ExContactBlockTL::Ddg(const Vector &d)
{
  // !!!!!!!!!!!!!!!!TO DO: call eval_jac_g ....
  // only do so after eval_jac_g has been updated in order 
  // that the gap function Jacobian data that is stored in 
  // the SparseMatrix member data M is updated
  //update_g();
  return M;
}


SparseMatrix* ExContactBlockTL::lDddg(const Vector &d, const Vector &l)
{
  // to do use dMi data
  return zeroMatdd;
}
