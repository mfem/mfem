// Copyright (c) 2010-2021, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.
#include <numeric>

#include "mfem.hpp"

#include "general/globals.hpp"
#include "general/forall.hpp"
#include "linalg/kernels.hpp"


// Kernels addons //////////////////////////////////////////////////////////////
#ifndef MFEM_USE_MPI
#define HYPRE_Int int
typedef int MPI_Session;
#define ParMesh Mesh
#define ParFESpace FESpace
#define GetParMesh GetMesh
#define GlobalTrueVSize GetVSize
#define ParLinearForm LinearForm
#define ParBilinearForm BilinearForm
#define ParGridFunction GridFunction
#define ParFiniteElementSpace FiniteElementSpace
#define PFesGetParMeshGetComm(...)
#define PFesGetParMeshGetComm0(...) 0
#define MPI_Init(...)
#define MPI_Barrier(...)
#define MPI_Finalize()
#define MPI_Comm_size(...)
#define MPI_Comm_rank(...)
#define MPI_Allreduce(src,dst,...) *dst = *src
#define MPI_Reduce(src, dst, n, T,...) *dst = *src
#define NewParMesh(pmesh, mesh, partitioning) pmesh = mesh
#else
#define NewParMesh(pmesh, mesh, partitioning) \
    pmesh = new ParMesh(MPI_COMM_WORLD, *mesh, partitioning);\
    delete mesh;
#endif

using namespace mfem;

namespace mfem
{

mfem::Mesh *CreateMeshEx7(int order);

using FE = mfem::FiniteElement;
using QI = mfem::QuadratureInterpolator;

// Kernels addons //////////////////////////////////////////////////////////////
namespace kernels
{

template<typename T> MFEM_HOST_DEVICE inline
void HouseholderReflect(T *A, const T *v,
                        const T b, const int m, const int n,
                        const int row, const int col)
{
   for (int j = 0; j < n; j++)
   {
      T w = A[0*row + j*col];
      for (int i = 1; i < m; i++) { w += v[i] * A[i*row + j*col]; }
      A[0*row + j*col] -= b * w;
      for (int i = 1; i < m; i++) { A[i*row + j*col] -= b * w * v[i]; }
   }
}

template<int Q1D, typename T> MFEM_HOST_DEVICE inline
void HouseholderApplyQ(T *A, const T *Q, const T *tau,
                       const int k, const int row, const int col)
{
   T v[Q1D];
   for (int ii=0; ii<k; ii++)
   {
      const int i = k-1-ii;
      for (int j = i+1; j < Q1D; j++) { v[j] = Q[j*k+i]; }
      // Apply Householder reflector (I - tau v v^T) coG^T
      HouseholderReflect(&A[i*row], &v[i], tau[i], Q1D-i, Q1D, row, col);
   }
}

template<int D1D, int Q1D, typename T> MFEM_HOST_DEVICE inline
void QRFactorization(T *mat, T *tau)
{
   T v[Q1D];
   DeviceMatrix B(mat, D1D, Q1D);
   for (int i = 0; i < D1D; i++)
   {
      // Calculate Householder vector, magnitude
      T sigma = 0.0;
      v[i] = B(i,i);
      for (int j = i + 1; j < Q1D; j++)
      {
         v[j] = B(i,j);
         sigma += v[j] * v[j];
      }
      T norm = std::sqrt(v[i]*v[i] + sigma); // norm of v[i:m]
      T Rii = -copysign(norm, v[i]);
      v[i] -= Rii;
      // norm of v[i:m] after modification above and scaling below
      //   norm = sqrt(v[i]*v[i] + sigma) / v[i];
      //   tau = 2 / (norm*norm)
      tau[i] = 2 * v[i]*v[i] / (v[i]*v[i] + sigma);
      for (int j=i+1; j<Q1D; j++) { v[j] /= v[i]; }
      // Apply Householder reflector to lower right panel
      HouseholderReflect(&mat[i*D1D+i+1], &v[i], tau[i],
                         Q1D-i, D1D-i-1, D1D, 1);
      // Save v
      B(i,i) = Rii;
      for (int j=i+1; j<Q1D; j++) { B(i,j) = v[j]; }
   }
}

template<int D1D, int Q1D, typename T = double> MFEM_HOST_DEVICE inline
void GetCollocatedGrad(DeviceTensor<2,const T> b,
                       DeviceTensor<2,const T> g,
                       DeviceTensor<2,T> CoG)
{
   T tau[Q1D];
   T B1d[Q1D*D1D];
   T G1d[Q1D*D1D];
   DeviceMatrix B(B1d, D1D, Q1D);
   DeviceMatrix G(G1d, D1D, Q1D);

   for (int d = 0; d < D1D; d++)
   {
      for (int q = 0; q < Q1D; q++)
      {
         B(d,q) = b(q,d);
         G(d,q) = g(q,d);
      }
   }
   QRFactorization<D1D,Q1D>(B1d, tau);
   // Apply Rinv, colograd1d = grad1d Rinv
   for (int i = 0; i < Q1D; i++)
   {
      CoG(0,i) = G(0,i)/B(0,0);
      for (int j = 1; j < D1D; j++)
      {
         CoG(j,i) = G(j,i);
         for (int k = 0; k < j; k++) { CoG(j,i) -= B(j,k)*CoG(k,i); }
         CoG(j,i) /= B(j,j);
      }
      for (int j = D1D; j < Q1D; j++) { CoG(j,i) = 0.0; }
   }
   // Apply Qtranspose, colograd = colograd Qtranspose
   HouseholderApplyQ<Q1D>((T*)CoG, B1d, tau, D1D, 1, Q1D);
}

}  // namespace kernels

// XFL addons //////////////////////////////////////////////////////////////////
namespace xfl
{

struct XElementRestriction : public ElementRestriction
{
   XElementRestriction(const ParFiniteElementSpace *fes,
                       ElementDofOrdering ordering)
      : ElementRestriction(*fes, ordering) { }
   const Array<int> &GatherMap() const { return gatherMap; }
};

/** ****************************************************************************
 * @brief The Operator class
 **************************************************************************** */
template <int DIM> class Operator;

/** ****************************************************************************
 * @brief The 2D Operator class
 **************************************************************************** */
template <>
class Operator<2> : public mfem::Operator
{
protected:
   static constexpr int DIM = 2;
   static constexpr int NBZ = 1;

   const DofToQuad::Mode mode = DofToQuad::TENSOR;
   const int flags = GeometricFactors::JACOBIANS |
                     GeometricFactors::COORDINATES |
                     GeometricFactors::DETERMINANTS;
   const ElementDofOrdering e_ordering = ElementDofOrdering::LEXICOGRAPHIC;

   mfem::ParMesh *mesh;
   const ParFiniteElementSpace *pfes;
   const GridFunction *nodes;
   const mfem::FiniteElementSpace *nfes;
   const int p, q;
   const xfl::XElementRestriction ER;
   const mfem::Operator *NR;
   const Geometry::Type type;
   const IntegrationRule &ir;
   const GeometricFactors *geom;
   const DofToQuad *maps;
   const QuadratureInterpolator *nqi;
   const int SDIM, VDIM, NDOFS, NE, NQ, D1D, Q1D;
   mutable Vector val_xq, grad_xq;
   Vector J0, dx;
   const mfem::Operator *P, *R;

public:
   Operator(const ParFiniteElementSpace *pfes)
      : mfem::Operator(pfes->GetVSize()),
        mesh(pfes->GetParMesh()),
        pfes(pfes),
        nodes((mesh->EnsureNodes(), mesh->GetNodes())),
        nfes(nodes->FESpace()),
        p(pfes->GetFE(0)->GetOrder()),
        q(2 * p + mesh->GetElementTransformation(0)->OrderW()),
        ER(pfes, e_ordering),
        NR(nfes->GetElementRestriction(e_ordering)),
        type(mesh->GetElementBaseGeometry(0)),
        ir(IntRules.Get(type, q)),
        geom(mesh->GetGeometricFactors(ir, flags)),
        maps(&pfes->GetFE(0)->GetDofToQuad(ir, mode)),
        nqi(nfes->GetQuadratureInterpolator(ir)),
        SDIM(mesh->SpaceDimension()),
        VDIM(pfes->GetVDim()),
        NDOFS(pfes->GetNDofs()),
        NE(mesh->GetNE()),
        NQ(ir.GetNPoints()),
        D1D(pfes->GetFE(0)->GetOrder() + 1),
        Q1D(IntRules.Get(Geometry::SEGMENT, ir.GetOrder()).GetNPoints()),
        val_xq(NQ * VDIM * NE),
        grad_xq(NQ * VDIM * DIM * NE),
        J0(SDIM * DIM * NQ * NE),
        dx(NQ * NE, MemoryType::HOST_32),
        P(pfes->GetProlongationMatrix()),
        R(pfes->GetRestrictionMatrix())
   {
      MFEM_VERIFY(DIM == 2, "");
      MFEM_VERIFY(VDIM == 1, "");
      MFEM_VERIFY(SDIM == DIM, "");
      MFEM_VERIFY(NQ == Q1D * Q1D, "");
      MFEM_VERIFY(DIM == mesh->Dimension(), "");
      nqi->SetOutputLayout(QVectorLayout::byVDIM);
      const FiniteElement *fe = nfes->GetFE(0);
      const int vdim = nfes->GetVDim();
      const int nd = fe->GetDof();
      Vector Enodes(vdim * nd * NE);
      NR->Mult(*nodes, Enodes);
      nqi->Derivatives(Enodes, J0);
   }

   virtual void Mult(const mfem::Vector &, mfem::Vector &) const {}

   virtual void QMult(const mfem::Vector &, mfem::Vector &) const {}

   virtual const mfem::Operator *GetProlongation() const { return P; }

   virtual const mfem::Operator *GetRestriction() const { return R; }
};

/** ****************************************************************************
 * @brief The 3D Operator class
 **************************************************************************** */
template <>
class Operator<3> : public mfem::Operator
{
protected:
   static constexpr int DIM = 3;

   const DofToQuad::Mode mode = DofToQuad::TENSOR;
   const int flags = GeometricFactors::JACOBIANS |
                     GeometricFactors::COORDINATES |
                     GeometricFactors::DETERMINANTS;
   const ElementDofOrdering e_ordering = ElementDofOrdering::LEXICOGRAPHIC;

   mfem::ParMesh *mesh;
   const ParFiniteElementSpace *pfes;
   const GridFunction *nodes;
   const mfem::FiniteElementSpace *nfes;
   const int p, q;
   const xfl::XElementRestriction ER;
   const mfem::Operator *NR;
   const Geometry::Type type;
   const IntegrationRule &ir;
   const GeometricFactors *geom;
   const DofToQuad *maps;
   const QuadratureInterpolator *qi, *nqi;
   const int SDIM, VDIM, NDOFS, NE, NQ, D1D, Q1D;
   mutable Vector val_xq, grad_xq;
   Vector J0, dx;
   const mfem::Operator *P, *R;
   mutable Array<double> CoG;

public:
   Operator(const ParFiniteElementSpace *pfes)
      : mfem::Operator(pfes->GetVSize()),
        mesh(pfes->GetParMesh()),
        pfes(pfes),
        nodes((mesh->EnsureNodes(), mesh->GetNodes())),
        nfes(nodes->FESpace()),
        p(pfes->GetFE(0)->GetOrder()),
        q(2 * p + mesh->GetElementTransformation(0)->OrderW()),
        ER(pfes, e_ordering),
        NR(nfes->GetElementRestriction(e_ordering)),
        type(mesh->GetElementBaseGeometry(0)),
        ir(IntRules.Get(type, q)),
        geom(mesh->GetGeometricFactors(ir, flags)),
        maps(&pfes->GetFE(0)->GetDofToQuad(ir, mode)),
        qi(pfes->GetQuadratureInterpolator(ir)),
        nqi(nfes->GetQuadratureInterpolator(ir)),
        SDIM(mesh->SpaceDimension()),
        VDIM(pfes->GetVDim()),
        NDOFS(pfes->GetNDofs()),
        NE(mesh->GetNE()),
        NQ(ir.GetNPoints()),
        D1D(pfes->GetFE(0)->GetOrder() + 1),
        Q1D(IntRules.Get(Geometry::SEGMENT, ir.GetOrder()).GetNPoints()),
        val_xq(NQ * VDIM * NE),
        grad_xq(NQ * VDIM * DIM * NE),
        J0(SDIM * DIM * NQ * NE),
        dx(NQ * NE),
        P(pfes->GetProlongationMatrix()),
        R(pfes->GetRestrictionMatrix())
   {
      MFEM_VERIFY(DIM == 3, "");
      MFEM_VERIFY(VDIM == 1, "");
      MFEM_VERIFY(SDIM == DIM, "");
      MFEM_VERIFY(NQ == Q1D * Q1D * Q1D, "");
      MFEM_VERIFY(DIM == mesh->Dimension(), "");
      qi->SetOutputLayout(QVectorLayout::byVDIM);
      nqi->SetOutputLayout(QVectorLayout::byVDIM);
      const FiniteElement *fe = nfes->GetFE(0);
      const int vdim = nfes->GetVDim();
      const int nd = fe->GetDof();
      Vector Enodes(vdim * nd * NE);
      NR->Mult(*nodes, Enodes);
      nqi->Derivatives(Enodes, J0);
   }

   virtual void Mult(const mfem::Vector &, mfem::Vector &) const {}

   virtual void QMult(const mfem::Vector &, mfem::Vector &) const {}

   virtual const mfem::Operator *GetProlongation() const { return P; }

   virtual const mfem::Operator *GetRestriction() const { return R; }
};

/** ****************************************************************************
 * @brief The Problem struct
 ******************************************************************************/
struct Problem
{
   mfem::Operator *QM {nullptr};
   mfem::ParLinearForm &b;
   Problem(mfem::ParLinearForm &b, mfem::Operator *QM) : QM(QM), b(b) {}
   ~Problem() { delete QM; }
};

/** ****************************************************************************
 * @brief The QForm class
 ******************************************************************************/
class QForm
{
public:
   const char *qs;
   mfem::Operator *QM;
   mfem::ParLinearForm *b = nullptr;
   mfem::ConstantCoefficient *constant_coeff = nullptr;
   mfem::FunctionCoefficient *function_coeff = nullptr;
   mfem::ParFiniteElementSpace *pfes;

public:

   QForm(ParFiniteElementSpace *pfes, const char *qs, mfem::Operator *QM)
      : qs(qs), QM(QM), pfes(pfes) { }

   ~QForm() { }

   // Create problem
   Problem *operator==(QForm &rhs)
   {
      assert(!b);
      mfem::ParLinearForm *b = new mfem::ParLinearForm(rhs.ParFESpace());
      assert(b);
      if (!rhs.ConstantCoeff() && !rhs.FunctionCoeff())
      {
         ConstantCoefficient *cst = new ConstantCoefficient(1.0);
         b->AddDomainIntegrator(new DomainLFIntegrator(*cst));
      }
      else if (rhs.ConstantCoeff())
      {
         ConstantCoefficient *cst = rhs.ConstantCoeff();
         b->AddDomainIntegrator(new DomainLFIntegrator(*cst));
      }
      else if (rhs.FunctionCoeff())
      {
         FunctionCoefficient *func = rhs.FunctionCoeff();
         b->AddDomainIntegrator(new DomainLFIntegrator(*func));
      }
      else
      {
         assert(false);
      }

      return new Problem(*b, QM);
   }

   // + operator on QForms
   QForm &operator+(QForm &)
   {
      assert(false);  // not supported
      return *this;
   }

   mfem::ParFiniteElementSpace *ParFESpace() { return pfes; }
   mfem::ConstantCoefficient *ConstantCoeff() const { return constant_coeff; }
   mfem::FunctionCoefficient *FunctionCoeff() const { return function_coeff; }
};

/** ****************************************************************************
 * @brief Function class
 ******************************************************************************/
class Function : public ParGridFunction
{
public:
   Function(ParFiniteElementSpace *pfes) : ParGridFunction(pfes)
   {
      assert(pfes);
      assert(pfes->GlobalTrueVSize() > 0);
   }
   void operator=(double value) { ParGridFunction::operator=(value); }
   int geometric_dimension() { return fes->GetMesh()->SpaceDimension(); }
   ParFiniteElementSpace *ParFESpace() { return ParGridFunction::ParFESpace(); }
   const ParFiniteElementSpace *ParFESpace() const { return ParGridFunction::ParFESpace(); }
   ConstantCoefficient *ConstantCoeff() const { return nullptr; }
   FunctionCoefficient *FunctionCoeff() const { return nullptr; }
};

/** ****************************************************************************
 * @brief TrialFunction class
 ******************************************************************************/
class TrialFunction : public Function
{
public:
   TrialFunction(ParFiniteElementSpace *pfes) : Function(pfes) { }
   ~TrialFunction() { }
};

/** ****************************************************************************
 * @brief TestFunction class
 ******************************************************************************/
class TestFunction : public Function
{
public:
   TestFunction(ParFiniteElementSpace *pfes) : Function(pfes) { }
   TestFunction(const TestFunction &) = default;
   ~TestFunction() { }
};

/** ****************************************************************************
 * @brief Constant class
 ******************************************************************************/
class Constant
{
   const double value = 0.0;
   ConstantCoefficient *cst = nullptr;

public:
   Constant(double val) : value(val), cst(new ConstantCoefficient(val)) { }
   ~Constant() { /*delete cst;*/ }
   ParFiniteElementSpace *ParFESpace() const { return nullptr; }
   double Value() const { return value; }
   double Value() { return value; }
   double operator*(TestFunction&) { return 0.0; }
   ConstantCoefficient *ConstantCoeff() const { return cst; }
   FunctionCoefficient *FunctionCoeff() const { return nullptr; }
   operator const double *() const { return nullptr; }  // qf eval
};

/** ****************************************************************************
 * @brief Expressions
 ******************************************************************************/
class Expression
{
   FunctionCoefficient *fct = nullptr;

public:
   Expression(std::function<double(const Vector &)> F)
      : fct(new FunctionCoefficient(F)) { }
   ~Expression() { /*delete fct;*/ }
   ParFiniteElementSpace *ParFESpace() const { return nullptr; }  // qf args
   ConstantCoefficient *ConstantCoeff() const { return nullptr; }
   FunctionCoefficient *FunctionCoeff() const { return fct; }
   operator const double *() const { return nullptr; }  // qf eval
   // double operator *(TestFunction &v) { return 0.0;} // qf eval
};

/** ****************************************************************************
 * @brief Mesh
 ******************************************************************************/
static mfem::ParMesh *MeshToPMesh(mfem::Mesh *mesh)
{
   int num_procs = 1;
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
   int n111[3] {1, 1, 1};
   int n211[3] {2, 1, 1};
   int n221[3] {2, 2, 1};
   int n222[3] {2, 2, 2};
   int n422[3] {4, 2, 2};
   int n442[3] {4, 4, 2};
   int n444[3] {4, 4, 4};
   int *nxyz = (num_procs == 1 ? n111 :
                num_procs == 2 ? n211 :
                num_procs == 4 ? n221 :
                num_procs == 8 ? n222 :
                num_procs == 16 ? n422 :
                num_procs == 32 ? n442 :
                num_procs == 64 ? n444 : nullptr);
   assert(nxyz);
   const int mesh_p = 1;
   mesh->SetCurvature(mesh_p, false, -1, Ordering::byNODES);
   //int *partitioning = mesh->CartesianPartitioning(nxyz);
   ParMesh *pmesh = nullptr;
   NewParMesh(pmesh, mesh, partitioning);
   return pmesh;
}

mfem::ParMesh &Mesh(const char *mesh_file)  // and & for mesh
{
   return *MeshToPMesh(new mfem::Mesh(mesh_file, 1, 1));
}

#ifdef MFEM_USE_MPI
mfem::ParMesh &Mesh(mfem::Mesh *mesh) { return *MeshToPMesh(mesh); }

mfem::ParMesh &Mesh(mfem::ParMesh *pmesh) { return *pmesh; }
#else
mfem::Mesh &Mesh(mfem::Mesh *mesh) { return *mesh; }
#endif

mfem::ParMesh &UnitSquareMesh(int nx, int ny)
{
   Element::Type quad = Element::Type::QUADRILATERAL;
   mfem::Mesh mesh = mfem::Mesh::MakeCartesian2D(nx, ny, quad);
   return *MeshToPMesh(&mesh);
}

mfem::ParMesh &UnitHexMesh(int nx, int ny, int nz)
{
   Element::Type hex = Element::Type::HEXAHEDRON;
   mfem::Mesh mesh = mfem::Mesh::MakeCartesian3D(nx, ny, nz, hex);
   return *MeshToPMesh(&mesh);
}

/** ****************************************************************************
 * @brief Device
 ******************************************************************************/
mfem::Device Device(const char *device_config) { return {device_config}; }

/** ****************************************************************************
 * @brief FiniteElement
 ******************************************************************************/
FiniteElementCollection *FiniteElement(std::string family, int type, int p)
{
   MFEM_VERIFY(family == "Lagrange", "Unsupported family!");
   MFEM_VERIFY(type == Element::Type::QUADRILATERAL ||
               type == Element::Type::HEXAHEDRON, "Unsupported type!");
   const int dim = (type == Element::Type::QUADRILATERAL) ? 2 :
                   (type == Element::Type::HEXAHEDRON)  ? 3 : 0;
   const int btype = BasisType::GaussLobatto;
   return new H1_FECollection(p, dim, btype);
}

/** ****************************************************************************
 * @brief Function Spaces
 ******************************************************************************/
class FunctionSpace : public ParFiniteElementSpace {};

ParFiniteElementSpace *FunctionSpace(mfem::ParMesh *pmesh,
                                     std::string family,
                                     int p)
{
   assert(false);
   const int dim = pmesh->Dimension();
   MFEM_VERIFY(family == "P", "Unsupported FE!");
   FiniteElementCollection *fec = new H1_FECollection(p, dim);
   return new ParFiniteElementSpace(pmesh, fec);
}

ParFiniteElementSpace *FunctionSpace(mfem::ParMesh &pmesh, std::string f, int p)
{
   assert(false);
   return FunctionSpace(&pmesh, f, p);
}

ParFiniteElementSpace *FunctionSpace(mfem::ParMesh &pmesh,
                                     FiniteElementCollection *fec)
{
   const int vdim = 1;
   const Ordering::Type ordering = Ordering::byNODES;
   ParFiniteElementSpace *pfes =
      new ParFiniteElementSpace(&pmesh, fec, vdim, ordering);
   return pfes;
}

ParFiniteElementSpace *FunctionSpace(mfem::ParMesh &pmesh,
                                     FiniteElementCollection *fec,
                                     const int vdim)
{
   assert(false);
   return new ParFiniteElementSpace(&pmesh, fec, vdim);
}

/** ****************************************************************************
 * @brief Vector Function Space
 ******************************************************************************/
ParFiniteElementSpace *VectorFunctionSpace(mfem::ParMesh *pmesh,
                                           std::string family,
                                           const int p)
{
   const int dim = pmesh->Dimension();
   MFEM_VERIFY(family == "P", "Unsupported FE!");
   FiniteElementCollection *fec = new H1_FECollection(p, dim);
   return new ParFiniteElementSpace(pmesh, fec, dim);
}

ParFiniteElementSpace *VectorFunctionSpace(mfem::ParMesh &pmesh,
                                           std::string family,
                                           const int p)
{
   return VectorFunctionSpace(&pmesh, family, p);
}

ParFiniteElementSpace *VectorFunctionSpace(mfem::ParMesh &pmesh,
                                           FiniteElementCollection *fec)
{
   return new ParFiniteElementSpace(&pmesh, fec, pmesh.Dimension());
}

/** ****************************************************************************
 * @brief Boundary Conditions
 ******************************************************************************/
Array<int> DirichletBC(mfem::ParFiniteElementSpace *pfes)
{
   assert(pfes);
   Array<int> ess_tdof_list;
   mfem::ParMesh *pmesh = pfes->GetParMesh();
   if (pmesh->bdr_attributes.Size())
   {
      Array<int> ess_bdr(pmesh->bdr_attributes.Max());
      ess_bdr = 1;
      pfes->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
   }
   return ess_tdof_list;
}

/** ****************************************************************************
 * @brief Math namespace
 ******************************************************************************/
namespace math
{

Constant Pow(Function &gf, double exp)
{
   return Constant(gf.Vector::Normlp(exp));
}

}  // namespace math

/** ****************************************************************************
 * @brief solve with boundary conditions
 ******************************************************************************/
int solve(xfl::Problem *pb, xfl::Function &x, Array<int> ess_tdof_list)
{
   assert(x.FESpace());
   ParFiniteElementSpace *fes = x.ParFESpace();
   MFEM_VERIFY(UsesTensorBasis(*fes), "FE Space must Use Tensor Basis!");

   Vector B, X;
   pb->b.Assemble();
   mfem::Operator *A = nullptr;

   mfem::Operator &op = *(pb->QM);
   op.FormLinearSystem(ess_tdof_list, x, pb->b, A, X, B);
   CG(*A, B, X, 1, 400, 1e-12, 0.0);
   op.RecoverFEMSolution(X, pb->b, x);
   x.HostReadWrite();

   delete pb;
   return 0;
}

/// solve with empty boundary conditions
int solve(xfl::Problem *pb, xfl::Function &x)
{
   Array<int> empty_tdof_list;
   return solve(pb, x, empty_tdof_list);
}

/** ****************************************************************************
 * @brief benchmark this prblem with boundary conditions
 ******************************************************************************/
int benchmark(xfl::Problem *pb, xfl::Function &x, Array<int> ess_tdof_list,
              const double rtol, const int max_it, const int print_lvl)
{
   assert(x.ParFESpace());
   ParFiniteElementSpace *pfes = x.ParFESpace();
   assert(pfes->GlobalTrueVSize() > 0);
   MFEM_VERIFY(UsesTensorBasis(*pfes), "FE Space must Use Tensor Basis!");

   mfem::ParLinearForm &b = pb->b;
   b.Assemble();

   Vector B, X;
   mfem::Operator *A;
   mfem::Operator &a = *(pb->QM);
   a.FormLinearSystem(ess_tdof_list, x, b, A, X, B);

#ifndef MFEM_USE_MPI
   CGSolver cg;
#else
   CGSolver cg(MPI_COMM_WORLD);
#endif

   cg.SetRelTol(rtol);
   cg.SetOperator(*A);

   // Warm-up CG solve (in case of JIT to avoid timing it)
   {
      Vector Y(X);
      cg.SetMaxIter(2);
      cg.SetPrintLevel(-1);
      cg.Mult(B, Y);
   }

   // benchmark this problem
   {
      tic_toc.Clear();
      cg.SetMaxIter(max_it);
      cg.SetPrintLevel(print_lvl);
      {
         tic_toc.Start();
         cg.Mult(B, X);
         tic_toc.Stop();
      }
   }

   // MFEM_VERIFY(cg.GetConverged(), "CG did not converged!");
   MFEM_VERIFY(cg.GetNumIterations() <= max_it, "");
   a.RecoverFEMSolution(X, b, x);
   x.HostReadWrite();

   int myid = 0;
   MPI_Comm_rank(MPI_COMM_WORLD, &myid);

   const double rt = tic_toc.RealTime();
   double rt_min, rt_max;
   MPI_Reduce(&rt, &rt_min, 1, MPI_DOUBLE, MPI_MIN, 0,
              pfes->GetParMesh()->GetComm());
   MPI_Reduce(&rt, &rt_max, 1, MPI_DOUBLE, MPI_MAX, 0,
              pfes->GetParMesh()->GetComm());

   const HYPRE_Int dofs = pfes->GlobalTrueVSize();
   const int cg_iter = cg.GetNumIterations();
   const double mdofs_max = ((1e-6 * dofs) * cg_iter) / rt_max;
   const double mdofs_min = ((1e-6 * dofs) * cg_iter) / rt_min;

   if (myid == 0)
   {
      mfem::out << "Number of finite element unknowns: " << dofs <<  std::endl;
      mfem::out << "Total CG time:    " << rt_max << " (" << rt_min << ") sec."
                << std::endl;
      mfem::out << "Time per CG step: "
                << rt_max / cg_iter << " ("
                << rt_min / cg_iter << ") sec." << std::endl;
      mfem::out << "\033[32m";
      mfem::out << "\"DOFs/sec\" in CG: " << mdofs_max << " ("
                << mdofs_min << ") million.";
      mfem::out << "\033[m" << std::endl;
   }
   delete pb;
   return 0;
}

int benchmark(xfl::Problem *pb, xfl::Function &x, Array<int> ess_tdof_list)
{
   return benchmark(pb, x, ess_tdof_list, 1e-12, 200, -1);
}

/** ****************************************************************************
 * @brief plot the x gridfunction
 ******************************************************************************/
int plot(xfl::Function &x)
{
   int num_procs = 1, myid = 0;
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
   MPI_Comm_rank(MPI_COMM_WORLD, &myid);

   ParFiniteElementSpace *fes = x.ParFESpace();
   assert(fes);
   mfem::ParMesh *pmesh = fes->GetParMesh();
   assert(pmesh);
   char vishost[] = "localhost";
   int visport = 19916;
   socketstream sol_sock(vishost, visport);
   sol_sock << "parallel " << num_procs << " " << myid << "\n";
   sol_sock.precision(8);
   sol_sock << "solution\n" << *pmesh << x << std::flush;
   return 0;
}

/** ****************************************************************************
 * @brief plot the mesh
 ******************************************************************************/
int plot(mfem::ParMesh *pmesh)
{
   int num_procs = 1, myid = 0;
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
   MPI_Comm_rank(MPI_COMM_WORLD, &myid);

   char vishost[] = "localhost";
   int visport = 19916;
   socketstream sol_sock(vishost, visport);
   sol_sock << "parallel " << num_procs << " " << myid << "\n";
   sol_sock.precision(8);
   sol_sock << "mesh\n" << *pmesh << std::flush;
   return 0;
}

/** ****************************************************************************
 * @brief save the x gridfunction
 ******************************************************************************/
int save(xfl::Function &x, const char *filename)
{
   int myid = 0;
   MPI_Comm_rank(MPI_COMM_WORLD, &myid);
   std::ostringstream sol_name;
   sol_name << filename << "." << std::setfill('0') << std::setw(6) << myid;

   std::ofstream sol_ofs(sol_name.str().c_str());
   sol_ofs.precision(8);
   x.Save(sol_ofs);
   return 0;
}

/** ****************************************************************************
 * @brief save the x gridfunction
 ******************************************************************************/
int save(mfem::ParMesh &mesh, const char *filename)
{
   int myid = 0;
   MPI_Comm_rank(MPI_COMM_WORLD, &myid);
   std::ostringstream mesh_name;
   mesh_name << filename << "." << std::setfill('0') << std::setw(6) << myid;

   std::ofstream mesh_ofs(mesh_name.str().c_str());
   mesh_ofs.precision(8);
   mesh.Print(mesh_ofs);
   return 0;
}

//constexpr int point = Element::Type::POINT;
//constexpr int segment = Element::Type::SEGMENT;
//constexpr int triangle = Element::Type::TRIANGLE;
constexpr int quadrilateral = Element::Type::QUADRILATERAL;
//constexpr int tetrahedron = Element::Type::TETRAHEDRON;
constexpr int hexahedron = Element::Type::HEXAHEDRON;
//constexpr int wedge = Element::Type::WEDGE;

}  // namespace xfl

template <typename... Args>
void print(const char *fmt, Args... args)
{
   mfem::out << std::flush;
   std::printf(fmt, args...);
   mfem::out << std::endl;
}

inline bool UsesTensorBasis(const FiniteElementSpace *fes)
{
   return mfem::UsesTensorBasis(*fes);
}

int sym(int u) { return u; }
int dot(int u, int v) { return u * v; }

/// CPP addons /////////////////////////////////////////////////////////////////
namespace cpp
{

// *****************************************************************************
struct Range : public std::vector<int>
{
   Range(const int n) : vector<int>(n)
   {
      // Fills the range with sequentially increasing values
      std::iota(std::begin(*this), std::end(*this), 0);
   }
};

}  // namespace mfem

}  // namespace cpp
