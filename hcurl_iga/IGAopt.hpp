#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include <functional>
#include "../../fem/integ/bilininteg_hcurl_kernels.hpp"
#include "../../fem/integ/bilininteg_diffusion_kernels.hpp"
// #include "nonlininteg.hpp"
// #include "../../linalg/dtensor.hpp"
// #include "../../general/forall.hpp"
// #include "../qfunction.hpp"

namespace mfem
{

/** @brief Structure representing the matrices/tensors needed to evaluate (in
    reference space) the values, gradients, divergences, or curls of a
    FiniteElement at the quadrature points of a given IntegrationRule. */
/** Objects of this type are typically created and owned by the respective
    FiniteElement object. */
class NURBSDofToQuad
{
public:
   Array<real_t> Bo,Bc;
   Array<real_t> Bot,Bct;
   Array<real_t> Go,Gc;
   Array<real_t> Got,Gct;
   int ne,Q1D,nq,cD1D,D1D;
   Vector jac;
   IntegrationRule ips_for_jac;
};

const NURBSDofToQuad &GetNURBSDofToQuad3D(Mesh *mesh, const FiniteElementSpace *fes, const IntegrationRule *ir)
{
   NURBSDofToQuad *d2q = nullptr;
   d2q = new NURBSDofToQuad;
   d2q->ne = mesh->NURBSext->GetNE();
   int dim = mesh->Dimension();
   int ne=d2q->ne;
   int patch;
   int nurbs_order = mesh->NURBSext->GetOrder();
   d2q->Q1D = ir->GetNPoints();
   int Q1D = d2q->Q1D;

   d2q->nq = Q1D*Q1D*Q1D;
   int nq = d2q->nq; 
   d2q->D1D = nurbs_order+1;
   d2q->cD1D = nurbs_order+2;
   
   int D1D = d2q->D1D;
   int cD1D = d2q->cD1D;

   d2q->Bo.SetSize(ne*dim*D1D*Q1D);
   d2q->Go.SetSize(ne*dim*D1D*Q1D);
   d2q->Bc.SetSize(ne*dim*cD1D*Q1D);
   d2q->Gc.SetSize(ne*dim*cD1D*Q1D);
   d2q->Bot.SetSize(ne*dim*D1D*Q1D);
   d2q->Got.SetSize(ne*dim*D1D*Q1D);
   d2q->Bct.SetSize(ne*dim*cD1D*Q1D);
   d2q->Gct.SetSize(ne*dim*cD1D*Q1D);
   d2q->Bo = 0;
   d2q->Go = 0;
   d2q->Bc = 0;
   d2q->Gc = 0;
   d2q->Bot = 0;
   d2q->Got = 0;
   d2q->Bct = 0;
   d2q->Gct = 0;
   d2q->jac.SetSize(dim * dim * nq * ne);
   d2q->ips_for_jac.SetSize(nq);
   // Set basis functions and gradients for elements
   // mfem::out<<"IGAopt.hpp row 70 ****************************"<<std::endl;
   for (int qz=0; qz<Q1D; ++qz)
   {
      for (int qy=0; qy<Q1D; ++qy)
      {
         for (int qx=0; qx<Q1D; ++qx)
         {
            const int p = qx + (qy * Q1D) + (qz * Q1D * Q1D);
            d2q->ips_for_jac[p].x = (*ir)[qx].x;
            d2q->ips_for_jac[p].weight = (*ir)[qx].weight;
            d2q->ips_for_jac[p].y = (*ir)[qy].x;
            d2q->ips_for_jac[p].weight *= (*ir)[qy].weight;
            d2q->ips_for_jac[p].z = (*ir)[qz].x;
            d2q->ips_for_jac[p].weight *= (*ir)[qz].weight;
         }
      }
   }
   //mfem::out<<std::endl;
   for(int e = 0; e < ne; e++)
   {
      const FiniteElement *fe = fes->GetFE(e);
      const NURBS_HCurl3DFiniteElement *el =
                                    dynamic_cast<const NURBS_HCurl3DFiniteElement*>(fe);
      MFEM_VERIFY(el != NULL, "Only NURBS_HCurl3DFiniteElement is supported!");
      patch = mesh->NURBSext->GetElementPatch(e);
      Array<const KnotVector*> pkv;
      Array<const KnotVector*> cpkv;
      mesh->NURBSext->GetPatchKnotVectors(patch, pkv);
      MFEM_VERIFY(pkv.Size() == dim, "");
      cpkv.SetSize(dim);
      int t = 1;
      for(int i = 0; i < dim; i++ )
      {
         cpkv[i] = pkv[i]->DegreeElevate(t);
      }
      ElementTransformation *tr = mesh->GetElementTransformation(e);
      for (int qz=0; qz<Q1D; ++qz)
      {
         for (int qy=0; qy<Q1D; ++qy)
         {
            for (int qx=0; qx<Q1D; ++qx)
            {
               const int p = qx + (qy * Q1D) + (qz * Q1D * Q1D);
               tr->SetIntPoint(&(d2q->ips_for_jac[p]));
               const DenseMatrix& Jp = tr->Jacobian();
               for (int i=0; i<dim; ++i)
                  for (int j=0; j<dim; ++j)
                  {
                     d2q->jac[p + ((i + (j * dim)) * nq) + dim*dim*nq*e] = Jp(i,j);
                  }
            }
         }
      }
      for (int d=0; d<dim; ++d)
      {
         MFEM_VERIFY(pkv[d]->GetOrder() == nurbs_order, "Order must be same!!!");
         MFEM_VERIFY(cpkv[d]->GetOrder() == nurbs_order+1, "closed basis order should be open basis order add one!!!");

         Vector shapeKV(D1D);
         Vector dshapeKV(D1D);

         Vector cshapeKV(cD1D);
         Vector cdshapeKV(cD1D);
         for (int i = 0; i < Q1D; i++)
         {
            const IntegrationPoint &ip = ir->IntPoint(i);
            const int ijk = el->GetIJK()[d];
           
            pkv[d]->CalcShape(shapeKV, ijk, ip.x);
            pkv[d]->CalcDShape(dshapeKV, ijk, ip.x);

            cpkv[d]->CalcShape(cshapeKV, ijk, ip.x);
            cpkv[d]->CalcDShape(cdshapeKV, ijk, ip.x);
            // Put shapeKV into array B storing shapes for all points.
            // TODO: This should be based on NURBS3DFiniteElement::CalcShape and CalcDShape.
            // For now, it works under the assumption that all NURBS weights are 1.
            for (int j=0; j<D1D; ++j)
            {
               d2q->Bo[((e*dim+d)*D1D+j)*Q1D+i] = d2q->Bot[((e*dim+d)*Q1D+i)*D1D+j] = shapeKV[j];//num_el*dim*D1D*Q1D
               d2q->Go[((e*dim+d)*D1D+j)*Q1D+i] = d2q->Got[((e*dim+d)*Q1D+i)*D1D+j] = dshapeKV[j];
            }
            for (int j=0; j<cD1D; ++j)
            {
               d2q->Bc[((e*dim+d)*cD1D+j)*Q1D+i] = d2q->Bct[((e*dim+d)*Q1D+i)*cD1D+j] =  cshapeKV[j];
               d2q->Gc[((e*dim+d)*cD1D+j)*Q1D+i] = d2q->Gct[((e*dim+d)*Q1D+i)*cD1D+j] = cdshapeKV[j];
            }
         }
      }
   }

   return *d2q;
}

const NURBSDofToQuad &GetNURBSDofToQuad2D(Mesh *mesh, const FiniteElementSpace *fes, const IntegrationRule *ir)
{
   NURBSDofToQuad *d2q = nullptr;
   d2q = new NURBSDofToQuad;
   d2q->ne = mesh->NURBSext->GetNE();
   int dim = mesh->Dimension();
   int ne=d2q->ne;
   int patch;
   int nurbs_order = mesh->NURBSext->GetOrder();
   d2q->Q1D = ir->GetNPoints();
   int Q1D = d2q->Q1D;

   d2q->nq = Q1D*Q1D;
   int nq = d2q->nq; 
   d2q->D1D = nurbs_order+1;
   d2q->cD1D = nurbs_order+2;
   
   int D1D = d2q->D1D;
   int cD1D = d2q->cD1D;

   d2q->Bo.SetSize(ne*dim*D1D*Q1D);
   d2q->Go.SetSize(ne*dim*D1D*Q1D);
   d2q->Bc.SetSize(ne*dim*cD1D*Q1D);
   d2q->Gc.SetSize(ne*dim*cD1D*Q1D);
   d2q->Bot.SetSize(ne*dim*D1D*Q1D);
   d2q->Got.SetSize(ne*dim*D1D*Q1D);
   d2q->Bct.SetSize(ne*dim*cD1D*Q1D);
   d2q->Gct.SetSize(ne*dim*cD1D*Q1D);
   d2q->Bo = 0;
   d2q->Go = 0;
   d2q->Bc = 0;
   d2q->Gc = 0;
   d2q->Bot = 0;
   d2q->Got = 0;
   d2q->Bct = 0;
   d2q->Gct = 0;
   d2q->jac.SetSize(dim * dim * nq * ne);
   d2q->ips_for_jac.SetSize(nq);
   // Set basis functions and gradients for elements
   // mfem::out<<"IGAopt.hpp row 70 ****************************"<<std::endl;
   for (int qy=0; qy<Q1D; ++qy)
   {
      for (int qx=0; qx<Q1D; ++qx)
      {
            const int p = qx + (qy * Q1D);
            d2q->ips_for_jac[p].x = (*ir)[qx].x;
            d2q->ips_for_jac[p].weight = (*ir)[qx].weight;
            d2q->ips_for_jac[p].y = (*ir)[qy].x;
            d2q->ips_for_jac[p].weight *= (*ir)[qy].weight;
      }
   }
   //mfem::out<<std::endl;
   for(int e = 0; e < ne; e++)
   {
      const FiniteElement *fe = fes->GetFE(e);
      const NURBS_HCurl2DFiniteElement *el =
                                    dynamic_cast<const NURBS_HCurl2DFiniteElement*>(fe);
      MFEM_VERIFY(el != NULL, "Only NURBS_HCurl2DFiniteElement is supported!");
      patch = mesh->NURBSext->GetElementPatch(e);
      Array<const KnotVector*> pkv;
      Array<const KnotVector*> cpkv;
      mesh->NURBSext->GetPatchKnotVectors(patch, pkv);
      MFEM_VERIFY(pkv.Size() == dim, "");
      cpkv.SetSize(dim);
      int t = 1;
      for(int i = 0; i < dim; i++ )
      {
         cpkv[i] = pkv[i]->DegreeElevate(t);
      }
      ElementTransformation *tr = mesh->GetElementTransformation(e);
      for (int qy=0; qy<Q1D; ++qy)
      {
         for (int qx=0; qx<Q1D; ++qx)
         {
            const int p = qx + (qy * Q1D);
            tr->SetIntPoint(&(d2q->ips_for_jac[p]));
            const DenseMatrix& Jp = tr->Jacobian();
            for (int i=0; i<dim; ++i)
               for (int j=0; j<dim; ++j)
               {
                  d2q->jac[p + ((i + (j * dim)) * nq) + dim*dim*nq*e] = Jp(i,j);
               }
          }
      }
      for (int d=0; d<dim; ++d)
      {
         MFEM_VERIFY(pkv[d]->GetOrder() == nurbs_order, "Order must be same!!!");
         MFEM_VERIFY(cpkv[d]->GetOrder() == nurbs_order+1, "closed basis order should be open basis order add one!!!");

         Vector shapeKV(D1D);
         Vector dshapeKV(D1D);

         Vector cshapeKV(cD1D);
         Vector cdshapeKV(cD1D);
         for (int i = 0; i < Q1D; i++)
         {
            const IntegrationPoint &ip = ir->IntPoint(i);
            const int ijk = el->GetIJK()[d];
           
            pkv[d]->CalcShape(shapeKV, ijk, ip.x);
            pkv[d]->CalcDShape(dshapeKV, ijk, ip.x);

            cpkv[d]->CalcShape(cshapeKV, ijk, ip.x);
            cpkv[d]->CalcDShape(cdshapeKV, ijk, ip.x);
            // Put shapeKV into array B storing shapes for all points.
            // TODO: This should be based on NURBS3DFiniteElement::CalcShape and CalcDShape.
            // For now, it works under the assumption that all NURBS weights are 1.
            for (int j=0; j<D1D; ++j)
            {
               d2q->Bo[((e*dim+d)*D1D+j)*Q1D+i] = d2q->Bot[((e*dim+d)*Q1D+i)*D1D+j] = shapeKV[j];//num_el*dim*D1D*Q1D
               d2q->Go[((e*dim+d)*D1D+j)*Q1D+i] = d2q->Got[((e*dim+d)*Q1D+i)*D1D+j] = dshapeKV[j];
            }
            for (int j=0; j<cD1D; ++j)
            {
               d2q->Bc[((e*dim+d)*cD1D+j)*Q1D+i] = d2q->Bct[((e*dim+d)*Q1D+i)*cD1D+j] =  cshapeKV[j];
               d2q->Gc[((e*dim+d)*cD1D+j)*Q1D+i] = d2q->Gct[((e*dim+d)*Q1D+i)*cD1D+j] = cdshapeKV[j];
            }
         }
      }
   }

   return *d2q;
}

const NURBSDofToQuad &GetNURBSDofToQuad(Mesh *mesh, const FiniteElementSpace *fes, const IntegrationRule *ir)
{
   int dim = mesh->Dimension();
   if(dim == 3)
   {
      return GetNURBSDofToQuad3D(mesh, fes, ir);
   }
   else if(dim == 2)
   {
      return GetNURBSDofToQuad2D(mesh, fes, ir);
   }
   else{
      MFEM_ABORT("No support dimension !!!");
   }
}

class NURBSCurlCurlIntegrator: public CurlCurlIntegrator
{

private:
   // PA extension
   int dim;
   Vector pa_data;
   bool symmetric = true; ///< False if using a nonsymmetric matrix coefficient
   int numPatches = 0;
   static constexpr int numTypes = 2;  // Number of rule types
   const NURBSDofToQuad *maps;

public:

   NURBSCurlCurlIntegrator() { Q = NULL; DQ = NULL; MQ = NULL; integrationMode = Mode::ELEMENTWISE;}
   /// Construct a bilinear form integrator for Nedelec elements
   NURBSCurlCurlIntegrator(Coefficient &q, const IntegrationRule *ir = NULL) :
      CurlCurlIntegrator(q,ir){ integrationMode = Mode::ELEMENTWISE; }
   NURBSCurlCurlIntegrator(DiagonalMatrixCoefficient &dq,
                      const IntegrationRule *ir = NULL) :
      CurlCurlIntegrator(dq, ir){ integrationMode = Mode::ELEMENTWISE;}
   NURBSCurlCurlIntegrator(MatrixCoefficient &mq, const IntegrationRule *ir = NULL) :
      CurlCurlIntegrator(mq, ir){integrationMode = Mode::ELEMENTWISE; }

   using BilinearFormIntegrator::AssemblePA;
   virtual void AssemblePA(const FiniteElementSpace &fes);
   virtual void AddMultPA(const Vector &x, Vector &y) const;
   virtual void AddMultTransposePA(const Vector &x, Vector &y) const;
   virtual void AddMultPA3D(  const int d1d,
                              const int cd1d,
                              const int q1d,
                              const bool symmetric,
                              const int NE,
                              const Array<real_t> &bo,
                              const Array<real_t> &bc,
                              const Array<real_t> &bot,
                              const Array<real_t> &bct,
                              const Array<real_t> &gc,
                              const Array<real_t> &gct,
                              const Vector &pa_data,
                              const Vector &x, Vector &y) const;
   virtual void AddMultPA2D(const int D1D,
                       const int Q1D,
                       const int NE,
                       const Array<real_t> &bo,
                       const Array<real_t> &bot,
                       const Array<real_t> &gc,
                       const Array<real_t> &gct,
                       const Vector &pa_data,
                       const Vector &x,
                       Vector &y) const;
   virtual void AssembleDiagonalPA(Vector& diag);
   virtual void AssembleDiagonalPA3D(const Array<real_t> &bo,
                                    const Array<real_t> &bc,
                                    const Array<real_t> &go,
                                    const Array<real_t> &gc,
                                    const Vector &pa_data,
                                    Vector& diag);
   virtual void AssembleDiagonalPA2D(Vector& diag);
   Coefficient *GetCoefficient() const { return Q; }
   ~NURBSCurlCurlIntegrator() {if(maps){delete maps;}};
};

void NURBSCurlCurlIntegrator::AssemblePA(const FiniteElementSpace &fes)
{
   // Assumes tensor-product elements
   Mesh *mesh = fes.GetMesh();
   const FiniteElement *el = fes.GetFE(0);
   const int dims = el->GetDim();
   dim = mesh->Dimension();
   int nurbs_order = mesh->NURBSext->GetOrder();
   int ir_order = 2*nurbs_order;
   // Assume all the same order
   const IntegrationRule *ir1d = &IntRules.Get(Geometry::SEGMENT, ir_order);
   maps = &GetNURBSDofToQuad(mesh, &fes, ir1d);
   quad1D = maps->Q1D;

   QuadratureSpace qs(*mesh, maps->ips_for_jac);
   CoefficientVector coeff(qs, CoefficientStorage::SYMMETRIC);
   //mfem::out<<"IGAopt.hpp row 376"<<std::endl;
   if (Q) { coeff.Project(*Q);}
   else if (MQ) { coeff.ProjectTranspose(*MQ);}
   else if (DQ) { coeff.Project(*DQ);}
   else { coeff.SetConstant(1.0);}

   const int coeff_dim = coeff.GetVDim();
   symmetric = (coeff_dim != dim*dim);
   const int sym_dims = (dims * (dims + 1)) / 2; // 1x1: 1, 2x2: 3, 3x3: 6
   const int ndata =  symmetric ? sym_dims : dim*dim;
   //mfem::out<<"IGAopt.hpp row 385 sym_dims: "<<coeff_dim<<" "<<sym_dims<<" "<<ndata<<std::endl;
   pa_data.SetSize(ndata * (maps->nq) * (maps->ne), Device::GetMemoryType());

   if (el->GetDerivType() != mfem::FiniteElement::CURL)
   {
      MFEM_ABORT("Unknown kernel.");
   }
   if(dim == 3)
   {
    internal::PACurlCurlSetup3D(maps->Q1D, coeff_dim, maps->ne, maps->ips_for_jac.GetWeights(), maps->jac,
                                  coeff, pa_data);
   }
   else if(dim == 2)
   {
    internal::PACurlCurlSetup2D(maps->Q1D, maps->ne, maps->ips_for_jac.GetWeights(), maps->jac,
                                  coeff, pa_data);  
   }
}

void NURBSCurlCurlIntegrator::AssembleDiagonalPA(Vector& diag)
{
   if(dim == 3)
   {
      return AssembleDiagonalPA3D(maps->Bo,maps->Bc,maps->Go,maps->Gc,pa_data,diag);
   }
   else if(dim == 2)
   {
      return AssembleDiagonalPA2D(diag);
   }
   else{
      MFEM_ABORT("Unsuportted Dim!!!");
   }
}

void NURBSCurlCurlIntegrator::AssembleDiagonalPA2D(Vector& diag)
{
   constexpr static int VDIM = 2;
   const int D1D = maps->cD1D;
   const int Q1D = maps->Q1D;
   const int NE = maps->ne;
   auto Bo = Reshape(maps->Bo.Read(), Q1D, D1D-1, VDIM, NE);
   auto Gc = Reshape(maps->Gc.Read(), Q1D, D1D, VDIM, NE);
   auto op = Reshape(pa_data.Read(), Q1D, Q1D, NE);
   auto D = Reshape(diag.ReadWrite(), 2*(D1D-1)*D1D, NE);

   mfem::forall(NE, [=] MFEM_HOST_DEVICE (int e)
   {
      constexpr static int MAX_Q1D = DofQuadLimits::HCURL_MAX_Q1D;

      int osc = 0;

      for (int c = 0; c < VDIM; ++c)  // loop over x, y components
      {
         const int D1Dy = (c == 1) ? D1D - 1 : D1D;
         const int D1Dx = (c == 0) ? D1D - 1 : D1D;

         real_t t[MAX_Q1D];

         for (int dy = 0; dy < D1Dy; ++dy)
         {
            for (int qx = 0; qx < Q1D; ++qx)
            {
               t[qx] = 0.0;
               for (int qy = 0; qy < Q1D; ++qy)
               {
                  const real_t wy = (c == 1) ? Bo(qy,dy,c,e) : -Gc(qy,dy,c,e);
                  t[qx] += wy * wy * op(qx,qy,e);
               }
            }

            for (int dx = 0; dx < D1Dx; ++dx)
            {
               for (int qx = 0; qx < Q1D; ++qx)
               {
                  const real_t wx = ((c == 0) ? Bo(qx,dx,c,e) : Gc(qx,dx,c,e));
                  D(dx + (dy * D1Dx) + osc, e) += t[qx] * wx * wx;
               }
            }
         }

         osc += D1Dx * D1Dy;
      }  // loop c
   }); // end of element loop
}

void NURBSCurlCurlIntegrator::AssembleDiagonalPA3D(const Array<real_t> &bo,
                                                   const Array<real_t> &bc,
                                                   const Array<real_t> &go,
                                                   const Array<real_t> &gc,
                                                   const Vector &pd,
                                                   Vector& diag)
{
   const int D1D = maps->cD1D;
   const int Q1D = maps->Q1D;
   const int NE = maps->ne;
   auto Bo = Reshape(bo.Read(), Q1D, D1D-1, 3, NE);
   auto Bc = Reshape(bc.Read(), Q1D, D1D, 3, NE);
   auto Go = Reshape(go.Read(), Q1D, D1D-1, 3, NE);
   auto Gc = Reshape(gc.Read(), Q1D, D1D, 3, NE);
   auto op = Reshape(pd.Read(), Q1D, Q1D, Q1D, (symmetric ? 6 : 9), NE);
   auto D = Reshape(diag.ReadWrite(), 3*(D1D-1)*D1D*D1D, NE);
   const int s = symmetric ? 6 : 9;
   const int i11 = 0;
   const int i12 = 1;
   const int i13 = 2;
   const int i21 = symmetric ? i12 : 3;
   const int i22 = symmetric ? 3 : 4;
   const int i23 = symmetric ? 4 : 5;
   const int i31 = symmetric ? i13 : 6;
   const int i32 = symmetric ? i23 : 7;
   const int i33 = symmetric ? 5 : 8;
   constexpr int VDIM = 3;
   constexpr int MD1D = 3;
   constexpr int MQ1D = 3;
   Vector Yt(MQ1D*MD1D*MD1D*9*3*3*NE);
   Vector Zt(MQ1D*MQ1D*MD1D*9*3*NE);
   Yt = 0.0;
   Zt = 0.0;
   auto yt = Reshape(Yt.ReadWrite(), MQ1D, MD1D, MD1D,9,3,3,NE);
   auto zt = Reshape(Zt.ReadWrite(), MQ1D, MQ1D, MD1D,9,3,NE);
   mfem::forall(NE, [=] MFEM_HOST_DEVICE (int e)
   {
      // Using (\nabla\times u) F = 1/det(dF) dF \hat{\nabla}\times\hat{u} (p. 78 of Monk), we get
      // (\nabla\times u) \cdot (\nabla\times u) = 1/det(dF)^2 \hat{\nabla}\times\hat{u}^T dF^T dF \hat{\nabla}\times\hat{u}
      // If c = 0, \hat{\nabla}\times\hat{u} reduces to [0, (u_0)_{x_2}, -(u_0)_{x_1}]
      // If c = 1, \hat{\nabla}\times\hat{u} reduces to [-(u_1)_{x_2}, 0, (u_1)_{x_0}]
      // If c = 2, \hat{\nabla}\times\hat{u} reduces to [(u_2)_{x_1}, -(u_2)_{x_0}, 0]

      // For each c, we will keep 9 arrays for derivatives multiplied by the 9 entries of the 3x3 matrix (dF^T C dF),
      // which may be non-symmetric depending on a possibly non-symmetric matrix coefficient.
      int osc = 0;
      for (int c = 0; c < VDIM; ++c)  // loop over x, y, z components
      {
         const int D1Dz = (c == 2) ? D1D - 1 : D1D;
         const int D1Dy = (c == 1) ? D1D - 1 : D1D;
         const int D1Dx = (c == 0) ? D1D - 1 : D1D;
         // z contraction
         for (int qx = 0; qx < Q1D; ++qx)
         {
            for (int qy = 0; qy < Q1D; ++qy)
            {
               for (int dz = 0; dz < D1Dz; ++dz)
               {
                  for (int i=0; i<s; ++i)
                  {
                     for (int d=0; d<3; ++d)
                     {
                        zt(qx,qy,dz,i,d,e) = 0.0;
                     }
                  }

                  for (int qz = 0; qz < Q1D; ++qz)
                  {
                     const real_t wz = ((c == 2) ? Bo(qz,dz,c,e) : Bc(qz,dz,c,e));
                     const real_t wDz = ((c == 2) ? Go(qz,dz,c,e) : Gc(qz,dz,c,e));

                     for (int i=0; i<s; ++i)
                     {
                        zt(qx,qy,dz,i,0,e) += wz * wz * op(qx,qy,qz,i,e);
                        zt(qx,qy,dz,i,1,e) += wDz * wz * op(qx,qy,qz,i,e);
                        zt(qx,qy,dz,i,2,e) += wDz * wDz * op(qx,qy,qz,i,e);
                     }
                  }
               }
            }
         }  // end of z contraction


         // y contraction
         for (int qx = 0; qx < Q1D; ++qx)
         {
            for (int dz = 0; dz < D1Dz; ++dz)
            {
               for (int dy = 0; dy < D1Dy; ++dy)
               {
                  for (int i=0; i<s; ++i)
                  {
                     for (int d=0; d<3; ++d)
                        for (int j=0; j<3; ++j)
                        {
                           yt(qx,dy,dz,i,d,j,e) = 0.0;
                        }
                  }

                  for (int qy = 0; qy < Q1D; ++qy)
                  {
                     const real_t wy = ((c == 1) ? Bo(qy,dy,c,e) : Bc(qy,dy,c,e));
                     const real_t wDy = ((c == 1) ? Go(qy,dy,c,e) : Gc(qy,dy,c,e));

                     for (int i=0; i<s; ++i)
                     {
                        for (int d=0; d<3; ++d)
                        {
                           yt(qx,dy,dz,i,d,0,e) += wy * wy * zt(qx,qy,dz,i,d,e);
                           yt(qx,dy,dz,i,d,1,e) += wDy * wy * zt(qx,qy,dz,i,d,e);
                           yt(qx,dy,dz,i,d,2,e) += wDy * wDy * zt(qx,qy,dz,i,d,e);
                        }
                     }
                  }
               }
            }
         }  // end of y contraction

         // x contraction
         for (int dz = 0; dz < D1Dz; ++dz)
         {
            for (int dy = 0; dy < D1Dy; ++dy)
            {
               for (int dx = 0; dx < D1Dx; ++dx)
               {
                  for (int qx = 0; qx < Q1D; ++qx)
                  {
                     const real_t wx = ((c == 0) ? Bo(qx,dx,c,e) : Bc(qx,dx,c,e));
                     const real_t wDx = ((c == 0) ? Go(qx,dx,c,e) : Gc(qx,dx,c,e));

                     if (c == 0)
                     {
                        // (u_0)_{x_2} (O22 (u_0)_{x_2} - O23 (u_0)_{x_1}) - (u_0)_{x_1} (O32 (u_0)_{x_2} - O33 (u_0)_{x_1})
                        const real_t sumy = yt(qx,dy,dz,i22,2,0,e) - yt(qx,dy,dz,i23,1,1,e)
                                            - yt(qx,dy,dz,i32,1,1,e) + yt(qx,dy,dz,i33,0,2,e);

                        D(dx + ((dy + (dz * D1Dy)) * D1Dx) + osc, e) += sumy * wx * wx;
                        //printf(" %f",sumy * wx * wx);
                     }
                     else if (c == 1)
                     {
                        // (u_1)_{x_2} (O11 (u_1)_{x_2} - O13 (u_1)_{x_0}) + (u_1)_{x_0} (-O31 (u_1)_{x_2} + O33 (u_1)_{x_0})
                        const real_t d = (yt(qx,dy,dz,i11,2,0,e) * wx * wx)
                                         - ((yt(qx,dy,dz,i13,1,0,e) + yt(qx,dy,dz,i31,1,0,e)) * wDx * wx)
                                         + (yt(qx,dy,dz,i33,0,0,e) * wDx * wDx);

                        D(dx + ((dy + (dz * D1Dy)) * D1Dx) + osc, e) += d;
                        //printf(" %f",d);
                     }
                     else
                     {
                        // (u_2)_{x_1} (O11 (u_2)_{x_1} - O12 (u_2)_{x_0}) - (u_2)_{x_0} (O21 (u_2)_{x_1} - O22 (u_2)_{x_0})
                        const real_t d = (yt(qx,dy,dz,i11,0,2,e) * wx * wx)
                                         - ((yt(qx,dy,dz,i12,0,1,e) + yt(qx,dy,dz,i21,0,1,e)) * wDx * wx)
                                         + (yt(qx,dy,dz,i22,0,0,e) * wDx * wDx);

                        D(dx + ((dy + (dz * D1Dy)) * D1Dx) + osc, e) += d;
                        //printf(" %f",d);
                     }
                  }
               }
            }
         }  // end of x contraction

      osc += D1Dx * D1Dy * D1Dz;
      }  // loop c
   }); // end of element loop
}

void NURBSCurlCurlIntegrator::AddMultPA(const Vector &x, Vector &y) const
{
   if(dim == 3)
   {
    AddMultPA3D(maps->D1D, maps->cD1D, maps->Q1D, symmetric, maps->ne, maps->Bo, maps->Bc, maps->Bot, maps->Bct, maps->Gc, maps->Gct, pa_data, x, y);
   }
   else if(dim == 2)
   {
    AddMultPA2D(maps->cD1D, maps->Q1D, maps->ne, maps->Bo, maps->Bot, maps->Gc, maps->Gct, pa_data, x, y);
   }
   else{
      //mfem::out<<"IGAopt.hpp: test row 643 dim: "<<dim<<std::endl;
      MFEM_ABORT("No support Dim!!")
   }
}
void NURBSCurlCurlIntegrator::AddMultTransposePA(const Vector &x, Vector &y) const
{
   MFEM_ASSERT(symmetric == true, "NURBSCurlCurlIntegrator AddMultTransposePA must need symmetric ceofficent now");
   AddMultPA(x,y);
}
void NURBSCurlCurlIntegrator::AddMultPA2D(const int D1D,
                       const int Q1D,
                       const int NE,
                       const Array<real_t> &bo,
                       const Array<real_t> &bot,
                       const Array<real_t> &gc,
                       const Array<real_t> &gct,
                       const Vector &pa_data,
                       const Vector &x,
                       Vector &y) const
{
   constexpr static int VDIM = 2;
   auto Bo = Reshape(bo.Read(), Q1D, D1D-1, VDIM, NE);
   auto Bot = Reshape(bot.Read(), D1D-1, Q1D, VDIM, NE);
   auto Gc = Reshape(gc.Read(), Q1D, D1D, VDIM, NE);
   auto Gct = Reshape(gct.Read(), D1D, Q1D, VDIM, NE);
   auto op = Reshape(pa_data.Read(), Q1D, Q1D, NE);
   auto X = Reshape(x.Read(), 2*(D1D-1)*D1D, NE);
   auto Y = Reshape(y.ReadWrite(), 2*(D1D-1)*D1D, NE);
   //mfem::out<<"bilininteg_hcurl_kernels.cpp row 690"<<std::endl;
   mfem::forall(NE, [=] MFEM_HOST_DEVICE (int e)
   {
      constexpr static int MAX_D1D = DofQuadLimits::HCURL_MAX_D1D;
      constexpr static int MAX_Q1D = DofQuadLimits::HCURL_MAX_Q1D;

      real_t curl[MAX_Q1D][MAX_Q1D];

      // curl[qy][qx] will be computed as du_y/dx - du_x/dy

      for (int qy = 0; qy < Q1D; ++qy)
      {
         for (int qx = 0; qx < Q1D; ++qx)
         {
            curl[qy][qx] = 0.0;
         }
      }

      int osc = 0;

      for (int c = 0; c < VDIM; ++c)  // loop over x, y components
      {
         const int D1Dy = (c == 1) ? D1D - 1 : D1D;
         const int D1Dx = (c == 0) ? D1D - 1 : D1D;

         for (int dy = 0; dy < D1Dy; ++dy)
         {
            real_t gradX[MAX_Q1D];
            for (int qx = 0; qx < Q1D; ++qx)
            {
               gradX[qx] = 0;
            }

            for (int dx = 0; dx < D1Dx; ++dx)
            {
               const real_t t = X(dx + (dy * D1Dx) + osc, e);
               for (int qx = 0; qx < Q1D; ++qx)
               {
                  gradX[qx] += t * ((c == 0) ? Bo(qx,dx,c,e) : Gc(qx,dx,c,e));
               }
            }

            for (int qy = 0; qy < Q1D; ++qy)
            {
               const real_t wy = (c == 0) ? -Gc(qy,dy,c,e) : Bo(qy,dy,c,e);
               for (int qx = 0; qx < Q1D; ++qx)
               {
                  curl[qy][qx] += gradX[qx] * wy;
               }
            }
         }

         osc += D1Dx * D1Dy;
      }  // loop (c) over components

      // Apply D operator.
      for (int qy = 0; qy < Q1D; ++qy)
      {
         for (int qx = 0; qx < Q1D; ++qx)
         {
            curl[qy][qx] *= op(qx,qy,e);
         }
      }

      for (int qy = 0; qy < Q1D; ++qy)
      {
         osc = 0;

         for (int c = 0; c < VDIM; ++c)  // loop over x, y components
         {
            const int D1Dy = (c == 1) ? D1D - 1 : D1D;
            const int D1Dx = (c == 0) ? D1D - 1 : D1D;

            real_t gradX[MAX_D1D];
            for (int dx = 0; dx < D1Dx; ++dx)
            {
               gradX[dx] = 0.0;
            }
            for (int qx = 0; qx < Q1D; ++qx)
            {
               for (int dx = 0; dx < D1Dx; ++dx)
               {
                  gradX[dx] += curl[qy][qx] * ((c == 0) ? Bot(dx,qx,c,e) : Gct(dx,qx,c,e));
               }
            }
            for (int dy = 0; dy < D1Dy; ++dy)
            {
               const real_t wy = (c == 0) ? -Gct(dy,qy,c,e) : Bot(dy,qy,c,e);

               for (int dx = 0; dx < D1Dx; ++dx)
               {
                  Y(dx + (dy * D1Dx) + osc, e) += gradX[dx] * wy;
               }
            }

            osc += D1Dx * D1Dy;
         }  // loop c
      }  // loop qy
   }); // end of element loop
}

void NURBSCurlCurlIntegrator::AddMultPA3D(const int d1d,
                              const int cd1d,
                              const int q1d,
                              const bool symmetric,
                              const int NE,
                              const Array<real_t> &bo,
                              const Array<real_t> &bc,
                              const Array<real_t> &bot,
                              const Array<real_t> &bct,
                              const Array<real_t> &gc,
                              const Array<real_t> &gct,
                              const Vector &pa_data,
                              const Vector &x, Vector &y) const
{  //mfem::out<<"IGAopt.hpp row 442 *********************"<<std::endl;
   constexpr int VDIM = 3;
   int Q1D = q1d;
   int D1D = d1d;
   int cD1D = cd1d;
   auto Bo = Reshape(bo.Read(), Q1D, D1D, VDIM, NE);//num_el*dim*D1D*Q1D
   auto Bc = Reshape(bc.Read(), Q1D, cD1D, VDIM, NE);
   auto Gc = Reshape(gc.Read(), Q1D, cD1D, VDIM, NE);

   auto Bot = Reshape(bot.Read(), D1D, Q1D, VDIM, NE);//num_el*dim*D1D*Q1D
   auto Bct = Reshape(bct.Read(), cD1D, Q1D, VDIM, NE);
   auto Gct = Reshape(gct.Read(), cD1D, Q1D, VDIM, NE);

   auto op = Reshape(pa_data.Read(), Q1D, Q1D, Q1D, (symmetric ? 6 : 9), NE);
   auto X = Reshape(x.Read(), 3*D1D*cD1D*cD1D, NE);
   auto Y = Reshape(y.ReadWrite(), 3*D1D*cD1D*cD1D, NE);
   bool s = symmetric;

   mfem::forall(NE, [=] MFEM_HOST_DEVICE (int e)
   {  
      // Using (\nabla\times u) F = 1/det(dF) dF \hat{\nabla}\times\hat{u} (p. 78 of Monk),
      // we get:
      // (\nabla\times u) \cdot (\nabla\times v)
      //     = 1/det(dF)^2 \hat{\nabla}\times\hat{u}^T dF^T dF \hat{\nabla}\times\hat{v}
      // If c = 0, \hat{\nabla}\times\hat{u} reduces to [0, (u_0)_{x_2}, -(u_0)_{x_1}]
      // If c = 1, \hat{\nabla}\times\hat{u} reduces to [-(u_1)_{x_2}, 0, (u_1)_{x_0}]
      // If c = 2, \hat{\nabla}\times\hat{u} reduces to [(u_2)_{x_1}, -(u_2)_{x_0}, 0]
      constexpr int MD1D = DofQuadLimits::HCURL_MAX_D1D;
      constexpr int MQ1D = DofQuadLimits::HCURL_MAX_Q1D;

      real_t curl[MQ1D][MQ1D][MQ1D][VDIM];

      for (int qz = 0; qz < Q1D; ++qz)
      {
         for (int qy = 0; qy < Q1D; ++qy)
         {
            for (int qx = 0; qx < Q1D; ++qx)
            {
               for (int c = 0; c < VDIM; ++c)
               {
                  curl[qz][qy][qx][c] = 0.0;
               }
            }
         }
      }
      // We treat x, y, z components separately for optimization specific to each.

      int osc = 0;

      {
         // x component
         const int D1Dz = cD1D;
         const int D1Dy = cD1D;
         const int D1Dx = D1D;

         for (int dz = 0; dz < D1Dz; ++dz)
         {
            real_t gradXY[MQ1D][MQ1D][2];
            for (int qy = 0; qy < Q1D; ++qy)
            {
               for (int qx = 0; qx < Q1D; ++qx)
               {
                  for (int d = 0; d < 2; ++d)
                  {
                     gradXY[qy][qx][d] = 0.0;
                  }
               }
            }

            for (int dy = 0; dy < D1Dy; ++dy)
            {
               real_t massX[MQ1D];
               for (int qx = 0; qx < Q1D; ++qx)
               {
                  massX[qx] = 0.0;
               }

               for (int dx = 0; dx < D1Dx; ++dx)
               {
                  const real_t t = X(dx + ((dy + (dz * D1Dy)) * D1Dx) + osc, e);
                  for (int qx = 0; qx < Q1D; ++qx)
                  {
                     massX[qx] += t * Bo(qx,dx,0,e);
                  }
               }

               for (int qy = 0; qy < Q1D; ++qy)
               {
                  const real_t wy = Bc(qy,dy,1,e);
                  const real_t wDy = Gc(qy,dy,1,e);
                  for (int qx = 0; qx < Q1D; ++qx)
                  {
                     const real_t wx = massX[qx];
                     gradXY[qy][qx][0] += wx * wDy;
                     gradXY[qy][qx][1] += wx * wy;
                  }
               }
            }

            for (int qz = 0; qz < Q1D; ++qz)
            {
               const real_t wz = Bc(qz,dz,2,e);
               const real_t wDz = Gc(qz,dz,2,e);
               for (int qy = 0; qy < Q1D; ++qy)
               {
                  for (int qx = 0; qx < Q1D; ++qx)
                  {
                     // \hat{\nabla}\times\hat{u} is [0, (u_0)_{x_2}, -(u_0)_{x_1}]
                     curl[qz][qy][qx][1] += gradXY[qy][qx][1] * wDz; // (u_0)_{x_2}
                     curl[qz][qy][qx][2] -= gradXY[qy][qx][0] * wz;  // -(u_0)_{x_1}
                  }
               }
            }
         }

         osc += D1Dx * D1Dy * D1Dz;
      }
      
      {
         // y component
         const int D1Dz = cD1D;
         const int D1Dy = D1D;
         const int D1Dx = cD1D;

         for (int dz = 0; dz < D1Dz; ++dz)
         {
            real_t gradXY[MQ1D][MQ1D][2];
            for (int qy = 0; qy < Q1D; ++qy)
            {
               for (int qx = 0; qx < Q1D; ++qx)
               {
                  for (int d = 0; d < 2; ++d)
                  {
                     gradXY[qy][qx][d] = 0.0;
                  }
               }
            }

            for (int dx = 0; dx < D1Dx; ++dx)
            {
               real_t massY[MQ1D];
               for (int qy = 0; qy < Q1D; ++qy)
               {
                  massY[qy] = 0.0;
               }

               for (int dy = 0; dy < D1Dy; ++dy)
               {
                  const real_t t = X(dx + ((dy + (dz * D1Dy)) * D1Dx) + osc, e);
                  for (int qy = 0; qy < Q1D; ++qy)
                  {
                     massY[qy] += t * Bo(qy,dy,1,e);
                  }
               }

               for (int qx = 0; qx < Q1D; ++qx)
               {
                  const real_t wx = Bc(qx,dx,0,e);
                  const real_t wDx = Gc(qx,dx,0,e);
                  for (int qy = 0; qy < Q1D; ++qy)
                  {
                     const real_t wy = massY[qy];
                     gradXY[qy][qx][0] += wDx * wy;
                     gradXY[qy][qx][1] += wx * wy;
                  }
               }
            }

            for (int qz = 0; qz < Q1D; ++qz)
            {
               const real_t wz = Bc(qz,dz,2,e);
               const real_t wDz = Gc(qz,dz,2,e);
               for (int qy = 0; qy < Q1D; ++qy)
               {
                  for (int qx = 0; qx < Q1D; ++qx)
                  {
                     // \hat{\nabla}\times\hat{u} is [-(u_1)_{x_2}, 0, (u_1)_{x_0}]
                     curl[qz][qy][qx][0] -= gradXY[qy][qx][1] * wDz; // -(u_1)_{x_2}
                     curl[qz][qy][qx][2] += gradXY[qy][qx][0] * wz;  // (u_1)_{x_0}
                  }
               }
            }
         }

         osc += D1Dx * D1Dy * D1Dz;
      }

      {
         // z component
         const int D1Dz = D1D;
         const int D1Dy = cD1D;
         const int D1Dx = cD1D;

         for (int dx = 0; dx < D1Dx; ++dx)
         {
            real_t gradYZ[MQ1D][MQ1D][2];
            for (int qz = 0; qz < Q1D; ++qz)
            {
               for (int qy = 0; qy < Q1D; ++qy)
               {
                  for (int d = 0; d < 2; ++d)
                  {
                     gradYZ[qz][qy][d] = 0.0;
                  }
               }
            }

            for (int dy = 0; dy < D1Dy; ++dy)
            {
               real_t massZ[MQ1D];
               for (int qz = 0; qz < Q1D; ++qz)
               {
                  massZ[qz] = 0.0;
               }

               for (int dz = 0; dz < D1Dz; ++dz)
               {
                  const real_t t = X(dx + ((dy + (dz * D1Dy)) * D1Dx) + osc, e);
                  for (int qz = 0; qz < Q1D; ++qz)
                  {
                     massZ[qz] += t * Bo(qz,dz,2,e);
                  }
               }

               for (int qy = 0; qy < Q1D; ++qy)
               {
                  const real_t wy = Bc(qy,dy,1,e);
                  const real_t wDy = Gc(qy,dy,1,e);
                  for (int qz = 0; qz < Q1D; ++qz)
                  {
                     const real_t wz = massZ[qz];
                     gradYZ[qz][qy][0] += wz * wy;
                     gradYZ[qz][qy][1] += wz * wDy;
                  }
               }
            }

            for (int qx = 0; qx < Q1D; ++qx)
            {
               const real_t wx = Bc(qx,dx,0,e);
               const real_t wDx = Gc(qx,dx,0,e);

               for (int qy = 0; qy < Q1D; ++qy)
               {
                  for (int qz = 0; qz < Q1D; ++qz)
                  {
                     // \hat{\nabla}\times\hat{u} is [(u_2)_{x_1}, -(u_2)_{x_0}, 0]
                     curl[qz][qy][qx][0] += gradYZ[qz][qy][1] * wx;  // (u_2)_{x_1}
                     curl[qz][qy][qx][1] -= gradYZ[qz][qy][0] * wDx; // -(u_2)_{x_0}
                  }
               }
            }
         }
      }

      // Apply D operator.
      for (int qz = 0; qz < Q1D; ++qz)
      {
         for (int qy = 0; qy < Q1D; ++qy)
         {
            for (int qx = 0; qx < Q1D; ++qx)
            {
               const real_t O11 = op(qx,qy,qz,0,e);
               const real_t O12 = op(qx,qy,qz,1,e);
               const real_t O13 = op(qx,qy,qz,2,e);
               const real_t O21 = s ? O12 : op(qx,qy,qz,3,e);
               const real_t O22 = s ? op(qx,qy,qz,3,e) : op(qx,qy,qz,4,e);
               const real_t O23 = s ? op(qx,qy,qz,4,e) : op(qx,qy,qz,5,e);
               const real_t O31 = s ? O13 : op(qx,qy,qz,6,e);
               const real_t O32 = s ? O23 : op(qx,qy,qz,7,e);
               const real_t O33 = s ? op(qx,qy,qz,5,e) : op(qx,qy,qz,8,e);

               const real_t c1 = (O11 * curl[qz][qy][qx][0]) + (O12 * curl[qz][qy][qx][1]) +
                                 (O13 * curl[qz][qy][qx][2]);
               const real_t c2 = (O21 * curl[qz][qy][qx][0]) + (O22 * curl[qz][qy][qx][1]) +
                                 (O23 * curl[qz][qy][qx][2]);
               const real_t c3 = (O31 * curl[qz][qy][qx][0]) + (O32 * curl[qz][qy][qx][1]) +
                                 (O33 * curl[qz][qy][qx][2]);

               curl[qz][qy][qx][0] = c1;
               curl[qz][qy][qx][1] = c2;
               curl[qz][qy][qx][2] = c3;
            }
         }
      }
      // x component
      osc = 0;
      {
         const int D1Dz = cD1D;
         const int D1Dy = cD1D;
         const int D1Dx = D1D;

         for (int qz = 0; qz < Q1D; ++qz)
         {
            real_t gradXY12[MD1D][MD1D];
            real_t gradXY21[MD1D][MD1D];

            for (int dy = 0; dy < D1Dy; ++dy)
            {
               for (int dx = 0; dx < D1Dx; ++dx)
               {
                  gradXY12[dy][dx] = 0.0;
                  gradXY21[dy][dx] = 0.0;
               }
            }
            for (int qy = 0; qy < Q1D; ++qy)
            {
               real_t massX[MD1D][2];
               for (int dx = 0; dx < D1Dx; ++dx)
               {
                  for (int n = 0; n < 2; ++n)
                  {
                     massX[dx][n] = 0.0;
                  }
               }
               for (int qx = 0; qx < Q1D; ++qx)
               {
                  for (int dx = 0; dx < D1Dx; ++dx)
                  {
                     const real_t wx = Bot(dx,qx,0,e);

                     massX[dx][0] += wx * curl[qz][qy][qx][1];
                     massX[dx][1] += wx * curl[qz][qy][qx][2];
                  }
               }
               for (int dy = 0; dy < D1Dy; ++dy)
               {
                  const real_t wy = Bct(dy,qy,1,e);
                  const real_t wDy = Gct(dy,qy,1,e);

                  for (int dx = 0; dx < D1Dx; ++dx)
                  {
                     gradXY21[dy][dx] += massX[dx][0] * wy;
                     gradXY12[dy][dx] += massX[dx][1] * wDy;
                  }
               }
            }

            for (int dz = 0; dz < D1Dz; ++dz)
            {
               const real_t wz = Bct(dz,qz,2,e);
               const real_t wDz = Gct(dz,qz,2,e);
               for (int dy = 0; dy < D1Dy; ++dy)
               {
                  for (int dx = 0; dx < D1Dx; ++dx)
                  {
                     // \hat{\nabla}\times\hat{u} is [0, (u_0)_{x_2}, -(u_0)_{x_1}]
                     // (u_0)_{x_2} * (op * curl)_1 - (u_0)_{x_1} * (op * curl)_2
                      Y(dx + ((dy + (dz * D1Dy)) * D1Dx) + osc,
                        e) += (gradXY21[dy][dx] * wDz) - (gradXY12[dy][dx] * wz);
                     //printf("%f ",(Y(dx + ((dy + (dz * D1Dy)) * D1Dx) + osc, e)));
                  }
               }
            }
         }  // loop qz

         osc += D1Dx * D1Dy * D1Dz;
      }

      // y component
      {
         const int D1Dz = cD1D;
         const int D1Dy = D1D;
         const int D1Dx = cD1D;

         for (int qz = 0; qz < Q1D; ++qz)
         {
            real_t gradXY02[MD1D][MD1D];
            real_t gradXY20[MD1D][MD1D];

            for (int dy = 0; dy < D1Dy; ++dy)
            {
               for (int dx = 0; dx < D1Dx; ++dx)
               {
                  gradXY02[dy][dx] = 0.0;
                  gradXY20[dy][dx] = 0.0;
               }
            }
            for (int qx = 0; qx < Q1D; ++qx)
            {
               real_t massY[MD1D][2];
               for (int dy = 0; dy < D1Dy; ++dy)
               {
                  massY[dy][0] = 0.0;
                  massY[dy][1] = 0.0;
               }
               for (int qy = 0; qy < Q1D; ++qy)
               {
                  for (int dy = 0; dy < D1Dy; ++dy)
                  {
                     const real_t wy = Bot(dy,qy,1,e);

                     massY[dy][0] += wy * curl[qz][qy][qx][2];
                     massY[dy][1] += wy * curl[qz][qy][qx][0];
                  }
               }
               for (int dx = 0; dx < D1Dx; ++dx)
               {
                  const real_t wx = Bct(dx,qx,0,e);
                  const real_t wDx = Gct(dx,qx,0,e);

                  for (int dy = 0; dy < D1Dy; ++dy)
                  {
                     gradXY02[dy][dx] += massY[dy][0] * wDx;
                     gradXY20[dy][dx] += massY[dy][1] * wx;
                  }
               }
            }

            for (int dz = 0; dz < D1Dz; ++dz)
            {
               const real_t wz = Bct(dz,qz,2,e);
               const real_t wDz = Gct(dz,qz,2,e);
               for (int dy = 0; dy < D1Dy; ++dy)
               {
                  for (int dx = 0; dx < D1Dx; ++dx)
                  {
                     // \hat{\nabla}\times\hat{u} is [-(u_1)_{x_2}, 0, (u_1)_{x_0}]
                     // -(u_1)_{x_2} * (op * curl)_0 + (u_1)_{x_0} * (op * curl)_2
                     Y(dx + ((dy + (dz * D1Dy)) * D1Dx) + osc,
                        e) += (-gradXY20[dy][dx] * wDz) + (gradXY02[dy][dx] * wz);
                     //printf("%f ",(Y(dx + ((dy + (dz * D1Dy)) * D1Dx) + osc, e)));
                  }
               }
            }
         }  // loop qz

         osc += D1Dx * D1Dy * D1Dz;
      }

      // z component
      {
         const int D1Dz = D1D;
         const int D1Dy = cD1D;
         const int D1Dx = cD1D;

         for (int qx = 0; qx < Q1D; ++qx)
         {
            real_t gradYZ01[MD1D][MD1D];
            real_t gradYZ10[MD1D][MD1D];

            for (int dy = 0; dy < D1Dy; ++dy)
            {
               for (int dz = 0; dz < D1Dz; ++dz)
               {
                  gradYZ01[dz][dy] = 0.0;
                  gradYZ10[dz][dy] = 0.0;
               }
            }
            for (int qy = 0; qy < Q1D; ++qy)
            {
               real_t massZ[MD1D][2];
               for (int dz = 0; dz < D1Dz; ++dz)
               {
                  for (int n = 0; n < 2; ++n)
                  {
                     massZ[dz][n] = 0.0;
                  }
               }
               for (int qz = 0; qz < Q1D; ++qz)
               {
                  for (int dz = 0; dz < D1Dz; ++dz)
                  {
                     const real_t wz = Bot(dz,qz,2,e);

                     massZ[dz][0] += wz * curl[qz][qy][qx][0];
                     massZ[dz][1] += wz * curl[qz][qy][qx][1];
                  }
               }
               for (int dy = 0; dy < D1Dy; ++dy)
               {
                  const real_t wy = Bct(dy,qy,1,e);
                  const real_t wDy = Gct(dy,qy,1,e);

                  for (int dz = 0; dz < D1Dz; ++dz)
                  {
                     gradYZ01[dz][dy] += wy * massZ[dz][1];
                     gradYZ10[dz][dy] += wDy * massZ[dz][0];
                  }
               }
            }

            for (int dx = 0; dx < D1Dx; ++dx)
            {
               const real_t wx = Bct(dx,qx,0,e);
               const real_t wDx = Gct(dx,qx,0,e);

               for (int dy = 0; dy < D1Dy; ++dy)
               {
                  for (int dz = 0; dz < D1Dz; ++dz)
                  {
                     // \hat{\nabla}\times\hat{u} is [(u_2)_{x_1}, -(u_2)_{x_0}, 0]
                     // (u_2)_{x_1} * (op * curl)_0 - (u_2)_{x_0} * (op * curl)_1
                     Y(dx + ((dy + (dz * D1Dy)) * D1Dx) + osc,
                        e) += (gradYZ10[dz][dy] * wx) - (gradYZ01[dz][dy] * wDx);
                     //printf("%f ",(Y(dx + ((dy + (dz * D1Dy)) * D1Dx) + osc, e)));
                  }
               }
            }
         }  // loop qx
      }
   }); // end of element loop
   //mfem::out<<std::endl;
   // mfem::out<<"IGAopt.cpp row 942: "<<std::endl;
   // for(int i = 0; i < y.Size(); i++)
   // {
   //    mfem::out<<y[i]<<" ";
   // }
   // mfem::out<<std::endl;
}

class NURBSHCurl_VectorMassIntegrator: public VectorMassIntegrator
{

private:
   // PA extension
   int dim;
   Vector pa_data;
   bool symmetric = true; ///< False if using a nonsymmetric matrix coefficient
   const NURBSDofToQuad *maps;

public:

   NURBSHCurl_VectorMassIntegrator() { Q = NULL; VQ = NULL; MQ = NULL;integrationMode = Mode::ELEMENTWISE;}
   /// Construct a bilinear form integrator for Nedelec elements
   NURBSHCurl_VectorMassIntegrator(Coefficient &q, const IntegrationRule *ir = NULL) :
      VectorMassIntegrator(q,ir){ integrationMode = Mode::ELEMENTWISE;}
   NURBSHCurl_VectorMassIntegrator(VectorCoefficient &vq,
                      const IntegrationRule *ir = NULL) :
      VectorMassIntegrator(vq){ integrationMode = Mode::ELEMENTWISE;}
   NURBSHCurl_VectorMassIntegrator(MatrixCoefficient &mq, const IntegrationRule *ir = NULL) :
      VectorMassIntegrator(mq){ integrationMode = Mode::ELEMENTWISE;}

   using BilinearFormIntegrator::AssemblePA;
   virtual void AssemblePA(const FiniteElementSpace &fes);
   virtual void AddMultPA(const Vector &x, Vector &y) const;
   virtual void AddMultTransposePA(const Vector &x, Vector &y) const;
   virtual void AddMultPA3D(  const int d1d,
                              const int q1d,
                              const bool symmetric,
                              const int NE,
                              const Array<real_t> &bo,
                              const Array<real_t> &bc,
                              const Array<real_t> &bot,
                              const Array<real_t> &bct,
                              const Vector &pa_data,
                              const Vector &x, Vector &y) const;
   virtual void AddMultPA2D(const int D1D,
                              const int Q1D,
                              const int NE,
                              const bool symmetric,
                              const Array<real_t> &bo,
                              const Array<real_t> &bc,
                              const Array<real_t> &bot,
                              const Array<real_t> &bct,
                              const Vector &pa_data,
                              const Vector &x,
                              Vector &y) const;
   virtual void AssembleDiagonalPA(Vector& diag);
   virtual void AssembleDiagonalPA2D(Vector& diag);
   virtual void AssembleDiagonalPA3D(Vector& diag);
   Coefficient *GetCoefficient() const { return Q; }
   ~NURBSHCurl_VectorMassIntegrator() {if(maps){delete maps;}};
};

void NURBSHCurl_VectorMassIntegrator::AssemblePA(const FiniteElementSpace &fes)
{  
   // Assumes tensor-product elements
   Mesh *mesh = fes.GetMesh();
   const FiniteElement *el = fes.GetFE(0);
   const int dims = el->GetDim();
   dim = mesh->Dimension();
   int nurbs_order = mesh->NURBSext->GetOrder();
   int ir_order = 2*nurbs_order;
   // Assume all the same order
   const IntegrationRule *ir1d = &IntRules.Get(Geometry::SEGMENT, ir_order);
   maps = &GetNURBSDofToQuad(mesh, &fes, ir1d);
   QuadratureSpace qs(*mesh, maps->ips_for_jac);
   CoefficientVector coeff(qs, CoefficientStorage::SYMMETRIC);
   if (Q) { coeff.Project(*Q); }
   else if (MQ) { coeff.ProjectTranspose(*MQ); }
   else if (VQ) { coeff.Project(*VQ); }
   else { coeff.SetConstant(1.0); }
   const int coeff_dim = coeff.GetVDim();
   symmetric = (coeff_dim != dim*dim);
   const int sym_dims = (dims * (dims + 1)) / 2; // 1x1: 1, 2x2: 3, 3x3: 6
   pa_data.SetSize((symmetric ? sym_dims : dims*dims) * maps->nq * maps->ne, Device::GetMemoryType());
   if (el->GetDerivType() != mfem::FiniteElement::CURL)
   {
      MFEM_ABORT("Unknown kernel.");
   }
   if(dim == 3)
   {
    internal::PADiffusionSetup3D(maps->Q1D, coeff_dim, maps->ne, maps->ips_for_jac.GetWeights(), maps->jac,
                                   coeff, pa_data);
   }
   else if(dim == 2)
   {
    internal::PADiffusionSetup2D<2>(maps->Q1D, coeff_dim, maps->ne, maps->ips_for_jac.GetWeights(),
                                      maps->jac, coeff, pa_data);
   }
}

void NURBSHCurl_VectorMassIntegrator::AddMultPA(const Vector &x, Vector &y) const
{
   if(dim ==3 )
   {
      return AddMultPA3D(maps->D1D, maps->Q1D, symmetric, maps->ne, maps->Bo, maps->Bc, maps->Bot, maps->Bct, pa_data,x, y);
   }
   else if(dim == 2)
   {
      return AddMultPA2D(maps->cD1D, maps->Q1D, maps->ne, symmetric, maps->Bo, maps->Bc, maps->Bot, maps->Bct, pa_data, x, y);
   }
}

void NURBSHCurl_VectorMassIntegrator::AddMultTransposePA(const Vector &x, Vector &y) const
{
   MFEM_ASSERT(symmetric == true, "NURBSHCurl_VectorMassIntegrator AddMultTransposePA must need symmetric ceofficent now");
   AddMultPA(x,y);
}

void NURBSHCurl_VectorMassIntegrator::AssembleDiagonalPA(Vector& diag)
{
   if(dim == 3)
   {
      return AssembleDiagonalPA3D(diag);
   }
   else if(dim == 2)
   {
      return AssembleDiagonalPA2D(diag);
   }
   else{
      MFEM_ABORT("No support Dim !!!");
   }
}

void NURBSHCurl_VectorMassIntegrator::AssembleDiagonalPA2D(Vector& diag)
{
   constexpr static int VDIM = 2;
   const int NE = maps->ne;
   const int Q1D = maps->Q1D;
   const int D1D = maps->cD1D;
   bool s = symmetric;
   auto Bo = Reshape(maps->Bo.Read(), Q1D, D1D-1, VDIM, NE);
   auto Bc = Reshape(maps->Bc.Read(), Q1D, D1D, VDIM, NE);
   auto op = Reshape(pa_data.Read(), Q1D, Q1D, symmetric ? 3 : 4, NE);
   auto D = Reshape(diag.ReadWrite(), 2*(D1D-1)*D1D, NE);

   mfem::forall(NE, [=] MFEM_HOST_DEVICE (int e)
   {
      constexpr static int VDIM = 2;
      constexpr static int MAX_Q1D = DofQuadLimits::HCURL_MAX_Q1D;

      int osc = 0;

      for (int c = 0; c < VDIM; ++c)  // loop over x, y components
      {
         const int D1Dy = (c == 1) ? D1D - 1 : D1D;
         const int D1Dx = (c == 0) ? D1D - 1 : D1D;

         real_t mass[MAX_Q1D];

         for (int dy = 0; dy < D1Dy; ++dy)
         {
            for (int qx = 0; qx < Q1D; ++qx)
            {
               mass[qx] = 0.0;
               for (int qy = 0; qy < Q1D; ++qy)
               {
                  const real_t wy = (c == 1) ? Bo(qy,dy,c,e) : Bc(qy,dy,c,e);

                  mass[qx] += wy * wy * ((c == 0) ? op(qx,qy,0,e) :
                                         op(qx,qy,s ? 2 : 3, e));
               }
            }

            for (int dx = 0; dx < D1Dx; ++dx)
            {
               for (int qx = 0; qx < Q1D; ++qx)
               {
                  const real_t wx = ((c == 0) ? Bo(qx,dx,c,e) : Bc(qx,dx,c,e));
                  D(dx + (dy * D1Dx) + osc, e) += mass[qx] * wx * wx;
               }
            }
         }
         osc += D1Dx * D1Dy;
      }  // loop c
   }); // end of element loop
}

void NURBSHCurl_VectorMassIntegrator::AssembleDiagonalPA3D(Vector& diag)
{  
   const int D1D = maps->cD1D;
   const int Q1D = maps->Q1D;
   const int NE = maps->ne;

   const int s = symmetric ? 6 : 9;
   auto Bo = Reshape(maps->Bo.Read(), Q1D, D1D-1, dim, NE);
   auto Bc = Reshape(maps->Bc.Read(), Q1D, D1D, dim, NE);
   auto op = Reshape(pa_data.Read(), Q1D, Q1D, Q1D, symmetric ? 6 : 9, NE);
   auto D = Reshape(diag.ReadWrite(), 3*(D1D-1)*D1D*D1D, NE);
   mfem::forall(NE, [=] MFEM_HOST_DEVICE (int e)
   {
      constexpr static int VDIM = 3;
      constexpr static int MQ1D = DofQuadLimits::HCURL_MAX_Q1D;
      int osc = 0;
      real_t mass[MQ1D];
      for (int ii = 0; ii < MQ1D; ii++)
      {
         mass[ii] = 0;
      }

      for (int c = 0; c < VDIM; ++c)  // loop over x, y, z components
      {
         const int D1Dz = (c == 2) ? D1D - 1 : D1D;
         const int D1Dy = (c == 1) ? D1D - 1 : D1D;
         const int D1Dx = (c == 0) ? D1D - 1 : D1D;

         const int opc = (c == 0) ? 0 : ((c == 1) ? ((s == 6) ? 3 : 4) : ((s == 6) ? 5 : 8));
         for (int dz = 0; dz < D1Dz; ++dz)
         {  
            for (int dy = 0; dy < D1Dy; ++dy)
            {
               for (int qx = 0; qx < Q1D; ++qx)
               {
                  mass[qx] = 0.0;
                  for (int qy = 0; qy < Q1D; ++qy)
                  {  
                     const real_t wy = (c == 1) ? Bo(qy,dy,c,e) : Bc(qy,dy,c,e);
                     for (int qz = 0; qz < Q1D; ++qz)
                     {
                        const real_t wz = (c == 2) ? Bo(qz,dz,c,e) : Bc(qz,dz,c,e);
                        mass[qx] += wy * wy * wz * wz * op(qx,qy,qz,opc,e);
                     }
                  }
               }
               for (int dx = 0; dx < D1Dx; ++dx)
               {
                  for (int qx = 0; qx < Q1D; ++qx)
                  {
                     const real_t wx = ((c == 0) ? Bo(qx,dx,c,e) : Bc(qx,dx,c,e));
                     D(dx + ((dy + (dz * D1Dy)) * D1Dx) + osc, e) += mass[qx] * wx * wx;
                  }
               }
            }
         }
         osc += D1Dx * D1Dy * D1Dz;
      }  // loop c
   }); // end of element loop
}

void NURBSHCurl_VectorMassIntegrator::AddMultPA2D(const int D1D,
                        const int Q1D,
                        const int NE,
                        const bool symmetric,
                        const Array<real_t> &bo,
                        const Array<real_t> &bc,
                        const Array<real_t> &bot,
                        const Array<real_t> &bct,
                        const Vector &pa_data,
                        const Vector &x,
                        Vector &y) const
{
   constexpr static int VDIM = 2;
   auto Bo = Reshape(bo.Read(), Q1D, D1D-1, VDIM, NE);
   auto Bc = Reshape(bc.Read(), Q1D, D1D, VDIM, NE);
   auto Bot = Reshape(bot.Read(), D1D-1, Q1D, VDIM, NE);
   auto Bct = Reshape(bct.Read(), D1D, Q1D, VDIM, NE);
   auto op = Reshape(pa_data.Read(), Q1D, Q1D, symmetric ? 3 : 4, NE);
   auto X = Reshape(x.Read(), 2*(D1D-1)*D1D, NE);
   auto Y = Reshape(y.ReadWrite(), 2*(D1D-1)*D1D, NE);
   bool s = symmetric;
   //mfem::out<<"bilininteg_hcurl_kernels.cpp row 169"<<std::endl;
   mfem::forall(NE, [=] MFEM_HOST_DEVICE (int e)
   {
      constexpr static int MAX_D1D = DofQuadLimits::HCURL_MAX_D1D;
      constexpr static int MAX_Q1D = DofQuadLimits::HCURL_MAX_Q1D;

      real_t mass[MAX_Q1D][MAX_Q1D][VDIM];

      for (int qy = 0; qy < Q1D; ++qy)
      {
         for (int qx = 0; qx < Q1D; ++qx)
         {
            for (int c = 0; c < VDIM; ++c)
            {
               mass[qy][qx][c] = 0.0;
            }
         }
      }

      int osc = 0;

      for (int c = 0; c < VDIM; ++c)  // loop over x, y components
      {
         const int D1Dy = (c == 1) ? D1D - 1 : D1D;
         const int D1Dx = (c == 0) ? D1D - 1 : D1D;

         for (int dy = 0; dy < D1Dy; ++dy)
         {
            real_t massX[MAX_Q1D];
            for (int qx = 0; qx < Q1D; ++qx)
            {
               massX[qx] = 0.0;
            }

            for (int dx = 0; dx < D1Dx; ++dx)
            {
               const real_t t = X(dx + (dy * D1Dx) + osc, e);
               for (int qx = 0; qx < Q1D; ++qx)
               {
                  massX[qx] += t * ((c == 0) ? Bo(qx,dx,c,e) : Bc(qx,dx,c,e));
               }
            }

            for (int qy = 0; qy < Q1D; ++qy)
            {
               const real_t wy = (c == 1) ? Bo(qy,dy,c,e) : Bc(qy,dy,c,e);
               for (int qx = 0; qx < Q1D; ++qx)
               {
                  mass[qy][qx][c] += massX[qx] * wy;
               }
            }
         }

         osc += D1Dx * D1Dy;
      }  // loop (c) over components

      // Apply D operator.
      for (int qy = 0; qy < Q1D; ++qy)
      {
         for (int qx = 0; qx < Q1D; ++qx)
         {
            const real_t O11 = op(qx,qy,0,e);
            const real_t O21 = op(qx,qy,1,e);
            const real_t O12 = s ? O21 : op(qx,qy,2,e);
            const real_t O22 = s ? op(qx,qy,2,e) : op(qx,qy,3,e);
            const real_t massX = mass[qy][qx][0];
            const real_t massY = mass[qy][qx][1];
            mass[qy][qx][0] = (O11*massX)+(O12*massY);
            mass[qy][qx][1] = (O21*massX)+(O22*massY);
         }
      }

      for (int qy = 0; qy < Q1D; ++qy)
      {
         osc = 0;

         for (int c = 0; c < VDIM; ++c)  // loop over x, y components
         {
            const int D1Dy = (c == 1) ? D1D - 1 : D1D;
            const int D1Dx = (c == 0) ? D1D - 1 : D1D;

            real_t massX[MAX_D1D];
            for (int dx = 0; dx < D1Dx; ++dx)
            {
               massX[dx] = 0.0;
            }
            for (int qx = 0; qx < Q1D; ++qx)
            {
               for (int dx = 0; dx < D1Dx; ++dx)
               {
                  massX[dx] += mass[qy][qx][c] * ((c == 0) ? Bot(dx,qx,c,e) : Bct(dx,qx,c,e));
               }
            }

            for (int dy = 0; dy < D1Dy; ++dy)
            {
               const real_t wy = (c == 1) ? Bot(dy,qy,c,e) : Bct(dy,qy,c,e);

               for (int dx = 0; dx < D1Dx; ++dx)
               {
                  Y(dx + (dy * D1Dx) + osc, e) += massX[dx] * wy;
                  //printf("%d ",Y(dx + (dy * D1Dx) + osc, e));
               }
            }

            osc += D1Dx * D1Dy;
         }  // loop c
      }  // loop qy
   }); // end of element loop
   // mfem::out<<std::endl;
   // mfem::out<<"bilininteg_hcurl_kernels.cpp row 279: "<<y.Size()<<std::endl;
   // for(int i = 0; i < y.Size(); i++)
   // {
   //    mfem::out<<y[i]<<" ";
   // }
   // mfem::out<<std::endl;
}

void NURBSHCurl_VectorMassIntegrator::AddMultPA3D( const int d1d,
                                                   const int q1d,
                                                   const bool symmetric,
                                                   const int NE,
                                                   const Array<real_t> &bo,
                                                   const Array<real_t> &bc,
                                                   const Array<real_t> &bot,
                                                   const Array<real_t> &bct,
                                                   const Vector &pa_data,
                                                   const Vector &x, Vector &y) const
{  //mfem::out<<"IGAopt.hpp row 1170"<<std::endl;
   MFEM_VERIFY(dim == 3, "Must be 3D");
   int Q1D = q1d;
   int D1D = d1d;
   int cD1D = D1D+1;
   constexpr static int VDIM = 3;
   auto Bo = Reshape(bo.Read(), Q1D, D1D, VDIM, NE);
   auto Bc = Reshape(bc.Read(), Q1D, cD1D, VDIM, NE);
   auto Bot = Reshape(bot.Read(), D1D, Q1D, VDIM, NE);
   auto Bct = Reshape(bct.Read(), cD1D, Q1D, VDIM, NE);
   auto op = Reshape(pa_data.Read(), Q1D, Q1D, Q1D, (symmetric ? 6 : 9), NE);
   auto X = Reshape(x.Read(), 3*D1D*cD1D*cD1D, NE);
   auto Y = Reshape(y.ReadWrite(), 3*D1D*cD1D*cD1D, NE);
   bool s = symmetric;
   //mfem::out<<"IGAopt.cpp row 1113: "<<std::endl;
   mfem::forall(NE, [=] MFEM_HOST_DEVICE (int e)
   {
      constexpr static int MAX_D1D = DofQuadLimits::HCURL_MAX_D1D;
      constexpr static int MAX_Q1D = DofQuadLimits::HCURL_MAX_Q1D;

      real_t mass[MAX_Q1D][MAX_Q1D][MAX_Q1D][VDIM];

      for (int qz = 0; qz < Q1D; ++qz)
      {
         for (int qy = 0; qy < Q1D; ++qy)
         {
            for (int qx = 0; qx < Q1D; ++qx)
            {
               for (int c = 0; c < VDIM; ++c)
               {
                  mass[qz][qy][qx][c] = 0.0;
               }
            }
         }
      }

      int osc = 0;

      for (int c = 0; c < VDIM; ++c)  // loop over x, y, z components
      {
         const int D1Dz = (c == 2) ? D1D : cD1D;
         const int D1Dy = (c == 1) ? D1D : cD1D;
         const int D1Dx = (c == 0) ? D1D : cD1D;

         for (int dz = 0; dz < D1Dz; ++dz)
         {
            real_t massXY[MAX_Q1D][MAX_Q1D];
            for (int qy = 0; qy < Q1D; ++qy)
            {
               for (int qx = 0; qx < Q1D; ++qx)
               {
                  massXY[qy][qx] = 0.0;
               }
            }

            for (int dy = 0; dy < D1Dy; ++dy)
            {
               real_t massX[MAX_Q1D];
               for (int qx = 0; qx < Q1D; ++qx)
               {
                  massX[qx] = 0.0;
               }

               for (int dx = 0; dx < D1Dx; ++dx)
               {
                  const real_t t = X(dx + ((dy + (dz * D1Dy)) * D1Dx) + osc, e);
                  for (int qx = 0; qx < Q1D; ++qx)
                  {
                     massX[qx] += t * ((c == 0) ? Bo(qx,dx,c,e) : Bc(qx,dx,c,e));
                  }
               }

               for (int qy = 0; qy < Q1D; ++qy)
               {
                  const real_t wy = (c == 1) ? Bo(qy,dy,c,e) : Bc(qy,dy,c,e);
                  for (int qx = 0; qx < Q1D; ++qx)
                  {
                     const real_t wx = massX[qx];
                     massXY[qy][qx] += wx * wy;
                  }
               }
            }

            for (int qz = 0; qz < Q1D; ++qz)
            {
               const real_t wz = (c == 2) ? Bo(qz,dz,c,e) : Bc(qz,dz,c,e);
               for (int qy = 0; qy < Q1D; ++qy)
               {
                  for (int qx = 0; qx < Q1D; ++qx)
                  {
                     mass[qz][qy][qx][c] += massXY[qy][qx] * wz;
                  }
               }
            }
         }

         osc += D1Dx * D1Dy * D1Dz;
      }  // loop (c) over components

      // Apply D operator.
      for (int qz = 0; qz < Q1D; ++qz)
      {
         for (int qy = 0; qy < Q1D; ++qy)
         {
            for (int qx = 0; qx < Q1D; ++qx)
            {
               const real_t O11 = op(qx,qy,qz,0,e);
               const real_t O12 = op(qx,qy,qz,1,e);
               const real_t O13 = op(qx,qy,qz,2,e);
               const real_t O21 = s ? O12 : op(qx,qy,qz,3,e);
               const real_t O22 = s ? op(qx,qy,qz,3,e) : op(qx,qy,qz,4,e);
               const real_t O23 = s ? op(qx,qy,qz,4,e) : op(qx,qy,qz,5,e);
               const real_t O31 = s ? O13 : op(qx,qy,qz,6,e);
               const real_t O32 = s ? O23 : op(qx,qy,qz,7,e);
               const real_t O33 = s ? op(qx,qy,qz,5,e) : op(qx,qy,qz,8,e);
               const real_t massX = mass[qz][qy][qx][0];
               const real_t massY = mass[qz][qy][qx][1];
               const real_t massZ = mass[qz][qy][qx][2];
               mass[qz][qy][qx][0] = (O11*massX)+(O12*massY)+(O13*massZ);
               mass[qz][qy][qx][1] = (O21*massX)+(O22*massY)+(O23*massZ);
               mass[qz][qy][qx][2] = (O31*massX)+(O32*massY)+(O33*massZ);
            }
         }
      }
      for (int qz = 0; qz < Q1D; ++qz)
      {
         real_t massXY[MAX_D1D][MAX_D1D];

         osc = 0;

         for (int c = 0; c < VDIM; ++c)  // loop over x, y, z components
         {
            const int D1Dz = (c == 2) ? D1D : cD1D;
            const int D1Dy = (c == 1) ? D1D : cD1D;
            const int D1Dx = (c == 0) ? D1D : cD1D;

            for (int dy = 0; dy < D1Dy; ++dy)
            {
               for (int dx = 0; dx < D1Dx; ++dx)
               {
                  massXY[dy][dx] = 0.0;
               }
            }
            for (int qy = 0; qy < Q1D; ++qy)
            {
               real_t massX[MAX_D1D];
               for (int dx = 0; dx < D1Dx; ++dx)
               {
                  massX[dx] = 0;
               }
               for (int qx = 0; qx < Q1D; ++qx)
               {
                  for (int dx = 0; dx < D1Dx; ++dx)
                  {
                     massX[dx] += mass[qz][qy][qx][c] * ((c == 0) ? Bot(dx,qx,c,e) : Bct(dx,qx,c,e));
                  }
               }
               for (int dy = 0; dy < D1Dy; ++dy)
               {
                  const real_t wy = (c == 1) ? Bot(dy,qy,c,e) : Bct(dy,qy,c,e);
                  for (int dx = 0; dx < D1Dx; ++dx)
                  {
                     massXY[dy][dx] += massX[dx] * wy;
                  }
               }
            }

            for (int dz = 0; dz < D1Dz; ++dz)
            {
               const real_t wz = (c == 2) ? Bot(dz,qz,c,e) : Bct(dz,qz,c,e);
               for (int dy = 0; dy < D1Dy; ++dy)
               {
                  for (int dx = 0; dx < D1Dx; ++dx)
                  {
                     Y(dx + ((dy + (dz * D1Dy)) * D1Dx) + osc, e) += massXY[dy][dx] * wz;
                     //printf("%f ",(Y(dx + ((dy + (dz * D1Dy)) * D1Dx) + osc, e)));
                  }
               }
            }
            osc += D1Dx * D1Dy * D1Dz;
         }  // loop c
      }  // loop qz
   }); // end of element loop

}

}
