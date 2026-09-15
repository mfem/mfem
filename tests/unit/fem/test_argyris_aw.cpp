// Copyright (c) 2010-2026, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.

#include "mfem.hpp"
#include "unit_tests.hpp"
#include <cmath>
#include <memory>

using namespace mfem;

namespace
{

constexpr real_t vertices[3][2] = {{0.0,0.0}, {1.0,0.0}, {0.0,1.0}};
constexpr int edge_vertices[3][2] = {{0,1}, {1,2}, {2,0}};

} // namespace

TEST_CASE("Argyris physical degrees of freedom", "[FiniteElement][Argyris]")
{
   ArgyrisTriangleFiniteElement fe;
   const bool reflected = GENERATE(false, true);
   IsoparametricTransformation T;
   T.SetIdentityTransformation(Geometry::TRIANGLE);
   DenseMatrix points(2,3);
   points(0,0)=0.2; points(1,0)=-0.3;
   points(0,1)=1.6; points(1,1)=0.1;
   points(0,2)=0.5; points(1,2)=reflected ? -1.1 : 0.8;
   T.SetPointMat(points);
   Vector shape(21);
   DenseMatrix grad(21,2), hess(21,3);
   for (int v = 0; v < 3; v++)
   {
      IntegrationPoint ip;
      ip.Set2(vertices[v][0],vertices[v][1]);
      T.SetIntPoint(&ip);
      fe.CalcPhysShape(T,shape);
      fe.CalcPhysDShape(T,grad);
      fe.CalcPhysHessian(T,hess);
      for (int j = 0; j < 21; j++)
      {
         REQUIRE(shape(j) == MFEM_Approx(j == 6*v ? 1.0 : 0.0));
         for (int d = 0; d < 2; d++)
         { REQUIRE(grad(j,d) == MFEM_Approx(j == 6*v+1+d ? 1.0 : 0.0)); }
         for (int d = 0; d < 3; d++)
         { REQUIRE(hess(j,d) == MFEM_Approx(j == 6*v+3+d ? 1.0 : 0.0)); }
      }
   }
   for (int e = 0; e < 3; e++)
   {
      const int a = edge_vertices[e][0], b = edge_vertices[e][1];
      const real_t nx = points(1,b)-points(1,a), ny = points(0,a)-points(0,b);
      const IntegrationPoint &ip = fe.GetNodes().IntPoint(18+e);
      T.SetIntPoint(&ip);
      fe.CalcPhysDShape(T,grad);
      for (int j = 0; j < 21; j++)
      { REQUIRE(nx*grad(j,0)+ny*grad(j,1) == MFEM_Approx(j == 18+e ? 1.0 : 0.0)); }
   }
   ArgyrisFECollection fec;
   REQUIRE(fec.DofForGeometry(Geometry::POINT) == 6);
   REQUIRE(fec.DofForGeometry(Geometry::SEGMENT) == 1);
   REQUIRE(fec.DofForGeometry(Geometry::TRIANGLE) == 0);
   std::unique_ptr<FiniteElementCollection> copy(FiniteElementCollection::New(
                                                    fec.Name()));
   REQUIRE(copy->FiniteElementForGeometry(Geometry::TRIANGLE)->GetDof() == 21);
}

TEST_CASE("Arnold-Winther physical degrees of freedom",
          "[FiniteElement][ArnoldWinther]")
{
   ArnoldWintherTriangleFiniteElement fe;
   const int geometry = GENERATE(0,1,2);
   IsoparametricTransformation T;
   T.SetIdentityTransformation(Geometry::TRIANGLE);
   DenseMatrix points(2,3);
   points = T.GetPointMat();
   if (geometry)
   {
      points(0,0)=0.2; points(1,0)=-0.3;
      points(0,1)=1.6; points(1,1)=0.1;
      points(0,2)=0.5; points(1,2)=geometry == 2 ? -1.1 : 0.8;
      T.SetPointMat(points);
   }
   DenseMatrix dofs(24);
   dofs = 0.0;
   DenseTensor shape(2,2,24);
   const int row[3] = {0,0,1}, col[3] = {0,1,1};
   for (int v = 0; v < 3; v++)
   {
      IntegrationPoint ip;
      ip.Set2(vertices[v][0],vertices[v][1]);
      T.SetIntPoint(&ip);
      fe.CalcMShape(T,shape);
      for (int c = 0; c < 3; c++)
      {
         for (int j = 0; j < 24; j++) { dofs(3*v+c,j)=shape(row[c],col[c],j); }
      }
   }
   const IntegrationRule &er = IntRules.Get(Geometry::SEGMENT,6);
   for (int e = 0; e < 3; e++)
   {
      const int a = edge_vertices[e][0], b = edge_vertices[e][1];
      const real_t dx = points(0,b)-points(0,a), dy = points(1,b)-points(1,a);
      const real_t length = std::hypot(dx,dy), tx = dx/length, ty = dy/length;
      const real_t nx = ty, ny = -tx;
      for (int q = 0; q < er.GetNPoints(); q++)
      {
         const IntegrationPoint &qp = er.IntPoint(q);
         IntegrationPoint ip;
         ip.Set2(vertices[a][0]+qp.x*(vertices[b][0]-vertices[a][0]),
                 vertices[a][1]+qp.x*(vertices[b][1]-vertices[a][1]));
         T.SetIntPoint(&ip);
         fe.CalcMShape(T,shape);
         for (int j = 0; j < 24; j++)
         {
            real_t nn = nx*nx*shape(0,0,j)+2*nx*ny*shape(0,1,j)+ny*ny*shape(1,1,j);
            real_t nt = tx*nx*shape(0,0,j)+(tx*ny+ty*nx)*shape(0,1,j)+ty*ny*shape(1,1,j);
            for (int m = 0; m < 2; m++)
            {
               real_t w = length*qp.weight*(m ? 2*qp.x-1 : 1);
               dofs(9+4*e+2*m,j) += w*nn;
               dofs(10+4*e+2*m,j) += w*nt;
            }
         }
      }
   }
   const IntegrationRule &ir = IntRules.Get(Geometry::TRIANGLE,6);
   for (int q = 0; q < ir.GetNPoints(); q++)
   {
      const IntegrationPoint &ip = ir.IntPoint(q);
      T.SetIntPoint(&ip);
      fe.CalcMShape(T,shape);
      const DenseMatrix &K = T.InverseJacobian();
      const real_t det2 = T.Jacobian().Det()*T.Jacobian().Det();
      for (int c = 0; c < 3; c++)
      {
         for (int j = 0; j < 24; j++)
         {
            for (int a = 0; a < 2; a++)
            {
               for (int b = 0; b < 2; b++)
               { dofs(21+c,j) += ip.weight*det2*K(row[c],a)*shape(a,b,j)*K(col[c],b); }
            }
         }
      }
   }
   for (int i = 0; i < 24; i++)
   {
      for (int j = 0; j < 24; j++)
      { REQUIRE(dofs(i,j) == MFEM_Approx(i == j ? 1.0 : 0.0)); }
   }
   ArnoldWintherFECollection fec;
   REQUIRE(fec.DofForGeometry(Geometry::POINT) == 3);
   REQUIRE(fec.DofForGeometry(Geometry::SEGMENT) == 4);
   REQUIRE(fec.DofForGeometry(Geometry::TRIANGLE) == 3);
   REQUIRE(fe.GetOrder() == 3);
   std::unique_ptr<FiniteElementCollection> copy(FiniteElementCollection::New(
                                                    fec.Name()));
   REQUIRE(copy->FiniteElementForGeometry(Geometry::TRIANGLE)->GetDof() == 24);
}

TEST_CASE("Argyris to Arnold-Winther Airy interpolation",
          "[DiscreteInterpolator][AiryInterpolator][ArnoldWinther]")
{
   Mesh mesh = Mesh::MakeCartesian2D(2, 2, Element::TRIANGLE, true);
   for (int vertex = 0; vertex < mesh.GetNV(); vertex++)
   {
      real_t *point = mesh.GetVertex(vertex);
      const real_t x = point[0];
      const real_t y = point[1];
      point[0] = 0.2 + 1.4*x + 0.3*y;
      point[1] = -0.1 + 0.2*x + 0.8*y;
   }
   ArgyrisFECollection hct_collection;
   ArnoldWintherFECollection jm_collection;
   FiniteElementSpace hct_space(&mesh, &hct_collection);
   FiniteElementSpace jm_space(&mesh, &jm_collection);
   DiscreteLinearOperator airy(&hct_space, &jm_space);
   airy.AddDomainInterpolator(new AiryInterpolator);
   airy.Assemble();
   airy.Finalize();

   Vector potential(hct_space.GetVSize()), stress(jm_space.GetVSize());
   for (int i = 0; i < potential.Size(); i++)
   {
      potential(i) = std::sin(real_t(i + 1));
   }
   airy.Mult(potential, stress);

   const IntegrationRule &base = IntRules.Get(Geometry::TRIANGLE, 6);
   std::unique_ptr<IntegrationRule> rule(base.ApplyToTriangleAlfeldSplit());
   real_t error = 0.0;
   real_t divergence = 0.0;
   for (int element = 0; element < mesh.GetNE(); element++)
   {
      Array<int> hct_dofs, jm_dofs;
      hct_space.GetElementDofs(element, hct_dofs);
      jm_space.GetElementDofs(element, jm_dofs);
      Vector local_potential, local_stress;
      potential.GetSubVector(hct_dofs, local_potential);
      stress.GetSubVector(jm_dofs, local_stress);
      ElementTransformation *T = mesh.GetElementTransformation(element);
      DenseMatrix hessian(21, 3), divshape(24, 2);
      DenseTensor matrix_shape(2, 2, 24);
      for (int q = 0; q < rule->GetNPoints(); q++)
      {
         const IntegrationPoint &ip = rule->IntPoint(q);
         T->SetIntPoint(&ip);
         hct_space.GetFE(element)->CalcPhysHessian(*T, hessian);
         jm_space.GetFE(element)->CalcMShape(*T, matrix_shape);
         jm_space.GetFE(element)->CalcPhysDivShape(*T, divshape);
         real_t computed[3] = {0.0, 0.0, 0.0};
         real_t div[2] = {0.0, 0.0};
         for (int i = 0; i < 24; i++)
         {
            computed[0] += local_stress(i)*matrix_shape(0,0,i);
            computed[1] += local_stress(i)*matrix_shape(0,1,i);
            computed[2] += local_stress(i)*matrix_shape(1,1,i);
            div[0] += local_stress(i)*divshape(i,0);
            div[1] += local_stress(i)*divshape(i,1);
         }
         real_t expected[3] = {0.0, 0.0, 0.0};
         for (int i = 0; i < 21; i++)
         {
            expected[0] += local_potential(i)*hessian(i,2);
            expected[1] -= local_potential(i)*hessian(i,1);
            expected[2] += local_potential(i)*hessian(i,0);
         }
         for (int i = 0; i < 3; i++)
         {
            error = std::max(error, std::abs(computed[i] - expected[i]));
         }
         divergence = std::max(divergence,
                               std::hypot(div[0], div[1]));
      }
   }
   BilinearForm mass(&jm_space), biharmonic(&hct_space), divdiv(&jm_space);
   mass.AddDomainIntegrator(new MatrixFEMassIntegrator);
   biharmonic.AddDomainIntegrator(new HessianIntegrator);
   divdiv.AddDomainIntegrator(new MatrixDivDivIntegrator);
   mass.Assemble(); mass.Finalize();
   biharmonic.Assemble(); biharmonic.Finalize();
   divdiv.Assemble(); divdiv.Finalize();
   Vector work(stress.Size()), pwork(potential.Size());
   mass.Mult(stress,work);
   biharmonic.Mult(potential,pwork);
   REQUIRE((stress*work) == Approx(potential*pwork).epsilon(1e-10));
   divdiv.Mult(stress,work);
   REQUIRE(work.Norml2() < 1e-7);
   REQUIRE(error < 1e-9);
   REQUIRE(divergence < 1e-9);
}

TEST_CASE("Arnold-Winther projection and physical refinement",
          "[Transfer][ArnoldWinther]")
{
   Mesh coarse_mesh = Mesh::MakeCartesian2D(2,1,Element::TRIANGLE,true);
   for (int v = 0; v < coarse_mesh.GetNV(); v++)
   {
      real_t *p = coarse_mesh.GetVertex(v);
      const real_t x = p[0], y = p[1];
      p[0] = 0.3+1.7*x+0.4*y;
      p[1] = -0.2+0.2*x+0.9*y;
   }
   Mesh fine_mesh(coarse_mesh);
   fine_mesh.UniformRefinement();
   ArnoldWintherFECollection fec;
   FiniteElementSpace coarse(&coarse_mesh,&fec), fine(&fine_mesh,&fec);
   OperatorHandle P(GENERATE(Operator::MFEM_SPARSEMAT,Operator::ANY_TYPE));
   fine.GetTransferOperator(coarse,P);
   H1_FECollection h1fec(2,2);
   VectorFunctionCoefficient exact(3,[](const Vector &p, Vector &s)
   {
      s.SetSize(3);
      s(0) = 1+p(0)*p(0)+p(1);
      s(1) = 0.2+p(0)*p(1);
      s(2) = -0.3+p(1)*p(1)+p(0);
   });
   Vector x(coarse.GetVSize()), y(fine.GetVSize());
   for (int level = 0; level < 2; level++)
   {
      FiniteElementSpace &fes = level ? fine : coarse;
      FiniteElementSpace h1(fes.GetMesh(),&h1fec,3,Ordering::byVDIM);
      GridFunction source(&h1);
      source.ProjectCoefficient(exact);
      DiscreteLinearOperator pi(&h1,&fes);
      pi.AddDomainInterpolator(new IdentityInterpolator);
      pi.Assemble(); pi.Finalize();
      Vector projected(fes.GetVSize());
      pi.Mult(source,projected);
      if (!level) { x = projected; P->Mult(x,y); }
      else
      {
         projected -= y;
         REQUIRE(projected.Normlinf() < 1e-10);
      }
      const Vector &coefficients = level ? y : x;
      MatrixFunctionCoefficient tensor(2, [](const Vector &p, DenseMatrix &value)
      {
         value.SetSize(2);
         value(0,0) = 1+p(0)*p(0)+p(1);
         value(0,1) = value(1,0) = 0.2+p(0)*p(1);
         value(1,1) = -0.3+p(1)*p(1)+p(0);
      });
      BilinearForm mass(&fes);
      mass.AddDomainIntegrator(new MatrixFEMassIntegrator);
      mass.Assemble(); mass.Finalize();
      LinearForm load(&fes);
      load.AddDomainIntegrator(new MatrixFEDomainLFIntegrator(tensor));
      load.Assemble();
      Vector residual(fes.GetVSize());
      mass.Mult(coefficients,residual);
      residual -= load;
      REQUIRE(residual.Normlinf() < 1e-10);
      const IntegrationRule &ir = IntRules.Get(Geometry::TRIANGLE,6);
      for (int e = 0; e < fes.GetNE(); e++)
      {
         Array<int> dofs;
         fes.GetElementDofs(e,dofs);
         Vector local, p(2), expected(3), div(2);
         coefficients.GetSubVector(dofs,local);
         auto *T = fes.GetElementTransformation(e);
         DenseTensor shape(2,2,24);
         DenseMatrix divshape(24,2);
         for (int q = 0; q < ir.GetNPoints(); q++)
         {
            const IntegrationPoint &ip = ir.IntPoint(q);
            T->SetIntPoint(&ip);
            fes.GetFE(e)->CalcMShape(*T,shape);
            fes.GetFE(e)->CalcPhysDivShape(*T,divshape);
            exact.Eval(expected,*T,ip);
            for (int j = 0; j < 24; j++)
            {
               expected(0) -= local(j)*shape(0,0,j);
               expected(1) -= local(j)*shape(0,1,j);
               expected(2) -= local(j)*shape(1,1,j);
            }
            REQUIRE(expected.Normlinf() < 1e-10);
            T->Transform(ip,p);
            divshape.MultTranspose(local,div);
            REQUIRE(div(0) == MFEM_Approx(3*p(0)));
            REQUIRE(div(1) == MFEM_Approx(3*p(1)));
         }
      }
   }
}

TEST_CASE("Argyris reproduces physical quintics", "[FiniteElement][Argyris]")
{
   ArgyrisTriangleFiniteElement fe;
   IsoparametricTransformation T;
   T.SetIdentityTransformation(Geometry::TRIANGLE);
   DenseMatrix points(2,3);
   points(0,0)=0.2; points(1,0)=-0.3;
   points(0,1)=1.6; points(1,1)=0.1;
   points(0,2)=0.5; points(1,2)=0.8;
   T.SetPointMat(points);
   auto polynomial = [](const Vector &p, Vector &jet)
   {
      const real_t x = p(0), y = p(1);
      jet(0) = x*x*x*x*x+2*x*x*y*y*y+1+x-y;
      jet(1) = 5*x*x*x*x+4*x*y*y*y+1;
      jet(2) = 6*x*x*y*y-1;
      jet(3) = 20*x*x*x+4*y*y*y;
      jet(4) = 12*x*y*y;
      jet(5) = 12*x*x*y;
   };
   Vector dofs(21), p(2), jet(6), shape(21), gradient(2), hessian(3);
   DenseMatrix grad(21,2), hess(21,3);
   for (int v = 0; v < 3; v++)
   {
      T.Transform(fe.GetNodes().IntPoint(6*v),p);
      polynomial(p,jet);
      for (int d = 0; d < 6; d++) { dofs(6*v+d) = jet(d); }
   }
   for (int e = 0; e < 3; e++)
   {
      const int a = edge_vertices[e][0], b = edge_vertices[e][1];
      T.Transform(fe.GetNodes().IntPoint(18+e),p);
      polynomial(p,jet);
      dofs(18+e) = (points(1,b)-points(1,a))*jet(1)
                   -(points(0,b)-points(0,a))*jet(2);
   }
   const IntegrationRule &ir = IntRules.Get(Geometry::TRIANGLE,10);
   for (int q = 0; q < ir.GetNPoints(); q++)
   {
      const IntegrationPoint &ip = ir.IntPoint(q);
      T.SetIntPoint(&ip);
      T.Transform(ip,p);
      polynomial(p,jet);
      fe.CalcPhysShape(T,shape);
      fe.CalcPhysDShape(T,grad);
      fe.CalcPhysHessian(T,hess);
      grad.MultTranspose(dofs,gradient);
      hess.MultTranspose(dofs,hessian);
      REQUIRE((shape*dofs) == MFEM_Approx(jet(0)));
      for (int d = 0; d < 2; d++)
      { REQUIRE(gradient(d) == MFEM_Approx(jet(1+d))); }
      for (int d = 0; d < 3; d++)
      { REQUIRE(hessian(d) == MFEM_Approx(jet(3+d))); }
   }
}

TEST_CASE("Arnold-Winther cubic transfer and linear divergence",
          "[FiniteElement][Transfer][ArnoldWinther]")
{
   ArnoldWintherTriangleFiniteElement fe;
   IsoparametricTransformation T;
   T.SetIdentityTransformation(Geometry::TRIANGLE);
   DenseMatrix points(2,3);
   points(0,0)=0.1; points(1,0)=0.2;
   points(0,1)=0.6; points(1,1)=0.1;
   points(0,2)=0.2; points(1,2)=0.6;
   T.SetPointMat(points);
   DenseMatrix transfer;
   fe.GetLocalInterpolation(T,transfer);
   DenseTensor coarse(2,2,24), fine(2,2,24);
   DenseMatrix div(24,2), div_vertex[3];
   for (int v = 0; v < 3; v++)
   {
      div_vertex[v].SetSize(24,2);
      IntegrationPoint ip;
      ip.Set2(vertices[v][0],vertices[v][1]);
      fe.CalcDivShape(ip,div_vertex[v]);
   }
   Vector point(2);
   const IntegrationRule &ir = IntRules.Get(Geometry::TRIANGLE,6);
   for (int q = 0; q < ir.GetNPoints(); q++)
   {
      const IntegrationPoint &ip = ir.IntPoint(q);
      T.SetIntPoint(&ip);
      T.Transform(ip,point);
      IntegrationPoint coarse_ip;
      coarse_ip.Set2(point(0),point(1));
      fe.CalcMShape(coarse_ip,coarse);
      fe.CalcMShape(T,fine);
      fe.CalcDivShape(ip,div);
      for (int j = 0; j < 24; j++)
      {
         for (int a = 0; a < 2; a++)
         {
            const real_t linear = (1-ip.x-ip.y)*div_vertex[0](j,a)
                                  +ip.x*div_vertex[1](j,a)+ip.y*div_vertex[2](j,a);
            REQUIRE(div(j,a) == MFEM_Approx(linear));
            for (int b = 0; b < 2; b++)
            {
               real_t value = 0.0;
               for (int k = 0; k < 24; k++) { value += transfer(k,j)*fine(a,b,k); }
               REQUIRE(value == MFEM_Approx(coarse(a,b,j)));
            }
         }
      }
   }
}

namespace
{
// Forward evaluations through an unrelated FE type to check that quadrature
// selection uses the partition interface rather than concrete element types.
class PartitionTestElement : public FiniteElement
{
   const FiniteElement &element;
public:
   PartitionTestElement(const FiniteElement &fe)
      : FiniteElement(fe.GetDim(), fe.GetGeomType(), fe.GetDof(), fe.GetOrder()),
        element(fe)
   {
      range_type = fe.GetRangeType();
      map_type = fe.GetMapType();
   }
   IntegrationPartition GetIntegrationPartition() const override
   { return element.GetIntegrationPartition(); }
   void CalcShape(const IntegrationPoint &ip, Vector &shape) const override
   { element.CalcShape(ip, shape); }
   void CalcDShape(const IntegrationPoint &ip, DenseMatrix &shape) const override
   { element.CalcDShape(ip, shape); }
   void CalcPhysHessian(ElementTransformation &T,
                        DenseMatrix &shape) const override
   { element.CalcPhysHessian(T, shape); }
   void CalcMShape(ElementTransformation &T, DenseTensor &shape) const override
   { element.CalcMShape(T, shape); }
   void CalcPhysDivShape(ElementTransformation &T,
                         DenseMatrix &shape) const override
   { element.CalcPhysDivShape(T, shape); }
};
}

TEST_CASE("Finite element integration partitions",
          "[FiniteElement][HCT][JohnsonMercier]")
{
   HCTTriangleFiniteElement hct;
   JohnsonMercierTriangleFiniteElement jm;
   ArgyrisTriangleFiniteElement argyris;
   ArnoldWintherTriangleFiniteElement aw;
   H1_TriangleElement h1(2);
   using Partition = FiniteElement::IntegrationPartition;
   const FiniteElement *elements[] = {&hct, &jm, &argyris, &aw, &h1};
   for (int i = 0; i < 5; i++)
   {
      REQUIRE(elements[i]->GetIntegrationPartition() ==
              (i < 2 ? Partition::ALFELD : Partition::UNSPLIT));
   }
   IsoparametricTransformation T;
   T.SetIdentityTransformation(Geometry::TRIANGLE);
   PartitionTestElement scalar(hct), tensor(jm);
   DenseMatrix actual, expected;
   HessianIntegrator hessian;
   hessian.AssembleElementMatrix(hct, T, expected);
   hessian.AssembleElementMatrix(scalar, T, actual);
   actual -= expected;
   REQUIRE(actual.MaxMaxNorm() < 1e-12);

   MatrixFEMassIntegrator mass;
   mass.AssembleElementMatrix(jm, T, expected);
   mass.AssembleElementMatrix(tensor, T, actual);
   actual -= expected;
   REQUIRE(actual.MaxMaxNorm() < 1e-12);

   MatrixDivDivIntegrator divdiv;
   divdiv.AssembleElementMatrix(jm, T, expected);
   divdiv.AssembleElementMatrix(tensor, T, actual);
   actual -= expected;
   REQUIRE(actual.MaxMaxNorm() < 1e-12);

   DenseMatrix identity(2);
   identity = 0.0;
   identity(0,0) = identity(1,1) = 1.0;
   MatrixConstantCoefficient coefficient(identity);
   MatrixFEDomainLFIntegrator load(coefficient);
   Vector actual_load, expected_load;
   load.AssembleRHSElementVect(jm, T, expected_load);
   load.AssembleRHSElementVect(tensor, T, actual_load);
   actual_load -= expected_load;
   REQUIRE(actual_load.Normlinf() < 1e-12);
}

TEST_CASE("Physical transfer through the finite element interface",
          "[FiniteElement][Transfer][ArnoldWinther][JohnsonMercier]")
{
   H1_TriangleElement h1(2);
   JohnsonMercierTriangleFiniteElement jm;
   ArnoldWintherTriangleFiniteElement aw;
   const FiniteElement *elements[] = {&h1, &jm, &aw};
   IsoparametricTransformation child, fine;
   child.SetIdentityTransformation(Geometry::TRIANGLE);
   fine.SetIdentityTransformation(Geometry::TRIANGLE);
   DenseMatrix points(2,3);
   points(0,0)=0.2; points(1,0)=-0.3;
   points(0,1)=1.6; points(1,1)=0.1;
   points(0,2)=0.5; points(1,2)=0.8;
   fine.SetPointMat(points);
   for (const FiniteElement *fe : elements)
   {
      REQUIRE(fe->RequiresPhysicalTransfer() == (fe != &h1));
      DenseMatrix reference, corrected;
      fe->GetLocalInterpolation(child, reference);
      fe->GetPhysicalTransferMatrix(reference, child, fine, corrected);
      // An identity child map must transfer the physical DOFs identically,
      // even when the parent/fine element is skewed.
      for (int i = 0; i < fe->GetDof(); i++)
      {
         for (int j = 0; j < fe->GetDof(); j++)
         { REQUIRE(corrected(i,j) == MFEM_Approx(i == j ? 1.0 : 0.0)); }
      }
   }
   // The default correction copies an arbitrary reference matrix without
   // inspecting either transformation, which need not be initialized.
   IsoparametricTransformation unused_child, unused_fine;
   DenseMatrix reference(6), corrected;
   for (int i = 0; i < 6; i++)
   {
      for (int j = 0; j < 6; j++) { reference(i,j) = i+2*j; }
   }
   const FiniteElement &fe = h1;
   fe.GetPhysicalTransferMatrix(reference, unused_child, unused_fine, corrected);
   corrected -= reference;
   REQUIRE(corrected.MaxMaxNorm() == 0.0);
}
