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

TEST_CASE("HZZZ physical degrees of freedom",
          "[FiniteElement][HZZZ]")
{
   HuangZhangZhouZhuTriangleFiniteElement fe;
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
   DenseMatrix dofs(21);
   dofs = 0.0;
   DenseTensor shape(2,2,21);
   const int row[3] = {0,0,1}, col[3] = {0,1,1};
   for (int v = 0; v < 3; v++)
   {
      IntegrationPoint ip;
      ip.Set2(vertices[v][0],vertices[v][1]);
      T.SetIntPoint(&ip);
      fe.CalcMShape(T,shape);
      for (int c = 0; c < 3; c++)
      {
         for (int j = 0; j < 21; j++) { dofs(3*v+c,j)=shape(row[c],col[c],j); }
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
         for (int j = 0; j < 21; j++)
         {
            real_t nn = nx*nx*shape(0,0,j)+2*nx*ny*shape(0,1,j)+ny*ny*shape(1,1,j);
            real_t nt = tx*nx*shape(0,0,j)+(tx*ny+ty*nx)*shape(0,1,j)+ty*ny*shape(1,1,j);
            for (int m = 0; m < 2; m++)
            {
               real_t w = length*qp.weight*(m ? 2*qp.x-1 : 1);
               dofs(9+3*e+2*m,j) += w*nn;
               if (!m) { dofs(10+3*e,j) += w*nt; }
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
         for (int j = 0; j < 21; j++)
         {
            for (int a = 0; a < 2; a++)
            {
               for (int b = 0; b < 2; b++)
               {
                  const real_t modes[1] = {1.0};
                  for (int m = 0; m < 1; m++)
                  {
                     dofs(18+3*m+c,j) += ip.weight*det2*modes[m]*K(row[c],a)*
                                         shape(a,b,j)*K(col[c],b);
                  }
               }
            }
         }
      }
   }
   for (int i = 0; i < 21; i++)
   {
      for (int j = 0; j < 21; j++)
      { REQUIRE(dofs(i,j) == MFEM_Approx(i == j ? 1.0 : 0.0)); }
   }
   HZZZFECollection fec;
   REQUIRE(fec.DofForGeometry(Geometry::POINT) == 3);
   REQUIRE(fec.DofForGeometry(Geometry::SEGMENT) == 3);
   REQUIRE(fec.DofForGeometry(Geometry::TRIANGLE) == 3);
   REQUIRE(fe.GetOrder() == 3);
   std::unique_ptr<FiniteElementCollection> copy(FiniteElementCollection::New(
                                                    fec.Name()));
   REQUIRE(copy->FiniteElementForGeometry(Geometry::TRIANGLE)->GetDof() == 21);
}

TEST_CASE("Bell to HZZZ Airy interpolation",
          "[DiscreteInterpolator][AiryInterpolator][HZZZ]")
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
   BellFECollection hct_collection;
   HZZZFECollection jm_collection;
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

   const IntegrationRule *rule = &IntRules.Get(Geometry::TRIANGLE, 6);
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
      DenseMatrix hessian(18, 3), divshape(21, 2);
      DenseTensor matrix_shape(2, 2, 21);
      for (int q = 0; q < rule->GetNPoints(); q++)
      {
         const IntegrationPoint &ip = rule->IntPoint(q);
         T->SetIntPoint(&ip);
         hct_space.GetFE(element)->CalcPhysHessian(*T, hessian);
         jm_space.GetFE(element)->CalcMShape(*T, matrix_shape);
         jm_space.GetFE(element)->CalcPhysDivShape(*T, divshape);
         real_t computed[3] = {0.0, 0.0, 0.0};
         real_t div[2] = {0.0, 0.0};
         for (int i = 0; i < 21; i++)
         {
            computed[0] += local_stress(i)*matrix_shape(0,0,i);
            computed[1] += local_stress(i)*matrix_shape(0,1,i);
            computed[2] += local_stress(i)*matrix_shape(1,1,i);
            div[0] += local_stress(i)*divshape(i,0);
            div[1] += local_stress(i)*divshape(i,1);
         }
         real_t expected[3] = {0.0, 0.0, 0.0};
         for (int i = 0; i < 18; i++)
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

TEST_CASE("HZZZ projection and physical refinement",
          "[Transfer][HZZZ]")
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
   HZZZFECollection fec;
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
         DenseTensor shape(2,2,21);
         DenseMatrix divshape(21,2);
         for (int q = 0; q < ir.GetNPoints(); q++)
         {
            const IntegrationPoint &ip = ir.IntPoint(q);
            T->SetIntPoint(&ip);
            fes.GetFE(e)->CalcMShape(*T,shape);
            fes.GetFE(e)->CalcPhysDivShape(*T,divshape);
            exact.Eval(expected,*T,ip);
            for (int j = 0; j < 21; j++)
            {
               expected(0) -= local(j)*shape(0,0,j);
               expected(1) -= local(j)*shape(0,1,j);
               expected(2) -= local(j)*shape(1,1,j);
            }
            REQUIRE(expected.Normlinf() < 1e-10);
            T->Transform(ip,p);
            divshape.MultTranspose(local,div);
            div(0) -= 3*p(0);
            div(1) -= 3*p(1);
            // Derivatives amplify cancellation in the cubic moment basis.
            REQUIRE(div.Normlinf() < 1e-10);
         }
      }
   }
}

TEST_CASE("HZZZ physical traction continuity",
          "[FiniteElementSpace][HZZZ]")
{
   Mesh mesh = Mesh::MakeCartesian2D(2, 2, Element::TRIANGLE, true);
   for (int v = 0; v < mesh.GetNV(); v++)
   {
      real_t *p = mesh.GetVertex(v);
      const real_t x = p[0], y = p[1];
      p[0] = 0.2+1.4*x+0.3*y;
      p[1] = -0.1+0.2*x+0.8*y;
   }
   HZZZFECollection fec;
   FiniteElementSpace fes(&mesh, &fec);
   Vector x(fes.GetVSize());
   for (int i = 0; i < x.Size(); i++) { x(i) = std::sin(real_t(i + 1)); }
   real_t jump = 0.0;
   for (int f = 0; f < mesh.GetNumFaces(); f++)
   {
      auto *T = mesh.GetFaceElementTransformations(f);
      if (T->Elem2No < 0) { continue; }
      for (real_t r : {0.0,0.23,0.57,0.81,1.0})
      {
         IntegrationPoint ip;
         ip.Set1w(r, 1.0);
         T->SetAllIntPoints(&ip);
         Vector normal(2);
         CalcOrtho(T->Jacobian(), normal);
         Vector traction[2] = {Vector(2), Vector(2)};
         for (int side = 0; side < 2; side++)
         {
            const int e = side ? T->Elem2No : T->Elem1No;
            Array<int> dofs;
            DofTransformation trans;
            fes.GetElementDofs(e, dofs, trans);
            Vector local;
            x.GetSubVector(dofs, local);
            trans.InvTransformPrimal(local);
            DenseTensor shape(2, 2, 21);
            fes.GetFE(e)->CalcMShape(side ? *T->Elem2 : *T->Elem1, shape);
            traction[side] = 0.0;
            for (int k = 0; k < 21; k++)
            {
               for (int i = 0; i < 2; i++)
               {
                  for (int j = 0; j < 2; j++)
                  { traction[side](i) += local(k)*shape(i,j,k)*normal(j); }
               }
            }
         }
         traction[0] -= traction[1];
         jump = std::max(jump, traction[0].Norml2());
      }
   }
   REQUIRE(jump < 1e-10);
}

TEST_CASE("Bell physical degrees of freedom", "[FiniteElement][Bell]")
{
   BellTriangleFiniteElement fe;
   const bool reflected = GENERATE(false, true);
   IsoparametricTransformation T;
   T.SetIdentityTransformation(Geometry::TRIANGLE);
   DenseMatrix points(2,3);
   points(0,0)=0.2; points(1,0)=-0.3;
   points(0,1)=1.6; points(1,1)=0.1;
   points(0,2)=0.5; points(1,2)=reflected ? -1.1 : 0.8;
   T.SetPointMat(points);
   Vector shape(18);
   DenseMatrix grad(18,2), hess(18,3);
   for (int v = 0; v < 3; v++)
   {
      IntegrationPoint ip;
      ip.Set2(vertices[v][0],vertices[v][1]);
      T.SetIntPoint(&ip);
      fe.CalcPhysShape(T,shape);
      fe.CalcPhysDShape(T,grad);
      fe.CalcPhysHessian(T,hess);
      for (int j = 0; j < 18; j++)
      {
         REQUIRE(shape(j) == MFEM_Approx(j == 6*v ? 1.0 : 0.0));
         for (int d = 0; d < 2; d++)
         { REQUIRE(grad(j,d) == MFEM_Approx(j == 6*v+1+d ? 1.0 : 0.0)); }
         for (int d = 0; d < 3; d++)
         { REQUIRE(hess(j,d) == MFEM_Approx(j == 6*v+3+d ? 1.0 : 0.0)); }
      }
   }
   BellFECollection fec;
   REQUIRE(fec.DofForGeometry(Geometry::POINT) == 6);
   REQUIRE(fec.DofForGeometry(Geometry::SEGMENT) == 0);
   REQUIRE(fec.DofForGeometry(Geometry::TRIANGLE) == 0);
   std::unique_ptr<FiniteElementCollection> copy(FiniteElementCollection::New(
                                                    fec.Name()));
   REQUIRE(copy->FiniteElementForGeometry(Geometry::TRIANGLE)->GetDof() == 18);
}

TEST_CASE("Bell reproduces physical quartics", "[FiniteElement][Bell]")
{
   BellTriangleFiniteElement fe;
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
      jet(0) = x*x*x*x+2*x*x*y*y+1+x-y;
      jet(1) = 4*x*x*x+4*x*y*y+1;
      jet(2) = 4*x*x*y-1;
      jet(3) = 12*x*x+4*y*y;
      jet(4) = 8*x*y;
      jet(5) = 4*x*x;
   };
   Vector dofs(18), p(2), jet(6), shape(18), gradient(2), hessian(3);
   DenseMatrix grad(18,2), hess(18,3);
   for (int v = 0; v < 3; v++)
   {
      T.Transform(fe.GetNodes().IntPoint(6*v),p);
      polynomial(p,jet);
      for (int d = 0; d < 6; d++) { dofs(6*v+d) = jet(d); }
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

TEST_CASE("Bell and HZZZ physical edge constraints",
          "[FiniteElement][Bell][HZZZ]")
{
   BellTriangleFiniteElement bell;
   HuangZhangZhouZhuTriangleFiniteElement stress;
   IsoparametricTransformation T;
   T.SetIdentityTransformation(Geometry::TRIANGLE);
   DenseMatrix points(2,3);
   const bool reflected = GENERATE(false,true);
   points(0,0)=0.2; points(1,0)=-0.3;
   points(0,1)=1.6; points(1,1)=0.1;
   points(0,2)=0.5; points(1,2)=reflected ? -1.1 : 0.8;
   T.SetPointMat(points);
   for (int e = 0; e < 3; e++)
   {
      const int a = edge_vertices[e][0], b = edge_vertices[e][1];
      const real_t tx = points(0,b)-points(0,a), ty = points(1,b)-points(1,a);
      const real_t nx = ty, ny = -tx;
      Vector fourth(18), third(21);
      fourth = 0.0; third = 0.0;
      for (int q = 0; q < 5; q++)
      {
         const real_t r = q/4.0;
         IntegrationPoint ip;
         ip.Set2((1-r)*vertices[a][0]+r*vertices[b][0],
                 (1-r)*vertices[a][1]+r*vertices[b][1]);
         T.SetIntPoint(&ip);
         DenseMatrix grad(18,2);
         bell.CalcPhysDShape(T,grad);
         const int weights[5] = {1,-4,6,-4,1};
         for (int j = 0; j < 18; j++)
         { fourth(j) += weights[q]*(nx*grad(j,0)+ny*grad(j,1)); }
      }
      for (int q = 0; q < 4; q++)
      {
         const real_t r = q/3.0;
         IntegrationPoint ip;
         ip.Set2((1-r)*vertices[a][0]+r*vertices[b][0],
                 (1-r)*vertices[a][1]+r*vertices[b][1]);
         T.SetIntPoint(&ip);
         DenseTensor shape(2,2,21);
         stress.CalcMShape(T,shape);
         const int weights[4] = {-1,3,-3,1};
         for (int j = 0; j < 21; j++)
         {
            third(j) += weights[q]*(tx*nx*shape(0,0,j)+
                                    (tx*ny+ty*nx)*shape(0,1,j)+ty*ny*shape(1,1,j));
         }
      }
      REQUIRE(fourth.Normlinf() < 1e-10);
      REQUIRE(third.Normlinf() < 1e-10);
   }
   DenseMatrix vertex_div[3], div(21,2);
   for (int v = 0; v < 3; v++)
   {
      IntegrationPoint ip;
      ip.Set2(vertices[v][0],vertices[v][1]);
      T.SetIntPoint(&ip);
      vertex_div[v].SetSize(21,2);
      stress.CalcPhysDivShape(T,vertex_div[v]);
   }
   for (int q = 0; q < 6; q++)
   {
      const auto &ip = IntRules.Get(Geometry::TRIANGLE,4).IntPoint(q);
      T.SetIntPoint(&ip);
      stress.CalcPhysDivShape(T,div);
      for (int i = 0; i < 21; i++)
      {
         for (int d = 0; d < 2; d++)
         {
            REQUIRE(div(i,d) == MFEM_Approx((1-ip.x-ip.y)*vertex_div[0](i,d)+
                                            ip.x*vertex_div[1](i,d)+ip.y*vertex_div[2](i,d)));
         }
      }
   }
}

TEST_CASE("Bell physical C1 continuity", "[FiniteElementSpace][Bell]")
{
   Mesh mesh = Mesh::MakeCartesian2D(2,2,Element::TRIANGLE,true);
   for (int v = 0; v < mesh.GetNV(); v++)
   {
      real_t *p = mesh.GetVertex(v);
      const real_t x = p[0], y = p[1];
      p[0] = 0.2+1.4*x+0.3*y;
      p[1] = -0.1+0.2*x+0.8*y;
   }
   BellFECollection fec;
   FiniteElementSpace fes(&mesh,&fec);
   Vector x(fes.GetVSize());
   x.Randomize(17);
   for (int f = 0; f < mesh.GetNumFaces(); f++)
   {
      auto *T = mesh.GetFaceElementTransformations(f);
      if (T->Elem2No < 0) { continue; }
      for (real_t r : {0.0,0.21,0.57,0.83,1.0})
      {
         IntegrationPoint ip;
         ip.Set1w(r,1.0);
         T->SetAllIntPoints(&ip);
         Vector trace[2] = {Vector(3),Vector(3)};
         for (int side = 0; side < 2; side++)
         {
            const int e = side ? T->Elem2No : T->Elem1No;
            Array<int> dofs;
            fes.GetElementDofs(e,dofs);
            Vector local, shape(18), derivative(2);
            x.GetSubVector(dofs,local);
            DenseMatrix grad(18,2);
            auto &eltrans = side ? *T->Elem2 : *T->Elem1;
            fes.GetFE(e)->CalcPhysShape(eltrans,shape);
            fes.GetFE(e)->CalcPhysDShape(eltrans,grad);
            grad.MultTranspose(local,derivative);
            trace[side](0) = shape*local;
            trace[side](1) = derivative(0);
            trace[side](2) = derivative(1);
         }
         trace[0] -= trace[1];
         REQUIRE(trace[0].Normlinf() < 1e-10);
      }
   }
}

TEST_CASE("Bell and HZZZ physical local interpolation",
          "[Transfer][Bell][HZZZ]")
{
   BellTriangleFiniteElement bell;
   HuangZhangZhouZhuTriangleFiniteElement stress;
   IsoparametricTransformation coarse, child, fine;
   coarse.SetIdentityTransformation(Geometry::TRIANGLE);
   child.SetIdentityTransformation(Geometry::TRIANGLE);
   fine.SetIdentityTransformation(Geometry::TRIANGLE);
   const bool reflected = GENERATE(false,true);
   DenseMatrix points(2,3), child_points(2,3), fine_points(2,3);
   points(0,0)=0.2; points(1,0)=-0.3;
   points(0,1)=1.6; points(1,1)=0.1;
   points(0,2)=0.5; points(1,2)=reflected ? -1.1 : 0.8;
   coarse.SetPointMat(points);
   child_points(0,0)=0.1; child_points(1,0)=0.2;
   child_points(0,1)=0.6; child_points(1,1)=0.1;
   child_points(0,2)=0.2; child_points(1,2)=0.6;
   child.SetPointMat(child_points);
   for (int v = 0; v < 3; v++)
   {
      IntegrationPoint ip;
      ip.Set2(child_points(0,v),child_points(1,v));
      Vector point(fine_points.GetColumn(v),2);
      coarse.Transform(ip,point);
   }
   fine.SetPointMat(fine_points);
   DenseMatrix reference, transfer;
   bell.GetLocalInterpolation(child,reference);
   bell.GetPhysicalTransferMatrix(reference,child,fine,transfer);
   for (int v = 0; v < 3; v++)
   {
      IntegrationPoint ip;
      ip.Set2(child_points(0,v),child_points(1,v));
      coarse.SetIntPoint(&ip);
      Vector shape(18);
      DenseMatrix grad(18,2), hess(18,3);
      bell.CalcPhysShape(coarse,shape);
      bell.CalcPhysDShape(coarse,grad);
      bell.CalcPhysHessian(coarse,hess);
      for (int j = 0; j < 18; j++)
      {
         REQUIRE(transfer(6*v,j) == MFEM_Approx(shape(j)));
         for (int d = 0; d < 2; d++)
         { REQUIRE(transfer(6*v+1+d,j) == MFEM_Approx(grad(j,d))); }
         for (int d = 0; d < 3; d++)
         { REQUIRE(transfer(6*v+3+d,j) == MFEM_Approx(hess(j,d))); }
      }
   }
   stress.GetLocalInterpolation(child,reference);
   stress.GetPhysicalTransferMatrix(reference,child,fine,transfer);
   // Independently interpolate each coarse cubic tensor into a fine H1
   // representation, then apply the physical canonical stress moments.
   H1_TriangleElement h1(3);
   const int n = h1.GetDof();
   DenseMatrix nodal(3*n,21), projection, expected(21);
   for (int q = 0; q < n; q++)
   {
      Vector point(2);
      child.Transform(h1.GetNodes().IntPoint(q),point);
      IntegrationPoint ip;
      ip.Set2(point(0),point(1));
      coarse.SetIntPoint(&ip);
      DenseTensor shape(2,2,21);
      stress.CalcMShape(coarse,shape);
      for (int j = 0; j < 21; j++)
      {
         nodal(q,j) = shape(0,0,j);
         nodal(n+q,j) = shape(0,1,j);
         nodal(2*n+q,j) = shape(1,1,j);
      }
   }
   stress.Project(h1,fine,projection);
   Mult(projection,nodal,expected);
   expected -= transfer;
   REQUIRE(expected.MaxMaxNorm() < 1e-10);
}

TEST_CASE("Bell physical quartic refinement", "[Transfer][Bell]")
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
   BellFECollection fec;
   FiniteElementSpace coarse(&coarse_mesh,&fec), fine(&fine_mesh,&fec);
   OperatorHandle P(GENERATE(Operator::MFEM_SPARSEMAT,Operator::ANY_TYPE));
   fine.GetTransferOperator(coarse,P);
   Vector coarse_values(coarse.GetVSize()), fine_values(fine.GetVSize());
   for (int level = 0; level < 2; level++)
   {
      auto &fes = level ? fine : coarse;
      Vector values(fes.GetVSize());
      for (int v = 0; v < fes.GetMesh()->GetNV(); v++)
      {
         const real_t *p = fes.GetMesh()->GetVertex(v);
         const real_t x = p[0], y = p[1];
         const real_t jet[6] = {x*x*x*x+x*y*y,4*x*x*x+y*y,2*x*y,
                                12*x*x,2*y,2*x
                               };
         Array<int> dofs;
         fes.GetVertexDofs(v,dofs);
         for (int d = 0; d < 6; d++) { values(dofs[d]) = jet[d]; }
      }
      if (!level) { coarse_values = values; P->Mult(coarse_values,fine_values); }
      else
      {
         values -= fine_values;
         REQUIRE(values.Normlinf() < 1e-9);
      }
   }
}
