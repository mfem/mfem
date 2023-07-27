#include "problems.hpp"


void ElasticityProblem::Init()
{
   int dim = mesh->Dimension();
   fec = new H1_FECollection(order,dim);
   fes = new FiniteElementSpace(mesh,fec,dim,Ordering::byVDIM);
   ndofs = fes->GetTrueVSize();
   mesh->SetNodalFESpace(fes);
   if (mesh->bdr_attributes.Size())
   {
      ess_bdr.SetSize(mesh->bdr_attributes.Max());
   }
   ess_bdr = 0; ess_bdr[1] = 1;
   fes->GetEssentialTrueDofs(ess_bdr,ess_tdof_list);
   // Solution GridFunction
   x.SetSpace(fes);  x = 0.0;
   // RHS
   b.Update(fes); 
   // Elasticity operator
   lambda.SetSize(mesh->attributes.Max()); lambda = 57.6923076923;
   mu.SetSize(mesh->attributes.Max()); mu = 38.4615384615;

   lambda_cf.UpdateConstants(lambda);
   mu_cf.UpdateConstants(mu);
   a = new BilinearForm(fes);
   a->AddDomainIntegrator(new ElasticityIntegrator(lambda_cf,mu_cf));
}

void ElasticityProblem::FormLinearSystem()
{
   if (!formsystem) 
   {
      formsystem = true;
      b.Assemble();
      a->Assemble();
      a->FormLinearSystem(ess_tdof_list, x, b, A, X, B);
   }
}
void ElasticityProblem::UpdateLinearSystem()
{
   if (formsystem)
   {
      b.Update();
      a->Update();
      formsystem = false;
   }
   FormLinearSystem();
}

ContactProblem::ContactProblem(ElasticityProblem * prob1_, ElasticityProblem * prob2_)
: prob1(prob1_), prob2(prob2_)
{
   // 1. Set up block system
   int dim = prob1->GetMesh()->Dimension();

   Vector delta1(dim);
   delta1 = 0.0; delta1[0] = 0.1;
   prob1->SetDisplacementDirichletData(delta1);
   prob1->FormLinearSystem();
   
   Vector delta2(dim);
   delta2 = 0.0; 
   prob2->SetDisplacementDirichletData(delta2);
   prob2->FormLinearSystem();

   int ndof1 = prob1->GetNumDofs();
   int ndof2 = prob2->GetNumDofs();

   Array<int> offsets(3);
   offsets[0] = 0;
   offsets[1] = ndof1;
   offsets[2] = ndof2;
   offsets.PartialSum();

   BlockMatrix Kb(offsets);
   Kb.SetBlock(0,0,&prob1->GetOperator());
   Kb.SetBlock(1,1,&prob2->GetOperator());
   K = Kb.CreateMonolithic();

   B = new BlockVector(offsets);
   B->GetBlock(0).Set(1.0, prob1->GetRHS());
   B->GetBlock(1).Set(1.0, prob2->GetRHS());

   // Vector displ1(ndof1), displ2(ndof2);
   // displ1 = 0.0;
   // displ2 = 0.0;

   // ComputeGapFunctionAndDerivatives(displ1, displ2);

}

void ContactProblem::ComputeContactVertrices()
{
   if (npoints>0) return;
   Mesh * mesh2 = prob2->GetMesh();
   Array<int> vert;
   for (int b=0; b<mesh2->GetNBE(); b++)
   {
      if (mesh2->GetBdrAttribute(b) == 3)
      {
         mesh2->GetBdrElementVertices(b, vert);
         for (auto v : vert)
         {
            contact_vertices.insert(v);
         }
      }
   }
   npoints = contact_vertices.size();
}

void ContactProblem::ComputeGapFunctionAndDerivatives(const Vector &displ1, 
                                                      const Vector & displ2, bool reduced)
{
   ComputeContactVertrices();

   Mesh * mesh1 = prob1->GetMesh();
   int dim = mesh1->Dimension();
   Mesh * mesh2 = prob2->GetMesh();

   int ndof1 = prob1->GetNumDofs();
   int ndof2 = prob2->GetNumDofs();
   int ndofs = ndof1 + ndof2;

   int nv1 = mesh1->GetNV();
   int nv2 = mesh2->GetNV();
   int nv = nv1 + nv2;
   // connectivity of the second mesh

   Array<int> conn2(npoints); 
   mesh2->MoveNodes(displ2);
   Vector xyz(dim * npoints);

   int cnt = 0;
   for (auto v : contact_vertices)
   {
      for (int d = 0; d<dim; d++)
      {
         xyz(cnt*dim + d) = mesh2->GetVertex(v)[d]+displ2[v*dim+d];
      }
      conn2[cnt] = v + nv1; 
      cnt++;
   }
   
   MFEM_VERIFY(cnt == npoints, "");
   gapv.SetSize(npoints*dim);

   // segment reference coordinates of the closest point
   Vector xi1(npoints*(dim-1));
   Array<int> conn1(npoints*4);
   
   mesh1->MoveNodes(displ1);
   FindPointsInMesh(*mesh1, xyz, conn1, xi1);

   DenseMatrix coordsm(npoints*4, dim);
   for (int i=0; i<npoints; i++)
   {
      for (int j=0; j<4; j++)
      {
         for (int k=0; k<dim; k++)
         {
            coordsm(i*4+j,k) = mesh1->GetVertex(conn1[i*4+j])[k]+displ1[dim*conn1[i*4+j]+k];
         }
      }
   }

   if (M)
   {
      delete M;
      for (int i = 0; i<dM.Size(); i++)
      {
         delete dM[i];
      }
      dM.SetSize(0);
   }

   int h = (reduced) ? npoints : nv;
   M = new SparseMatrix(h,ndofs);
   dM.SetSize(npoints);
   for (int i = 0; i<npoints; i++)
   {
      dM[i] = new SparseMatrix(ndofs,ndofs);
   }

   Assemble_Contact(nv, xyz, xi1, coordsm, conn2, conn1, gapv, *M, dM, reduced);
   
   M->Finalize();
   for (int i = 0; i<npoints; i++)
   {
      dM[i]->Finalize();
   }
}


double ContactProblem::E(const Vector & d) const
{
  return (0.5 * K->InnerProduct(d, d) - InnerProduct(d, *B));
}

void ContactProblem::DdE(const Vector &d, Vector &gradE) const
{
   gradE.SetSize(K->Height());
   K->Mult(d, gradE);
   gradE.Add(-1.0, *B); 
}

SparseMatrix* ContactProblem::DddE(const Vector &d)
{
  return K; 
}

void ContactProblem::g(const Vector &d, Vector &gd, bool reduced)
{
   int ndof1 = prob1->GetNumDofs();
   int ndof2 = prob2->GetNumDofs();
   double * data = d.GetData();
   Vector displ1(data,ndof1);
   Vector displ2(&data[ndof1],ndof2);
   ComputeGapFunctionAndDerivatives(displ1, displ2);
   gd = GetGapFunction();
}

SparseMatrix* ContactProblem::Ddg(const Vector &d)
{
  return GetJacobian();
}

SparseMatrix* ContactProblem::lDddg(const Vector &d, const Vector &l)
{
   return nullptr; // for now
}
