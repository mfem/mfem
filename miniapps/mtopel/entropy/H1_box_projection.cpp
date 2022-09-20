
#include "H1_box_projection.hpp"

BoxProjection::BoxProjection(ParMesh* pmesh_, int order_, Coefficient * g_cf_, VectorCoefficient * grad_g_cf_)
:pmesh(pmesh_), order(order_), g_cf(g_cf_), grad_g_cf(grad_g_cf_)
{
   dim = pmesh->Dimension();
   H1fec = new H1_FECollection(order, dim);
   L2fec = new L2_FECollection(order-1, dim);
   H1pfes = new ParFiniteElementSpace(pmesh,H1fec);
   L2pfes = new ParFiniteElementSpace(pmesh,L2fec);

   P_H1 = H1pfes->GetProlongationMatrix();
   R_H1 = H1pfes->GetRestrictionOperator();


   a_H1H1 = new ParBilinearForm(H1pfes);
   a_H1L2 = new ParMixedBilinearForm(H1pfes,L2pfes);
   b_H1 = new ParLinearForm(H1pfes);

   // H1H1 block
   a_H1H1->AddDomainIntegrator(new DiffusionIntegrator());
   a_H1H1->AddDomainIntegrator(new MassIntegrator());
   a_H1H1->Assemble();
   A_H1H1 = a_H1H1->ParallelAssemble();

   // H1L2 block
   a_H1L2->AddDomainIntegrator(new MixedScalarMassIntegrator());
   a_H1L2->Assemble();
   A_H1L2 = a_H1L2->ParallelAssemble();
   A_L2H1 = A_H1L2->Transpose();

   offsets.SetSize(3); 
   toffsets.SetSize(3);
   offsets[0] = 0; offsets[1] = H1pfes->GetVSize(); offsets[2] = L2pfes->GetVSize();
   toffsets[0] = 0; toffsets[1] = H1pfes->GetTrueVSize(); toffsets[2] = L2pfes->GetTrueVSize();
   offsets.PartialSum();
   toffsets.PartialSum();

   x.Update(offsets); x = 0.0;
   rhs.Update(offsets); rhs = 0.0;
   tx.Update(toffsets); tx = 0.0;
   trhs.Update(toffsets); trhs = 0.0;

   u_gf.MakeRef(H1pfes,x.GetBlock(0));
   psi_gf.MakeRef(L2pfes,x.GetBlock(1));
}

double BoxProjection::NewtonStep(const ParLinearForm & b_H1_, ParGridFunction & psi_kl_gf, ParGridFunction & u_kl_gf)
{
   GridFunctionCoefficient psi_kl_cf(&psi_kl_gf);
   ParLinearForm l_H1(H1pfes);
   l_H1.AddDomainIntegrator(new DomainLFIntegrator(psi_kl_cf));
   l_H1.Assemble();
   l_H1-=b_H1_;
   l_H1.Neg();

   ParLinearForm l_L2(L2pfes);
   ExpitGridFunctionCoefficient expit_psi_cf(psi_kl_gf);
   l_L2.AddDomainIntegrator(new DomainLFIntegrator(expit_psi_cf));
   l_L2.Assemble();

   BlockVector B(toffsets);
   P_H1->MultTranspose(l_H1,B.GetBlock(0));
   B.GetBlock(1).MakeRef(l_L2,0);

   ParBilinearForm a_L2L2(L2pfes);
   dExpitdxGridFunctionCoefficient dexpit_psi_cf(psi_kl_gf);
   a_L2L2.AddDomainIntegrator(new MassIntegrator(dexpit_psi_cf));
   a_L2L2.Assemble();
   HypreParMatrix *A_L2L2 = a_L2L2.ParallelAssemble();

   BlockVector X(toffsets);

   Array2D<HypreParMatrix *> BlockA(2,2);
   Array2D<double> scale(2,2);
   scale(0,0) = alpha + beta;
   scale(0,1) = 1.0;
   scale(1,0) = 1.0;
   scale(1,1) = -1.0;
   BlockA(0,0) = A_H1H1;
   BlockA(0,1) = A_H1L2;
   BlockA(1,0) = A_L2H1;
   BlockA(1,1) = A_L2L2;
   HypreParMatrix * Ah = HypreParMatrixFromBlocks(BlockA, &scale);
         
   MUMPSSolver mumps;
   mumps.SetPrintLevel(0);
   mumps.SetMatrixSymType(MUMPSSolver::MatType::UNSYMMETRIC);
   mumps.SetOperator(*Ah);
   mumps.Mult(B,X);
   delete Ah;
   delete A_L2L2;

   ParGridFunction u_kl_gf_previous(u_kl_gf);
   GridFunctionCoefficient u_kl_cf(&u_kl_gf_previous);
   u_kl_gf.SetFromTrueDofs(X.GetBlock(0));
   psi_kl_gf += X.GetBlock(1);
   
   return u_kl_gf.ComputeL2Error(u_kl_cf);
}


double BoxProjection::BregmanStep(ParGridFunction & u_gf_, ParGridFunction & psi_gf_)
{

   GridFunctionCoefficient psi_cf(&psi_gf_);
   GridFunctionCoefficient u_cf(&u_gf);
   GradientGridFunctionCoefficient gradu_cf(&u_gf);
   SumCoefficient sum_cf(*g_cf,u_cf,alpha,beta);
   VectorSumCoefficient vsum_cf(*grad_g_cf,gradu_cf,alpha,beta);

   ParLinearForm b_H1_(H1pfes);
   b_H1_.AddDomainIntegrator(new DomainLFGradIntegrator(vsum_cf));
   b_H1_.AddDomainIntegrator(new DomainLFIntegrator(sum_cf));
   b_H1_.AddDomainIntegrator(new DomainLFIntegrator(psi_cf));
   b_H1_.Assemble();

   ParGridFunction u_kl(H1pfes);

   for (int l = 0; l < max_newton_it; l++)
   {
      double update_norm = NewtonStep(b_H1_,psi_gf_,u_kl);
      if (update_norm < newton_tol) break; 
   }
}


BoxProjection::~BoxProjection()
{
   delete b_H1;
   delete a_H1H1;
   delete a_H1L2;
   delete H1fec;
   delete L2fec;
   delete H1pfes;
   delete L2pfes;
}



int main(int argc, char *argv[])
{
   MPI_Session mpi;
   int num_procs = mpi.WorldSize();
   int myid = mpi.WorldRank();


}


double LnitGridFunctionCoefficient::Eval(ElementTransformation &T,
                               const IntegrationPoint &ip)
{
   MFEM_ASSERT(u != NULL, "grid function is not set");

   double val = u->GetValue(T, ip);
   return min(max_val, max(min_val, lnit(val)));
}

double ExpitGridFunctionCoefficient::Eval(ElementTransformation &T,
                               const IntegrationPoint &ip)
{
   MFEM_ASSERT(u != NULL, "grid function is not set");

   double val = u->GetValue(T, ip);
   return min(max_val, max(min_val, expit(val)));
}

double dExpitdxGridFunctionCoefficient::Eval(ElementTransformation &T,
                               const IntegrationPoint &ip)
{
   MFEM_ASSERT(u != NULL, "grid function is not set");

   double val = u->GetValue(T, ip);
   return min(max_val, max(min_val, dexpitdx(val)));
}