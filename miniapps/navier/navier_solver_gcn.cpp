#include "navier_solver_gcn.hpp"
#include "../../general/forall.hpp"
#include <fstream>
#include <iomanip>

namespace mfem {


NavierSolverGCN::NavierSolverGCN(ParMesh* mesh, int order_, std::shared_ptr<Coefficient> visc_):
    pmesh(mesh), order(order), 
    thet1(real_t(0.5)), thet2(real_t(0.5)),thet3(real_t(0.5)),thet4(real_t(0.5))
{

   vfec.reset(new H1_FECollection(order, pmesh->Dimension()));
   pfec.reset(new H1_FECollection(order));
   vfes.reset(new ParFiniteElementSpace(pmesh, vfec.get(), pmesh->Dimension()));
   pfes.reset(new ParFiniteElementSpace(pmesh, pfec.get()));

   int vfes_truevsize = vfes->GetTrueVSize();
   int pfes_truevsize = pfes->GetTrueVSize();

   //velocity
   cvel.reset(new ParGridFunction(vfes.get())); *cvel=real_t(0.0);
   nvel.reset(new ParGridFunction(vfes.get())); *nvel=real_t(0.0);
   pvel.reset(new ParGridFunction(vfes.get())); *pvel=real_t(0.0);
   //pressure
   ppres.reset(new ParGridFunction(pfes.get())); *ppres=real_t(0.0);
   npres.reset(new ParGridFunction(pfes.get())); *npres=real_t(0.0);
   cpres.reset(new ParGridFunction(pfes.get())); *cpres=real_t(0.0);


   nvelc.SetGridFunction(nvel.get());
   pvelc.SetGridFunction(pvel.get());
   cvelc.SetGridFunction(cvel.get());

   ppresc.SetGridFunction(ppres.get());   
   npresc.SetGridFunction(npres.get());
   cpresc.SetGridFunction(cpres.get());

   brink.reset();
   if (visc_ != nullptr)
   {
      visc = visc_;
   }
   else
   {
      visc.reset(new ConstantCoefficient(1.0));
   }

   onecoeff.constant = 1.0;
   zerocoef.constant = 0.0;

   block_true_offsets.SetSize(3);
   block_true_offsets[0] = 0;
   block_true_offsets[1] = vfes->TrueVSize();
   block_true_offsets[2] = pfes->TrueVSize();
   block_true_offsets.PartialSum();

}

NavierSolverGCN::~NavierSolverGCN()
{

}


void NavierSolverGCN::SetEssTDofs(real_t t, ParGridFunction& pgf, Array<int>& ess_dofs)
{
   // Set the essential boundary conditions
   ess_dofs.DeleteAll();

   Array<int> ess_tdofv_temp;

   for(auto it=vel_bcs.begin(); it!=vel_bcs.end(); ++it)
   {
      int attr = it->first;
      std::shared_ptr<VectorCoefficient> coeff = it->second;
      coeff->SetTime(t);
      Array<int> ess_bdr(pmesh->bdr_attributes.Max());
      ess_bdr=0;
      ess_bdr[attr-1] = 1;
      ess_tdofv_temp.DeleteAll();
      vfes->GetEssentialTrueDofs(ess_bdr,ess_tdofv_temp);
      ess_dofs.Append(ess_tdofv_temp);

      pgf.ProjectBdrCoefficient(*coeff,ess_bdr);
   }
}  

void NavierSolverGCN::SetEssTDofs(real_t t, ParGridFunction& pgf)
{

   for(auto it=vel_bcs.begin(); it!=vel_bcs.end(); ++it)
   {
      int attr = it->first;
      std::shared_ptr<VectorCoefficient> coeff = it->second;
      coeff->SetTime(t);
      Array<int> ess_bdr(pmesh->bdr_attributes.Max());
      ess_bdr=0;
      ess_bdr[attr-1] = 1;

      pgf.ProjectBdrCoefficient(*coeff,ess_bdr);
   }
}  

void NavierSolverGCN::SetEssTDofs(Array<int>& ess_dofs)
{
   // Set the essential boundary conditions
   ess_dofs.DeleteAll();

   Array<int> ess_tdofv_temp;

   for(auto it=vel_bcs.begin(); it!=vel_bcs.end(); ++it)
   {
      int attr = it->first;
      Array<int> ess_bdr(pmesh->bdr_attributes.Max());
      ess_bdr=0;
      ess_bdr[attr-1] = 1;
      ess_tdofv_temp.DeleteAll();
      vfes->GetEssentialTrueDofs(ess_bdr,ess_tdofv_temp);
      ess_dofs.Append(ess_tdofv_temp);
   }
}

void NavierSolverGCN::SetupOperator(real_t t, real_t dt)
{

   // Set up boundary conditions
   SetEssTDofs(t+dt, *nvel, ess_tdofv);

   // Set up the velocity and pressure coefficients
   nvelc.SetGridFunction(nvel.get());
   pvelc.SetGridFunction(pvel.get());
   cvelc.SetGridFunction(cvel.get());
   ppresc.SetGridFunction(ppres.get());
   npresc.SetGridFunction(npres.get());


   
   // Set up the bilinear form for A11
   A11.reset(new ParBilinearForm(vfes.get()));

   visc->SetTime(t+dt);
   nvisc.reset(new ProductCoefficient(dt*thet1, *visc));

   A11->AddDomainIntegrator(new VectorMassIntegrator(onecoeff));
   if(brink != nullptr)
   {
      brink->SetTime(t+dt);
      nbrink.reset(new ProductCoefficient(dt*thet1, *brink));
      A11->AddDomainIntegrator(new VectorMassIntegrator(*nbrink));
   }

   A11->AddDomainIntegrator(new VectorConvectionIntegrator(nvelc, dt*thet1));
   A11->AddDomainIntegrator(new ElasticityIntegrator(zerocoef,*nvisc));

   A11->Update();   
   A11->Assemble();
   A11->Finalize();
   A11->FormSystemMatrix(ess_tdofv, A11H);

   icoeff.constant = dt*thet1;

   if(A21==nullptr)
   {
      A12.reset(new ParMixedBilinearForm(vfes.get(), pfes.get()));  
      A21.reset(new ParMixedBilinearForm(pfes.get(), vfes.get()));
      A12->AddDomainIntegrator(new VectorDivergenceIntegrator());
      A21->AddDomainIntegrator(new GradientIntegrator());

      A21->Update();
      A12->Assemble();
      A12->Finalize();
      A12->FormRectangularSystemMatrix(ess_tdofv, ess_tdofp, A12H);

      A21->Update();
      A21->Assemble();  
      A21->Finalize();
      A21->FormRectangularSystemMatrix(ess_tdofp, ess_tdofv, A21H);
   }


   //set the block operator

   AB.reset(new BlockOperator(block_true_offsets));
   AB->SetBlock(0,0,A11H.Ptr());
   AB->SetBlock(0,1,A12H.Ptr(),dt*thet1);
   AB->SetBlock(1,0,A21H.Ptr(),dt*thet1);

   //set the preconditioner
   

}

void NavierSolverGCN::SetupRHS(real_t t, real_t dt)
{

}

void NavierSolverGCN::Step(real_t &time, real_t dt, int cur_step, bool provisional)
{

}

//VectorConvectionIntegrator

void VectorConvectionIntegrator::AssembleElementMatrix(
   const FiniteElement &el, ElementTransformation &Trans, DenseMatrix &elmat)
{

    int nd = el.GetDof();
    dim = el.GetDim();

 #ifdef MFEM_THREAD_SAFE
    DenseMatrix dshape, adjJ, Q_ir;
    Vector shape, vec2, BdFidxT;
 #endif
    elmat.SetSize(nd*dim);
    dshape.SetSize(nd,dim);
    adjJ.SetSize(dim);
    shape.SetSize(nd);
    vec2.SetSize(dim);
    BdFidxT.SetSize(nd);
    partelmat.SetSize(nd);

    Vector vec1;


    const IntegrationRule *ir = GetIntegrationRule(el, Trans);
    if (ir == NULL)
    {
       int order = Trans.OrderGrad(&el) + Trans.Order() + el.GetOrder();
       ir = &IntRules.Get(el.GetGeomType(), order);
    }

    Q->Eval(Q_ir, Trans, *ir);

    elmat = 0.0;
    partelmat=0.0;
    for (int i = 0; i < ir->GetNPoints(); i++)
    {
       const IntegrationPoint &ip = ir->IntPoint(i);
       el.CalcDShape(ip, dshape);
       el.CalcShape(ip, shape);

       Trans.SetIntPoint(&ip);
       CalcAdjugate(Trans.Jacobian(), adjJ);
       Q_ir.GetColumnReference(i, vec1);
       vec1 *= alpha * ip.weight;

       adjJ.Mult(vec1, vec2);
       dshape.Mult(vec2, BdFidxT);

       AddMultVWt(shape, BdFidxT, partelmat);
    }

    for (int k = 0; k < dim; k++)
    {
       elmat.AddMatrix(partelmat, nd*k, nd*k);
    }

}

const IntegrationRule &VectorConvectionIntegrator::GetRule(
   const FiniteElement &trial_fe, const FiniteElement &test_fe,
   const ElementTransformation &Trans)
{
   int order = Trans.OrderGrad(&trial_fe) + Trans.Order() + test_fe.GetOrder();

   return IntRules.Get(trial_fe.GetGeomType(), order);
}

const IntegrationRule &VectorConvectionIntegrator::GetRule(
   const FiniteElement &el, const ElementTransformation &Trans)
{
   return GetRule(el,el,Trans);
}

}//end namespace mfem
