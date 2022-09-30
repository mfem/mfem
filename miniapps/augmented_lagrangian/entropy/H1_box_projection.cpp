#include "H1_box_projection.hpp"

BoxProjection::BoxProjection(ParMesh* pmesh_, int order_, Coefficient * g_cf_,
                             VectorCoefficient * grad_g_cf_, bool H1H1)
:pmesh(pmesh_), order(order_), g_cf(g_cf_), grad_g_cf(grad_g_cf_)
{
   dim = pmesh->Dimension();
   if (H1H1)
   {
      u_fec = new H1_FECollection(order+1, dim);
      p_fec = new H1_FECollection(order, dim);
   }
   else
   {
      u_fec = new H1_FECollection(order, dim);
      p_fec = new L2_FECollection(order-1, dim);
   }
   u_fes = new ParFiniteElementSpace(pmesh,u_fec);
   p_fes = new ParFiniteElementSpace(pmesh,p_fec);

   P_u = u_fes->GetProlongationMatrix();
   P_p = p_fes->GetProlongationMatrix();

   b_u = new ParLinearForm(u_fes);

   // Assemble u-p block
   a_up = new ParMixedBilinearForm(u_fes,p_fes);
   a_up->AddDomainIntegrator(new MixedScalarMassIntegrator());
   a_up->Assemble();
   a_up->Finalize();
   A_up = a_up->ParallelAssemble();
   A_pu = A_up->Transpose();

   offsets.SetSize(3); 
   toffsets.SetSize(3);
   offsets[0] = 0; offsets[1] = u_fes->GetVSize(); offsets[2] = p_fes->GetVSize();
   toffsets[0] = 0; toffsets[1] = u_fes->GetTrueVSize(); toffsets[2] = p_fes->GetTrueVSize();
   offsets.PartialSum();
   toffsets.PartialSum();
}

void BoxProjection::Update_A_uu(ParGridFunction & u_gf_)
{
   delete a_uu;
   a_uu = new ParBilinearForm(u_fes);

   ActiveSetCoefficient as_cf(u_gf_);
   SumCoefficient mass_cf(1.0, as_cf, 1.0, gamma);

   // Assemble u-u block
   ConstantCoefficient epsilon_cf(epsilon);
   a_uu->AddDomainIntegrator(new DiffusionIntegrator(epsilon_cf));
   a_uu->AddDomainIntegrator(new MassIntegrator(mass_cf));
   a_uu->Assemble();
   a_uu->Finalize();
   A_uu = a_uu->ParallelAssemble();
}

double BoxProjection::NewtonStep(const ParLinearForm & b_u_, ParGridFunction & p_kl_gf, ParGridFunction & u_kl_gf)
{
   newton_cntr++;

   // ND_FECollection fec(order,dim);
   // ParFiniteElementSpace fes(pmesh,&fec);
   // ParGridFunction grad_u_gf(&fes);
   // GradientGridFunctionCoefficient grad_u_cf(&u_kl_gf);
   // grad_u_gf.ProjectCoefficient(grad_u_cf);

   // char vishost[] = "localhost";
   // int visport = 19916;
   // socketstream u_sock(vishost, visport);
   // u_sock.precision(8);
   // u_sock << "solution\n" << *pmesh << u_kl_gf  
   //        << "window_title 'Newton u'" << flush;

   // socketstream socksock(vishost, visport);
   // socksock.precision(8);
   // socksock << "solution\n" << *pmesh << grad_u_gf 
   //        << "window_title 'Newton Grad u'" << flush;
   
   // cin.get();

   Update_A_uu(u_kl_gf);
   
   GridFunctionCoefficient p_kl_cf(&p_kl_gf);
   ParLinearForm l_u(u_fes);
   l_u.AddDomainIntegrator(new DomainLFIntegrator(p_kl_cf));
   l_u.Assemble(false);
   l_u-=b_u_;
   l_u.Neg();

   ParLinearForm l_p(p_fes);
   ExpitGridFunctionCoefficient expit_p_cf(p_kl_gf);
   expit_p_cf.SetBounds(min_val,max_val);
   l_p.AddDomainIntegrator(new DomainLFIntegrator(expit_p_cf));
   l_p.Assemble(false);

   BlockVector B(toffsets);
   
   P_u->MultTranspose(l_u,B.GetBlock(0));
   if (P_p)
   {
      P_p->MultTranspose(l_p,B.GetBlock(1));
   }
   else
   {
      B.GetBlock(1) = l_p;
   }

   ParBilinearForm a_pp(p_fes);
   dExpitdxGridFunctionCoefficient dexpit_p_cf(p_kl_gf);
   dexpit_p_cf.SetBounds(min_val,max_val);
   a_pp.AddDomainIntegrator(new MassIntegrator(dexpit_p_cf));
   a_pp.Assemble();
   a_pp.Finalize();
   HypreParMatrix *A_pp = a_pp.ParallelAssemble();

   BlockVector X(toffsets);

   Array2D<HypreParMatrix *> BlockA(2,2);
   Array2D<double> scale(2,2);
   scale(0,0) = alpha + beta;
   scale(0,1) = 1.0;
   scale(1,0) = 1.0;
   scale(1,1) = -1.0;
   BlockA(0,0) = A_uu;
   BlockA(0,1) = A_pu;
   BlockA(1,0) = A_up;
   BlockA(1,1) = A_pp;
   HypreParMatrix * Ah = HypreParMatrixFromBlocks(BlockA, &scale);
         
   MUMPSSolver mumps;
   mumps.SetPrintLevel(0);
   mumps.SetMatrixSymType(MUMPSSolver::MatType::SYMMETRIC_INDEFINITE);
   mumps.SetOperator(*Ah);
   mumps.Mult(B,X);
   delete Ah;
   delete A_pp;

   ParGridFunction u_kl_gf_previous(u_kl_gf);
   GridFunctionCoefficient u_kl_cf(&u_kl_gf_previous);
   u_kl_gf.SetFromTrueDofs(X.GetBlock(0));
   
   Vector tmp;
   if (P_p)
   {
      tmp.SetSize(p_kl_gf.Size());
      P_p->Mult(X.GetBlock(1),tmp);
   }
   else
   {
      tmp.SetDataAndSize(X.GetBlock(1).GetData(),p_kl_gf.Size());
   }
   p_kl_gf.Add(theta,tmp);
   
   return u_kl_gf.ComputeL2Error(u_kl_cf);
}


double BoxProjection::BregmanStep(ParGridFunction & u_gf_, ParGridFunction & p_gf_)
{
   bregman_cntr++;

   ParGridFunction u_gf_old(u_gf_);
   GridFunctionCoefficient p_cf(&p_gf_);
   GridFunctionCoefficient u_cf(&u_gf_old);
   GradientGridFunctionCoefficient gradu_cf(&u_gf_old);
   SumCoefficient sum_cf(*g_cf,u_cf,alpha,beta);
   VectorSumCoefficient vsum_cf(*grad_g_cf,gradu_cf,epsilon*alpha,epsilon*beta);
   ScalarVectorProductCoefficient eps_grad_g_cf(-epsilon*alpha, *grad_g_cf);

   ParLinearForm b_u_(u_fes);
   b_u_.AddDomainIntegrator(new DomainLFGradIntegrator(vsum_cf));
   b_u_.AddDomainIntegrator(new DomainLFIntegrator(sum_cf));
   b_u_.AddDomainIntegrator(new DomainLFIntegrator(p_cf));
   // b_u_.AddBoundaryIntegrator(new BoundaryNormalLFIntegrator(eps_grad_g_cf));
   b_u_.Assemble(false);
   int myid;
   MPI_Comm_rank(pmesh->GetComm(), &myid);
   for (int l = 0; l < max_newton_it; l++)
   {
      // both u and p are updated inside NewtonStep
      double update_norm = NewtonStep(b_u_,p_gf_,u_gf_);
      if (print_level > 1)
      {
         if (myid == 0)
         {
            mfem::out << "Newton step update norm = " << update_norm << endl;
         }
      }
      if (update_norm < newton_tol) 
      {
         break;
      } 
   }
   return u_gf_.ComputeL2Error(u_cf);
}

void BoxProjection::Solve()
{
   BoxFunctionCoefficient u0_cf(*g_cf,min_val, max_val);
   u_gf.SetSpace(u_fes);
   u_gf.ProjectCoefficient(u0_cf);
   int myid;
   MPI_Comm_rank(pmesh->GetComm(), &myid);
   if (print_level >= 0)
   {
      double grad_error = u_gf.ComputeGradError(grad_g_cf);
      double L2_error = u_gf.ComputeL2Error(&g_cf);
      double H1_error = sqrt(epsilon*grad_error*grad_error + L2_error*L2_error);
      if (myid == 0)
      {
         mfem::out << " | u - g |_H1      = " << grad_error << endl;
         mfem::out << "|| u - g ||_L2     = " << L2_error << endl;
         mfem::out << "|| u - g ||_H1_eps = " << H1_error << endl;
      }
   }

   // char vishost[] = "localhost";
   // int visport = 19916;
   // socketstream g_sock(vishost, visport);
   // g_sock.precision(8);
   // g_sock << "solution\n" << *pmesh << u_gf
   //             << "window_title 'Function g'" << flush;

   u_gf = 0.5*(min_val + max_val);
   LnitGridFunctionCoefficient p0_cf(u_gf);
   p0_cf.SetBounds(min_val,max_val);
   p_gf.SetSpace(p_fes);
   p_gf.ProjectCoefficient(p0_cf);

   // socketstream p0_sock(vishost, visport);
   // p0_sock.precision(8);
   // p0_sock << "solution\n" << *pmesh << p_gf
   //             << "window_title 'Function p_0'" << flush;       

   for (int i = 0; i<max_bregman_it; i++)
   {
      if (i > 1) { theta = 1.0; }
      double update_norm = BregmanStep(u_gf, p_gf);
      if (print_level > 0)
      {
         if (myid == 0)
         {
            mfem::out << "Bregman step update norm = " << update_norm << endl;
         }
      }
      if (print_level >= 0)
      {
         double grad_error = u_gf.ComputeGradError(grad_g_cf);
         double L2_error = u_gf.ComputeL2Error(&g_cf);
         double H1_error = sqrt(epsilon*grad_error*grad_error + L2_error*L2_error);
         if (myid == 0)
         {
            mfem::out << " | u - g |_H1      = " << grad_error << endl;
            mfem::out << "|| u - g ||_L2     = " << L2_error << endl;
            mfem::out << "|| u - g ||_H1_eps = " << H1_error << endl;
         }
      }
      if (update_norm/alpha < bregman_tol) break;
      // if (update_norm < bregman_tol/sqrt(bregman_cntr)) break;
      // if (bregman_cntr > 1)
      // {
      //    double tmp = (double) bregman_cntr;
      //    // alpha *= sqrt( (tmp-1.0)/tmp );
      //    alpha *= sqrt( tmp/(tmp-1.0) );
      //    // mfem::out << "alpha = " << alpha << endl;
      // }
   }

   if (print_level >= 0)
   {
      if (myid == 0)
      {
         mfem::out << "\n # of outer iterations = " << bregman_cntr << endl;
         mfem::out << " # of inner iterations = " << newton_cntr << endl;
      }
   }
}


BoxProjection::~BoxProjection()
{
   delete b_u;
   delete a_uu;
   delete a_up;
   delete u_fec;
   delete p_fec;
   delete u_fes;
   delete p_fes;
}

double LnitGridFunctionCoefficient::Eval(ElementTransformation &T,
                               const IntegrationPoint &ip)
{
   MFEM_ASSERT(u != NULL, "grid function is not set");

   double val = u->GetValue(T, ip);
   return min(max_val, max(min_val, lnit(val, a, b)));
}

double ExpitGridFunctionCoefficient::Eval(ElementTransformation &T,
                               const IntegrationPoint &ip)
{
   MFEM_ASSERT(u != NULL, "grid function is not set");

   double val = u->GetValue(T, ip);
   // return min(max_val, max(min_val, expit(val)));
   return expit(val,a,b);
}

double dExpitdxGridFunctionCoefficient::Eval(ElementTransformation &T,
                               const IntegrationPoint &ip)
{
   MFEM_ASSERT(u != NULL, "grid function is not set");

   double val = u->GetValue(T, ip);
   return max(min_val, dexpitdx(val, a, b));
}

double lnit(double x, double xmin, double xmax)
{
   double tol = 1e-12;
   double y = (x - xmin)/(xmax-xmin);
   y = min(max(tol,y),1.0-tol);
   return log(y/(1.0-y));
}

double expit(double x, double ymin, double ymax)
{
   double y;
   if (x >= 0)
   {
      y = 1.0/(1.0+exp(-x));
   }
   else
   {
      y = exp(x)/(1.0+exp(x));
   }
   return (ymax - ymin) * y + ymin;
}

double dexpitdx(double x, double ymin, double ymax)
{
   double dydx;
   if (x >= 0)
   {
      dydx = exp(-x)/pow(1.0+exp(-x),2);
   }
   else
   {
      dydx = exp(x)/pow(1.0+exp(x),2);
   }
   return (ymax - ymin) * dydx;
}
