#include "maxwell_utils.hpp"

real_t AzimuthalECoefficient::Eval(ElementTransformation &T,
                                   const IntegrationPoint &ip)
{
   Vector X, E;
   vgf->GetVectorValue(T,ip,E);
   T.Transform(ip, X);
   real_t x = X(0);
   real_t y = X(1);
   real_t r = sqrt(x*x + y*y);

   real_t val = -x*E[1] + y*E[0];
   return val/r;
}

real_t ParallelECoefficient::Eval(ElementTransformation &T,
                                  const IntegrationPoint &ip)
{
   Vector X, E;
   vgf->GetVectorValue(T,ip,E);
   T.Transform(ip, X);
   Vector b;
   ComputeB(X, b);
   return E*b;
}

EpsilonMatrixCoefficient::EpsilonMatrixCoefficient(const char * filename,
                                                   Mesh * mesh_, ParMesh * pmesh_,
                                                   real_t scale)
   : MatrixArrayCoefficient(mesh_->Dimension()), mesh(mesh_), pmesh(pmesh_),
     dim(mesh->Dimension())
{
   std::filebuf fb;
   fb.open(filename,std::ios::in);
   std::istream is(&fb);
   vgf = new GridFunction(mesh,is);
   fb.close();
   FiniteElementSpace * vfes = vgf->FESpace();
   int vdim = vfes->GetVDim();
   const FiniteElementCollection * fec = vfes->FEColl();
   FiniteElementSpace * fes = new FiniteElementSpace(mesh, fec);
   int * partitioning = mesh->GeneratePartitioning(num_procs);
   double *data = vgf->GetData();
   GridFunction gf;
   pgfs.SetSize(vdim);
   gf_cfs.SetSize(vdim);
   sdim = sqrt(vdim);
   for (int i = 0; i<sdim; i++)
   {
      for (int j = 0; j<sdim; j++)
      {
         int k = i*sdim+j;
         gf.MakeRef(fes,&data[k*fes->GetVSize()]);
         pgfs[k] = new ParGridFunction(pmesh,&gf,partitioning);
         (*pgfs[k])*=scale;
         gf_cfs[k] = new GridFunctionCoefficient(pgfs[k]);
         if (i<dim && j<dim)
         {
            Set(i,j,gf_cfs[k], true);
         }
      }
   }
}

void EpsilonMatrixCoefficient::VisualizeMatrixCoefficient()
{
   Array<socketstream *> sol_sock(pgfs.Size());
   for (int k = 0; k<pgfs.Size(); k++)
   {
      if (Mpi::Root()) { mfem::out << "Visualizing component " << k << endl; }
      char vishost[] = "localhost";
      int visport = 19916;
      sol_sock[k] = new socketstream(vishost, visport);
      sol_sock[k]->precision(8);
      *sol_sock[k] << "parallel " << num_procs << " " << myid << "\n";
      int i = k/sdim;
      int j = k%sdim;
      *sol_sock[k] << "solution\n" << *pmesh << *pgfs[k]
                   << "window_title 'Epsilon Matrix Coefficient Component (" << i << "," << j <<
                   ")'" << flush;
   }
}

void EpsilonMatrixCoefficient::Update()
{
   pgfs[0]->ParFESpace()->Update();
   for (int k = 0; k<pgfs.Size(); k++)
   {
      pgfs[k]->Update();
   }
}

EpsilonMatrixCoefficient::~EpsilonMatrixCoefficient()
{
   for (int i = 0; i<pgfs.Size(); i++)
   {
      delete pgfs[i];
   }
   pgfs.DeleteAll();
}

DielectricTensorComponentCoefficient::DielectricTensorComponentCoefficient(
   real_t delta_, real_t a0_, real_t a1_,
   int row_, int col_,
   bool use_imag_)
   : delta(delta_), a0(a0_), a1(a1_), row(row_), col(col_), use_imag(use_imag_) { }


real_t DielectricTensorComponentCoefficient::Eval(ElementTransformation &T,
                                                  const IntegrationPoint &ip)
{
   Vector x;
   T.Transform(ip, x);
   return use_imag ? ComputeImagPart(x) : ComputeRealPart(x);
}

real_t DielectricTensorComponentCoefficient::ComputeRealPart(const Vector &x)
{
   Vector b;
   ComputeB(x, b);

   real_t r = std::sqrt(x(0)*x(0) + x(1)*x(1));
   real_t S = 1.0;
   real_t P = a0 + a1 * (r - 0.9);

   real_t bb_ij = b(row) * b(col);
   return S * (row == col) + (P - S) * bb_ij;
}

real_t DielectricTensorComponentCoefficient::ComputeImagPart(const Vector &x)
{
   return (row == col) ? delta : 0.0;
}

void VisualizeMatrixArrayCoefficient(MatrixArrayCoefficient &mc, ParMesh *pmesh,
                                     int order, bool paraview, const char *name)
{
   MFEM_VERIFY(pmesh != nullptr, "ParMesh pointer must not be null.");
   int dim = mc.GetVDim();

   mfem::out << "Visualizing matrix coefficient with dimension: " << dim << endl;
   mfem::out << "order = " << order << endl;
   mfem::out << "pmesh dimension = " << pmesh->Dimension() << endl;

   auto fec = new H1_FECollection(order, pmesh->Dimension());
   auto pfes = new ParFiniteElementSpace(pmesh, fec);

   Array<ParGridFunction *> pgfs(dim * dim);

   Array<GridFunctionCoefficient *> gf_cfs(dim * dim);
   ParaViewDataCollection * pvdc = nullptr;
   std::ostringstream label;
   if (name)
   {
      label << name;
   }
   else
   {
      label << "eps";
   }
   if (paraview)
   {
      pvdc = new ParaViewDataCollection(label.str(), pmesh);
      pvdc->SetPrefixPath("ParaView");
      pvdc->SetLevelsOfDetail(order);
      pvdc->SetCycle(0);
      pvdc->SetDataFormat(VTKFormat::BINARY);
   }
   for (int i = 0; i < dim; ++i)
   {
      for (int j = 0; j < dim; ++j)
      {
         Coefficient *c_ij = mc.GetCoeff(i, j);
         if (!c_ij) { continue; }

         pgfs[i*dim + j] = new ParGridFunction(pfes);
         *pgfs[i*dim + j] = 0.0;
         pgfs[i*dim + j]->ProjectCoefficient(*c_ij);

         mfem::out << "Visualizing component (" << i << "," << j << ")" << endl;
         // mfem::out << "GridFunction size: " << gf->Size() << endl;
         // gf->Print(mfem::out);

         char vishost[] = "localhost";
         int  visport   = 19916;
         socketstream sol_sock(vishost, visport);

         sol_sock << "parallel " << pmesh->GetNRanks() << " " << pmesh->GetMyRank() <<
                  "\n";
         sol_sock << "solution\n" << * pmesh <<  *pgfs[i*dim + j];
         sol_sock << "window_title '" << label.str() <<"_" << i << j << "'\n";
         sol_sock << flush;

         if (paraview)
         {
            pvdc->RegisterField(label.str() + std::to_string(i) + std::to_string(j),
                                pgfs[i*dim + j]);
         }
      }
   }
   if (paraview)
   {
      pvdc->Save();
      delete pvdc;
      for (int i = 0; i < dim * dim; ++i)
      {
         delete pgfs[i];
      }
   }
   delete pfes;
   delete fec;
}

void ComputeB(const Vector &x, Vector &b)
{
   real_t x0 = x(0), x1 = x(1);
   real_t r = std::sqrt(x0 * x0 + x1 * x1);
   int dim = x.Size();
   b.SetSize(dim); b = 0.0;
   b(0) = -x1 / r;
   b(1) =  x0 / r;
   if (dim == 3) { b(2) = 0.0; }
}

void DirectionalVectorDiffusionIntegrator::AssembleElementMatrix(
   const FiniteElement &el,
   ElementTransformation &Trans,
   DenseMatrix &elmat)
{
   int dof = el.GetDof();
   int dim = el.GetDim();
   int vdim = Trans.GetSpaceDim();

   elmat.SetSize(dof * vdim, dof * vdim);
   elmat = 0.0;

   const IntegrationRule *ir = IntRule;
   if (ir == NULL)
   {
      int order = 2 * el.GetOrder(); // Integration order
      ir = &IntRules.Get(el.GetGeomType(), order);
   }

   // Get shape functions and their derivatives
   DenseMatrix dshape(dof, dim);
   Vector vec(dim);

   for (int k = 0; k < ir->GetNPoints(); k++)
   {
      const IntegrationPoint &ip = ir->IntPoint(k);
      Trans.SetIntPoint(&ip);
      double w = ip.weight * Trans.Weight();
      VQ->Eval(vec, Trans, ip);
      el.CalcPhysDShape(Trans, dshape);

      // Compute (vq·∇)φ for each basis function
      Vector vq_grad_phi(dof); vq_grad_phi = 0.0;
      for (int j = 0; j < dof; j++)
      {
         for (int d = 0; d < dim; d++)
         {
            vq_grad_phi(j) += vec(d) * dshape(j, d);
         }
      }

      for (int comp = 0; comp < vdim; comp++)
      {
         int offset = comp * dof;
         for (int j = 0; j < dof; j++)
         {
            int jj = j + offset;
            for (int i = 0; i < dof; i++)
            {
               int ii = i + offset;
               elmat(jj, ii) += w * vq_grad_phi(j)
                                * vq_grad_phi(i);
            }
         }
      }
   }
}

void DirectionalVectorDiffusionIntegrator::AssembleElementMatrix2(
   const FiniteElement &trial_fe, const FiniteElement &test_fe,
   ElementTransformation &Trans,
   DenseMatrix &elmat)
{
   int trial_dof = trial_fe.GetDof();
   int test_dof = test_fe.GetDof();
   int dim = trial_fe.GetDim();
   int vdim = Trans.GetSpaceDim();

   elmat.SetSize(test_dof * vdim, trial_dof * vdim);
   elmat = 0.0;

   const IntegrationRule *ir = IntRule;
   if (ir == NULL)
   {
      int order = trial_fe.GetOrder() + test_fe.GetOrder() + Trans.OrderW() - 1;
      ir = &IntRules.Get(trial_fe.GetGeomType(), order);
   }

   Vector vec(dim);
   DenseMatrix trial_dshape(trial_dof, dim);
   DenseMatrix test_dshape(test_dof, dim);

   for (int k = 0; k < ir->GetNPoints(); k++)
   {
      const IntegrationPoint &ip = ir->IntPoint(k);
      Trans.SetIntPoint(&ip);
      double w = ip.weight * Trans.Weight();
      VQ->Eval(vec, Trans, ip);
      trial_fe.CalcPhysDShape(Trans, trial_dshape);
      test_fe.CalcPhysDShape(Trans, test_dshape);

      // Compute (vq·∇)φ for each basis function
      Vector vq_grad_phi_trial(trial_dof); vq_grad_phi_trial = 0.0;
      for (int j = 0; j < trial_dof; j++)
      {
         for (int d = 0; d < dim; d++)
         {
            vq_grad_phi_trial(j) += vec(d) * trial_dshape(j, d);
         }
      }

      Vector vq_grad_phi_test(test_dof); vq_grad_phi_test = 0.0;
      for (int j = 0; j < test_dof; j++)
      {
         for (int d = 0; d < dim; d++)
         {
            vq_grad_phi_test(j) += vec(d) * test_dshape(j, d);
         }
      }

      for (int trial_comp = 0; trial_comp < vdim; trial_comp++)
      {
         int offset_trial = trial_comp * trial_dof;
         int offset_test = trial_comp * test_dof;
         for (int j = 0; j < test_dof; j++)
         {
            int jj = j + offset_test;
            for (int i = 0; i < trial_dof; i++)
            {
               int ii = i + offset_trial;
               elmat(jj, ii) += w * vq_grad_phi_test(j) * vq_grad_phi_trial(i);
            }
         }
      }
   }
}



void DirectionalVectorGradientIntegrator::AssembleElementMatrix(
   const FiniteElement &el,
   ElementTransformation &Trans,
   DenseMatrix &elmat)
{
   int dof = el.GetDof();
   int dim = el.GetDim();
   int vdim = Trans.GetSpaceDim();

   elmat.SetSize(dof * vdim, dof * vdim);
   elmat = 0.0;

   const IntegrationRule *ir = IntRule;
   if (ir == NULL)
   {
      int order = 2 * el.GetOrder(); // Integration order
      ir = &IntRules.Get(el.GetGeomType(), order);
   }

   // Get shape functions and their derivatives
   DenseMatrix dshape(dof, dim);
   Vector shape(dof);
   Vector vec(dim);

   for (int k = 0; k < ir->GetNPoints(); k++)
   {
      const IntegrationPoint &ip = ir->IntPoint(k);
      Trans.SetIntPoint(&ip);
      double w = ip.weight * Trans.Weight();
      VQ->Eval(vec, Trans, ip);

      // gradient of trial (u) in physical space
      el.CalcPhysDShape(Trans, dshape);

      // value of test (v) in physical space
      el.CalcPhysShape(Trans, shape);

      // (vq·∇)φ_i on the trial side
      Vector vq_grad_phi(dof); vq_grad_phi = 0.0;
      for (int j = 0; j < dof; j++)
      {
         for (int d = 0; d < dim; d++)
         {
            vq_grad_phi(j) += vec(d) * dshape(j, d);
         }
      }

      for (int comp = 0; comp < vdim; comp++)
      {
         int offset = comp * dof;
         for (int j = 0; j < dof; j++)
         {
            int jj = j + offset;
            for (int i = 0; i < dof; i++)
            {
               int ii = i + offset;
               elmat(jj, ii) += w * shape(jj) * vq_grad_phi(i);
            }
         }
      }
   }
}

void DirectionalVectorGradientIntegrator::AssembleElementMatrix2(
   const FiniteElement &trial_fe, const FiniteElement &test_fe,
   ElementTransformation &Trans,
   DenseMatrix &elmat)
{
   int trial_dof = trial_fe.GetDof();
   int test_dof = test_fe.GetDof();
   int dim = trial_fe.GetDim();
   int vdim = Trans.GetSpaceDim();

   elmat.SetSize(test_dof * vdim, trial_dof * vdim);
   elmat = 0.0;

   const IntegrationRule *ir = IntRule;
   if (ir == NULL)
   {
      int order = trial_fe.GetOrder() + test_fe.GetOrder() + Trans.OrderW();
      ir = &IntRules.Get(trial_fe.GetGeomType(), order);
   }

   Vector vec(dim);
   DenseMatrix trial_dshape(trial_dof, dim);
   Vector test_shape(test_dof);

   for (int k = 0; k < ir->GetNPoints(); k++)
   {
      const IntegrationPoint &ip = ir->IntPoint(k);
      Trans.SetIntPoint(&ip);
      double w = ip.weight * Trans.Weight();
      VQ->Eval(vec, Trans, ip);
      // gradient of trial (u) in physical space
      trial_fe.CalcPhysDShape(Trans, trial_dshape);
      // value of test (v) in physical space
      test_fe.CalcPhysShape(Trans, test_shape);

      // (vq·∇)φ_i on the trial side
      Vector vq_grad_phi_trial(trial_dof);
      vq_grad_phi_trial = 0.0;
      for (int i = 0; i < trial_dof; i++)
      {
         for (int d = 0; d < dim; d++)
         {
            vq_grad_phi_trial(i) += vec(d) * trial_dshape(i, d);
         }
      }

      for (int comp = 0; comp < vdim; comp++)
      {
         int offset_trial = comp * trial_dof;
         int offset_test = comp * test_dof;
         for (int j = 0; j < test_dof; j++)
         {
            int jj = j + offset_test;
            for (int i = 0; i < trial_dof; i++)
            {
               int ii = i + offset_trial;
               elmat(jj, ii) += w * test_shape(j) * vq_grad_phi_trial(i);
            }
         }
      }
   }
}