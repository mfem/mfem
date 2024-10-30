#include "topopt.hpp"

namespace mfem
{

void GLVis::Append(GridFunction &gf, const char window_title[],
                   const char keys[])
{
   sockets.Append(new socketstream(hostname, port, secure));
   socketstream *socket = sockets.Last();
   if (!socket->is_open())
   {
      return;
   }
   Mesh *mesh = gf.FESpace()->GetMesh();
   gfs.Append(&gf);
   meshes.Append(mesh);
   socket->precision(8);
#ifdef MFEM_USE_MPI
   if (parallel)
   {
      *socket << "parallel " << Mpi::WorldSize() << " " << Mpi::WorldRank() << "\n";
   }
#endif
   *socket << "solution\n" << *mesh << gf;
   if (keys)
   {
      *socket << "keys " << keys << "\n";
   }
   if (window_title)
   {
      *socket << "window_title '" << window_title <<"'\n";
   }
   *socket << std::flush;
}

void GLVis::Update()
{
   for (int i=0; i<sockets.Size(); i++)
   {
      if (!sockets[i]->good() && !sockets[i]->is_open())
      {
         continue;
      }
#ifdef MFEM_USE_MPI
      if (parallel)
      {
         *sockets[i] << "parallel " << Mpi::WorldSize() << " " << Mpi::WorldRank() <<
                     "\n";
      }
#endif
      *sockets[i] << "solution\n" << *meshes[i] << *gfs[i];
      *sockets[i] << std::flush;
   }
}

void DirectionalHookesLawBdrIntegrator::AssembleFaceMatrix(
   const FiniteElement &el, const FiniteElement &dummy,
   FaceElementTransformations &Trans, DenseMatrix &elmat)
{

   real_t kw;
   int dim = el.GetDim();
   int ndof = el.GetDof();
   Vector dir(dim), nor(dim);

   shape.SetSize(ndof);
   elmat.SetSize(ndof*dim);
   elmat = 0.0;

   const IntegrationRule *ir = IntRule;
   if (ir == NULL)
   {
      int order;
      // Assuming order(u)==order(mesh)
      order = Trans.Elem1->OrderW() + 2*el.GetOrder();
      if (el.Space() == FunctionSpace::Pk)
      {
         order++;
      }
      ir = &IntRules.Get(Trans.GetGeometryType(), order);
   }

   for (int p = 0; p < ir->GetNPoints(); p++)
   {
      const IntegrationPoint &ip = ir->IntPoint(p);

      // Set the integration point in the face and the neighboring elements
      Trans.SetAllIntPoints(&ip);

      // Access the neighboring elements' integration points
      // Note: eip2 will only contain valid data if Elem2 exists
      const IntegrationPoint &eip1 = Trans.GetElement1IntPoint();

      el.CalcPhysShape(*Trans.Elem1, shape);

      direction->Eval(dir, *Trans.Elem1, eip1);

      kw = k*ip.weight;
      for (int d2=0; d2<dim; d2++)
      {
         for (int j = 0; j < ndof; j++)
         {
            for (int d1 = 0; d1 < dim; d1++)
            {
               for (int i = 0; i < ndof; i++)
               {
                  elmat(i+d1*ndof, j+d2*ndof) += kw * dir[d1] * dir[d2] * shape(i) * shape(j);
               }
            }
         }
      }
   }
}

real_t StrainEnergyDensityCoefficient::Eval(
   ElementTransformation &T, const IntegrationPoint &ip)
{
   real_t L = lambda.Eval(T, ip);
   real_t M = mu.Eval(T, ip)*0.5;

   state_gf.GetVectorGradient(T, grad);
   grad.Symmetrize();
   if (adjstate_gf) { adjstate_gf->GetVectorGradient(T, adjgrad); adjgrad.Symmetrize(); }
   else {adjgrad.UseExternalData(grad.GetData(), grad.NumCols(), grad.NumRows());}

   real_t density = L*grad.Trace()*adjgrad.Trace();
   int dim = T.GetSpaceDim();
   for (int i=0; i<dim; i++)
   {
      for (int j=0; j<dim; j++)
      {
         density += 2.0*M*grad(i,j)*adjgrad(i,j);
      }
   }
   return -der_simp_cf.Eval(T, ip)*density;
}

real_t DiffusionEnergyDensityCoefficient::Eval(
   ElementTransformation &T, const IntegrationPoint &ip)
{
   real_t K_val = K.Eval(T, ip);

   state_gf.GetGradient(T, grad);
   if (adjstate_gf) { adjstate_gf->GetGradient(T, adjgrad); }
   else {adjgrad.MakeRef(grad, 0);}

   real_t density = K_val*InnerProduct(grad, adjgrad);
   return -der_simp_cf.Eval(T, ip)*density;
}

void ProjectCoefficient(GridFunction &x, Coefficient &coeff, int attribute)
{
   int i;
   Array<int> dofs;
   Vector vals;

   DofTransformation * doftrans = NULL;

   FiniteElementSpace *fes = x.FESpace();

   for (i = 0; i < fes->GetNE(); i++)
   {
      if (fes->GetAttribute(i) != attribute)
      {
         continue;
      }

      doftrans = fes->GetElementDofs(i, dofs);
      vals.SetSize(dofs.Size());
      fes->GetFE(i)->Project(coeff, *fes->GetElementTransformation(i), vals);
      if (doftrans)
      {
         doftrans->TransformPrimal(vals);
      }
      x.SetSubVector(dofs, vals);
   }
}

DesignDensity::DesignDensity(
   FiniteElementSpace &fes_control, const real_t tot_vol,
   const real_t min_vol, const real_t max_vol,
   LegendreEntropy *entropy)
   :fes_control(fes_control), tot_vol(tot_vol),
    min_vol(min_vol), max_vol(max_vol), entropy(entropy)
{
#ifdef MFEM_USE_MPI
   ParFiniteElementSpace * pfes_control = dynamic_cast<ParFiniteElementSpace*>
                                          (&fes_control);
   if (pfes_control)
   {
      zero.reset(new ParGridFunction(pfes_control));
   }
   else
   {
      zero.reset(new GridFunction(&fes_control));
   }
#else
   zero.reset(new GridFunction(&fes_control));
#endif
   *zero = 0.0;
}

void DesignDensity::ProjectedStep(GridFunction &x, const real_t step_size,
                                  const GridFunction &grad, real_t &mu, real_t &vol)
{
   const real_t maxval = entropy->GetFiniteUpperBound();
   const real_t minval = entropy->GetFiniteLowerBound();
   const real_t x_max = GetMaxval(x);
   const real_t x_min = GetMinval(x);
   real_t beta_max = infinity();
   real_t beta_min = infinity();
   if (x_max > maxval){
      beta_max = maxval/x_max;
   }
   if (x_min < minval){
      beta_min = minval / x_min;
   }
   if (IsFinite(beta_max) || IsFinite(beta_min))
   {
      x *= std::min(beta_max, beta_min);
   }


   x.Add(-step_size, grad);

   mu = 0.0;
   const real_t abs_step_size = std::fabs(step_size);
   MappedGFCoefficient newx_cf(x, [&mu, maxval, minval,
                                        abs_step_size](const real_t psi)
   {
      return psi + mu*abs_step_size;
   });
   MaskedCoefficient masked_newx_cf(newx_cf);
   ConstantCoefficient maxval_cf(maxval);
   ConstantCoefficient minval_cf(minval);
   if (solid_attr_id) { masked_newx_cf.AddMasking(maxval_cf, solid_attr_id); }
   if (void_attr_id) { masked_newx_cf.AddMasking(minval_cf, void_attr_id); }
   CompositeCoefficient density_cf(masked_newx_cf, entropy->backward);
   vol = zero->ComputeL1Error(density_cf);
   if (vol <= max_vol && vol >= min_vol)
   {
      x.ProjectCoefficient(masked_newx_cf);
      return;
   }

   GridFunctionCoefficient grad_cf(&grad);
   real_t max_grad = zero->ComputeMaxError(grad_cf);
   real_t mu_max(max_grad), mu_min(-max_grad);
   // real_t mu_max(1e10), mu_min(-1e10);
#ifdef MFEM_USE_MPI
   const ParFiniteElementSpace *pfes = dynamic_cast<const ParFiniteElementSpace*>(grad.FESpace());
   if (pfes)
   {
      MPI_Allreduce(MPI_IN_PLACE, &mu_max, 1, MFEM_MPI_REAL_T, MPI_MAX,
                    pfes->GetComm());
      MPI_Allreduce(MPI_IN_PLACE, &mu_min, 1, MFEM_MPI_REAL_T, MPI_MIN,
                    pfes->GetComm());
   }
   // out << mu_max << ", " << mu_min << std::endl;
#endif
   // real_t mu_max(max_grad), mu_min(-max_grad);
   if (mu_max == mu_min)
   {
      mu_max = entropy->GetFiniteUpperBound();
      mu_min = entropy->GetFiniteLowerBound();
   }
   const real_t targ_vol = vol > max_vol ? max_vol : min_vol;

   for (int i=0; i< 200; i++)
   {
      mu = (mu_max+mu_min)*0.5;
      vol = zero->ComputeL1Error(density_cf);
      if (vol > targ_vol) { mu_max = mu; }
      else if (vol < targ_vol) { mu_min = mu; }
      else { break; }
      if (mu_max - mu_min < 1e-12) { break; }
   }
   x.ProjectCoefficient(masked_newx_cf);
}

real_t DesignDensity::ApplyVolumeProjection(GridFunction &x, bool use_entropy)
{
   real_t mu = 0.0; // constant perturbation

   // define density with perturbation
   real_t maxval = entropy && use_entropy ? entropy->GetFiniteUpperBound() : 1.0;
   real_t minval = entropy && use_entropy ? entropy->GetFiniteLowerBound() : 0.0;

   ConstantCoefficient const_cf(1.0);
   if (solid_attr_id)
   {
      const_cf.constant = maxval;
      ProjectCoefficient(x, const_cf, solid_attr_id);
   }
   if (void_attr_id)
   {
      const_cf.constant = minval;
      ProjectCoefficient(x, const_cf, void_attr_id);
   }

   std::unique_ptr<Coefficient> density_cf, latent_cf;
   if (entropy && use_entropy)
   {
      latent_cf.reset(new MappedGFCoefficient(
                         x, [&mu, maxval, minval](const real_t psi)
      {return std::max(minval, std::min(maxval, psi + mu));}));
      x.ProjectCoefficient(*latent_cf); // apply clipping.
      density_cf.reset(new CompositeCoefficient(*latent_cf, [this](
      const real_t x) {return entropy->backward(x);}));
   }
   else
   {
      density_cf.reset(new MappedGFCoefficient(x, [&mu](const real_t x) {return std::max(0.0, std::min(1.0, x+mu));}));
   }

   // Check the volume constraints and determine the target volume
   real_t curr_vol = zero->ComputeL1Error(*density_cf);
   real_t target_vol=-1;
   if (curr_vol > max_vol)
   {
      target_vol = max_vol;
   }
   else if (curr_vol < min_vol)
   {
      target_vol = min_vol;
   }

   // if target volume is -1, then it is already satisfied
   if (target_vol == -1) { return curr_vol; }

   // Get lower and upper bound
   // Baseline is computed by considering a constant density that has target volume
   // Then the lower/upper bounds of mu can be found by
   // subtracting max/min of the current variable from the baseline
   // This is possible because our density mapping is an increasing function
   real_t baseline = entropy && use_entropy
                     ? entropy->forward(target_vol / tot_vol)
                     : target_vol / tot_vol;
   real_t upper = baseline - x.Min();
   real_t lower = baseline - x.Max();

#ifdef MFEM_USE_MPI
   ParGridFunction *px = dynamic_cast<ParGridFunction*>(&x);
   MPI_Comm comm;
   if (px)
   {
      comm = px->ParFESpace()->GetComm();
      MPI_Allreduce(MPI_IN_PLACE, &upper, 1, MFEM_MPI_REAL_T, MPI_MAX, comm);
      MPI_Allreduce(MPI_IN_PLACE, &lower, 1, MFEM_MPI_REAL_T, MPI_MIN, comm);
   }
#endif
   if (upper == lower) {x = baseline; return target_vol;}

   // bisection
   real_t target_accuracy = 1e-12;
   real_t dc = (upper - lower)*0.5; // increament
   mu = (upper + lower) * 0.5; // initial choice
   while (dc > target_accuracy)
   {
      curr_vol = zero->ComputeL1Error(*density_cf);
      dc *= 0.5;
      mu += curr_vol < target_vol ? dc : -dc;
   }
   if (entropy && use_entropy)
   {
      x.ProjectCoefficient(*latent_cf);
   }
   else
   {
      x.ProjectCoefficient(*density_cf);
   }

   return curr_vol;
}
real_t DesignDensity::ComputeVolume(const GridFunction &x)
{
   MappedGFCoefficient density_cf=entropy->GetBackwardCoeff(x);
   return zero->ComputeL1Error(density_cf);
}

DensityBasedTopOpt::DensityBasedTopOpt(
   DesignDensity &density, GridFunction &control_gf, GridFunction &grad_control,
   HelmholtzFilter &filter, GridFunction &filter_gf, GridFunction &grad_filter,
   EllipticProblem &state_eq, GridFunction &state_gf,
   bool enforce_volume_constraint)
   :density(density), control_gf(control_gf), grad_control(grad_control),
    filter(filter), filter_gf(filter_gf), grad_filter(grad_filter),
    elasticity(state_eq), state_gf(state_gf),
    obj(state_eq.HasAdjoint() ? *state_eq.GetAdjLinearForm():
        *state_eq.GetLinearForm()),
    enforce_volume_constraint(enforce_volume_constraint)
{
   Array<int> empty(control_gf.FESpace()->GetMesh()->bdr_attributes.Max());
   empty = 0;
   L2projector.reset(new L2Projection(*control_gf.FESpace(), empty));
   grad_filter_cf.SetGridFunction(&grad_filter);
   L2projector->GetLinearForm()->AddDomainIntegrator(new DomainLFIntegrator(
                                                        grad_filter_cf));
   if (state_eq.HasAdjoint())
   {
#ifdef MFEM_USE_MPI
      ParFiniteElementSpace *pfes = dynamic_cast<ParFiniteElementSpace*>
                                    (state_gf.FESpace());
      if (pfes) {adj_state_gf.reset(new ParGridFunction(pfes));}
      else {adj_state_gf.reset(new GridFunction(state_gf.FESpace()));}
#else
      adj_state_gf.reset(new GridFunction(state_gf.FESpace()));
#endif
      *adj_state_gf = 0.0;
   }
}

real_t DensityBasedTopOpt::Eval()
{
   if (enforce_volume_constraint)
   {
      current_volume = density.ApplyVolumeProjection(control_gf,
                                                     density.hasEntropy());
   }
   else
   {
      current_volume = density.ComputeVolume(control_gf);
   }
   filter.Solve(filter_gf);
   elasticity.Solve(state_gf);
   obj.Assemble();
   if (elasticity.IsParallel())
   {
#ifdef MFEM_USE_MPI
      objval = InnerProduct(elasticity.GetComm(), obj, state_gf);
#endif
   }
   else
   {
      objval = InnerProduct(obj, state_gf);
   }
   return objval;
}

void DensityBasedTopOpt::UpdateGradient()
{
   if (elasticity.HasAdjoint())
   {
      // Since the elasticity coefficient is the same, reuse the solver
      elasticity.SolveAdjoint(*adj_state_gf, true);
   }

   filter.SolveAdjoint(grad_filter);
   L2projector->Solve(grad_control);
}

int sign(const real_t x) { return x > 0 ? 1 : x < 0 ? -1 : 0; }
real_t ComputeKKT(const ParGridFunction &control_gf,
                  const ParGridFunction &grad, LegendreEntropy &entropy,
                  const int solid_attr, const int void_attr,
                  const real_t min_vol, const real_t max_vol, const real_t cur_vol,
                  const ParGridFunction &one_gf, const ParGridFunction &zero_gf,
                  const ParGridFunction &dv,
                  ParGridFunction &kkt, real_t &dual_V)
{
   MPI_Comm comm = control_gf.ParFESpace()->GetComm();
   ParGridFunction &dual_B = kkt;
   MappedPairedGFCoefficient signmismatch(control_gf, grad, [&dual_V,
                                                             &entropy](const real_t psi, const real_t g)
   {
      return sign(entropy.backward(psi)-0.5)!=sign(-g+dual_V);
   });
   ConstantCoefficient zero_cf(0.0);
   MaskedCoefficient masked_signmismatch(signmismatch);
   if (solid_attr) { masked_signmismatch.AddMasking(zero_cf, solid_attr); }
   if (void_attr) { masked_signmismatch.AddMasking(zero_cf, void_attr); }
   GridFunctionCoefficient grad_cf(&grad);
   MappedGFCoefficient shifted_grad_cf(grad, [&dual_V](const real_t x) {return x-dual_V;});
   ProductCoefficient mismatch_grad(shifted_grad_cf, masked_signmismatch);
   const real_t tot_vol = InnerProduct(comm, one_gf, dv);
   const real_t vol_res = 2*cur_vol - min_vol - max_vol;
   const real_t avg_grad = InnerProduct(comm, grad, dv)/tot_vol;

   dual_V = avg_grad;
   real_t upper_bound = infinity();
   real_t lower_bound = -infinity();
   real_t old_dual_V = infinity();
   kkt.ProjectCoefficient(mismatch_grad);
   real_t best_mismatch = zero_gf.ComputeL1Error(mismatch_grad);
   real_t best_guess = dual_V;
   real_t cur_mismatch;
   for (int i=0; i< 100; i++)
   {
      old_dual_V = dual_V;
      kkt.ProjectCoefficient(mismatch_grad);
      real_t delta_dual = InnerProduct(comm, kkt, dv);
      dual_V += delta_dual / tot_vol;
      kkt.ProjectCoefficient(mismatch_grad);
      cur_mismatch = zero_gf.ComputeL1Error(mismatch_grad);
      // if (Mpi::Root())
      // {
      //    out << dual_V << ", " << cur_mismatch << std::endl;
      // }
      if (cur_mismatch < best_mismatch)
      {
         best_guess = dual_V;
         best_mismatch = cur_mismatch;
      }
      if (std::fabs(old_dual_V - dual_V) < 1e-12)
      {
         break;
      }
   }
   dual_V = best_guess;
   zero_gf.ComputeElementMaxErrors(mismatch_grad, kkt);
   real_t grad_res = zero_gf.ComputeL1Error(mismatch_grad);
   if (Mpi::Root()) { out << dual_V << ", " << cur_mismatch << ", " << best_mismatch << ", " << dual_V << std::endl; }
   return grad_res + std::max(0.0, dual_V*vol_res / tot_vol);
}


} // end of namespace mfem
