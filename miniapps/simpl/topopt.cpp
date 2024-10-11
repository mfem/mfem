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

real_t DesignDensity::ApplyVolumeProjection(GridFunction &x, bool use_entropy)
{
   real_t mu = 0.0; // constant perturbation

   // define density with perturbation
   std::function<real_t(const real_t)> density_fun;
   if (entropy && use_entropy)
   {
      // if entropy exists, then use the Bregman projection
      // assuming x is the dual variable
      density_fun = [this, &mu](const real_t psi) { return std::max(0.0, std::min(1.0, this->entropy->backward(psi + mu))); };
   }
   else
   {
      // if entropy does not exist, then use the L2 projection
      // assuming x is the primal variable
      density_fun = [&mu](const real_t rho) {return std::max(0.0, std::min(rho + mu, 1.0));};
   }
   MappedGFCoefficient density(x, density_fun);

   real_t maxval = entropy && use_entropy ? entropy->forward(1-1e-12) : 1-1e-12;
   real_t minval = entropy && use_entropy ? entropy->forward(1e-12) : 1e-12;
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

   // Check the volume constraints and determine the target volume
   real_t curr_vol = zero->ComputeL1Error(density);
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
      curr_vol = zero->ComputeL1Error(density);
      dc *= 0.5;
      mu += curr_vol < target_vol ? dc : -dc;
      if (solid_attr_id)
      {
         const_cf.constant = maxval-mu;
         ProjectCoefficient(x, const_cf, solid_attr_id);
      }
      if (void_attr_id)
      {
         const_cf.constant = minval-mu;
         ProjectCoefficient(x, const_cf, void_attr_id);
      }
   }
   if (entropy && use_entropy)
   {
      x += mu;
      if (std::isfinite(entropy->GetUpperBound()))
      {
         real_t uval = entropy->GetUpperBound();
         if (Mpi::Root()) { out << "Enforcing Upper Bound " << uval << std::endl; }
         for (real_t &val:x) { val = std::min(uval, val); }
      }
      if (std::isfinite(entropy->GetLowerBound()))
      {
         real_t lval = entropy->GetLowerBound();
         if (Mpi::Root()) { out << "Enforcing Lower Bound " << lval << std::endl; }
         for (real_t &val:x) { val = std::max(lval, val); }
      }
   }
   else
   {
      x.ProjectCoefficient(density);
   }

   return curr_vol;
}

DensityBasedTopOpt::DensityBasedTopOpt(
   DesignDensity &density, GridFunction &control_gf, GridFunction &grad_control,
   HelmholtzFilter &filter, GridFunction &filter_gf, GridFunction &grad_filter,
   ElasticityProblem &elasticity, GridFunction &state_gf)
   :density(density), control_gf(control_gf), grad_control(grad_control),
    filter(filter), filter_gf(filter_gf), grad_filter(grad_filter),
    elasticity(elasticity), state_gf(state_gf),
    obj(elasticity.HasAdjoint() ? *elasticity.GetAdjLinearForm():
        *elasticity.GetLinearForm())
{
   Array<int> empty(control_gf.FESpace()->GetMesh()->bdr_attributes.Max());
   empty = 0;
   L2projector.reset(new L2Projection(*control_gf.FESpace(), empty));
   grad_filter_cf.SetGridFunction(&grad_filter);
   L2projector->GetLinearForm()->AddDomainIntegrator(new DomainLFIntegrator(
                                                        grad_filter_cf));
   if (elasticity.HasAdjoint())
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
   current_volume = density.ApplyVolumeProjection(control_gf,
                                                  density.hasEntropy());
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


} // end of namespace mfem
