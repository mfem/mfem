//                       MFEM Example 41 - Parallel Version
//
// Compile with: make ex41p
//
// Sample runs:  mpirun -np 4 ./examples/ex41p -ref 1 -kv 0.001 -o 4
//               mpirun -np 4 ./examples/ex41p -ref 3 -kv 0.001
//
// Description:  This example demonstrates the technique of adaptive mesh
//               refinement using dual-weighted residuals in the context of a
//               steady state Navier-Stokes benchmark problem. The problem setup
//               is the flow around cylinder with a laminar flow field at
//               Reynolds number 20.

#include "mfem.hpp"

using namespace std;
using namespace mfem;

template <typename T>
vector<size_t> sort_indexes(const vector<T> &v)
{
   vector<size_t> idx(v.size());
   iota(idx.begin(), idx.end(), 0);

   sort(idx.begin(), idx.end(),
   [&v](size_t i1, size_t i2) {return v[i1] < v[i2];});

   return idx;
}


/**
 * @brief Compute the drag and lift forces F = (f_d, f_l)
 *        on marked boundary attributes
 *
 *        F = ∫ σ ⋅ n̂ dΓ
 *        with σ = -pI + ν∇u
 *
 * @param u_gf velocity grid function
 * @param p_gf pressure grid function
 * @param kinematic_viscosity viscosity coefficient
 * @param marker boundary attributes over which to compute
 *               lift and drag forces
 * @param ir_face integration rule for surface integrals
 *
 * @return std::tuple<real_t, real_t> tuple of lift and drag
 */
std::tuple<real_t, real_t> DragLift(
   GridFunction &u_gf,
   GridFunction &p_gf,
   real_t kinematic_viscosity,
   Array<int> &marker,
   IntegrationRule &ir_face)
{
   const auto &u_fes = *u_gf.FESpace();
   const auto &mesh = *u_fes.GetMesh();
   const auto dim = mesh.Dimension();

   Vector draglift(2);
   draglift = 0.0;
   Vector nor(dim);
   DenseMatrix dudx, A(dim, dim);

   for (int be = 0; be < u_fes.GetNBE(); be++)
   {
      const int bdr_el_attr = mesh.GetBdrAttribute(be);
      if (marker[bdr_el_attr-1] == 0)
      {
         continue;
      }

      const FiniteElement &bdr_el = *u_fes.GetBE(be);
      ElementTransformation &Tr = *u_fes.GetBdrElementTransformation(be);

      for (int q = 0; q < ir_face.GetNPoints(); q++)
      {
         const IntegrationPoint &ip = ir_face.IntPoint(q);
         Tr.SetIntPoint(&ip);
         CalcOrtho(Tr.Jacobian(), nor);

         const real_t scale = nor.Norml2();
         nor /= -scale;

         u_gf.GetVectorGradient(Tr, dudx);
         auto p = p_gf.GetValue(Tr, ip);

         // σ = (-pI + ν∇u)
         for (int i = 0; i < dim; i++)
         {
            for (int j = 0; j < dim; j++)
            {
               A(i, j) = -p * (i == j) + kinematic_viscosity * dudx(i, j);
            }
         }

         // F = ∫ σ⋅n̂ dΓ
         for (int i = 0; i < dim; i++)
         {
            real_t s = 0.0;
            for (int j = 0; j < dim; j++)
            {
               s -= A(i, j) * nor(j);
            }
            draglift(i) += s * Tr.Weight() * ip.weight;
         }
      }
   }

   return {draglift(0), draglift(1)};
}



/**
 * @brief Integrator for computing dC(u,p)/du on the right
 *        hand side of the dual problem, with
 *
 *        C(u,p) = ∫ (-pI + ν∇u) ⋅ n̂ dΓ, and
 *
 *        dC(u,p)/du = ∫ d(ν∇u)/du ⋅ n̂ dΓ, since
 *
 *        u = ∑uᵢψᵢ, where ψᵢ are the FE basis functions,
 *        d(∇u)/du = ∇ψ
 *
 *        Note: we integrate the y-component, since our
 *        goal/cost function is lift.
 */
class DLiftDuIntegrator : public LinearFormIntegrator
{
public:
   DLiftDuIntegrator(real_t nu) : nu(nu) {}

   // Goal function is a face integral, hence volume integration is not needed
   void AssembleRHSElementVect(const FiniteElement &el,
                               ElementTransformation &Tr,
                               Vector &elvect) override
   {
      MFEM_ABORT("not implemented");
   }

   // Perform integration over a face
   void AssembleRHSElementVect(const FiniteElement &el,
                               FaceElementTransformations &Tr,
                               Vector &elvect) override
   {
      int dof = el.GetDof();
      int spaceDim = Tr.GetSpaceDim();

      Vector nor(spaceDim);

      dpsi.SetSize(dof, spaceDim);

      elvect.SetSize(dof * spaceDim);
      elvect = 0.0;

      const IntegrationRule *ir = IntRule;
      if (ir == NULL)
      {
         int intorder = 2 * el.GetOrder() + 2;
         ir = &IntRules.Get(el.GetGeomType(), intorder);
      }

      for (int i = 0; i < ir->GetNPoints(); i++)
      {
         const IntegrationPoint &ip = ir->IntPoint(i);
         Tr.SetAllIntPoints(&ip);

         const IntegrationPoint &eip = Tr.GetElement1IntPoint();

         CalcOrtho(Tr.Jacobian(), nor);

         // Also note the (-) negative scaling to get the correct
         // sign on the rhs of the adjoint
         const real_t scale = nor.Norml2();
         nor /= -scale;

         el.CalcPhysDShape(Tr.GetElement1Transformation(), dpsi);

         for (int d = 0; d < dof; d++)
         {
            for (int i = 0; i < spaceDim; i++)
            {
               Vector dpsi_dot_nor(spaceDim);
               dpsi_dot_nor = 0.0;
               for (int j = 0; j < spaceDim; j++)
               {
                  dpsi_dot_nor(i) += dpsi(d, j) * nor(j);
               }
               elvect(dof*i+d) += ip.weight * Tr.Weight() * nu * dpsi_dot_nor(1);
            }
         }
      }
   }

   DenseMatrix dpsi;
   const real_t nu;
};


/**
 * @brief Integrator for computing dC(u,p)/dp on the right
 *        hand side of the dual problem, with
 *
 *        C(u,p) = ∫ (-pI + ν∇u) ⋅ n̂ dΓ, and
 *
 *        dC(u,p)/dp = ∫ -pI ⋅ n̂ dΓ, since
 *
 *        p = ∑pᵢψᵢ, where ψᵢ are the FE basis functions,
 *        d(-pI)/dp = -ψ
 *
 *        Note: we integrate the y-component, since our
 *        goal/cost function is lift.
 */
class DLiftDpIntegrator : public LinearFormIntegrator
{
public:
   DLiftDpIntegrator() {}

   // Goal function is a face integral, hence volume integration is not needed
   void AssembleRHSElementVect(const FiniteElement &el,
                               ElementTransformation &Tr,
                               Vector &elvect) override
   {
      MFEM_ABORT("nope");
   }

   // Perform integration over a face
   void AssembleRHSElementVect(const FiniteElement &el,
                               FaceElementTransformations &Tr,
                               Vector &elvect) override
   {
      int dof = el.GetDof();
      int spaceDim = Tr.GetSpaceDim();

      Vector nor(spaceDim);

      psi.SetSize(dof);

      elvect.SetSize(dof * spaceDim);
      elvect = 0.0;

      const IntegrationRule *ir = IntRule;
      if (ir == nullptr)
      {
         int intorder = 2 * el.GetOrder() + 2;
         ir = &IntRules.Get(el.GetGeomType(), intorder);
      }

      for (int i = 0; i < ir->GetNPoints(); i++)
      {
         const IntegrationPoint &ip = ir->IntPoint(i);
         Tr.SetAllIntPoints(&ip);

         const IntegrationPoint &eip = Tr.GetElement1IntPoint();

         CalcOrtho(Tr.Jacobian(), nor);

         // Note the (+) positive scaling to get the correct sign
         // on the rhs of the adjoint
         const real_t scale = nor.Norml2();
         nor /= scale;

         el.CalcPhysShape(Tr.GetElement1Transformation(), psi);

         for (int d = 0; d < dof; d++)
         {
            elvect(d) += ip.weight * Tr.Weight() * psi(d) * nor(1);
         }
      }
   }

   Vector psi;
};

/**
 * @brief Compute the dual-weighted residual (DWR). The momentum residual, R₁(u,p), and
 *        continuity residual, R₂(u), are computed using the velocity and pressure (u,p)
 *        from the forward problem. Then, given the dual variables z=(z_u,z_p), the
 *        dual-weighted residual, R_z, is computed as
 *
 *        R_z = (R₁(u,p),z_u) + (R₂(u),z_p) with
 *
 *        R₁ = (u ⋅ ∇)u + ∇p  - ν∇²u
 *        R₂ = ∇⋅u
 *
 *        The primal and dual solutions (u,p) and (z_u,z_p) (and therefore R_z) live in a
 *        higher-order space and are projected to a space of piecewise-constant basis, ψ.
 *
 *        R_{z,0} = (R_z,ψ)
 */
class DWRIntegrator : public LinearFormIntegrator
{
public:
   DWRIntegrator(ParGridFunction &uhgf, ParGridFunction &phgf,
                 ParGridFunction &zuhgf, ParGridFunction &zphgf, real_t nu) :
      nu(nu), uhgf(uhgf), phgf(phgf), zuhgf(zuhgf), zphgf(zphgf) {}

   void AssembleRHSElementVect(const FiniteElement &el,
                               ElementTransformation &Tr,
                               Vector &elvect) override
   {
      int dof = el.GetDof();
      int spaceDim = Tr.GetSpaceDim();

      psi.SetSize(dof);
      dzh_duhT.SetSize(spaceDim, spaceDim);
      elvect.SetSize(dof);
      elvect = 0.0;

      const IntegrationRule *ir = IntRule;
      if (ir == nullptr)
      {
         int intorder = 6;
         ir = &IntRules.Get(el.GetGeomType(), intorder);
      }

      for (int i = 0; i < ir->GetNPoints(); i++)
      {
         const IntegrationPoint &ip = ir->IntPoint(i);

         Tr.SetIntPoint(&ip);
         el.CalcPhysShape(Tr, psi);

         uhgf.GetVectorValue(Tr, ip, uh);
         zuhgf.GetVectorValue(Tr, ip, zuh);
         real_t zph = zphgf.GetValue(Tr);
         phgf.GetGradient(Tr, dph);
         uhgf.GetVectorGradient(Tr, duh);
         zuhgf.GetVectorGradient(Tr, dzuh);

         // nu * (grad(uh) : grad(zuh), psi)
         real_t wrk1 = 0.0;
         for (int i = 0; i < duh.NumRows(); i++)
         {
            for (int j = 0; j < duh.NumCols(); j++)
            {
               wrk1 += duh(i, j) * dzuh(i, j);
            }
         }
         add(elvect, ip.weight * Tr.Weight() * nu * wrk1, psi, elvect);

         // (grad(uh), zuh grad(psi)) -> 0

         // (zuh dot(grad(uh), uh), psi)
         wrk1 = 0.0;
         for (int i = 0; i < duh.NumRows(); i++)
         {
            for (int j = 0; j < duh.NumCols(); j++)
            {
               wrk1 += duh(i, j) * uh(j) * zuh(i);
            }
         }
         add(elvect, ip.weight * Tr.Weight() * wrk1, psi, elvect);

         // (zuh grad(ph), psi)
         wrk1 = 0.0;
         for (int i = 0; i < dph.Size(); i++)
         {
            wrk1 += dph(i) * zuh(i);
         }
         add(elvect, ip.weight * Tr.Weight() * wrk1, psi, elvect);

         // (zph * div(uh), psi)
         add(elvect, ip.weight * Tr.Weight() * duh.Trace() * zph, psi, elvect);
      }

      for (int i = 0; i < dof; i ++)
      {
         elvect(i) = abs(elvect(i));
      }
   }

   Vector psi;
   Vector zuh, uh, dph;
   DenseMatrix dzuh, duh, dzh_duhT;

   const real_t nu;
   ParGridFunction uhgf, phgf, zuhgf, zphgf;
};

/**
 * @brief Constructs the operators in steady-state, incompressible
 *        Navier-Stokes equations (NSE)
 *
 *        (u ⋅ ∇)u + ∇p - ν∇²u = 0
 *        ∇⋅u = 0
 *
 *        in block form
 *
 *        [ N(u) + L    G ] [u] = [0]
 *        [ D           0 ] [p] = [0]
 *
 *        with the (L)aplacian, (G)radient, (D)ivergence and
 *        (N)onliner convection operator.
 */
class NavierStokes : public Operator
{
public:
   NavierStokes(ParFiniteElementSpace &ufes,
                ParFiniteElementSpace &pfes,
                Array<int> &u_ess_tdof,
                Array<int> &p_ess_tdof,
                const real_t kinematic_viscosity) :
      ufes(ufes),
      pfes(pfes),
      u_ess_tdof(u_ess_tdof),
      p_ess_tdof(p_ess_tdof),
      N(&ufes),
      Ne(&ufes),
      g(&pfes, &ufes),
      ge(&pfes, &ufes),
      d(&ufes, &pfes),
      de(&ufes, &pfes),
      offsets({0, ufes.GetTrueVSize(), pfes.GetTrueVSize()}),
           mp(&pfes),
           pq(&pfes)
   {
      offsets.PartialSum();
      pc = new BlockLowerTriangularPreconditioner(offsets);

      this->height = offsets.Last();
      this->width = this->height;

      kinematic_viscosity_coef.constant = kinematic_viscosity;
      inverse_kinematic_viscosity_coef.constant = 1.0 / kinematic_viscosity;

      Array<int> empty;

      // Build the nonlinear form with the convection and
      // vector diffusion (Laplacian) terms
      N.AddDomainIntegrator(new VectorConvectionNLFIntegrator);
      N.AddDomainIntegrator(
         new VectorDiffusionIntegrator(kinematic_viscosity_coef));
      N.SetAssemblyLevel(AssemblyLevel::PARTIAL);
      N.Setup();

      Ne.AddDomainIntegrator(new VectorConvectionNLFIntegrator);
      Ne.AddDomainIntegrator(
         new VectorDiffusionIntegrator(kinematic_viscosity_coef));
      Ne.SetEssentialTrueDofs(u_ess_tdof);
      Ne.Setup();

      // Build the gradient integrator for the pressure term
      // in the momentum equation
      g.AddDomainIntegrator(new GradientIntegrator);
      g.SetAssemblyLevel(AssemblyLevel::PARTIAL);
      g.Assemble();
      g.Finalize();
      g.FormRectangularSystemMatrix(empty, empty, G);

      ge.AddDomainIntegrator(new GradientIntegrator);
      ge.SetAssemblyLevel(AssemblyLevel::PARTIAL);
      ge.Assemble();
      ge.Finalize();
      ge.FormRectangularSystemMatrix(p_ess_tdof, u_ess_tdof, Ge);

      // Build the velocity divergence term in the continuity equation
      d.AddDomainIntegrator(new VectorDivergenceIntegrator);
      d.SetAssemblyLevel(AssemblyLevel::PARTIAL);
      d.Assemble();
      d.Finalize();
      d.FormRectangularSystemMatrix(empty, empty, D);

      de.AddDomainIntegrator(new VectorDivergenceIntegrator);
      de.SetAssemblyLevel(AssemblyLevel::PARTIAL);
      de.Assemble();
      de.Finalize();
      de.FormRectangularSystemMatrix(u_ess_tdof, p_ess_tdof, De);

      // Form the (1,1) P block to enforce pressure dirichlet BC in the
      // linear solves at essential DOFs
      zero_coeff.constant = 0.0;
      pq.AddDomainIntegrator(new MassIntegrator(zero_coeff));
      pq.Assemble();
      pq.Finalize();
      pq.FormSystemMatrix(p_ess_tdof, PQ);

      // Form the Navier-Stokes block operator
      R = new BlockOperator(offsets);
      R->SetBlock(0, 0, &N);
      R->SetBlock(0, 1, G.Ptr());
      R->SetBlock(1, 0, D.Ptr());

      // Form the block operator representing the gradient of the
      // Navier-Stokes equations (i.e. linearized) with-respect-to the velocity and pressure
      // (0,1) and (1,0) blocks are linear and do not change, while the
      // (0,0) is the nonlinear convection operator, hence the gradient
      // is computed when needed (below). The (1,1) block is used to enforce
      // pressure dirichlet BC
      Ae = new BlockOperator(offsets);
      Ae->SetBlock(0, 1, Ge.Ptr());
      Ae->SetBlock(1, 0, De.Ptr());
      Ae->SetBlock(1, 1, PQ.Ptr());

      mp.AddDomainIntegrator(new MassIntegrator(inverse_kinematic_viscosity_coef));
      mp.Assemble();
      mp.Finalize();
      mp.FormSystemMatrix(p_ess_tdof, Mpe);

      // Construct block lower triangular preconditioner
      // [N(u)       0  ]
      // [Div    ν⁻¹ M]
      // where M is the mass matrix on the pressure space
      mpe_inv = new OperatorJacobiSmoother(mp, p_ess_tdof);
      pc->SetBlock(1, 1, mpe_inv);
      pc->SetBlock(1, 0, De.Ptr());
   }

   // Apply NSE block operator
   void Mult(const Vector &x, Vector &y) const override
   {
      BlockVector yb(y.GetData(), offsets);
      R->Mult(x, y);
      yb.GetBlock(0).SetSubVector(u_ess_tdof, 0.0);
      yb.GetBlock(1).SetSubVector(p_ess_tdof, 0.0);
   }

   // Compute gradient about current velocity and return
   // gradient of NSE operator
   Operator &GetGradient(const Vector &x) const override
   {
      BlockVector xb(x.GetData(), offsets);
      Ae->SetBlock(0, 0, &Ne.GetGradient(xb.GetBlock(0)));

      RebuildPC();

      return *Ae;
   }

   void RebuildPC() const
   {
      delete a_inv;

      auto amg = new HypreBoomerAMG;
      HYPRE_BoomerAMGSetSmoothType(*amg, 5);
      amg->SetOperator(*static_cast<HypreParMatrix*>(&Ae->GetBlock(0, 0)));
      amg->SetSystemsOptions(2, true);
      amg->SetPrintLevel(0);
      a_inv = amg;

      pc->SetBlock(0, 0, a_inv);
   }

   ~NavierStokes()
   {
      delete R;
      delete Ae;
      delete mpe_inv;
      delete a_inv;
      delete pc;
   }

   ParFiniteElementSpace &ufes;
   ParFiniteElementSpace &pfes;
   Array<int> &u_ess_tdof;
   Array<int> &p_ess_tdof;
   ConstantCoefficient kinematic_viscosity_coef;
   ConstantCoefficient inverse_kinematic_viscosity_coef;
   ParNonlinearForm N, Ne;
   ParBilinearForm pq;
   ParMixedBilinearForm g, ge, d, de;
   Array<int> offsets;
   BlockOperator *R = nullptr;
   mutable BlockOperator *Ae = nullptr;

   OperatorHandle K, Ke;
   OperatorHandle G, Ge;
   OperatorHandle D, De;
   OperatorHandle PQ, Mpe;

   mutable BlockLowerTriangularPreconditioner *pc = nullptr;
   ParBilinearForm mp;
   Solver *mpe_inv = nullptr;
   mutable Solver *a_inv = nullptr;
   ConstantCoefficient zero_coeff;
};

/**
 * @brief Constructs the linearized dual/adjoint of the steady-state,
 *        incompressible Navier-Stokes equations (NSE) in block form
 *
 *        [ N'(u) + L   G^T ] [z_u] = [ N'(u) + L  D ][z_u]
 *        [ D^T          0  ] [z_p] = [ G          0 ][z_p]
 *
 *        with the (L)aplacian, (G)radient, (D)ivergence and linearized
 *        (N')onliner convection operator. Note that the bilinear form
 *        corresponding to the diffusion/Laplacian term is self-adjoint
 */
class NavierStokesAdjoint : public Operator
{
public:
   NavierStokesAdjoint(ParFiniteElementSpace &ufes,
                       ParFiniteElementSpace &pfes,
                       Array<int> &u_ess_tdof,
                       Array<int> &p_ess_tdof,
                       const real_t kinematic_viscosity) :
      ufes(ufes),
      pfes(pfes),
      u_ess_tdof(u_ess_tdof),
      p_ess_tdof(p_ess_tdof),
      Ne(&ufes),
      ge_tr(&ufes, &pfes),
      de_tr(&pfes, &ufes),
      offsets({0, ufes.GetTrueVSize(), pfes.GetTrueVSize()}),
           mp(&pfes),
           pq(&pfes)
   {
      offsets.PartialSum();
      pc = new BlockLowerTriangularPreconditioner(offsets);

      this->height = offsets.Last();
      this->width = this->height;

      kinematic_viscosity_coef.constant = kinematic_viscosity;
      inverse_kinematic_viscosity_coef.constant = 1.0 / kinematic_viscosity;

      Array<int> empty;
   }

   void Setup(const Vector &x)
   {

      // Build the nonlinear form with the convection integrator and
      // vector diffusion (Laplacian) terms. Note that the bilinear
      // form corresponding to the Laplacian term is self-adjoint
      Ne.AddDomainIntegrator(new VectorConvectionNLFIntegrator);
      Ne.AddDomainIntegrator(
         new VectorDiffusionIntegrator(kinematic_viscosity_coef));
      Ne.SetEssentialTrueDofs(u_ess_tdof);
      Ne.Setup();

      // Compute the nonlinear operator's linearized dual
      auto KeTr = static_cast<HypreParMatrix &>(Ne.GetGradient(x)).Transpose();

      // Build the dual of the pressure gradient term
      // Note that dual is weak divergence operator which equivalent to the
      // transpose of the gradient
      ge_tr.AddDomainIntegrator(new TransposeIntegrator(new GradientIntegrator));
      ge_tr.SetAssemblyLevel(AssemblyLevel::PARTIAL);
      ge_tr.Assemble();
      ge_tr.Finalize();
      ge_tr.FormRectangularSystemMatrix(u_ess_tdof, p_ess_tdof, GeTr);

      // Build the dual of the velocity divergence term in the continuity eq.
      // Note that dual is the gradiant operator which equivalent to the
      // transpose of the divergence operator
      de_tr.AddDomainIntegrator(new TransposeIntegrator(new
                                                        VectorDivergenceIntegrator));
      de_tr.SetAssemblyLevel(AssemblyLevel::PARTIAL);
      de_tr.Assemble();
      de_tr.Finalize();
      de_tr.FormRectangularSystemMatrix(p_ess_tdof, u_ess_tdof, DeTr);

      // Form the (1,1) P block to enforce pressure dirichlet BC in the
      // linear solves at essential DOFs
      zero_coeff.constant = 0.0;
      pq.AddDomainIntegrator(new MassIntegrator(zero_coeff));
      pq.Assemble();
      pq.Finalize();
      pq.FormSystemMatrix(p_ess_tdof, PQ);


      // Construct the block operator corresponding to the linearized dual
      // of the Navier-Stokes operator
      Ae = new BlockOperator(offsets);
      Ae->SetBlock(0, 0, KeTr);
      Ae->SetBlock(0, 1, DeTr.Ptr());
      Ae->SetBlock(1, 0, GeTr.Ptr());
      Ae->SetBlock(1, 1, PQ.Ptr());

      mp.AddDomainIntegrator(new MassIntegrator(inverse_kinematic_viscosity_coef));
      mp.Assemble();
      mp.Finalize();
      mp.FormSystemMatrix(p_ess_tdof, Mpe);

      // Construct block lower triangular preconditioner
      // [N'(u)      0  ]
      // [Div    ν⁻¹ M]
      // where N' is the linearzed convection term,
      // M is the mass matrix on the dual pressure space
      mpe_inv = new OperatorJacobiSmoother(mp, p_ess_tdof);
      pc->SetBlock(1, 1, mpe_inv);
      pc->SetBlock(1, 0, GeTr.Ptr());

      auto amg = new HypreBoomerAMG;
      amg->SetOperator(*static_cast<HypreParMatrix*>(KeTr));
      amg->SetPrintLevel(0);
      a_inv = amg;

      pc->SetBlock(0, 0, a_inv);
   }

   void Mult(const Vector &x, Vector &y) const override
   {
      Ae->Mult(x, y);
   }

   ~NavierStokesAdjoint()
   {
      delete Ae;
      delete mpe_inv;
      delete a_inv;
      delete pc;
   }

   ParFiniteElementSpace &ufes;
   ParFiniteElementSpace &pfes;
   Array<int> &u_ess_tdof;
   Array<int> &p_ess_tdof;
   ConstantCoefficient kinematic_viscosity_coef;
   ConstantCoefficient inverse_kinematic_viscosity_coef;
   ParNonlinearForm Ne;
   ParBilinearForm pq;
   ParMixedBilinearForm ge_tr, de_tr;
   Array<int> offsets;
   mutable BlockOperator *Ae = nullptr;

   OperatorHandle Ke;
   OperatorHandle GeTr;
   OperatorHandle DeTr;
   OperatorHandle PQ, Mpe;

   mutable BlockLowerTriangularPreconditioner *pc = nullptr;
   ParBilinearForm mp;
   Solver *mpe_inv = nullptr;
   mutable Solver *a_inv = nullptr;
   ConstantCoefficient zero_coeff;
};


/**
 * @brief Solve the steady-state, incompressible Navier-Stokes equations
 *        and return velocity and pressure fields
 *
 * @param h1vfes FESpace for velocity
 * @param h1fes  FESpace for pressure
 * @param kinematic_viscosity viscosity coefficient
 * @return std::tuple<ParGridFunction, ParGridFunction> tuple for (velocity, pressure) grid functions
 */
std::tuple<ParGridFunction, ParGridFunction>
SolveForwardProblem(
   ParFiniteElementSpace &h1vfes,
   ParFiniteElementSpace &h1fes,
   real_t kinematic_viscosity)
{
   const int myid = Mpi::WorldRank();
   ParMesh &mesh = *h1fes.GetParMesh();
   const int dim = mesh.Dimension();

   HYPRE_BigInt size = h1fes.GlobalTrueVSize();
   if (myid == 0)
   {
      cout << "Number of finite element unknowns: " << size * (dim + 1) << endl;
   }

   // Get dofs corresponding to the essential BC for velocity
   Array<int> vel_ess_tdof_list;
   {
      Array<int> ess_bdr(mesh.bdr_attributes.Max());
      ess_bdr = 1;
      ess_bdr[1] = 0;
      ess_bdr[4] = 0;
      h1vfes.GetEssentialTrueDofs(ess_bdr, vel_ess_tdof_list);
   }

   // Get dofs corresponding to the essential BC for pressure
   Array<int> pres_ess_tdof_list;
   {
      Array<int> ess_bdr(mesh.bdr_attributes.Max());
      ess_bdr = 0;
      ess_bdr[1] = 1;
      h1fes.GetEssentialTrueDofs(ess_bdr, pres_ess_tdof_list);
   }

   NavierStokes ns(h1vfes, h1fes, vel_ess_tdof_list, pres_ess_tdof_list,
                   kinematic_viscosity);

   ParGridFunction u(&h1vfes), p(&h1fes);
   u = 0.0;
   p = 0.0;

   auto inflow = [](const Vector &coords, Vector &u)
   {
      const real_t y = coords(1);
      const real_t U = 0.3;
      u(0) = 4.0 * U * y * (0.41 - y) / pow(0.41, 2.0);
      u(1) = 0.0;
   };

   // Set initial guess for the Newton solve. The initial guess
   // at essential BC dofs also sets the boundary condition
   // at those dofs.
   Array<int> inflow_attr(mesh.bdr_attributes.Max());
   inflow_attr = 0;
   inflow_attr[0] = 1;
   VectorFunctionCoefficient inflow_coef(dim, inflow);
   u.ProjectBdrCoefficient(inflow_coef, inflow_attr);

   BlockVector x(ns.offsets);
   u.GetTrueDofs(x.GetBlock(0));
   p.GetTrueDofs(x.GetBlock(1));

   GMRESSolver krylov(MPI_COMM_WORLD);
   krylov.SetRelTol(1e-4);
   krylov.SetAbsTol(1e-12);
   krylov.SetKDim(300);
   krylov.SetMaxIter(1000);
   krylov.SetPreconditioner(*ns.pc);
   krylov.SetPrintLevel(IterativeSolver::PrintLevel().Summary());

   NewtonSolver newton(MPI_COMM_WORLD);
   newton.SetRelTol(1e-8);
   newton.SetAbsTol(1e-9);
   newton.SetMaxIter(15);
   newton.SetPrintLevel(IterativeSolver::PrintLevel().Iterations());
   newton.SetOperator(ns);
   newton.SetPreconditioner(krylov);

   Vector zero;
   newton.Mult(zero, x);

   u.SetFromTrueDofs(x.GetBlock(0));
   p.SetFromTrueDofs(x.GetBlock(1));

   return {u, p};
}


/**
 * @brief Solve the linearized dual problem. Given state variables q=(u,p),
 *        a goal/cost function C(q) (e.g. lift) and equality constraints of
 *        a governing equation F(q) = 0 (e.g. Navier-Stokes) the Lagrangian ℒ is
 *
 *        ℒ = C(q) + z F(q)
 *
 *        where z = (z_u,z_p) is the dual of the velocity and pressure. The linerized dual
 *        formulation is obtained from the first order optimality condition dℒ/dq=0
 *
 *        dℒ/dq = (dC(q)/dq)ᵀ + zᵀ dF(q)/dq = 0
 *              → (dF(q)/dq)ᵀ z = -dC(q)/dq
 *
 *              → dC(q)/dq = [ dC(u,p)/du, dC(u,p)/dp ] in block form
 *
 *        which is solved for the dual variables z. Here, (dF(q)/dq)ᵀ is the adjoint of the
 *        linearized Navier-Stokes operator, and dC(q)/dq are the derivatives of the goal (lift)
 *        with-respect-to the state q=(u,p) computed by the forward solve.
 *
 * @param h1vfes FESpace for dual of velocity
 * @param h1fes  FESpace for dual of pressure
 * @param ul grid function for velocity computed from the forward problem
 * @param pl grid function for pressure computed from the forward problem
 * @param kinematic_viscosity coefficient of viscosity
 * @return std::tuple<ParGridFunction, ParGridFunction> tuple of grid functions for
 *                                                      velocity, pressure duals
 */
std::tuple<ParGridFunction, ParGridFunction>
SolveDualProblem(
   ParFiniteElementSpace &h1vfes,
   ParFiniteElementSpace &h1fes,
   ParGridFunction &ul,
   ParGridFunction &pl,
   real_t kinematic_viscosity)
{
   const int myid = Mpi::WorldRank();
   ParMesh &mesh = *h1fes.GetParMesh();
   const int dim = mesh.Dimension();

   // Get dofs corresponding to the essential BC for velocity-dual
   Array<int> vel_ess_tdof_list;
   {
      Array<int> ess_bdr(mesh.bdr_attributes.Max());
      ess_bdr = 1;
      ess_bdr[1] = 0;
      ess_bdr[4] = 0;
      h1vfes.GetEssentialTrueDofs(ess_bdr, vel_ess_tdof_list);
   }

   // Get dofs corresponding to the essential BC for pressure-dual
   Array<int> pres_ess_tdof_list;
   {
      Array<int> ess_bdr(mesh.bdr_attributes.Max());
      ess_bdr = 0;
      ess_bdr[1] = 1;
      h1fes.GetEssentialTrueDofs(ess_bdr, pres_ess_tdof_list);
   }

   NavierStokesAdjoint adjoint(h1vfes, h1fes, vel_ess_tdof_list,
                               pres_ess_tdof_list,
                               kinematic_viscosity);

   // Project ul and pl from the forward problem defined on a coarse finite element
   // space to the finer space of the dual variables
   ParGridFunction u(&h1vfes);
   PRefinementTransferOperator(static_cast<FiniteElementSpace>(*ul.ParFESpace()),
                               static_cast<FiniteElementSpace>(h1vfes)).Mult(ul, u);

   ParGridFunction p(&h1fes);
   PRefinementTransferOperator(static_cast<FiniteElementSpace>(*pl.ParFESpace()),
                               static_cast<FiniteElementSpace>(h1fes)).Mult(pl, p);

   BlockVector x(adjoint.offsets);
   u.GetTrueDofs(x.GetBlock(0));
   p.GetTrueDofs(x.GetBlock(1));

   // Build the forms in the linear adjoint of the Navier-Stokes
   // Here, the linearization of the convection term is performed
   // about the velocity from the forward problem
   adjoint.Setup(x.GetBlock(0));

   BlockVector b(adjoint.offsets);

   Vector v(2);
   v = 1.0;
   VectorConstantCoefficient vone(v);
   ConstantCoefficient one(1.0);

   // Mark mesh boundary attribute which the QoI was computd
   Array<int> qoi_attr(mesh.bdr_attributes.Max());
   qoi_attr = 0;
   qoi_attr[5] = 1;
   qoi_attr[6] = 1;

   // Construct the linear form corresponding to dC(u,p)/du in the
   // right hand side of the dual problem
   ParLinearForm dldu(&h1vfes);
   auto lfi1 = new DLiftDuIntegrator(kinematic_viscosity);
   dldu.AddBdrFaceIntegrator(lfi1, qoi_attr);
   dldu.Assemble();
   dldu.ParallelAssemble(b.GetBlock(0));

   // Construct the linear form corresponding to dC(u,p)/dp in the
   // right hand side of the dual problem
   ParLinearForm dldp(&h1fes);
   auto lfi2 = new DLiftDpIntegrator;
   dldp.AddBdrFaceIntegrator(lfi2, qoi_attr);
   dldp.Assemble();
   dldp.ParallelAssemble(b.GetBlock(1));

   GMRESSolver krylov(MPI_COMM_WORLD);
   krylov.SetRelTol(1e-8);
   krylov.SetAbsTol(1e-12);
   krylov.SetKDim(300);
   krylov.SetMaxIter(5000);
   krylov.SetOperator(adjoint);
   krylov.SetPreconditioner(*adjoint.pc);
   krylov.SetPrintLevel(IterativeSolver::PrintLevel().Summary());

   b.GetBlock(0).SetSubVector(vel_ess_tdof_list, 0.0);
   b.GetBlock(1).SetSubVector(pres_ess_tdof_list, 0.0);

   krylov.Mult(b, x); // Solve the dual problem for z = (z_u, z_p)

   u.SetFromTrueDofs(x.GetBlock(0));
   p.SetFromTrueDofs(x.GetBlock(1));

   return {u, p};
}

/**
 * @brief Compute the quantity of interst (QoI) for use in DWR based AMR
 *
 * @param u velocity grid function
 * @param p pressure grid function
 * @param kinematic_viscosity coefficient of viscosity
 * @return std::tuple<real_t, real_t> tuple containing QoIs
 */
std::tuple<real_t, real_t>
ComputeQoI(ParGridFunction &u, ParGridFunction &p,
           real_t kinematic_viscosity)
{
   ParFiniteElementSpace &h1vfes = *u.ParFESpace();
   ParMesh &mesh = *h1vfes.GetParMesh();
   const int dim = mesh.Dimension();
   const int polynomial_order = h1vfes.GetOrder(0);

   Array<int> qoi_attr(mesh.bdr_attributes.Max());
   qoi_attr = 0;
   qoi_attr[5] = 1;
   qoi_attr[6] = 1;

   auto ir_face = IntRules.Get(h1vfes.GetMesh()->GetFaceGeometry(0),
                               2 * polynomial_order + 1);
   auto [drag_local, lift_local] = DragLift(u, p, kinematic_viscosity, qoi_attr,
                                            ir_face);

   real_t drag_global = 0.0, lift_global = 0.0;
   MPI_Allreduce(&drag_local, &drag_global, 1, MFEM_MPI_REAL_T, MPI_SUM,
                 MPI_COMM_WORLD);
   MPI_Allreduce(&lift_local, &lift_global, 1, MFEM_MPI_REAL_T, MPI_SUM,
                 MPI_COMM_WORLD);

   return {drag_global, lift_global};
}

int main(int argc, char *argv[])
{
   Mpi::Init();
   int num_procs = Mpi::WorldSize();
   int myid = Mpi::WorldRank();
   Hypre::Init();

   const char *mesh_file = "../data/channel-cylinder.msh";
   int polynomial_order = 2;
   const char *device_config = "cpu";
   bool visualization = true;
   real_t kinematic_viscosity = 1.0e-3;
   int uniform_refinements = 3;
   int max_iterations = 3;

   OptionsParser args(argc, argv);
   args.AddOption(&polynomial_order, "-o", "--order",
                  "Finite element order (polynomial degree)");
   args.AddOption(&mesh_file, "-m", "--mesh", "Mesh file to use.");
   args.AddOption(&uniform_refinements, "-ref", "--refinements",
                  "uniform refinements");
   args.AddOption(&kinematic_viscosity, "-kv", "--kinematic_viscosity",
                  "kinematic viscosity");
   args.AddOption(&max_iterations, "-max_it", "--max_iterations",
                  "maximum refinement iterations");
   args.AddOption(&device_config, "-d", "--device",
                  "Device configuration string, see Device::Configure().");
   args.ParseCheck();

   Mesh serial_mesh(mesh_file);
   serial_mesh.EnsureNCMesh();
   const int dim = serial_mesh.Dimension();

   for (int i = 0; i < uniform_refinements; i++)
   {
      serial_mesh.UniformRefinement();
   }

   ParMesh mesh(MPI_COMM_WORLD, serial_mesh);
   serial_mesh.Clear();

   ParaViewDataCollection paraview_dc("navier_stokes_dwr", &mesh);
   paraview_dc.SetDataFormat(VTKFormat::BINARY);
   paraview_dc.SetHighOrderOutput(true);

   for (int iterations = 0; iterations < max_iterations; iterations++)
   {
      H1_FECollection fec(polynomial_order, dim);
      ParFiniteElementSpace h1vfes(&mesh, &fec, dim, Ordering::byNODES);

      H1_FECollection fecl(polynomial_order - 1, dim);
      ParFiniteElementSpace h1fes(&mesh, &fecl);

      // Solve the forward problem (steady, incompressible, Navier-Stokes)
      // to compute velocity and pressure
      auto [u, p] = SolveForwardProblem(h1vfes, h1fes, kinematic_viscosity);

      // Compute the quantity of interest (lift and drag forces) over the boundary
      auto [drag, lift] = ComputeQoI(u, p, kinematic_viscosity);

      const real_t U_mean = 0.2;
      const real_t c0 = 2.0 / (U_mean*U_mean * 0.1);

      if (Mpi::Root())
      {
         printf("CD = %1.8E CL = %1.8E\n", -c0*drag, -c0*lift);
      }

      // Solve the dual problem in an enriched space for velocity
      // and pressure dual variables, zu and zp, respectively.
      H1_FECollection fecstar(polynomial_order + 1, dim);
      ParFiniteElementSpace h1vfesstar(&mesh, &fecstar, dim, Ordering::byNODES);

      H1_FECollection feclstar(polynomial_order, dim);
      ParFiniteElementSpace h1fesstar(&mesh, &feclstar);

      auto [zu, zp] = SolveDualProblem(h1vfesstar, h1fesstar, u, p,
                                       kinematic_viscosity);

      // Compute the dual-weighted residual (DWR) given both the primal, (u,p), and
      // dual, (zu,zp), solutions. The error indicator is constructed per-element, hence
      // piecewise-constant basis are used.
      L2_FECollection l2fec(0, mesh.Dimension());
      ParFiniteElementSpace marker_fes(&mesh, &l2fec);

      ParGridFunction eta(&marker_fes);

      ParLinearForm dwr_lf(&marker_fes);
      dwr_lf.AddDomainIntegrator(new DWRIntegrator(u, p, zu, zp,
                                                   kinematic_viscosity));
      dwr_lf.Assemble();
      eta = dwr_lf;

      paraview_dc.SetCycle(iterations);
      paraview_dc.SetTime(iterations);
      paraview_dc.SetLevelsOfDetail(polynomial_order);
      paraview_dc.RegisterField("velocity", &u);
      paraview_dc.RegisterField("pressure", &p);
      paraview_dc.RegisterField("velocity_dual", &zu);
      paraview_dc.RegisterField("pressure_dual", &zp);
      paraview_dc.RegisterField("error_indicator", &eta);
      paraview_dc.Save();

      // Refinement marking
      {
         const int num_ranks = Mpi::WorldSize();
         const int local_ne = mesh.GetNE();
         const int global_ne = mesh.GetGlobalNE();
         // Limit refinements to 10% of the global number of elements
         int element_refinement_limit = std::round(global_ne * 0.1);

         // Compute error indicate per mesh element
         std::vector<real_t> error_indicator(global_ne);
         std::fill(error_indicator.begin(), error_indicator.end(), 0.0);
         Vector dofs;
         for (int ge = 0; ge < global_ne; ge++)
         {
            const int e = mesh.GetLocalElementNum(ge);
            if (e != -1)
            {
               eta.GetElementDofValues(e, dofs);
               error_indicator[ge] = abs(dofs(0));
            }
         }

         std::vector<real_t> global_error_indicator(global_ne);
         MPI_Allreduce(error_indicator.data(), global_error_indicator.data(),
                       global_ne, MFEM_MPI_REAL_T, MPI_SUM, MPI_COMM_WORLD);

         // Sort error indicator (indices) in descending order
         auto ranking = sort_indexes(global_error_indicator);
         std::reverse(ranking.begin(), ranking.end());

         std::vector<int> ranking_pruned(element_refinement_limit);
         for (int i = 0; i < element_refinement_limit; i++)
         {
            ranking_pruned[i] = ranking[i];
         }

         // Mark error prone elements for refinement
         Array<int> el_to_refine;
         for (auto &ge : ranking_pruned)
         {
            const int e = mesh.GetLocalElementNum(ge);
            if (e != -1)
            {
               el_to_refine.Append(e);
            }
         }
         mesh.GeneralRefinement(el_to_refine, 1, 1);
      }
   }

   return 0;
}
