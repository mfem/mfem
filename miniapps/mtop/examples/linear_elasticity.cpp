#include "linear_elasticity.hpp"

using namespace mfem;

using mfem::future::dual;
using mfem::future::tuple;
using mfem::future::tensor;

using mfem::future::Weight;
using mfem::future::Gradient;
using mfem::future::Identity;


LinearElasticityTimeDependentOperator::LinearElasticityTimeDependentOperator(ParMesh &mesh_, int vorder)
    : TimeDependentOperator(),
      mesh(mesh_),
      order(vorder)
{
    mesh.EnsureNodes();
    dim = mesh.Dimension();
    space_dim = mesh.SpaceDimension();

    fec = std::make_unique<H1_FECollection>(order, dim);
    fespace = std::make_unique<ParFiniteElementSpace>(&mesh, fec.get(), dim);

    nodes = static_cast<ParGridFunction *>(mesh.GetNodes());
    mfes = nodes->ParFESpace();

    domain_attributes.SetSize(mesh_.attributes.Max());

    const mfem::FiniteElement *fe= fespace->GetFE(0);
    ir = &(IntRules.Get(fe->GetGeomType(),
                      fe->GetOrder() + fe->GetOrder() + fe->GetDim() - 1));

    qs.reset(new QuadratureSpace(mesh, *ir));

    fqs.reset(new FaceQuadratureSpace(mesh, order+1, FaceType::Boundary));

    ups.reset(new future::UniformParameterSpace(
        mesh, *ir, 1, false /* used_in_tensor_product */));
    
    if (mesh.attributes.Size() > 0)
    {
      domain_attributes.SetSize(mesh.attributes.Max());
      domain_attributes = 1;
    }

    //set the block sizes for the solution, rhs and tmp vectors
    block_true_offsets.SetSize(3);
    block_true_offsets[0] = 0;
    block_true_offsets[1] = fespace->TrueVSize();
    block_true_offsets[2] = 2*fespace->TrueVSize();

    sol.Update(block_true_offsets); sol=0.0;
    rhs.Update(block_true_offsets); rhs=0.0;
    tmp.Update(block_true_offsets); tmp=0.0;        

    this->width = 2*fespace->TrueVSize();
    this->height = 2*fespace->TrueVSize();

    MPI_Comm_rank(mesh.GetComm(),&myrank);

} 

template <int DI, typename scalar_t=real_t> struct QElasticityFunction
{
    using matd_t = tensor<real_t, DI, DI>;
    using vecd_t = tensor<scalar_t, DI>;

    struct Mass
    {
        MFEM_HOST_DEVICE inline auto operator()(const vecd_t &u,
                                                const real_t &dens1,
                                                const real_t &dens2,
                                                const scalar_t &density,
                                                const matd_t &J,
                                                const real_t &w) const
        {
            const auto dens = density*dens2 + (1.0-density)*dens1;
            const auto detJ = mfem::future::det(J);
            return tuple{dens * u * detJ * w};
        }
        
    };

    struct Elasticity
    {
        MFEM_HOST_DEVICE inline auto operator()(const matd_t &dudxi,
                                                const real_t &L1,
                                                const real_t &M1,
                                                const real_t &L2,
                                                const real_t &M2,
                                                const scalar_t &density,
                                                const matd_t &J,
                                                const real_t &w) const
        {
            const matd_t JxW = transpose(inv(J)) * det(J) * w;
            constexpr auto I = mfem::future::IsotropicIdentity<DI>();
            const auto eps = mfem::future::sym(dudxi * mfem::future::inv(J));
            const auto L = density*L2 + (1.0-density)*L1;
            const auto M = density*M2 + (1.0-density)*M1;
            return tuple{(L * tr(eps) * I + 2.0 * M * eps) * JxW};
        }
    };

    struct DynamicForce
    {
        real_t time=0.0;
        real_t period=1.0;

        MFEM_HOST_DEVICE inline auto operator()(const vecd_t &u,
                                                const matd_t &J,
                                                const real_t &w) const
        {
            const auto detJ = mfem::future::det(J);
            // time dependent force in x direction
            const real_t force_amplitude = (time > 0.0) ? sin(M_PI*time/period) : 0.0;
            vecd_t force = vecd_t::Zero();
            force(0) = force_amplitude;
            return tuple{force * detJ * w};
        }
    };

};

class InterpolatedCoefficient : public mfem::Coefficient
{

public:
    InterpolatedCoefficient(mfem::Coefficient &c1, mfem::Coefficient &c2, mfem::Coefficient &c3)
        : coeff1(c1), coeff2(c2), coeff3(c3) {}

    virtual double Eval(mfem::ElementTransformation &T,
                        const mfem::IntegrationPoint &ip) override
    {
        real_t c1=coeff1.Eval(T, ip);
        real_t c2=coeff2.Eval(T, ip);
        real_t dens=coeff3.Eval(T, ip);
        return c2*dens + (1.0-dens)*c1;
    }
private:
    mfem::Coefficient &coeff1;
    mfem::Coefficient &coeff2;
    mfem::Coefficient &coeff3;

};

void LinearElasticityTimeDependentOperator::AssembleExplicit()
{
    // define the mass differentiable operator
    {
        dfem_mass_op = std::make_unique<mfem::future::DifferentiableOperator>(
            std::vector<mfem::future::FieldDescriptor>{ {FDispl, fespace.get()} },
            std::vector<mfem::future::FieldDescriptor>{ 
                {Dens1, ups.get()},
                {Dens2, ups.get()},
                {Density, ups.get()},
                {Coords, mfes}
            },
            mesh);

        dfem_mass_op->SetParameters({ dens1.get(), dens2.get(), density.get(), nodes });

        const auto minputs =
            mfem::future::tuple{
                mfem::future::Value<FDispl>{},
                mfem::future::Identity<Dens1>{},
                mfem::future::Identity<Dens2>{},
                mfem::future::Identity<Density>{},
                mfem::future::Gradient<Coords>{},
                mfem::future::Weight{}
            };

        const auto moutputs =
            mfem::future::tuple{
                mfem::future::Value<FDispl>{} 
            };

        if (2 == space_dim)
        {
            typename QElasticityFunction<2>::Mass mass_func;
            dfem_mass_op->AddDomainIntegrator(mass_func, minputs, moutputs, *ir, domain_attributes);
        }
        else if (3 == space_dim)
        {
            typename QElasticityFunction<3>::Mass mass_func;
            dfem_mass_op->AddDomainIntegrator(mass_func, minputs, moutputs, *ir, domain_attributes); 
        }
    }

    // define the linear elasticity differentiable operator
    /*
    {
        dfem_forward_op = std::make_unique<mfem::future::DifferentiableOperator>(
            std::vector<FieldDescriptor>{ {FDispl, fespace.get()} },
            std::vector<FieldDescriptor>{ 
                {Lame1, ups.get()},
                {Mu1, ups.get()},
                {Lame2, ups.get()},
                {Mu2, ups.get()},
                {Density, ups.get()},
                {Coords, mfes}
            },
            mesh);

        dfem_forward_op->SetParameters({ &l1, &m1, &l2, &m2, &density, nodes });

        const auto finputs =
            mfem::future::tupel{
                mfem::future::Value<FDispl>{},
                mfem::future::Identity<Lame1>{},
                mfem::future::Identity<Mu1>{},
                mfem::future::Identity<Lame2>{},
                mfem::future::Identity<Mu2>{},
                mfem::future::Identity<Density>{},
                mfem::future::Gradient<Coords>{},
                mfem::future::Weight{}
            };

        const auto foutputs =
            mfem::future::tupel{
                mfem::future::Value<FDispl>{} 
            };

        if (2 == space_dim)
        {
            typename QElasticityFunction<2>::Elasticity elasticity_func;
            dfem_forward_op->SetFunction(elasticity_func, finputs, foutputs, ir, domain_attributes);
        }
        else if (3 == space_dim)
        {
            typename QElasticityFunction<3>::Elasticity elasticity_func;
            dfem_forward_op->SetFunction(elasticity_func, finputs, foutputs, ir, domain_attributes); 
        }
    }*/

    //LOR mass matrix
    {
        std::unique_ptr<mfem::ParLORDiscretization> lor_disc;
        lor_disc = std::make_unique<mfem::ParLORDiscretization>(*fespace);
        ParFiniteElementSpace &lor_space = lor_disc->GetParFESpace();
        /*
        ParMesh &lor_mesh = *lor_space.GetParMesh();
        lor_mesh.EnsureNodes();
        ParGridFunction* lor_nodes=static_cast<ParGridFunction *>(lor_mesh.GetNodes());
        ParFiniteElementSpace* lor_nodes_fes = lor_nodes->ParFESpace();
        */

        InterpolatedCoefficient interp_dens1(*cdens1, *cdens2, *cdensity);

        ParBilinearForm bf_lor(&lor_space);
        bf_lor.AddDomainIntegrator(new MassIntegrator(interp_dens1));
        bf_lor.Assemble();
        bf_lor.Finalize();
        M_lor.reset(bf_lor.ParallelAssemble());
    }

    // allocate the AMG preconditioner and CG solver
    // for the mass matrix
    {
        amg = std::make_unique<HypreBoomerAMG>();
        amg->SetPrintLevel(1);
        amg->SetOperator(*M_lor);

        cg = std::make_unique<CGSolver>(mesh.GetComm());
        cg->SetRelTol(1e-12);
        cg->SetAbsTol(0.0);
        cg->SetMaxIter(500);
        cg->SetPrintLevel(1);
        cg->SetOperator(*dfem_mass_op);
        cg->SetPreconditioner(*amg);
    }
    
}

void LinearElasticityTimeDependentOperator::Mult(const Vector &x,
                                                 Vector &y) const
{
    real_t time = this->GetTime();

    BlockVector bx(const_cast<Vector&>(x), block_true_offsets);
    BlockVector by(y, block_true_offsets);

    displ.SetFromTrueDofs(bx.GetBlock(0));
    veloc.SetFromTrueDofs(bx.GetBlock(1));

    Array<int> atrs; atrs.SetSize(mesh.bdr_attributes.Max()); atrs = 0;
    // set the BC for displacement field
    for (const auto &it : bdr_displ)
    {
        int bdr_attr = it.first;
        std::shared_ptr<VectorCoefficient> bc = it.second;
        VectorCoefficient& vc=*bc;
        bc->SetTime(time);
        atrs[bdr_attr-1] = 1;   
        displ.ProjectBdrCoefficient(vc, atrs);
        atrs[bdr_attr-1] = 0;
    }
    // set BC for velocity field
    for (const auto &it : bdr_veloc)
    {
        int bdr_attr = it.first;
        std::shared_ptr<VectorCoefficient> bc = it.second;
        VectorCoefficient& vc=*bc;
        bc->SetTime(time);
        atrs[bdr_attr-1] = 1;   
        veloc.ProjectBdrCoefficient(vc, atrs);
        atrs[bdr_attr-1] = 0;
    }

    displ.SetTrueVector();
    veloc.SetTrueVector();

    by.GetBlock(0).Set(1.0, veloc.GetTrueVector()); // dx/dt = velocity

    // compute the residual
    res.SetSize(bx.GetBlock(1).Size());
    //add external forces if any
    res = 0.0;

    //compute the mass proportional viscous damping term
    dfem_mass_op->SetParameters({ cm1.get(), cm2.get(), density.get(), nodes });
    dfem_mass_op->Mult(veloc.GetTrueVector(), tmp.GetBlock(1)); 
    res -= tmp.GetBlock(1);

    // add the stiffness proportional viscous damping term

    // add the elastic force term

    cg->Mult(res, by.GetBlock(1)); // solve for acceleration
}

void LinearElasticityTimeDependentOperator::ImplicitSolve(
    const real_t dt,
    const Vector &x,
    Vector &k)
{
}

