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
    fespace = std::make_unique<ParFiniteElementSpace>(&mesh, fec.get(), dim, Ordering::byNODES);

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
    block_true_offsets[2] = fespace->TrueVSize();

    block_true_offsets.PartialSum();

    sol.Update(block_true_offsets); sol=0.0; sol.UseDevice(true);
    rhs.Update(block_true_offsets); rhs=0.0; rhs.UseDevice(true);
    tmp.Update(block_true_offsets); tmp=0.0; tmp.UseDevice(true);   
    
    res.SetSize(fespace->GetTrueVSize()); res=0.0; res.UseDevice(true);

    displ.SetSpace(fespace.get()); displ=0.0; 
    displ.SetTrueVector(); 
    displ.GetTrueVector().UseDevice(true);
    
    veloc.SetSpace(fespace.get()); veloc=0.0; 
    veloc.SetTrueVector(); 
    veloc.GetTrueVector().UseDevice(true);


    this->width = 2*fespace->TrueVSize();
    this->height = 2*fespace->TrueVSize();

    MPI_Comm_rank(mesh.GetComm(),&myrank);

    vol_force_mem.SetSize(10);
    vol_force_mem.UseDevice(true);
    vol_force_mem(0) = 0.0; // time
    vol_force_mem(1) = 1.0; // period
    vol_force_mem(2) = 0.0; // amplitude
    vol_force_mem(3) = 0.5; // radius
    vol_force_mem(4) = 0.0; // x coordinate of the center
    vol_force_mem(5) = 0.0; // y coordinate of the center
    vol_force_mem(6) = 0.0; // z coordinate of the center   
    vol_force_mem(7) = 5*vol_force_mem(1); // total train length 
    vol_force_mem(8) = vol_force_mem(7)/2.0;
    vol_force_mem(9) = 2.0;

    bdr_force_mem.SetSize(3);
    bdr_force_mem.UseDevice(true);
    bdr_force_mem(0) = 0.0; // time
    bdr_force_mem(1) = 1.0; // period
    bdr_force_mem(2) = 0.0; // amplitude

    obj.reset();

} 

void LinearElasticityTimeDependentOperator::SetObjective(std::shared_ptr<Operator> op_)
{
    if(op_.get()!=nullptr)
    {
        obj=op_;
        //set the new objective and readjust the size of the operator and the state
        block_true_offsets.SetSize(4);
        block_true_offsets[0] = 0;
        block_true_offsets[1] = fespace->TrueVSize();
        block_true_offsets[2] = fespace->TrueVSize();
        block_true_offsets[3] = op_->Height();

        block_true_offsets.PartialSum();

        sol.Update(block_true_offsets); sol=0.0; sol.UseDevice(true);
        rhs.Update(block_true_offsets); rhs=0.0; rhs.UseDevice(true);
        tmp.Update(block_true_offsets); tmp=0.0; tmp.UseDevice(true);   

        this->width = block_true_offsets[3];
        this->height = block_true_offsets[3];        
    }else{

        obj.reset();

        block_true_offsets.SetSize(3);
        block_true_offsets[0] = 0;
        block_true_offsets[1] = fespace->TrueVSize();
        block_true_offsets[2] = fespace->TrueVSize();

        block_true_offsets.PartialSum();

        sol.Update(block_true_offsets); sol=0.0; sol.UseDevice(true);
        rhs.Update(block_true_offsets); rhs=0.0; rhs.UseDevice(true);
        tmp.Update(block_true_offsets); tmp=0.0; tmp.UseDevice(true);   

        this->width = 2*fespace->TrueVSize();
        this->height = 2*fespace->TrueVSize();
    }
}

template <int DI, typename scalar_t=real_t> struct QElasticityFunction
{
    using matd_t = tensor<scalar_t, DI, DI>;
    using vecd_t = tensor<scalar_t, DI>;
    using vec_t = tensor<real_t, DI>;
    using mat_t = tensor<real_t, DI, DI>;

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

    struct DynamicBdrForce
    {
        //real_t time=0.0;
        //real_t period=1.0;
        mfem::Vector* time_mem; 

        //mfem::Memory<int> alt_time; check the documentation about Memory class for more details

        DynamicBdrForce(mfem::Vector& tm) // the Read method should be called on the vector passed as tm 
                                       // before calling the Mult on the differentiable operator when 
                                       // the time is changing, i.e., the values between the host 
                                       // and device have to be synchronized.
        {   
            time_mem = tm.Read(); //get the device pointer
        }

        MFEM_HOST_DEVICE inline auto operator()(const vecd_t &u,
                                                const matd_t &J,
                                                const real_t &w
                                            ) const
        {
            const real_t time = (*time_mem)(0);
            const real_t period = (*time_mem)(1);
            const real_t amplitude = (*time_mem)(2);
            const auto detJ = mfem::future::det(J);
            // time dependent force in x direction
            const real_t force_amplitude = (time > 0.0) ? amplitude*sin(M_PI*time/period) : 0.0;
            vecd_t force {0};//= vecd_t::Zero();
            force(0) = force_amplitude;
            return tuple{force * detJ * w};
        }

    };

    
    struct DynamicVolForce
    {
        const real_t* time_mem; 
        DynamicVolForce(mfem::Vector& tm)
        {   
            time_mem = tm.Read(); //get the device pointer
        }

        
        MFEM_HOST_DEVICE inline auto operator()(const vecd_t &u,
                                                const vec_t &x,
                                                const matd_t &J,
                                                const real_t &w
                                                ) const
            
        {
            const real_t time = *(time_mem+0);
            const real_t period = *(time_mem+1);
            const real_t amplitude = *(time_mem+2);
            const real_t radius = *(time_mem+3);

            const real_t L=*(time_mem+7);
            const real_t t0=*(time_mem+8);
            const real_t n=*(time_mem+9);

            const real_t envelope_ampl= (time< L) ?pow(cos(M_PI*(time-t0)/L),n) : 0.0;
            const real_t force_amplitude = (time > 0.0) ? amplitude*sin(2.0*M_PI*time/period) : 0.0;

            vecd_t force {0};

            // time dependent force in x direction
            force(0) = force_amplitude*envelope_ampl;

            //compute the distance from the center of the force application
            real_t dist_sq = 0.0;
            for (int i = 0; i < DI; i++)
            {
                const real_t diff = x(i) - *(time_mem+4+i);
                dist_sq += diff * diff;
            }
                
            // apply the force only within the specified radius
            if(dist_sq > radius*radius)
            {
                force(0) = 0.0;
            }                

            const auto detJ = mfem::future::det(J);
            return tuple{force * detJ * w};
        }

        struct  Objective
        {
            /* data */
            const real_t* obj_mem;
            Objective(mfem::Vector& tm)
            {
                obj_mem = tm.Read(); //get the device pointer
            }

            // takes velocity and returns squared velocity 
            MFEM_HOST_DEVICE inline auto operator()(const vecd_t &u,
                                                const vec_t &x,
                                                const mat_t &J,
                                                const real_t &w
                                                ) const
            
            {
                const real_t time = *(obj_mem+0);
                const real_t radius = *(obj_mem+1);
                //compute the distance from the center of the objective circle/sphere
                real_t dist_sq = 0.0;
                
                scalar_t obj = 0.0;

                for (int i = 0; i < DI; i++)
                {
                    const real_t diff = x(i) - *(time_mem+2+i);
                    dist_sq += diff * diff;

                    obj  +=  u(i) * u(i);
                }
                
                // apply the obj only within the specified radius
                if(dist_sq > radius*radius)
                {
                    obj = 0.0;
                }
                
                const auto detJ = mfem::future::det(J);
                return tuple{obj * detJ * w};

            }

        };

        struct  ObjectiveGrad
        {
            /* data */
            const real_t* obj_mem;

            ObjectiveGrad(mfem::Vector& tm)
            {
                obj_mem = tm.Read(); //get the device pointer
            }

            // takes velocity and returns squared velocity 
            MFEM_HOST_DEVICE inline auto operator()(const vecd_t &u,
                                                const vecd_t &x,
                                                const matd_t &J,
                                                const real_t &w
                                                ) const
            
            {
                const real_t time = *(obj_mem+0);
                const real_t radius = *(obj_mem+1);
                //compute the distance from the center of the objective circle/sphere
                scalar_t dist_sq = 0.0;
                vecd_t obj_grad;
                real_t objc = 1.0;

                for (int i = 0; i < DI; i++)
                {
                    const real_t diff = x(i) - *(time_mem+2+i);
                    dist_sq += diff * diff;

                    obj_grad(i) = 2.0 * u(i);
                }
                
                // apply the obj only within the specified radius
                if(dist_sq > radius*radius)
                {
                    objc = 0.0;
                }
                
                const auto detJ = mfem::future::det(J);
                return tuple{objc* obj_grad * detJ * w};

            }

        };
        
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
    
    // define the damp differentiable operator
    {
        dfem_damp_op = std::make_unique<mfem::future::DifferentiableOperator>(
            std::vector<mfem::future::FieldDescriptor>{ {FVeloc, fespace.get()} },
            std::vector<mfem::future::FieldDescriptor>{ 
                {CMass1, ups.get()},
                {CMass2, ups.get()},
                {Density, ups.get()},
                {Coords, mfes}
            },
            mesh);

        dfem_damp_op->SetParameters({ cm1.get(), cm2.get(), density.get(), nodes });

        const auto dinputs =
            mfem::future::tuple{
                mfem::future::Value<FVeloc>{},
                mfem::future::Identity<CMass1>{},
                mfem::future::Identity<CMass2>{},
                mfem::future::Identity<Density>{},
                mfem::future::Gradient<Coords>{},
                mfem::future::Weight{}
            };

        const auto doutputs =
            mfem::future::tuple{
                mfem::future::Value<FVeloc>{} 
            };

        if (2 == space_dim)
        {
            typename QElasticityFunction<2>::Mass damp_func;
            dfem_damp_op->AddDomainIntegrator(damp_func, dinputs, doutputs, *ir, domain_attributes);
        }
        else if (3 == space_dim)
        {
            typename QElasticityFunction<3>::Mass damp_func;
            dfem_damp_op->AddDomainIntegrator(damp_func, dinputs, doutputs, *ir, domain_attributes); 
        }
    }



    //define the volumetric force differentiable operator
    {
        dfem_vol_force_op = std::make_unique<mfem::future::DifferentiableOperator>(
            std::vector<mfem::future::FieldDescriptor>{ {FDispl, fespace.get()} },
            std::vector<mfem::future::FieldDescriptor>{ 
                {Coords, mfes}
            },
            mesh);

        dfem_vol_force_op->SetParameters({ nodes });

        const auto finputs =
            mfem::future::tuple{
                mfem::future::Value<FDispl>{},
                mfem::future::Value<Coords>{},
                mfem::future::Gradient<Coords>{},
                mfem::future::Weight{}
            };

        const auto foutputs =
            mfem::future::tuple{
                mfem::future::Value<FDispl>{} 
            };

        if (2 == space_dim)
        {
            typename QElasticityFunction<2>::DynamicVolForce vol_force_func(vol_force_mem);
            dfem_vol_force_op->AddDomainIntegrator(vol_force_func, finputs, foutputs, *ir, domain_attributes);
        }
        else if (3 == space_dim)
        {
            typename QElasticityFunction<3>::DynamicVolForce vol_force_func(vol_force_mem);
            dfem_vol_force_op->AddDomainIntegrator(vol_force_func, finputs, foutputs, *ir, domain_attributes); 
        }
    }   

    // define the linear elasticity differentiable operator
    {
        dfem_forward_op = std::make_unique<mfem::future::DifferentiableOperator>(
            std::vector<mfem::future::FieldDescriptor>{ {FDispl, fespace.get()} },
            std::vector<mfem::future::FieldDescriptor>{ 
                {Lambda1, ups.get()},
                {Mu1, ups.get()},
                {Lambda2, ups.get()},
                {Mu2, ups.get()},
                {Density, ups.get()},
                {Coords, mfes}
            },
            mesh);

        dfem_forward_op->SetParameters({ l1.get(), m1.get(), l2.get(), m2.get(), density.get(), nodes });

        const auto finputs =
            mfem::future::tuple{
                mfem::future::Gradient<FDispl>{},
                mfem::future::Identity<Lambda1>{},
                mfem::future::Identity<Mu1>{},
                mfem::future::Identity<Lambda2>{},
                mfem::future::Identity<Mu2>{},
                mfem::future::Identity<Density>{},
                mfem::future::Gradient<Coords>{},
                mfem::future::Weight{}
            };

        const auto foutputs =
            mfem::future::tuple{
                mfem::future::Gradient<FDispl>{} 
            };

        if (2 == space_dim)
        {
            typename QElasticityFunction<2>::Elasticity elasticity_func;
            dfem_forward_op->AddDomainIntegrator(elasticity_func, finputs, foutputs, *ir, domain_attributes);
        }
        else if (3 == space_dim)
        {
            typename QElasticityFunction<3>::Elasticity elasticity_func;
            dfem_forward_op->AddDomainIntegrator(elasticity_func, finputs, foutputs, *ir, domain_attributes); 
        }
    }

    //Spectral mass-matrix
    {
        InterpolatedCoefficient interp_dens1(*cdens1, *cdens2, *cdensity);

        IntegrationRules gll_rules(0, Quadrature1D::GaussLobatto);

        const IntegrationRule &ir_ni = gll_rules.Get(mesh.GetTypicalElementGeometry(),
                                                        2 * order - 1);

        ParBilinearForm bf_lor(fespace.get());
        auto *mv_blfi = new VectorMassIntegrator(interp_dens1);
        mv_blfi->SetIntRule(&ir_ni);
        //bf_lor.AddDomainIntegrator(new LumpedIntegrator(mv_blfi));
        bf_lor.AddDomainIntegrator(mv_blfi);
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
        cg->SetRelTol(1e-7);
        cg->SetAbsTol(1e-12);
        cg->SetMaxIter(500);
        cg->SetPrintLevel(0);
        cg->SetOperator(*dfem_mass_op);
        //cg->SetOperator(*M_lor);
        cg->SetPreconditioner(*amg);
        cg->iterative_mode=false;
    }

    //set the zero bdr conditions
    {
        Array<int> bdr_attr; bdr_attr.SetSize(mesh.bdr_attributes.Max());
        bdr_attr=0;
        for(const auto &it:zero_bdrs)
        {
            bdr_attr[it-1]=1.0;
        }
        fespace->GetEssentialTrueDofs(bdr_attr,ess_tdof_list);
    }
    
}

void LinearElasticityTimeDependentOperator::Mult(const Vector &x,
                                                 Vector &y) const
{
    real_t time = this->GetTime();

    BlockVector bx(const_cast<Vector&>(x), block_true_offsets);
    BlockVector by(y, block_true_offsets);

    displ.GetTrueVector().Set(1.0,bx.GetBlock(0));
    veloc.GetTrueVector().Set(1.0,bx.GetBlock(1));    
    //set zero BC
    {
        int N = ess_tdof_list.Size();
        real_t *dp=displ.GetTrueVector().ReadWrite();
        real_t *vp=veloc.GetTrueVector().ReadWrite();
        const int *ep = ess_tdof_list.Read();
        mfem::forall(N, [=] MFEM_HOST_DEVICE(int i) { 
                        dp[ep[i]] = 0.0;
                        vp[ep[i]] = 0.0;
                     });
    }
    //displ.SetFromTrueVector();
    //veloc.SetFromTrueVector();


    by.GetBlock(0).Set(1.0, veloc.GetTrueVector()); // dx/dt = velocity

    // compute the residual
    
    // 1) add external volumetric forces 
    real_t* pvol_force_mem=vol_force_mem.HostReadWrite(); //get the host pointer
    pvol_force_mem[0]=time; //set the current time to be pass to the integrator
    vol_force_mem.Read(); //copy force_mem from host to device
    // call the kernel computing f_ext
    // dfem_vol_force_op->SetParameters({nodes}); // it is already set
    dfem_vol_force_op->Mult(veloc.GetTrueVector(),res);

    // 2) compute the mass proportional viscous damping term
    // dfem_damp_op->SetParameters({ cm1.get(), cm2.get(), density.get(), nodes });
    dfem_damp_op->Mult(veloc.GetTrueVector(), tmp.GetBlock(1)); 
    res -= tmp.GetBlock(1);

    // 3) add the stiffness proportional viscous damping term

    // 4) add the elastic force term
    dfem_forward_op->Mult(displ.GetTrueVector(),tmp.GetBlock(0));
    res-= tmp.GetBlock(0);


    //dfem_mass_op->SetParameters({dens1.get(), dens2.get(), density.get(), nodes});
    cg->Mult(res, by.GetBlock(1)); // solve for acceleration

    //check if objective is valid
    if(obj.get()!=nullptr)
    {
        //evaluate the objective contribution
        obj->Mult(x,by.GetBlock(2));
    }
}

// implements the adjoint reverse time integration 
// i.e. x=[l_q,l_v, L_\rho]^T y=x' - i.e. the derivative with respect to \tau=T-t 
// before calling MultTranspose one should set the sol vector with the
// solution for the forward problem at time t
void LinearElasticityTimeDependentOperator::AdjointMult(const Vector &x,
                                                            Vector &y) const
{
    BlockVector bx(const_cast<Vector&>(x), block_true_offsets);
    BlockVector by(y, block_true_offsets);

    y=0.0;
}


void LinearElasticityTimeDependentOperator::ImplicitSolve(
    const real_t dt,
    const Vector &x,
    Vector &k)
{
}



template <int DI, typename scalar_t=real_t> struct QObjectiveFunction
{
    using matd_t = tensor<scalar_t, DI, DI>;
    using vecd_t = tensor<scalar_t, DI>;
    using vec_t = tensor<real_t, DI>;
    using mat_t = tensor<real_t, DI, DI>;

    struct Objective1
    {
        MFEM_HOST_DEVICE inline auto operator()(const vecd_t &u,
                                                const real_t &co,
                                                const matd_t &J,
                                                const real_t &w) const
        {
            scalar_t rez; 
            rez=0.0;
            for(int i=0;i<DI;i++)
            {
                rez=rez+u(i)*u(i);
            }
            rez=rez*co;
            const auto detJ = mfem::future::det(J);
            return tuple{rez * detJ * w};
        }
    };

    struct Objective2
    {
        const real_t s1;
        const real_t s2;

        Objective2(real_t s1_ = real_t(1.0), real_t s2_ = real_t(1.0)):s1(s1_),s2(s2_)
        {

        }

        MFEM_HOST_DEVICE inline auto operator()(const vecd_t &u,
                                                const vecd_t &v,
                                                const real_t &co,
                                                const matd_t &J,
                                                const real_t &w) const
        {
            scalar_t rez=0.0;
            for(int i=0;i<DI;i++)
            {
                rez=rez+u(i)*u(i)*s1+v(i)*v(i)*s2;
            }
            const auto detJ = mfem::future::det(J);
            return tuple{rez * detJ * w};
        }
    };

};


ExampleObjectiveIntegrand::ExampleObjectiveIntegrand(ParFiniteElementSpace* fes_, 
                                std::shared_ptr<mfem::Coefficient> objc_) 
{
    fes=fes_;

    fes->GetParMesh()->EnsureNodes();

    disp.SetSpace(fes); disp=0.0;
    velo.SetSpace(fes); velo=0.0;

    this->width=2*fes->GetTrueVSize(); //disp.Size() + veloc.Size()
    this->height=1; //returns 3 objectives     

    grad=nullptr;

     //set the block sizes for the solution, rhs and tmp vectors
    block_true_offsets.SetSize(3);
    block_true_offsets[0] = 0;
    block_true_offsets[1] = fes->TrueVSize();
    block_true_offsets[2] = fes->TrueVSize();

    block_true_offsets.PartialSum();

    const mfem::FiniteElement *fe= fes->GetFE(0);
    ir = &(IntRules.Get(fe->GetGeomType(),
                      fe->GetOrder() + fe->GetOrder() + fe->GetDim() - 1));

    qs.reset(new QuadratureSpace(*(fes->GetParMesh()), *ir));

    ups.reset(new future::UniformParameterSpace(
        *(fes->GetParMesh()), *ir, 1, false /* used_in_tensor_product */));
    
    if (fes->GetParMesh()->attributes.Size() > 0)
    {
      domain_attributes.SetSize(fes->GetParMesh()->attributes.Max());
      domain_attributes = 1;
    }

    nodes = static_cast<ParGridFunction *>(fes->GetParMesh()->GetNodes());
    mfes = nodes->ParFESpace();

    SetCoefficients(objc_);

}

void ExampleObjectiveIntegrand::SetCoefficients( std::shared_ptr<mfem::Coefficient> objc)
{
    co=objc;
    if(co.get()!=nullptr)
    {
        //project the coefficient
        density.reset(new mfem::CoefficientVector(*qs, mfem::CoefficientStorage::FULL)); 
        density->Project(*co);
    }else{
        density.reset(new mfem::CoefficientVector(*qs, mfem::CoefficientStorage::FULL));
        density->SetConstant(1.0);
    }

    res.SetSize(density->Size());

    //allocate the differentiable operator
    {
        obj = std::make_unique<mfem::future::DifferentiableOperator>(
            std::vector<mfem::future::FieldDescriptor>{ {FDispl, fes} },
            std::vector<mfem::future::FieldDescriptor>{
                {Density, ups.get()},
                {Coords, mfes}
            },
            *(fes->GetParMesh())
        );

        obj->SetParameters( {density.get(), nodes} );

        const auto finputs =
            mfem::future::tuple{
                mfem::future::Value<FDispl>{},
                mfem::future::Identity<Density>{}, 
                mfem::future::Gradient<Coords>{},
                mfem::future::Weight{}
            };
        
        const auto foutputs =
            mfem::future::tuple{
                mfem::future::Identity<Density>{} 
            };
        
        int space_dim=fes->GetParMesh()->SpaceDimension();

        if(2 == space_dim)
        {
            using mfem::future::dual;
            using dual_t = dual<real_t, real_t>;
            typename QObjectiveFunction<2,dual_t>::Objective1 obj_func;
            auto derivatives = std::integer_sequence<size_t, FDispl, Coords> {};
            obj->AddDomainIntegrator(obj_func, finputs, foutputs, *ir, domain_attributes, derivatives);
        }
        else if( 3 == space_dim)
        {
            using mfem::future::dual;
            using dual_t = dual<real_t, real_t>;
            typename QObjectiveFunction<3,dual_t>::Objective1 obj_func;
            auto derivatives = std::integer_sequence<size_t, FDispl, Coords> {};
            obj->AddDomainIntegrator(obj_func, finputs, foutputs, *ir, domain_attributes, derivatives);
        }

    }
}

void ExampleObjectiveIntegrand::Mult(const Vector &x, Vector &y) const
{

    mfem::Array<int> lblock_true_offsets;
    lblock_true_offsets.SetSize(4);
    lblock_true_offsets[0] = 0;
    lblock_true_offsets[1] = fes->TrueVSize();
    lblock_true_offsets[2] = fes->TrueVSize();
    lblock_true_offsets[3] = x.Size()-2*fes->TrueVSize();
    lblock_true_offsets.PartialSum();

    BlockVector bx(const_cast<Vector&>(x), lblock_true_offsets);

    obj->Mult(bx.GetBlock(0),res);
    //sum up the weighted values
    real_t lp=mfem::InnerProduct(fes->GetComm(), res, *density);
    y[0]=lp;
}

void ExampleObjectiveIntegrand::EvalGradient(const Vector &x, Vector &grad) const
{
    mfem::Array<int> lblock_true_offsets;
    lblock_true_offsets.SetSize(4);
    lblock_true_offsets[0] = 0;
    lblock_true_offsets[1] = fes->TrueVSize();
    lblock_true_offsets[2] = fes->TrueVSize();
    lblock_true_offsets[3] = x.Size()-2*fes->TrueVSize();
    lblock_true_offsets.PartialSum();

    BlockVector bx(const_cast<Vector&>(x), lblock_true_offsets);
    BlockVector by(grad, lblock_true_offsets); by=0.0;
    disp.SetFromTrueDofs(bx.GetBlock(0));

    std::shared_ptr<mfem::future::DerivativeOperator> dobj_du;
    dobj_du=obj->GetDerivative(FDispl, {&disp},{density.get(), nodes});


    if (Mpi::Root())
    {
       std::cout << "Op size: " << dobj_du->Height()<<" "<<dobj_du->Width()<< std::endl;
       std::cout << " disp size:"<< bx.GetBlock(0).Size()<<std::endl;
       std::cout << " dens size:"<< density->Size()<<std::endl;
    }

    dobj_du->MultTranspose(*density,by.GetBlock(0));
}
