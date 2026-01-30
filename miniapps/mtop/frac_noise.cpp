#include "frac_noise.hpp"

using namespace mfem;
using namespace std;

FracRandomFieldGenerator::FracRandomFieldGenerator(ParMesh &pmesh_,
                                                   const int par_ref_levels_, 
                                                   const int order_, 
                                                   real_t sigma_, real_t s_)
    : pmesh(pmesh_), par_ref_levels(par_ref_levels_),
      order(order_), sigma(sigma_), s(s_)
{
    fec.reset(new H1_FECollection(order, pmesh.Dimension()));
    ParFiniteElementSpace *coarse_fespace = new ParFiniteElementSpace(&pmesh, fec.get());

    fespaces.reset(new ParFiniteElementSpaceHierarchy(&pmesh, coarse_fespace,
                                                      false, true)); // take ownership of coarse space

    for(int l = 0; l < par_ref_levels; l++)
    {
        fespaces->AddUniformlyRefinedLevel(1, Ordering::byVDIM);
    }
    
    // set the prolongation operators for the multigrid
    const int nlevels = fespaces->GetNumLevels();
    prolongations.SetSize(nlevels - 1);
    for (int level = 0; level < nlevels - 1; ++level)
    {
        prolongations[level] = fespaces->GetProlongationAtLevel(level);
    }

    //set the operators and the smoothers for the multigrid
    operators.SetSize(nlevels);
    smoothers.SetSize(nlevels);

    const Array<int> ess_tdofs;

    for (int level = 0; level < nlevels; ++level)
    {
        // mass and diffusion operators on each level
        ParFiniteElementSpace &fespace = fespaces->GetFESpaceAtLevel(level);
        unique_ptr<ParBilinearForm> bf(new ParBilinearForm(&fespace));
        ConstantCoefficient one(1.0);
        ConstantCoefficient mass_coef(sigma*sigma);

        bf->AddDomainIntegrator(new MassIntegrator(mass_coef));
        bf->AddDomainIntegrator(new DiffusionIntegrator(one));
        bf->Assemble();
        bf->Finalize();
        HypreParMatrix *mat= bf->ParallelAssemble();
        operators[level] = mat;

        // Jacobi smoother on each level
        Vector diag(fespace.GetTrueVSize());
        Vector tmp(fespace.GetTrueVSize()); 
        mat->GetDiag(diag);
        if(false)
        {
            tmp=1.0;
            mat->AbsMult(tmp, diag); // diag = |A|*1
        }
        else{
            for(int i=0; i<diag.Size(); i++)
            {
                diag[i]=1.0/sqrt(diag[i]);
            }
            mat->Mult(diag, tmp); 
            real_t omega=InnerProduct(fespace.GetComm(), diag, tmp); 
            mat->GetDiag(diag); diag *= omega;
        }

        OperatorJacobiSmoother *jac = new OperatorJacobiSmoother(diag,
                                                                 ess_tdofs,
                                                                 1.0 /*weight*/);
        jac->iterative_mode = false;
        jac->Setup(diag);
        smoothers[level]=jac;
    }
};

FracRandomFieldGenerator::~FracRandomFieldGenerator()
{
    for(int i = 0; i < operators.Size(); i++){delete operators[i];}
    for(int i = 0; i < smoothers.Size(); i++){delete smoothers[i];}
}

void FracRandomFieldGenerator::Mult(const Vector &x, Vector &y) const
{

    const int nlevels = fespaces->GetNumLevels();
    SymmetrizedSmoother ms(smoothers[nlevels-1], operators[nlevels-1]);
    vector<Vector*> u;
    vector<Vector*> v;

    int myrank;
    MPI_Comm_rank(pmesh.GetComm(), &myrank);

    
    for(int level = 0; level < nlevels; level++)
    {
        Vector *w = new Vector(); w->SetSize( fespaces->GetFESpaceAtLevel(level).GetTrueVSize() );
        u.push_back(w);

        w = new Vector(); w->SetSize( fespaces->GetFESpaceAtLevel(level).GetTrueVSize() );
        v.push_back(w);

        if(0==myrank)  {cout << "Level " << level << 
            " size: " << fespaces->GetFESpaceAtLevel(level).GlobalTrueVSize()<<  
                " loc:" << fespaces->GetFESpaceAtLevel(level).GetTrueVSize() << endl;

                cout<<" smoothe size:"<< smoothers[level]->Height() 
                    << " operator size:" << operators[level]->Height() << endl;

                if(level<nlevels-1)
                {
                    cout<<" prolong size:"<< prolongations[level]->Height() 
                        << " x " << prolongations[level]->Width() << endl;
                }
        }    

    }
    

    ms.Mult(x, *(u[nlevels -1]));

    Vector  tva, tvb, tvc; 
    tva.SetSize(y.Size()); tva=x;
    tvb.SetSize(y.Size());
    tvc.SetSize(y.Size()); 

    for(int level = nlevels - 2; level >=0; level--)
    {
         if(1==myrank)  {cout << "Level " << level << 
                " size: " << fespaces->GetFESpaceAtLevel(level).GlobalTrueVSize() << endl;}

        smoothers[level+1]->Mult(tva, tvb);
        operators[level+1]->Mult(tvb, tvc);
        add(-1.0, tvc, 1.0, tva, tvb); 
        tva.SetSize(prolongations[level]->Width());

        if(1==myrank)  {cout << "Level " << level << 
                " restrict from size: " << tvb.Size() << 
                " to size: " << tva.Size() << endl;}

        prolongations[level]->MultTranspose(tvb, tva);

        
        ms.SetSmoother(*smoothers[level]);
        ms.SetOperator(*operators[level]);
        //smoothing
        ms.Mult(tva, *(u[level]));
        

        tvb.SetSize( tva.Size() );
        tvc.SetSize( tva.Size() );
    }


    // compute h on each level
    Array<real_t> h; h.SetSize(nlevels);
    {
        real_t h_min, h_max, kappa_min, kappa_max;
        for(int level =0; level < nlevels; level++)
        {
            ParFiniteElementSpace &fes = fespaces->GetFESpaceAtLevel(level);
            fes.GetParMesh()->GetCharacteristics(h_min, h_max, kappa_min, kappa_max);
            h[level]=h_max;
            if(0==myrank)
            {
                cout << " Level " << level << " h_min: " << h_min << 
                    " h_max: " << h_max << endl;
            }
        }
    }

    // coarse to fine loop
    real_t mu;
    mu=pow(sigma*sigma +1.0/(h[0]*h[0]),s);
    v[0]->Set(mu, *(u[0]) );

    for(int level =1; level < nlevels; level++)
    {
        mu = sigma*sigma + 1.0/(h[level]*h[level]);
        mu = pow(mu, s);

        tva.SetSize( v[level]->Size() );
        tvb.SetSize( v[level]->Size() );
        tvc.SetSize( v[level]->Size() );

        prolongations[level-1]->Mult(*(v[level-1]), tva );
        operators[level]->Mult(tva, tvb);
        smoothers[level]->MultTranspose(tvb, tvc);
        tva.Add(-1.0, tvc);


        v[level]->Set(mu, *(u[level]) );
        v[level]->Add(1.0, tva);
    }

    // final scaling
    mu=pow(sigma*sigma +1.0/(h[0]*h[0]),s);
    y.Set(1.0/mu, *(v[nlevels -1]));


    for(int level = 0; level < nlevels; level++)
    {
        delete u[level];
        delete v[level];
    }
}

