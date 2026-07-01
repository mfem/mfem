
#include "mfem.hpp"
#include "../QuantityOfInterest.hpp"
#include <limits>

using namespace std;
using namespace mfem;

// Max-length constraint  G = 1/2 ∫_Ω (γ − α)² dx  
class MaxFilterResidual : public QuantityOfInterest
{
protected:
    ParFiniteElementSpace *fes;
    MPI_Comm comm;

    GridFunctionCoefficient gamma_cf, alpha_cf;
    SumCoefficient     diff_cf;          //  γ − α  
    ProductCoefficient diff2_cf;         // (γ − α)²

public:
    MaxFilterResidual(MPI_Comm comm_, ParGridFunction &gamma_, ParGridFunction &alpha_)
    : fes(alpha_.ParFESpace()), comm(comm_),
      gamma_cf(&gamma_), alpha_cf(&alpha_),
      diff_cf(gamma_cf, alpha_cf, 1.0, -1.0),
      diff2_cf(diff_cf, diff_cf) { }
    ~MaxFilterResidual() { }

    // return coefficient evaluating (γ − α) 
    Coefficient *GetResidualCoefficient() { return &diff_cf; }

    // G = 1/2 ∫_Ω (γ − α)² dx
    real_t Eval() override
    {
        ParLinearForm lf(fes);
        lf.AddDomainIntegrator(new DomainLFIntegrator(diff2_cf));
        lf.Assemble();
        std::unique_ptr<HypreParVector> v(lf.ParallelAssemble());   // ∫_Ω (γ−α)² 

        real_t loc, val;
        loc = v->Sum();
        MPI_Allreduce(&loc, &val, 1, MPITypeMap<real_t>::mpi_type, MPI_SUM, comm);
        return 0.5 * val;
    }

    // dG/dα = (α − γ, ·)_L2.
    void GetGrad(Vector &dGdalpha) override
    {
        ParLinearForm la(fes);
        la.AddDomainIntegrator(new DomainLFIntegrator(diff_cf));
        la.Assemble();
        std::unique_ptr<HypreParVector> va(la.ParallelAssemble());

        dGdalpha = *va;
        dGdalpha.Neg();              // - (γ - α) = (α - γ)
    }
};

#ifdef MFEM_USE_GSLIB
class ThicknessResidual : public QuantityOfInterest
{
    protected:
    ParFiniteElementSpace *fes;
    ParGridFunction *rho_filter;  // filtered design density
    FindPointsGSLIB finder;

    
    Vector interp_vals;         // interpolated values at all sample points
    Array<unsigned int> code;   // gslib point location codes
    Array<unsigned int> proc;   // gslib processor indices for all sample points
    Array<unsigned int> elem;   // gslib element indices for all sample points
    
    const int dim;
    int nrays, nsamples;        // number of rays and samples per ray
    Array<real_t> ds;           // segment length for each ray
    const Vector *alpha;        // per-ray thickness design variables (live, owned by caller)
    
    Vector ray_residuals;       // residuals (A_i - α_i)
    
    public:
    ThicknessResidual(ParMesh &pmesh_, ParGridFunction &rho_filter_,
    Vector *ray_starts, Vector *ray_ends, int nrays_,
    const Vector &alpha_, int nsamples_=100)
    : fes(rho_filter_.ParFESpace()), rho_filter(&rho_filter_),
    dim(pmesh_.Dimension()), nrays(nrays_),
    nsamples(nsamples_), alpha(&alpha_), ray_residuals(nrays_)
    {
        finder.Setup(pmesh_);
        
        ds.SetSize(nrays);
        Vector xyz(dim * nsamples * nrays);
        
        // Setup sampling points for all rays
        for (int r = 0; r < nrays; r++)
        {
            Vector ray(dim);
            subtract(ray_ends[r], ray_starts[r], ray);
            real_t ray_len = ray.Norml2();
            ds[r] = ray_len / nsamples;        // segment length for this ray
            
            for (int i = 0; i < nsamples; i++)
            {
                const double t = (i + 0.5) / nsamples;   // midpoint of segment i
                const int idx = r * nsamples + i;
                for (int d = 0; d < dim; d++)
                {
                    xyz(d * nsamples * nrays + idx) = ray_starts[r](d) + t * ray(d);  // byNODES ordering
                }
            }
        }

        finder.FindPoints(xyz, Ordering::byNODES);

        // Get point location codes and elements
        code = finder.GetCode();
        proc = finder.GetProc();
        elem = finder.GetElem();
    }
    ~ThicknessResidual() { }

    // Compute total residual: ∑_i 1/2 (A_i - α_i)^2
    real_t Eval() override
    {
        finder.Interpolate(*rho_filter, interp_vals);

        real_t total_residual = 0.0;

        for (int r = 0; r < nrays; r++)
        {
            // Compute thickness A_i for ray i
            real_t thickness = 0.0;
            for (int i = 0; i < nsamples; i++)
            {
                const int idx = r * nsamples + i;
                if (code[idx] == 2) { continue; }   // skip points outside the mesh
                thickness += interp_vals(idx) * ds[r];  // midpoint rule
            }

            real_t res = thickness - (*alpha)(r);
            ray_residuals(r) = res;
            total_residual += res * res;
        }

        return 0.5 * total_residual;
    }

    // Gradient w.r.t. alpha: ∂R/∂α_i = - (A_i - α_i)
    void GetGrad(Vector &grad) override
    {
        grad.SetSize(nrays);
        grad = ray_residuals;
        grad.Neg();  // - (A_i - α_i)
    }

    // Sensitivity of R w.r.t. the FILTERED density
    //     ℓ = Σ_i (A_i − α_i) ds_i Σ_k φ(x_ik),
    // where φ(x_ik) are the filter shape functions sampling ρ̃ at ray point x_ik.
    void GetGradRHS(Vector &ell_tdof)
    {
        const int myid = fes->GetMyRank();
        const Vector &ref = finder.GetReferencePosition();   // byVDim, ref in [0,1]

        const int npts = code.Size();
        Array<int> owner(npts);
        for (int p = 0; p < npts; p++)
        {
            owner[p] = (code[p] != 2 && proc[p] == (unsigned)myid)
                       ? myid : std::numeric_limits<int>::max();
        }
        MPI_Allreduce(MPI_IN_PLACE, owner.GetData(), npts, MPI_INT, MPI_MIN,
                      fes->GetComm());

        Vector ell_ldof(fes->GetVSize());                   // L-vector accumulation
        ell_ldof = 0.0;

        Array<int> vdofs;
        Vector shape;
        IntegrationPoint ip;

        for (int r = 0; r < nrays; r++)
        {
            const real_t w = ray_residuals(r) * ds[r];     // (A_i − α_i) * ds_i
            for (int k = 0; k < nsamples; k++)
            {
                const int idx = r * nsamples + k;
                if (owner[idx] != myid) { continue; }              // unique global owner

                const int e = elem[idx];
                const FiniteElement *fe = fes->GetFE(e);
                const int ndof = fe->GetDof();
                shape.SetSize(ndof);

                if (dim == 2) { ip.Set2(ref(idx*dim+0), ref(idx*dim+1)); }
                else          { ip.Set3(ref(idx*dim+0), ref(idx*dim+1), ref(idx*dim+2)); }
                fe->CalcShape(ip, shape);

                fes->GetElementDofs(e, vdofs);
                for (int j = 0; j < ndof; j++)
                {
                    const int vd  = vdofs[j];
                    const int dof = (vd >= 0) ? vd : -1 - vd;
                    const real_t s = (vd >= 0) ? shape(j) : -shape(j);
                    ell_ldof(dof) += w * s;
                }
            }
        }

        // Reduce the local-dof functional to true dofs:  ℓ_true = P^T ℓ_ldof.
        ell_tdof.SetSize(fes->GetTrueVSize());
        fes->GetProlongationMatrix()->MultTranspose(ell_ldof, ell_tdof);
    }
};
#endif


class AdvectThicknessResidual : public QuantityOfInterest
{
private:
    MPI_Comm comm;
    
    ParFiniteElementSpace *sub_fes;  // DG space for rho_a_sub

    ParSubMesh      *submesh;    // outflow boundary submesh (borrowed)
    ParGridFunction *rho_a_full; // live full-DG forward field (borrowed from solver)
    ParGridFunction *rho_a_sub;  // outflow trace of rho_a (refreshed each eval)
    ParGridFunction *alpha;      // per-ray thickness design variables (live, owned by caller)

    void Refresh() { submesh->Transfer(*rho_a_full, *rho_a_sub); }

public:
    AdvectThicknessResidual(ParSubMesh &submesh_, ParGridFunction &rho_a_, ParGridFunction &alpha_)
    : comm(alpha_.ParFESpace()->GetComm()), sub_fes(alpha_.ParFESpace()),
        submesh(&submesh_), rho_a_full(&rho_a_),
        rho_a_sub(new ParGridFunction(sub_fes)), alpha(&alpha_)
    {
        Refresh();
    }
    ~AdvectThicknessResidual() { delete rho_a_sub; }

    // forward solve for rho_a over the pseudo-time interval, then integrate
    // the accumulated density over the outflow boundary
    real_t Eval() override
    {
        Refresh();
        ParGridFunction r(sub_fes);
        r  = *rho_a_sub;
        r -= *alpha;                   // r = rho_a - alpha

        ConstantCoefficient zero(0.0);
        const real_t rnorm = r.ComputeL2Error(zero);
        return 0.5 * rnorm * rnorm;
    }

    // dG/drho_a = (rho_a - alpha)_gamma_out
    // dG/dalpha = (alpha - rho_a)_gamma_out
    void GetGrad(Vector &grad_rho, Vector &grad_al) override
    {
        Refresh();
        ParGridFunction r(sub_fes);
        r  = *alpha;
        r -= *rho_a_sub;                // alpha - rho_a   on Gamma_out

        GridFunctionCoefficient r_cf(&r);
        ParLinearForm lf1(sub_fes);
        lf1.AddDomainIntegrator(new DomainLFIntegrator(r_cf));
        lf1.Assemble();                // M_Gamma (alpha - rho_a)
        std::unique_ptr<HypreParVector> dv(lf1.ParallelAssemble());

        grad_al.SetSize(dv->Size());
        grad_al = *dv;

        grad_rho.SetSize(dv->Size());
        grad_rho = *dv;
        grad_rho.Neg();                // - (alpha - rho_a) = (rho_a - alpha)
    }
};