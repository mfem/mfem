//                       MFEM Example 19 - Parallel Version
//
// Compile with: make ex19p
//
// Sample runs:
//    mpirun -np 2 ex19p -m ../data/beam-quad.mesh
//    mpirun -np 2 ex19p -m ../data/beam-tri.mesh
//    mpirun -np 2 ex19p -m ../data/beam-hex.mesh
//    mpirun -np 2 ex19p -m ../data/beam-tet.mesh
//    mpirun -np 2 ex19p -m ../data/beam-wedge.mesh
//
// Description:  This examples solves a quasi-static incompressible nonlinear
//               elasticity problem of the form 0 = H(x), where H is an
//               incompressible hyperelastic model and x is a block state vector
//               containing displacement and pressure variables. The geometry of
//               the domain is assumed to be as follows:
//
//                                 +---------------------+
//                    boundary --->|                     |<--- boundary
//                    attribute 1  |                     |     attribute 2
//                    (fixed)      +---------------------+     (fixed, nonzero)
//
//               The example demonstrates the use of block nonlinear operators
//               (the class RubberOperator defining H(x)) as well as a nonlinear
//               Newton solver for the quasi-static problem. Each Newton step
//               requires the inversion of a Jacobian matrix, which is done
//               through a (preconditioned) inner solver. The specialized block
//               preconditioner is implemented as a user-defined solver.
//
//               We recommend viewing examples 2, 5, and 10 before viewing this
//               example.

#include "mfem.hpp"
#include <memory>
#include <iostream>
#include <fstream>

namespace mfem {

class pLapIntegrator: public ADQIntegratorH
{
private:


    template<typename DType, typename MVType>
    DType MyQIntegrator(const mfem::Vector& vparam, MVType& uu)
    {
        double pp=vparam[0];
        double ee=vparam[1];
        double ff=vparam[2];

        DType  u=uu[3];
        DType  norm2=uu[0]*uu[0]+uu[1]*uu[1]+uu[2]*uu[2];

        DType rez= pow(ee*ee+norm2,pp/2.0)/pp-ff*u;
        return rez;
    }


public:
    pLapIntegrator(){}

    virtual ~pLapIntegrator(){}

    virtual double QIntegrator(const mfem::Vector &vparam,const mfem::Vector &uu) override
    {
        double rez=MyQIntegrator<double,const mfem::Vector>(vparam,uu);
        return rez;
    }

    virtual ADFType QIntegrator(const mfem::Vector &vparam, ADFVector& uu) override
    {
        ADFType rez=MyQIntegrator<ADFType,ADFVector>(vparam,uu);
        return rez;
    }

    virtual ADSType QIntegrator(const mfem::Vector &vparam, ADSVector& uu) override
    {
        ADSType rez=MyQIntegrator<ADSType,ADSVector>(vparam,uu);
        return rez;
    }

};

class pLaplaceH: public mfem::NonlinearFormIntegrator
{
protected:
    mfem::Coefficient* pp;
    mfem::Coefficient* coeff;
    mfem::Coefficient* load;

    pLapIntegrator qint;
public:
    pLaplaceH()
    {
        coeff=nullptr;
        pp=nullptr;
    }

    pLaplaceH(mfem::Coefficient& pp_):pp(&pp_), coeff(nullptr), load(nullptr)
    {

    }

    pLaplaceH(mfem::Coefficient &pp_,mfem::Coefficient& q, mfem::Coefficient& ld_): pp(&pp_), coeff(&q), load(&ld_)
    {

    }

    virtual ~pLaplaceH()
    {

    }

    virtual double GetElementEnergy(const mfem::FiniteElement &el, mfem::ElementTransformation &trans, const mfem::Vector &elfun) override
    {
        double energy=0.0;
        int ndof = el.GetDof();
        int ndim = el.GetDim();
        int spaceDim = trans.GetSpaceDim();
        bool square = (ndim == spaceDim);
        const mfem::IntegrationRule *ir = NULL;
        int order = 2 * trans.OrderGrad(&el) - 1; // correct order?
        ir = &mfem::IntRules.Get(el.GetGeomType(), order);

        mfem::Vector shapef(ndof);
        mfem::DenseMatrix dshape_iso(ndof,ndim);
        mfem::DenseMatrix dshape_xyz(ndof,spaceDim);
        mfem::Vector grad(spaceDim);

        mfem::Vector vparam(3);//[power, epsilon, load]
        mfem::Vector uu(4);//[diff_x,diff_y,diff_z,u]

        uu=0.0;
        vparam[0]=2.0;  //default power
        vparam[1]=1e-8; //default epsilon
        vparam[2]=1.0;  //default load

        double w;
        double detJ;

        for(int i = 0; i < ir -> GetNPoints(); i++)
        {
            const mfem::IntegrationPoint &ip = ir->IntPoint(i);
            trans.SetIntPoint(&ip);
            w = trans.Weight();
            detJ = (square ? w : w*w);
            w = ip.weight *w;

            el.CalcDShape(ip,dshape_iso);
            el.CalcShape(ip,shapef);
            // AdjugateJacobian = / adj(J),         if J is square
            //                    \ adj(J^t.J).J^t, otherwise
            mfem::Mult(dshape_iso, trans.AdjugateJacobian(), dshape_xyz);
            // dshape_xyz should be devided by detJ for obtaining the real value
            // calculate the gradient
            dshape_xyz.MultTranspose(elfun,grad);

            //set the power
            if(pp!=nullptr)
            {
                vparam[0]=pp->Eval(trans,ip);
            }

            //set the coefficient ensuring possitiveness of the tangent matrix
            if(coeff!=nullptr)
            {
                vparam[1]=coeff->Eval(trans,ip);
            }
            //add the contribution from the load
            if(load!=nullptr)
            {
                vparam[2]=load->Eval(trans,ip);
            }
            //fill the vector uu
            for(int jj=0;jj<spaceDim;jj++)
            {
                uu[jj]=grad[jj]/detJ;
            }
            uu[3]=shapef*elfun;

            energy = energy + w * (qint.QIntegrator(vparam,uu));

        }
        return energy;
    }

    virtual void AssembleElementVector(const mfem::FiniteElement & el,
                                       mfem::ElementTransformation & trans,
                                       const mfem::Vector & elfun,
                                       mfem::Vector & elvect) override
    {
            int ndof = el.GetDof();
            int ndim = el.GetDim();
            int spaceDim = trans.GetSpaceDim();
            bool square = (ndim == spaceDim);
            const mfem::IntegrationRule *ir = NULL;
            int order = 2 * trans.OrderGrad(&el) - 1; // correct order?
            ir = &mfem::IntRules.Get(el.GetGeomType(), order);

            mfem::Vector shapef(ndof);
            mfem::DenseMatrix dshape_iso(ndof,ndim);
            mfem::DenseMatrix dshape_xyz(ndof,spaceDim);
            mfem::Vector lvec(ndof);
            elvect.SetSize(ndof);
            elvect=0.0;

            mfem::DenseMatrix B(ndof,4); //[diff_x,diff_y,diff_z, shape]
            mfem::Vector vparam(3);//[power, epsilon, load]
            mfem::Vector uu(4);//[diff_x,diff_y,diff_z,u]
            mfem::Vector du(4);
            B=0.0;
            uu=0.0;
            //initialize the parameters - keep the same order
            //utilized in the pLapIntegrator definition
            vparam[0]=2.0;  //default power
            vparam[1]=1e-8; //default epsilon
            vparam[2]=1.0;  //default load

            double w;
            double detJ;

            for (int i = 0; i < ir -> GetNPoints(); i++)
            {
                const mfem::IntegrationPoint &ip = ir->IntPoint(i);
                trans.SetIntPoint(&ip);
                w = trans.Weight();
                detJ = (square ? w : w*w);
                w = ip.weight * w;

                el.CalcDShape(ip,dshape_iso);
                el.CalcShape(ip,shapef);
                mfem::Mult(dshape_iso, trans.InverseJacobian(), dshape_xyz);

                //set the matrix B
                for(int jj=0;jj<spaceDim;jj++)
                {
                    B.SetCol(jj,dshape_xyz.GetColumn(jj));
                }
                B.SetCol(3,shapef);


                //set the power
                if(pp!=nullptr)
                {
                    vparam[0]=pp->Eval(trans,ip);
                }
                //set the coefficient ensuring possitiveness of the tangent matrix
                if(coeff!=nullptr)
                {
                    vparam[1]=coeff->Eval(trans,ip);
                }
                //add the contribution from the load
                if(load!=nullptr)
                {
                    vparam[2]=load->Eval(trans,ip);
                }

                //calculate uu
                B.MultTranspose(elfun,uu);
                //calculate derivative of the energy with respect to uu
                qint.QIntegratorDU(vparam,uu,du);

                B.Mult(du,lvec);
                elvect.Add( w, lvec);
            }// end integration loop
    }

    virtual void AssembleElementGrad(const mfem::FiniteElement & el,
                                         mfem::ElementTransformation & trans,
                                         const mfem::Vector & elfun, mfem::DenseMatrix & elmat) override
    {
        int ndof = el.GetDof();
        int ndim = el.GetDim();
        int spaceDim = trans.GetSpaceDim();
        bool square = (ndim == spaceDim);
        const mfem::IntegrationRule *ir = NULL;
        int order = 2 * trans.OrderGrad(&el) - 1; // correct order?
        ir = &mfem::IntRules.Get(el.GetGeomType(), order);

        mfem::Vector shapef(ndof);
        mfem::DenseMatrix dshape_iso(ndof,ndim);
        mfem::DenseMatrix dshape_xyz(ndof,spaceDim);
        elmat.SetSize(ndof,ndof);
        elmat=0.0;

        mfem::DenseMatrix B(ndof,4); //[diff_x,diff_y,diff_z, shape]
        mfem::DenseMatrix A(ndof,4);
        mfem::Vector vparam(3);//[power, epsilon, load]
        mfem::Vector uu(4);//[diff_x,diff_y,diff_z,u]
        mfem::DenseMatrix duu(4,4);
        B=0.0;
        uu=0.0;
        //initialize the parameters - keep the same order
        //utilized in the pLapIntegrator definition
        vparam[0]=2.0;  //default power
        vparam[1]=1e-8; //default epsilon
        vparam[2]=1.0;  //default load

        double w;
        double detJ;

        for (int i = 0; i < ir -> GetNPoints(); i++)
        {
            const mfem::IntegrationPoint &ip = ir->IntPoint(i);
            trans.SetIntPoint(&ip);
            w = trans.Weight();
            detJ = (square ? w : w*w);
            w = ip.weight * w;

            el.CalcDShape(ip,dshape_iso);
            el.CalcShape(ip,shapef);
            mfem::Mult(dshape_iso, trans.InverseJacobian(), dshape_xyz);

            //set the matrix B
            for(int jj=0;jj<spaceDim;jj++)
            {
                B.SetCol(jj,dshape_xyz.GetColumn(jj));
            }
            B.SetCol(3,shapef);


            //set the power
            if(pp!=nullptr)
            {
                vparam[0]=pp->Eval(trans,ip);
            }
            //set the coefficient ensuring possitiveness of the tangent matrix
            if(coeff!=nullptr)
            {
                vparam[1]=coeff->Eval(trans,ip);
            }
            //add the contribution from the load
            if(load!=nullptr)
            {
                vparam[2]=load->Eval(trans,ip);
            }

            //calculate uu
            B.MultTranspose(elfun,uu);
            //calculate derivative of the energy with respect to uu
            qint.QIntegratorDD(vparam,uu,duu);

            mfem::Mult(B,duu,A);
            mfem::AddMult_a_ABt(w,A,B,elmat);

        }//end integration loop
    }


};


class pLaplace: public mfem::NonlinearFormIntegrator
{
protected:
    mfem::Coefficient* pp;
    mfem::Coefficient* coeff;
    mfem::Coefficient* load;
public:
    pLaplace()
    {
        coeff=nullptr;
        pp=nullptr;
    }

    pLaplace(mfem::Coefficient& pp_):pp(&pp_), coeff(nullptr), load(nullptr)
    {

    }

    pLaplace(mfem::Coefficient &pp_,mfem::Coefficient& q, mfem::Coefficient& ld_): pp(&pp_), coeff(&q), load(&ld_)
    {

    }

    virtual ~pLaplace()
    {

    }

    virtual double GetElementEnergy(const mfem::FiniteElement &el, mfem::ElementTransformation &trans, const mfem::Vector &elfun) override
    {
        double energy=0.0;
        int ndof = el.GetDof();
        int ndim = el.GetDim();
        int spaceDim = trans.GetSpaceDim();
        bool square = (ndim == spaceDim);
        const mfem::IntegrationRule *ir = NULL;
        int order = 2 * trans.OrderGrad(&el) - 1; // correct order?
        ir = &mfem::IntRules.Get(el.GetGeomType(), order);

        mfem::Vector shapef(ndof);
        mfem::DenseMatrix dshape_iso(ndof,ndim);
        mfem::DenseMatrix dshape_xyz(ndof,spaceDim);
        mfem::Vector grad(spaceDim);

        double w;
        double detJ;
        double nrgrad2;
        double ppp=2.0;
        double eee=0.0;

        for(int i = 0; i < ir -> GetNPoints(); i++)
        {
            const mfem::IntegrationPoint &ip = ir->IntPoint(i);
            trans.SetIntPoint(&ip);
            w = trans.Weight();
            detJ = (square ? w : w*w);
            w = ip.weight *w;

            el.CalcDShape(ip,dshape_iso);
            el.CalcShape(ip,shapef);
            // AdjugateJacobian = / adj(J),         if J is square
            //                    \ adj(J^t.J).J^t, otherwise
            mfem::Mult(dshape_iso, trans.AdjugateJacobian(), dshape_xyz);
            // dshape_xyz should be devided by detJ for obtaining the real value
            // calculate the gradient
            dshape_xyz.MultTranspose(elfun,grad);
            nrgrad2=grad*grad/(detJ*detJ);

            //set the power
            if(pp!=nullptr)
            {
                ppp=pp->Eval(trans,ip);
            }

            //set the coefficient ensuring possitiveness of the tangent matrix
            if(coeff!=nullptr)
            {
                eee=coeff->Eval(trans,ip);
            }

            energy = energy + w * std::pow( nrgrad2 + eee * eee , ppp / 2.0 ) / ppp;

            //add the contribution from the load
            if(load!=nullptr)
            {
                energy = energy - w *  (shapef*elfun) * load->Eval(trans,ip);
            }
        }
        return energy;
    }

    virtual void  AssembleElementVector(const mfem::FiniteElement & el,
                                                mfem::ElementTransformation & trans,
                                                const mfem::Vector & elfun,
                                                mfem::Vector & elvect) override
    {
            int ndof = el.GetDof();
            int ndim = el.GetDim();
            int spaceDim = trans.GetSpaceDim();
            bool square = (ndim == spaceDim);
            const mfem::IntegrationRule *ir = NULL;
            int order = 2 * trans.OrderGrad(&el) - 1; // correct order?
            ir = &mfem::IntRules.Get(el.GetGeomType(), order);

            mfem::Vector shapef(ndof);
            mfem::DenseMatrix dshape_iso(ndof,ndim);
            mfem::DenseMatrix dshape_xyz(ndof,spaceDim);
            mfem::Vector grad(spaceDim);
            mfem::Vector lvec(ndof);
            elvect.SetSize(ndof);
            elvect=0.0;

            double w;
            double detJ;
            double nrgrad;
            double aa;
            double ppp=2.0;
            double eee=0.0;

            for (int i = 0; i < ir -> GetNPoints(); i++)
            {
                const mfem::IntegrationPoint &ip = ir->IntPoint(i);
                trans.SetIntPoint(&ip);
                w = trans.Weight();
                detJ = (square ? w : w*w);
                w = ip.weight * w;//w;

                el.CalcDShape(ip,dshape_iso);
                el.CalcShape(ip,shapef);
                // AdjugateJacobian = / adj(J),         if J is square
                //                    \ adj(J^t.J).J^t, otherwise
                mfem::Mult(dshape_iso, trans.AdjugateJacobian(), dshape_xyz);
                // dshape_xyz should be devided by detJ for obtaining the real value

                //calculate the gradient
                dshape_xyz.MultTranspose(elfun,grad);
                nrgrad=grad.Norml2()/detJ;
                //grad is not scaled so far, i.e., grad=grad/detJ

                //set the power
                if(pp!=nullptr)
                {
                    ppp=pp->Eval(trans,ip);
                }

                //set the coefficient ensuring possitiveness of the tangent matrix
                if(coeff!=nullptr)
                {
                    eee=coeff->Eval(trans,ip);
                }

                aa = nrgrad * nrgrad + eee * eee;
                aa=std::pow( aa , ( ppp - 2.0 ) / 2.0 );
                dshape_xyz.Mult(grad,lvec);
                elvect.Add( w * aa / ( detJ * detJ ), lvec);


                //add loading
                if(load!=nullptr)
                {
                    elvect.Add(-w*load->Eval(trans,ip),shapef);
                }
            }// end integration loop
    }

    virtual void AssembleElementGrad(const mfem::FiniteElement & el,
                                         mfem::ElementTransformation & trans,
                                         const mfem::Vector & elfun, mfem::DenseMatrix & elmat) override
    {
        int ndof = el.GetDof();
        int ndim = el.GetDim();
        int spaceDim = trans.GetSpaceDim();
        bool square = (ndim == spaceDim);
        const mfem::IntegrationRule *ir = NULL;
        int order = 2 * trans.OrderGrad(&el) - 1; // correct order?
        ir = &mfem::IntRules.Get(el.GetGeomType(), order);

        mfem::DenseMatrix dshape_iso(ndof,ndim);
        mfem::DenseMatrix dshape_xyz(ndof,spaceDim);
        mfem::Vector grad(spaceDim);
        mfem::Vector lvec(ndof);
        elmat.SetSize(ndof,ndof);
        elmat=0.0;

        double w;
        double detJ;
        double nrgrad;
        double aa0;
        double aa1;
        double ppp=2.0;
        double eee=0.0;

        for (int i = 0; i < ir -> GetNPoints(); i++)
        {
            const mfem::IntegrationPoint &ip = ir->IntPoint(i);
            trans.SetIntPoint(&ip);
            w = trans.Weight();
            detJ = (square ? w : w*w);
            w = ip.weight * w;

            el.CalcDShape(ip,dshape_iso);
            // AdjugateJacobian = / adj(J),         if J is square
            //                    \ adj(J^t.J).J^t, otherwise
            mfem::Mult(dshape_iso, trans.AdjugateJacobian(), dshape_xyz);
            // dshape_xyz should be devided by detJ for obtaining the real value
            // grad is not scaled so far,i.e., grad=grad/detJ

            //set the power
            if(pp!=nullptr)
            {
                ppp=pp->Eval(trans,ip);
            }
            //set the coefficient ensuring possitiveness of the tangent matrix
            if(coeff!=nullptr)
            {
                eee=coeff->Eval(trans,ip);
            }

            //calculate the gradient
            dshape_xyz.MultTranspose(elfun,grad);
            nrgrad = grad.Norml2() / detJ;
            aa0 = nrgrad * nrgrad + eee * eee;
            aa1 = std::pow( aa0 , ( ppp - 2.0 ) / 2.0 );
            aa0 = ( ppp - 2.0 ) * std::pow(aa0, ( ppp - 4.0 ) / 2.0 );
            dshape_xyz.Mult(grad,lvec);
            w = w / ( detJ * detJ );
            mfem::AddMult_a_VVt( w * aa0 / ( detJ * detJ ), lvec, elmat);
            mfem::AddMult_a_AAt( w * aa1 , dshape_xyz, elmat);

        }//end integration loop
    }
};

}


int main(int argc, char *argv[])
{
   // 1. Initialize MPI
   int num_procs, myrank;
   MPI_Init(&argc, &argv);
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
   MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

   // 2. Parse command-line options
   const char *mesh_file = "../data/beam-tet.mesh";
   int ser_ref_levels = 0;
   int par_ref_levels = 0;
   int order = 2;
   bool visualization = true;
   double newton_rel_tol = 1e-4;
   double newton_abs_tol = 1e-6;
   int newton_iter = 500;
   int print_level = 0;
   double pp = 2.0;
   int integrator=0;
   mfem::StopWatch* timer=new mfem::StopWatch();

   mfem::OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&ser_ref_levels, "-rs", "--refine-serial",
                  "Number of times to refine the mesh uniformly in serial.");
   args.AddOption(&par_ref_levels, "-rp", "--refine-parallel",
                  "Number of times to refine the mesh uniformly in parallel.");
   args.AddOption(&order, "-o", "--order",
                  "Order (degree) of the finite elements.");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&newton_rel_tol, "-rel", "--relative-tolerance",
                  "Relative tolerance for the Newton solve.");
   args.AddOption(&newton_abs_tol, "-abs", "--absolute-tolerance",
                  "Absolute tolerance for the Newton solve.");
   args.AddOption(&newton_iter, "-it", "--newton-iterations",
                  "Maximum iterations for the Newton solve.");
   args.AddOption(&pp, "-pp", "--power-parameter",
                  "Power parameter (>=2.0) for the p-Laplacian.");
   args.AddOption((&print_level),"-prt","--print-level",
                  "Print level.");
   args.AddOption(&integrator, "-int","--integrator",
                  "Integrator 0: standard; 1: AD uaing energy; 2: AD using gradients");

   args.Parse();
   if (!args.Good())
   {
      if (myrank == 0)
      {
         args.PrintUsage(std::cout);
      }
      MPI_Finalize();
      return 1;
   }
   if (myrank == 0)
   {
      args.PrintOptions(std::cout);
   }

   // 3. Read the (serial) mesh from the given mesh file on all processors.  We
   //    can handle triangular, quadrilateral, tetrahedral and hexahedral meshes
   //    with the same code.
   mfem::Mesh *mesh = new mfem::Mesh(mesh_file, 1, 1);
   int dim = mesh->Dimension();

   // 4. Refine the mesh in serial to increase the resolution. In this example
   //    we do 'ser_ref_levels' of uniform refinement, where 'ser_ref_levels' is
   //    a command-line parameter.
   for (int lev = 0; lev < ser_ref_levels; lev++)
   {
      mesh->UniformRefinement();
   }

   // 5. Define a parallel mesh by a partitioning of the serial mesh. Refine
   //    this mesh further in parallel to increase the resolution. Once the
   //    parallel mesh is defined, the serial mesh can be deleted.
   mfem::ParMesh *pmesh = new mfem::ParMesh(MPI_COMM_WORLD, *mesh);
   delete mesh;
   for (int lev = 0; lev < par_ref_levels; lev++)
   {
      pmesh->UniformRefinement();
   }

   // 6. Define the power parameter for the p-Laplacian and all other
   //    coefficients
   mfem::ConstantCoefficient c_pp(pp);
   mfem::ConstantCoefficient load(1.000000000);
   mfem::ConstantCoefficient c_ee(0.000000001);

   // 7. Define the finite element spaces for the solution
   mfem::H1_FECollection fec(order,dim);
   mfem::ParFiniteElementSpace fespace(pmesh,&fec,1,mfem::Ordering::byVDIM);
   HYPRE_Int glob_size=fespace.GlobalTrueVSize();
   if (myrank == 0)
   {
        std::cout << "Number of finite element unknowns: " << glob_size << std::endl;
   }

   // 8. Define the Dirichlet conditions
   mfem::Array<int> ess_bdr(pmesh->bdr_attributes.Max());
   ess_bdr = 1;

   // 9. Define the nonlinear form
   mfem::ParNonlinearForm* nf=new mfem::ParNonlinearForm(&fespace);

   // 10. Define the solution vector x as a parallel finite element grid function
   //     corresponding to fespace. Initialize x with initial guess of zero,
   //     which satisfies the boundary conditions.
   mfem::ParGridFunction x(&fespace);
   x = 0.0;
   mfem::HypreParVector* tv=x.GetTrueDofs();
   mfem::HypreParVector* sv=x.GetTrueDofs();

   // 11. Define ParaView DataCollection
   mfem::ParaViewDataCollection *dacol=new mfem::ParaViewDataCollection("pLap",pmesh);
   dacol->SetLevelsOfDetail(order);
   dacol->RegisterField("sol",&x);


   // 11. Set domain integrators - start with linear diffusion
   {
       // the default power coefficient is 2.0
       mfem::ConstantCoefficient lpp(2.0);
       if(integrator==0)
       {
            nf->AddDomainIntegrator(new mfem::pLaplace(lpp,c_ee,load));
       }else
       if(integrator==1)
       {
           nf->AddDomainIntegrator(new mfem::pLaplaceH(lpp,c_ee,load));
       }
       nf->SetEssentialBC(ess_bdr);
       // compute the energy
       double energy=nf->GetEnergy(*tv);
       if(myrank==0){
           std::cout<<"[2] The total energy of the system is E="<<energy<<std::endl;}
       // time the assembly
       timer->Clear();
       timer->Start();
       mfem::Operator &op=nf->GetGradient(*sv);
       timer->Stop();
       if(myrank==0){
           std::cout<<"[2] The assembly time is: "<<timer->RealTime()<<std::endl;}
       mfem::Solver *prec=new mfem::HypreBoomerAMG();
       mfem::GMRESSolver *j_gmres = new mfem::GMRESSolver(MPI_COMM_WORLD);
       j_gmres->SetRelTol(1e-7);
       j_gmres->SetAbsTol(1e-15);
       j_gmres->SetMaxIter(300);
       j_gmres->SetPrintLevel(print_level);
       j_gmres->SetPreconditioner(*prec);

       mfem::NewtonSolver* ns;
       ns=new mfem::NewtonSolver(MPI_COMM_WORLD);
       ns->iterative_mode = true;
       ns->SetSolver(*j_gmres);
       ns->SetOperator(*nf);
       ns->SetPrintLevel(print_level);
       ns->SetRelTol(1e-6);
       ns->SetAbsTol(1e-12);
       ns->SetMaxIter(3);
       //solve the problem
       timer->Clear();
       timer->Start();
       ns->Mult(*tv, *sv);
       timer->Stop();
       if(myrank==0){
           std::cout<<"Time for the NewtonSolver: "<<timer->RealTime()<<std::endl;}

       energy=nf->GetEnergy(*sv);
       if(myrank==0){
                 std::cout<<"[pp=2] The total energy of the system is E="<<energy<<std::endl;}

       delete ns;
       delete j_gmres;
       delete prec;

       x.SetFromTrueDofs(*sv);
       dacol->SetTime(2.0);
       dacol->SetCycle(2);
       dacol->Save();
   }

   // 12. Continue with powers higher than 2
   for(int i=3;i<pp;i++)
   {
       delete nf;
       nf=new mfem::ParNonlinearForm(&fespace);
       mfem::ConstantCoefficient lpp((double)i);
       if(integrator==0)
       {
            nf->AddDomainIntegrator(new mfem::pLaplace(lpp,c_ee,load));
       }else
       if(integrator==1)
       {
           nf->AddDomainIntegrator(new mfem::pLaplaceH(lpp,c_ee,load));
       }
       nf->SetEssentialBC(ess_bdr);
       // compute the energy
       double energy=nf->GetEnergy(*sv);
       if(myrank==0){
           std::cout<<"[pp="<<i<<"] The total energy of the system is E="<<energy<<std::endl;}
       // time the assembly
       timer->Clear();
       timer->Start();
       mfem::Operator &op=nf->GetGradient(*sv);
       timer->Stop();
       if(myrank==0){
           std::cout<<"[pp="<<i<<"] The assembly time is: "<<timer->RealTime()<<std::endl;}
       mfem::Solver *prec=new mfem::HypreBoomerAMG();
       mfem::GMRESSolver *j_gmres = new mfem::GMRESSolver(MPI_COMM_WORLD);
       j_gmres->SetRelTol(1e-7);
       j_gmres->SetAbsTol(1e-15);
       j_gmres->SetMaxIter(300);
       j_gmres->SetPrintLevel(print_level);
       j_gmres->SetPreconditioner(*prec);

       mfem::NewtonSolver* ns;
       ns=new mfem::NewtonSolver(MPI_COMM_WORLD);
       ns->iterative_mode = true;
       ns->SetSolver(*j_gmres);
       ns->SetOperator(*nf);
       ns->SetPrintLevel(print_level);
       ns->SetRelTol(1e-6);
       ns->SetAbsTol(1e-12);
       ns->SetMaxIter(3);
       //solve the problem
       timer->Clear();
       timer->Start();
       ns->Mult(*tv, *sv);
       timer->Stop();
       if(myrank==0){
           std::cout<<"Time for the NewtonSolver: "<<timer->RealTime()<<std::endl;}

       energy=nf->GetEnergy(*sv);
       if(myrank==0){
                 std::cout<<"[pp="<<i<<"] The total energy of the system is E="<<energy<<std::endl;}

       delete ns;
       delete j_gmres;
       delete prec;

       x.SetFromTrueDofs(*sv);
       dacol->SetTime(i);
       dacol->SetCycle(i);
       dacol->Save();
   }

   // 13. Continue with the final power
   if( std::abs(pp-2.0) > std::numeric_limits<double>::epsilon())
   {
       delete nf;
       nf=new mfem::ParNonlinearForm(&fespace);
       if(integrator==0)
       {
            nf->AddDomainIntegrator(new mfem::pLaplace(c_pp,c_ee,load));
       }else
       if(integrator==1)
       {
           nf->AddDomainIntegrator(new mfem::pLaplaceH(c_pp,c_ee,load));
       }
       nf->SetEssentialBC(ess_bdr);
       // compute the energy
       double energy=nf->GetEnergy(*sv);
       if(myrank==0){
           std::cout<<"[pp="<<pp<<"] The total energy of the system is E="<<energy<<std::endl;}
       // time the assembly
       timer->Clear();
       timer->Start();
       mfem::Operator &op=nf->GetGradient(*sv);
       timer->Stop();
       if(myrank==0){
           std::cout<<"[pp="<<pp<<"] The assembly time is: "<<timer->RealTime()<<std::endl;}
       mfem::Solver *prec=new mfem::HypreBoomerAMG();
       mfem::GMRESSolver *j_gmres = new mfem::GMRESSolver(MPI_COMM_WORLD);
       j_gmres->SetRelTol(1e-8);
       j_gmres->SetAbsTol(1e-15);
       j_gmres->SetMaxIter(300);
       j_gmres->SetPrintLevel(print_level);
       j_gmres->SetPreconditioner(*prec);

       mfem::NewtonSolver* ns;
       ns=new mfem::NewtonSolver(MPI_COMM_WORLD);
       ns->iterative_mode = true;
       ns->SetSolver(*j_gmres);
       ns->SetOperator(*nf);
       ns->SetPrintLevel(print_level);
       ns->SetRelTol(1e-6);
       ns->SetAbsTol(1e-12);
       ns->SetMaxIter(3);
       //solve the problem
       timer->Clear();
       timer->Start();
       ns->Mult(*tv, *sv);
       timer->Stop();
       if(myrank==0){
           std::cout<<"Time for the NewtonSolver: "<<timer->RealTime()<<std::endl;}

       energy=nf->GetEnergy(*sv);
       if(myrank==0){
                 std::cout<<"[pp="<<pp<<"] The total energy of the system is E="<<energy<<std::endl;}

       delete ns;
       delete j_gmres;
       delete prec;

       x.SetFromTrueDofs(*sv);
       dacol->SetTime(pp);
       if(pp<2.0)
       {
           dacol->SetCycle(std::floor(pp));
       }
       else
       {
           dacol->SetCycle(std::ceil(pp));
       }
       dacol->Save();
   }



   // 19. Free the used memory
   delete dacol;
   delete sv;
   delete tv;
   delete nf;
   delete pmesh;
   delete timer;

   MPI_Finalize();

   return 0;
}


