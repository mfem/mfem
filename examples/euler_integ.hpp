#ifndef MFEM_EULER_INTEG
#define MFEM_EULER_INTEG
#include "mfem.hpp"
using namespace mfem;
using namespace std;
#include "euler_fluxes.hpp"
namespace mfem
{

    template <int dim>
    class EulerDomainIntegrator : public NonlinearFormIntegrator
    {
    public:
        EulerDomainIntegrator(int num_state, double a = 1.0) : num_states(num_state), alpha(a) {}

        /// Euler flux function in a given (scaled) direction
        /// \param[in] dir - direction in which the flux is desired
        /// \param[in] q - conservative variables
        /// \param[out] flux - fluxes in the direction `dir`
        void calcFlux(const mfem::Vector &dir, const mfem::Vector &q,
                      mfem::Vector &flux)
        {
            calcEulerFlux<dim>(dir.GetData(), q.GetData(), flux.GetData());
        }

        /// Construct the element local residual
        /// \param[in] el - the finite element whose residual we want
        /// \param[in] trans - defines the reference to physical element mapping
        /// \param[in] elfun - element local state function
        /// \param[out] elvect - element local residual
        virtual void AssembleElementVector(const mfem::FiniteElement &el,
                                           mfem::ElementTransformation &trans,
                                           const mfem::Vector &elfun,
                                           mfem::Vector &elvect)
        {
            using namespace mfem;
            using namespace std;
            const int num_nodes = el.GetDof();
            //int dim = el.GetDim();
            elvect.SetSize(num_states * num_nodes);
            elvect = 0.0;
            DenseMatrix u_mat(elfun.GetData(), num_nodes, num_states);
            DenseMatrix res(elvect.GetData(), num_nodes, num_states);
            DenseMatrix adjJ_i, elflux, dshape, dshapedx;
            Vector shape, dxidx, dshapedxi, fluxi, u;
            u.SetSize(num_states);
            dxidx.SetSize(dim);
            fluxi.SetSize(num_states);
            dshapedxi.SetSize(num_nodes);
            shape.SetSize(num_nodes);
            dshape.SetSize(num_nodes, dim);
            dshapedx.SetSize(num_nodes, dim);
            elflux.SetSize(num_states, dim);
            adjJ_i.SetSize(dim);
            int intorder = trans.OrderGrad(&el) + trans.Order() + el.GetOrder();
            const IntegrationRule *ir = IntRule;
            if (ir == NULL)
            {
                ir = &IntRules.Get(el.GetGeomType(), intorder);
            }

            for (int i = 0; i < ir->GetNPoints(); i++)
            {
                const IntegrationPoint &ip = ir->IntPoint(i);
                trans.SetIntPoint(&ip);
                // Calculate the shape function
                el.CalcShape(ip, shape);
                // Compute the physical gradient
                el.CalcDShape(ip, dshape);
                // Mult(dshape, trans.AdjugateJacobian(), dshapedx);
                u_mat.MultTranspose(shape, u);
                CalcAdjugate(trans.Jacobian(), adjJ_i);
                for (int di = 0; di < dim; ++di)
                {
                    adjJ_i.GetRow(di, dxidx);
                    calcFlux(dxidx, u, fluxi);
                    dshape.GetColumn(di, dshapedxi);
                    AddMult_a_VWt(-ip.weight, dshapedxi, fluxi, res);
                }
            }
            res *= alpha;
        }

        void AssembleElementGrad(
            const mfem::FiniteElement &el, mfem::ElementTransformation &trans,
            const mfem::Vector &elfun, mfem::DenseMatrix &elmat)
        {
            int num_nodes = el.GetDof();
            int ndof = elfun.Size();
            elmat.SetSize(ndof);
            elmat = 0.0;
            double delta = 1e-6;
            for (int i = 0; i < ndof; ++i)
            {
                Vector elfun_plus(elfun);
                Vector elfun_minus(elfun);
                elfun_plus(i) += delta;
                Vector elvect_plus;
                AssembleElementVector(el, trans, elfun_plus, elvect_plus);
                elfun_minus(i) -= delta;
                Vector elvect_minus;
                AssembleElementVector(el, trans, elfun_minus, elvect_minus);

                elvect_plus -= elvect_minus;
                elvect_plus /= 2.0 * delta;

                for (int j = 0; j < ndof; ++j)
                {
                    elmat(j, i) = elvect_plus(j);
                }
            }
        }
        virtual double GetElementEnergy(const FiniteElement &el,
                                        ElementTransformation &Ttr,
                                        const Vector &elfun)
        {

            double energy;

            energy = 0.0;

            return energy;
        }

    protected:
        /// number of states
        int num_states;
        /// scales the terms; can be used to move to rhs/lhs
        double alpha;
#ifndef MFEM_THREAD_SAFE
        /// the coordinates of node i
        mfem::Vector x_i;
        /// used to reference the states at node i
        mfem::Vector ui;
        /// used to reference the residual at node i
        mfem::Vector resi;
        /// stores a row of the adjugate of the mapping Jacobian
        mfem::Vector dxidx;
        /// stores the result of calling the flux function
        mfem::Vector fluxi;
        /// used to store the adjugate of the mapping Jacobian at node i
        mfem::DenseMatrix adjJ_i;
        /// used to store the flux Jacobian at node i
        mfem::DenseMatrix flux_jaci;
        /// used to store the flux at each node
        mfem::DenseMatrix elflux;
        /// used to store the residual in (num_states, Dof) format
        mfem::DenseMatrix elres;
#endif
    };

    /// Integrator for inviscid boundary fluxes
    template <int dim, int bndinteg, bool entvar = false>
    class EulerBoundaryIntegrator : public NonlinearFormIntegrator
    {
    public:
        /// Constructs an integrator for isentropic vortex boundary flux
        /// \param[in] diff_stack - for algorithmic differentiation
        /// \param[in] fe_coll - used to determine the face elements
        /// \param[in] a - used to move residual to lhs (1.0) or rhs(-1.0)
        EulerBoundaryIntegrator(const mfem::FiniteElementCollection *fe_coll, int num_state, mfem::Vector &qf,
                                double a = 1.0)
            : fec(fe_coll), num_states(num_state), qfs(qf), alpha(a), work_vec(dim + 2) {}

        /// Compute a characteristic boundary flux for the isentropic vortex
        /// \param[in] x - coordinate location at which flux is evaluated
        /// \param[in] dir - vector normal to the boundary at `x`
        /// \param[in] q - conservative variables at which to evaluate the flux
        /// \param[out] flux_vec - value of the flux
        void calcFlux(const mfem::Vector &x, const mfem::Vector &dir,
                      const mfem::Vector &q, mfem::Vector &flux_vec)
        {
            if (bndinteg == 1)
            {
                calcIsentropicVortexFlux<entvar>(x.GetData(), dir.GetData(),
                                                 q.GetData(), flux_vec.GetData());
            }
            else if (bndinteg == 2)
            {
                calcSlipWallFlux<dim, entvar>(x.GetData(), dir.GetData(),
                                              q.GetData(), flux_vec.GetData());
            }
            else
            {
                calcFarFieldFlux<dim, entvar>(dir.GetData(), qfs.GetData(),
                                              q.GetData(), work_vec.GetData(),
                                              flux_vec.GetData());
            }
        }

        /// Construct the contribution to the element local residual
        /// \param[in] el_bnd - the finite element whose residual we want to update
        /// \param[in] el_unused - dummy element that is not used for boundaries
        /// \param[in] trans - holds geometry and mapping information about the face
        /// \param[in] elfun - element local state function
        /// \param[out] elvect - element local residual
        virtual void AssembleFaceVector(const mfem::FiniteElement &el_bnd,
                                        const mfem::FiniteElement &el_unused,
                                        mfem::FaceElementTransformations &trans,
                                        const mfem::Vector &elfun,
                                        mfem::Vector &elvect)
        {
            // using namespace mfem;
            const int dof = el_bnd.GetDof();
#ifdef MFEM_THREAD_SAFE
            Vector u_face, x, nrm, flux_face, shape;
#endif
            // int dim = el_bnd.GetDim();
            u_face.SetSize(num_states);
            x.SetSize(dim);
            nrm.SetSize(dim);
            flux_face.SetSize(num_states);
            elvect.SetSize(num_states * dof);
            elvect = 0.0;
            shape.SetSize(dof);
            DenseMatrix u(elfun.GetData(), dof, num_states);
            DenseMatrix res(elvect.GetData(), dof, num_states);
            int intorder;
            intorder = trans.Elem1->OrderW() + 2 * el_bnd.GetOrder();
            const IntegrationRule *ir = &IntRules.Get(trans.FaceGeom, intorder);
            IntegrationPoint eip1;
            for (int i = 0; i < ir->GetNPoints(); i++)
            {
                const IntegrationPoint &ip = ir->IntPoint(i);
                trans.Loc1.Transform(ip, eip1);
                trans.Elem1->Transform(eip1, x);
                el_bnd.CalcShape(eip1, shape);
                // get the normal vector and the flux on the face
                trans.Face->SetIntPoint(&ip);
                CalcOrtho(trans.Face->Jacobian(), nrm);
                // Interpolate elfun at the point
                u.MultTranspose(shape, u_face);
                calcFlux(x, nrm, u_face, flux_face);
                flux_face *= ip.weight;
                // multiply by test function
                for (int n = 0; n < num_states; ++n)
                {
                    for (int s = 0; s < dof; s++)
                    {
                        res(s, n) += shape(s) * flux_face(n);
                    }
                }
            }
            res *= alpha;
        }

        void AssembleFaceGrad(
            const mfem::FiniteElement &el_bnd,
            const mfem::FiniteElement &el_unused,
            mfem::FaceElementTransformations &trans,
            const mfem::Vector &elfun,
            mfem::DenseMatrix &elmat)
        {
            int ndof = elfun.Size();
            elmat.SetSize(ndof);
            elmat = 0.0;
            double delta = 1e-6;
            for (int i = 0; i < ndof; ++i)
            {
                Vector elfun_plus(elfun);
                Vector elfun_minus(elfun);
                elfun_plus(i) += delta;
                Vector elvect_plus;
                AssembleFaceVector(el_bnd, el_unused, trans, elfun_plus, elvect_plus);
                elfun_minus(i) -= delta;
                Vector elvect_minus;
                AssembleFaceVector(el_bnd, el_unused, trans, elfun_minus, elvect_minus);

                elvect_plus -= elvect_minus;
                elvect_plus /= 2 * delta;

                for (int j = 0; j < ndof; ++j)
                {
                    elmat(j, i) = elvect_plus(j);
                }
            }
        }

        double calcBndryFun(
            const mfem::Vector &x, const mfem::Vector &dir,
            const mfem::Vector &q)
        {
            mfem::Vector flux_vec(q.Size());
            calcFlux(x, dir, q, flux_vec);
            mfem::Vector w(q.Size());
            if (entvar)
            {
                w = q;
            }
            else
            {
                calcEntropyVars<dim>(q.GetData(), w.GetData());
            }
            return w * flux_vec;
        }

        double GetFaceEnergy(
            const mfem::FiniteElement &el_bnd,
            const mfem::FiniteElement &el_unused,
            mfem::FaceElementTransformations &trans,
            const mfem::Vector &elfun)
        {
            const int num_nodes = el_bnd.GetDof();
#ifdef MFEM_THREAD_SAFE
            Vector u_face, x, nrm, flux_face;
#endif
            u_face.SetSize(num_states);
            x.SetSize(dim);
            nrm.SetSize(dim);
            shape.SetSize(num_nodes);
            double fun = 0.0; // initialize the functional value
            DenseMatrix u(elfun.GetData(), num_nodes, num_states);

            int intorder;
            intorder = trans.Elem1->OrderW() + 2 * el_bnd.GetOrder();
            const IntegrationRule *ir = &IntRules.Get(trans.FaceGeom, intorder);
            IntegrationPoint el_ip;
            for (int i = 0; i < ir->GetNPoints(); i++)
            {
                const IntegrationPoint &face_ip = ir->IntPoint(i);
                trans.Loc1.Transform(face_ip, el_ip);
                trans.Elem1->Transform(el_ip, x);
                el_bnd.CalcShape(el_ip, shape);
                u.MultTranspose(shape, u_face);
                // get the normal vector, and then add contribution to function
                trans.Face->SetIntPoint(&face_ip);
                CalcOrtho(trans.Face->Jacobian(), nrm);
                fun += calcBndryFun(x, nrm, u_face) * face_ip.weight * alpha;
            }
            return fun;
        }

    protected:
        /// number of states
        int num_states;
        /// scales the terms; can be used to move to rhs/lhs
        double alpha;
        /// used to select the appropriate face element
        const mfem::FiniteElementCollection *fec;
#ifndef MFEM_THREAD_SAFE
        /// used to reference the state at face node
        mfem::Vector u_face;
        /// store the physical location of a node
        mfem::Vector x;
        /// farfield state value
        mfem::Vector qfs;
        /// work vector
        mfem::Vector work_vec;
        /// the outward pointing (scaled) normal to the boundary at a node
        mfem::Vector nrm;
        mfem::Vector shape;
        /// stores the flux evaluated by `bnd_flux`
        mfem::Vector flux_face;
        /// stores the jacobian of the flux with respect to the state at `u_face`
        mfem::DenseMatrix flux_jac_face;

#endif
    };

    /// Integrator for inviscid interface fluxes (fluxes that do not need gradient)
    /// \tparam Derived - a class Derived from this one (needed for CRTP)
    template <int dim>
    class EulerFaceIntegrator : public NonlinearFormIntegrator
    {
    public:
        /// Constructs a face integrator based on a given interface flux
        /// \param[in] diff_stack - for algorithmic differentiation
        /// \param[in] fe_coll - used to determine the face elements
        /// \param[in] num_state_vars - the number of state variables
        /// \param[in] a - used to move residual to lhs (1.0) or rhs(-1.0)
        EulerFaceIntegrator(const mfem::FiniteElementCollection *fe_coll,
                            double coeff = 1.0, int num_state_vars = 1, double a = 1.0)
            : diss_coeff(coeff), num_states(num_state_vars), alpha(a), fec(fe_coll) {}

        /// Construct the contribution to the element local residuals
        /// \param[in] el_left - "left" element whose residual we want to update
        /// \param[in] el_right - "right" element whose residual we want to update
        /// \param[in] trans - holds geometry and mapping information about the face
        /// \param[in] elfun - element local state function
        /// \param[out] elvect - element local residual
        virtual void AssembleFaceVector(const mfem::FiniteElement &el_left,
                                        const mfem::FiniteElement &el_right,
                                        mfem::FaceElementTransformations &trans,
                                        const mfem::Vector &elfun,
                                        mfem::Vector &elvect)
        {
            // using namespace mfem;
#ifdef MFEM_THREAD_SAFE
            Vector shape1, shape2, funval1, funval2, nrm, fluxN;
#endif
            // Compute the term <F.n(u),[w]> on the interior faces.
            const int dof1 = el_left.GetDof();
            const int dof2 = el_right.GetDof();
            //int dim = el_left.GetDim();
            nrm.SetSize(dim);
            shape1.SetSize(dof1);
            shape2.SetSize(dof2);
            funval1.SetSize(num_states);
            funval2.SetSize(num_states);
            fluxN.SetSize(num_states);
            elvect.SetSize((dof1 + dof2) * num_states);
            elvect = 0.0;

            DenseMatrix elfun1_mat(elfun.GetData(), dof1, num_states);
            DenseMatrix elfun2_mat(elfun.GetData() + dof1 * num_states, dof2,
                                   num_states);

            DenseMatrix elvect1_mat(elvect.GetData(), dof1, num_states);
            DenseMatrix elvect2_mat(elvect.GetData() + dof1 * num_states, dof2,
                                    num_states);

            // Integration order calculation from DGTraceIntegrator
            int intorder;
            if (trans.Elem2No >= 0)
                intorder = (min(trans.Elem1->OrderW(), trans.Elem2->OrderW()) +
                            2 * max(el_left.GetOrder(), el_right.GetOrder()));
            else
            {
                intorder = trans.Elem1->OrderW() + 2 * el_left.GetOrder();
            }

            const IntegrationRule *ir = &IntRules.Get(trans.FaceGeom, intorder);
            //cout << "face elements are " << trans.Elem1No << " , " << trans.Elem2No << endl;
            for (int i = 0; i < ir->GetNPoints(); i++)
            {
                const IntegrationPoint &ip = ir->IntPoint(i);
                IntegrationPoint eip1;
                IntegrationPoint eip2;
                trans.Loc1.Transform(ip, eip1);
                trans.Loc2.Transform(ip, eip2);

                // Calculate basis functions on both elements at the face
                el_left.CalcShape(eip1, shape1);
                el_right.CalcShape(eip2, shape2);

                // Interpolate elfun at the point
                elfun1_mat.MultTranspose(shape1, funval1);
                elfun2_mat.MultTranspose(shape2, funval2);

                trans.Face->SetIntPoint(&ip);

                // Get the normal vector and the flux on the face
                CalcOrtho(trans.Face->Jacobian(), nrm);

                flux(nrm, funval1, funval2, fluxN);

                fluxN *= ip.weight;
                for (int k = 0; k < num_states; k++)
                {
                    for (int s = 0; s < dof1; s++)
                    {
                        elvect1_mat(s, k) += fluxN(k) * shape1(s);
                    }
                    for (int s = 0; s < dof2; s++)
                    {
                        elvect2_mat(s, k) -= fluxN(k) * shape2(s);
                    }
                }
                elvect *= alpha;
            }
        }
        void AssembleFaceGrad(
            const mfem::FiniteElement &el_left,
            const mfem::FiniteElement &el_right,
            mfem::FaceElementTransformations &trans,
            const mfem::Vector &elfun,
            mfem::DenseMatrix &elmat)
        {
            int ndof = elfun.Size();
            elmat.SetSize(ndof);
            elmat = 0.0;
            double delta = 1e-6;
            for (int i = 0; i < ndof; ++i)
            {
                Vector elfun_plus(elfun);
                Vector elfun_minus(elfun);
                elfun_plus(i) += delta;
                Vector elvect_plus;
                AssembleFaceVector(el_left, el_right, trans, elfun_plus, elvect_plus);
                elfun_minus(i) -= delta;
                Vector elvect_minus;
                AssembleFaceVector(el_left, el_right, trans, elfun_minus, elvect_minus);

                elvect_plus -= elvect_minus;
                elvect_plus /= 2 * delta;

                for (int j = 0; j < ndof; ++j)
                {
                    elmat(j, i) = elvect_plus(j);
                }
            }
        }
        double GetFaceEnergy(
            const mfem::FiniteElement &el_bnd,
            const mfem::FiniteElement &el_unused,
            mfem::FaceElementTransformations &trans,
            const mfem::Vector &elfun)
        {
            return 0.0;
        }

    protected:
        /// number of states
        int num_states;
        /// scales the terms; can be used to move to rhs/lhs
        double alpha;
        /// dissipation coefficient
        double diss_coeff;
        /// used to select the appropriate face element
        const mfem::FiniteElementCollection *fec;
#ifndef MFEM_THREAD_SAFE
        /// used to reference the left state at face node
        mfem::Vector u_face_left;
        /// used to reference the right state at face node
        mfem::Vector u_face_right;
        /// the outward pointing (scaled) normal to the boundary at a node
        mfem::Vector nrm;
        /// stores the flux evaluated by `bnd_flux`
        mfem::Vector flux_face;
        mfem::Vector shape1, shape2, funval1, funval2, fluxN;
        /// stores the jacobian of the flux with respect to the left state
        mfem::DenseMatrix flux_jac_left;
        /// stores the jacobian of the flux with respect to the right state
        mfem::DenseMatrix flux_jac_right;
#endif

        /// Compute an interface flux function
        /// \param[in] dir - vector normal to the face
        /// \param[in] u_left - "left" state at which to evaluate the flux
        /// \param[in] u_right - "right" state at which to evaluate the flux
        /// \param[out] flux_vec - value of the flux
        /// \note This uses the CRTP, so it wraps a call to `calcFlux` in Derived.
        void flux(const mfem::Vector &dir, const mfem::Vector &u_left,
                  const mfem::Vector &u_right, mfem::Vector &flux_vec)
        {
            // calcLaxFriedrichsFlux<dim>(dir.GetData(), u_left.GetData(), u_right.GetData(),
            //                            flux_vec.GetData());
            calcRoeFaceFlux<dim>(dir.GetData(), u_left.GetData(), u_right.GetData(),
                                       flux_vec.GetData());
        }
    };

    /// Integrator for mass matrix
    class EulerMassIntegrator : public mfem::BilinearFormIntegrator
    {
    public:
        /// Constructs a diagonal-mass matrix integrator.
        /// \param[in] nvar - number of state variables
        EulerMassIntegrator(int nvar = 1) : num_state(nvar) {}
        /// Finds the mass matrix for the given element.
        /// \param[in] el - the element for which the mass matrix is desired
        /// \param[in,out] trans -  transformation
        /// \param[out] elmat - the element mass matrix
        void AssembleElementMatrix(const mfem::FiniteElement &el,
                                   mfem::ElementTransformation &trans,
                                   mfem::DenseMatrix &elmat)
        {
            using namespace mfem;
            int num_nodes = el.GetDof();
            double w;

#ifdef MFEM_THREAD_SAFE
            Vector shape;
#endif
            elmat.SetSize(num_nodes * num_state);
            shape.SetSize(num_nodes);
            DenseMatrix elmat1;
            elmat1.SetSize(num_nodes);

            const IntegrationRule *ir = IntRule;
            if (ir == NULL)
            {
                int order = 2 * el.GetOrder() + trans.OrderW();

                if (el.Space() == FunctionSpace::rQk)
                {
                    ir = &RefinedIntRules.Get(el.GetGeomType(), order);
                }
                else
                {
                    ir = &IntRules.Get(el.GetGeomType(), order);
                }
            }
            elmat = 0.0;
            for (int i = 0; i < ir->GetNPoints(); i++)
            {
                const IntegrationPoint &ip = ir->IntPoint(i);
                el.CalcShape(ip, shape);

                trans.SetIntPoint(&ip);
                w = trans.Weight() * ip.weight;

                AddMult_a_VVt(w, shape, elmat1);
                for (int k = 0; k < num_state; k++)
                {
                    elmat.AddMatrix(elmat1, num_nodes * k, num_nodes * k);
                }
            }
        }

    protected:
        mfem::Vector shape;
        mfem::DenseMatrix elmat;
        int num_state;
    };

    template <int dim, bool entvar = false>
    class PressureForce : public NonlinearFormIntegrator
    {
        public:
        PressureForce(const mfem::Vector &force_dir, int num_state, double a) : force_nrm(force_dir),
                                             work_vec(dim + 2),
                                             num_states(num_state), alpha(a) {}
        
        double calcBndryFun(const mfem::Vector &x,
                            const mfem::Vector &dir,
                            const mfem::Vector &q)
        {
            calcSlipWallFlux<dim, entvar>(x.GetData(), dir.GetData(),
                                                  q.GetData(), work_vec.GetData());
            return dot<dim>(force_nrm.GetData(), work_vec.GetData() + 1);
        }

        double GetFaceEnergy(
            const mfem::FiniteElement &el_bnd,
            const mfem::FiniteElement &el_unused,
            mfem::FaceElementTransformations &trans,
            const mfem::Vector &elfun)
        {
            const int num_nodes = el_bnd.GetDof();
#ifdef MFEM_THREAD_SAFE
            Vector u_face, x, nrm, flux_face;
#endif
            u_face.SetSize(num_states);
            x.SetSize(dim);
            nrm.SetSize(dim);
            shape.SetSize(num_nodes);
            double fun = 0.0; // initialize the functional value
            DenseMatrix u(elfun.GetData(), num_nodes, num_states);

            int intorder;
            intorder = trans.Elem1->OrderW() + 2 * el_bnd.GetOrder();
            const IntegrationRule *ir = &IntRules.Get(trans.FaceGeom, intorder);
            IntegrationPoint el_ip;
            for (int i = 0; i < ir->GetNPoints(); i++)
            {
                const IntegrationPoint &face_ip = ir->IntPoint(i);
                trans.Loc1.Transform(face_ip, el_ip);
                trans.Elem1->Transform(el_ip, x);
                el_bnd.CalcShape(el_ip, shape);
                u.MultTranspose(shape, u_face);
                // get the normal vector, and then add contribution to function
                trans.Face->SetIntPoint(&face_ip);
                CalcOrtho(trans.Face->Jacobian(), nrm);
                fun += calcBndryFun(x, nrm, u_face) * face_ip.weight * alpha;
            }
            return fun;
        }

    protected:
        /// number of states
        int num_states;
        /// scales the terms; can be used to move to rhs/lhs
        double alpha;
        /// `dim` entry unit normal vector specifying the direction of the force
        mfem::Vector force_nrm;
        /// work vector used to stored the flux
        mfem::Vector work_vec;
            /// used to reference the state at face node
        mfem::Vector u_face;
        /// store the physical location of a node
        mfem::Vector x;
        /// farfield state value
        mfem::Vector qfs;
        /// the outward pointing (scaled) normal to the boundary at a node
        mfem::Vector nrm;
        mfem::Vector shape;
        /// stores the flux evaluated by `bnd_flux`
        mfem::Vector flux_face;
    };
} // namespace mfem
#endif