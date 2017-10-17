#ifndef MFEML2_MORTAR_INTEGRATOR_HPP
#define MFEML2_MORTAR_INTEGRATOR_HPP 

#include "../fem/fem.hpp"

namespace mfem {

	/*!
	 * @brief Interface for mortar element assembly
	 */
	class MortarIntegrator {
	public:

		/*!
		 * @brief Implements the assembly routine
		 * @param trial is the master/source element 
		 * @param trial_ir is the quadrature formula for evaluating quantities within the trial element
		 * @param trial_Trans the geometric transformation of the trial element
		 * @param test  is the slave/destination element
		 * @param test_ir is the quadrature formula for evaluating quantities within the test element
		 * @param test_Trans the geometric transformation of the test element
		 * @param elemmat the result of the assembly
		 */
		virtual void AssembleElementMatrix(const FiniteElement   &trial,
			const IntegrationRule &trial_ir,
			ElementTransformation &trial_Trans,
			const FiniteElement   &test,
			const IntegrationRule &test_ir,
			ElementTransformation &test_Trans,
			DenseMatrix	  		  &elemmat
			) = 0;


		/*!
		 * @return the additional orders of quadarture required by the integrator.
		 * It is 0 by default, override method to change that.
		 */
		virtual int GetQuadratureOrder() const
		{
			return 0;
		}

		virtual ~MortarIntegrator() {}
	};

	/*!
	 * @brief Integrator for scalar finite elements 
	 * \f$ (u, v)_{L^2(\mathcal{T}_m \cap \mathcal{T}_s)}, u \in U(\mathcal{T}_m ) and v \in V(\mathcal{T}_s ) \f$
	 */
	class L2MortarIntegrator : public MortarIntegrator {
	public:
		void AssembleElementMatrix(
			const FiniteElement   	&trial,
			const IntegrationRule 	&trial_ir,
			ElementTransformation   &trial_Trans,
			const FiniteElement   	&test,
			const IntegrationRule   &test_ir,
			ElementTransformation   &test_Trans,
			DenseMatrix	  			&elemmat
			) override;
	};

	/*!
	 * @brief Integrator for vector finite elements 
	 * \f$ (u, v)_{L^2(\mathcal{T}_m \cap \mathcal{T}_s)}, u \in U(\mathcal{T}_m ) and v \in V(\mathcal{T}_s ) \f$
	 */
	class VectorL2MortarIntegrator : public MortarIntegrator {
	public:

	#ifndef MFEM_THREAD_SAFE
		Vector shape;
		Vector D;
		DenseMatrix K;
		DenseMatrix test_vshape;
		DenseMatrix trial_vshape;
	#endif

	public:
		VectorL2MortarIntegrator() { Init(NULL, NULL, NULL); }
		VectorL2MortarIntegrator(Coefficient *_q) { Init(_q, NULL, NULL); }
		VectorL2MortarIntegrator(Coefficient &q) { Init(&q, NULL, NULL); }
		VectorL2MortarIntegrator(VectorCoefficient *_vq) { Init(NULL, _vq, NULL); }
		VectorL2MortarIntegrator(VectorCoefficient &vq) { Init(NULL, &vq, NULL); }
		VectorL2MortarIntegrator(MatrixCoefficient *_mq) { Init(NULL, NULL, _mq); }
		VectorL2MortarIntegrator(MatrixCoefficient &mq) { Init(NULL, NULL, &mq); }

		void AssembleElementMatrix(
			const FiniteElement   	&trial,
			const IntegrationRule 	&trial_ir,
			ElementTransformation   &trial_Trans,
			const FiniteElement   	&test,
			const IntegrationRule   &test_ir,
			ElementTransformation   &test_Trans,
			DenseMatrix	  			&elemmat
			) override;

	private:
		Coefficient *Q;
		VectorCoefficient *VQ;
		MatrixCoefficient *MQ;

		void Init(Coefficient *q, VectorCoefficient *vq, MatrixCoefficient *mq)
		{ Q = q; VQ = vq; MQ = mq; }
	};

}

#endif //MFEML2_MORTAR_INTEGRATOR_HPP
