#ifndef MFEM_L2P_MORTAR_ASSEMBLE_HPP
#define MFEM_L2P_MORTAR_ASSEMBLE_HPP

#include <memory>
#include <math.h>

#include "../fem/fem.hpp"

#include "opencl_adapter.hpp"

#define USE_DOUBLE_PRECISION
#define DEFAULT_TOLLERANCE 1e-12

namespace mfem {

	class Intersector : public moonolith::OpenCLAdapter {
	public:
		//I do not know why the compiler wants this...
		template<typename T>
		inline static T sqrt(const T v) {
			return std::sqrt(v);
		}

		#include "all_kernels.cl"
	};


	typedef mfem::Intersector::PMesh Polyhedron;

	void Print(const IntegrationRule &ir, std::ostream &os = std::cout);

	double SumOfWeights(const IntegrationRule &ir);
	double Sum(const DenseMatrix &mat);

	void MakeCompositeQuadrature2D(const DenseMatrix &polygon, const double weight, const int order, IntegrationRule &c_ir);
	void MakeCompositeQuadrature3D(const Polyhedron &polyhedron, const double weight, const int order, IntegrationRule &c_ir);

	void TransformToReference(ElementTransformation &Trans, const int type, const IntegrationRule &global_ir, IntegrationRule &ref_ir);

	void MortarAssemble(const FiniteElement &trial_fe, const IntegrationRule &trial_ir,
						const FiniteElement &test_fe, const IntegrationRule &test_ir,
						ElementTransformation &Trans, DenseMatrix &elmat);

	void MakePolyhedron(const Mesh &m, const int el_index, Polyhedron &polyhedron);

	bool Intersect2D(const DenseMatrix &poly1, const DenseMatrix &poly2, DenseMatrix &intersection);
	bool Intersect3D(const Mesh &m1, const int el1, const Mesh &m2, const int el2, Polyhedron &intersection);
}

#undef mortar_assemble
#endif //MFEM_L2P_MORTAR_ASSEMBLE_HPP
