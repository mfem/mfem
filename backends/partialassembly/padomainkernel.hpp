// Copyright (c) 2010, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-443211. All Rights
// reserved. See file COPYRIGHT for details.
//
// This file is part of the MFEM library. For more information and source code
// availability see http://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the GNU Lesser General Public License (as published by the Free
// Software Foundation) version 2.1 dated February 1999.


#ifndef MFEM_BACKENDS_PA_DOMAINKERNEL_HPP
#define MFEM_BACKENDS_PA_DOMAINKERNEL_HPP

#if defined(MFEM_USE_BACKENDS) && defined(MFEM_USE_PA)

#include "hostdomainkernel.hpp"
#include "tensor.hpp"

namespace mfem
{

namespace pa
{

template <PAOp OpName>
class QuadDimVal{
public:
	static const int value = 1;
};

template <>
class QuadDimVal<BtDB>{
public:
	static const int value = 0;
};

template <>
class QuadDimVal<GtDG>{
public:
	static const int value = 2;
};

template <typename Equation>
class QuadDim{
public:
	static const int value = QuadDimVal<Equation::OpName>::value;
};

template <typename Equation>
class EltDim{
public:
	static const int value = QuadDim<Equation>::value + 1;
};

template <typename Equation>
class TensorDim{
public:
	static const int value = EltDim<Equation>::value + 1;
};

template <int Dim, typename Vector>
class TensorType;

template <int Dim>
class TensorType<Dim,Vector<double>>{
public:
	typedef Tensor<Dim,double> type;
};

template <typename Vector>
class FESpaceType;

template <>
class FESpaceType<mfem::pa::Vector<double>>{
public:
	typedef mfem::FiniteElementSpace type;
};

template <typename Vector, bool IsLinear, bool IsTimeConstant>
class MeshJac;

template <>
class MeshJac<Vector<double>, false, true>
{
private:
	const int dim;
	Tensor<4> J;
public:
	MeshJac(mfem::FiniteElementSpace& fes, const int dim, const int quads, const int nbElts, const int ir_order)
	: dim(dim), J(dim, dim, quads, nbElts) {
		Tensor<1> J1d(J.getData(), J.length());
		EvalJacobians(dim, &fes, ir_order, J1d);
	}

	const Tensor<2> operator()(const int& e, const int& k) const {
		return Tensor<2>(&J(0, 0, k, e), dim, dim);
	}
};

template <>
class MeshJac<Vector<double>, true, true>
{
private:
	const int dim;
	Tensor<3> J;
public:
	MeshJac(mfem::FiniteElementSpace& fes, const int dim, const int quads, const int nbElts, const int ir_order)
	: dim(dim), J(dim, dim, nbElts) {
		//Needlessly expensive
		MeshJac<Vector<double>, false, true> Jac(fes,dim,quads,nbElts,ir_order);
		Tensor<2> locJ(J.getData(),dim,dim);
		for (int i = 0; i < nbElts; ++i)
		{
			locJ.setView(&J(0,0,i)) = Jac(i,0);
		}
	}

	Tensor<2> operator()(const int& e, const int& k) const {
		return Tensor<2>(&J(0, 0, e), dim, dim);
	}
};

//Only for CPU
struct QuadInfo
{
	int dim;
	int k;
	int e;
	ElementTransformation* tr;
	IntegrationPoint ip;
	Tensor<2> J_ek;
};

//Might only be for CPU too
template <PAOp Op, typename Vector, bool IsLinear=false, bool IsTimeConstant=true>
class Equation
{
public:
	typedef Vector VectorType;
	static const PAOp OpName = Op;
protected:
	typedef typename TensorType<QuadDimVal<OpName>::value, Vector>::type QuadTensor;
private:
	typedef typename FESpaceType<Vector>::type FESpace;
	typedef typename TensorType<2, Vector>::type JTensor;
	FESpace& fes;
	const int dim;
	const IntegrationRule& ir;//FIXME: not yet on GPU
	const IntegrationRule& ir1d;//FIXME: not yet on GPU
	MeshJac<Vector,IsLinear,IsTimeConstant> Jac;
public:
	// Equation(mfem::FiniteElementSpace& fes, const IntegrationRule& ir): fes(fes), dim(fes.GetFE(0)->GetDim()), ir(ir) {}
	Equation(mfem::FiniteElementSpace& fes, const int ir_order)
	: fes(fes)
	, dim(fes.GetFE(0)->GetDim())
	, ir(IntRules.Get(fes.GetFE(0)->GetGeomType(), ir_order))
	, ir1d(IntRules.Get(Geometry::SEGMENT, ir_order))
	, Jac(fes,dim,getNbQuads(),getNbElts(),ir_order) {}

	const FESpace& getTrialFESpace() const { return fes; }
	const FESpace& getTestFESpace() const { return fes; }
	const int getNbDofs1d() const { return fes.GetFE(0)->GetOrder() + 1;}
	const int getNbQuads() const { return ir.GetNPoints(); }
	const int getNbQuads1d() const { return ir1d.GetNPoints(); }
	const int getNbElts() const {return fes.GetNE(); }
	void getJac(const int e, const int k, JTensor& t) const { t = Jac(e,k); }
	const IntegrationPoint& getIntPoint(const int k) const { return ir.IntPoint(k); }
	const IntegrationRule& getIntRule1d() const { return ir1d; }
	const int getDim() const { return fes.GetFE(0)->GetDim(); }

	// ElementInfo getQuadInfo(int e, int k) {
	// 	return {dim, k, e, fes.GetElementTransformation(e)};
	// }
};

class TestEq: public Equation<BtDB,Vector<double>>
{
public:
	TestEq() = delete;
	// TestEq(mfem::FiniteElementSpace& fes, const IntegrationRule& ir): Equation(fes, ir) {}
	TestEq(mfem::FiniteElementSpace& fes, const int ir_order): Equation(fes, ir_order) {}

	void evalD(QuadTensor& D_ek, QuadInfo& info) const {
		D_ek = 1.0;
	}

	void evalD(QuadTensor& D_ek, const int dim, const int k , const int e, ElementTransformation* Tr, const IntegrationPoint& ip, const Tensor<2>& J_ek) const {

	}
};

class HostMassEq: public Equation<BtDB,Vector<double>,true>
{
public:
	HostMassEq() = delete;
	// HostMassEq(mfem::FiniteElementSpace& fes, const IntegrationRule& ir): Equation(fes, ir) {}
	HostMassEq(mfem::FiniteElementSpace& fes, const int ir_order): Equation(fes, ir_order) {}

	void evalD(QuadTensor& D_ek, const int dim, const int k , const int e, ElementTransformation* Tr, const IntegrationPoint& ip, const Tensor<2>& J_ek) const {
		D_ek = ip.weight * det(J_ek);
	}

	void evalD(QuadTensor& D_ek, QuadInfo& info) const {
		D_ek = info.ip.weight * det(info.J_ek);
	}
};

template <typename Equation, typename Vector>
class QuadTensorFunc
{
private:
	typedef typename FESpaceType<Vector>::type FESpace;
	typedef typename TensorType<EltDim<Equation>::value, Vector>::type Tensor; //Defines the Host/Device Tensor type for quadrature data
	typedef typename TensorType<QuadDim<Equation>::value, Vector>::type QuadTensor;
	typedef typename TensorType<2, Vector>::type JTensor;
	Equation eq;//Maybe Equation needs to be moved on GPU Device<Equation,Vector>& eq?
	mutable QuadTensor D_ek;
	mutable JTensor J_ek;
public:
	QuadTensorFunc() = delete;
	QuadTensorFunc(Equation& eq): eq(eq), D_ek(), J_ek(eq.getDim(), eq.getDim()) { }

	const FESpace& getTrialFESpace() const {return eq.getTrialFESpace(); }
	const FESpace& getTestFESpace() const {return eq.getTestFESpace(); }

	//This will not work on GPU
	void evalD(const int e, Tensor& D_e) const {
		ElementTransformation *Tr = eq.getTrialFESpace().GetElementTransformation(e);
		for (int k = 0; k < eq.getNbQuads(); ++k)
		{
			D_ek.slice(D_e, k);
			eq.getJac(e, k, J_ek);
			const IntegrationPoint &ip = eq.getIntPoint(k);
			Tr->SetIntPoint(&ip);
			eq.evalD(D_ek, eq.getDim(), k, e, Tr, ip, J_ek);
		}
	}
};

template <typename Vector>
class PAIntegrator
{
public:
	virtual void Mult(const Vector& x, Vector& y) const = 0;
	virtual void MultAdd(const Vector& x, Vector& y) const = 0;
};


template <typename Equation, typename Vector>
class DomainKernel;

template <typename Equation>
class DomainKernel<Equation,Vector<double>>
{
public:
	typedef HostDomainKernel<Equation::OpName> type;
	
};

/**
*	Can be used as a Matrix Free operator
*/
template <typename Equation, typename Vector>
class PADomainIntegrator: public PAIntegrator<Vector>
{
private:
	typedef typename TensorType<TensorDim<Equation>::value, Vector>::type Tensor;
	typedef typename TensorType<EltDim<Equation>::value, Vector>::type EltTensor;
	typedef QuadTensorFunc<Equation, Vector> QFunc;
	typedef typename DomainKernel<Equation, Vector>::type Kernel;
	Kernel kernel;
	QFunc qfunc;
	mutable EltTensor D_e;
public:
	PADomainIntegrator() = delete;
	PADomainIntegrator(Equation& eq): kernel(eq), qfunc(eq) {}

	//This is unlikely to work on GPU
	void evalD(Tensor& D) const {
		for (int e = 0; e < qfunc.getTrialFESpace().GetNE(); ++e)
		{
			D_e.slice(D, e);
			qfunc.evalD(e, D_e);
		}
	}

	virtual void Mult(const Vector& x, Vector& y) const {
		kernel.Mult(qfunc, x, y);
	}

	virtual void MultAdd(const Vector& x, Vector& y) const {
		kernel.MultAdd(qfunc, x, y);
	}
};

template <typename Equation, typename Vector = typename Equation::VectorType>
PADomainIntegrator<Equation,Vector>* createMFDomainKernel(Equation& eq){ return new PADomainIntegrator<Equation,Vector>(eq); }

static void initFESpaceTensor(const int dim, const int nbQuads, const int nbElts, Tensor<2>& D){
	D.setSize(nbQuads,nbElts);
}
static void initFESpaceTensor(const int dim, const int nbQuads, const int nbElts, Tensor<3>& D){
	D.setSize(dim,nbQuads,nbElts);
}
static void initFESpaceTensor(const int dim, const int nbQuads, const int nbElts, Tensor<4>& D){
	D.setSize(dim,dim,nbQuads,nbElts);
}

/**
*	Partial Assembly operator
*/
template <typename Equation, typename Vector>
class PADomainKernel: public PAIntegrator<Vector>
{
private:
	typedef typename TensorType<TensorDim<Equation>::value, Vector>::type Tensor;
	typedef typename DomainKernel<Equation, Vector>::type Kernel;
	typedef typename FESpaceType<Vector>::type FESpace;
	Kernel kernel;
	Tensor D;
	// const FESpace& trial_fes, test_fes;
public:
	PADomainKernel() = delete;
	//This should work on GPU
	PADomainKernel(Equation& eq): kernel(eq), D() {
		const int dim = eq.getDim();
		const int nbQuads = eq.getNbQuads();
		const int nbElts = eq.getNbElts();
		initFESpaceTensor(dim, nbQuads, nbElts, D);
		PADomainIntegrator<Equation, Vector> integ(eq);
		integ.evalD(D);
	}

	virtual void Mult(const Vector& x, Vector& y) const {
		kernel.Mult(D, x, y);
	}

	virtual void MultAdd(const Vector& x, Vector& y) const {
		kernel.MultAdd(D, x, y);
	}
};


template <typename Equation, typename Vector = typename Equation::VectorType>
PADomainKernel<Equation,Vector>* createPADomainKernel(Equation& eq){ return new PADomainKernel<Equation,Vector>(eq); }

template <typename Vector>
class MatrixType;

template <>
class MatrixType<Vector<double>>{
public:
	typedef mfem::Array<Tensor<2,double>> type;
};

/**
*	Local Matrices operator
*/
template <typename Equation, typename Vector>
class LocMatKernel: public PAIntegrator<Vector>
{
private:
	typedef typename MatrixType<Vector>::type MatrixSet;
	MatrixSet A;//We should find a way to accumulate those...
public:
	LocMatKernel() = delete;
	LocMatKernel(Equation& eq) {
		PADomainIntegrator<Equation, Vector> qfunc(eq);
		auto trial_fes = qfunc.getTrialFES();
		auto test_fes = qfunc.getTestFES();
		//TODO
		// A = QuadBasisOut<Equation::OpOut>(test_fes) * qfunc * QuadBasisIn<Equation::OpIn>(trial_fes);
	}

	virtual void Mult(const Vector& x, Vector& y) const {
		// y = A * x;
		A.Mult(x, y);
	}

	virtual void MultAdd(const Vector& x, Vector& y) const {
		// y += A * x;
		A.MultAdd(x, y);
	}
};

template <typename Vector>
class SpMatrixType;

template <>
class SpMatrixType<Vector<double>>{
public:
	typedef Tensor<2,double> type;
};

/**
*	Sparse Matrix operator
*/
template <typename Equation, typename Vector>
class SpMatKernel: public PAIntegrator<Vector>
{
private:
	typedef typename SpMatrixType<Vector>::type SpMat;
	SpMat A;
public:
	SpMatKernel() = delete;
	SpMatKernel(Equation& eq) {
		LocMatKernel<Equation, Vector> locA(eq);
		A = locA;
	}

	virtual void Mult(const Vector& x, Vector& y) const {
		// y = A * x;
		A.Mult(x, y);
	}

	virtual void MultAdd(const Vector& x, Vector& y) const {
		// y += A * x;
		A.MultAdd(x, y);
	}
};

}

}

#endif

#endif