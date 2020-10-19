
#include <assert.h>
#include "Box.hpp"


namespace mfem {

	void MaxCol(const DenseMatrix &mat, Vector &vec, bool include_vec_elements)
	{
		int start = 0;
		if(!include_vec_elements) {
			//I am not modifying mat
			const_cast<DenseMatrix &>(mat).GetColumn(0, vec);
			start = 1;
		} else {
			vec.SetSize(mat.Height());
		}

		for(int i = 0; i < mat.Height(); ++i) {
			for(int j = start; j < mat.Width(); ++j) {
				const double e = mat.Elem(i, j);

				if(vec(i) < e) {
					vec(i) = e;
				}
			}
		}
	}

	void MinCol(const DenseMatrix &mat, Vector &vec, bool include_vec_elements)
	{
		int start = 0;
		if(!include_vec_elements) {
			//I am not modifying mat
			const_cast<DenseMatrix &>(mat).GetColumn(0, vec);
			start = 1;
		} else {
			vec.SetSize(mat.Height());
		}

		for(int i = 0; i < mat.Height(); ++i) {
			for(int j = start; j < mat.Width(); ++j) {
				const double e = mat.Elem(i, j);

				if(vec(i) > e) {
					vec(i) = e;
				}
			}
		}
	}


	Box::Box(const int n)
	: min_(n), max_(n)
	{
		Reset();
	}

	Box::Box() {}

	Box::~Box() {}

	void Box::Reset(const int n)
	{
		min_.SetSize(n);
		max_.SetSize(n);
		Reset();
	}

	void Box::Reset()
	{
		min_ = 	std::numeric_limits<double>::max();
		max_ = -std::numeric_limits<double>::max();
	}

	Box::Box(const DenseMatrix &points)
	{
		MinCol(points, min_, false);
		MaxCol(points, max_, false);
	}

	Box & Box::operator += (const DenseMatrix &points)
	{
		bool is_empty = min_.Size();
		MinCol(points, min_, !is_empty);
		MaxCol(points, max_, !is_empty);
		return *this;
	}

	Box & Box::operator +=(const Box &box)
	{
		using std::min;
		using std::max;

		if(Empty()) {
			*this = box;
			return *this;
		}

		int n = GetDims();

		for(int i = 0; i < n; ++i) {
			min_(i) = min(min_(i), box.min_(i));
			max_(i) = max(max_(i), box.max_(i));
		}

		return *this;
	}
	
	bool Box::Intersects(const Box &other) const
	{
		int n = GetDims();
		
		assert(n == other.GetDims() && "must have same dimensions");
		
		for(int i = 0; i < n; ++i) {
			if(other.GetMax(i) < GetMin(i) || GetMax(i) < other.GetMin(i)) {
				return false;
			}
		}

		return true;
	}

	bool Box::Intersects(const Box &other, const double tol) const
	{
		int n = GetDims();
		
		assert(n == other.GetDims() && "must have same dimensions");
		
		for(int i = 0; i < n; ++i) {
			if(other.GetMax(i) + tol <= GetMin(i) || GetMax(i) + tol <= other.GetMin(i)) {
				return false;
			}
		}

		return true;
	}

	void Box::Enlarge(const double value)
	{
		const int n = GetDims();
		for(int i = 0; i < n; ++i) {
			min_(i) -= value;
			max_(i) += value;
		}
	}

	void Box::Print(std::ostream &os) const
	{
		os << "[\n";
		for(int i = 0; i < GetDims(); ++i) {
			os << "\t" << GetMin(i) << " " << GetMax(i) << "\n";
		}
		os << "]\n";
	}

}


