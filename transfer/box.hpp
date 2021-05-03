#ifndef MFEM_L2P_BOX_HPP
#define MFEM_L2P_BOX_HPP

#include "../fem/fem.hpp"


namespace mfem {

	void MaxCol(const DenseMatrix &mat, Vector &vec, bool include_vec_elements);
	void MinCol(const DenseMatrix &mat, Vector &vec, bool include_vec_elements);

	class Box {
	public:

		Box(const int n);
		Box();

		virtual ~Box();

		void Reset(const int n);
		void Reset();

		Box(const DenseMatrix &points);
		Box & operator += (const DenseMatrix &points);
		Box & operator += (const Box &box);

		bool Intersects(const Box &other) const;
		bool Intersects(const Box &other, const double tol) const;

		void Enlarge(const double value);

		void Print(std::ostream &os = std::cout) const;

		inline double GetMin(const int index) const
		{
			return min_(index);
		}

		inline double GetMax(const int index) const
		{
			return max_(index);
		}

		inline const Vector &GetMin() const
		{
			return min_;
		}

		inline const Vector &GetMax() const
		{
			return max_;
		}

		inline Vector &GetMin()
		{
			return min_;
		}

		inline Vector &GetMax()
		{
			return max_;
		}

		inline int GetDims() const
		{
			return min_.Size();
		}

		inline bool Empty() const
		{
			if(min_.Size() == 0) return true;
			return GetMin(0) > GetMax(0);
		}

	private:
		Vector min_;
		Vector max_;
	};

}

#endif //MFEM_L2P_BOX_HPP
