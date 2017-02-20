#ifndef OPENCL_ADAPTER_HPP
#define OPENCL_ADAPTER_HPP 

#include <assert.h>
#include <cmath>

#include <string>
#include <algorithm> 

#ifndef __kernel
#define __kernel
#define __global
#define __local
#define __constant
#endif

#define CLIPP_HOST_CL

namespace clipp {
	template<typename T, int D>
	class Vector { };

	template<typename T>
	class Vector<T, 2> { 
	public:
		T x, y;

		enum {
			n_dims = 2
		};

		inline Vector(const T x = 0.0, const T y = 0.0)
		: x(x), y(y)
		{}

		inline T operator[](const int i) const
		{
			switch(i) { case 0: {return x;} case 1: {return y;} default: {assert(false);} }
			return 0;
		}

		inline T &operator[](const int i)
		{
			static T null_object = 0;
			switch(i) { case 0: {return x;} case 1: {return y;} default: {assert(false);} }
			return null_object;
		}

		inline friend T dot(const Vector &left, const Vector &right)
		{
			return left.x * right.x + left.y * right.y;
		}

		inline friend T length(const Vector &left)
		{
			return sqrt(left.x * left.x  + left.y * left.y);
		}

		inline friend Vector operator+(const Vector &left, const Vector &right)
		{
			return Vector(left.x + right.x, left.y + right.y);
		}

		inline friend Vector operator-(const Vector &left, const Vector &right)
		{
			return Vector(left.x - right.x, left.y - right.y);
		}

		inline friend Vector operator*(const T &left, const Vector &right)
		{
			return Vector(left * right.x, left * right.y);
		}

		inline friend Vector operator*(const Vector &left, const T &right)
		{
			return Vector(left.x * right, left.y * right);
		}

		inline friend Vector operator/(const Vector &left, const T &right)
		{
			return Vector(left.x / right, left.y / right);
		}

		//in-place
		inline Vector & operator+=(const Vector &other)
		{
			x += other.x;
			y += other.y;
			return *this;
		}

		inline Vector & operator-=(const Vector &other)
		{
			x -= other.x;
			y -= other.y;
			return *this;
		}

		inline Vector & operator*=(const T &other)
		{
			x *= other;
			y *= other;
			return *this;
		}

		inline Vector & operator/=(const T &other)
		{
			x /= other;
			y /= other;
			return *this;
		}

		inline friend Vector operator-(const Vector &other)
		{
			return Vector(-other.x, -other.y);
		}

		inline friend T distance(const Vector &left, const Vector &right)
		{
			return length(left - right);
		}

		inline friend Vector normalize(const Vector &v)
		{
			return v * (1.0/length(v));
		}
	};

	template<typename T>
	class Vector<T, 3> { 
	public:
		T x, y, z;

		enum {
			n_dims = 3
		};

		Vector(const T x = 0.0, const T y = 0.0, const T z = 0.0)
		: x(x), y(y), z(z)
		{}

		T operator[](const int i) const
		{
			switch(i) { case 0: {return x;} case 1: {return y;} case 2: {return z;} default: {assert(false);} }
			return 0;
		}

		T &operator[](const int i)
		{
			static T null_object = 0;
			switch(i) { case 0: {return x;} case 1: {return y;} case 2: {return z;} default: {assert(false);} }
			return null_object;
		}

		inline friend T dot(const Vector &left, const Vector &right)
		{
			return left.x * right.x + left.y * right.y + left.z * right.z;
		}

		inline friend T length(const Vector &left)
		{
			return sqrt(left.x * left.x  + left.y * left.y + left.z * left.z );
		}

		inline friend Vector operator+(const Vector &left, const Vector &right)
		{
			return Vector(left.x + right.x, left.y + right.y, left.z + right.z);
		}

		inline friend Vector operator-(const Vector &left, const Vector &right)
		{
			return Vector(left.x - right.x, left.y - right.y, left.z - right.z);
		}

		inline friend Vector operator*(const T &left, const Vector &right)
		{
			return Vector(left * right.x, left * right.y, left * right.z);
		}

		inline friend Vector operator*(const Vector &left, const T &right)
		{
			return Vector(left.x * right, left.y * right, left.z * right);
		}

		inline friend Vector operator/(const Vector &left, const T &right)
		{
			return Vector(left.x / right, left.y / right, left.z / right);
		}

		//in-place
		inline Vector & operator+=(const Vector &other)
		{
			x += other.x;
			y += other.y;
			z += other.z;
			return *this;
		}

		inline Vector & operator-=(const Vector &other)
		{
			x -= other.x;
			y -= other.y;
			z -= other.z;
			return *this;
		}

		inline Vector & operator*=(const T &other)
		{
			x *= other;
			y *= other;
			z *= other;
			return *this;
		}

		inline Vector & operator/=(const T &other)
		{
			x /= other;
			y /= other;
			z /= other;
			return *this;
		}

		inline friend Vector operator-(const Vector &other)
		{
			return Vector(-other.x, -other.y, -other.z);
		}

		inline friend T distance(const Vector &left, const Vector &right)
		{
			return length(left - right);
		}

		inline friend Vector cross(const Vector &left, const Vector &right)
		{
			return Vector( (left.y * right.z) - ( right.y * left.z ),
				(left.z * right.x) - ( right.z * left.x ),
				(left.x * right.y) - ( right.x * left.y ) ); 
		}

		inline friend Vector normalize(const Vector &v)
		{
			return v * (1.0/length(v));
		}
	};

	class OpenCLAdapter {
	public:
		typedef Vector<double, 2> double2;
		typedef Vector<double, 3> double3;

		typedef Vector<float, 2> float2;
		typedef Vector<float, 3> float3;

		typedef double Scalar;
		typedef Vector<Scalar, 2> Vector2;
		typedef Vector<Scalar, 3> Vector3;

		typedef int SizeType;

		int global_size[3];

		OpenCLAdapter();

		int get_global_id(const int i) const;
		int get_global_size(const int i) const;
		int get_group_id(const int i) const;
		int get_local_size(const int i) const;
		int get_local_id(const int i) const;

		void set_global_size(const int i, const int size);

		static const int CLK_LOCAL_MEM_FENCE  = 0;
		static const int CLK_GLOBAL_MEM_FENCE = 1;

		inline static Scalar min(const Scalar x, const Scalar y)
		{
			return std::min(x, y);
		}

		inline static Scalar max(const Scalar x, const Scalar y)
		{
			return std::max(x, y);
		}

		inline static Vector3 vec_3(const Scalar x, const Scalar y,const Scalar z)
		{
			return Vector3(x, y, z);
		}

		inline static Vector2 vec_2(const Scalar x, const Scalar y)
		{
			return Vector2(x, y);
		}

		inline static const int sign(const Scalar x)
		{
			return (x < 0)? -1 : (x > 0? 1 : 0);
		}

		inline static void barrier(const int barrier_type) {
			(void) barrier_type;
		}

		static void assertion_failure(const char *assertion, const char *file, unsigned line, const char *function) {
			fprintf(stderr, "%s:%u: %s%sAssertion `%s' failed.\n",
				file,
				line,
				function ? function : "",
				function ? ": " : "",
				assertion
				);
			
			abort();
		}
	};
}

#endif //OPENCL_ADAPTER_HPP

