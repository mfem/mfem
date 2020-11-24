// tensor
#include <iostream>

#pragma once

template<typename T, int... n>
struct tensor;

template<typename T>
struct tensor<T, 1>
{
   static constexpr int shape[1] = {1};
   operator T() { return value; }
   T value;
};

template<typename T, int n>
struct tensor<T, n>
{
   static constexpr int shape[1] = {n};
   constexpr auto &operator[](int i) { return value[i]; };
   constexpr auto operator[](int i) const { return value[i]; };
   T value[n];
};

template<typename T, int first, int... rest>
struct tensor<T, first, rest...>
{
   static constexpr int shape[1 + sizeof...(rest)] = {first, rest...};
   constexpr auto &operator[](int i) { return value[i]; };
   constexpr auto operator[](int i) const { return value[i]; };
   tensor<T, rest...> value[first];
};

template<int n>
constexpr int product(int (&values)[n])
{
   int p = 1;
   for (int i = 0; i < n; i++)
   {
      p *= values[i];
   }
   return p;
}

template<typename S, typename T, int... n>
auto operator+(tensor<S, n...> A, tensor<T, n...> B)
{
   tensor<decltype(S{} + T{}), n...> C{};
   for (int i = 0; i < tensor<T, n...>::shape[0]; i++)
   {
      C[i] = A[i] + B[i];
   }
   return C;
}

template<typename S, typename T, int... n>
auto operator-(tensor<S, n...> A, tensor<T, n...> B)
{
   tensor<decltype(S{} - T{}), n...> C{};
   for (int i = 0; i < tensor<T, n...>::shape[0]; i++)
   {
      C[i] = A[i] - B[i];
   }
   return C;
}

template<typename S, typename T, int... n>
auto operator*(S scale, tensor<T, n...> A)
{
   tensor<decltype(S{} * T{}), n...> C{};
   for (int i = 0; i < tensor<T, n...>::shape[0]; i++)
   {
      C[i] = scale * A[i];
   }
   return C;
}

template<typename S, typename T, int... n>
auto operator*(tensor<T, n...> A, S scale)
{
   tensor<decltype(T{} * S{}), n...> C{};
   for (int i = 0; i < tensor<T, n...>::shape[0]; i++)
   {
      C[i] = A[i] * scale;
   }
   return C;
}

template<typename S, typename T, int... n>
auto operator/(S scale, tensor<T, n...> A)
{
   tensor<decltype(S{} / T{}), n...> C{};
   for (int i = 0; i < tensor<T, n...>::shape[0]; i++)
   {
      C[i] = scale / A[i];
   }
   return C;
}

template<typename S, typename T, int... n>
auto operator/(tensor<T, n...> A, S scale)
{
   tensor<decltype(T{} / S{}), n...> C{};
   for (int i = 0; i < tensor<T, n...>::shape[0]; i++)
   {
      C[i] = A[i] / scale;
   }
   return C;
}

template<typename S, typename T, int m, int n, int p>
auto dot(tensor<S, m, n> A, tensor<T, n, p> B)
{
   tensor<decltype(S{} * T{}), m, p> AB{};
   for (int i = 0; i < m; i++)
   {
      for (int j = 0; j < p; j++)
      {
         for (int k = 0; k < n; k++)
         {
            AB[i][j] = AB[i][j] + A[i][k] * B[k][j];
         }
      }
   }
   return AB;
}

template<typename T, int m, int n>
auto inner(tensor<T, m, n> A, tensor<T, m, n> B)
{
   double value = 0.0;
   for (int i = 0; i < m; i++)
   {
      for (int j = 0; j < n; j++)
      {
         value += A[i][j] * B[i][j];
      }
   }
   return value;
}

auto inner(double a, double b)
{
   return a * b;
}

template<typename T, int n>
auto tr(tensor<T, n, n> A)
{
   T trA{};
   for (int i = 0; i < n; i++)
   {
      trA = trA + A[i][i];
   }
   return trA;
}

template<int dim>
constexpr tensor<double, dim, dim> Identity()
{
   tensor<double, dim, dim> I{};
   for (int i = 0; i < dim; i++)
   {
      for (int j = 0; j < dim; j++)
      {
         I[i][j] = (i == j);
      }
   }
   return I;
}

template<typename T, int m, int n>
auto transpose(const tensor<T, m, n> &A)
{
   tensor<T, n, m> AT{};
   for (int i = 0; i < n; i++)
   {
      for (int j = 0; j < m; j++)
      {
         AT[i][j] = A[j][i];
      }
   }
   return AT;
}

template<typename T, int n>
auto norm(const tensor<T, n> &A)
{
   T r = {};
   for (int i = 0; i < n; i++)
   {
      r = r + A[i] * A[i];
   }
   return pow(r, 0.5);
}

template<typename gradient_type>
struct dual
{
   double value;
   gradient_type gradient;
};

template<typename gradient_type>
auto operator+(dual<gradient_type> a, double b)
{
   return dual<gradient_type>{a.value + b, a.gradient};
}

template<typename gradient_type>
auto operator+(double a, dual<gradient_type> b)
{
   return dual<gradient_type>{a + b.value, b.gradient};
}

template<typename gradient_type>
auto operator+(dual<gradient_type> a, dual<gradient_type> b)
{
   return dual<gradient_type>{a.value + b.value, a.gradient + b.gradient};
}

template<typename gradient_type>
auto operator*(dual<gradient_type> a, double b)
{
   return dual<gradient_type>{a.value * b, a.gradient * b};
}

template<typename gradient_type>
auto operator*(double a, dual<gradient_type> b)
{
   return dual<gradient_type>{a * b.value, a * b.gradient};
}

template<typename gradient_type>
auto operator*(dual<gradient_type> a, dual<gradient_type> b)
{
   return dual<gradient_type>{a.value * b.value,
                              b.value * a.gradient + a.value * b.gradient};
}

template<typename gradient_type>
auto cos(dual<gradient_type> a)
{
   return dual<gradient_type>{cos(a.value), -a.gradient * sin(a.value)};
}

template<typename gradient_type>
auto exp(dual<gradient_type> a)
{
   return dual<gradient_type>{exp(a.value), exp(a.value)};
}

template<typename gradient_type>
auto log(dual<gradient_type> a)
{
   return dual<gradient_type>{log(a.value), a.gradient / a.value};
}

template<typename gradient_type>
auto pow(dual<gradient_type> a, dual<gradient_type> b)
{
   double value = pow(a.value, b.value);
   return dual<gradient_type>{value,
                              value
                                 * (a.gradient * (b.value / a.value)
                                    + b.gradient * log(a.value))};
}

template<typename gradient_type>
auto pow(double a, dual<gradient_type> b)
{
   double value = pow(a, b.value);
   return dual<gradient_type>{value, value * b.gradient * log(a)};
}

template<typename gradient_type>
auto pow(dual<gradient_type> a, double b)
{
   double value = pow(a.value, b);
   return dual<gradient_type>{value, value * a.gradient * b / a.value};
}

template<typename T, int... n>
auto &operator<<(std::ostream &out, dual<T> A)
{
   out << '(' << A.value << ' ' << A.gradient << ')';
   return out;
}

template<typename T, int... n>
auto &operator<<(std::ostream &out, tensor<T, n...> A)
{
   out << '{' << A[0];
   for (int i = 1; i < tensor<T, n...>::shape[0]; i++)
   {
      out << ", " << A[i];
   }
   out << '}';
   return out;
}

auto derivative_wrt(double a)
{
   return dual<double>{a, 1};
}

template<typename T, int m>
auto derivative_wrt(tensor<T, m> A)
{
   tensor<dual<tensor<double, m>>, m> A_dual{};
   for (int i = 0; i < m; i++)
   {
      A_dual[i].value = A[i];
      A_dual[i].gradient[i] = 1.0;
   }
   return A_dual;
}

template<typename T, int m, int n>
auto derivative_wrt(tensor<T, m, n> A)
{
   tensor<dual<tensor<double, m, n>>, m, n> A_dual{};
   for (int i = 0; i < m; i++)
   {
      for (int j = 0; j < n; j++)
      {
         A_dual[i][j].value = A[i][j];
         A_dual[i][j].gradient[i][j] = 1.0;
      }
   }
   return A_dual;
}

template<typename grad_type, int nrows, int ncols>
auto directional_derivative(tensor<dual<grad_type>, nrows, ncols> A, grad_type n)
{
   tensor<double, nrows, ncols> dA_dn{};
   for (int i = 0; i < nrows; i++)
   {
      for (int j = 0; j < ncols; j++)
      {
         dA_dn[i][j] = inner(A[i][j].gradient, n);
      }
   }
   return dA_dn;
}
