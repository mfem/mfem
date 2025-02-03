#include <tuple>
#include <type_traits>
#include <iostream>
#include <enzyme/enzyme>

template <typename T>
constexpr auto get_type_name() -> std::string_view
{
#if defined(__clang__)
   constexpr auto prefix = std::string_view {"[T = "};
   constexpr auto suffix = "]";
   constexpr auto function = std::string_view{__PRETTY_FUNCTION__};
#elif defined(__GNUC__)
   constexpr auto prefix = std::string_view {"with T = "};
   constexpr auto suffix = "; ";
   constexpr auto function = std::string_view{__PRETTY_FUNCTION__};
#elif defined(_MSC_VER)
   constexpr auto prefix = std::string_view {"get_type_name<"};
   constexpr auto suffix = ">(void)";
   constexpr auto function = std::string_view{__FUNCSIG__};
#else
#error Unsupported compiler
#endif

   const auto start = function.find(prefix) + prefix.size();
   const auto end = function.find(suffix);
   const auto size = end - start;

   return function.substr(start, size);
}

template <typename ... Ts>
constexpr auto decay_types(std::tuple<Ts...> const &)
-> std::tuple<std::remove_cv_t<std::remove_reference_t<Ts>>...>;

template <typename T>
using decay_tuple = decltype(decay_types(std::declval<T>()));

template <class F> struct FunctionSignature;

template <typename output_t, typename... input_ts>
struct FunctionSignature<output_t(input_ts...)>
{
   using return_t = output_t;
   using parameter_ts = std::tuple<input_ts...>;
};

template <class T> struct create_function_signature;

template <typename output_t, typename T, typename... input_ts>
struct create_function_signature<output_t (T::*)(input_ts...) const>
{
   using type = FunctionSignature<output_t(input_ts...)>;
};

template <typename arg_ts, std::size_t... Is>
auto create_enzyme_args(arg_ts &args,
                        arg_ts &shadow_args,
                        std::index_sequence<Is...>)
{
   // (std::cout << ... << std::get<Is>(shadow_args));
   return std::tuple<enzyme::Duplicated<decltype(std::get<Is>(args))>...>
   {
      { std::get<Is>(args), std::get<Is>(shadow_args) }...
   };
}

template <typename kernel_t, typename arg_ts>
auto fwddiff_apply_enzyme(kernel_t kernel, arg_ts &&args, arg_ts &&shadow_args)
{
   auto arg_indices =
      std::make_index_sequence<std::tuple_size_v<std::remove_reference_t<arg_ts>>> {};

   auto enzyme_args = create_enzyme_args(args, shadow_args, arg_indices);

   // using kf_return_t = typename create_function_signature<
   //                     decltype(&kernel_t::operator())>::type::return_t;

   std::cout << "\n";
   std::cout << "args is " << get_type_name<decltype(args)>() << "\n\n";
   std::cout << "enzyme_args type is " << get_type_name<decltype(enzyme_args)>() <<
             "\n\n";
   // std::cout << "return type is " << get_type_name<decltype(kf_return_t{})>() <<
   //           "\n\n";

   std::cout << "args " << std::get<0>(args) << "\n";
   std::cout << "shadow args " << std::get<0>(shadow_args) << "\n";

   return std::apply([&](auto &&...args)
   {
      // std::cout << enzyme::autodiff<enzyme::Forward>(+kernel, args...) << "\n";
      return enzyme::get<0>
             (enzyme::autodiff<enzyme::Forward>(+kernel, args...));
   },
   enzyme_args);
}

int main()
{

   auto func = [](const double &x, double &y)
   {
      std::cout << "func( x = " << x << " )\n";
      return x*x;
   };

   using kf_param_ts = typename create_function_signature<
                       decltype(&decltype(func)::operator())>::type::parameter_ts;
   using kf_output_t = typename create_function_signature<
                       decltype(&decltype(func)::operator())>::type::return_t;
   auto kernel_args = decay_tuple<kf_param_ts> {};
   auto kernel_shadow_args = decay_tuple<kf_param_ts> {};

   std::get<0>(kernel_args) = 3;
   std::get<0>(kernel_shadow_args) = 1;

   auto dx = fwddiff_apply_enzyme(func, kernel_args, kernel_shadow_args);

   std::cout << "dfdx = " << dx << "\n";

   return 0;
}
