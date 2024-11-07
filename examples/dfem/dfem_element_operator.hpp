#include "dfem_util.hpp"

namespace mfem
{

template <typename... input_ts, size_t... Is>
constexpr auto make_dependency_map_impl(mfem::tuple<input_ts...> inputs,
                                        std::index_sequence<Is...>)
{
   auto make_dependency_tuple = [&](auto i)
   {
      return std::make_tuple((mfem::get<i>(inputs).GetFieldId() == mfem::get<Is>
                              (inputs).GetFieldId())...);
   };

   return std::make_tuple(make_dependency_tuple(std::integral_constant<size_t, Is> {})...);
}

template <typename... input_ts>
constexpr auto make_dependency_map(mfem::tuple<input_ts...> inputs)
{
   return make_dependency_map_impl(inputs, std::index_sequence_for<input_ts...> {});
}

template <typename func_t, typename input_t, typename output_t, typename dependency_map_t>
struct ElementOperator;

template <typename func_t, typename... input_ts, typename... output_ts, typename dependency_map_t>
struct ElementOperator<func_t, mfem::tuple<input_ts...>, mfem::tuple<output_ts...>, dependency_map_t>
{
   using entity_t = Entity::Element;

   func_t qfunc;

   mfem::tuple<input_ts...> inputs;
   mfem::tuple<output_ts...> outputs;

   dependency_map_t dependency_map;

   using qf_param_ts = typename create_function_signature<
                       decltype(&func_t::operator())>::type::parameter_ts;
   using qf_output_t = typename create_function_signature<
                       decltype(&func_t::operator())>::type::return_t;

   static constexpr size_t num_inputs =
      mfem::tuple_size<decltype(inputs)>::value;
   static constexpr size_t num_outputs =
      mfem::tuple_size<decltype(outputs)>::value;

   ElementOperator(func_t qfunc,
                   mfem::tuple<input_ts...> inputs,
                   mfem::tuple<output_ts...> outputs)
      : qfunc(qfunc), inputs(inputs), outputs(outputs),
        dependency_map(make_dependency_map(inputs))
   {
      // Consistency checks
      if constexpr (num_outputs > 1)
      {
         static_assert(always_false<func_t>,
                       "more than one output per kernel is not supported right now");
      }

      constexpr size_t num_qfinputs = mfem::tuple_size<qf_param_ts>::value;
      static_assert(num_qfinputs == num_inputs,
                    "kernel function inputs and descriptor inputs have to match");

      constexpr size_t num_qf_outputs = mfem::tuple_size<qf_output_t>::value;
      static_assert(num_qf_outputs == num_qf_outputs,
                    "kernel function outputs and descriptor outputs have to match");
   }
};

template <typename func_t, typename... input_ts, typename... output_ts>
ElementOperator(func_t, mfem::tuple<input_ts...>, mfem::tuple<output_ts...>)
-> ElementOperator<func_t, mfem::tuple<input_ts...>, mfem::tuple<output_ts...>,
decltype(make_dependency_map(std::declval<mfem::tuple<input_ts...>>()))>;

// template <typename func_t, typename input_t, typename output_t>
// struct BoundaryElementOperator : public
//    ElementOperator<func_t, input_t, output_t>
// {
// public:
//    using entity_t = Entity::BoundaryElement;
//    BoundaryElementOperator(func_t func, input_t inputs, output_t outputs)
//       : ElementOperator<func_t, input_t, output_t>(func, inputs, outputs) {}
// };

// template <typename func_t, typename input_t, typename output_t>
// struct FaceOperator : public
//    ElementOperator<func_t, input_t, output_t>
// {
// public:
//    using entity_t = Entity::Face;
//    FaceOperator(func_t func, input_t inputs, output_t outputs)
//       : ElementOperator<func_t, input_t, output_t>(func, inputs, outputs) {}
// };

} // namespace mfem
