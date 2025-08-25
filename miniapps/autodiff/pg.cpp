#include "pg.hpp"
namespace mfem
{
PGStepSizeRule::PGStepSizeRule(int rule_type,
                               real_t alpha0, real_t max_alpha,
                               real_t ratio, real_t ratio2)
   : rule_type(static_cast<RuleType>(rule_type))
   , max_alpha(max_alpha), alpha0(alpha0), ratio(ratio), ratio2(ratio2)
{
   MFEM_VERIFY(rule_type < RuleType::INVALID,
               "PGStepSizeRule: Invalid rule type");
   MFEM_VERIFY(alpha0 > 0, "PGStepSizeRule: alpha0 must be positive");
   MFEM_VERIFY(max_alpha >= alpha0,
               "PGStepSizeRule: max_alpha must be greater than or equal to alpha0");
   if (rule_type == RuleType::CONSTANT)
   {
   }
   else if (rule_type == RuleType::POLY)
   {
      MFEM_VERIFY(ratio > 0, "PGStepSizeRule: ratio must be positive for POLY rule");
   }
   else if (rule_type == RuleType::EXP)
   {
      MFEM_VERIFY(ratio > 1,
                  "PGStepSizeRule: ratio must be greater than 1 for EXP rule");
   }
   else if (rule_type == RuleType::DOUBLE_EXP)
   {
      MFEM_VERIFY(ratio > 1 && ratio2 > 1,
                  "PGStepSizeRule: ratio and ratio2 must be greater than 1 for DOUBLE_EXP rule");
   }
}

real_t PGStepSizeRule::Get(int iter) const
{
   real_t alpha = alpha0;
   switch (rule_type)
   {
      case RuleType::CONSTANT:
         break;
      case RuleType::POLY:
         alpha *= std::pow(iter+1, ratio);
         break;
      case RuleType::EXP:
         alpha *= std::pow(ratio, iter);
         break;
      case RuleType::DOUBLE_EXP:
         alpha *= std::pow(ratio, std::pow(ratio2, iter));
         break;
      default:
         break;
   }
   return std::min(alpha, max_alpha);
}

const GridFunction& ADPGFunctional::GetPrevLatent(int i) const
{
   Evaluator::param_t param = evaluator.Get(i);
   const GridFunction* gf = std::visit([&](auto arg)
   {
      using T = std::decay_t<decltype(arg)>;
      if constexpr (std::is_same_v<T, const GridFunction*> ||
                    std::is_same_v<T, const ParGridFunction*>)
      {
         return (const GridFunction*)arg;
      }
      else
      {
         MFEM_ABORT("Parameter at index " << i
                    << " is not a GridFunction or ParGridFunction");
         return (const GridFunction*)nullptr;
      }
   }, param);
   MFEM_VERIFY(gf != nullptr,
               "ADPGFunctional: GetPrevLatent(" << i << ") is null");
   return *gf;
}

}
