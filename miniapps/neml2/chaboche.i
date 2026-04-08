[Models]
  [isoharden]
    type = VoceIsotropicHardening
    saturated_hardening = 100
    saturation_rate = 1.2
  []
  [kinharden]
    type = SR2LinearCombination
    from_var = 'state/internal/X1 state/internal/X2'
    to_var = 'state/internal/X'
  []
  [mandel_stress]
    type = IsotropicMandelStress
    cauchy_stress = 'state/stress'
  []
  [overstress]
    type = SR2LinearCombination
    to_var = 'state/internal/O'
    from_var = 'state/internal/M state/internal/X'
    coefficients = '1 -1'
  []
  [vonmises]
    type = SR2Invariant
    invariant_type = 'VONMISES'
    tensor = 'state/internal/O'
    invariant = 'state/internal/s'
  []
  [yield]
    type = YieldFunction
    yield_stress = 5
    isotropic_hardening = 'state/internal/k'
  []
  [flow]
    type = ComposedModel
    models = 'overstress vonmises yield'
  []
  [normality]
    type = Normality
    model = 'flow'
    function = 'state/internal/fp'
    from = 'state/internal/M state/internal/k'
    to = 'state/internal/NM state/internal/Nk'
  []
  [flow_rate]
    type = PerzynaPlasticFlowRate
    reference_stress = 100
    exponent = 2
  []
  [eprate]
    type = AssociativeIsotropicPlasticHardening
  []
  [X1rate]
    type = ChabochePlasticHardening
    back_stress = 'state/internal/X1'
    C = 10000
    g = 100
    A = 1e-8
    a = 1.2
  []
  [X2rate]
    type = ChabochePlasticHardening
    back_stress = 'state/internal/X2'
    C = 1000
    g = 9
    A = 1e-10
    a = 3.2
  []
  [Eprate]
    type = AssociativePlasticFlow
  []
  [Erate]
    type = SR2VariableRate
    variable = 'forces/strain'
    rate = 'forces/E_rate'
  []
  [Eerate]
    type = SR2LinearCombination
    from_var = 'forces/E_rate state/internal/Ep_rate'
    to_var = 'state/internal/Ee_rate'
    coefficients = '1 -1'
  []
  [elasticity]
    type = LinearIsotropicElasticity
    coefficients = '1e5 0.3'
    coefficient_types = 'YOUNGS_MODULUS POISSONS_RATIO'
    stress = 'state/stress'
    rate_form = true
  []
  [integrate_ep]
    type = ScalarBackwardEulerTimeIntegration
    variable = 'state/internal/ep'
  []
  [integrate_X1]
    type = SR2BackwardEulerTimeIntegration
    variable = 'state/internal/X1'
  []
  [integrate_X2]
    type = SR2BackwardEulerTimeIntegration
    variable = 'state/internal/X2'
  []
  [integrate_stress]
    type = SR2BackwardEulerTimeIntegration
    variable = 'state/stress'
  []
[]

[EquationSystems]
  [eq_sys]
    type = NonlinearSystem
    model = 'implicit_rate'
  []
[]

[Solvers]
  [newton]
    type = Newton
    linear_solver = 'lu'
    verbose = false
  []
  [lu]
    type = DenseLU
  []
[]

[Models]
  [model]
    type = ImplicitUpdate
    equation_system = 'eq_sys'
    solver = 'newton'
  []
  [implicit_rate]
    type = ComposedModel
    models = 'isoharden kinharden mandel_stress overstress vonmises yield normality flow_rate eprate Eprate X1rate X2rate Erate Eerate elasticity integrate_stress integrate_ep integrate_X1 integrate_X2'
  []
[]
