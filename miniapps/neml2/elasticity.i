[Models]
  [model]
    type = LinearIsotropicElasticity
    coefficients = '100 0.3'
    coefficient_types = 'YOUNGS_MODULUS POISSONS_RATIO'
    strain = 'forces/strain'
    stress = 'state/stress'
  []
[]
