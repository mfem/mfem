[Models]
  [elasticity]
    type = LinearIsotropicElasticity
    coefficients = '100 0.3'
    coefficient_types = 'YOUNGS_MODULUS POISSONS_RATIO'
    strain = 'state/strain'
    stress = 'state/stress'
  []
[]
