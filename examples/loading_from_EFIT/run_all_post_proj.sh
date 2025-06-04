make all -j 8
# first group
./J_field_projector_J_tor_direct &
./J_field_projector_J_pol_direct &
./B_field_projector_B_pol_vec_CG && ./B_field_vec_CG_projector -gf output/B_pol_vec_CG.gf -gn B_pol_vec_CG && ./J_field_projector_J_tor_vec_CG &
./B_field_projector_B_pol_Hcurl && ./B_field_vector_projector -gf output/B_pol_Hcurl.gf -gn B_pol_Hcurl &&./J_field_projector_J_tor_Hcurl &
./B_field_projector_B_pol_Hdiv && ./B_field_vector_projector -gf output/B_pol_Hdiv.gf -gn B_pol_Hdiv && ./J_field_projector_J_tor_Hdiv &
./B_field_projector_B_tor_DG && ./B_field_scalar_projector -gf output/B_tor_DG.gf -gn B_tor_DG && ./J_field_projector_J_pol_Hcurl &
./B_field_projector_B_tor_CG && ./B_field_scalar_projector -gf output/B_tor_CG.gf -gn B_tor_CG && ./J_field_projector_J_pol_Hdiv && ./J_field_projector_J_pol_vec_CG
# second group
./JxB_validator_JxB_tor_A &
./JxB_validator_JxB_pol_A &
./JxB_validator_JxB_tor_B &
./JxB_validator_JxB_pol_B &
./JxB_validator_JxB_tor_C &
./JxB_validator_JxB_pol_C
# third group
./B_field_projector_div_B_pol_vec_CG &
./B_field_projector_div_B_pol_Hcurl &
./B_field_projector_div_B_pol_Hdiv &
./JxB_validator_grad_p
