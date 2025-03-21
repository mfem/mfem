make all -j
./B_field_projector_B_tor & ./B_field_projector_B_pol_Hcurl & ./B_field_projector_B_pol_Hdiv
./J_field_projector_J_pol & ./J_field_projector_J_tor_Hcurl & ./J_field_projector_J_tor_Hdiv
./JxB_validator_JxB_tor & ./JxB_validator_JxB_pol_Hcurl
./JxB_validator_grad_p