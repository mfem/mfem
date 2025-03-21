make all -j
./B_field_projector_B_tor & ./B_field_projector_B_perp_Hcurl & ./B_field_projector_B_perp_Hdiv
./J_field_projector_J_perp & ./J_field_projector_J_tor_Hcurl & ./J_field_projector_J_tor_Hdiv
./JxB_validator_JxB_tor & ./JxB_validator_JxB_perp_Hcurl
./JxB_validator_grad_p