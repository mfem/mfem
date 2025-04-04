make all -j 8
./B_field_projector_B_pol_Hcurl && ./J_field_projector_J_tor_Hcurl&
./B_field_projector_B_pol_Hdiv && ./J_field_projector_J_tor_Hdiv&
./B_field_projector_B_tor
./J_field_projector_J_pol_Hcurl&
./J_field_projector_J_pol_Hdiv
./JxB_validator_JxB_tor_Hcurl & ./JxB_validator_JxB_pol_Hcurl
./JxB_validator_grad_p