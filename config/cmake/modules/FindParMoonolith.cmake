cmake_minimum_required(VERSION 2.8)

if(MOONOLITH_DIR OR DEFINED ENV{MOONOLITH_DIR})

	find_path(MOONOLITH_INSTALLATION_PATH
		NAME config/moonolith_config.cmake
		HINTS ${MOONOLITH_DIR}
		$ENV{MOONOLITH_DIR}
		)

	
	if(MOONOLITH_INSTALLATION_PATH)
		message(STATUS "Found moonolith installation at ${MOONOLITH_INSTALLATION_PATH}")
		include(${MOONOLITH_INSTALLATION_PATH}/config/moonolith_config.cmake)
		include(FindPackageHandleStandardArgs)

		find_package_handle_standard_args(MOONOLITH
		  REQUIRED_VARS MOONOLITH_LIBRARIES MOONOLITH_INCLUDES
		)

		mark_as_advanced(MOONOLITH_INCLUDES MOONOLITH_LIBRARIES)

		if(MOONOLITH_FOUND)
			add_custom_target(par_moonolith)
		endif()
	else()

	endif()

endif()


if(NOT MOONOLITH_FOUND)
	#Automatically download
	include(ExternalProject)

	set(STAGE_DIR 				"${CMAKE_BINARY_DIR}/stage")
	set(MOONOLITH_URL 			https://zulianp@bitbucket.org/zulianp/par_moonolith.git)
	set(MOONOLITH_SOURCE_DIR 	${STAGE_DIR}/par_moonolith)
	set(MOONOLITH_BIN_DIR 		${STAGE_DIR}/par_moonolith/bin)

	if(MOONOLITH_INSTALL_PREFIX)
		set(MOONOLITH_INSTALL_DIR ${MOONOLITH_INSTALL_PREFIX})
	elseif(DEFINED ENV{MOONOLITH_INSTALL_PREFIX})
		set(MOONOLITH_INSTALL_DIR $ENV{MOONOLITH_INSTALL_PREFIX})
	else()
		set(MOONOLITH_INSTALL_DIR ${CMAKE_BINARY_DIR}/external)
		message(STATUS "Moonolith will be installed in ${MOONOLITH_INSTALL_DIR}.\nIf you want a different location set MOONOLITH_INSTALL_PREFIX")
	endif()


	list(APPEND MOONOLITH_CMAKE_ARGS
		"-DCMAKE_INSTALL_PREFIX=${MOONOLITH_INSTALL_DIR}"
		"-DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}"
		"-DENABLE_SANITIZER=${ENABLE_SANITIZER}"
		"-DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}"
		# "-DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}"
		"-DENABLE_MOONOLITH_PROFILER=${ENABLE_MOONOLITH_PROFILER}"
		)

	message(STATUS "${MOONOLITH_CMAKE_ARGS}")

	ExternalProject_Add(
		par_moonolith 
		UPDATE_COMMAND		"" #FIXME
		PREFIX ${STAGE_DIR}
		GIT_REPOSITORY 		${MOONOLITH_URL}
		DOWNLOAD_DIR 		${STAGE_DIR} 
		INSTALL_DIR         ${MOONOLITH_INSTALL_DIR}
		BINARY_DIR 			${MOONOLITH_SOURCE_DIR}
		CMAKE_ARGS 			"${MOONOLITH_CMAKE_ARGS}"
		LOG_CONFIGURE		1
		LOG_BUILD 			1
	)

	list(APPEND MOONOLITH_INCLUDES 
		${MOONOLITH_INSTALL_DIR}/include
		${MOONOLITH_INSTALL_DIR}/include/kernels
		)

	set(MOONOLITH_LIBRARIES "")
	list(APPEND MOONOLITH_LIBRARIES 
		"-L${MOONOLITH_INSTALL_DIR}/lib"
		"-lmoonolith_opencl"
		"-lpar_moonolith"
		"-lpar_moonolith_intersection"
		"-lpar_moonolith_mpi"
		"-lpar_moonolith_tree"
		"-lpar_moonolith_utils"
		)

	set(MOONOLITH_FOUND TRUE)

endif()
