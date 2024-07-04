cmake_minimum_required(VERSION 3.20 FATAL_ERROR)
project(fvlog LANGUAGES CXX CUDA)

set(CMAKE_EXPORT_COMPILE_COMMANDS TRUE CACHE INTERNAL "")

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 80)
endif()

# find_package(MPI REQUIRED)

# # openmp and tbb for CPU backend
# find_package(OpenMP REQUIRED)
# find_package(TBB REQUIRED)

if(OPENMP_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    # set nvcc cmake flag for openmp
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -fopenmp")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

# Add the cuco and rmm dependencies
if(NOT EXISTS ${CMAKE_CURRENT_BINARY_DIR}/gdlog_RAPIDS.cmake)
    file(DOWNLOAD https://raw.githubusercontent.com/rapidsai/rapids-cmake/branch-24.06/RAPIDS.cmake
        ${CMAKE_CURRENT_BINARY_DIR}/gdlog_RAPIDS.cmake)
endif()

include(${CMAKE_CURRENT_BINARY_DIR}/gdlog_RAPIDS.cmake)

include(rapids-cmake)
include(rapids-cpm)
include(rapids-cuda)
include(rapids-export)
include(rapids-find)

rapids_cpm_init()

function(rapids_cpm_cuco)
    list(APPEND CMAKE_MESSAGE_CONTEXT "rapids.cpm.cuco")

    set(options)
    set(one_value INSTALL_EXPORT_SET)
    set(multi_value)
    cmake_parse_arguments(_RAPIDS "${options}" "${one_value}" "${multi_value}" ${ARGN})

    # Fix up _RAPIDS_UNPARSED_ARGUMENTS to have INSTALL_EXPORT_SET as this is need for rapids_cpm_find
    set(to_install OFF)

    if(_RAPIDS_INSTALL_EXPORT_SET)
        list(APPEND _RAPIDS_UNPARSED_ARGUMENTS INSTALL_EXPORT_SET ${_RAPIDS_INSTALL_EXPORT_SET})
        set(to_install ON)
    endif()

    include("${rapids-cmake-dir}/cpm/detail/package_details.cmake")
    rapids_cpm_package_details(cuco version repository tag shallow exclude)

    set(to_exclude OFF)

    if(NOT to_install OR exclude)
        set(to_exclude ON)
    endif()

    include("${rapids-cmake-dir}/cpm/detail/generate_patch_command.cmake")
    rapids_cpm_generate_patch_command(cuco ${version} patch_command)

    include("${rapids-cmake-dir}/cpm/find.cmake")
    rapids_cpm_find(cuco ${version} ${_RAPIDS_UNPARSED_ARGUMENTS}
        GLOBAL_TARGETS cuco::cuco
        CPM_ARGS
        GIT_REPOSITORY ${repository}
        GIT_TAG ${tag}
        GIT_SHALLOW ${shallow} ${patch_command}
        EXCLUDE_FROM_ALL ${to_exclude}
        OPTIONS "BUILD_TESTS OFF" "BUILD_BENCHMARKS OFF" "BUILD_EXAMPLES OFF"
        "INSTALL_CUCO ${to_install}")

    include("${rapids-cmake-dir}/cpm/detail/display_patch_status.cmake")
    rapids_cpm_display_patch_status(cuco)

    # Propagate up variables that CPMFindPackage provide
    set(cuco_SOURCE_DIR "${cuco_SOURCE_DIR}" PARENT_SCOPE)
    set(cuco_BINARY_DIR "${cuco_BINARY_DIR}" PARENT_SCOPE)
    set(cuco_ADDED "${cuco_ADDED}" PARENT_SCOPE)
    set(cuco_VERSION ${version} PARENT_SCOPE)
endfunction()

rapids_cpm_cuco()

function(rapids_cpm_rmm)
    list(APPEND CMAKE_MESSAGE_CONTEXT "rapids.cpm.rmm")

    set(options)
    set(one_value INSTALL_EXPORT_SET)
    set(multi_value)
    cmake_parse_arguments(_RAPIDS "${options}" "${one_value}" "${multi_value}" ${ARGN})

    # Fix up RAPIDS_UNPARSED_ARGUMENTS to have EXPORT_SETS as this is need for rapids_cpm_find
    if(_RAPIDS_INSTALL_EXPORT_SET)
        list(APPEND _RAPIDS_UNPARSED_ARGUMENTS INSTALL_EXPORT_SET ${_RAPIDS_INSTALL_EXPORT_SET})
    endif()

    include("${rapids-cmake-dir}/cpm/detail/package_details.cmake")
    rapids_cpm_package_details(rmm version repository tag shallow exclude)
    set(to_exclude OFF)

    if(NOT _RAPIDS_INSTALL_EXPORT_SET OR exclude)
        set(to_exclude ON)
    endif()

    include("${rapids-cmake-dir}/cpm/detail/generate_patch_command.cmake")
    rapids_cpm_generate_patch_command(rmm ${version} patch_command)

    include("${rapids-cmake-dir}/cpm/find.cmake")
    rapids_cpm_find(rmm ${version} ${ARGN} ${_RAPIDS_UNPARSED_ARGUMENTS}
        GLOBAL_TARGETS rmm::rmm
        CPM_ARGS
        GIT_REPOSITORY ${repository}
        GIT_TAG ${tag}
        GIT_SHALLOW ${shallow} ${patch_command}
        EXCLUDE_FROM_ALL ${to_exclude}
        OPTIONS "BUILD_TESTS OFF" "BUILD_BENCHMARKS OFF")

    include("${rapids-cmake-dir}/cpm/detail/display_patch_status.cmake")
    rapids_cpm_display_patch_status(rmm)

    # Propagate up variables that CPMFindPackage provide
    set(rmm_SOURCE_DIR "${rmm_SOURCE_DIR}" PARENT_SCOPE)
    set(rmm_BINARY_DIR "${rmm_BINARY_DIR}" PARENT_SCOPE)
    set(rmm_ADDED "${rmm_ADDED}" PARENT_SCOPE)
    set(rmm_VERSION ${version} PARENT_SCOPE)

    # rmm creates the correct namespace aliases
endfunction()

rapids_cpm_rmm()

# using test
include(CTest)

if(${CMAKE_SOURCE_DIR} STREQUAL ${CMAKE_CURRENT_SOURCE_DIR})
    add_subdirectory(test)
endif()

enable_testing()

# Add the source files
set(gpu_ra_src "${PROJECT_SOURCE_DIR}/src")
set(gpu_ra_include "${PROJECT_SOURCE_DIR}/include")
# set(test_dir "${PROJECT_SOURCE_DIR}/test")

file(GLOB source_file_gpu_ra
    "${gpu_ra_src}/*.cu"
    "${gpu_ra_src}/*.cpp"
    "${gpu_ra_include}/*.h"
    "${gpu_ra_include}/*.cuh"
)

add_library(gpu_ra "${source_file_gpu_ra}")
target_compile_features(gpu_ra PUBLIC cxx_std_17)
target_compile_definitions(gpu_ra PUBLIC)
target_include_directories(gpu_ra PUBLIC ${gpu_ra_include})

target_link_libraries(gpu_ra rmm::rmm cuco)

set_target_properties(gpu_ra PROPERTIES CUDA_SEPARABLE_COMPILATION ON CUDA_ARCHITECTURES "80")
target_compile_options(gpu_ra PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:
    -lineinfo
    --expt-relaxed-constexpr
    --default-stream per-thread
    --extended-lambda>)
