project(unittest LANGUAGES CXX CUDA)

add_executable(TC_TEST src/tc_test.cu)
target_include_directories(TC_TEST PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(TC_TEST PRIVATE gpu_ra)
target_compile_features(TC_TEST PRIVATE cxx_std_17)
set_target_properties(TC_TEST PROPERTIES CUDA_SEPARABLE_COMPILATION ON CUDA_ARCHITECTURES "80")
target_compile_options(TC_TEST PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:
    -lineinfo
    --expt-relaxed-constexpr
    --default-stream per-thread
    --expt-extended-lambda>)
add_test(NAME TC_TEST COMMAND TC_TEST)

# SG
add_executable(SG_TEST src/sg_test.cu)
target_include_directories(SG_TEST PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(SG_TEST PRIVATE gpu_ra)
target_compile_features(SG_TEST PRIVATE cxx_std_17)
set_target_properties(SG_TEST PROPERTIES CUDA_SEPARABLE_COMPILATION ON CUDA_ARCHITECTURES "80")
target_compile_options(SG_TEST PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:
    -lineinfo
    --expt-relaxed-constexpr
    --default-stream per-thread
    --expt-extended-lambda>)
add_test(NAME SG_TEST COMMAND SG_TEST)

# SG shared library
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
add_library(SG SHARED src/sg_shared.cu)
target_include_directories(SG PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(SG PRIVATE gpu_ra)
target_compile_features(SG PRIVATE cxx_std_17)
set_target_properties(SG PROPERTIES CUDA_SEPARABLE_COMPILATION ON CUDA_ARCHITECTURES "80")
target_compile_options(SG PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:
    -lineinfo
    --expt-relaxed-constexpr
    --default-stream per-thread
    --expt-extended-lambda>)

#LUBM
add_executable(LUBM_TEST src/lubm_test.cu)
target_include_directories(LUBM_TEST PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(LUBM_TEST PRIVATE gpu_ra)
target_compile_features(LUBM_TEST PRIVATE cxx_std_17)
set_target_properties(LUBM_TEST PROPERTIES CUDA_SEPARABLE_COMPILATION ON CUDA_ARCHITECTURES "80")
target_compile_options(LUBM_TEST PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:
    -lineinfo
    --expt-relaxed-constexpr
    --default-stream per-thread
    --expt-extended-lambda>)
add_test(NAME LUBM_TEST COMMAND LUBM_TEST)

# owl_el
add_executable(OWL_EL_TEST src/owl_el.cu)
target_include_directories(OWL_EL_TEST PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(OWL_EL_TEST PRIVATE gpu_ra)
target_compile_features(OWL_EL_TEST PRIVATE cxx_std_17)
set_target_properties(OWL_EL_TEST PROPERTIES CUDA_SEPARABLE_COMPILATION ON CUDA_ARCHITECTURES "80")
target_compile_options(OWL_EL_TEST PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:
    -lineinfo
    --expt-relaxed-constexpr
    --default-stream per-thread
    --expt-extended-lambda>)
add_test(NAME OWL_EL_TEST COMMAND OWL_EL_TEST)

# mir test
add_executable(MIR_TEST src/mir_test.cu)
target_include_directories(MIR_TEST PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(MIR_TEST PRIVATE gpu_ra)
target_compile_features(MIR_TEST PRIVATE cxx_std_17)
set_target_properties(MIR_TEST PROPERTIES CUDA_SEPARABLE_COMPILATION ON CUDA_ARCHITECTURES "80")
target_compile_options(MIR_TEST PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:
    -lineinfo
    --expt-relaxed-constexpr
    --default-stream per-thread
    --expt-extended-lambda>)
add_test(NAME MIR_TEST COMMAND MIR_TEST)

# cspa test
add_executable(CSPA_TEST src/cspa.cu)
target_include_directories(CSPA_TEST PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(CSPA_TEST PRIVATE gpu_ra)
target_compile_features(CSPA_TEST PRIVATE cxx_std_17)
set_target_properties(CSPA_TEST PROPERTIES CUDA_SEPARABLE_COMPILATION ON CUDA_ARCHITECTURES "80")
target_compile_options(CSPA_TEST PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:
    -lineinfo
    --expt-relaxed-constexpr
    --default-stream per-thread
    --expt-extended-lambda>)
add_test(NAME CSPA_TEST COMMAND CSPA_TEST)

# polonius test
add_executable(POLONIUS_TEST src/polonius.cu)
target_include_directories(POLONIUS_TEST PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(POLONIUS_TEST PRIVATE gpu_ra)
target_compile_features(POLONIUS_TEST PRIVATE cxx_std_17)
set_target_properties(POLONIUS_TEST PROPERTIES CUDA_SEPARABLE_COMPILATION ON CUDA_ARCHITECTURES "80")
target_compile_options(POLONIUS_TEST PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:
    -lineinfo
    --expt-relaxed-constexpr
    --default-stream per-thread
    --expt-extended-lambda>)
add_test(NAME POLONIUS_TEST COMMAND POLONIUS_TEST)
