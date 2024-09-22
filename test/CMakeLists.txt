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
