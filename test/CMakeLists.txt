project(unittest LANGUAGES CXX CUDA)

add_executable(HISA_TEST src/hisa_test.cu)
target_include_directories(HISA_TEST PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(HISA_TEST PRIVATE gpu_ra)
target_compile_features(HISA_TEST PRIVATE cxx_std_17)
set_target_properties(HISA_TEST PROPERTIES CUDA_SEPARABLE_COMPILATION ON CUDA_ARCHITECTURES "80")
target_compile_options(HISA_TEST PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:
    -lineinfo
    --expt-relaxed-constexpr
    --default-stream per-thread
    --expt-extended-lambda>)
add_test(NAME HISA_TEST COMMAND HISA_TEST)

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
