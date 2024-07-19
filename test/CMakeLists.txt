project(unittest LANGUAGES CXX HIP)

add_executable(TC_TEST src/tc_test.hip)
target_include_directories(TC_TEST PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(TC_TEST PRIVATE gpu_ra)
target_compile_features(TC_TEST PRIVATE cxx_std_17)
set_target_properties(TC_TEST PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_compile_options(TC_TEST PUBLIC $<$<COMPILE_LANGUAGE:HIP>:
    -lineinfo>)

add_test(NAME TC_TEST COMMAND TC_TEST)

# SG
add_executable(SG_TEST src/sg_test.hip)
target_include_directories(SG_TEST PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(SG_TEST PRIVATE gpu_ra)
target_compile_features(SG_TEST PRIVATE cxx_std_17)
set_target_properties(SG_TEST PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_compile_options(SG_TEST PUBLIC $<$<COMPILE_LANGUAGE:HIP>:
    -lineinfo>)
add_test(NAME SG_TEST COMMAND SG_TEST)

#LUBM
add_executable(LUBM_TEST src/lubm_test.hip)
target_include_directories(LUBM_TEST PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(LUBM_TEST PRIVATE gpu_ra)
target_compile_features(LUBM_TEST PRIVATE cxx_std_17)
set_target_properties(LUBM_TEST PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_compile_options(LUBM_TEST PUBLIC $<$<COMPILE_LANGUAGE:HIP>:
    -lineinfo>)
add_test(NAME LUBM_TEST COMMAND LUBM_TEST)

