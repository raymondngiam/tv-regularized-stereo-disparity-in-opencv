set(TEST_BINARY ${PROJECT_NAME}_test)

add_executable(${TEST_BINARY}
        test_primalDualStereo.cpp
        test_cvMatUtils.cpp)

target_link_libraries(${TEST_BINARY}
        gtest
        gtest_main
        ${OpenCV_LIBS}
        tbb
        OpenMP::OpenMP_CXX)

add_test(NAME ${TEST_BINARY}
        COMMAND ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${TEST_BINARY})