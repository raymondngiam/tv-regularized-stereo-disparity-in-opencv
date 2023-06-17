add_executable(main main.cpp)

target_link_libraries(main
        ${OpenCV_LIBS}
        tbb
        OpenMP::OpenMP_CXX
        Python3::Python
        Python3::NumPy
        )