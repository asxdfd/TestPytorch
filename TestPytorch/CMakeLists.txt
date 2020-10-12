# CMakeList.txt : CMake project for TestPytorch, include source and define
# project specific logic here.
#
cmake_minimum_required (VERSION 3.8 FATAL_ERROR)

set(CMAKE_PREFIX_PATH "E:/pytorch/libtorch")
set(OPENCV_PREFIX_PATH "E:/opencv-4.3.0-contrib/build_new/install")

find_package(Torch REQUIRED)

include_directories("./include")
include_directories("${OPENCV_PREFIX_PATH}/include")
link_directories("${OPENCV_PREFIX_PATH}/x64/vc16/lib")

# Add source to this project's executable.
add_executable (TestPytorch "src/TestPytorch.cpp")

# TODO: Add tests and install targets if needed.
target_link_libraries(TestPytorch
    ${TORCH_LIBRARIES}
    opencv_img_hash430
	opencv_world430
)

set_property(TARGET TestPytorch PROPERTY CXX_STANDARD 14)

if (MSVC)
  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
  file(GLOB OPENCV_DLLS "${OPENCV_PREFIX_PATH}/x64/vc16/bin/*.dll")
  add_custom_command(TARGET TestPytorch
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     ${TORCH_DLLS} ${OPENCV_DLLS}
                     $<TARGET_FILE_DIR:TestPytorch>)
endif (MSVC)
