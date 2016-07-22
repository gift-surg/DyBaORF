cmake_minimum_required(VERSION 2.8)
PROJECT(DyBaORF)

add_subdirectory(src)
add_subdirectory(Test)

include_directories ("${PROJECT_SOURCE_DIR}/src/RF")
include_directories ("${PROJECT_SOURCE_DIR}/Test/RF")

add_executable(DyBaORF_test ${PROJECT_SOURCE_DIR}/Test/main.cpp)
target_link_libraries (DyBaORF_test dybaorf dybaorf_test)


