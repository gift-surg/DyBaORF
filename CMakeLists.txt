
cmake_minimum_required(VERSION 2.8)
PROJECT(DyBaORF)

set (RF_SOURCES_DIR RF)

set (RF_SOURCES_HEADER ${RF_SOURCES_DIR}/Node.h
                       ${RF_SOURCES_DIR}/ODTree.h
                       ${RF_SOURCES_DIR}/ORForest.h)
set (RF_SOURCES_SOURCE ${RF_SOURCES_DIR}/Node.cpp
                       ${RF_SOURCES_DIR}/ODTree.cpp
                       ${RF_SOURCES_DIR}/ORForest.cpp)


FILE(GLOB HEADERS RF_SOURCES_HEADER Test/*.h)
FILE(GLOB SOURCES_FILES main.cpp ${RF_SOURCES_SOURCE} Test/*.cpp ${HEADERS})

include_directories(${CMAKE_CURRENT_SOURCE_DIR} ${RF_SOURCES_DIR})

ADD_EXECUTABLE(DyBaORF_test ${SOURCES_FILES})

source_group(RF\\Headers FILES ${RF_SOURCES_HEADER})
source_group(RF\\Sources FILES ${RF_SOURCES_SOURCE})

source_group(Test\\Headers REGULAR_EXPRESSION "Test/.*\\.h")
source_group(Test\\Sources REGULAR_EXPRESSION "Test/.*\\.cpp")

