
cmake_minimum_required(VERSION 2.8)
PROJECT(ORF_test)

set (RF_SOURCES_DIR /Users/guotaiwang/Documents/workspace/SegTool/source/algorithm/RandomForest)
set (RF_SOURCES_HEADER ${RF_SOURCES_DIR}/Node.h
                       ${RF_SOURCES_DIR}/ODTree.h
                       ${RF_SOURCES_DIR}/ORForest.h)
set (RF_SOURCES_SOURCE ${RF_SOURCES_DIR}/Node.cpp
                       ${RF_SOURCES_DIR}/ODTree.cpp
                       ${RF_SOURCES_DIR}/ORForest.cpp)

set (LASVM_SOURCES_HEADER ./LASVM/lasvm.h ./LASVM/lasvmEnsemble.h)
set (LASVM_SOURCES_SOURCE ./LASVM/lasvm.cpp ./LASVM/lasvmEnsemble.cpp)

FILE(GLOB HEADERS Bayes/*.h ${RF_SOURCES_HEADER} Test/*.h)
FILE(GLOB SOURCES_FILES main.cpp Bayes/*.cpp ${RF_SOURCES_SOURCE} Test/*.cpp ${HEADERS} ${LASVM_SOURCES_HEADER} ${LASVM_SOURCES_SOURCE})

include_directories(${CMAKE_CURRENT_SOURCE_DIR} ${RF_SOURCES_DIR})

ADD_EXECUTABLE(ORF_test ${SOURCES_FILES})

source_group(RF\\Headers FILES ${RF_SOURCES_HEADER})
source_group(RF\\Sources FILES ${RF_SOURCES_SOURCE})

source_group(LASVM\\Headers FILES ${LASVM_SOURCES_HEADER})
source_group(LASVM\\Sources FILES ${LASVM_SOURCES_SOURCE})

source_group(Bayes\\Headers REGULAR_EXPRESSION "Bayes/.*\\.h")
source_group(Bayes\\Sources REGULAR_EXPRESSION "Bayes/.*\\.cpp")

source_group(Test\\Headers REGULAR_EXPRESSION "Test/.*\\.h")
source_group(Test\\Sources REGULAR_EXPRESSION "Test/.*\\.cpp")

