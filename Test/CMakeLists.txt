
set (TEST_HEADER
    RF/AbstractTestExample.h
    RF/RFTestExample.h
)

set (TEST_SOURCE
    RF/AbstractTestExample.cpp
    RF/RFTestExample.cpp
)

include_directories ("${PROJECT_SOURCE_DIR}/src/RF")
include_directories ("${PROJECT_SOURCE_DIR}/Test/RF")

add_library(dybaorf_test
  ${TEST_SOURCE}
  ${TEST_HEADER}
)


target_link_libraries(dybaorf_test dybaorf)
