set (RF_SOURCES_SOURCE
	RF/Node.cpp
    RF/ODTree.cpp
    RF/ORForest.cpp
)

set (RF_SOURCES_HEADER
	RF/Node.h
    RF/ODTree.h
    RF/ORForest.h
)

add_library(dybaorf
  ${RF_SOURCES_SOURCE}
  ${RF_SOURCES_HEADER}
)
