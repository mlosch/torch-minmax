CMAKE_MINIMUM_REQUIRED(VERSION 2.8 FATAL_ERROR)
CMAKE_POLICY(VERSION 2.8)

IF(NOT Torch_FOUND)
  FIND_PACKAGE(Torch REQUIRED)
ENDIF()

IF(NOT CUDA_FOUND)
  FIND_PACKAGE(CUDA 4.0 REQUIRED)
ENDIF()

IF(NOT MINMAX_INSTALL_LIB_SUBDIR)
  SET(MINMAX_INSTALL_LIB_SUBDIR "lib" CACHE PATH "MINMAX install library directory")
ENDIF()

FILE(GLOB src-cuda *.cu)

CUDA_ADD_LIBRARY(MINMAX MODULE ${src-cuda})

INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR})
TARGET_LINK_LIBRARIES(MINMAX THC TH ${CUDA_cusparse_LIBRARY})

INSTALL(TARGETS MINMAX LIBRARY DESTINATION ${MINMAX_INSTALL_LIB_SUBDIR})
