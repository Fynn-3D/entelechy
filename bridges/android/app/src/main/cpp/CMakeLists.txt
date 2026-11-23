cmake_minimum_required(VERSION 3.22.1)
project(entelechy_android C)

if(NOT DEFINED ENTELECHY_LIB_NAME)
    set(ENTELECHY_LIB_NAME "entelechy_app")
endif()

add_library(entelechy_ffi SHARED IMPORTED)
set_target_properties(entelechy_ffi PROPERTIES
    IMPORTED_LOCATION "${CMAKE_CURRENT_SOURCE_DIR}/../jniLibs/${ANDROID_ABI}/libentelechy_ffi.so"
)

add_library(${ENTELECHY_LIB_NAME} SHARED main.c)
target_include_directories(${ENTELECHY_LIB_NAME} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(${ENTELECHY_LIB_NAME}
    entelechy_ffi
    android
    aaudio
    log
)
