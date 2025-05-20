cmake_minimum_required(VERSION 3.20)
project(Project)
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
set(CMAKE_CXX_STANDARD 20)

# Add nlohmann_json
include(FetchContent)
FetchContent_Declare(
    json
    URL https://github.com/nlohmann/json/releases/download/v3.11.2/json.tar.xz
)
FetchContent_MakeAvailable(json)

# Main project executable
add_executable(Project main.cpp
        proj.cpp
        proj.h)
target_link_libraries(Project ${OpenCV_LIBS})

# Test executable
add_executable(test_program test.cpp
        proj.cpp
        proj.h)
target_link_libraries(test_program ${OpenCV_LIBS} nlohmann_json::nlohmann_json)
