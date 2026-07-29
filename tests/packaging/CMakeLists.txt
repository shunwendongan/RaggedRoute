cmake_minimum_required(VERSION 3.24)

project(RaggedRoutePackageConsumer LANGUAGES CXX)

find_package(RaggedRoute CONFIG REQUIRED)

add_executable(raggedroute_package_consumer consumer.cpp)
target_link_libraries(raggedroute_package_consumer PRIVATE
  RaggedRoute::baseline_ops
  RaggedRoute::correctness_framework)
