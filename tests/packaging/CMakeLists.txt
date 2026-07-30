cmake_minimum_required(VERSION 3.24)

project(RaggedRoutePackageConsumer LANGUAGES CXX)

set(RAGGEDROUTE_REQUIRED_VERSION "0.2" CACHE STRING
    "RaggedRoute package version requested by the consumer test")
find_package(RaggedRoute ${RAGGEDROUTE_REQUIRED_VERSION} CONFIG REQUIRED)

add_executable(raggedroute_package_consumer consumer.cpp)
target_link_libraries(raggedroute_package_consumer PRIVATE
  RaggedRoute::runtime
  RaggedRoute::correctness_framework)
