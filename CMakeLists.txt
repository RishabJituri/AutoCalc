cmake_minimum_required(VERSION 3.20)
project(autograd_remote LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Point CMake to Homebrew configs (optional if auto-found)
# set(Protobuf_DIR "/opt/homebrew/lib/cmake/protobuf")
# set(gRPC_DIR "/opt/homebrew/lib/cmake/grpc")

# IMPORTANT: Use CONFIG mode for both
find_package(Protobuf CONFIG REQUIRED)
find_package(gRPC CONFIG REQUIRED)

# Your proto
set(PROTO_DIR ${CMAKE_CURRENT_SOURCE_DIR}/proto)
set(PROTO_FILE ${PROTO_DIR}/engine.proto)
set(GEN_CPP_DIR ${CMAKE_CURRENT_BINARY_DIR}/generated)
file(MAKE_DIRECTORY ${GEN_CPP_DIR})

# Get tool paths from imported targets
get_target_property(PROTOC_EXE protobuf::protoc IMPORTED_LOCATION)
get_target_property(GRPC_CPP_PLUGIN gRPC::grpc_cpp_plugin IMPORTED_LOCATION)

# Generate *.pb.cc/h and *.grpc.pb.cc/h
add_custom_command(
  OUTPUT
    ${GEN_CPP_DIR}/engine.pb.cc ${GEN_CPP_DIR}/engine.pb.h
    ${GEN_CPP_DIR}/engine.grpc.pb.cc ${GEN_CPP_DIR}/engine.grpc.pb.h
  COMMAND ${PROTOC_EXE}
    --proto_path=${PROTO_DIR}
    --cpp_out=${GEN_CPP_DIR}
    --grpc_out=${GEN_CPP_DIR}
    --plugin=protoc-gen-grpc=${GRPC_CPP_PLUGIN}
    ${PROTO_FILE}
  DEPENDS ${PROTO_FILE}
  VERBATIM
)

add_library(proto_objs
  ${GEN_CPP_DIR}/engine.pb.cc
  ${GEN_CPP_DIR}/engine.grpc.pb.cc
)
target_include_directories(proto_objs PUBLIC ${GEN_CPP_DIR})
target_link_libraries(proto_objs PUBLIC protobuf::libprotobuf gRPC::grpc++)

# Your core + service
add_library(core src/variable.cpp)
target_include_directories(core PUBLIC src)

add_library(engine_service src/service.cpp)
target_include_directories(engine_service PUBLIC src ${GEN_CPP_DIR})
target_link_libraries(engine_service PUBLIC core proto_objs)

add_executable(server src/server_main.cpp)
target_link_libraries(server PRIVATE engine_service)
