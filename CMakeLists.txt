cmake_minimum_required(VERSION 3.20)
project(autocalc LANGUAGES CXX)

# ------------------------ C++ standard ------------------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ------------------------ Options/Args ------------------------
# Allow: cmake -DARGS="--epochs 1 --data ./Data" ..
set(ARGS "" CACHE STRING "Arguments passed to run-* convenience targets")

# ------------------------ Include paths -----------------------
# Preserve Makefile's explicit include flags
# (These are only used to seed include directories; keep them explicit as requested.)
set(AUTO_INCLUDES "-Iinclude")

# Parse -I... items into include_directories
separate_arguments(AUTO_INCLUDE_TOKENS NATIVE_COMMAND "${AUTO_INCLUDES}")
set(_incs)
foreach(tok ${AUTO_INCLUDE_TOKENS})
  if(tok MATCHES "^-I(.+)$")
    list(APPEND _incs "${CMAKE_MATCH_1}")
  endif()
endforeach()
if(_incs)
  include_directories(${_incs})
endif()

# ------------------------ Flags: Debug/Sanitized ---------------
# Keep original flags from Makefile
set(AUTO_CXXFLAGS_DBG "-std=c++17 -O0 -g -Wall -Wextra -Wpedantic -fno-omit-frame-pointer -fsanitize=address,undefined -MMD -MP")
set(AUTO_LDFLAGS_DBG  "-fsanitize=address,undefined")

# ------------------------ Flags: Fast/Release ------------------
set(AUTO_FAST_CXXFLAGS "-std=c++17 -O3 -DNDEBUG -Wall -Wextra -fno-omit-frame-pointer -ffast-math ${AUTO_FAST_ARCH} -MMD -MP")
set(AUTO_FAST_LDFLAGS  "")

# Extract arch flags (-mcpu/-march) for fast targets so they can be isolated
set(AUTO_FAST_ARCH "")
string(REGEX MATCHALL "-m(cpu|arch)=[^ ]+" _arch_tokens "${AUTO_FAST_CXXFLAGS}")
if(_arch_tokens)
  list(GET _arch_tokens 0 AUTO_FAST_ARCH)
endif()

# Common warnings from Makefile
set(AUTO_WARN_FLAGS "")
string(REGEX MATCHALL "-W[^ ]+" _warn_tokens "${AUTO_CXXFLAGS_DBG}")
foreach(w ${_warn_tokens})
  if(NOT w MATCHES "-Wall|-Wextra|-Wpedantic")
    continue()
  endif()
  list(APPEND AUTO_WARN_FLAGS ${w})
endforeach()

# Common extras (frame-pointer etc.)
set(AUTO_COMMON_FLAGS "")
foreach(flag "-fno-omit-frame-pointer" "-ffast-math")
  string(FIND "${AUTO_CXXFLAGS_DBG} ${AUTO_FAST_CXXFLAGS}" "${flag}" _found)
  if(NOT _found EQUAL -1)
    list(APPEND AUTO_COMMON_FLAGS ${flag})
  endif()
endforeach()

# ------------------------ Source discovery ---------------------
# Mirror Makefile's globbing approach
file(GLOB_RECURSE SRC_FILES CONFIGURE_DEPENDS "src/*.cpp")
file(GLOB TEST_FILES CONFIGURE_DEPENDS "tests/*.cpp")
set(MNIST_EXAMPLE "examples/mnist_demo.cpp")
set(RESNET_EXAMPLE "examples/ResNet_demo.cpp")
set(LSTM_EXAMPLE  "examples/lstm_shakespeare.cpp")

# ------------------------ Output layout ------------------------
# Recreate output paths similar to the Makefile
set(RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)  # default
set(LIBRARY_OUTPUT_DIRECTORY  ${CMAKE_BINARY_DIR}/lib)
set(ARCHIVE_OUTPUT_DIRECTORY  ${CMAKE_BINARY_DIR}/lib)

# Helper function to set an exe's output directory
function(place_exe tgt dir)
  set_target_properties(${tgt} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/${dir}"
  )
endfunction()

# ------------------------ Threads (Linux) ----------------------
find_package(Threads)
set(THREADS_LIB "")
if(Threads_FOUND)
  set(THREADS_LIB Threads::Threads)
endif()

# ===============================================================
# Targets
# ===============================================================

# -------- Tests (sanitized) --------
add_executable(tests ${SRC_FILES} ${TEST_FILES})
target_compile_options(tests PRIVATE ${AUTO_WARN_FLAGS} -O0 -g -fno-omit-frame-pointer)
# Apply sanitizer/linker flags from Makefile
separate_arguments(_dbg_cxx NATIVE_COMMAND "${AUTO_CXXFLAGS_DBG}")
separate_arguments(_dbg_ld  NATIVE_COMMAND "${AUTO_LDFLAGS_DBG}")
target_compile_options(tests PRIVATE ${_dbg_cxx})
target_link_options(tests PRIVATE ${_dbg_ld})
target_link_libraries(tests PRIVATE ${THREADS_LIB})
place_exe(tests "tests")

# -------- Sanity pool (sanitized) --------
if(EXISTS ${CMAKE_SOURCE_DIR}/parallel_sanity/sanity_pool.cpp)
  add_executable(sanity_pool parallel_sanity/sanity_pool.cpp ${SRC_FILES})
  target_compile_options(sanity_pool PRIVATE ${AUTO_WARN_FLAGS} -O0 -g -fno-omit-frame-pointer)
  target_compile_options(sanity_pool PRIVATE ${_dbg_cxx})
  target_link_options(sanity_pool PRIVATE ${_dbg_ld})
  target_link_libraries(sanity_pool PRIVATE ${THREADS_LIB})
  place_exe(sanity_pool ".")
  add_custom_target(run-sanity
    COMMAND $<TARGET_FILE:sanity_pool> ${ARGS}
    DEPENDS sanity_pool
    USES_TERMINAL
  )
endif()

# -------- Debug demo (sanitized) --------
if(EXISTS ${CMAKE_SOURCE_DIR}/${MNIST_EXAMPLE})
  add_executable(mnist_demo_dbg ${MNIST_EXAMPLE} ${SRC_FILES})
  # Warning: tests/*.cpp are NOT added here (mirrors Makefile's filter-out)
  target_compile_options(mnist_demo_dbg PRIVATE ${AUTO_WARN_FLAGS} -O0 -g -fno-omit-frame-pointer)
  target_compile_options(mnist_demo_dbg PRIVATE ${_dbg_cxx})
  target_link_options(mnist_demo_dbg PRIVATE ${_dbg_ld})
  target_link_libraries(mnist_demo_dbg PRIVATE ${THREADS_LIB})
  place_exe(mnist_demo_dbg "demo_slow")
  add_custom_target(run-demo
    COMMAND $<TARGET_FILE:mnist_demo_dbg> ${ARGS}
    DEPENDS mnist_demo_dbg
    USES_TERMINAL
  )
endif()

# -------- Fast demos (release-ish) --------
# Common fast flags
separate_arguments(_fast_cxx NATIVE_COMMAND "${AUTO_FAST_CXXFLAGS}")
separate_arguments(_fast_ld  NATIVE_COMMAND "${AUTO_FAST_LDFLAGS}")

# MNIST
if(EXISTS ${CMAKE_SOURCE_DIR}/${MNIST_EXAMPLE})
  add_executable(fast_mnist ${MNIST_EXAMPLE} ${SRC_FILES})
  target_compile_options(fast_mnist PRIVATE ${AUTO_WARN_FLAGS} -DNDEBUG -O3 ${AUTO_COMMON_FLAGS} ${_fast_cxx})
  target_link_options(fast_mnist PRIVATE ${_fast_ld})
  target_link_libraries(fast_mnist PRIVATE ${THREADS_LIB})
  place_exe(fast_mnist "fast")
  add_custom_target(run-fast-mnist
    COMMAND $<TARGET_FILE:fast_mnist> ${ARGS}
    DEPENDS fast_mnist
    USES_TERMINAL
  )
endif()

# ResNet
if(EXISTS ${CMAKE_SOURCE_DIR}/${RESNET_EXAMPLE})
  add_executable(fast_resnet ${RESNET_EXAMPLE} ${SRC_FILES})
  target_compile_options(fast_resnet PRIVATE ${AUTO_WARN_FLAGS} -DNDEBUG -O3 ${AUTO_COMMON_FLAGS} ${_fast_cxx})
  target_link_options(fast_resnet PRIVATE ${_fast_ld})
  target_link_libraries(fast_resnet PRIVATE ${THREADS_LIB})
  place_exe(fast_resnet "fast")
  add_custom_target(run-fast-resnet
    COMMAND $<TARGET_FILE:fast_resnet> ${ARGS}
    DEPENDS fast_resnet
    USES_TERMINAL
  )
endif()

# LSTM
if(EXISTS ${CMAKE_SOURCE_DIR}/${LSTM_EXAMPLE})
  add_executable(fast_lstm ${LSTM_EXAMPLE} ${SRC_FILES})
  target_compile_options(fast_lstm PRIVATE ${AUTO_WARN_FLAGS} -DNDEBUG -O3 ${AUTO_COMMON_FLAGS} ${_fast_cxx})
  target_link_options(fast_lstm PRIVATE ${_fast_ld})
  target_link_libraries(fast_lstm PRIVATE ${THREADS_LIB})
  place_exe(fast_lstm "fast")
  add_custom_target(run-fast-lstm
    COMMAND $<TARGET_FILE:fast_lstm> ${ARGS}
    DEPENDS fast_lstm
    USES_TERMINAL
  )
endif()

# Convenience "all" similar to Makefile: build tests + fast mnist + fast lstm
add_custom_target(build-all
  DEPENDS tests
)
if(TARGET fast_mnist)
  add_dependencies(build-all fast_mnist)
endif()
if(TARGET fast_lstm)
  add_dependencies(build-all fast_lstm)
endif()

# ------------------------ Notes ------------------------
message(STATUS "Includes: ${_incs}")
message(STATUS "Debug CXXFLAGS: ${AUTO_CXXFLAGS_DBG}")
message(STATUS "Fast  CXXFLAGS: ${AUTO_FAST_CXXFLAGS}")
