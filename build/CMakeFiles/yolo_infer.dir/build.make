# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.22

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/alfeak/lidar/workspace/implement/yolo_tensorrt_infer_module

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/alfeak/lidar/workspace/implement/yolo_tensorrt_infer_module/build

# Include any dependencies generated for this target.
include CMakeFiles/yolo_infer.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/yolo_infer.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/yolo_infer.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/yolo_infer.dir/flags.make

CMakeFiles/yolo_infer.dir/src/dataloader.cpp.o: CMakeFiles/yolo_infer.dir/flags.make
CMakeFiles/yolo_infer.dir/src/dataloader.cpp.o: ../src/dataloader.cpp
CMakeFiles/yolo_infer.dir/src/dataloader.cpp.o: CMakeFiles/yolo_infer.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/alfeak/lidar/workspace/implement/yolo_tensorrt_infer_module/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/yolo_infer.dir/src/dataloader.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/yolo_infer.dir/src/dataloader.cpp.o -MF CMakeFiles/yolo_infer.dir/src/dataloader.cpp.o.d -o CMakeFiles/yolo_infer.dir/src/dataloader.cpp.o -c /home/alfeak/lidar/workspace/implement/yolo_tensorrt_infer_module/src/dataloader.cpp

CMakeFiles/yolo_infer.dir/src/dataloader.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/yolo_infer.dir/src/dataloader.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/alfeak/lidar/workspace/implement/yolo_tensorrt_infer_module/src/dataloader.cpp > CMakeFiles/yolo_infer.dir/src/dataloader.cpp.i

CMakeFiles/yolo_infer.dir/src/dataloader.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/yolo_infer.dir/src/dataloader.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/alfeak/lidar/workspace/implement/yolo_tensorrt_infer_module/src/dataloader.cpp -o CMakeFiles/yolo_infer.dir/src/dataloader.cpp.s

CMakeFiles/yolo_infer.dir/src/engine.cpp.o: CMakeFiles/yolo_infer.dir/flags.make
CMakeFiles/yolo_infer.dir/src/engine.cpp.o: ../src/engine.cpp
CMakeFiles/yolo_infer.dir/src/engine.cpp.o: CMakeFiles/yolo_infer.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/alfeak/lidar/workspace/implement/yolo_tensorrt_infer_module/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/yolo_infer.dir/src/engine.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/yolo_infer.dir/src/engine.cpp.o -MF CMakeFiles/yolo_infer.dir/src/engine.cpp.o.d -o CMakeFiles/yolo_infer.dir/src/engine.cpp.o -c /home/alfeak/lidar/workspace/implement/yolo_tensorrt_infer_module/src/engine.cpp

CMakeFiles/yolo_infer.dir/src/engine.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/yolo_infer.dir/src/engine.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/alfeak/lidar/workspace/implement/yolo_tensorrt_infer_module/src/engine.cpp > CMakeFiles/yolo_infer.dir/src/engine.cpp.i

CMakeFiles/yolo_infer.dir/src/engine.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/yolo_infer.dir/src/engine.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/alfeak/lidar/workspace/implement/yolo_tensorrt_infer_module/src/engine.cpp -o CMakeFiles/yolo_infer.dir/src/engine.cpp.s

CMakeFiles/yolo_infer.dir/src/postprocess.cu.o: CMakeFiles/yolo_infer.dir/flags.make
CMakeFiles/yolo_infer.dir/src/postprocess.cu.o: ../src/postprocess.cu
CMakeFiles/yolo_infer.dir/src/postprocess.cu.o: CMakeFiles/yolo_infer.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/alfeak/lidar/workspace/implement/yolo_tensorrt_infer_module/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CUDA object CMakeFiles/yolo_infer.dir/src/postprocess.cu.o"
	/usr/local/cuda-12.6/bin/nvcc -forward-unknown-to-host-compiler $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -MD -MT CMakeFiles/yolo_infer.dir/src/postprocess.cu.o -MF CMakeFiles/yolo_infer.dir/src/postprocess.cu.o.d -x cu -c /home/alfeak/lidar/workspace/implement/yolo_tensorrt_infer_module/src/postprocess.cu -o CMakeFiles/yolo_infer.dir/src/postprocess.cu.o

CMakeFiles/yolo_infer.dir/src/postprocess.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CUDA source to CMakeFiles/yolo_infer.dir/src/postprocess.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

CMakeFiles/yolo_infer.dir/src/postprocess.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CUDA source to assembly CMakeFiles/yolo_infer.dir/src/postprocess.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

CMakeFiles/yolo_infer.dir/src/preprocess.cu.o: CMakeFiles/yolo_infer.dir/flags.make
CMakeFiles/yolo_infer.dir/src/preprocess.cu.o: ../src/preprocess.cu
CMakeFiles/yolo_infer.dir/src/preprocess.cu.o: CMakeFiles/yolo_infer.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/alfeak/lidar/workspace/implement/yolo_tensorrt_infer_module/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building CUDA object CMakeFiles/yolo_infer.dir/src/preprocess.cu.o"
	/usr/local/cuda-12.6/bin/nvcc -forward-unknown-to-host-compiler $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -MD -MT CMakeFiles/yolo_infer.dir/src/preprocess.cu.o -MF CMakeFiles/yolo_infer.dir/src/preprocess.cu.o.d -x cu -c /home/alfeak/lidar/workspace/implement/yolo_tensorrt_infer_module/src/preprocess.cu -o CMakeFiles/yolo_infer.dir/src/preprocess.cu.o

CMakeFiles/yolo_infer.dir/src/preprocess.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CUDA source to CMakeFiles/yolo_infer.dir/src/preprocess.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

CMakeFiles/yolo_infer.dir/src/preprocess.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CUDA source to assembly CMakeFiles/yolo_infer.dir/src/preprocess.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

CMakeFiles/yolo_infer.dir/src/test.cpp.o: CMakeFiles/yolo_infer.dir/flags.make
CMakeFiles/yolo_infer.dir/src/test.cpp.o: ../src/test.cpp
CMakeFiles/yolo_infer.dir/src/test.cpp.o: CMakeFiles/yolo_infer.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/alfeak/lidar/workspace/implement/yolo_tensorrt_infer_module/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Building CXX object CMakeFiles/yolo_infer.dir/src/test.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/yolo_infer.dir/src/test.cpp.o -MF CMakeFiles/yolo_infer.dir/src/test.cpp.o.d -o CMakeFiles/yolo_infer.dir/src/test.cpp.o -c /home/alfeak/lidar/workspace/implement/yolo_tensorrt_infer_module/src/test.cpp

CMakeFiles/yolo_infer.dir/src/test.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/yolo_infer.dir/src/test.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/alfeak/lidar/workspace/implement/yolo_tensorrt_infer_module/src/test.cpp > CMakeFiles/yolo_infer.dir/src/test.cpp.i

CMakeFiles/yolo_infer.dir/src/test.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/yolo_infer.dir/src/test.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/alfeak/lidar/workspace/implement/yolo_tensorrt_infer_module/src/test.cpp -o CMakeFiles/yolo_infer.dir/src/test.cpp.s

# Object files for target yolo_infer
yolo_infer_OBJECTS = \
"CMakeFiles/yolo_infer.dir/src/dataloader.cpp.o" \
"CMakeFiles/yolo_infer.dir/src/engine.cpp.o" \
"CMakeFiles/yolo_infer.dir/src/postprocess.cu.o" \
"CMakeFiles/yolo_infer.dir/src/preprocess.cu.o" \
"CMakeFiles/yolo_infer.dir/src/test.cpp.o"

# External object files for target yolo_infer
yolo_infer_EXTERNAL_OBJECTS =

yolo_infer: CMakeFiles/yolo_infer.dir/src/dataloader.cpp.o
yolo_infer: CMakeFiles/yolo_infer.dir/src/engine.cpp.o
yolo_infer: CMakeFiles/yolo_infer.dir/src/postprocess.cu.o
yolo_infer: CMakeFiles/yolo_infer.dir/src/preprocess.cu.o
yolo_infer: CMakeFiles/yolo_infer.dir/src/test.cpp.o
yolo_infer: CMakeFiles/yolo_infer.dir/build.make
yolo_infer: /usr/lib/libopencv_gapi.so.4.8.0
yolo_infer: /usr/lib/libopencv_highgui.so.4.8.0
yolo_infer: /usr/lib/libopencv_ml.so.4.8.0
yolo_infer: /usr/lib/libopencv_objdetect.so.4.8.0
yolo_infer: /usr/lib/libopencv_photo.so.4.8.0
yolo_infer: /usr/lib/libopencv_stitching.so.4.8.0
yolo_infer: /usr/lib/libopencv_video.so.4.8.0
yolo_infer: /usr/lib/libopencv_videoio.so.4.8.0
yolo_infer: /usr/local/cuda-12.6/lib64/libcudart_static.a
yolo_infer: /usr/lib/aarch64-linux-gnu/librt.a
yolo_infer: /usr/lib/libopencv_imgcodecs.so.4.8.0
yolo_infer: /usr/lib/libopencv_dnn.so.4.8.0
yolo_infer: /usr/lib/libopencv_calib3d.so.4.8.0
yolo_infer: /usr/lib/libopencv_features2d.so.4.8.0
yolo_infer: /usr/lib/libopencv_flann.so.4.8.0
yolo_infer: /usr/lib/libopencv_imgproc.so.4.8.0
yolo_infer: /usr/lib/libopencv_core.so.4.8.0
yolo_infer: CMakeFiles/yolo_infer.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/alfeak/lidar/workspace/implement/yolo_tensorrt_infer_module/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_6) "Linking CXX executable yolo_infer"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/yolo_infer.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/yolo_infer.dir/build: yolo_infer
.PHONY : CMakeFiles/yolo_infer.dir/build

CMakeFiles/yolo_infer.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/yolo_infer.dir/cmake_clean.cmake
.PHONY : CMakeFiles/yolo_infer.dir/clean

CMakeFiles/yolo_infer.dir/depend:
	cd /home/alfeak/lidar/workspace/implement/yolo_tensorrt_infer_module/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/alfeak/lidar/workspace/implement/yolo_tensorrt_infer_module /home/alfeak/lidar/workspace/implement/yolo_tensorrt_infer_module /home/alfeak/lidar/workspace/implement/yolo_tensorrt_infer_module/build /home/alfeak/lidar/workspace/implement/yolo_tensorrt_infer_module/build /home/alfeak/lidar/workspace/implement/yolo_tensorrt_infer_module/build/CMakeFiles/yolo_infer.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/yolo_infer.dir/depend

