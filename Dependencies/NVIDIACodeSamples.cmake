# Find the NVIDIA Code Samples
set(NVIDIA_CODE_SAMPLES_DIRECTORY "$ENV{HOME}/code-samples")
add_compile_definitions(NVIDIA_CODE_SAMPLES_DIRECTORY="${NVIDIA_CODE_SAMPLES_DIRECTORY}")
message(STATUS "NVIDIA code samples directory: ${NVIDIA_CODE_SAMPLES_DIRECTORY}")

set(NVIDIA_DEVELOPER_BLOG "/applications/nvidia-developer-blog/code-samples")
add_compile_definitions(NVIDIA_DEVELOPER_BLOG="${NVIDIA_DEVELOPER_BLOG}")
message(STATUS "NVIDIA developer blog code samples directory: ${NVIDIA_DEVELOPER_BLOG}")