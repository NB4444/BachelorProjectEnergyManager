# Find the NVIDIA Code Samples
set(NVIDIA_CODE_SAMPLES_DIRECTORY "/home/xqbakker/code-samples")
if (NOT EXISTS ${NVIDIA_CODE_SAMPLES_DIRECTORY})
    set(NVIDIA_CODE_SAMPLES_DIRECTORY "/home/qub1/code-samples")
endif ()
add_compile_definitions(NVIDIA_CODE_SAMPLES_DIRECTORY="${NVIDIA_CODE_SAMPLES_DIRECTORY}")
message(STATUS "NVIDIA code samples directory: ${NVIDIA_CODE_SAMPLES_DIRECTORY}")