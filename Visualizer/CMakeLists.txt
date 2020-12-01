# Add the environment
add_custom_command(
        OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/CondaEnvironment"
        COMMAND
        conda env create --prefix "${CMAKE_CURRENT_BINARY_DIR}/CondaEnvironment" --file "${CMAKE_CURRENT_SOURCE_DIR}/environment.yml"
        && "${CMAKE_CURRENT_BINARY_DIR}/CondaEnvironment/bin/jupyter" labextension install @jupyter-widgets/jupyterlab-manager
        && "${CMAKE_CURRENT_BINARY_DIR}/CondaEnvironment/bin/jupyter" labextension install @jupyterlab/toc
        && "${CMAKE_CURRENT_BINARY_DIR}/CondaEnvironment/bin/jupyter" labextension install jupyter-matplotlib
)

# Set up the target
add_custom_target(
        ${PROJECT_NAME}-Visualizer
        DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/CondaEnvironment"
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/Source"
)