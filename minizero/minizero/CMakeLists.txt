file(GLOB SRCS *.cpp)

add_executable(minizero ${SRCS})
target_link_libraries(
    minizero
    actor
    config
    console
    environment
    network
    utils
    zero
    ${TORCH_LIBRARIES}
    /home/cthsu/miniconda3/envs/ml_hw2/lib/python3.10/site-packages/nvidia/cuda_runtime/lib/libcudart.so.12
)