#include "CudaRosNode_impl.hpp"
#include <cstdio>

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    rclcpp::executors::StaticSingleThreadedExecutor executor;
    rclcpp::NodeOptions options;

    auto pCudaNode = std::make_shared<cuda_ros_node::CudaRosNode>(options);
    executor.add_node(pCudaNode);
    executor.spin();

    rclcpp::shutdown();
    return 0;
}