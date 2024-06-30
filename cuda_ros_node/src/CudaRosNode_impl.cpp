#include "CudaRosNode_impl.hpp"

// Forward declaration of cuda lib functions
// void cudaAdd(int *c);
// void cudaPrintDeviceProperties();

namespace cuda_ros_node
{
CudaRosNode::CudaRosNode(const rclcpp::NodeOptions& options) : Node("cuda_ros_node", options)
{
    publisher_ = this->create_publisher<std_msgs::msg::Float32>("/CudaOutputTopic", 10);
    timer_ = this->create_wall_timer(std::chrono::seconds(1), std::bind(&CudaRosNode::timer_callback, this));
    count_ = 0;
    // Print the device properties
    cudaPrintDeviceProperties();
}

void CudaRosNode::timer_callback()
{
    auto msg = std_msgs::msg::Float32();
    // Run the cuda code
    int* ptr = (int*)malloc(sizeof(int));
    cudaAdd(ptr);
    msg.data = static_cast<float>(*ptr);
    publisher_->publish(msg);
    // Cleanup
    free(ptr);
}
awefawef
} // namespace cuda_ros_node

// Register the class with the component factory
#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(cuda_ros_node::CudaRosNode)