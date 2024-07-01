#include "CudaRosNode_impl.hpp"
#include <vector>

namespace cuda_ros_node
{
CudaRosNode::CudaRosNode(const rclcpp::NodeOptions& options) : Node("cuda_ros_node", options)
{
    publisher_ = this->create_publisher<std_msgs::msg::Int16MultiArray>("/CudaOutputTopic", 10);
    timer_ = this->create_wall_timer(std::chrono::seconds(1), std::bind(&CudaRosNode::timer_callback, this));
    count_ = 0;
    // Print the device properties
    cudaPrintDeviceProperties();
}

void CudaRosNode::timer_callback()
{
    auto msg = std_msgs::msg::Int16MultiArray();
    // Run the cuda code
    int size = 10;
    // Arrays to store our data to be added together
    std::vector<int> a, b;
    // Array to store our output
    std::vector<int> c(size, 0);
    // Fill the arrays
    for (int i = 0; i < size; i++)
    {
        a.push_back(-i);
        b.push_back(i * i);
    }
    cudaAdd(&a[0], &b[0], &c[0], a.size());
    // Add the summed vector to our message
    for (auto val : c)
    {
        msg.data.push_back(val);
    }
    publisher_->publish(msg);
}
} // namespace cuda_ros_node

// Register the class with the component factory
#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(cuda_ros_node::CudaRosNode)