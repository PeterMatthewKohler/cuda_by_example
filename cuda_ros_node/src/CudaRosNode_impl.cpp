#include "CudaRosNode_impl.hpp"
#include <vector>

namespace cuda_ros_node
{
CudaRosNode::CudaRosNode(const rclcpp::NodeOptions& options) : Node("cuda_ros_node", options)
{
    // Create our vector addition publisher and callback
    addPublisher = this->create_publisher<std_msgs::msg::Int64MultiArray>("/CudaAddOutputTopic", 10);
    addTimer = this->create_wall_timer(std::chrono::seconds(1), std::bind(&CudaRosNode::addTimerCallback, this));
    // Create our dot product publisher and callback
    dotPublisher = this->create_publisher<std_msgs::msg::Float32>("/CudaDotOutputTopic", 10);
    dotTimer = this->create_wall_timer(std::chrono::seconds(1), std::bind(&CudaRosNode::dotTimerCallback, this));
    // Print the device properties
    cudaPrintDeviceProperties();
}

void CudaRosNode::addTimerCallback()
{
    auto msg = std_msgs::msg::Int64MultiArray();
    // Run the cuda code
    int size = 1025;    // # Threads in block for my 3070 is 1024
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
    addPublisher->publish(msg);
}

void CudaRosNode::dotTimerCallback()
{
    auto msg = std_msgs::msg::Float32();
    // Run the cuda code
    // Arrays to store our data
    std::vector<float> a = {1, 2, 3, 4}; 
    std::vector<float> b = {4, 5, 6, 7};
    msg.data = cudaDot(&a[0], &b[0], a.size());
    dotPublisher->publish(msg);
}
} // namespace cuda_ros_node

// Register the class with the component factory
#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(cuda_ros_node::CudaRosNode)