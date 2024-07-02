#include <chrono>
#include <functional>
#include <memory>
#include <string>

#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/int64_multi_array.hpp>

namespace cuda_ros_node {
    // Cuda functions prototypes so we can use the functions defined in the Cuda lib
    void cudaAdd(int *a, int *b, int *c, int arrSize);
    void cudaPrintDeviceProperties();

    class CudaRosNode : public rclcpp::Node
    {
        public:
            CudaRosNode(const rclcpp::NodeOptions& options);

        private:
            void timer_callback();
            rclcpp::TimerBase::SharedPtr timer_;
            rclcpp::Publisher<std_msgs::msg::Int64MultiArray>::SharedPtr publisher_;
    };
}
