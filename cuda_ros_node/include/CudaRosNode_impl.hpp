#include <chrono>
#include <functional>
#include <memory>
#include <string>

#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/int64_multi_array.hpp>
#include <std_msgs/msg/float32.hpp>

namespace cuda_ros_node {
    // Cuda functions prototypes so we can use the functions defined in the Cuda lib
    void cudaAdd(int *a, int *b, int *c, int arrSize);
    float cudaDot(float* a, float* b, int arrSize);
    void cudaPrintDeviceProperties();

    class CudaRosNode : public rclcpp::Node
    {
        public:
            CudaRosNode(const rclcpp::NodeOptions& options);

        private:
            void addTimerCallback();
            rclcpp::TimerBase::SharedPtr addTimer;
            rclcpp::Publisher<std_msgs::msg::Int64MultiArray>::SharedPtr addPublisher;
            void dotTimerCallback();
            rclcpp::TimerBase::SharedPtr dotTimer;
            rclcpp::Publisher<std_msgs::msg::Float32>::SharedPtr dotPublisher;
    };
}
