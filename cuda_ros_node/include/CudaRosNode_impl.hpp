#include <chrono>
#include <functional>
#include <memory>

#include "CPUBitmap.h"

#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/int64_multi_array.hpp>
#include <std_msgs/msg/float32.hpp>
#include <sensor_msgs/msg/image.hpp>

namespace cuda_ros_node {
    // Cuda functions prototypes so we can use the functions defined in the Cuda lib
    void cudaAdd(int *a, int *b, int *c, int arrSize);
    float cudaDot(float* a, float* b, int arrSize);
    CPUBitmap* rayTrace();
    void cudaPrintDeviceProperties();
    void compute_histogram(unsigned char* inputBuffer, std::size_t inputBufferSize, unsigned int* histogram);

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
            void rayTraceTimerCallback();
            rclcpp::TimerBase::SharedPtr rayTraceTimer;
            rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr rayTraceImgPublisher;
            void computeHistogramTimerCallback();
            rclcpp::TimerBase::SharedPtr computeHistogramTimer;
            rclcpp::Publisher<std_msgs::msg::Int64MultiArray>::SharedPtr computeHistogramPublisher;
    };
}
