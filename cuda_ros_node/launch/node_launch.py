from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

PACKAGE="cuda_ros_node"

## define custom tag handler
def join(loader, node):
    seq = loader.construct_sequence(node)
    return ''.join([str(i) for i in seq])

def generate_launch_description():

    # Each node call is a new object in the launch description array

    ld = LaunchDescription(
        [
            Node(
                package="cuda_ros_node",
                executable="CudaRosNode_exe",
                name="CudaRosNode",
                output="screen",
                emulate_tty=True,
                #parameters = [nothing for now]
                arguments=["--ros-args", "--log-level", "info"],
            )
        ]
    )
    return ld

