import argparse
import matplotlib.pyplot as plt
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import JointState
import rosbag2_py

def extract_and_plot(db3_path, target_topic='/rx200/joint_states'):
    # 1. Configure for SQLite3 (.db3)
    storage_options = rosbag2_py.StorageOptions(
        uri=db3_path,
        storage_id='sqlite3'  # <--- Explicitly set for .db3 files
    )
    
    converter_options = rosbag2_py.ConverterOptions(
        input_serialization_format='cdr',
        output_serialization_format='cdr'
    )

    reader = rosbag2_py.SequentialReader()
    
    try:
        reader.open(storage_options, converter_options)
    except Exception as e:
        print(f"Error opening bag: {e}")
        print("Tip: Make sure you point to the .db3 file, not just the folder.")
        return

    # 2. Filter for joint states
    storage_filter = rosbag2_py.StorageFilter(topics=[target_topic])
    reader.set_filter(storage_filter)

    joint_data = {}
    start_time = None

    print(f"Reading from {db3_path}...")
    
    # 3. Read loop
    while reader.has_next():
        (topic, data, t) = reader.read_next()
        msg = deserialize_message(data, JointState)

        # Time handling
        # Convert ROS Time (seconds + nanoseconds) to float seconds
        timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        
        if start_time is None:
            start_time = timestamp
        
        rel_time = timestamp - start_time

        # Extract positions, velocities, efforts
        for i, name in enumerate(msg.name):
            if name not in joint_data:
                joint_data[name] = {"t": [], "pos": [], "vel": [], "eff": []}
            
            joint_data[name]["t"].append(rel_time)
            
            # Safe append (check bounds in case hardware doesn't report vel/eff)
            if i < len(msg.position):
                joint_data[name]["pos"].append(msg.position[i])
            if i < len(msg.velocity):
                joint_data[name]["vel"].append(msg.velocity[i])
            if i < len(msg.effort):
                joint_data[name]["eff"].append(msg.effort[i])

    if not joint_data:
        print(f"No data found on topic {target_topic}")
        return

    print(f"Plotting {len(joint_data)} joints...")
    plot_data(joint_data)

def plot_data(joint_data):
    fig, axs = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    
    # 1. Position
    for name, data in joint_data.items():
        if data["pos"]:
            axs[0].plot(data["t"], data["pos"], label=name)
    axs[0].set_ylabel('Position (rad or m)')
    axs[0].set_title('Joint Positions')
    axs[0].grid(True)
    # Legend only on the top plot to save space
    axs[0].legend(loc='upper right', bbox_to_anchor=(1.25, 1.0))

    # 2. Velocity
    for name, data in joint_data.items():
        if data["vel"]:
            axs[1].plot(data["t"], data["vel"], label=name)
    axs[1].set_ylabel('Velocity (rad/s)')
    axs[1].set_title('Joint Velocities')
    axs[1].grid(True)

    # 3. Effort
    for name, data in joint_data.items():
        if data["eff"]:
            axs[2].plot(data["t"], data["eff"], label=name)
    axs[2].set_ylabel('Effort (Nm)')
    axs[2].set_title('Joint Efforts')
    axs[2].set_xlabel('Time (s)')
    axs[2].grid(True)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot .db3 joint states")
    parser.add_argument("db3_file", help="Path to the .db3 file inside the bag folder")
    args = parser.parse_args()

    extract_and_plot(args.db3_file)