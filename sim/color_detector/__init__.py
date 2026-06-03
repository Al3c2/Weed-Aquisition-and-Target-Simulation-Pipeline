class BlueObjectDetector(Node):
    def __init__(self):
        super().__init__('blue_object_detector')

        self.declare_parameter(
            'rgb_topic',
            '/world/empty/model/swincar_ur3/model/ur3/link/base_link/sensor/rgbd_camera/image'
        )
        self.declare_parameter(
            'depth_topic',
            '/world/empty/model/swincar_ur3/model/ur3/link/base_link/sensor/rgbd_camera/depth_image'
        )
        self.declare_parameter(
            'camera_info_topic',
            '/world/empty/model/swincar_ur3/model/ur3/link/base_link/sensor/rgbd_camera/camera_info'
        )
        
        # Frame names
        self.declare_parameter('camera_frame', 'camera_optical_link')
        self.declare_parameter('target_frame', 'world')
        self.declare_parameter('primary_target_topic', '/blue_target_primary')

        rgb_topic = self.get_parameter('rgb_topic').get_parameter_value().string_value
        depth_topic = self.get_parameter('depth_topic').get_parameter_value().string_value
        cam_info_topic = self.get_parameter('camera_info_topic').get_parameter_value().string_value
        self.camera_frame = self.get_parameter('camera_frame').get_parameter_value().string_value
        self.target_frame = self.get_parameter('target_frame').get_parameter_value().string_value
        primary_topic = self.get_parameter('primary_target_topic').get_parameter_value().string_value

        self.get_logger().info(f"RGB topic:       {rgb_topic}")
        self.get_logger().info(f"Depth topic:     {depth_topic}")
        self.get_logger().info(f"Camera info:     {cam_info_topic}")
        self.get_logger().info(f"Camera frame:    {self.camera_frame}")
        self.get_logger().info(f"Target frame:    {self.target_frame}")
        self.get_logger().info(f"Primary out pub: {primary_topic}")
                # -------- Intrinsics --------
        self.fx = None
        self.fy = None
        self.cx = None
        self.cy = None
        self.cam_info_width = None
        self.cam_info_height = None
        self.intrinsics_scaled = False

        # Intrinsics
        self.fx = self.fy = self.cx = self.cy = None

        self.latest_depth = None
        self.latest_depth_stamp = None

        self.bridge = CvBridge()

        # TF buffer + listener
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Subs / pubs
        self.rgb_sub = self.create_subscription(Image, rgb_topic, self.rgb_callback, 10)
        self.depth_sub = self.create_subscription(Image, depth_topic, self.depth_callback, 10)
        self.cam_info_sub = self.create_subscription(CameraInfo, cam_info_topic, self.cam_info_callback, 10)

        self.primary_target_pub = self.create_publisher(PointStamped, primary_topic, 10)
