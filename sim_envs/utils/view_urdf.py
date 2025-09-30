import pybullet as p
import pybullet_data
import time
import os

from pybullet_utils import UrdfObject

class URDFViewer:
    """
    一个用于加载和可视化URDF文件的简单PyBullet查看器。
    """
    def __init__(self, window_title="URDF Viewer"):
        """
        初始化PyBullet仿真环境。
        """
        try:
            # 尝试连接到GUI，如果已有连接则复用
            self.client_id = p.connect(p.GUI, options=f'--title={window_title}')
            print("成功连接到PyBullet GUI。")
        except p.error:
            # 如果连接失败（例如，在无头服务器上），则退出
            print("错误: 无法连接到PyBullet GUI。请确保您在有图形界面的环境中运行。")
            exit()
            
        # 设置默认环境
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.8)
        p.setRealTimeSimulation(0) # 关闭实时模拟，我们手动步进
        
        # 加载地面
        p.loadURDF("plane.urdf")
        
        # 设置一个合适的观察视角
        p.resetDebugVisualizerCamera(
            cameraDistance=1.5,
            cameraYaw=30,
            cameraPitch=-30,
            cameraTargetPosition=[0, 0, 0.5]
        )
        
        # 用于存储加载的物体ID
        self.loaded_objects = {}
        print("查看器已初始化，按 Ctrl+C 或关闭窗口退出。")

    def load_urdf(self, urdf_path, position, orientation_euler=(0, 0, 0), scale=1.0, use_fixed_base=True):
        """
        加载一个URDF文件到场景中。

        参数:
            urdf_path (str): URDF文件的路径。
            position (list/tuple): 物体的初始位置 [x, y, z]。
            orientation_euler (list/tuple): 物体的初始欧拉角姿态 [roll, pitch, yaw]。
            scale (float): 模型的全局缩放比例。
            use_fixed_base (bool): 是否将基座固定在世界上。

        返回:
            int: 加载的物体的唯一ID, 如果失败则返回-1。
        """
        if not os.path.exists(urdf_path):
            print(f"错误: 找不到文件 '{urdf_path}'")
            return -1

        try:
            # 将欧拉角转换为四元数
            orientation_quat = p.getQuaternionFromEuler(orientation_euler)
            
            # 加载URDF
            obj_id = p.loadURDF(
                fileName=urdf_path,
                basePosition=position,
                baseOrientation=orientation_quat,
                globalScaling=scale,
                useFixedBase=use_fixed_base,
                physicsClientId=self.client_id
            )

            urdf_object = UrdfObject(obj_id)
            urdf_object.reset_base_pose(position, orientation_quat)
            
            self.loaded_objects[obj_id] = urdf_path
            print(f"成功加载 '{urdf_path}', 物体ID为: {obj_id}")
            return obj_id, urdf_object
            
        except p.error as e:
            print(f"加载 '{urdf_path}' 失败: {e}")
            return -1, -1

    def run(self):
        """
        运行仿真循环，保持窗口开启。
        """
        try:
            while p.isConnected(self.client_id):
                p.stepSimulation()
                time.sleep(1./240.)
        except p.error:
            # 当用户关闭窗口时，p.isConnected会抛出异常
            print("\nPyBullet窗口已关闭。")
        finally:
            self.close()

    def close(self):
        """
        断开与PyBullet的连接。
        """
        if p.isConnected(self.client_id):
            p.disconnect(self.client_id)
            print("已从PyBullet断开连接。")


# --- 主程序入口 ---
if __name__ == "__main__":
    dir_path = os.path.dirname(os.path.realpath(__file__))
    asset_path = os.path.join(os.path.dirname(dir_path), "assets")
    
    # 1. 实例化查看器
    viewer = URDFViewer(window_title="URDF Viewer")

    # 2. 加载你的第一个（主要）URDF文件
    # !!! 修改为你自己URDF文件的路径 !!!
    my_urdf_file = os.path.join(asset_path, "needle/needle_40mm.urdf") # 这是一个pybullet_data自带的例子，你可以换成自己的路径

    target_pos = [0, 0, 0.5]  # 你想要放置物体的位置
    target_orn = [0, 0, 0]    # 物体的欧拉角姿态

    obj_id, urdf_object = viewer.load_urdf(
        urdf_path=my_urdf_file,
        position=target_pos,
        orientation_euler=target_orn,
        scale=1,
        use_fixed_base=True
    )

    read_pos, read_orn = urdf_object.get_base_pose()

    # [验证]
    print("\n--- 验证 ---")
    print(f"设置的基座位置: {[f'{x:.4f}' for x in target_pos]}")
    print(f"读取的基座位置: {[f'{x:.4f}' for x in read_pos]}")

    # [对比]
    # 直接用pybullet API读取的是重心位置，你会看到它和我们读取的位置不同
    com_pos, _ = p.getBasePositionAndOrientation(urdf_object.obj_id)
    print(f"PyBullet API直接读取的重心位置: {[f'{x:.4f}' for x in com_pos]}")

    # 4. 运行查看器，直到窗口被关闭
    viewer.run()