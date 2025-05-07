import os
import numpy as np
from tumor_analyzer import EllipsoidFitter


def sample_within_ellipsoid(ellipsoid_model, num_samples=1, random_state=None):
    """
    从椭球体内部均匀采样点
    
    参数:
        ellipsoid_model: EllipsoidFitter实例
        num_samples: 需要采样的点数量
        random_state: 随机数种子
    
    返回:
        采样的点坐标数组 shape=(num_samples, 3)
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    center = ellipsoid_model.center
    axes = ellipsoid_model.axes
    radii = ellipsoid_model.radii
    
    # 生成均匀分布的球面坐标
    samples = []
    for _ in range(num_samples):
        # 生成随机方向
        phi = np.random.uniform(0, 2 * np.pi)
        costheta = np.random.uniform(-1, 1)
        theta = np.arccos(costheta)
        
        # 生成随机半径 (r^3分布确保体积均匀性)
        u = np.random.uniform(0, 1)
        r = u ** (1 / 3)
        
        # 转换为笛卡尔坐标
        x = r * np.sin(theta) * np.cos(phi)
        y = r * np.sin(theta) * np.sin(phi)
        z = r * np.cos(theta)
        
        # 变换到椭球体坐标系
        point = center + np.dot(np.array([x, y, z]) * radii, axes)
        samples.append(point)
    
    return np.array(samples)


def get_absolute_coordinate(relative_coordinate, original_shape, target_volume, start):
    """
    将相对坐标转换为绝对坐标
    
    参数:
        relative_coordinate: 相对坐标
        original_shape: 原始体积形状
        target_volume: 目标体积形状
        start: 裁剪起始点
        
    返回:
        绝对坐标
    """
    x_ratio = original_shape[0] / target_volume[0]
    y_ratio = original_shape[1] / target_volume[1]
    z_ratio = original_shape[2] / target_volume[2]

    absolute_x = relative_coordinate[0] * x_ratio
    absolute_y = relative_coordinate[1] * y_ratio
    absolute_z = relative_coordinate[2] * z_ratio

    absolute_x += start[0]
    absolute_y += start[1]
    absolute_z += start[2]

    return np.array([absolute_x, absolute_y, absolute_z], dtype=float)


def is_edge_point(mask, point, edge_op="erosion", neighborhood_size=(3, 3, 3), volume_threshold=5,
                 sobel_threshold=400, erosion_kernel_size=5):
    """
    判断点是否在边缘
    
    参数:
        mask: 3D掩码数组
        point: 点坐标
        edge_op: 边缘检测方法，可选值包括：'erosion', 'volume', 'sobel', 'any', 'both', 'volume_erosion', 'sobel_erosion', 'all', 'none'
        neighborhood_size: 检查体积时的邻域大小
        volume_threshold: 体积阈值
        sobel_threshold: 梯度阈值
        erosion_kernel_size: 腐蚀核大小
        
    返回:
        是否为边缘点
    """
    from scipy import ndimage
    import cv2
    
    point = np.round(point).astype(int)
    
    # 确保点在掩码范围内
    if not (0 <= point[0] < mask.shape[0] and 
            0 <= point[1] < mask.shape[1] and 
            0 <= point[2] < mask.shape[2]):
        return True
    
    # 确保点在器官内
    if mask[tuple(point)] != 1:
        return True
    
    def check_volume():
        # 定义邻域边界
        min_bounds = np.maximum(point - np.array(neighborhood_size) // 2, 0)
        max_bounds = np.minimum(point + np.array(neighborhood_size) // 2, np.array(mask.shape) - 1)

        # 提取邻域体积
        neighborhood_volume = mask[min_bounds[0]:max_bounds[0] + 1,
                                  min_bounds[1]:max_bounds[1] + 1,
                                  min_bounds[2]:max_bounds[2] + 1]

        # 计算肝脏体素数量
        liver_voxel_count = np.sum(neighborhood_volume == 1)

        # 检查是否低于阈值
        return liver_voxel_count < volume_threshold
        
    def check_sobel():
        # 计算当前点邻域的边界
        min_bounds = np.maximum(point - np.array([7, 7, 7]) // 2, 0)
        max_bounds = np.minimum(point + np.array([7, 7, 7]) // 2, np.array(mask.shape) - 1)

        # 提取邻域子区域
        neighborhood_sn = mask[min_bounds[0]:max_bounds[0] + 1,
                              min_bounds[1]:max_bounds[1] + 1,
                              min_bounds[2]:max_bounds[2] + 1]

        # 仅在邻域区域计算Sobel滤波
        sobel_x = ndimage.sobel(neighborhood_sn, axis=0)
        sobel_y = ndimage.sobel(neighborhood_sn, axis=1)
        sobel_z = ndimage.sobel(neighborhood_sn, axis=2)

        # 计算梯度幅值
        gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2 + sobel_z**2)

        # 获取邻域中心点的梯度幅值
        central_point = np.array([7, 7, 7]) // 2
        gradient_value = gradient_magnitude[tuple(central_point)]

        # 如果梯度幅值超过阈值，则返回True
        return gradient_value > sobel_threshold
        
    def check_erosion():
        # 获取当前点所在的z轴位置
        z = point[2]
        # 确保z在有效范围内
        if z < 0 or z >= mask.shape[2]:
            return True  # 如果在边界外，认为是边缘点
            
        # 获取z轴切片
        mask_slice = mask[..., z].copy()  # 使用copy避免修改原始掩码
        
        # 创建腐蚀核
        kernel = np.ones((erosion_kernel_size, erosion_kernel_size), dtype=np.uint8)
        
        # 腐蚀肝脏掩码
        eroded_mask = cv2.erode(mask_slice.astype(np.uint8), kernel, iterations=1)
        
        # 检查当前点在原始掩码中是肝脏点(值为1)，但在腐蚀后的掩码中不是肝脏点(值为0)
        x, y = point[0], point[1]
        # 确保x,y在有效范围内
        if x < 0 or x >= mask.shape[0] or y < 0 or y >= mask.shape[1]:
            return True
            
        return mask_slice[x, y] == 1 and eroded_mask[x, y] == 0

    # 根据选择的操作模式判断是否为边缘点
    if edge_op == "volume":
        return check_volume()
    elif edge_op == "sobel":
        return check_sobel()
    elif edge_op == "erosion":
        return check_erosion()
    elif edge_op == "any":
        return check_volume() and check_sobel()
    elif edge_op == "both":
        return check_volume() or check_sobel()
    elif edge_op == "volume_erosion":
        return check_volume() or check_erosion()
    elif edge_op == "sobel_erosion":
        return check_sobel() or check_erosion()
    elif edge_op == "all":
        return check_volume() or check_sobel() or check_erosion()
    elif edge_op == "none":
        return False
    else:
        raise ValueError("Invalid edge_op option. Choose from 'volume', 'sobel', 'erosion', 'any', 'both', 'volume_erosion', 'sobel_erosion', 'all', or 'none'.")


def is_within_middle_z_range(point, z_start, z_end, margin_ratio=0.2):
    """
    判断点是否在z轴的中间区域
    
    参数:
        point: 点坐标
        z_start: z轴起始位置
        z_end: z轴结束位置
        margin_ratio: 边缘比例
        
    返回:
        是否在中间区域
    """
    z = point[2]
    z_range = z_end - z_start
    z_margin = z_range * margin_ratio
    
    return (z_start + z_margin) <= z <= (z_end - z_margin)


def optimized_ellipsoid_select(mask_scan, ellipsoid_model, max_attempts=1000, edge_op="erosion", 
                              middle_z_only=False, random_state=None):
    """
    基于优化椭球体模型选择肿瘤点位置
    
    参数:
        mask_scan: 3D掩码数组
        ellipsoid_model: 椭球体模型
        max_attempts: 最大尝试次数
        edge_op: 边缘检测方法
        middle_z_only: 是否只在z轴中间区域选择
        random_state: 随机数种子
        
    返回:
        选择的点坐标
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    # 找到肝脏区域边界
    x_start, x_end = np.where(np.any(mask_scan, axis=(1, 2)))[0][[0, -1]]
    y_start, y_end = np.where(np.any(mask_scan, axis=(0, 2)))[0][[0, -1]]
    z_start, z_end = np.where(np.any(mask_scan, axis=(0, 1)))[0][[0, -1]]

    # 缩小边界以避免边缘
    x_start, x_end = max(0, x_start + 1), min(mask_scan.shape[0], x_end - 1)
    y_start, y_end = max(0, y_start + 1), min(mask_scan.shape[1], y_end - 1)
    z_start, z_end = max(0, z_start + 1), min(mask_scan.shape[2], z_end - 1)

    # 提取肝脏掩码
    liver_mask = mask_scan[x_start:x_end, y_start:y_end, z_start:z_end]
    target_volume = (300, 250, 140)
    start = (x_start, y_start, z_start)

    # 批量生成多个候选点以提高效率
    batch_size = min(100, max_attempts // 10)
    
    attempts = 0
    while attempts < max_attempts:
        # 从椭球体模型批量采样
        candidate_points = sample_within_ellipsoid(ellipsoid_model, num_samples=batch_size)
        
        for potential_point in candidate_points:
            if attempts >= max_attempts:
                break
                
            # 将相对坐标转换为绝对坐标
            absolute_point = get_absolute_coordinate(
                potential_point, liver_mask.shape, target_volume, start)
            
            # 确保点在掩码边界内
            if any(coord < 0 or coord >= dim for coord, dim in zip(absolute_point, mask_scan.shape)):
                attempts += 1
                continue
                
            # 四舍五入并转换为整数
            absolute_point = np.round(absolute_point).astype(int)
            
            # 检查点是否在器官内且不在边缘
            if mask_scan[tuple(absolute_point)] == 1:
                if not is_edge_point(mask_scan, absolute_point, edge_op):
                    if not middle_z_only or is_within_middle_z_range(absolute_point, z_start, z_end):
                        return absolute_point
            
            attempts += 1
    
    # 如果达到最大尝试次数仍未找到合适点，使用随机选择
    print("警告：达到最大尝试次数，使用随机选择")
    return random_select(mask_scan)


def random_select(mask_scan, max_attempts=500):
    """
    随机选择肿瘤点位置（备用方法）
    
    参数:
        mask_scan: 3D掩码数组
        max_attempts: 最大尝试次数
        
    返回:
        选择的点坐标
    """
    # 找到所有值为1的点
    points = np.argwhere(mask_scan == 1)
    
    # 如果没有符合条件的点，返回None
    if len(points) == 0:
        return None
    
    # 随机选择一个点
    idx = np.random.randint(0, len(points))
    point = points[idx]
    
    # 确保不在边缘
    attempts = 0
    while is_edge_point(mask_scan, point) and attempts < max_attempts:
        idx = np.random.randint(0, len(points))
        point = points[idx]
        attempts += 1
    
    return point


def demo_sampling(ellipsoid_model, n_samples=10):
    """
    演示椭球体采样
    
    参数:
        ellipsoid_model: 椭球体模型
        n_samples: 采样点数量
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    
    # 采样点
    samples = sample_within_ellipsoid(ellipsoid_model, num_samples=n_samples)
    
    # 绘制椭球体和采样点
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # 绘制采样点
    ax.scatter(samples[:, 0], samples[:, 1], samples[:, 2], 
               color='blue', s=50, label='采样点')
    
    # 绘制椭球体中心
    center = ellipsoid_model.center
    ax.scatter([center[0]], [center[1]], [center[2]], 
               color='red', s=100, label='椭球体中心')
    
    # 绘制椭球体
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 50)
    
    radii = ellipsoid_model.radii
    axes = ellipsoid_model.axes
    
    x = radii[0] * np.outer(np.cos(u), np.sin(v))
    y = radii[1] * np.outer(np.sin(u), np.sin(v))
    z = radii[2] * np.outer(np.ones_like(u), np.cos(v))
    
    for i in range(len(x)):
        for j in range(len(x[i])):
            [x[i,j], y[i,j], z[i,j]] = center + np.dot([x[i,j], y[i,j], z[i,j]], axes)
    
    ax.plot_surface(x, y, z, color='red', alpha=0.1)
    
    # 设置坐标轴标签和图例
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    
    plt.title('椭球体内均匀采样演示')
    plt.show()


def main():
    # 演示采样函数使用方法
    from tumor_analyzer import EllipsoidFitter
    
    # 默认参数 (这些应该替换为优化后的参数)
    center = [190, 140, 90]
    axes = [[-0.8, 0.6, 0.2], 
        [0.6, 0.8, -0.2], 
        [-0.3, 0.0, -1.0]]
    radii = [230, 130, 80]
    
    # 加载椭球体模型
    model_path = "models/optimized_ellipsoid_model.py"
    if os.path.exists(model_path):
        print(f"加载已优化的椭球体模型: {model_path}")
        # 动态导入模型加载函数
        import importlib.util
        spec = importlib.util.spec_from_file_location("ellipsoid_module", model_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        # 调用加载函数
        ellipsoid_model = module.load_optimized_ellipsoid_model()
    else:
        print(f"未找到优化模型，使用默认参数...")
        normalized_axes = np.zeros_like(axes)
        for i in range(3):
            normalized_axes[i] = axes[i] / np.linalg.norm(axes[i])
        ellipsoid_model = EllipsoidFitter.from_precomputed_parameters(center, normalized_axes, radii)
    
    # 演示采样
    demo_sampling(ellipsoid_model, n_samples=30)
    
    print("采样点示例:")
    samples = sample_within_ellipsoid(ellipsoid_model, num_samples=5)
    for i, sample in enumerate(samples):
        print(f"采样点 {i+1}: {sample}")


if __name__ == "__main__":
    main() 