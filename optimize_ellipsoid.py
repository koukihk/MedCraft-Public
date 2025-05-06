import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.optimize import minimize
import pickle
import argparse
from mpl_toolkits.mplot3d import Axes3D

from tumor_analyzer import TumorAnalyzer, EllipsoidFitter


def round_to_tens(value):
    """将大于10的数值四舍五入到最近的10的倍数"""
    if abs(value) > 10:
        return round(value / 10) * 10
    return value


def optimize_ellipsoid(tumor_positions, coverage_weight=0.6, compactness_weight=0.4, verbose=True):
    """
    使用直接优化方法找到最佳椭球体模型
    
    参数:
        tumor_positions: 肿瘤位置点的数组
        coverage_weight: 覆盖率权重
        compactness_weight: 紧凑性权重
        verbose: 是否显示详细信息
    
    返回:
        优化后的椭球体参数(中心点, 轴, 半径)
    """
    # 计算初始中心点和主轴
    mean_pos = np.mean(tumor_positions, axis=0)
    
    # 中心化数据
    centered_data = tumor_positions - mean_pos
    
    # 计算协方差矩阵
    cov_matrix = np.cov(centered_data, rowvar=False)
    
    # 计算特征值和特征向量
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    
    # 排序特征值和特征向量（从大到小）
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # 初始半径估计（基于特征值）
    initial_radii = np.sqrt(eigenvalues) * 3
    
    # 将初始参数放入优化器
    initial_params = np.concatenate([mean_pos, 
                                     eigenvectors[:, 0], 
                                     eigenvectors[:, 1], 
                                     eigenvectors[:, 2], 
                                     initial_radii])
    
    # 定义目标函数
    def objective(params):
        center = params[0:3]
        axes = params[3:12].reshape(3, 3)
        radii = params[12:15]
        
        # 计算椭球体内外的点
        distances = []
        for pos in tumor_positions:
            # 计算点到中心的向量
            centered_pos = pos - center
            
            # 将点投影到椭球体坐标系
            proj_x = np.dot(centered_pos, axes[0])
            proj_y = np.dot(centered_pos, axes[1])
            proj_z = np.dot(centered_pos, axes[2])
            
            # 计算归一化距离
            dist = np.sqrt((proj_x/radii[0])**2 + (proj_y/radii[1])**2 + (proj_z/radii[2])**2)
            distances.append(dist)
        
        distances = np.array(distances)
        
        # 计算覆盖率（椭球体内点的比例）
        coverage = np.mean(distances <= 1.0)
        
        # 计算紧凑性（椭球体体积的倒数）
        volume = (4/3) * np.pi * np.prod(radii)
        compactness = 1.0 / volume
        
        # 总体目标是最大化(权重×覆盖率 + 权重×紧凑性)
        objective_value = -(coverage_weight * coverage + compactness_weight * compactness)
        
        return objective_value
    
    # 约束条件：确保轴向量是单位向量
    def constraint_axis1(params):
        return np.sum(params[3:6]**2) - 1.0
    
    def constraint_axis2(params):
        return np.sum(params[6:9]**2) - 1.0
    
    def constraint_axis3(params):
        return np.sum(params[9:12]**2) - 1.0
    
    def constraint_orthogonal12(params):
        return np.abs(np.dot(params[3:6], params[6:9]))
    
    def constraint_orthogonal13(params):
        return np.abs(np.dot(params[3:6], params[9:12]))
    
    def constraint_orthogonal23(params):
        return np.abs(np.dot(params[6:9], params[9:12]))
    
    constraints = [
        {'type': 'eq', 'fun': constraint_axis1},
        {'type': 'eq', 'fun': constraint_axis2},
        {'type': 'eq', 'fun': constraint_axis3},
        {'type': 'eq', 'fun': constraint_orthogonal12},
        {'type': 'eq', 'fun': constraint_orthogonal13},
        {'type': 'eq', 'fun': constraint_orthogonal23}
    ]
    
    # 设置半径的边界（确保半径为正）
    bounds = [(None, None)] * 12 + [(1.0, None)] * 3
    
    if verbose:
        print("开始优化椭球体参数...")
    
    # 进行优化
    result = minimize(objective, initial_params, method='SLSQP', 
                      constraints=constraints, bounds=bounds,
                      options={'maxiter': 300, 'disp': verbose})
    
    if verbose:
        print(f"优化结果: {result.success}, 消息: {result.message}")
    
    # 提取优化后的参数
    opt_params = result.x
    opt_center = opt_params[0:3]
    opt_axes = opt_params[3:12].reshape(3, 3)
    opt_radii = opt_params[12:15]
    
    # 对大于10的半径取整到十位
    opt_radii = np.array([round_to_tens(r) for r in opt_radii])
    
    # 确保轴向量是正交的单位向量
    # Gram-Schmidt正交化
    v1 = opt_axes[0]
    v1 = v1 / np.linalg.norm(v1)
    
    v2 = opt_axes[1] - np.dot(opt_axes[1], v1) * v1
    v2 = v2 / np.linalg.norm(v2)
    
    v3 = opt_axes[2] - np.dot(opt_axes[2], v1) * v1 - np.dot(opt_axes[2], v2) * v2
    v3 = v3 / np.linalg.norm(v3)
    
    # 创建简化版的轴向量（1位小数）
    v1_simple = np.round(v1, 1)
    v2_simple = np.round(v2, 1)
    v3_simple = np.round(v3, 1)
    
    # 直接使用简化版的轴向量，但记得对其进行归一化处理
    v1_simple = v1_simple / np.linalg.norm(v1_simple)
    v2_simple = v2_simple / np.linalg.norm(v2_simple)
    v3_simple = v3_simple / np.linalg.norm(v3_simple)
    
    # 使用简化后的轴向量
    simple_axes = np.array([v1_simple, v2_simple, v3_simple])
    
    # 更简化的表示方式 - 四舍五入到仅保留0.1, 0.2等简单值
    # 将向量元素舍入到最接近的0.1
    def simplify_axis(v):
        # 四舍五入到1位小数
        v_rounded = np.round(v, 1)
        # 归一化
        return v_rounded / np.linalg.norm(v_rounded)
    
    # 应用简化
    v1_simple = simplify_axis(v1)
    v2_simple = simplify_axis(v2)
    v3_simple = simplify_axis(v3)
    
    # 确保第三个向量与前两个正交
    v3_simple = np.cross(v1_simple, v2_simple)
    v3_simple = v3_simple / np.linalg.norm(v3_simple)
    
    # 再次四舍五入以获得更简洁的表示
    v1_simple = np.round(v1_simple, 1)
    v2_simple = np.round(v2_simple, 1)
    v3_simple = np.round(v3_simple, 1)
    
    # 归一化
    v1_simple = v1_simple / np.linalg.norm(v1_simple)
    v2_simple = v2_simple / np.linalg.norm(v2_simple)
    v3_simple = v3_simple / np.linalg.norm(v3_simple)
    
    opt_axes = np.array([v1_simple, v2_simple, v3_simple])
    
    # 将中心点也取整到十位
    opt_center = np.array([round_to_tens(c) for c in opt_center]).astype(int)
    
    # 计算最终目标函数值
    final_params = np.concatenate([opt_center, opt_axes.flatten(), opt_radii])
    final_objective = objective(final_params)
    
    # 对显示和存储进行格式化，保留一位小数
    display_axes = np.zeros_like(opt_axes)
    for i in range(3):
        for j in range(3):
            display_axes[i, j] = round(opt_axes[i, j], 1)
    
    if verbose:
        print(f"最终目标函数值: {-final_objective}")
        print(f"最优中心点: {opt_center}")
        print(f"最优轴:\n{display_axes}")
        print(f"最优半径: {opt_radii}")
    
    return opt_center, display_axes, opt_radii


def validate_ellipsoid(tumor_positions, center, axes, radii):
    """
    验证椭球体模型的表现
    
    参数:
        tumor_positions: 肿瘤位置点的数组
        center: 椭球体中心点
        axes: 椭球体轴向量
        radii: 椭球体半径
    
    返回:
        覆盖率和紧凑性
    """
    # 确保轴向量是单位向量
    normalized_axes = np.zeros_like(axes)
    for i in range(3):
        normalized_axes[i] = axes[i] / np.linalg.norm(axes[i])
        
    ellipsoid = EllipsoidFitter.from_precomputed_parameters(center, normalized_axes, radii)
    
    distances = []
    for pos in tumor_positions:
        # 计算点到中心的向量
        centered_pos = pos - center
        
        # 将点投影到椭球体坐标系
        proj_x = np.dot(centered_pos, normalized_axes[0])
        proj_y = np.dot(centered_pos, normalized_axes[1])
        proj_z = np.dot(centered_pos, normalized_axes[2])
        
        # 计算归一化距离
        dist = np.sqrt((proj_x/radii[0])**2 + (proj_y/radii[1])**2 + (proj_z/radii[2])**2)
        distances.append(dist)
    
    distances = np.array(distances)
    
    # 计算覆盖率（椭球体内点的比例）
    coverage = np.mean(distances <= 1.0)
    
    # 计算紧凑性（椭球体体积的倒数）
    volume = (4/3) * np.pi * np.prod(radii)
    compactness = 1.0 / volume
    
    print(f"椭球体覆盖率: {coverage:.4f} ({int(coverage*100)}%)")
    print(f"椭球体体积: {volume:.2f}")
    print(f"椭球体紧凑性: {compactness:.8f}")
    
    return coverage, compactness


def plot_ellipsoid_with_data(tumor_positions, center, axes, radii, save_path=None):
    """
    绘制椭球体和数据点
    """
    # 确保轴向量是单位向量
    normalized_axes = np.zeros_like(axes)
    for i in range(3):
        normalized_axes[i] = axes[i] / np.linalg.norm(axes[i])
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # 绘制数据点
    ax.scatter(tumor_positions[:, 0], tumor_positions[:, 1], tumor_positions[:, 2], 
               color='blue', alpha=0.3, s=1, label='肿瘤位置')
    
    # 绘制椭球体中心
    ax.scatter([center[0]], [center[1]], [center[2]], color='red', s=50, label='椭球体中心')
    
    # 绘制椭球体
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 50)
    
    x = radii[0] * np.outer(np.cos(u), np.sin(v))
    y = radii[1] * np.outer(np.sin(u), np.sin(v))
    z = radii[2] * np.outer(np.ones_like(u), np.cos(v))
    
    for i in range(len(x)):
        for j in range(len(x[i])):
            [x[i,j], y[i,j], z[i,j]] = center + np.dot([x[i,j], y[i,j], z[i,j]], normalized_axes)
    
    ax.plot_surface(x, y, z, color='red', alpha=0.1)
    
    # 绘制椭球体主轴
    for i in range(3):
        axis_endpoint = center + normalized_axes[i] * radii[i]
        ax.plot([center[0], axis_endpoint[0]], 
                [center[1], axis_endpoint[1]], 
                [center[2], axis_endpoint[2]], 'k-', linewidth=2)
    
    # 设置坐标轴标签和图例
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    
    plt.title('椭球体拟合模型与肿瘤位置数据')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"图像已保存到 {save_path}")
    
    plt.show()


def save_ellipsoid_model(center, axes, radii, output_file):
    """保存椭球体模型参数"""
    # 确保轴向量是单位向量
    normalized_axes = np.zeros_like(axes)
    for i in range(3):
        normalized_axes[i] = axes[i] / np.linalg.norm(axes[i])
    
    model_data = {
        'center': center.tolist(),
        'axes': normalized_axes.tolist(),
        'radii': radii.tolist()
    }
    
    with open(output_file, 'wb') as f:
        pickle.dump(model_data, f)
    
    print(f"椭球体模型已保存到 {output_file}")
    
    # 创建更简洁的显示版本，仅保留一位小数
    display_axes = np.zeros_like(normalized_axes)
    for i in range(3):
        for j in range(3):
            display_axes[i, j] = round(normalized_axes[i, j], 1)
    
    # 同时生成一个Python函数代码文件，可以直接加载模型
    py_code = f"""
def load_optimized_ellipsoid_model():
    \"\"\"
    加载优化后的椭球体模型
    
    返回:
        EllipsoidFitter类实例，包含优化后的椭球体参数
    \"\"\"
    from tumor_analyzer import EllipsoidFitter
    import numpy as np
    
    center = {center.tolist()}
    axes = {display_axes.tolist()}
    radii = {radii.tolist()}
    
    # 对轴向量进行归一化
    normalized_axes = np.zeros_like(axes)
    for i in range(3):
        normalized_axes[i] = axes[i] / np.linalg.norm(axes[i])
    
    ellipsoid_model = EllipsoidFitter.from_precomputed_parameters(center, normalized_axes, radii)
    return ellipsoid_model
"""
    
    py_file = os.path.splitext(output_file)[0] + '.py'
    with open(py_file, 'w') as f:
        f.write(py_code)
    
    print(f"椭球体模型加载函数已保存到 {py_file}")


def main():
    parser = argparse.ArgumentParser(description='优化肿瘤位置的椭球体模型')
    parser.add_argument('--data_dir', type=str, default='datafolds/04_LiTS', 
                        help='包含肿瘤数据的目录')
    parser.add_argument('--output_dir', type=str, default='models', 
                        help='模型和图像输出目录')
    parser.add_argument('--coverage_weight', type=float, default=0.6,
                        help='覆盖率权重 (0-1)')
    parser.add_argument('--compactness_weight', type=float, default=0.4,
                        help='紧凑性权重 (0-1)')
    parser.add_argument('--plot', action='store_true', 
                        help='是否绘制椭球体图')
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 加载肿瘤数据
    analyzer = TumorAnalyzer()
    all_tumors = analyzer.get_all_tumors(args.data_dir, args.data_dir, save=True)
    
    if not all_tumors:
        print("未找到肿瘤数据，请检查数据目录是否正确")
        return
    
    print(f"加载了 {len(all_tumors)} 个肿瘤数据点")
    
    # 提取肿瘤位置
    tumor_positions = np.array([tumor.position for tumor in all_tumors])
    
    # 优化椭球体参数
    center, axes, radii = optimize_ellipsoid(
        tumor_positions, 
        coverage_weight=args.coverage_weight, 
        compactness_weight=args.compactness_weight
    )
    
    # 验证椭球体模型
    validate_ellipsoid(tumor_positions, center, axes, radii)
    
    # 保存模型
    model_file = os.path.join(args.output_dir, 'optimized_ellipsoid_model.pkl')
    save_ellipsoid_model(center, axes, radii, model_file)
    
    # 绘图
    if args.plot:
        plot_file = os.path.join(args.output_dir, 'ellipsoid_visualization.png')
        plot_ellipsoid_with_data(tumor_positions, center, axes, radii, plot_file)


if __name__ == "__main__":
    main() 