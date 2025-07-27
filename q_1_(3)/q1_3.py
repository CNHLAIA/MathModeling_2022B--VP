import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

def calculate_fitness(particle):

    positions = particle.reshape((9, 2))
    
    # 半径偏差
    radii = np.sqrt(np.sum(positions**2, axis=1))
    avg_radius = np.mean(radii)
    radius_cost = np.sum((radii - avg_radius)**2)
    
    # 角度间隔
    angles = np.arctan2(positions[:, 1], positions[:, 0]) * 180 / np.pi
    
    sorted_angles = np.sort(angles)
    
    # 相邻角度差
    diffs = np.diff(sorted_angles)
    last_diff = (sorted_angles[0] + 360) - sorted_angles[-1]
    all_diffs = np.append(diffs, last_diff)
    
    ideal_angle = 40.0
    angle_cost = np.sum((all_diffs - ideal_angle)**2)
    
    w1 = 1.0  # 半径成本的权重
    w2 = 1.0  # 角度成本的权重
    
    total_fitness = w1 * radius_cost + w2 * angle_cost
    
    return total_fitness

def pso_optimization(initial_positions, n_particles=500, n_iterations=1000):

    dimensions = 18       # 维度
    w = 0.5               # 惯性
    c1 = 0.8              # 认知
    c2 = 0.9              # 社会
    
    # 搜索范围
    initial_flat = np.array(initial_positions[1:]).flatten()  # 排除FY00
    search_space_min = initial_flat - 50
    search_space_max = initial_flat + 50
    
    # 粒子的初始位置和速度
    particles_pos = np.random.uniform(search_space_min, search_space_max, (n_particles, dimensions))
    particles_vel = np.zeros((n_particles, dimensions))
    
    pbest_pos = particles_pos.copy()
    pbest_fitness = np.array([calculate_fitness(p) for p in particles_pos])
    gbest_pos = pbest_pos[np.argmin(pbest_fitness)]
    gbest_fitness = np.min(pbest_fitness)
    
    print(f"初始最佳适应度: {gbest_fitness:.6f}")
    

    for i in range(n_iterations):
        for j in range(n_particles):
            
            r1 = np.random.rand(dimensions)
            r2 = np.random.rand(dimensions)
            cognitive_vel = c1 * r1 * (pbest_pos[j] - particles_pos[j])
            social_vel = c2 * r2 * (gbest_pos - particles_pos[j])
            particles_vel[j] = w * particles_vel[j] + cognitive_vel + social_vel
            
            
            particles_pos[j] += particles_vel[j]
            

            particles_pos[j] = np.clip(particles_pos[j], search_space_min, search_space_max)
            
            # 个体最优
            current_fitness = calculate_fitness(particles_pos[j])
            if current_fitness < pbest_fitness[j]:
                pbest_fitness[j] = current_fitness
                pbest_pos[j] = particles_pos[j].copy()
        
        # 全局最优
        if np.min(pbest_fitness) < gbest_fitness:
            gbest_fitness = np.min(pbest_fitness)
            gbest_pos = pbest_pos[np.argmin(pbest_fitness)]
        
        if i % 200 == 0:
            print(f"迭代次数 {i}: 最佳适应度 = {gbest_fitness:.6f}")
    
    print(f"最终最佳适应度: {gbest_fitness:.6f}")
    

    optimal_coords = gbest_pos.reshape((9, 2))
    return optimal_coords

def calculate_adjustment_plan(initial_positions, optimal_positions):

    adjustment_steps = []
    
    for i in range(1, 10):  
        initial_pos = initial_positions[i]
        target_pos = optimal_positions[i-1]  
        

        displacement = np.array(target_pos) - np.array(initial_pos)
        distance = np.linalg.norm(displacement)
        
        if distance > 0.1:  
            angle = np.arctan2(displacement[1], displacement[0]) * 180 / np.pi
            
            step = {
                'drone_id': f'FY{i:02d}',
                'from': initial_pos,
                'to': target_pos,
                'distance': distance,
                'direction': angle
            }
            adjustment_steps.append(step)
    
    return adjustment_steps

def visualize_positions(initial_positions, optimal_positions):
    """
    可视化初始位置和优化后的位置
    """

    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    initial_x = [pos[0] for pos in initial_positions]
    initial_y = [pos[1] for pos in initial_positions]
    plt.scatter(initial_x, initial_y, c='red', s=100, alpha=0.7, label='Initial Positions')
    plt.scatter(0, 0, c='blue', s=200, marker='s', label='FY00 (Center)')

    for i, (x, y) in enumerate(initial_positions):
        plt.annotate(f'FY{i:02d}', (x, y), xytext=(5, 5), textcoords='offset points')
    
    plt.title('Initial UAV Positions')
    plt.xlabel('X Coordinate (m)')
    plt.ylabel('Y Coordinate (m)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.axis('equal')

    plt.subplot(1, 2, 2)
    opt_x = [pos[0] for pos in optimal_positions]
    opt_y = [pos[1] for pos in optimal_positions]
    plt.scatter(opt_x, opt_y, c='green', s=100, alpha=0.7, label='Optimized Positions')
    plt.scatter(0, 0, c='blue', s=200, marker='s', label='FY00 (Center)')
    

    for i, (x, y) in enumerate(optimal_positions):
        plt.annotate(f'FY{i:02d}', (x, y), xytext=(5, 5), textcoords='offset points')
    
    plt.title('Optimized UAV Positions')
    plt.xlabel('X Coordinate (m)')
    plt.ylabel('Y Coordinate (m)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.axis('equal')
    
    plt.tight_layout()
    plt.show()

def analyze_formation_quality(positions):


    radii = [np.sqrt(x**2 + y**2) for x, y in positions[1:]] 
    avg_radius = np.mean(radii)
    radius_std = np.std(radii)

    angles = [np.arctan2(y, x) * 180 / np.pi for x, y in positions[1:]]
    sorted_angles = sorted(angles)
    
    angle_diffs = []
    for i in range(len(sorted_angles)):
        if i == len(sorted_angles) - 1:
            diff = (sorted_angles[0] + 360) - sorted_angles[i]
        else:
            diff = sorted_angles[i+1] - sorted_angles[i]
        angle_diffs.append(diff)
    
    ideal_angle = 40.0
    angle_std = np.std(angle_diffs)
    
    print(f"\n编队质量分析:")
    print(f"平均半径: {avg_radius:.2f} m")
    print(f"半径标准差: {radius_std:.2f} m")
    print(f"理想角度间隔: {ideal_angle:.1f}°")
    print(f"实际角度间隔: {[f'{diff:.1f}°' for diff in angle_diffs]}")
    print(f"角度间隔标准差: {angle_std:.2f}°")
    
    return {
        'avg_radius': avg_radius,
        'radius_std': radius_std,
        'angle_diffs': angle_diffs,
        'angle_std': angle_std
    }

def to_polar_zb(zj_positions):
    
    polar_positions = []
    for x, y in zj_positions:
        r = np.sqrt(x**2 + y**2)
        theta = np.arctan2(y, x)
        polar_positions.append((r, np.degrees(theta)))
    return polar_positions


if __name__ == "__main__":
    
    dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(dir, 'q1_3_jizb.csv')
    data = pd.read_csv(file_path)
    lie = []
    for index, row in data.iterrows():
        id = int(row['无人机编号'])
        jzb = row['极坐标m']
        angle = np.radians(row['极坐标°'])
        lie.append((jzb * np.cos(angle), jzb * np.sin(angle)))
    
    print("初始无人机位置 (直角坐标):")
    for i, pos in enumerate(lie):
        print(f"FY{i:02d}: ({pos[0]:.2f}, {pos[1]:.2f})")
    
    # PSO
    optimal_positions = pso_optimization(lie)
    
    print("\n优化后的FY01-FY09位置坐标 (x, y):")
    for i, pos in enumerate(optimal_positions):
        print(f"FY{i+1:02d}: ({pos[0]:.2f}, {pos[1]:.2f})")

    polar_op_positions = to_polar_zb(optimal_positions)

    print("\n优化后的FY01-FY09的极坐标:")
    for i, (r, theta) in enumerate(polar_op_positions):
        print(f"FY{i+1:02d}: 半径={r:.6f}m, 角度={theta:.6f}°")
    
    
    # 可视化
    all_optimal = [(0, 0)] + [tuple(pos) for pos in optimal_positions] 
    visualize_positions(lie, all_optimal)
    
    # 分析编队质量
    print("\n初始编队质量:")
    initial_quality = analyze_formation_quality(lie)
    
    print("\n优化后编队质量:")
    optimal_quality = analyze_formation_quality(all_optimal)
    
    results_df = pd.DataFrame({
        'UAV_ID': [f'FY{i+1:02d}' for i in range(9)],
        'Initial_X': [pos[0] for pos in lie[1:]],
        'Initial_Y': [pos[1] for pos in lie[1:]],
        'Optimal_X': [pos[0] for pos in optimal_positions],
        'Optimal_Y': [pos[1] for pos in optimal_positions],
        'Movement_Distance': [np.linalg.norm(np.array(optimal_positions[i]) - np.array(lie[i+1])) 
                             for i in range(9)]
    })
    
    output_file = os.path.join(dir, 'q1_3_optimization_results.csv')
    results_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    
    