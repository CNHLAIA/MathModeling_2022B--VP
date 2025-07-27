import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

def load_ideal_coordinates():

    dir = os.path.dirname(os.path.abspath(__file__))
    original_file = os.path.join(dir, 'coordinate_original.csv')
    
    data = pd.read_csv(original_file)
    ideal_coords = {}
    
    uav_to_index = {
        'FY11': 0, 'FY01': 1, 'FY02': 2, 'FY03': 3, 'FY04': 4, 'FY05': 5,
        'FY06': 6, 'FY07': 7, 'FY08': 8, 'FY09': 9, 'FY10': 10,
        'FY12': 11, 'FY13': 12, 'FY14': 13, 'FY15': 14
    }
    
    for _, row in data.iterrows():
        uav_id = row['无人机编号']
        if uav_id in uav_to_index:
            index = uav_to_index[uav_id]
            ideal_coords[index] = [row['x坐标'], row['y坐标']]
    
    return ideal_coords

def calculate_fitness(particle):

    movable_positions = particle.reshape((14, 2))
    

    positions = np.zeros((15, 2))
    positions[0] = [0, 0] 
    positions[1:] = movable_positions
    

    ideal_coords = load_ideal_coordinates()
    

    position_cost = 0.0
    for i in range(15):
        ideal_pos = np.array(ideal_coords[i])
        actual_pos = positions[i]
        position_cost += np.linalg.norm(actual_pos - ideal_pos) ** 2
    

    key_distance_pairs = [
        (1, 2), (1, 3),  
        (2, 4), (2, 5), (3, 5), (3, 6),  
        (4, 7), (4, 8), (5, 8), (5, 9), (6, 9), (6, 10), 
        (7, 11), (8, 12), (9, 13), (10, 14), 
        (2, 3), (4, 5), (5, 6), (7, 8), (8, 9), (9, 10),
        (11, 12), (12, 13), (13, 14)
    ]
    
    distance_cost = 0.0
    ideal_distance = 50.0
    for i, j in key_distance_pairs:
        actual_distance = np.linalg.norm(positions[i] - positions[j])
        distance_cost += (actual_distance - ideal_distance) ** 2
    

    w1 = 10.0   # 位置收敛权重(主)
    w2 = 0.1    # 距离约束权重(辅)
    
    total_fitness = w1 * position_cost + w2 * distance_cost
    
    return total_fitness

def pso_optimization(initial_positions, n_particles=500, n_iterations=100):
    """
    目标导向的PSO优化
    """
    dimensions = 28
    w = 0.5
    c1 = 0.8
    c2 = 0.9
    

    ideal_coords_dict = load_ideal_coordinates()
    ideal_coords = np.array([
        ideal_coords_dict[1],   # FY01
        ideal_coords_dict[2],   # FY02
        ideal_coords_dict[3],   # FY03
        ideal_coords_dict[4],   # FY04
        ideal_coords_dict[5],   # FY05
        ideal_coords_dict[6],   # FY06
        ideal_coords_dict[7],   # FY07
        ideal_coords_dict[8],   # FY08
        ideal_coords_dict[9],   # FY09
        ideal_coords_dict[10],  # FY10
        ideal_coords_dict[11],  # FY12
        ideal_coords_dict[12],  # FY13
        ideal_coords_dict[13],  # FY14
        ideal_coords_dict[14]   # FY15
    ]).flatten()
    

    search_space_min = ideal_coords - 30
    search_space_max = ideal_coords + 30
    

    particles_pos = np.zeros((n_particles, dimensions))
    
    current_flat = np.array(initial_positions[1:]).flatten()
    for i in range(n_particles // 2):
        particles_pos[i] = current_flat + np.random.normal(0, 10, dimensions)
    
    for i in range(n_particles // 2, n_particles):
        particles_pos[i] = ideal_coords + np.random.normal(0, 5, dimensions)
    
    particles_pos = np.clip(particles_pos, search_space_min, search_space_max)
    particles_vel = np.zeros((n_particles, dimensions))
    
    pbest_pos = particles_pos.copy()
    pbest_fitness = np.array([calculate_fitness(p) for p in particles_pos])
    gbest_pos = pbest_pos[np.argmin(pbest_fitness)]
    gbest_fitness = np.min(pbest_fitness)
    
    print(f"初始最佳适应度: {gbest_fitness:.2f}")
    
    for i in range(n_iterations):
        for j in range(n_particles):
            r1 = np.random.rand(dimensions)
            r2 = np.random.rand(dimensions)
            cognitive_vel = c1 * r1 * (pbest_pos[j] - particles_pos[j])
            social_vel = c2 * r2 * (gbest_pos - particles_pos[j])
            particles_vel[j] = w * particles_vel[j] + cognitive_vel + social_vel
            
            particles_pos[j] += particles_vel[j]
            particles_pos[j] = np.clip(particles_pos[j], search_space_min, search_space_max)
            
            current_fitness = calculate_fitness(particles_pos[j])
            if current_fitness < pbest_fitness[j]:
                pbest_fitness[j] = current_fitness
                pbest_pos[j] = particles_pos[j].copy()
        
        if np.min(pbest_fitness) < gbest_fitness:
            gbest_fitness = np.min(pbest_fitness)
            gbest_pos = pbest_pos[np.argmin(pbest_fitness)]
        
        if i % 50 == 0:
            print(f"迭代 {i}: 适应度 = {gbest_fitness:.2f}")
    
    print(f"最终适应度: {gbest_fitness:.2f}")
    return gbest_pos.reshape((14, 2))

def analyze_formation_accuracy(initial_positions, optimal_positions):
    """
    分析编队与理想坐标的接近程度
    """
    print("\n=== 编队精度分析 ===")
    
    ideal_coords = load_ideal_coordinates()
    
    mapping = [
        (1, 'FY01'), (2, 'FY02'), (3, 'FY03'), (4, 'FY04'), (5, 'FY05'),
        (6, 'FY06'), (7, 'FY07'), (8, 'FY08'), (9, 'FY09'), (10, 'FY10'),
        (11, 'FY12'), (12, 'FY13'), (13, 'FY14'), (14, 'FY15')
    ]
    
    print("初始位置与理想位置偏差:")
    initial_errors = []
    for i, (initial_idx, drone_id) in enumerate(mapping):
        initial_pos = np.array(initial_positions[initial_idx])
        ideal_pos = np.array(ideal_coords[initial_idx])
        error = np.linalg.norm(initial_pos - ideal_pos)
        initial_errors.append(error)
        print(f"  {drone_id}: {error:.2f}m")
    
    print(f"\n初始平均偏差: {np.mean(initial_errors):.2f}m")
    print(f"初始最大偏差: {max(initial_errors):.2f}m")
    
    print("\n优化后位置与理想位置偏差:")
    optimal_errors = []
    all_optimal = [(0, 0)] + [tuple(pos) for pos in optimal_positions]
    
    for i, (initial_idx, drone_id) in enumerate(mapping):
        optimal_pos = np.array(all_optimal[initial_idx])
        ideal_pos = np.array(ideal_coords[initial_idx])
        error = np.linalg.norm(optimal_pos - ideal_pos)
        optimal_errors.append(error)
        print(f"  {drone_id}: {error:.2f}m")
    
    print(f"\n优化后平均偏差: {np.mean(optimal_errors):.2f}m")
    print(f"优化后最大偏差: {max(optimal_errors):.2f}m")
    
    improvement = ((np.mean(initial_errors) - np.mean(optimal_errors)) / np.mean(initial_errors)) * 100
    print(f"\n精度改进: {improvement:.1f}%")
    
    return {
        'initial_avg_error': np.mean(initial_errors),
        'optimal_avg_error': np.mean(optimal_errors),
        'improvement': improvement
    }

def visualize_comparison(initial_positions, optimal_positions):

    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    
    ideal_coords_dict = load_ideal_coordinates()
    ideal_coords = [
        ideal_coords_dict[0],   #同上
        ideal_coords_dict[1],   
        ideal_coords_dict[2],  
        ideal_coords_dict[3],  
        ideal_coords_dict[4],  
        ideal_coords_dict[5],  
        ideal_coords_dict[6],   
        ideal_coords_dict[7],  
        ideal_coords_dict[8],   
        ideal_coords_dict[9],   
        ideal_coords_dict[10], 
        ideal_coords_dict[11],  
        ideal_coords_dict[12],  
        ideal_coords_dict[13],  
        ideal_coords_dict[14]  
    ]
    
    uav_labels = ['FY11', 'FY01', 'FY02', 'FY03', 'FY04', 'FY05', 'FY06', 'FY07', 'FY08', 'FY09', 'FY10', 'FY12', 'FY13', 'FY14', 'FY15']

    ideal_x = [pos[0] for pos in ideal_coords]
    ideal_y = [pos[1] for pos in ideal_coords]
    ax1.scatter(ideal_x, ideal_y, c='blue', s=100, alpha=0.8, label='理想位置')
    ax1.scatter(0, 0, c='red', s=200, marker='s', label='FY11')
    
    for i, (x, y) in enumerate(ideal_coords):
        ax1.annotate(uav_labels[i], (x, y), xytext=(3, 3), textcoords='offset points', fontsize=8)
    
    ax1.set_title('理想锥形编队')
    ax1.set_xlabel('X坐标 (m)')
    ax1.set_ylabel('Y坐标 (m)')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.axis('equal')
    
    initial_x = [pos[0] for pos in initial_positions]
    initial_y = [pos[1] for pos in initial_positions]
    ax2.scatter(initial_x, initial_y, c='red', s=100, alpha=0.7, label='偏差位置')
    ax2.scatter(0, 0, c='blue', s=200, marker='s', label='FY11')
    
    for i, (x, y) in enumerate(initial_positions):
        ax2.annotate(uav_labels[i], (x, y), xytext=(3, 3), textcoords='offset points', fontsize=8)
    
    ax2.set_title('初始偏差位置')
    ax2.set_xlabel('X坐标 (m)')
    ax2.set_ylabel('Y坐标 (m)')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.axis('equal')
    
    all_optimal = [(0, 0)] + [tuple(pos) for pos in optimal_positions]
    opt_x = [pos[0] for pos in all_optimal]
    opt_y = [pos[1] for pos in all_optimal]
    ax3.scatter(opt_x, opt_y, c='green', s=100, alpha=0.7, label='优化位置')
    ax3.scatter(0, 0, c='blue', s=200, marker='s', label='FY11')
    
    for i, (x, y) in enumerate(all_optimal):
        ax3.annotate(uav_labels[i], (x, y), xytext=(3, 3), textcoords='offset points', fontsize=8)
    
    ax3.set_title('优化后位置')
    ax3.set_xlabel('X坐标 (m)')
    ax3.set_ylabel('Y坐标 (m)')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    ax3.axis('equal')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(dir, 'coordinate_mess.csv')
    data = pd.read_csv(file_path)
    
    lie = []
    for index, row in data.iterrows():
        lie.append((row['x坐标'], row['y坐标']))
    
    print("=== 目标导向锥形编队优化 ===")
    print("策略：以理想坐标为目标进行收敛优化")
    
    print("\n开始PSO优化...")
    optimal_positions = pso_optimization(lie)
    
    # 精度分析
    accuracy_results = analyze_formation_accuracy(lie, optimal_positions)
    
    # 移动方案
    movements = []
    mapping = [
        (1, 'FY01'), (2, 'FY02'), (3, 'FY03'), (4, 'FY04'), (5, 'FY05'),
        (6, 'FY06'), (7, 'FY07'), (8, 'FY08'), (9, 'FY09'), (10, 'FY10'),
        (11, 'FY12'), (12, 'FY13'), (13, 'FY14'), (14, 'FY15')
    ]
    
    for i, (initial_idx, drone_id) in enumerate(mapping):
        initial_pos = lie[initial_idx]
        target_pos = optimal_positions[i]
        displacement = np.array(target_pos) - np.array(initial_pos)
        distance = np.linalg.norm(displacement)
        
        if distance > 0.1:
            angle = np.arctan2(displacement[1], displacement[0]) * 180 / np.pi
            movements.append({
                'drone_id': drone_id,
                'distance': distance,
                'direction': angle
            })
    
    print(f"\n调整方案 (共{len(movements)}架无人机):")
    for move in movements:
        print(f"   {move['drone_id']}: 移动{move['distance']:.2f}m, 方向{move['direction']:.1f}°")
    
    # 可视化
    visualize_comparison(lie, optimal_positions)
    
    results_data = []
    for i, (initial_idx, drone_id) in enumerate(mapping):
        results_data.append({
            'UAV_ID': drone_id,
            'Initial_X': lie[initial_idx][0],
            'Initial_Y': lie[initial_idx][1],
            'Optimal_X': optimal_positions[i][0],
            'Optimal_Y': optimal_positions[i][1],
            'Movement_Distance': np.linalg.norm(np.array(optimal_positions[i]) - np.array(lie[initial_idx]))
        })
    
    results_df = pd.DataFrame(results_data)
    output_file = os.path.join(dir, 'q2_target_guided_results.csv')
    results_df.to_csv(output_file, index=False, encoding='utf-8-sig')
