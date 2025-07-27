import numpy as np;
import matplotlib.pyplot as plt
import sympy as sp

class node:
    #已知无人机的直角坐标
    x_0 = 0
    y_0 = 0
    def __init__(self, i, j, a1, a2, a3, q):
        self.x_1 = np.cos(2 * (i-1) * np.pi / 9)
        self.y_1 = np.sin(2 * (i-1) * np.pi / 9)
        self.x_2 = np.cos(2 * (j-1) * np.pi / 9)
        self.y_2 = np.sin(2 * (j-1) * np.pi / 9)
        self.delx = sp.symbols('delta_x')
        self.dely = sp.symbols('delta_y')
        self.x_q = sp.cos(2 * (q-1) * sp.pi / 9) + self.delx
        self.y_q = sp.sin(2 * (q-1) * sp.pi / 9) + self.dely
        self.a1 = a1
        self.a2 = a2
        self.a3 = a3


def to_vector(startp, endp):
    start = np.array(startp)
    end = np.array(endp)
    return end - start

def solve(n):
    #三个向量 0i 0j ij
    v1 = to_vector([n.x_q, n.y_q], [n.x_0, n.y_0])
    v2 = to_vector([n.x_q, n.y_q], [n.x_1, n.y_1])
    v3 = to_vector([n.x_q, n.y_q], [n.x_2, n.y_2])

    m_v1 = sp.sqrt(v1[0]**2 + v1[1]**2)
    m_v2 = sp.sqrt(v2[0]**2 + v2[1]**2)
    m_v3 = sp.sqrt(v3[0]**2 + v3[1]**2)

    cos_a1 = sp.cos(sp.rad(float(n.a1)))
    cos_a2 = sp.cos(sp.rad(float(n.a2)))
    cos_a3 = sp.cos(sp.rad(float(n.a3)))

    eq1 = sp.Eq((v1[0]*v2[0] + v1[1]*v2[1]) / (m_v1 * m_v2), cos_a1)
    eq2 = sp.Eq((v1[0]*v3[0] + v1[1]*v3[1]) / (m_v1 * m_v3), cos_a2)
    eq3 = sp.Eq((v2[0]*v3[0] + v2[1]*v3[1]) / (m_v2 * m_v3), cos_a3)

    solution = sp.nsolve((eq1, eq2, eq3), (n.delx, n.dely), (0, 0), verify = False)
    print("偏移量求解结果[Δx, Δy]是:" ,solution)

    x_q_real = np.cos(2 * (q - 1) * np.pi / 9) + float(solution[0])
    y_q_real = np.sin(2 * (q - 1) * np.pi / 9) + float(solution[1])
    print(f"实际坐标: ({x_q_real:.6f}, {y_q_real:.6f})")

    with open("q1_1_result.csv", "a") as f:
        f.write(f"{q},{x_q_real:.6f},{y_q_real:.6f}\n")
    return
    

if __name__ == "__main__":
    with open("q1_1_result.csv", "w") as f:
        f.write("接收信号无人机编号,x坐标,y坐标\n")

    n1, n2 = map(int, input("请输入2个发射信号的无人机编号，用空格分隔:").split())
    #a1为q 0和q n1对应的夹角，a2为q 0和q n2对应的夹角，a3为q n1和q n2对应的夹角
    for i in range(1,8):
        q = int(input("请输入当前接收信号的无人机编号:"))
        a1, a2, a3 = map(float, input("请输入已知角度，用空格分隔:").split())
        n = node(n1, n2, a1, a2, a3, q)
        solve(n)


'''
测试数据:
input:
1 2
3
49.5 69.5 20
4
29 49 20
7
28.5 8.5 20
9
67.5 47.5 20
'''