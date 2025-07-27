#本代码部分借鉴了教材'司守奎，孙玺菁 数学建模算法与应用 Python'

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Arc
from matplotlib.patches import Circle

fig, ax = plt.subplots()

def draw_angle(ax, p_center, p1, p2, r=0.4, color='red', label=None, fontsize=12):

    v1 = np.array([p1[0] - p_center[0], p1[1] - p_center[1]])
    v2 = np.array([p2[0] - p_center[0], p2[1] - p_center[1]])

    def angle(v):
        return np.degrees(np.arctan2(v[1], v[0]))

    angle1 = angle(v1)
    angle2 = angle(v2)


    if angle2 < angle1:
        angle1, angle2 = angle2, angle1


    arc = Arc(p_center, width=r*2, height=r*2, theta1=angle1, theta2=angle2, edgecolor=color, lw=2)
    ax.add_patch(arc)


    if label:
        mid_angle = np.radians((angle1 + angle2) / 2)
        label_x = p_center[0] + (r + 0.1) * np.cos(mid_angle)
        label_y = p_center[1] + (r + 0.1) * np.sin(mid_angle)

        ax.text(label_x, label_y, label, fontsize=fontsize, color=color,
                horizontalalignment='center', verticalalignment='center')


def draw_single_circle(C = (0,1), R = 1, p1 = None, p2 = None, p3 = None):

    p_half = ((p1[0] + p2[0])/2, (p1[1] + p2[1])/2)

    #circle = plt.Circle(C, R, color = 'blue', fill = False)
    #ax.add_patch(circle)
    ax.set_aspect('equal')
    
    ax.scatter(p1[0], p1[1], color = 'black')
    ax.scatter(p2[0], p2[1], color = 'black')
    ax.scatter(p3[0], p3[1], color = 'black')
    ax.scatter(C[0], C[1], color = 'blue')
    ax.scatter(p_half[0], p_half[1], color = 'black')

    ax.text(p1[0], p1[1] - 0.13, p1[2], fontsize = 12, color = 'green', verticalalignment = 'bottom')
    ax.text(p2[0]-0.25, p2[1] - 0.13, p2[2], fontsize = 12, color = 'green', verticalalignment = 'bottom')
    ax.text(p3[0]-0.25, p3[1], 'FY_q', fontsize = 12, color = 'green', verticalalignment = 'bottom')

    x_line = [p1[0], p2[0], p3[0], p1[0]]
    y_line = [p1[1], p2[1], p3[1], p1[1]]
    ax.plot(x_line, y_line, color = 'black')
    ax.plot([p2[0], C[0], p_half[0]], [p2[1], C[1], p_half[1]], color = 'black', linestyle = '--')

    draw_angle(ax, p3, p1, p2, r = 0.1, color = 'black', label = r'$\alpha_1$', fontsize = 8)
    draw_angle(ax, C, p2, p_half, r = 0.1, color = 'black', label = r'$\alpha_1$', fontsize = 8)

    v1 = np.array([p1[0] - C[0], p1[1] - C[1]])
    v2 = np.array([p2[0] - C[0], p2[1] - C[1]])

    angle = lambda v: np.degrees(np.arctan2(v[1], v[0]))
    angle1 = angle(v1)
    angle2 = angle(v2)

    if angle2 < angle1:
        angle1, angle2 = angle2, angle1

    if angle2 - angle1 > 180:
        angle1 += 360

    arc_dash = Arc(C, width=2*R, height=2*R, theta1=angle1, theta2=angle2, edgecolor='blue', linestyle='--', lw=2)
    ax.add_patch(arc_dash)

    arc_solid1 = Arc(C, width=2*R, height=2*R, theta1=angle2, theta2=angle1 + 360, edgecolor='blue', linestyle='-', lw=2)
    ax.add_patch(arc_solid1)

def distance(a, b):
    return ((a[0] - b[0])**2 + (a[1] - b[1])**2)**0.5

def draw_3c(ax, P, C1, C2, C3, color = ('red', 'blue', 'green')):

    R1 = distance(C1, P)
    R2 = distance(C2, P)
    R3 = distance(C3, P)

    circle1 = Circle(C1, R1, color = color[0], fill = False)
    circle2 = Circle(C2, R2, color = color[1], fill = False)
    circle3 = Circle(C3, R3, color = color[2], fill = False)

    ax.add_patch(circle1)
    ax.add_patch(circle2)
    ax.add_patch(circle3)

    ax.scatter(P[0], P[1], color = 'black', zorder = 5)
    ax.text(P[0] + 0.03, P[1] + 0.2, 'P', fontsize = 8, color = 'green')

    jiao1, jiao2, jiao3 = (0.07, 5.62), (0.36, 0.53), (4.09, -0.24)
    ax.scatter([jiao1[0], jiao2[0], jiao3[0]], [jiao1[1], jiao2[1], jiao3[1]])
    ax.text(jiao1[0] + 0.2, jiao1[1], 'A', fontsize = 8, va = 'center')
    ax.text(jiao2[0] + 0.2, jiao2[1], 'B', fontsize = 8, va = 'center')
    ax.text(jiao3[0] + 0.2, jiao3[1] - 0.2, 'C', fontsize = 8, va = 'center')

    arrowprops = dict(arrowstyle="->", color='black', lw=1.5)
    
    ax.annotate("", xy=jiao1, xytext=P, arrowprops=dict(arrowstyle="->", color='black', lw=0.9))
    ax.annotate("", xy=jiao2, xytext=P, arrowprops=dict(arrowstyle="->", color='black', lw=0.9))
    ax.annotate("", xy=jiao3, xytext=P, arrowprops=dict(arrowstyle="->", color='black', lw=0.9))

    '''
    ax.scatter([C1[0], C2[0], C3[0]], [C1[1], C2[1], C3[1]], c=color, s=50)
    ax.text(C1[0] + 0.1, C1[1], 'C1', fontsize=8, va='center')
    ax.text(C2[0] + 0.1, C2[1], 'C2', fontsize=8, va='center')
    ax.text(C3[0] + 0.1, C3[1], 'C3', fontsize=8, va='center')
    '''

if __name__ == "__main__":
    draw_single_circle(C = (0,1), R = 1, p1 = (0.8660254037844386, 0.5, 'FY01'), p2 = (-0.8660254037844386, 0.5, 'FY00'), p3 = (-0.6, 1.8))
    #draw_3c(ax, (1,1), (-1,3), (4,4), (2,-1))
    plt.title("questions 1_2_one-circle")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(False)
    plt.axis('equal')
    plt.show()
    #print((1-0.8**2)**0.5)