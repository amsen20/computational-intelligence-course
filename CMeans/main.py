import csv
import sys
import random
from vector import *
import matplotlib.pyplot as plt

CNT = 100

path = sys.argv[1]

points = []
dims = []
intervals = []

with open(path, 'r') as f:
    csv_reader = csv.DictReader(f)
    frstln = True
    for row in csv_reader:
        if frstln:
            frstln = False
            for dim in row:
                dims.append(dim)
        point = [float(row[dim]) for dim in dims]
        point = tuple(point)
        points.append(point)

for i in range(len(dims)):
    mx = -1
    mn = 1e9
    for point in points:
        mn = min(mn, point[i])
        mx = max(mx, point[i])
    intervals.append((mn, mx))

def random_point():
    point = []
    for (l, r) in intervals:
        point.append(random.random()*(r-l) + l)
    return tuple(point)

def C_means(C, m):
    centroids = [random_point() for _ in range(C)]

    for _ in range(CNT):
        # calculate uij
        u = []
        for i in range(len(points)):
            point  = points[i]
            row = []

            for j in range(len(centroids)):
                centroid = centroids[j]
                cur = 0
                X_C = cord(sub(point, centroid))
                for ocentroid in centroids:
                    X_OC = cord(sub(point, ocentroid))
                    cur += (X_C/X_OC)**(1/(m-1))
                uij = 1/cur
                row.append(uij)
            
            u.append(row)
        
        # move centroid
        for i in range(len(centroids)):
            centroid = centroids[i]
            cur = zeros(centroid)
            div = 0
            for j in range(len(points)):
                point = points[j]
                cur = add(cur, mul(point, u[j][i]**m))
                div += u[j][i]**m
            cur = mul(cur, 1/div)
            
            centroids[i] = cur
        
        # calculate cost function
        cost = 0
        for i in range(len(points)):
            point = points[i]
            for j in range(len(centroids)):
                centroid = centroids[j]
                cost += u[i][j]**m * cord(sub(point, centroid))
        
    return cost, u, centroids

def draw(us, crisp=False):
    assert len(dims) == 2
    x = [point[0] for point in points]
    y = [point[1] for point in points]
    for C, m, u, centroids in us:
        cx = [centroid[0] for centroid in centroids]
        cy = [centroid[1] for centroid in centroids]
        colors = []
        for i, point in enumerate(points):
            col = 0
            all = 0
            mx, id = -1, -1
            for j in range(C):
               col += u[i][j]**m * (j*100/C)
               all += u[i][j]**m
               if u[i][j] > mx:
                   mx, id = u[i][j], j
            col /= all
            colors.append(col if not crisp else id*100/C)
        print("C: {}, m = {}".format(C, m))
        plt.scatter(x, y, c=colors, cmap='viridis')
        if '-center' in sys.argv:
            plt.scatter(cx, cy, color='red', marker='o')
        plt.colorbar()
        plt.show()

def eval(Cs, ms, toplt='C'):
    assert len(Cs) == 1 or len(ms) == 1
    xs = []
    costs = []
    us = []

    for C in Cs:
        for m in ms:
            cost, u, centroids = C_means(C, m)
            xs.append(C if toplt == 'C' else m)
            costs.append(cost)
            us.append((C, m, u, centroids))
            print("C: {}, m: {}, cost: {}".format(C, m, cost))

    plt.plot(xs, costs)
    plt.xlabel(toplt)
    plt.ylabel('cost')
    plt.show()
    try:
        draw(us, ('-crisp' in sys.argv))
    except Exception as e:
        print(e)

if '-plotc' in sys.argv:
    eval([1, 2, 3, 4, 5], [3], 'C')

if '-plotm' in sys.argv:
    eval([3], [2, 3, 4, 5, 6, 7, 8, 9, 10], 'm')
