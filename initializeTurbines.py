from shapely.geometry import Polygon, Point
from scipy.spatial import distance
import numpy as np

def initializeTurbines(boundaries, n_wt, min_distance):
    polygon = Polygon(boundaries)
    
    min_x, min_y, max_x, max_y = polygon.bounds
    
    
    points = []
    while len(points) < n_wt:
        x = np.random.uniform(min_x, max_x)
        y = np.random.uniform(min_y, max_y)
        point = Point(x, y)

        if len(points) > 0: 
            
            if polygon.contains(point):
                
                if all(distance.euclidean((x,y), existing_point) >= min_distance for existing_point in points):
                    points.append([x, y])  
        else:
            points.append([x,y])

    return np.array(points)
