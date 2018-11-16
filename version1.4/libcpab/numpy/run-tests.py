import nplibcpab
import numpy as np

if __name__ == "__main__":
    N = 100
    batch = 1
    num_cells = 4
    nStepSolver = 50
    points = np.linspace(0, 1, N).reshape((1, -1))
    #Trels = np.ones((batch, num_cells, 1, 2))
    #Trels[:, :, :, 1] = 0
    Trels = np.random.random((batch, num_cells, 1, 2))
    
    ## The C implementation
    new_points = np.zeros((batch, 1, N))
    nplibcpab.npcpab_forward(points, Trels, new_points, np.array([4.0]), nStepSolver)
    
    ## The python implementation
    new_points2 = np.zeros((batch, 1, N))
    for t in range(batch):
        for i in range(N):
            point = points[0, i]
            
            for n in range(nStepSolver):
                cellidx = min(0, max(np.floor(point * num_cells), num_cells))
                A = Trels[t, cellidx]
                point = A[0, 0] * point + A[0, 1]
            new_points2[t, 0, i] = point
    
    ## Compare the two
    print(np.max(np.abs(new_points - new_points2)))

