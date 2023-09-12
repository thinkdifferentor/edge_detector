import numpy as np


roberts_kernel = np.array([
    [[
        [1,  0],
        [0, -1]
    ]],
    [[
        [0, -1],
        [1,  0]
    ]]
])


prewitt_kernel = np.array([
    [[
        [-1, -1, -1],
        [0,  0,  0],
        [1,  1,  1]
    ]],
    [[
        [-1,  0,  1],
        [-1,  0,  1],
        [-1,  0,  1]
    ]],
    [[
        [0,  1,  1],
        [-1,  0,  1],
        [-1, -1,  0]
    ]],
    [[
        [-1, -1,  0],
        [-1,  0,  1],
        [0,  1,  1]
    ]]
])


sobel_kernel = np.array([
    [[
        [-1, -2, -1],
        [0,  0,  0],
        [1,  2,  1]
    ]],
    [[
        [-1,  0,  1],
        [-2,  0,  2],
        [-1,  0,  1]
    ]],
    [[
        [0,  1,  2],
        [-1,  0,  1],
        [-2, -1,  0]
    ]],
    [[
        [-2, -1,  0],
        [-1,  0,  1],
        [0,  1,  2]
    ]]
])


scharr_kernel = np.array([
    [[
        [-3, -10, -3],
        [0,   0,  0],
        [3,  10,  3]
    ]],
    [[
        [-3,  0,   3],
        [-10, 0,  10],
        [-3,  0,   3]
    ]],
    [[
        [0,  3,  10],
        [-3,  0,  3],
        [-10, -3,  0]
    ]],
    [[
        [-10, -3, 0],
        [-3,  0, 3],
        [0,  3,  10]
    ]]
])


krisch_kernel = np.array([
    [[
        [5, 5, 5],
        [-3, 0, -3],
        [-3, -3, -3]
    ]],
    [[
        [-3, 5, 5],
        [-3, 0, 5],
        [-3, -3, -3]
    ]],
    [[
        [-3, -3, 5],
        [-3, 0, 5],
        [-3, -3, 5]
    ]],
    [[
        [-3, -3, -3],
        [-3, 0, 5],
        [-3, 5, 5]
    ]],
    [[
        [-3, -3, -3],
        [-3, 0, -3],
        [5, 5, 5]
    ]],
    [[
        [-3, -3, -3],
        [5, 0, -3],
        [5, 5, -3]
    ]],
    [[
        [5, -3, -3],
        [5, 0, -3],
        [5, -3, -3]
    ]],
    [[
        [5, 5, -3],
        [5, 0, -3],
        [-3, -3, -3]
    ]]
])


robinson_kernel = np.array([
    [[
        [1, 2, 1],
        [0, 0, 0],
        [-1, -2, -1]
    ]],
    [[
        [0, 1, 2],
        [-1, 0, 1],
        [-2, -1, 0]
    ]],
    [[
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ]],
    [[
        [-2, -1, 0],
        [-1, 0, 1],
        [0, 1, 2]
    ]],
    [[
        [-1, -2, -1],
        [0, 0, 0],
        [1, 2, 1]
    ]],
    [[
        [0, -1, -2],
        [1, 0, -1],
        [2, 1, 0]
    ]],
    [[
        [1, 0, -1],
        [2, 0, -2],
        [1, 0, -1]
    ]],
    [[
        [2, 1, 0],
        [1, 0, -1],
        [0, -1, -2]
    ]]
])


laplacian_kernel = np.array([
    [[
        [1, 1, 1],
        [1, -8, 1],
        [1, 1, 1]
    ]],
    [[
        [0, 1, 0],
        [1, -4, 1],
        [0, 1, 0]
    ]]
])