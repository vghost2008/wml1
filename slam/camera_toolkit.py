import numpy as np

def project_to_2d(X, camera_params):
    """
    Project 3D points to 2D
    Arguments:
    X -- 3D points in *camera space* to transform (N, *, 3)
    camera_params -- intrinsic parameteres (N, 2+2+3+2=9)
    """
    assert X.shape[-1] == 3
    assert len(camera_params.shape) == 2
    assert camera_params.shape[-1] == 9
    assert X.shape[0] == camera_params.shape[0]

    while len(camera_params.shape) < len(X.shape):
        camera_params = np.expand_dims(camera_params, 1)
    f = camera_params[..., :2]
    c = camera_params[..., 2:4]
    k = camera_params[..., 4:7]
    p = camera_params[..., 7:]

    # XX = np.clip(X[..., :2] / X[..., 2:], a_min=-1, a_max=1)
    XX = X[..., :2] / X[..., 2:]
    r2 = np.sum(XX[..., :2] ** 2, axis=len(XX.shape) - 1, keepdims=True)

    radial = 1 + np.sum(k * np.concatenate((r2, r2 ** 2, r2 ** 3), axis=len(r2.shape) - 1), axis=len(r2.shape) - 1,
                        keepdims=True)
    tan = np.sum(p * XX, axis=len(XX.shape) - 1, keepdims=True)

    XXX = XX * (radial + tan) + p * r2

    return f * XXX + c

def trans_cam_intrinsic(points,src_K,dst_K):
    '''

    Args:
        points: [N,2] points from camera with intrinsic src_K
        src_K:
        dst_K:

    Returns:
        points: [N,2]

    '''
    points = pixel2cam(points,src_K)
    return cam2pixel(points,dst_K)

def pixel2cam(P,K):
    '''
    inverst of project_to_2d, as the depth info is lost, use default depth 1.0
    Args:
        P: [N,2]
        K: [3,3]

    Returns:

    '''
    P = P-K[:2,2]
    focal_length = np.array([[K[0,0],K[1,1]]],dtype=np.float32)
    return P/focal_length

def cam2pixel(P,K):
    '''
    almost do the same thing as project_to_2d
    Args:
        P: [N,2]
        K: [3,3]

    Returns:
        P: [N,2]

    '''
    focal_length = np.array([[K[0,0],K[1,1]]],dtype=np.float32)
    P = P*focal_length
    P = P+K[:2,2]
    return P

def get_homogeneous_coordinates(points):
    '''

    Args:
        points: [N,X]

    Returns:
        points: [N,X+1]

    '''
    shape = points.shape[:-1]+[1]
    v = np.ones(shape,dtype=points.dtype)
    return np.concatenate([points,v],axis=-1)


def get_inhomogeneous_coordinates(points):
    '''

    Args:
        points: [N,X]

    Returns:
        points: [N,X-1]
    '''
    shape = list(points.shape)
    shape[-1] = shape[-1]-1
    res = points[...,:-1]/points[...,-1:]
    return res


