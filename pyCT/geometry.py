import pyCT
from pyCT.parameter import _Parameters
import numpy as np
from skimage.transform import rescale
import matplotlib.pyplot as plt
from matplotlib import cm

def show(params : _Parameters, 
         obj    : np.ndarray = None, 
         scale  : int        = 4,
         view   : list       = None,
         *args):
    
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(projection='3d', computed_zorder=False)
    proj = pyCT.project(obj, params)[0]
    
    params2 = params.copy()
    params2.object.size.set(np.array(params.object.size.get())//scale)
    params2.object.spacing.set(np.array(params.object.spacing.get())*scale)
    params2.set()
    tf = pyCT.getTransformation(params2, 1, 0, 1)
    cube = rescale(obj, 1/scale, preserve_range=True, anti_aliasing=False)

    su, sv = params.detector.length.get()
    nu, nv = params.detector.size.get()
    ou, ov = params.detector.motion.translation.get()[0]
    oa = params.detector.motion.rotation.get()[0]

    if obj is not None:
        # volume
        z, y, x = np.indices(np.array(cube.shape)+1)
        x, y, z, _ = tf.worldTransformation[0] @ np.stack([x.flatten(), y.flatten(), z.flatten(),np.ones_like(z.flatten())])
        x = x.reshape(params2.object.size.z+1, params2.object.size.y+1, params2.object.size.x+1) - params2.object.spacing.x/2
        y = y.reshape(params2.object.size.z+1, params2.object.size.y+1, params2.object.size.x+1) - params2.object.spacing.y/2
        z = z.reshape(params2.object.size.z+1, params2.object.size.y+1, params2.object.size.x+1) - params2.object.spacing.z/2
        filled = cube>0
        facecolors = cm.viridis(cube)
        facecolors[...,-1] = 1-cube
        ax.voxels(x, y, z, filled=filled, facecolors=facecolors, shade=True)

        # projection
        u = np.linspace(.5,su-.5,nu)-su/2
        v = np.linspace(.5,sv-.5,nv)-sv/2
        u,v = np.meshgrid(u, v)
        u,v = u.flatten(), v.flatten()
        X,Y = np.array([[np.cos(oa), np.sin(oa)],[-np.sin(oa),np.cos(oa)]]) @ np.stack([u,v]) + [[ou],[ov]]
        Z = np.ones_like(Y) * -params.source.distance.source2detector
        X,Y,Z,_ = np.linalg.inv(tf.cameraTransformation[0]) @ np.array([X, Y, Z, np.ones_like(X)])
        X = X.reshape(proj.shape)
        Y = Y.reshape(proj.shape)
        Z = Z.reshape(proj.shape)
        ax.plot_surface(X, Y, Z, facecolors=np.repeat(proj[...,None], axis=-1, repeats=3)/proj.max(), alpha=.3, zorder=0)
        
    # volume outline
    x, y, z = params.object.length.get()
    ax.plot([-x/2,-x/2,x/2,x/2,-x/2], [-y/2,y/2,y/2,-y/2,-y/2], [z/2,z/2,z/2,z/2,z/2], '--', color='gray', alpha=.5)
    ax.plot([-x/2,-x/2,x/2,x/2,-x/2], [-y/2,y/2,y/2,-y/2,-y/2], [-z/2,-z/2,-z/2,-z/2,-z/2], '--', color='gray', alpha=.5)
    ax.plot([-x/2,-x/2],[-y/2,-y/2],[-z/2,z/2], '--', color='gray', alpha=.5)
    ax.plot([x/2,x/2],[-y/2,-y/2],[-z/2,z/2], '--', color='gray', alpha=.5)
    ax.plot([-x/2,-x/2],[y/2,y/2],[-z/2,z/2], '--', color='gray', alpha=.5)
    ax.plot([x/2,x/2],[y/2,y/2],[-z/2,z/2], '--', color='gray', alpha=.5)
    
    # axis
    sx, sy, sz = params.object.length.get()/2
    ax.plot([-sx,sx],[0,0],[0,0], 'k')
    ax.text(sx,0,0,'x')
    ax.plot([0,0],[-sy,sy],[0,0], 'k')
    ax.text(0,sy,0,'y')
    ax.plot([0,0],[0,0],[-sz,sz], 'k')
    ax.text(0,0,sz,'z')

    # camera
    rot = np.array([[np.cos(oa), np.sin(oa), 0, 0],[-np.sin(oa), np.cos(oa), 0, 0],[0,0,1,0],[0,0,0,1]])
    cam = np.linalg.inv(tf.cameraTransformation[0]) @ rot
    right = cam[:-1,0]
    up = cam[:-1,1]
    back = cam[:-1,2]
    source = cam[:-1,3]
    detector_center = source - back*params.source.distance.source2detector + [0,ou,ov]
    detector_right = detector_center + right*su/2
    detector_up = detector_center + up*sv/2
    detector_right_up = detector_center + right*su/2 + up*sv/2
    detector_left_up = detector_center - right*su/2 + up*sv/2
    detector_right_down = detector_center + right*su/2 - up*sv/2
    detector_left_down = detector_center - right*su/2 - up*sv/2
    
    temp = np.stack([source, detector_center]).T
    ax.plot(temp[0],temp[1],temp[2],color='orange',marker='o',linestyle='dashed')
    temp = np.stack([detector_right_up, detector_right_down, detector_left_down, detector_left_up, detector_right_up]).T
    ax.plot(temp[0],temp[1],temp[2],color='red')
    temp = np.stack([detector_center, detector_right]).T
    ax.plot(temp[0], temp[1], temp[2], 'r')
    ax.text(detector_right[0], detector_right[1], detector_right[2], 'u')
    temp = np.stack([detector_center, detector_up]).T
    ax.plot(temp[0], temp[1], temp[2], 'r')
    ax.text(detector_up[0], detector_up[1], detector_up[2], 'v')
    
    ax.axis('equal')
    if view is not None:
        ax.view_init(azim=view[0], elev=view[1], roll=view[2])
    
    return fig