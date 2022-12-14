# refernece:
# https://github.com/alyssaq/face_morpher/blob/dlib/facemorpher/warper.py
# https://github.com/mjkwon2021/CAT-Net/blob/90739212417fe78b6bc7bc3b3a5fd93902fa67b1/Splicing/data/AbstractDataset.py#L17
from matplotlib import path
import numpy as np
import scipy.spatial as spatial

from torchvision.transforms.functional import to_pil_image
import lpips
import torch
import jpegio
import torch

def to_dct_volume(dct_coef_tensor : torch.Tensor, T = 20):    
    dct_volume_tensor = torch.zeros(size=(T+1, dct_coef_tensor.shape[1], dct_coef_tensor.shape[2]))
    dct_volume_tensor[0] += (dct_coef_tensor == 0).float().squeeze()
    for i in range(1, T):
        dct_volume_tensor[i] += (dct_coef_tensor == i).float().squeeze()
        dct_volume_tensor[i] += (dct_coef_tensor == -i).float().squeeze()
    dct_volume_tensor[T] += (dct_coef_tensor >= T).float().squeeze()
    dct_volume_tensor[T] += (dct_coef_tensor <= -T).float().squeeze()
    return dct_volume_tensor



def get_jpeg_info(im_path):
        """
        :param im_path: JPEG image path
        :return: DCT_coef (Y,Cb,Cr), qtables (Y,Cb,Cr)
        """
        num_channels = 3
        jpeg = jpegio.read(str(im_path))

        # determine which axes to up-sample
        ci = jpeg.comp_info
        need_scale = [[ci[i].v_samp_factor, ci[i].h_samp_factor] for i in range(num_channels)]
        if num_channels == 3:
            if ci[0].v_samp_factor == ci[1].v_samp_factor == ci[2].v_samp_factor:
                need_scale[0][0] = need_scale[1][0] = need_scale[2][0] = 2
            if ci[0].h_samp_factor == ci[1].h_samp_factor == ci[2].h_samp_factor:
                need_scale[0][1] = need_scale[1][1] = need_scale[2][1] = 2
        else:
            need_scale[0][0] = 2
            need_scale[0][1] = 2

        # up-sample DCT coefficients to match image size
        DCT_coef = []
        for i in range(num_channels):
            r, c = jpeg.coef_arrays[i].shape
            coef_view = jpeg.coef_arrays[i].reshape(r//8, 8, c//8, 8).transpose(0, 2, 1, 3)
            # case 1: row scale (O) and col scale (O)
            if need_scale[i][0]==1 and need_scale[i][1]==1:
                out_arr = np.zeros((r * 2, c * 2))
                out_view = out_arr.reshape(r * 2 // 8, 8, c * 2 // 8, 8).transpose(0, 2, 1, 3)
                out_view[::2, ::2, :, :] = coef_view[:, :, :, :]
                out_view[1::2, ::2, :, :] = coef_view[:, :, :, :]
                out_view[::2, 1::2, :, :] = coef_view[:, :, :, :]
                out_view[1::2, 1::2, :, :] = coef_view[:, :, :, :]

            # case 2: row scale (O) and col scale (X)
            elif need_scale[i][0]==1 and need_scale[i][1]==2:
                out_arr = np.zeros((r * 2, c))
                DCT_coef.append(out_arr)
                out_view = out_arr.reshape(r*2//8, 8, c // 8, 8).transpose(0, 2, 1, 3)
                out_view[::2, :, :, :] = coef_view[:, :, :, :]
                out_view[1::2, :, :, :] = coef_view[:, :, :, :]

            # case 3: row scale (X) and col scale (O)
            elif need_scale[i][0]==2 and need_scale[i][1]==1:
                out_arr = np.zeros((r, c * 2))
                out_view = out_arr.reshape(r // 8, 8, c * 2 // 8, 8).transpose(0, 2, 1, 3)
                out_view[:, ::2, :, :] = coef_view[:, :, :, :]
                out_view[:, 1::2, :, :] = coef_view[:, :, :, :]

            # case 4: row scale (X) and col scale (X)
            elif need_scale[i][0]==2 and need_scale[i][1]==2:
                out_arr = np.zeros((r, c))
                out_view = out_arr.reshape(r // 8, 8, c // 8, 8).transpose(0, 2, 1, 3)
                out_view[:, :, :, :] = coef_view[:, :, :, :]

            else:
                raise KeyError("Something wrong here.")

            DCT_coef.append(out_arr)

        # quantization tables
        qtables = [jpeg.quant_tables[ci[i].quant_tbl_no].astype(np.float) for i in range(num_channels)]

        return DCT_coef, qtables

class LPIPS_filter():
    def __init__(self,lpips_threshold, unmask_value = 0): #  unmask_value = 0 | 1
        self.lpips_threshold = lpips_threshold
        self.loss_fn_alex = lpips.LPIPS(net='alex')
        self.unmask_value = unmask_value
    def __call__(self,mask,origin_tensor,warpped_tensor):
        
        lpips_threshold = self.lpips_threshold
        loss_fn_alex = self.loss_fn_alex
        image_size = origin_tensor.shape[-2:]
        
        NUM_SIZE = 16
        PATCH_SIZE = k_size=image_size[0]//NUM_SIZE
        patches = mask.unfold(1, PATCH_SIZE, PATCH_SIZE).unfold(2, PATCH_SIZE, PATCH_SIZE)
        warpped_patches = warpped_tensor.unfold(1, PATCH_SIZE, PATCH_SIZE).unfold(2, PATCH_SIZE, PATCH_SIZE)
        origin_patches = origin_tensor.unfold(1, PATCH_SIZE, PATCH_SIZE).unfold(2, PATCH_SIZE, PATCH_SIZE)

        have_mask_list = []
        for i in range(NUM_SIZE):
            for j in range(NUM_SIZE):
                sub_img = patches[:, i, j]
                if(torch.count_nonzero(sub_img)!=0):
                    have_mask_list.append((i,j))
        threshold_mask_patches = patches.clone()
        
        unmask_patch = torch.zeros(patches.shape[-2:]) if self.unmask_value == 0 else torch.ones(patches.shape[-2:])

        for idx,(i,j) in enumerate(have_mask_list):
            w_patch = warpped_patches[:,i,j]
            o_patch = origin_patches[:,i,j]
            loss = loss_fn_alex(w_patch,o_patch).mean().item()
            if loss < lpips_threshold:
                threshold_mask_patches[:,i,j] = unmask_patch.clone()
                
        return threshold_mask_patches.permute(0,1,3,2,4).flatten(start_dim=3,end_dim=4).flatten(start_dim=1,end_dim=2)
        
        

def create_mesh(image_size,mesh_size: int):
    assert len(image_size) == 2, print("image_size len must be 2", len(image_size))
    # Create Mesh
    x = np.linspace(0,  image_size[0], image_size[0]//mesh_size)
    y = np.linspace(0,  image_size[1], image_size[1]//mesh_size)
    # x = np.linspace(self.mesh_size,  image_size[0]-self.mesh_size, image_size[0]//self.mesh_size)
    # y = np.linspace(self.mesh_size,  image_size[1]-self.mesh_size, image_size[1]//self.mesh_size)
    xv, yv = np.meshgrid( y,x)
    mesh = np.concatenate( (xv[np.newaxis,:], yv[np.newaxis,:]) )
    mesh = np.int32(mesh)
    # mesh_pts = mesh.reshape(2,-1).T
    
    return mesh


def bilinear_interpolate(img, coords):
    """ Interpolates over every image channel
    http://en.wikipedia.org/wiki/Bilinear_interpolation
    :param img: max 3 channel image
    :param coords: 2 x _m_ array. 1st row = xcoords, 2nd row = ycoords
    :returns: array of interpolated pixels with same shape as coords
    """
    int_coords = np.int32(coords)
    x0, y0 = int_coords
    dx, dy = coords - int_coords

    # 4 Neighour pixels
    q11 = img[y0, x0]
    q21 = img[y0, x0+1]
    q12 = img[y0+1, x0]
    q22 = img[y0+1, x0+1]

    btm = q21.T * dx + q11.T * (1 - dx)
    top = q22.T * dx + q12.T * (1 - dx)
    inter_pixel = top * dy + btm * (1 - dy)

    return inter_pixel.T

def grid_coordinates(points):
    """ x,y grid coordinates within the ROI of supplied points
    :param points: points to generate grid coordinates
    :returns: array of (x, y) coordinates
    """
    xmin = np.min(points[:, 0])
    xmax = np.max(points[:, 0]) + 1
    ymin = np.min(points[:, 1])
    ymax = np.max(points[:, 1]) + 1
    return np.asarray([(x, y) for y in range(ymin, ymax)
                        for x in range(xmin, xmax)], np.uint32)

def process_warp(src_img, result_img, tri_affines, dst_points, delaunay):
    """
    Warp each triangle from the src_image only within the
    ROI of the destination image (points in dst_points).
    """
    roi_coords = grid_coordinates(dst_points)
    # indices to vertices. -1 if pixel is not in any triangle
    roi_tri_indices = delaunay.find_simplex(roi_coords)

    for simplex_index in range(len(delaunay.simplices)):
        coords = roi_coords[roi_tri_indices == simplex_index]
        num_coords = len(coords)
        out_coords = np.dot(tri_affines[simplex_index],
                            np.vstack((coords.T, np.ones(num_coords))))
        x, y = coords.T
        result_img[y, x] = bilinear_interpolate(src_img, out_coords)

    return None

def triangular_affine_matrices(vertices, src_points, dest_points):
    """
    Calculate the affine transformation matrix for each
    triangle (x,y) vertex from dest_points to src_points
    :param vertices: array of triplet indices to corners of triangle
    :param src_points: array of [x, y] points to landmarks for source image
    :param dest_points: array of [x, y] points to landmarks for destination image
    :returns: 2 x 3 affine matrix transformation for a triangle
    """
    ones = [1, 1, 1]
    for tri_indices in vertices:
        src_tri = np.vstack((src_points[tri_indices, :].T, ones))
        dst_tri = np.vstack((dest_points[tri_indices, :].T, ones))
        mat = np.dot(src_tri, np.linalg.inv(dst_tri))[:2, :]
        yield mat

def warp_image(src_img, src_points, dest_points, dest_shape, dtype=np.uint8):
    # Resultant image will not have an alpha channel
    num_chans = 3
    src_img = src_img[:, :, :3]

    rows, cols = dest_shape[:2]
    # result_img = np.zeros((rows, cols, num_chans), dtype)
    result_img = src_img.copy()

    delaunay = spatial.Delaunay(dest_points)
    tri_affines = np.asarray(list(triangular_affine_matrices(
    delaunay.simplices, src_points, dest_points)))

    process_warp(src_img, result_img, tri_affines, dest_points, delaunay)

    return result_img

def rgb2gray(rgb):
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray

def get_var_map(img,mesh):
    img = rgb2gray(img)[...,np.newaxis]
    var_map = np.zeros((img.shape[0],img.shape[1],1))
    for x in range(mesh.shape[1]-1):
        for y in range(mesh.shape[2]-1):
            start_x,start_y = mesh[:,x,y]
            end_x,end_y = mesh[:,x+1,y+1]
            
            grid = img[start_y:end_y,start_x:end_x,:]
            grid_var = np.var(grid)
            var_map[start_y:end_y,start_x:end_x,:] =  grid_var
    # normalize
    var_map = (var_map - var_map.min()) / (var_map.mean() - var_map.min())
    var_map = np.clip(var_map, 0, 1)
    return var_map

def get_tri_varmap(img,mesh_pts):
    tri = spatial.Delaunay(mesh_pts)

    """ ?????? traingle_var_map """
    tri_varmap = np.zeros((img.shape[0],img.shape[1],1))

    x, y = np.meshgrid(np.arange(tri_varmap.shape[1]), np.arange(tri_varmap.shape[0]))
    x, y = x.flatten(), y.flatten()

    points = np.vstack((x,y)).T

    for ith in range(len(tri.simplices)):
        p = path.Path(tri.points[tri.simplices][ith])
        sample_pixel = p.contains_points(points).reshape(tri_varmap.shape[0],tri_varmap.shape[1])
        # print(" sample_pixel", sample_pixel)
        tri_area = img[sample_pixel]
        # print( tri_area.shape)
        tri_var = np.var(tri_area,keepdims=False)
        tri_varmap[sample_pixel] =  tri_var
    return tri_varmap

def mix_mask_var(mask,varmap,threshold):
    # inverse mask to only select mask area
    inv_mask = np.abs(mask - 1)
    mask_varmap = inv_mask.squeeze() * varmap.squeeze()
    mask_varmap = mask_varmap.reshape(mask.shape)
    

    # normalize
    # mask_varmap = mask_varmap/mask_varmap.max()
    
    # inverse mask_varmap (is the truthly result varmap)
    # make mask_area --> 0, unmask --> 1
    # [0,1] -1 --> [-1,0] --> abs([-1,0]) --> [1,0] 
    mask_varmap = np.abs(mask_varmap -1 )
    
    if threshold == -1:
        return mask_varmap
    
    # Threshold
    # make the final mask like trimap only 0, 0.5, 1 --> mask, undefined, unmask
    # mask area, set to 0
    mask_varmap = np.where(mask_varmap<threshold,np.zeros_like(mask_varmap),mask_varmap)
    # unmask area
    # set to -1 first, to avoid become undefined area, will use abs latter to recover 
    mask_varmap = np.where(mask_varmap==1,-1 * np.ones_like(mask_varmap),mask_varmap)
    # undefined area, set to 0.5
    mask_varmap = np.where(mask_varmap>=threshold,0.5 * np.ones_like(mask_varmap),mask_varmap)
    # recover unmask area from -1 to 1
    mask_varmap = np.abs(mask_varmap)

    
    # print(mask_varmap.min(),mask_varmap.max())
    return mask_varmap

def create_grid_mask(src_img,sample_pts_mesh_xy,mesh_tran):
    # Create mask
    mask = np.ones((src_img.shape[0],src_img.shape[1],1))
    # print("mask",mask.shape)
    for x,y in sample_pts_mesh_xy:
        start_x,start_y = mesh_tran[:,x-1,y-1]
        end_x,end_y = mesh_tran[:,x+1,y+1]
        # print("fdewef",mesh_tran[:,x-1,y-1], mesh_tran[:,x,y], mesh_tran[:,x+1,y+1])
        mask[start_y:end_y,start_x:end_x] = 0
    return mask

def create_triangle_mask(src_img,sample_pts,mesh_tran_pts):
    tri = spatial.Delaunay(mesh_tran_pts)
    """ ?????? sample_pts ????????? mesh_tran_pts ??? index """
    indexes = []
    for idx,(x,y) in enumerate(mesh_tran_pts):
        for sx,sy in sample_pts:
            if sx == x and sy == y:
                indexes.append(idx)
    # print(indexes)

    """ ?????? sample_pts ?????????????????????????????????????????????????????? """
    sample_traingle_idxs = [ [] for i in range(len(indexes))]
    for idx in range(len(tri.simplices)):
        i1,i2,i3 = tri.simplices[idx]
        for target_idx in range(len(indexes)):
            target = indexes[target_idx]
            if i1 == target or target == i2 or target == i3:
                sample_traingle_idxs[target_idx].append(idx)

    """ ????????? sample_pts ?????????????????? ???????????? """            
    sample_triangles=[]
    for idx in range(len(sample_traingle_idxs)):
        sample_triangles.append( tri.points[tri.simplices[sample_traingle_idxs[idx]]])
        # print(tri.points[tri.simplices[sample_traingle_idxs[idx]]].shape)
    sample_triangles= np.vstack(sample_triangles)

    """ ?????? traingle_mask """
    mask_sample = np.ones((src_img.shape[0],src_img.shape[1],1))

    x, y = np.meshgrid(np.arange(mask_sample.shape[1]), np.arange(mask_sample.shape[0]))
    x, y = x.flatten(), y.flatten()

    points = np.vstack((x,y)).T

    for ith in range(len(sample_triangles)):
        p = path.Path(sample_triangles[ith])
        grid = p.contains_points(points).reshape(mask_sample.shape[0],mask_sample.shape[1],1)
        mask_sample[grid] = 0
    
    return mask_sample