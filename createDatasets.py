#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


# # Classes

# In[2]:


import torch
import torchvision
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from natsort import natsorted
import cv2


# In[22]:


#. https://github.com/alyssaq/face_morpher/blob/dlib/facemorpher/warper.py
import numpy as np
import scipy.spatial as spatial

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

def get_var_map(img,mesh):
    var_map = np.zeros((img.shape[0],img.shape[1],1))
    for x in range(mesh.shape[1]-1):
        for y in range(mesh.shape[2]-1):
            start_x,start_y = mesh[:,x,y]
            end_x,end_y = mesh[:,x+1,y+1]
            
            gird = img[start_y:end_y,start_x:end_x,:]
            grid_var = np.var(gird,keepdims=False)
            var_map[start_y:end_y,start_x:end_x,:] =  grid_var
    return var_map

def mix_mask_var(mask,varmap,threshold):
    # inverse mask to only select mask area
    inv_mask = np.abs(mask - 1)
    mask_varmap = inv_mask * varmap

    # normalize
    mask_varmap = mask_varmap/mask_varmap.max()
    
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
    from matplotlib import path
    tri = spatial.Delaunay(mesh_tran_pts)
    """ 找出 sample_pts 對應在 mesh_tran_pts 的 index """
    indexes = []
    for idx,(x,y) in enumerate(mesh_tran_pts):
        for sx,sy in sample_pts:
            if sx == x and sy == y:
                indexes.append(idx)
    # print(indexes)

    """ 找出 sample_pts 所在的三角形，理論上大概要有五個左右 """
    sample_traingle_idxs = [ [] for i in range(len(indexes))]
    for idx in range(len(tri.simplices)):
        i1,i2,i3 = tri.simplices[idx]
        for target_idx in range(len(indexes)):
            target = indexes[target_idx]
            if i1 == target or target == i2 or target == i3:
                sample_traingle_idxs[target_idx].append(idx)

    """ 將所有 sample_pts 所在的三角形 搜集起來 """            
    sample_triangles=[]
    for idx in range(len(sample_traingle_idxs)):
        sample_triangles.append( tri.points[tri.simplices[sample_traingle_idxs[idx]]])
        # print(tri.points[tri.simplices[sample_traingle_idxs[idx]]].shape)
    sample_triangles= np.vstack(sample_triangles)


    """ 製作 traingle_mask """
    mask_sample = np.ones((src_img.shape[0],src_img.shape[1],1))

    x, y = np.meshgrid(np.arange(mask_sample.shape[1]), np.arange(mask_sample.shape[0]))
    x, y = x.flatten(), y.flatten()

    points = np.vstack((x,y)).T

    for ith in range(len(sample_triangles)):
        p = path.Path(sample_triangles[ith])
        grid = p.contains_points(points).reshape(mask_sample.shape[0],mask_sample.shape[1],1)
        mask_sample[grid] = 0
    
    return mask_sample

def createGTmask(mask_type,
                  varmap_type,
                  src_img,
                  warpped_img,
                  sample_pts,
                  sample_pts_mesh_xy,
                  mesh,
                  mesh_pts,
                  mesh_tran,
                  mesh_tran_pts,
                  image_size,
                  var_map_threshold):
    # Create mask
    mask = None
    if mask_type == "grid":
        mask = create_grid_mask(src_img,sample_pts_mesh_xy,mesh_tran)
    elif mask_type == "tri":
        mask = create_triangle_mask(src_img,sample_pts,mesh_tran_pts)
    else:
        raise NotImplementedError(f"mask_type {mask_type} not implemented!")
        
    # Create VarMap
    var_map_use = None
    if varmap_type == "notuse":
        return mask

    elif varmap_type == "var(warp)":
        var_warpped_map = get_var_map(np.array(warpped_img),mesh)
        var_map_use = var_warpped_map
        
    elif varmap_type == "warp(var)":
        var_map = get_var_map(src_img,mesh)
        warpped_var_map = warp_image(var_map ,mesh_pts, mesh_tran_pts, image_size )
        var_map_use = warpped_var_map
        
    else:
        raise NotImplementedError(f"varmap_type {varmap_type} not implemented!")
    
    mask_var_map = mix_mask_var(mask,var_map_use,threshold = var_map_threshold)
    return mask_var_map
    


class GridTriangularWarp:
    
    def __init__(self, args,mesh_size = 18 ,num_vertex_wanted_to_move=10, warp_min_factor=1,warp_max_factor=9,return_mesh=True, debug = False ):
        self.mesh_size = mesh_size
        self.num_vertex_wanted_to_move = num_vertex_wanted_to_move
        self.warp_min_factor = warp_min_factor
        self.warp_max_factor = warp_max_factor
        self.return_mesh = return_mesh
        
        self.mask_type = args.mask_type
        self.varmap_type = args.varmap_type
        self.varmap_threshold= args.varmap_threshold
        
        self.debug = debug
        
        
        
    def __call__(self, pillow_img):
        src_img = np.array(pillow_img)
        image_size = (src_img.shape[0],src_img.shape[1])

        # Create Mesh
        x = np.linspace(2,  image_size[0]-2, image_size[0]//self.mesh_size)
        y = np.linspace(2,  image_size[1]-2, image_size[1]//self.mesh_size)
        # x = np.linspace(self.mesh_size,  image_size[0]-self.mesh_size, image_size[0]//self.mesh_size)
        # y = np.linspace(self.mesh_size,  image_size[1]-self.mesh_size, image_size[1]//self.mesh_size)
        xv, yv = np.meshgrid( y,x)
        mesh = np.concatenate( (xv[np.newaxis,:], yv[np.newaxis,:]) )
        mesh = np.int32(mesh)
        mesh_pts = mesh.reshape(2,-1).T
    
        
        # Create Deform Mesh
        # print("mesh",mesh.shape)
        
        sample_pts_mesh_xy = np.concatenate(
            (
                np.random.randint( 3,mesh.shape[1]-3, size=(self.num_vertex_wanted_to_move, 1)),
                np.random.randint( 3,mesh.shape[2]-3, size=(self.num_vertex_wanted_to_move, 1))
                # np.random.randint( 1,mesh.shape[1]-2, size=(self.num_vertex_wanted_to_move, 1)),
                # np.random.randint( 1,mesh.shape[2]-2, size=(self.num_vertex_wanted_to_move, 1))
            ),axis=1
        )
        

        # print("sample_pts_mesh_xy",sample_pts_mesh_xy.shape) 
        mesh_tran = mesh.copy()
        sample_pts = []
        for x, y in sample_pts_mesh_xy:     
            effect = 0.1 * np.random.randint(self.warp_min_factor,self.warp_max_factor,size = 2)
            sign = 2 * np.random.randint(2,size = 2) -1 
            shift_vector = np.ones(2)
            shift_vector[0] *=  self.mesh_size * effect[0] * sign[0]
            shift_vector[1] *=  self.mesh_size * effect[1] * sign[1]
            # print(image_size[1]//self.mesh_size)
            # print(shift_vector,np.int32(shift_vector))
            mesh_tran[:,x,y] += np.int32(shift_vector)
            sample_pts.append(mesh_tran[:,x,y])
        sample_pts = np.stack(sample_pts)
        sample_pts = np.array(sample_pts, dtype=np.int32)
        
        mesh_tran = np.array(mesh_tran, dtype= np.int32)
        mesh_tran_pts =mesh_tran.reshape(2,-1).T
      
        
        mesh_pts = np.array(mesh_pts,dtype=np.int32)
        mesh_tran_pts = np.array(mesh_tran_pts,dtype= np.int32)
        result_img = warp_image(src_img ,mesh_pts, mesh_tran_pts, image_size )
        warpped_img = Image.fromarray(np.clip(result_img, 0.0, 255.0).astype(np.uint8))
              
        if self.return_mesh:
            # Create mask
            var_map_threshold=self.varmap_threshold
            mask = createGTmask(self.mask_type,
                  self.varmap_type,
                  src_img,
                  warpped_img,
                  sample_pts,
                  sample_pts_mesh_xy,
                  mesh,
                  mesh_pts,
                  mesh_tran,
                  mesh_tran_pts,
                  image_size,
                  var_map_threshold=var_map_threshold)
            
            if self.debug:
                masked_img = Image.fromarray(np.uint8(np.array(warpped_img) * mask))
                plt.figure(figsize=(16,8))
                plt.imshow(masked_img)
                plt.plot(mesh_tran_pts[:,0], mesh_tran_pts[:,1], 'o')
                plt.plot(sample_pts[:,0],sample_pts[:,1],'o',color="red")
                plt.triplot(mesh_tran_pts[:,0], mesh_tran_pts[:,1], spatial.Delaunay(mesh_tran_pts).simplices.copy())
                plt.show()
                return warpped_img,mesh_pts,mesh_tran_pts, mask
            else:
                return warpped_img,mesh_pts,mesh_tran_pts, mask
        
        return warpped_img
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


# In[23]:


def check_create_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


# In[67]:


class CelebADataset(torch.utils.data.Dataset):
    def __init__(self, warp_f,
                 root_dir, 
                 image_size=(256,256),
                 transform=None, 
                 return_mesh=False,
                 savePath=False):
        """
        Args:
        root_dir (string): Directory with all the images
        transform (callable, optional): transform to be applied to each image sample
        """
        # Read names of images in the root directory
        image_names = os.listdir(root_dir)

        self.root_dir = root_dir
        # self.resize = transforms.Resize(size= image_size)
        self.image_size = image_size
        self.centerCrop = transforms.CenterCrop(size= image_size)
        self.basic_transform = transforms.Compose(
        [
             transforms.ToTensor(),
             transforms.Normalize((0., 0., 0.), (1., 1., 1.))
        ])
        
        self.transform = transform 
        self.image_names = natsorted(image_names)
        
        self.warp_f = warp_f
        
        self.return_mesh = return_mesh
        
        self.savePath = savePath
        self.origin_dir = f"{savePath}/origin/"
        check_create_dir(self.origin_dir)
        self.warpped_dir = f"{savePath}/warpped/"
        check_create_dir(self.warpped_dir)
        self.mask_dir = f"{savePath}/mask/"
        check_create_dir(self.mask_dir)
        self.mesh_dir = f"{savePath}/mesh/"
        check_create_dir(self.mesh_dir)
        
        self._test_for_open_img()
        
        
        
    def __len__(self): 
        return len(self.image_names)
    
    def _test_for_open_img(self):
        img_path = os.path.join(self.root_dir, self.image_names[np.random.randint(len(self.image_names))])
        img_pillow_origin = Image.open(img_path).convert('RGB')
        img_pillow_array = np.array(img_pillow_origin)
        print("[test_for_open_img]",img_pillow_array.shape)
        
    
    def __getitem__(self, idx):
        # Get the path to the image 
        img_path = os.path.join(self.root_dir, self.image_names[idx])
        
        # Load image and convert it to RGB
        img_pillow_origin = Image.open(img_path).convert('RGB')        
        if img_pillow_origin.size[0] < self.image_size[0]  and img_pillow_origin.size[1] < self.image_size[1]:
            origin_size = max(img_pillow_origin.size[0],img_pillow_origin.size[1])
            scale_factor = min(self.image_size[0],self.image_size[1]) / origin_size
            scale_size = [ int(img_pillow_origin.size[1] * scale_factor) ,int(img_pillow_origin.size[0] * scale_factor) ]
            # print("img_pillow_origin.size",img_pillow_origin.size)
            # print("scale_factor",scale_factor, "scale_size ",scale_size )
            img_pillow_origin = torchvision.transforms.Resize(size = scale_size)(img_pillow_origin)
        # plt.imshow(img_pillow_origin)
        # print("img_pillow_origin",img_pillow_origin.size)
            
        img_pillow = img_pillow_origin
        if self.transform:
            img_pillow = self.transfrom(img_pillow)
        
        img_pillow = self.centerCrop(img_pillow)
        padding_mask_pillow = self.centerCrop(Image.fromarray(np.ones_like(img_pillow_origin)*255))
        padding_mask = (np.array(padding_mask_pillow)/255)[...,0][...,np.newaxis]
        
        
            
        if self.return_mesh:
            warpped_img, mesh_no_last_row, mesh_trans_no_last_row,mask = self.warp_f(img_pillow)
           
            
            
            if self.savePath:
                # img_pillow = self.centerCrop(img_pillow)
                origin_path = f"{self.origin_dir}/{self.image_names[idx]}"
                img_pillow.save( origin_path)
                
                # warpped_img = self.centerCrop(warpped_img)
                warpped_path = f"{self.warpped_dir}/{self.image_names[idx]}"
                warpped_img.save(warpped_path )
                
                # mask = (np.array( \
                #         self.centerCrop( \
                #             Image.fromarray(np.uint8(mask * 255)[:,:,0]) \
                #         ) \
                #     )/255)[...,np.newaxis]
                mask = mask * padding_mask 
                mask_path = f"{self.mask_dir}/{self.image_names[idx].split('.')[0]}"
                np.save(mask_path,mask)
                
                # mesh_pts,mesh_tran_pts
                mesh_path = f"{self.mesh_dir}/{self.image_names[idx].split('.')[0]}.npz"
                np.savez(mesh_path, mesh=mesh_no_last_row, mesh_tran=mesh_trans_no_last_row)
                
                
                print("origin:",origin_path)
                print("warpped:", warpped_path)
                print("mask_path", mask_path)
                print("mesh_path", mesh_path)
                print("---")
            
            img = self.basic_transform(img_pillow) 
            warpped_img = self.basic_transform(warpped_img)
            
            
               
                
            return img
        else:
            warpped_img = self.warp_f(img_pillow)
            warpped_img = self.basic_transform(warpped_img) 
            return img


# # Run

# In[68]:


src_data_dir = "/workspace/inpaint_mask/data/CIHP/instance-level_human_parsing/Training/Images/"
target_data_dir = "/workspace/inpaint_mask/data/warpData/CIHP/Training/"


# In[69]:


image_size = (512,512)
args = type('', (), {})()
args.mask_type = "tri"
args.varmap_type = "notuse"
args.varmap_threshold = -1


# In[70]:


batch_size = 16
warp_f = GridTriangularWarp(args=args, 
                            # mesh_size = 16,
                            mesh_size = 24,
                            num_vertex_wanted_to_move=10, 
                            warp_min_factor=1,
                            warp_max_factor=9,
                            return_mesh=True, 
                            debug = False )
dataset = CelebADataset(warp_f,
                        root_dir=src_data_dir,
                        image_size=image_size,
                        return_mesh=True,
                        savePath=target_data_dir )
data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, 
                                          shuffle=False,
                                          drop_last=False,
                                          num_workers=16)


# In[71]:

cnt = 0
for data in data_loader:
    cnt += data.shape[0]
    # break
print("total length:",cnt)

# In[72]:


sample_id = "0000006"

origin = Image.open(f"{target_data_dir}/origin/{sample_id}.jpg")
warpped = Image.open(f"{target_data_dir}/warpped/{sample_id}.jpg")
mask = np.load(f"{target_data_dir}/mask/{sample_id}.npy")
mesh_tran_pts = np.load(f"{target_data_dir}/mesh/{sample_id}.npz")["mesh_tran"]


# In[73]:


# (np.array(
#     Image.fromarray(np.uint8(mask * 255)[:,:,0]))
#     /255)[...,np.newaxis].shape


# In[74]:


fig, axs = plt.subplots(2, 3, figsize=(16,8))
axs[0,0].imshow(origin)
axs[0,1].imshow(warpped)
axs[0,2].imshow(mask ,cmap="gray")

axs[1,0].imshow( Image.fromarray(np.uint8(np.array(origin) * mask)))
axs[1,1].imshow( Image.fromarray(np.uint8(np.array(warpped) * mask)))
# axs[1,2].imshow(np.ones_like(origin)*255)
axs[1,2].imshow(mask ,cmap="gray")
axs[1,2].triplot(mesh_tran_pts[:,0], mesh_tran_pts[:,1], spatial.Delaunay(mesh_tran_pts).simplices.copy())
fig.savefig(f"{target_data_dir}/sample.jpg")


# In[75]:


# center_crop_f = transforms.CenterCrop(size= (512,512))
# img_pillow_origin = Image.open('/workspace/inpaint_mask/data/CIHP/instance-level_human_parsing/Training/Images/0000006.jpg')
# img_pillow = center_crop_f(img_pillow_origin)
# padding_mask_pillow= center_crop_f(Image.fromarray(np.ones_like(img_pillow_origin)*255))
# padding_mask = (np.array(padding_mask_pillow)/255)[...,0][...,np.newaxis]
# Image.fromarray(np.uint8(np.ones_like(img_pillow)*255 * padding_mask))
# # img_pillow


# In[76]:


# origin.size


# In[ ]:




