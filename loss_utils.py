import torch
def calculate_mask_loss_with_split(gt_masks, fake_masks, in_area_weight, out_area_weight, loss_f, mask_inverse=False, debug=False):
    # in_out_area_split
    in_area_mask_loss = 0.0
    out_area_mask_loss = 0.0
    # calculate in-area out-area each image
    for i in range(len(gt_masks)):
        
        if mask_inverse:
            out_area = (gt_masks[i] == 0)
            in_area = (out_area == False)
        else:
            out_area = (gt_masks[i] == 1)
            in_area = (out_area == False)
        if in_area_weight != 0.0:
            if len(fake_masks[i][in_area]) == 0 or len(gt_masks[i][in_area]) == 0:
                pass
            else:
                in_area_mask_loss_per_image = loss_f(fake_masks[i][in_area],gt_masks[i][in_area]).mean()
              
                if in_area_mask_loss == 0.0:
                    in_area_mask_loss = in_area_mask_loss_per_image
                else:
                    in_area_mask_loss += in_area_mask_loss_per_image 

        if out_area_weight != 0.0:
            out_area_mask_loss_per_image = loss_f(fake_masks[i][out_area], gt_masks[i][out_area]).mean() 
            if out_area_mask_loss == 0.0:
                out_area_mask_loss = out_area_mask_loss_per_image
            else:
                out_area_mask_loss += out_area_mask_loss_per_image 
                
    if in_area_mask_loss != 0.0:
        in_area_mask_loss /= len(gt_masks)

    if out_area_mask_loss != 0.0:
        out_area_mask_loss /= len(gt_masks)
        
    mask_loss = in_area_weight * in_area_mask_loss + out_area_weight * out_area_mask_loss

    return mask_loss, in_area_mask_loss, out_area_mask_loss



def calc_gradient_penalty(D, real_images, fake_images, device ,args):
    
    batch_size = fake_images.shape[0]
    # cuda = bool(device == "cuda")

    eta = torch.FloatTensor(batch_size,1,1,1).uniform_(0,1)
    eta = eta.expand(batch_size, real_images.size(1), real_images.size(2), real_images.size(3))
    eta = eta.to(device)
  

    interpolated = eta * real_images + ((1 - eta) * fake_images)

    interpolated = interpolated.to(device)
   

    # define it to calculate gradient
    interpolated = torch.autograd.Variable(interpolated, requires_grad=True)

    # calculate probability of interpolated examples
    prob_interpolated = D(interpolated)
   
    # calculate gradients of probabilities with respect to examples
    gradients = torch.autograd.grad(outputs=prob_interpolated, inputs=interpolated,
                            grad_outputs=torch.ones(prob_interpolated.size()).to(device),
                            create_graph=True, retain_graph=True)[0]

    grad_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() 
    return grad_penalty
