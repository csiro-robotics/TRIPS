import torch
import torchvision

   
def data_augmentation_cutout(image, target_label, domain_label):
    images_rotate = images
    mask_size = 16
    mask_size_half = 8
    size, c, h, w = images.shape
    cxmin, cxmax = mask_size_half, w - mask_size_half
    cymin, cymax = mask_size_half, h - mask_size_half
    for i in range(size):
        cx = np.random.randint(cxmin, cxmax)
        cy = np.random.randint(cymin, cymax)
        xmin = cx - mask_size_half
        ymin = cy - mask_size_half
        xmax = xmin + mask_size
        ymax = ymin + mask_size
        xmin = max(0, xmin)
        ymin = max(0, ymin)
        xmax = min(w, xmax)
        ymax = min(h, ymax)
        images_rotate[i, :, ymin:ymax, xmin:xmax] = torch.zeros(3, mask_size, mask_size)
    
    final_image = torch.cat(images, images_rotate)
    final_target_label = torch.cat(target_label, target_label)
    if domain_label != None:
        final_domain_label = torch.cat(domain_label, domain_label)
    else:
        final_domain_label = None

    return final_image, final_target_label, final_domain_label


def data_augmentation_rotation(image, target_label, domain_label):
    image = torch.stack([torch.rot90(image, k, (2, 3)) for k in range(4)], 1)
    image = image.view(-1, 3, 224, 224)
    target_label = torch.stack([target_label * 4 + k for k in range(4)], 1).view(-1)
    if domain_label !=None:
        domain_label = torch.stack([domain_label for k in range(4)], 1).view(-1)
    else:
        domain_label = None

    return image, target_label, domain_label
