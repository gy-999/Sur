import nibabel as nib
import numpy as np
import torch
import torchvision.transforms as transforms
import os
import nibabel as nib
import numpy as np
import torch
import torchvision.transforms as transforms
import os


def load_nifti_image(file_path):
    """Load NIfTI image from file and return as numpy array."""
    img = nib.load(file_path)
    img_data = img.get_fdata()
    return img_data


def center_crop(img, target_shape):
    """Center crop a 3D image to the target shape."""
    start_x = (img.shape[0] - target_shape[0]) // 2
    start_y = (img.shape[1] - target_shape[1]) // 2
    start_z = (img.shape[2] - target_shape[2]) // 2
    return img[start_x:start_x + target_shape[0],
           start_y:start_y + target_shape[1],
           start_z:start_z + target_shape[2]]


class NiftiToTensor(object):
    """Convert NIfTI image in numpy array format to torch tensor."""

    def __call__(self, img):
        return torch.from_numpy(img).float()


class MaskImage(object):
    """Apply mask to the image."""

    def __init__(self, mask):
        self.mask = mask

    def __call__(self, img):
        if img.shape != self.mask.shape:
            raise ValueError(f"Image shape {img.shape} and mask shape {self.mask.shape} do not match.")
        return img * self.mask


def save_nifti_image(data, file_path):
    """Save numpy array as NIfTI image."""
    nifti_img = nib.Nifti1Image(data, np.eye(4))
    nib.save(nifti_img, file_path)


# Define the target shape
target_shape = (200, 200, 130)


# Define the transform pipeline
def get_transform(mask):
    return transforms.Compose([
        transforms.Lambda(lambda img: (img - np.min(img)) / (np.max(img) - np.min(img))),  # Normalize
        transforms.Lambda(lambda img: center_crop(img, target_shape)),  # Center crop
        MaskImage(mask),  # Apply mask
        transforms.Lambda(lambda img: np.expand_dims(img, axis=0)),  # Add channel dimension
        NiftiToTensor()  # Convert to tensor
    ])


# Define paths
base_image_dir = r'D:\xunlei\PKG - UCSF-PDGM-v3-20230111\UCSF-PDGM-v3'
base_mask_dir = r'D:\xunlei\PKG - UCSF-PDGM-v3-20230111\segment'
output_dir = r'D:\xunlei\PKG - UCSF-PDGM-v3-20230111\output'
os.makedirs(output_dir, exist_ok=True)

# Traverse through the directories
for root, _, files in os.walk(base_image_dir):
    for file in files:
        if file.endswith('T2.nii.gz'):
            image_path = os.path.join(root, file)
            mask_path = os.path.join(base_mask_dir, os.path.basename(root), 'Necrosis_mask.nii.gz')

            if os.path.exists(mask_path):
                # Load image and mask
                t2_image = load_nifti_image(image_path)
                mask = load_nifti_image(mask_path)

                # Center crop mask to target shape
                mask = center_crop(mask, target_shape)

                # Apply transformations
                transform = get_transform(mask)
                preprocessed_image = transform(t2_image)

                # Save the preprocessed image
                output_file_path = os.path.join(output_dir, f'preprocessed_{file}')
                save_nifti_image(preprocessed_image.numpy()[0], output_file_path)

                print(f'Preprocessed image saved to {output_file_path}')

