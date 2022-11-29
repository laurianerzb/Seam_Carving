
# Content-aware image resizing

eam Carving, also known as Content Aware Image-Resizing, Image Retargeting, is a technique to "smart" resize the images. The technique basically resizes images based on the content of the image i.e. it preserves the content in order of its importance. The goal of this project is to implement seam carving and use it to retarget images.
    
The goal of this project is to implement seam carving of images and extends to object removal. Seam carving is a method of resizing an image. We implemented this project based on the method described in the paper by Avidan and Shamir.

## Setup    

You need Python 3 to run this code and any IDE like Vs code, Atom,etc   

## Requirements

- OpenCV
- scipy
- numba
- numpy

## Usage

The program is run in terminal   
>Syntax: python seam_carving.py (-resize | -remove) -im image_path -out output_file_name [-mask MASK] [-remove_mask_path remove_mask] [-dy DY] [-dx DX] [-visualize_seam]

Open usage.tx and follow the procedure

## Process:

- IMAGE RESIZING   
   > For image resizing input mask is optonal
    - dx < 0 we perform seam removal
    - dx > 0 we perform seam insertion
    - dy < 0 we rotate image,we rotate mask if any, we perform seam removal,we reset the image to original position
    - dy > 0 we rotate image,we rotate mask if any, we perform seam insertion,we reset the image to original position

- OBJECT REMOVAL   
    > For object removal input mask is required

## Algorithm Overview

### Seam Removal

1. Calculate energy map:
> Energy is calculated by sum the absolute value of the gradient in both x direction and y direction for all three channel (R, G, B).
2. Build accumulated cost matrix using forward energy:
> This step is implemented with dynamic programming. The value of each pixel is equal to its corresponding value in the energy map added to the minimum new neighbor energy introduced by removing one of its three top neighbors (top-left, top-center, and top-right)
3. Find and remove minimum seam from top to bottom edge:
> Backtracking from the bottom to the top edge of the accumulated cost matrix to find the minimum seam. All the pixels in each row after the pixel to be removed are shifted over one column to the left if it has index greater than the minimum seam.
4. Repeat step 1 - 3 until achieving targeting width
Seam Insertion

### Seam Insertion

Seam insertion can be thought of as inversion of seam removal and insert new artificial pixels into the image. We first perform seam removal for n seams on a duplicated input image and record all the coordinates in the same order when removing. Then, we insert new seams to original input image in the same order at the recorded coordinates location. The inserted artificial pixel values are derived from an average of left and right neighbors.


### Object Removal

1. Remove object by seam removal

> When generating energy map, the region protected by mask are weighted with a very high negative value. This guarantees that the minimum seam will be routed through the masked region. Seam removal is performed repeatly until masked region has been completely removed as stated above with one more step; the same minimum seam also has to be removed from the mask in order to get correct energy map for the next step.
2. Seam insertion
> Insering seams to return the image back to it's original dimensions.

## Example Results

The input image is on top and the result of the algorithm is on the bottom.
- image resizing with dx and dy < 0 i.e we remove seam in both x and y axis
<img src="images/ratatouille.jpg" height="342"> <img src="out_images/imge_resize_wxnyn.jpg" height="342">

- image removal object     
<img src="images/tour_eiffel.jpg" height="300"> <img src="out_images/obj_remove.jpg" height="300">

## References