
# Content-aware image resizing

Seam Carving, also known as Content Aware Image-Resizing, Image Retargeting, is a technique to resize the images. The technique basically resizes images based on the content of the image i.e. it preserves the content in order of its importance.In Seam-carving the image is reduced in size by one pixel of height (or width) at a time. A vertical seam in an image is a path of pixels connected from the top to the bottom with one pixel in each row. An horizontal seam in an image is a path of pixels connected from the left to the right with one pixel in each column.    
    
The goal of this project is to implement seam carving of images and extends to object removal. We implemented this project based on the method described in the paper by Avidan and Shamir.

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
> Usage of numba @jit for faster processing      
Open [usage.tx](usage.tx) and follow the procedure

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
> backward energy is calculated by a simple gradient magnitude at energy map
2. Calculate cumulative energy map and seam paths for image
3. Find the seam of lowest energy
4. Remove seam of lowest energy
5. step 1 - 4 until achieving targeting width



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
<img src="images/ratatouille.jpg" height="300"> <img src="out_images/imge_resize_wxnyn.jpg" height="300">

- image removal object     
<img src="images/Picture1.png" height="300"> <img src="out_images/obj_remove.jpg" height="300">

## References

**[Wikipedia seam carving](https://en.wikipedia.org/wiki/Seam_carving)**    
**[GeeksforGeeks](https://www.geeksforgeeks.org/image-resizing-using-seam-carving-using-opencv-in-python/?ref=gcse)**   
**[Avidan and Shamir paper](http://graphics.cs.cmu.edu/courses/15-463/2007_fall/hw/proj2/imret.pdf)**

