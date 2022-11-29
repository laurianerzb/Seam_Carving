import numpy as np
import cv2
import argparse
from numba import jit
from scipy import ndimage as ndi


seam_color_val = np.array([0, 255, 0])    # seam visualization color (BGR)
SHOULD_DOWNSIZE = True                    # if True, downsize image for faster carving
DOWNSIZE_WIDTH = 500                      # resized image width if SHOULD_DOWNSIZE is True
ENERGY_MASK_CONST = 100000.0              # large energy value for protective masking
MASK_THRESHOLD = 10                       # minimum pixel intensity for binary mask
FORWARUSE_FORWARD_ENERGYD_ENERGY = True                 # if True, use forward energy algorithm

#visualize_process function for visualization

def visualize_process(im, boolean_mask=None, rotate=False):
    visualize_seam = im.astype(np.uint8)
    if boolean_mask is not None:
        visualize_seam[np.where(boolean_mask == False)] = seam_color_val
    if rotate:
        visualize_seam = rotate_image(visualize_seam, False)
    cv2.imshow("visualization", visualize_seam)
    cv2.waitKey(1)
    return visualize_seam

#resize image function for visualization

def resize(image, width):
    dim = None
    ht, wt = image.shape[:2]
    dim = (width, int(ht * width / float(wt)))
    return cv2.resize(image, dim)

def rotate_image(image, clock):
    k = 1 if clock else 3
    return np.rot90(image, k)    

#BACKWARD ENERGY FUNCTIONS WITH Simple gradient magnitude energy map.

def backward_energy(im):
    
    xgrad = ndi.convolve1d(im, np.array([1, 0, -1]), axis=1, mode='wrap')
    ygrad = ndi.convolve1d(im, np.array([1, 0, -1]), axis=0, mode='wrap')
    
    gradient_magnitude = np.sqrt(np.sum(xgrad**2, axis=2) + np.sum(ygrad**2, axis=2))

    return gradient_magnitude

@jit

#FORWARD ENERGY FUNCTIONS 

def forward_energy(im):
   
    ht, wt = im.shape[:2]
    im = cv2.cvtColor(im.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float64)

    energy = np.zeros((ht, wt))
    m = np.zeros((ht, wt))
    
    U = np.roll(im, 1, axis=0)
    L = np.roll(im, 1, axis=1)
    R = np.roll(im, -1, axis=1)
    
    cU = np.abs(R - L)
    cL = np.abs(U - L) + cU
    cR = np.abs(U - R) + cU
    
    for i in range(1, ht):
        mU = m[i-1]
        mL = np.roll(mU, 1)
        mR = np.roll(mU, -1)
        
        mULR = np.array([mU, mL, mR])
        cULR = np.array([cU[i], cL[i], cR[i]])
        mULR += cULR

        argmins = np.argmin(mULR, axis=0)
        m[i] = np.choose(argmins, mULR)
        energy[i] = np.choose(argmins, cULR)    
        
    return energy

@jit

# ADD SEAM FUNCTION to a 3-channel color image by averaging the pixels values to the left and right of the seam.

def add_seam(im, seam_idx):
  
    ht, wt = im.shape[:2]
    output = np.zeros((ht, wt + 1, 3))
    for row in range(ht):
        col = seam_idx[row]
        for ch in range(3):
            if col == 0:
                p = np.average(im[row, col: col + 2, ch])
                output[row, col, ch] = im[row, col, ch]
                output[row, col + 1, ch] = p
                output[row, col + 1:, ch] = im[row, col:, ch]
            else:
                p = np.average(im[row, col - 1: col + 1, ch])
                output[row, : col, ch] = im[row, : col, ch]
                output[row, col, ch] = p
                output[row, col + 1:, ch] = im[row, col:, ch]

    return output

@jit

# ADD SEAM TO THE CREATED GRAYSCALE IMAGE FUNCTION

def add_seam_grayscale(im, seam_idx):
      
    ht, wt = im.shape[:2]
    output = np.zeros((ht, wt + 1))
    for row in range(ht):
        col = seam_idx[row]
        if col == 0:
            p = np.average(im[row, col: col + 2])
            output[row, col] = im[row, col]
            output[row, col + 1] = p
            output[row, col + 1:] = im[row, col:]
        else:
            p = np.average(im[row, col - 1: col + 1])
            output[row, : col] = im[row, : col]
            output[row, col] = p
            output[row, col + 1:] = im[row, col:]

    return output

@jit

# REMOVE SEAM FUNCTION

def remove_seam(im, boolean_mask):
    ht, wt = im.shape[:2]
    boolmask3c = np.stack([boolean_mask] * 3, axis=2)
    return im[boolmask3c].reshape((ht, wt - 1, 3))

@jit

# REMOVE GRAYSCALE SEAM FUNCTION

def remove_seam_grayscale(im, boolean_mask):
    ht, wt = im.shape[:2]
    return im[boolean_mask].reshape((ht, wt - 1))

@jit

# GET MINIMUM SEAM FUNCTION USING DP algorithm for finding the seam of minimum energy.

def get_minimum_seam(im, mask=None, remove_mask=None):
  
    ht, wt = im.shape[:2]
    energyfn = forward_energy if USE_FORWARD_ENERGY else backward_energy
    M = energyfn(im)

    if mask is not None:
        M[np.where(mask > MASK_THRESHOLD)] = ENERGY_MASK_CONST

    # give removal mask priority over protective mask by using larger negative value
    if remove_mask is not None:
        M[np.where(remove_mask > MASK_THRESHOLD)] = -ENERGY_MASK_CONST * 100

    backtrack = np.zeros_like(M, dtype=np.int)

    # populate DP matrix
    for i in range(1, ht):
        for j in range(0, wt):
            if j == 0:
                idx = np.argmin(M[i - 1, j:j + 2])
                backtrack[i, j] = idx + j
                min_energy = M[i-1, idx + j]
            else:
                idx = np.argmin(M[i - 1, j - 1:j + 2])
                backtrack[i, j] = idx + j - 1
                min_energy = M[i - 1, idx + j - 1]

            M[i, j] += min_energy

    # backtrack to find path
    seam_idx = []
    boolean_mask = np.ones((ht, wt), dtype=np.bool)
    j = np.argmin(M[-1])
    for i in range(ht-1, -1, -1):
        boolean_mask[i, j] = False
        seam_idx.append(j)
        j = backtrack[i, j]

    seam_idx.reverse()
    return np.array(seam_idx), boolean_mask


# MAIN SEAM REMOVAL FUNCTION

def seams_removal_main(im, num_remove, mask=None, visualize_seam=False, rot=False):
    for _ in range(num_remove):
        seam_idx, boolean_mask = get_minimum_seam(im, mask)
        if visualize_seam:
            visualize_process(im, boolean_mask, rotate=rot)
        im = remove_seam(im, boolean_mask)
        if mask is not None:
            mask = remove_seam_grayscale(mask, boolean_mask)
    return im, mask

# SEAM INSERTION FUNCTION

def insert_seams(im, num_add, mask=None, visualize_seam=False, rot=False):
    seams_arr = []
    temporay_image = im.copy()
    temporay_mask = mask.copy() if mask is not None else None

    for _ in range(num_add):
        seam_idx, boolean_mask = get_minimum_seam(temporay_image, temporay_mask)
        if visualize_seam:
            visualize_process(temporay_image, boolean_mask, rotate=rot)

        seams_arr.append(seam_idx)
        temporay_image = remove_seam(temporay_image, boolean_mask)
        if temporay_mask is not None:
            temporay_mask = remove_seam_grayscale(temporay_mask, boolean_mask)

    seams_arr.reverse()

    for _ in range(num_add):
        seam = seams_arr.pop()
        im = add_seam(im, seam)
        if visualize_seam:
            visualize_process(im, rotate=rot)
        if mask is not None:
            mask = add_seam_grayscale(mask, seam)

        for remaining_seam in seams_arr:
            remaining_seam[np.where(remaining_seam >= seam)] += 2         

    return im, mask

# SEAM CARVING FUNCTION

def seam_carving(im, dy, dx, mask=None, visualize_seam=False):
    im = im.astype(np.float64)
    ht, wt = im.shape[:2]
    assert ht + dy > 0 and wt + dx > 0 and dy <= ht and dx <= wt
    
    if mask is not None:
        mask = mask.astype(np.float64)

    output = im

    if dx < 0:
        output, mask = seams_removal_main(output, -dx, mask, visualize_seam)

    elif dx > 0:
        output, mask = insert_seams(output, dx, mask, visualize_seam)

    if dy < 0:
        output = rotate_image(output, True)
        if mask is not None:
            mask = rotate_image(mask, True)
        output, mask = seams_removal_main(output, -dy, mask, visualize_seam, rot=True)
        output = rotate_image(output, False)

    elif dy > 0:
        output = rotate_image(output, True)
        if mask is not None:
            mask = rotate_image(mask, True)
        output, mask = insert_seams(output, dy, mask, visualize_seam, rot=True)
        output = rotate_image(output, False)

    return output

# OBJECT REMOVAL FUNCTION

def object_removal(im, remove_mask, mask=None, visualize_seam=False, remove_horizontal_seams=False):
    im = im.astype(np.float64)
    remove_mask = remove_mask.astype(np.float64)
    if mask is not None:
        mask = mask.astype(np.float64)
    output = im

    ht, wt = im.shape[:2]

    if remove_horizontal_seams:
        output = rotate_image(output, True)
        remove_mask = rotate_image(remove_mask, True)
        if mask is not None:
            mask = rotate_image(mask, True)

    while len(np.where(remove_mask > MASK_THRESHOLD)[0]) > 0:
        seam_idx, boolean_mask = get_minimum_seam(output, mask, remove_mask)
        if visualize_seam:
            visualize_process(output, boolean_mask, rotate=remove_horizontal_seams)            
        output = remove_seam(output, boolean_mask)
        remove_mask = remove_seam_grayscale(remove_mask, boolean_mask)
        if mask is not None:
            mask = remove_seam_grayscale(mask, boolean_mask)

    num_add = (ht if remove_horizontal_seams else wt) - output.shape[1]
    output, mask = insert_seams(output, num_add, mask, visualize_seam, rot=remove_horizontal_seams)
    if remove_horizontal_seams:
        output = rotate_image(output, False)

    return output        

# MAIN FUNCTION

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("-resize", action='store_true')
    group.add_argument("-remove", action='store_true')

    ap.add_argument("-im", help="Path to image", required=True)
    ap.add_argument("-out", help="Output file name", required=True)
    ap.add_argument("-mask", help="Path to (protective) mask")
    ap.add_argument("-remove_mask", help="Path to removal mask")
    ap.add_argument("-dy", help="Number of vertical seams to add/subtract", type=int, default=0)
    ap.add_argument("-dx", help="Number of horizontal seams to add/subtract", type=int, default=0)
    ap.add_argument("-visualize_seam", help="visualize_process the seam removal process", action='store_true')
    ap.add_argument("-hremove", help="Remove horizontal seams for object removal", action='store_true')
    ap.add_argument("-backward_energy", help="Use backward energy map (default is forward)", action='store_true')
    args = vars(ap.parse_args())

    image_path, image_mask_path, output_file_name, remove_mask_path = args["im"], args["mask"], args["out"], args["remove_mask"]

    im = cv2.imread(image_path)
    assert im is not None
    mask = cv2.imread(image_mask_path, 0) if image_mask_path else None
    remove_mask = cv2.imread(remove_mask_path, 0) if remove_mask_path else None

    USE_FORWARD_ENERGY = not args["backward_energy"]

    # downsize image for faster processing
    ht, wt = im.shape[:2]
    if SHOULD_DOWNSIZE and wt > DOWNSIZE_WIDTH:
        im = resize(im, width=DOWNSIZE_WIDTH)
        if mask is not None:
            mask = resize(mask, width=DOWNSIZE_WIDTH)
        if remove_mask is not None:
            remove_mask = resize(remove_mask, width=DOWNSIZE_WIDTH)

    # image resize mode
    if args["resize"]:
        dy, dx = args["dy"], args["dx"]
        assert dy is not None and dx is not None
        output = seam_carving(im, dy, dx, mask, args["visualize_seam"])
        print("Shape of original image ",im.shape)
        print("Shape of resized image ",output.shape)
        cv2.imwrite(output_file_name, output)

    # object removal mode
    elif args["remove"]:
        assert remove_mask is not None
        output = object_removal(im, remove_mask, mask, args["visualize_seam"], args["hremove"])
        print("Shape of original image ",im.shape)
        print("Shape of object removal image ",output.shape)
        cv2.imwrite(output_file_name, output)


