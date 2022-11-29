import cv2
import numpy as np
import sys
import time


def create_mask(image):
    global pts; pts = []
    global x_lst; x_lst = []
    global img_m; img_m = image.copy()
    global mask_m; mask_m = np.zeros((img.shape[0], img.shape[1]))
    #mask_m = np.zeros(img_m.shape[:2])
   

    def draw_mask(pts):
        pts = np.array(pts, dtype=np.int32)
        pts.reshape((-1,1,2))
        cv2.fillPoly(mask_m,[pts],(255))
        cv2.imshow("image",mask_m)
        cv2.imwrite("mask.jpg", mask_m)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

   
    def mousePoints(event, x,y, flags, params):
        if event == cv2.EVENT_LBUTTONDOWN:
            img_1 = cv2.circle(img_m, (x,y), radius=2, color=(0, 0, 255), thickness=-1)
            cv2.imshow("original", img_1)
            pts.append((x,y))
            x_lst.append(x)
        elif event == cv2.EVENT_RBUTTONDOWN:
            draw_mask(pts)

            #cv2.circle(img_m, (x,y), radius=2, color=(0, 0, 255), thickness=-1)


    cv2.imshow("original", img_m)
    cv2.setMouseCallback("original", mousePoints)
    cv2.waitKey(0)
    return mask_m, (max(x_lst) - min(x_lst)) 

#img=cv2.imread('/Users/ruzuba/Documents/DOCUMENTS/SCHOOL/NKU/Programming/Videos-Course/Course1/CPP/S_M/SEAM-CARVING/Images/castle.jpg')
img=cv2.imread('/Users/ruzuba/Documents/DOCUMENTS/SCHOOL/NKU/Programming/Videos-Course/Course1/CPP/Comp/images/eiffel.jpg')
h = int((img.shape[0] * 500) / img.shape[1])
h = h+1 if h%2 != 0 else h
img = cv2.resize(img, (500, h))

while True:
    #print('\nEnter 1 for Normal Seam Carving\nEnter 2 for Object Removal\nEnter 3 for Seam Carving using Guassian pyramid: ')
    print('\nEnter 1 for Normal Seam Carving\nEnter 2 for Object Removal\nEnter 3 for Seam Carving using Guassian pyramid: ')

    op = (input())
    op = int(input("Choice:"))
    
    if op == 2:
        print("Initial Dimensions: ",img.shape[1]," x ",img.shape[0]," x ",img.shape[2])
        mask, dx = create_mask(img)
        start_time = time.time()
        #object_removal(img, mask, dx, start_time)  
        break
   
    else:
        print("Please enter valid input...")