import numpy as np
import cv2
import math

def line_insert(image,i,j,image_inf,image_segma,grad,image_inf_max):
    if image_inf[i,j,0]<0.1*image_inf_max:
        image[i, j]=0
    else:
        if grad[i,j,0] !=0:
            if 0 < image_segma[i,j]<= 1:
                color_x = image_inf[i - 1, j + 1, 0]
                color_y = image_inf[i, j + 1, 0]
                color_a = image_segma[i, j] * (color_x - color_y) + color_y
                color_x = image_inf[i + 1, j - 1, 0]
                color_y = image_inf[i, j - 1, 0]
                color_b = image_segma[i, j] * (color_x - color_y) + color_y
                if image_inf[i, j, 0] > max(color_a, color_b):
                    image[i, j] = image_inf[i,j,0]
            elif 1 < image_segma[i, j]:
                color_x = image_inf[i - 1, j + 1, 0]
                color_y = image_inf[i - 1, j, 0]
                color_a = (1 / image_segma[i, j]) * (color_x - color_y) + color_y
                color_x = image_inf[i + 1, j - 1, 0]
                color_y = image_inf[i + 1, j, 0]
                color_b = (1 / image_segma[i, j]) * (color_x - color_y) + color_y
                if image_inf[i, j, 0] > max(color_a, color_b):
                    image[i, j] = image_inf[i,j,0]
            elif -1 < image_segma[i, j] <= 0:
                color_x = image_inf[i + 1, j + 1, 0]
                color_y = image_inf[i, j + 1, 0]
                color_a = abs(image_segma[i, j]) * (color_x - color_y) + color_y
                color_x = image_inf[i - 1, j - 1, 0]
                color_y = image_inf[i, j - 1, 0]
                color_b = abs(image_segma[i, j]) * (color_x - color_y) + color_y
                if image_inf[i, j, 0] > max(color_a, color_b):
                    image[i, j] = image_inf[i,j,0]
            elif  image_segma[i, j] <= -1:
                color_x = image_inf[i + 1, j + 1, 0]
                color_y = image_inf[i + 1, j, 0]
                color_a = abs(1 / image_segma[i, j]) * (color_x - color_y) + color_y
                color_x = image_inf[i - 1, j - 1, 0]
                color_y = image_inf[i - 1, j, 0]
                color_b = abs(1 / image_segma[i, j]) * (color_x - color_y) + color_y
                if image_inf[i, j, 0] > max(color_a, color_b):
                    image[i, j] = image_inf[i,j,0]
    return image

def double_edge(h,w,image_inf,image_segma,grad):
    threshold_max=0.2
    threshold_min=0.1
    image=np.zeros((h,w),dtype=np.uint8)
    image_inf_max=np.max(image_inf[:,:,0])
    # print(image_inf_max)
    for i in range(1,h-1):
        for j in range(1,w-1):
            if abs(image_inf[i,j,0])>=image_inf_max*threshold_max:
                image=line_insert(image,i,j,image_inf,image_segma,grad,image_inf_max)
            elif image_inf_max * threshold_min<= abs(image_inf[i, j, 0]) < image_inf_max * threshold_max:
                image = line_insert(image, i, j, image_inf, image_segma,grad,image_inf_max)
            else :
                image = line_insert(image, i, j, image_inf, image_segma,grad,image_inf_max)
    TH=image_inf_max * threshold_max
    TL=image_inf_max*threshold_min

    for i in range(1,h-1):     # myself function
        for j in range(1,w-1):
            if image[i,j]>=TH:
                image[i,j]=255
            elif image[i,j]<=TL:
                image[i,j]=0
    for i in range(1,h-1):
        for j in range(1,w-1):
            if TL<image[i,j]<TH:
                if (image[i-2:i+2,j-2:j+2]>TH).any() :
                    image[i,j]=255
                else:
                    image[i,j]=0

    return image

def canny_edge(img):

    gray=cv2.cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h=gray.shape[0]
    w=gray.shape[1]

    # guas_img=np.zeros(h,w,dtype=np.float32)
    guas_img=cv2.GaussianBlur(gray,ksize=(5,5),sigmaX=0.5)

    grad = np.zeros((h,w,2),dtype=np.float32)
    grad[:,:,0] = cv2.Sobel(guas_img,cv2.CV_16S,1,0,ksize=3)
    grad[:,:,1] = cv2.Sobel(guas_img,cv2.CV_16S,0,1,ksize=3)

    image_inf=np.zeros((h,w,2),dtype=np.float32)
    image_inf[:,:,0]=np.sqrt(grad[:,:,0]**2+grad[:,:,1]**2)
    image_inf[:,:,1]=grad[:,:,1]/grad[:,:,0]
    image_segma=image_inf[:,:,1]
    for i in range(h):
        for j in range(w):
            image_inf[i,j,1]=math.atan(image_inf[i,j,1])*(180/math.pi)

    image=double_edge(h,w,image_inf,image_segma,grad)
    return image

if __name__=='__main__':
    img=cv2.imread("/home/longtong/PycharmProjects/homework/image/person.png")
    gray=cv2.cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img=canny_edge(img)

    cv2.imshow('',gray)
    cv2.imshow(' ',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

