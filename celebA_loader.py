import cv2
import numpy as np
import os
import random


def celeb_loader(dir='/home/airscan-razer04/Documents/datasets/img_align_celeba/',
                    randomize=True,
                    batch_size=64,
                    height=64,
                    width=64):
    list = os.listdir(dir) # dir is your directory path
    number_files = len(list)
    list.sort()
    #print(list)
    #print (number_files)
    while(1):
        if randomize == True:
            random.shuffle(list)
        img_list = list[:]

        while img_list:
            img_stack = np.zeros((batch_size, height, width,3),dtype=np.float32)

            for i in range(batch_size):
                #print(len(img_list), len(list))

                filename = dir + img_list.pop(0)
                #print(filename)
                img = cv2.imread(filename)
                #cv2.imshow("image",img)
                #cv2.waitKey(0)
                #cv2.destroyAllWindows()

                #print(img.shape)
                if img.shape != (64,64,3):
                    img_stack[i, :, :,:] = cv2.resize(img,(height,width))/255
                    #cv2.imshow("image",img_stack[i, :, :,:])
                    #cv2.waitKey(0)
                    #cv2.destroyAllWindows()

                    #print(img.shape)
                #print("imgs left",len(img_list))
                if len(img_list)==0:
                    if randomize == True:
                        random.shuffle(list)
                    img_list = list[:]
            yield img_stack, None
            #cv2.imshow("image",a)
            #cv2.waitKey(0)
            #cv2.destroyAllWindows()

if __name__ == '__main__':
    print("MAIN")
    some_gen = celeb_loader()
    a,b = next(some_gen)
    print(a[0,:,:,:])
    for i in range (a.shape[0]):
        cv2.imshow("image",a[i])
        cv2.waitKey(0)
        cv2.destroyAllWindows()
