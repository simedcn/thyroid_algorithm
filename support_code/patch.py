import os
import cv2

pics_dir=''
save_dir=''
img_length=img_width=299
overlap=100
dirs=os.listdir(pics_dir)
index=1
for file in dirs:
    print(str(index)+"\\"+str(len(dirs)))
    filename = os.path.join(pics_dir, file)
    portion=os.path.splitext(file)
    img = cv2.imread(filename)
    rows,cols= img.shape[:2]
    print(rows,cols)
    i=0
    ix = iy = 0
    while iy<rows:
        if iy+img_width<rows:
            while ix<cols:
                if ix+img_length<cols:
                    cropimg=img[iy:iy+img_width,ix:ix+img_length]
                    #print(1)
                else:
                    cropimg = img[iy: iy+img_width,cols-img_length:cols]
                    #print(cropimg.shape[:2])
                #print(pics_dir,portion[0],i,portion[1])
                cv2.imwrite(str(save_dir) + str(portion[0]) + "_" + str(i) + str(portion[1]),cropimg)
                i=i+1
                ix=ix+img_length-overlap
            ix=0
            iy=iy+img_width-overlap
        else:
            while ix<cols:
                if ix+img_length<cols:
                    cropimg=img[rows-img_width:rows,ix:ix+img_length]
                else:
                    cropimg = img[rows-img_width:rows,cols-img_length:cols]
                cv2.imwrite(str(save_dir) + str(portion[0]) + "_"+ str(i) + str(portion[1]),cropimg)
                i=i+1
                ix=ix+img_length-overlap
            break
    #cv2.imshow('img',img)
    #cv2.waitKey (0)
    index=index+1
    #print(index)
