#import caffe
import cv2
import numpy as np
import os
import time
import random

class PreProcess(object):
    """description of class"""
    def ConvertToGray(self,Image,filename):
        GrayImage=cv2.cvtColor(Image,cv2.COLOR_BGR2GRAY)
#        cv2.imwrite(outpath+filename,GrayImage)
        return GrayImage

    def RemoveRedColor(self,Image,filename):
        for r in range(0,Image.shape[0]):
            for c in range(0,Image.shape[1]):
                if(Image[r,c,0]<=127 and Image[r,c,1]<=127 and Image[r,c,2]>=127):
                    Image[r,c,0]=255
                    Image[r,c,1]=255
                    Image[r,c,2]=255
                if(Image[r,c,0]>63 and Image[r,c,1]>63 and Image[r,c,2]>63):
                    Image[r,c,0]=255
                    Image[r,c,1]=255
                    Image[r,c,2]=255
        #r_max=np.linspace(255,255,Image.shape[0]*Image.shape[1]).reshape(Image.shape[0],Image.shape[1])
        #Image[:,:,2]=r_max
#        cv2.imwrite(outpath+filename,Image)
        return Image

    def ConvertTo1Bpp(self,GrayImage,filename):
#        img = cv2.medianBlur(GrayImage,3)
#        arr col_count = []
        img = GrayImage
        ret, th1 = cv2.threshold(img,90,255,cv2.THRESH_BINARY)
        th2 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
                cv2.THRESH_BINARY,201,127)
        th3 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
                cv2.THRESH_BINARY,101,63)
        Bpp = th1
#        print Bpp
#        return Bpp
#        img2, contours, hierarchy = cv2.findContours(Bpp,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
#        cv2.drawContours(img2,contours,-1,(0,0,255),3) 
        for c in range(0,Bpp.shape[1]):
            c_count=0
            for r in range(0,Bpp.shape[0]):
                if(Bpp[r,c]==0):
                    c_count=c_count+1
            if(c_count<=3):
                for i in range(0,Bpp.shape[0]):
                    Bpp[i,c]=255
                c_count=0
#            print c_count
#            col_count[c]=c_count
        cv2.imwrite(outpath+filename,Bpp)
        return Bpp
    
    def FindBppContours(self,Bpp,filename):
#        contours, hierarchy = cv2.findContours(Bpp[1],cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE) 
#        cv2.imwrite(outpath+filename,Bpp[1])
####1. scan
        col_count = np.zeros(Bpp.shape[1])
        for c in range(0,Bpp.shape[1]):
            c_count=0
            for r in range(0,Bpp.shape[0]):
                if(Bpp[r,c]==0):
                    c_count=c_count+1
            col_count[c]=c_count
#        cv2.imwrite(outpath+filename,Bpp)
        print col_count
####2. find blocks
        blocks=[]
        left=-1
        right=-1
        for i in range(0,Bpp.shape[1]):
            if(i<=right):
                continue
            block_size=col_count[i]
            if(block_size>0):
                left=i
                right=i
                for j in range(i+1,Bpp.shape[1]):
                    if(col_count[j]>0):
                        right=j
                        block_size=block_size+col_count[j]
                    else:
                        break
                blocks.append([left,right,block_size])
        print "blocks="
        print blocks
####3. merge blocks
        merged_blocks=[]
        merged_block=[-1,-1,-1]
        next_index=-1
        checked_index=-1
        for i in range(0,len(blocks)-1):
            if (i<checked_index):
                continue
            curr_block=blocks[i]
            merged_block[0]=curr_block[0]
            merged_block[1]=curr_block[1]
            merged_block[2]=curr_block[2]
            for j in range(i+1,len(blocks)):
                checked_index=j
                next_block=blocks[j]
                gap=next_block[0]-merged_block[1]
                if(gap<=10):
                    merged_block[1]=next_block[1]
                    merged_block[2]=merged_block[2]+next_block[2]
                else:
                    break
            merged_blocks.append([merged_block[0],merged_block[1],merged_block[2]])
        print "merged_blocks="
        print merged_blocks
####4. find max 2 blocks[start,width,size]
        max_block_info=[[-1,-1,-1],[-1,-1,-1]]
        for i in range(0,len(merged_blocks)):
            curr_width=merged_blocks[i][1]-merged_blocks[i][0]
            if(curr_width>max_block_info[0][1]):
                max_block_info[1][0]=max_block_info[0][0]
                max_block_info[1][1]=max_block_info[0][1]
                max_block_info[1][2]=max_block_info[0][2]
                max_block_info[0][0]=i
                max_block_info[0][1]=curr_width
                max_block_info[0][2]=merged_blocks[i][2]
            elif(curr_width>max_block_info[1][1]):
                max_block_info[1][0]=i
                max_block_info[1][1]=curr_width
                max_block_info[1][2]=merged_blocks[i][2]
        print "max_blocks="
        print max_block_info
####5. mark
        block_index0=max_block_info[0][0]
        block_index1=max_block_info[1][0]
        max_1_size=max_block_info[0][1]
        max_2_size=max_block_info[1][1]
        if(max_1_size<max_2_size*2):
            for i in range(0,len(max_block_info)):
                block_index=max_block_info[i][0]
                start=merged_blocks[block_index][0]
                end=merged_blocks[block_index][1]
                for r in range(0,Bpp.shape[0]):
                    Bpp[r,start]=0
                    Bpp[r,end]=0
        else:
            block_index=max_block_info[0][0]
            start=merged_blocks[block_index][0]
            end=merged_blocks[block_index][1]
            middle=(start+end)/2
            m1=max(middle-14,start)
            m2=min(middle+14,end)
            min_c_size=merged_blocks[block_index][2]
            min_c=-1
#            print "c_size="
            for c in range(m1,m2):
                c_size=0
                for r in range(0,Bpp.shape[0]):
                    if(Bpp[r,c]==0):
                        c_size=c_size+1
                    if(Bpp[r,c+1]==0):
                        c_size=c_size+1
                    if(Bpp[r,c+2]==0):
                        c_size=c_size+1
#                print c_size
                if(c_size<min_c_size):
                    min_c_size=c_size
                    min_c=c
#            print middle,min_c
            for r in range(0,Bpp.shape[0]):
                Bpp[r,start]=0
                Bpp[r,end]=0
                Bpp[r,min_c]=0
        cv2.imwrite(outpath2+filename,Bpp)

        return Bpp

    def CutImage(self,Bpp,filename):
        b1=np.zeros((Bpp.shape[0],20))
        for i in range(78,98):
            for j in range(0,Bpp.shape[0]):
                b1[j][i-78]=Bpp[j][i]
        cv2.imwrite(outpath+filename.decode('gbk')[0].encode('gbk')+'_'+'%d' %(time.time()*1000)+str(random.randint(1000,9999))+'.png',b1)

        b2=np.zeros((Bpp.shape[0],19))
        for i in range(99,118):
            for j in range(0,Bpp.shape[0]):
                b2[j][i-99]=Bpp[j][i]
        cv2.imwrite(outpath+filename.decode('gbk')[1].encode('gbk')+'_'+'%d' %(time.time()*1000)+str(random.randint(1000,9999))+'.png',b2)

        b3=np.zeros((Bpp.shape[0],19))
        for i in range(119,138):
            for j in range(0,Bpp.shape[0]):
                b3[j][i-119]=Bpp[j][i]
        cv2.imwrite(outpath+filename.decode('gbk')[2].encode('gbk')+'_'+'%d' %(time.time()*1000)+str(random.randint(1000,9999))+'.png',b3)

        b4=np.zeros((Bpp.shape[0],19))
        for i in range(139,158):
            for j in range(0,Bpp.shape[0]):
                b4[j][i-139]=Bpp[j][i]
        cv2.imwrite(outpath+filename.decode('gbk')[3].encode('gbk')+'_'+'%d' %(time.time()*1000)+str(random.randint(1000,9999))+'.png',b4)
        #return (b1,b2,b3,b4)

PP=PreProcess()
inpath='E:/program/recognize_image/images'
outpath='E:/program/recognize_image/images1/'
outpath2='E:/program/recognize_image/images2/'
#print '1'
for root,dirs,files in os.walk(inpath):
    for filename in files:
        print filename
        Img=cv2.imread(root+'/'+filename)#No Chinese char
        NoRedImg=PP.RemoveRedColor(Img,filename)
        #print Img.shape
        GrayImage=PP.ConvertToGray(NoRedImg,filename)
        #print GrayImage.shape
        Bpp=PP.ConvertTo1Bpp(GrayImage,filename)
        Bpp2=PP.FindBppContours(Bpp,filename)
        #print(Bpp[1].shape[0],Bpp[1].shape[1])
        #Bpp_new=PP.InterferLine(Bpp[1],filename)
        #cv2.imwrite(outpath2+filename,Bpp_new)
        #b=PP.CutImage(Bpp_new,filename)