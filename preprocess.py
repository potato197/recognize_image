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
            if(col_count[i])>0:
                left=i
                right=left
                for j in range(i+1,Bpp.shape[1]):
                    if(col_count[j]>0):
                        right=j
                    else:
                        break
                blocks.append([left,right])
        print "blocks="
        print blocks
####3. merge blocks
        merged_blocks=[]
        merged_block=[-1,-1]
        next_index=-1
        checked_index=-1
        for i in range(0,len(blocks)-1):
            if (i<checked_index):
                continue
            curr_block=blocks[i]
            merged_block[0]=curr_block[0]
            merged_block[1]=curr_block[1]
            for j in range(i+1,len(blocks)-1):
                checked_index=j
                next_block=blocks[j]
                gap=next_block[0]-merged_block[1]
                if(gap<=8):
                    merged_block[1]=next_block[1]
                else:
                    break
            merged_blocks.append([merged_block[0],merged_block[1]])
        print "merged_blocks="
        print merged_blocks
####4. find max 2 blocks
        max_block_info=[[-1,0],[-1,0]]
        for i in range(0,len(merged_blocks)):
            curr_size=merged_blocks[i][1]-merged_blocks[i][0]
            if(curr_size>max_block_info[0][1]):
                max_block_info[1][0]=max_block_info[0][0]
                max_block_info[1][1]=max_block_info[0][1]
                max_block_info[0][0]=i
                max_block_info[0][1]=curr_size
            elif(curr_size>max_block_info[1][1]):
                max_block_info[1][0]=i
                max_block_info[1][1]=curr_size
        print "max_blocks="
        print max_block_info
####5. mark
        for i in range(0,len(max_block_info)):
            block_index=max_block_info[i][0]
            start=merged_blocks[block_index][0]
            end=merged_blocks[block_index][1]
            for r in range(0,Bpp.shape[0]):
                Bpp[r,start]=0
                Bpp[r,end]=0
        cv2.imwrite(outpath2+filename,Bpp)

        return Bpp

    def InterferLine(self,Bpp,filename):
        #print(type(Bpp.shape[0]), Bpp.shape[0])
        #print(type(Bpp.shape[1]), Bpp.shape[1])
        for i in range(0,76):
            for j in range(0,Bpp.shape[0]):
                Bpp[j][i]=255
        for i in range(161,Bpp.shape[1]):
            for j in range(0,Bpp.shape[0]):
                Bpp[j][i]=255        
        m=1
        n=1
        for i in range(76,161):
            while(m<Bpp.shape[0]-1):
                if Bpp[m][i]==0:
                    if Bpp[m+1][i]==0:
                        n=m+1
                    elif m>0 and Bpp[m-1][i]==0:
                        n=m
                        m=n-1
                    else:
                        n=m+1
                    break
                elif m!=Bpp.shape[0]:
                    l=0
                    k=0
                    ll=m
                    kk=m
                    while(ll>0):
                        if Bpp[ll][i]==0:
                            ll=11-1
                            l=l+1
                        else:
                            break
                    while(kk>0):
                        if Bpp[kk][i]==0:
                            kk=kk-1
                            k=k+1
                        else:
                            break
                    if (l<=k and l!=0) or (k==0 and l!=0):
                        m=m-1
                    else:
                        m=m+1
                else:
                    break
                #endif
            #endwhile
            if m>0 and Bpp[m-1][i]==0 and Bpp[n-1][i]==0:
                continue
            else:
                Bpp[m][i]=255
                Bpp[n][i]=255
            #endif
        #endfor
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