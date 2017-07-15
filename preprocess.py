#import caffe
import cv2
import numpy as np
import os
import time
import random
import os

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
#        cv2.imwrite(outpath+filename,Bpp)
        return Bpp

    def FindBppBoundaryRow(self,Bpp,col_start,col_end):
####1. scan
        row_count = np.zeros(Bpp.shape[0])
        for r in range(0,Bpp.shape[0]):
            r_count=0
            for c in range(col_start,col_end):
                if(Bpp[r,c]==0):
                    r_count=r_count+1
            row_count[r]=r_count
        print row_count
####2. find blocks
        blocks=[]
        up=-1
        down=-1
        r_sum=0
        r_no_empty_count=0
        r_avarage=0
        for i in range(0,Bpp.shape[0]):
            if(i<=down):
                continue
            block_size=row_count[i]
            if(block_size>0):
                up=i
                down=i
                for j in range(i+1,Bpp.shape[0]):
                    if(row_count[j]>0):
                        down=j
                        block_size=block_size+row_count[j]
                    else:
                        break
                r_sum=r_sum+block_size
                r_no_empty_count=r_no_empty_count+(down-up+1)
                blocks.append([up,down,block_size])
        r_avarage=r_sum/r_no_empty_count
        print "row_blocks="
        print blocks
        print "r_avarage="
        print r_avarage
####3. merge blocks
        merged_blocks=[]
        merged_block=[-1,-1,-1]
        next_index=-1
        checked_index=-1
        for i in range(0,len(blocks)):
            if (i<checked_index):
                continue
            curr_block=blocks[i]
            if(curr_block[2]<(r_avarage-1)*(curr_block[1]-curr_block[0]+1)):
                continue
            merged_block[0]=curr_block[0]
            merged_block[1]=curr_block[1]
            merged_block[2]=curr_block[2]
            for j in range(i+1,len(blocks)):
                checked_index=j
                next_block=blocks[j]
                if(next_block[2]<(r_avarage-1)*(next_block[1]-next_block[0]+1)):
                    break
                gap=next_block[0]-merged_block[1]
                if(gap<=3):
                    merged_block[1]=next_block[1]
                    merged_block[2]=merged_block[2]+next_block[2]
                else:
                    break
            merged_blocks.append([merged_block[0],merged_block[1],merged_block[2]])
        print "merged_row_blocks="
        print merged_blocks
####4. find max row blocks[start,height,size]
        max_block_info=[[-1,-1,-1]]
        for i in range(0,len(merged_blocks)):
            curr_height=merged_blocks[i][1]-merged_blocks[i][0]
            if(curr_height>max_block_info[0][1]):
                max_block_info[0][0]=i
                max_block_info[0][1]=curr_height
                max_block_info[0][2]=merged_blocks[i][2]
        print "max_row_blocks="
        print max_block_info
        i=max_block_info[0][0]
        return (merged_blocks[i][0],merged_blocks[i][1])

    def FindBppBoundary(self,Bpp,root,filename):
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
        c_sum=0
        c_no_empty_count=0
        c_avarage=0
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
                c_sum=c_sum+block_size
                c_no_empty_count=c_no_empty_count+(right-left+1)
        c_avarage=c_sum/c_no_empty_count
        print "blocks="
        print blocks
        print "c_avarage=",c_avarage
####3. merge blocks
        merged_blocks=[]
        merged_block=[-1,-1,-1]
        next_index=-1
        checked_index=-1
        for i in range(0,len(blocks)):
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
#                if(gap<=10 and ((curr_block[2]+next_block[2])>(c_avarage-1)*(next_block[1]-curr_block[0]+1))):
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
        max_1_size=max_block_info[0][1]
        max_2_size=max_block_info[1][1]
        if(max_1_size<max_2_size*2):
            sort_block_info=[]
            max_block_index0=max_block_info[0][0]
            max_block_index1=max_block_info[1][0]
            if(max_block_index0>max_block_index1):
                sort_block_info.append(merged_blocks[max_block_index1])
                sort_block_info.append(merged_blocks[max_block_index0])
            else:
                sort_block_info.append(merged_blocks[max_block_index0])
                sort_block_info.append(merged_blocks[max_block_index1])
            print "sort_block_info=",sort_block_info
            for i in range(0,len(sort_block_info)):
                start=sort_block_info[i][0]
                end=sort_block_info[i][1]
#                for r in range(0,Bpp.shape[0]):
#                    Bpp[r,start]=0
#                    Bpp[r,end]=0
                (row_start,row_end)=self.FindBppBoundaryRow(Bpp,start+1,end)
                self.CutImage(Bpp,root,filename,i,row_start,row_end,start,end)
#                for c in range(start,end):
#                    Bpp[row_start,c]=0
#                    Bpp[row_end,c]=0                
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
#            for r in range(0,Bpp.shape[0]):
#                Bpp[r,start]=0
#                Bpp[r,end]=0
#                Bpp[r,min_c]=0
            (row_start,row_end)=self.FindBppBoundaryRow(Bpp,start+1,min_c)
#            for c in range(start,min_c):
#                Bpp[row_start,c]=0
#                Bpp[row_end,c]=0  
            self.CutImage(Bpp,root,filename,0,row_start,row_end,start,min_c)
            (row_start,row_end)=self.FindBppBoundaryRow(Bpp,min_c+1,end)
#            for c in range(min_c,end):
#                Bpp[row_start,c]=0
#                Bpp[row_end,c]=0    
            self.CutImage(Bpp,root,filename,1,row_start,row_end,min_c,end)
#        cv2.imwrite(outpath2+filename,Bpp)
        return Bpp

    def CutImage(self,Bpp,root,filename,dir_str_index,x_start,x_end,y_start,y_end):
        w=x_end-x_start+1
        h=y_end-y_start+1
        b1=np.zeros((w,h))
        for i in range(0,w):
            for j in range(0,h):
                b1[i][j]=Bpp[x_start+i][y_start+j]
        res = cv2.resize(b1,(32, 32), interpolation = cv2.INTER_LINEAR)
        n=root.rfind("\\")
        dir_name=root[n+1:]
        out_path=outpath2+"\\"+dir_name.decode('gbk')[dir_str_index].encode('gbk')
        if not os.path.exists(out_path):
            os.mkdir(out_path)
#        cv2.imwrite(out_path+"\\"+filename.decode('gbk')[0].encode('gbk')+'_'+'%d' %(time.time()*1000)+str(random.randint(1000,9999))+'.png',res)
        cv2.imwrite(out_path+"\\"+filename,res)

PP=PreProcess()
curr_path=os.getcwd()
src_dir="images"
out_dir="images1"
out_dir2="images2"
inpath=curr_path+"\\"+src_dir
outpath=curr_path+"\\"+out_dir
outpath2=curr_path+"\\"+out_dir2
if not os.path.exists(outpath2):
    os.mkdir(outpath2)
#print '1'
for root,dirs,files in os.walk(inpath):
    print "root=",root
    print "dirs=",dirs
    for filename in files:
        print filename
        Img=cv2.imread(root+'/'+filename)#No Chinese char
        NoRedImg=PP.RemoveRedColor(Img,filename)
        #print Img.shape
        GrayImage=PP.ConvertToGray(NoRedImg,filename)
        #print GrayImage.shape
        Bpp=PP.ConvertTo1Bpp(GrayImage,filename)
        Bpp2=PP.FindBppBoundary(Bpp,root,filename)
        #print(Bpp[1].shape[0],Bpp[1].shape[1])
        #Bpp_new=PP.InterferLine(Bpp[1],filename)
        #cv2.imwrite(outpath2+filename,Bpp_new)
        #b=PP.CutImage(Bpp_new,filename)