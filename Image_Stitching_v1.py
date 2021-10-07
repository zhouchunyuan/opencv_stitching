import cv2
import numpy as np
import sys
from pathlib import Path
import re
from PyQt5 import QtWidgets, uic
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import QEvent
from PyQt5.QtWidgets import QApplication, QLabel,QWidget, QPushButton, QVBoxLayout

debugmode = False

class MyLabelPixmap(QLabel):
    def __init__(self,imgfile):
        QLabel.__init__(self)
        self.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Ignored)
        self.pixmap = QPixmap(imgfile)
        self.setPixmap(self.pixmap)
        self.installEventFilter(self)

    def eventFilter(self, source, event):
        if (source is self and event.type() == QEvent.Resize):
            self.setPixmap(self.pixmap.scaled(self.size()))
        return super(QLabel, self).eventFilter(source, event)
def n2yx(cols,N,type="snake"):
    n = N-1
    y = int(n/cols)
    x = n % cols
    if type=="snake":
        if y%2 != 0:
            x = cols-1 - x
    return y,x
    
def find_Left_And_Up_Neighbors(cols,N,type="snake"):
    left_neighbor = -1;
    up_neighbor = -1;
    y,x = n2yx(cols,N,type=type)
    if y>0 :
        if y%2==0:# current on evenline, up_neighbor is ord line
            up_neighbor = (y-1)*cols + cols-1 - x
        else:
            up_neighbor = (y-1)*cols + x
        up_neighbor += 1
        
    if x>0 :
        if y%2==0:# current on evenline, left_neighbor same line
            left_neighbor = y*cols + x-1
        else:
            left_neighbor = y*cols + cols-1 - x +1  
        left_neighbor += 1
    return left_neighbor,up_neighbor
def showNeighborImages(cols,N,type="snake"):
    filename = "faces_"+str(N)+".tif"
    print(filename)
    im = cv2.imread(filename)
    h = im.shape[0]
    w = im.shape[1]
    
    left,up = find_Left_And_Up_Neighbors(cols,N,type=type)
    
    cv2.namedWindow("current")
    cv2.namedWindow("left")
    cv2.namedWindow("up")
    cv2.moveWindow("current", 500, 200)
    cv2.moveWindow("left", 500-w, 200)
    cv2.moveWindow("up", 500, 200-h)
    cv2.imshow("current",im)
    print(N,left,up)
    if left != -1 : 
        filenameleft = "faces_"+str(left)+".tif"
        imleft = cv2.imread(filenameleft)
        cv2.imshow("left",imleft)
    if up != -1 : 
        filenameup = "faces_"+str(up)+".tif"
        imup = cv2.imread(filenameup)
        cv2.imshow("up",imup)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
class Image_Stitching():
    def __init__(self) :
        self.cols = 10;
        self.rows = 10;
        self.N = 100;
        self.overlap = 0.1
        self.ratio=0.7
        self.min_match=10
        self.sift=cv2.SIFT.create()
        self.smoothing_window_size=800
        self.working_dir = "."
        self.registered_dir = "./registered"

    def GUI(self,cvImg):
        app = QApplication([])
        window = QWidget()
        layout = QVBoxLayout()
        layout.addWidget(QPushButton('Top'))
        layout.addWidget(QPushButton('Bottom'))
        label = MyLabelPixmap(cvImg)
        layout.addWidget(label)
        window.setLayout(layout)
        window.resize(800, 600)
        window.show()
        app.exec()
    
    def showMap(self,imfile,m,n):
        parentFolder = Path(imfile).parent.absolute()
        tiffiles = list(parentFolder.glob('*.tif'))
        
        img = cv2.imread(imfile)
        imgH = img.shape[0]
        imgW = img.shape[1]
        k = 1-self.overlap
        kimgW = int(imgW*k)
        kimgH = int(imgH*k)
        mask = np.zeros((kimgH*m, kimgW*n,3))
        for tif in tiffiles:
            im = cv2.imread(str(tif))
            numTxt = re.search(r".*_(\d+).tif", str(tif))
            y,x = n2yx(10,int(numTxt.group(1)))
            mask[y*kimgH:(y+1)*kimgH,
                 x*kimgW:(x+1)*kimgW] = im[0:kimgH,0:kimgW]

        cv2.imwrite('test.jpg', mask) 
    def showRegisteredMap(self,imfile):
        m = self.cols
        n = self.rows
        parentFolder = Path(imfile).parent.absolute()
        tiffiles = list(parentFolder.glob('*.tif'))
        
        img = cv2.imread(imfile)
        imgH = img.shape[0]
        imgW = img.shape[1]
        k = 1-self.overlap
        kimgW = int(imgW*k)
        kimgH = int(imgH*k)
        mask = np.zeros((kimgH*m, kimgW*n,3))
        offsetH = int(self.overlap*imgH/2)
        offsetW = int(self.overlap*imgW/2)
        for tif in tiffiles:
            im = cv2.imread(str(tif))
            numTxt = re.search(r".*_(\d+).tif", str(tif))
            y,x = n2yx(10,int(numTxt.group(1)))
            mask[y*kimgH:(y+1)*kimgH,
                 x*kimgW:(x+1)*kimgW] = im[offsetH:kimgH+offsetH,offsetW:kimgW+offsetW]

        cv2.imwrite('registered.jpg', mask)         
    def doRegister(self,imfile):
        parentFolder = Path(imfile).parent.absolute()
        tiffiles = list(parentFolder.glob('*.tif'))
        
        for tif in tiffiles:
            im = cv2.imread(str(tif))
            numTxt = re.search(r".*_(\d+).tif", str(tif))
            self.registrationNeighbour(int(numTxt.group(1)))

    def registrationNeighbour(self,N):

        filename = "faces_"+str(N)+".tif"
        im = cv2.imread(filename)
        imgH = im.shape[0]
        imgW = im.shape[1]

        left_up_list = find_Left_And_Up_Neighbors(self.cols,N=N)

        if debugmode:
            print(left_up_list)
        for i,num in enumerate(left_up_list):
            if num != -1:
                neighborName = "faces_"+str(num)+".tif"
                imNeighbor = cv2.imread(neighborName)
                imCopy = im.copy()
                if i==0:#left
                    x1 = int(imgW*(1-self.overlap))
                    x0 = int(imgW*self.overlap)
                    imNeighbor[:,0:x0] = imNeighbor[:,x1:]
                    imNeighbor[:,x0:]= 0
                    imCopy[:,x0:]=0
                else:#up
                    y1 = int(imgH*(1-self.overlap))
                    y0 = int(imgH*self.overlap)
                    imNeighbor[0:y0,:] = imNeighbor[y1:,:]
                    imNeighbor[y0:,:] = 0
                    imCopy[y0:,:] = 0
                #cv2.imshow("left_up",imNeighbor)
                #cv2.waitKey(0)
                    
                kp1, des1 = self.sift.detectAndCompute(imNeighbor, None)
                kp2, des2 = self.sift.detectAndCompute(imCopy, None)
                matcher = cv2.BFMatcher()
                raw_matches = matcher.knnMatch(des1, des2, k=2)
                good_points = []
                good_matches=[]
                for m1, m2 in raw_matches:
                    x1,y1 = kp1[ m1.queryIdx ].pt;
                    x2,y2 = kp2[ m2.trainIdx ].pt;
                    #make sure only compare the overlap area
                    #if x1<imgW*(1-self.overlap) or x2>imgW*self.overlap : continue

                    if m1.distance < self.ratio * m2.distance:
                        good_points.append((m1.trainIdx, m1.queryIdx))
                        good_matches.append([m1])
                if debugmode==True:
                    img3 = cv2.drawMatchesKnn(imNeighbor, kp1, imCopy, kp2, good_matches, None, flags=2)
                    cv2.imwrite('matching'+str(i)+'.jpg', img3)
                
                rangx = range(int(imgW*self.overlap),imgW,int(imgW/6))# make 5 non-change points (x)
                rangy = range(int(imgH*self.overlap),imgH,int(imgH/6))# make 5 non-change points (y)
                kp2_nonChange = np.float32([(x,y) for x in rangx for y in rangy])# non-change points 5x5
                kp1_nonChange = kp2_nonChange #np.float32([(x+240-24,y) for x in rangx for y in rangy])# non-change points
                imNeighbor_kp = kp1_nonChange;
                im_kp = kp2_nonChange;
                if len(good_points) > self.min_match:
                    _kp1 = np.float32(
                        [kp1[i].pt for (_, i) in good_points])
                    _kp2 = np.float32(
                        [kp2[i].pt for (i, _) in good_points])
                   
                    imNeighbor_kp = np.append(imNeighbor_kp,_kp1,axis=0)
                    im_kp = np.append(im_kp,_kp2,axis=0)

                H, status = cv2.findHomography(im_kp,imNeighbor_kp, cv2.RANSAC,5.0)
                im = cv2.warpPerspective(im, H, (imgW, imgH))
        cv2.imwrite('registered/'+filename, im)
             
    def registration(self,img1,img2):
        imgH = img1.shape[0]
        imgW = img1.shape[1]

        kp1, des1 = self.sift.detectAndCompute(img1, None)
        kp2, des2 = self.sift.detectAndCompute(img2, None)
        matcher = cv2.BFMatcher()
        raw_matches = matcher.knnMatch(des1, des2, k=2)
        good_points = []
        good_matches=[]
        for m1, m2 in raw_matches:
            x1,y1 = kp1[ m1.queryIdx ].pt;
            x2,y2 = kp2[ m2.trainIdx ].pt;
            #make sure only compare the overlap area
            if x1<imgW*(1-self.overlap) or x2>imgW*self.overlap : continue

            if m1.distance < self.ratio * m2.distance:
                good_points.append((m1.trainIdx, m1.queryIdx))
                good_matches.append([m1])
        img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good_matches, None, flags=2)
        cv2.imwrite('matching.jpg', img3)

        rangx = range(int(imgW/2),imgW,int(imgW/5))# make 5 non-change points (x)
        rangy = range(0,imgH,int(imgH/5))          # make 5 non-change points (y)
        kp2_nonChange = np.float32([(x,y) for x in rangx for y in rangy])# non-change points 5x5
        kp1_nonChange = np.float32([(x+240-24,y) for x in rangx for y in rangy])# non-change points
        image1_kp = kp1_nonChange;
        image2_kp = kp2_nonChange;
        if len(good_points) > self.min_match:
            _kp1 = np.float32(
                [kp1[i].pt for (_, i) in good_points])
            _kp2 = np.float32(
                [kp2[i].pt for (i, _) in good_points])
           
            image1_kp = np.append(image1_kp,_kp1,axis=0)
            image2_kp = np.append(image2_kp,_kp2,axis=0)

        H, status = cv2.findHomography(image2_kp, image1_kp, cv2.RANSAC,5.0)
        return H

    def create_mask(self,img1,img2,version):
        height_img1 = img1.shape[0]
        width_img1 = img1.shape[1]
        width_img2 = img2.shape[1]
        self.smoothing_window_size = width_img1 ## added by chunyuan zhou
        height_panorama = height_img1
        width_panorama = width_img1 +width_img2
        offset = int(self.smoothing_window_size / 2)
        barrier = img1.shape[1] - int(self.smoothing_window_size / 2)
        mask = np.zeros((height_panorama, width_panorama))
        if version== 'left_image':
            mask[:, barrier - offset:barrier + offset ] = np.tile(np.linspace(1, 0, 2 * offset ).T, (height_panorama, 1))
            mask[:, :barrier - offset] = 1
        else:
            mask[:, barrier - offset :barrier + offset ] = np.tile(np.linspace(0, 1, 2 * offset ).T, (height_panorama, 1))
            mask[:, barrier + offset:] = 1
        return cv2.merge([mask, mask, mask])

    def blending(self,img1,img2):
        H = self.registration(img1,img2)
        height_img1 = img1.shape[0]
        width_img1 = img1.shape[1]
        width_img2 = img2.shape[1]
        height_panorama = height_img1
        width_panorama = width_img1 +width_img2

        panorama1 = np.zeros((height_panorama, width_panorama, 3))
        mask1 = self.create_mask(img1,img2,version='left_image')
        panorama1[0:img1.shape[0], 0:img1.shape[1], :] = img1
        #panorama1 *= mask1
        cv2.imwrite('mask1.jpg', panorama1) # debug
        mask2 = self.create_mask(img1,img2,version='right_image')
        panorama2 = cv2.warpPerspective(img2, H, (width_panorama, height_panorama))*mask2
        cv2.imwrite('panorama2.jpg', panorama2) # debug
        panorama1[:,int(width_img1*(1-self.overlap/2)):width_img1,:]=0#clean first half overlap area
        panorama2[:,0:int(width_img1*(1-self.overlap/2)),:]=0#clean 2nd overlap area
        result=panorama1+panorama2
        

        rows, cols = np.where(result[:, :, 0] != 0)
        min_row, max_row = min(rows), max(rows) + 1
        min_col, max_col = min(cols), max(cols) + 1
        final_result = result[min_row:max_row, min_col:max_col, :]
        return final_result
def main(argv1,argv2):
    debugmode=True
    #img1 = cv2.imread(argv1)
    #img2 = cv2.imread(argv2)
    #final=Image_Stitching().blending(img1,img2)
    #cv2.imwrite('panorama.jpg', final)
    #Image_Stitching().showMap(argv1,10,10)
    #Image_Stitching().GUI("test.jpg")
    #for i in range(100):
    #showNeighborImages(10,42)
    #Image_Stitching().registrationNeighbour(42)
    Image_Stitching().doRegister(argv1)
    Image_Stitching().showRegisteredMap("./registered/faces_1.tif")
if __name__ == '__main__':
    try: 
        main(sys.argv[1],sys.argv[2])
    except IndexError:
        print ("Please input two source images: ")
        print ("For example: python Image_Stitching.py '/Users/linrl3/Desktop/picture/p1.jpg' '/Users/linrl3/Desktop/picture/p2.jpg'")