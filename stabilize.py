
import cv2
import numpy as np
import os,sys
import math as m

def getMotion(filename, start, frames, psone):
    cap = cv2.VideoCapture(filename)
    cap.set(cv2.CAP_PROP_POS_FRAMES,start)

    skip=1
    largeSkip=60
    if psone:
        skip=largeSkip

    prev = np.array([])
    dx=0.0
    dy=0.0
    dth=0.0
    global allTransforms
    for tt in range(frames):
        # Capture frame-by-frame
        _, frame = cap.read()
        if frame is None:
            break
        if (tt%skip)==0:
            print(tt)
            if not(prev.size):
                prev = frame.copy()       
            rigid_mat = cv2.estimateRigidTransform(frame, prev, False)
            #rigid_mat2 = cv2.estimateRigidTransform(prev,frame, False)
            prev = frame.copy()       
            if rigid_mat is None:
                continue
            #dx+=0.5*(rigid_mat[0,2]-rigid_mat2[0,2])
            dx+=rigid_mat[0,2]
            #dy+=0.5*(rigid_mat[1,2]-rigid_mat2[1,2])
            dy+=rigid_mat[1,2]
            #dth+=0.5*(m.atan2(rigid_mat[1,0],rigid_mat[0,0])-m.atan2(rigid_mat2[1,0],rigid_mat2[0,0]))
            dth+=m.atan2(rigid_mat[1,0],rigid_mat[0,0])
            if not psone:
                if (tt%largeSkip)==0:
                    ddx = (dx-allTransforms[tt,0])
                    ddy = (dy-allTransforms[tt,1])
                    ddth = (dth-allTransforms[tt,2])
                    for back in range(int(0.75*largeSkip)):
                        allTransforms[tt-1-back,0]-=ddx*m.exp(-float(back)/(0.15*(float(largeSkip))))
                        allTransforms[tt-1-back,1]-=ddy*m.exp(-float(back)/(0.15*(float(largeSkip))))
                        allTransforms[tt-1-back,2]-=ddth*m.exp(-float(back)/(0.15*(float(largeSkip))))
                    dx=allTransforms[tt,0]
                    dy=allTransforms[tt,1]
                    dth=allTransforms[tt,2]
            allTransforms[tt,0]=dx
            allTransforms[tt,1]=dy
            allTransforms[tt,2]=dth
    cap.release()

def outputMovie(filename, outputfilename, start, frames):
    cap = cv2.VideoCapture(filename)
    cap.set(cv2.CAP_PROP_POS_FRAMES,start)

    S = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    out = cv2.VideoWriter(outputfilename, cv2.VideoWriter_fourcc('M','J','P','G'), cap.get(cv2.CAP_PROP_FPS), S, True)

    MC = np.zeros((2,3))
  
    global allTransforms
    for tt in range(frames):
        # Capture frame-by-frame
        _, frame = cap.read()
        if frame is None:
            break
        dx=allTransforms[tt,0]
        dy=allTransforms[tt,1]
        dth=allTransforms[tt,2]
        MC[0,0]=m.cos(dth)
        MC[0,1]=-m.sin(dth)
        MC[1,0]=m.sin(dth)
        MC[1,1]=m.cos(dth)
        MC[0,2]=dx
        MC[1,2]=dy
        finalframe = cv2.warpAffine(frame,MC,S)#, cv2.WARP_INVERSE_MAP) 
        finalframe[:,0:50,...]=0.0
        finalframe[:,1870:,...]=0.0
        finalframe[0:50,...]=0.0
        finalframe[1030:,...]=0.0
        out.write(finalframe)

    # When everything done, release the capture
    cap.release()

if __name__ == '__main__':
    FULLNAME = sys.argv[1]
    frameStart = int(sys.argv[2])
    frameLength = int(sys.argv[3])
    path, filename = os.path.split(FULLNAME)
    noext, ext = os.path.splitext(filename)
    allTransforms=np.zeros((frameLength,3))
    outputname = noext + '_FRW' + str(frameStart) + '.avi' 
    getMotion(FULLNAME, frameStart, frameLength, True)
    getMotion(FULLNAME, frameStart, frameLength, False)
    outputMovie(FULLNAME, outputname, frameStart, frameLength)
    print(outputname)

