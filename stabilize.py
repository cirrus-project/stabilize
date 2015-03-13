
import cv2
import numpy as np
import os,sys
import math as m



def cut_movie(filename, outputfilename, start, frames):
    cap = cv2.VideoCapture(filename)
    cap.set(cv2.CAP_PROP_POS_FRAMES,start)

    S = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    out = cv2.VideoWriter(outputfilename, cv2.VideoWriter_fourcc('M','J','P','G'), cap.get(cv2.CAP_PROP_FPS), S, True)

    MC = np.eye(3)
    prev = np.array([])
    for tt in range(frames):
        # Capture frame-by-frame
        _, frame = cap.read()
        if not(prev.size):
            prev = frame.copy()       
        rigid_mat = cv2.estimateRigidTransform(frame, prev, False)
        sr = rigid_mat[0:2,0:2]

        scale = m.sqrt(np.linalg.det(sr))
        rigid_mat[0:2,0:2] = sr/scale
        
        MC = np.dot(np.vstack([rigid_mat,(0,0,1)]),MC)
        finalframe = cv2.warpAffine(frame,MC[0:2,:],S, cv2.WARP_INVERSE_MAP) 
        out.write(finalframe)
        prev = frame.copy()       



    # When everything done, release the capture
    cap.release()

if __name__ == '__main__':
    FULLNAME = sys.argv[1]
    frameStart = int(sys.argv[2])
    frameLength = int(sys.argv[3])
    path, filename = os.path.split(FULLNAME)
    noext, ext = os.path.splitext(filename)
    outputname = noext + '_FR' + str(frameStart) + '.avi' 
    cut_movie(FULLNAME, outputname, frameStart, frameLength)
    print outputname

