# Analyzer for direct beam pointing of CICADA Terminal A


import cv2
import numpy as np
import matplotlib.pyplot as plt
from dtools import basic_functions as bf

# Create a VideoCapture object and read from input file
# If the input is the camera, pass 0 instead of the video file name
#cap = cv2.VideoCapture('Iris_33FPS_60_100_0_v2.mp4')
cap = cv2.VideoCapture('Iris_33FPS_gentleTap.mp4')
#cap = cv2.VideoCapture('No_IRIS_lowExposure.mp4')

maxPos = np.array([[0,0]])
maxValues = np.array([])
counter = 0
N=150
#while cap.isOpened():
for i in range(0,N):    
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(gray)

    cv2.circle(gray,maxLoc,20,(255,0,0),2)
    counter = counter + 1
    #imgplot = plt.imshow(gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    maxPos = np.vstack((maxPos,[maxLoc[0],maxLoc[1]]))
    maxValues = np.append(maxValues,[maxVal])
    curve = gray[:,maxLoc[0]]
    

maxPos = np.delete(maxPos,0,axis=0)

dev_horizontal = np.std(maxPos[:,0])
dev_vertical = np.std(maxPos[:,1])
#plt.plot(maxPos[:,1])
#plt.plot(maxPos[:,0])
print(dev_horizontal*3.75, " microns")
print(dev_vertical*3.75, " microns")


# Fourier analysis
Sf = np.fft.fftshift(np.fft.fft(maxPos[:,0]))
dT = 33e-3

freq = bf.fft_freq_axis_simple(N,dT)

Sf_max = np.fft.fftshift(np.fft.fft(maxValues))
plt.plot(freq,10*np.log10(np.abs(Sf_max)**2))

#plt.plot(10*np.log10(np.abs(Sf)**2/max(np.abs(Sf)**2)))
plt.xlim([0,16])
plt.xlabel('Fourier Frequency [Hz]')
plt.ylabel('Relative Intensity [dB]')
#plt.ylim([-20,5])


theta = 0
avg_pos_x = 3.75e-6*np.mean(maxPos[:,0])
f = 0.5
for i in range(0,N):
    delD = 3.75e-6*maxPos[i,0]-avg_pos_x
    #print(delD)
    theta = theta + np.sqrt((1/N)*(delD/f)**2)


print(theta/1e-6, 'urad')


cap.release()
#cap.destroyAllWindows()