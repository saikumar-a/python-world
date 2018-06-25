import matplotlib.pyplot as plt
import numpy as np
from numpy.ma.core import sin, cos

# plt.plot([1,2,3],[5,7,4])
# plt.show()

t=np.arange(0,0.98,0.01)

y1=sin(2*np.pi*4*t)
#Use subplots to display two plots side by side
# plt.subplot(1,2,1) 
plt.plot(t,y1,'b', label='sin')

y2=cos(2*np.pi*4*t)
# plt.subplot(1,2,2)
plt.plot(t,y2,'r', label='cos')
plt.xlabel('`time')
plt.ylabel('value')
plt.title('My Plot')


plt.tight_layout()
plt.show()

