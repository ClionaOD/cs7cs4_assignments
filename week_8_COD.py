import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def convolve(arr, kernel):
    """
    Takes an n*n array and convolves it with the k*k kernel

    args:
        arr: the n*n numpy array
        kernel: the k*k numpy array
    
    returns:
        conv: the convolved matrix
    """
    output = np.zeros(((arr.shape[0]-kernel.shape[0])+1, (arr.shape[1]-kernel.shape[1])+1))
    
    k = kernel.shape[0]
    for i in range(output.shape[0]):
        sub_arr = arr[i:k+i,:]
        for j in range(output.shape[1]):
            filt_arr = sub_arr[:,j:k+j]

            val = np.sum(filt_arr * kernel)
            output[i,j] = val
    
    return output

img = Image.open('./imagenet_shark.jpg')
img = img.resize((200,200))
rgb = np.array(img.convert('RGB'))
r = rgb[:,:,0]
plt.imshow(r)
plt.savefig('./r_channel.jpg')
plt.show()
plt.close()

kernel1 = np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]])
kernel2 = np.array([[0,-1,0],[-1,8,-1],[0,-1,0]])

output1 = convolve(r, kernel1)
output2 = convolve(r, kernel2)

plt.imshow(output1)
plt.savefig('./kernel1.jpg')
plt.show()
plt.close()

plt.imshow(output2)
plt.savefig('./kernel2.jpg')
plt.show()
plt.close()

