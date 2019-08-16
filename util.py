import matplotlib.pyplot as plt
import numpy as np

def plot_image_labels(images,labels=None,index=0,rows=4,cols=4,cmap=None):
    for i in range(rows * cols):
        ax = plt.subplot(rows,cols,i+1)#generate with 5 rows and 5 clos,and it's i+1 th picture
        ax.imshow(np.squeeze(images[index+i]),cmap = cmap)
        if labels is not None:
            title="label="+str(np.argmax(labels[i+index]))
            ax.set_title(title,fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
    plt.show()