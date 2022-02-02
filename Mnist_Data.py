import gzip
import numpy as np
import matplotlib.pyplot as plt
f = gzip.open('/home/semi/Machine_Learning/train-images-idx3-ubyte.gz','r')

image_size = 28
num_images = 1000

import numpy as np
f.read(16)
buf = f.read(image_size * image_size * num_images)
data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
data = data.reshape(num_images, image_size, image_size, 1)


for i in range (num_images):
    image_i = np.array(data[i]).squeeze()
    plt.imshow(image_i) 
    path = "/home/semi/Machine_Learning/Images/image_{index}.png".format(index = i)
    plt.savefig(path)


f = gzip.open('/home/semi/Machine_Learning/train-labels-idx1-ubyte.gz','r')
f.read(8)
LINES = ['image,caption\n']
for i in range(num_images):   
    buf = f.read(1)
    labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
    if (labels[0] == 0) : 
      LINES.append('Image_'+ str(i) + ' , zero zero zero zero zero zero zero zero zero zero zero\n')
    elif (labels[0] == 1) : 
      LINES.append('Image_'+ str(i) + ' , one one one one one one one one one one one\n')
    elif (labels[0] == 2) : 
      LINES.append('Image_'+ str(i) + ' , two two two two two two two two two two two\n')
    elif (labels[0] == 3) : 
      LINES.append('Image_'+ str(i) + ' , tree tree tree tree tree tree tree tree tree tree\n')
    elif (labels[0] == 4) : 
      LINES.append('Image_'+ str(i) + ' , four four four four four four four four four four\n')
    elif (labels[0] == 5) : 
      LINES.append('Image_'+ str(i) + ' , five five five five five five five five five five\n')
    elif (labels[0] == 6) : 
      LINES.append('Image_'+ str(i) + ' , six six six six six six six six six  six\n')
    elif (labels[0] == 7) : 
      LINES.append('Image_'+ str(i) + ' , seven seven seven seven seven seven seven seven seven seven\n')
    elif (labels[0] == 8) : 
      LINES.append('Image_'+ str(i) + ' , eight eight eight eight eight eight eight eight  eight eight\n')
    elif (labels[0] == 9) : 
      LINES.append('Image_'+ str(i) + ' , nine nine nine nine nine nine nine nine nine nine\n')
a_file = open("/home/semi/Machine_Learning/captions.txt", "w")
a_file.writelines(LINES)
a_file.close()
