from statistics import mean
from PIL import Image
import logging
import sys
import numpy as np

logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)


class ColorCube(object):
    def __init__(self, colors):
        self.colors = colors or []
        self.red = [r[0] for r in colors]
        self.green = [g[1] for g in colors]
        self.blue = [b[2] for b in colors]
        self.max_range = max(self.size) 
        self.size = (max(self.red) - min(self.red),  
                     max(self.green) - min(self.green),
                     max(self.blue) - min(self.blue))       
        self.max_channel = self.size.index(self.max_range) #mark which channel has max_range

    def split(self): # split a cube into two cubes by the color with max_range 
        middle = len(self.colors) // 2
        colors = sorted(self.colors, key=lambda c: c[self.max_channel])
        return ColorCube(colors[:middle]), ColorCube(colors[middle:])

    def average(self): # calculate the representative color       
        r = int(mean(self.red))
        g = int(mean(self.green))
        b = int(mean(self.blue))
        return r, g, b

    def __lt__(self, other):  # use for sort
        return self.max_range < other.max_range


def median_cut(img):
    colors = []
    for color_count, color in img.getcolors(img.width * img.height):
            colors += [color] * color_count
    cubes = [ColorCube(colors)]
    while len(cubes) < 256:
        cubes.sort()
        cubes += cubes.pop().split()

    return [c.average() for c in cubes]


def show_lut(colors):
    color_width = img.width / len(colors)
    color_height = int(max(100, color_width))
    color_size = (int(color_width), color_height)
    color_x = 0
    palette = Image.new('RGB', (img.width, color_height))

    for color in colors:
        color = Image.new('RGB', color_size, color)
        palette.paste(color, (int(color_x), 0))
        color_x += color_width
    palette.show()
    return palette



img = Image.open('./multimedia/redapple.jpg')
colors = median_cut(img)
palette = show_lut(colors) 

#get colors of the picture
rgb=(img.getdata())
rgb=list(rgb)
print(len(set(rgb)))
rgb=np.array(rgb)
rgbRepresent=[]
#get represent color for every color in origin picture
for i in rgb:
    print i
    dist=[]
    for j in colors:
        dist.append(np.sqrt(sum((i-j)**2))) #calculate distance 
    dist=np.array(dist)
    rgbRepresent.append([colors[dist.argsort()[0]]]) # choose the nearest cube

# create new images with replace color
rgbRepresent= [list(i[0]) for i in rgbRepresent]
data=[tuple(pixel) for pixel in rgbRepresent]
img2 = Image.new(img.mode, img.size)
img2.putdata(data)
#save result
imageio.imwrite("./apple.jpg",img2)

len(img2.getcolors(img2.size[0]*img2.size[1]))
