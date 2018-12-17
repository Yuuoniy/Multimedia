# -*- coding: utf-8 -*- 
from PIL import Image
import numpy as np
import math
import cv2
import logging

#亮度量化矩阵
brightness_quantization_matrix = np.array([
 [16, 11, 10, 16, 24, 40, 51, 61],
 [12, 12, 14, 19, 26, 58, 60, 55],
 [14, 13, 16, 24, 40, 57, 69, 56],
 [14, 17, 22, 29, 51, 87, 80, 62],
 [18, 22, 37, 56, 68, 109, 103, 77],
 [24, 35, 55, 64, 81, 104, 113, 92],
 [49, 64, 78, 87, 103, 121, 120, 101],
 [72, 92, 95, 98, 112, 100, 103, 99]
])
#色度量化矩阵
chroma_quantization_matrix = np.array([[17, 18, 24, 47, 99, 99, 99, 99],
       [18, 21, 26, 66, 99, 99, 99, 99],
       [24, 26, 56, 99, 99, 99, 99, 99],
       [47, 66, 99, 99, 99, 99, 99, 99],
       [99, 99, 99, 99, 99, 99, 99, 99],
       [99, 99, 99, 99, 99, 99, 99, 99],
       [99, 99, 99, 99, 99, 99, 99, 99],
       [99, 99, 99, 99, 99, 99, 99, 99]])


def rgb2ycbcr(im):
    xform = np.array([[.299, .587, .114], [-.1687, -.3313, .5], [.5, -.4187, -.0813]])
    ycbcr = im.dot(xform.T)
    ycbcr[:,:,[1,2]] += 128
    return np.uint8(ycbcr)
  
def ycbcr2rgb(img, src_cb, src_cr, src_y):
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            y = src_y[i, j]
            cb = src_cb[i, j]
            cr = src_cr[i, j]
            img[i, j, 2] = min(255, max(0, round(1.402 * (cr - 128) + y)))
            img[i, j, 1] = min(255, max(0, round(-0.344136 * (cb - 128) - 0.714136 * (cr - 128) + y)))
            img[i, j, 0] = min(255, max(0, round(1.772 * (cb - 128) + y)))
    return img


def get_blocks(image,block_size=8):
    block_rows = np.split(image, image.shape[0] / block_size) # 行切分
    blocks = []
    for rows in block_rows:
        blocks.append(np.split(rows, rows.shape[1] / block_size, axis=1)) #列切分
    return blocks

def combine_blocks(blocks,shape,block_size=8):
    img = np.zeros(shape,np.uint8);
    indices = [(i,j) for i in range(0, shape[0], block_size) for j in range(0, shape[1], block_size)]
    for block, index in zip(blocks, indices):
      i, j = index
      img[i:i+block_size, j:j+block_size] = block
    return img

def get_dct_matrix(block_size=8):
    dct_matrix = np.zeros((block_size, block_size))
    for i in range(0, block_size):
        for j in range(0, block_size):
            if i == 0:
                dct_matrix[i][j] = 1/math.sqrt(block_size)
            else:
                dct_matrix[i][j] = math.sqrt(2*1.0/block_size) * math.cos((math.pi*(2 * j + 1) * i)/(2*block_size))
    return dct_matrix


#反量化 
def de_quantization(dct_result,quantization_matrix,quality_factor=0.1):
    dct_coefficient = quantization_matrix*dct_result*quality_factor
    return dct_coefficient
    

def quantization(dct_coefficient,quantization_matrix,quality_factor=0.1):
    qm = quantization_matrix*quality_factor # 乘以质量因子改变压缩率
    return np.round(dct_coefficient/qm)  # rint = round to int 


def fit_block_size(image,block_size=8):
    new_height = (image.shape[0] // 16) * 16 # Taking off the extra bytes
    new_width = (image.shape[1] // 16) * 16
    return image[0:new_height, 0:new_width]


def sampling(img):
    samp_y, samp_cb, samp_cr = [], [], []
    for i in range(img.shape[0]):
      for j in range(img.shape[1]):
        samp_y.append(img[i, j, 0])
        if i % 2 == 0 and j % 2 == 0:
            samp_cb.append(img[i, j, 1])
        elif j % 2 == 0:
            samp_cr.append(img[i, j, 2])
    samp_y = np.array(samp_y).reshape([img.shape[0], img.shape[1]])
    samp_cb = np.array(samp_cb).reshape([img.shape[0] // 2,img.shape[1] // 2])
    samp_cr = np.array(samp_cr).reshape([img.shape[0] // 2, img.shape[1] // 2])
    return samp_y, samp_cb, samp_cr

def inverse_sampling(samp_cb, samp_cb,height,width):
    src_cb, src_cr = np.zeros([height, width]), np.zeros([height, width])
    for i in range(samp_cb.shape[0]):
        for j in range(samp_cb.shape[1]):
            val_cb, val_cr = samp_cb[i][j], samp_cb[i][j]
            src_cb[i*2:i*2+2, j*2:j*2+2] = np.array([val_cb, val_cb, val_cb, val_cb]).reshape([2, 2])
            src_cr[i*2:i*2+2, j*2:j*2+2] = np.array([val_cr, val_cr, val_cr, val_cr]).reshape([2, 2])
    return src_cb, origin_cr

def dct(block_matrix, dct_matrix):
    res_dct_matrix = np.dot(np.dot(dct_matrix,block_matrix), np.transpose(dct_matrix))
    return res_dct_matrix

def idct(block_matrix, dct_matrix):
    idct_matrix = np.dot(np.dot(dct_matrix.transpose(),block_matrix), dct_matrix)
    return idct_matrix

#单独对每个通道进行操作
def process(img,quantization_matrix):
    img = fit_block_size(img)
    blocks = get_blocks(img, 8)
    res_blocks =[]
    dct_matrix = get_dct_matrix(8)
    #对每个块进行处理
    for i in range(len(blocks)):
        for j in range(len(blocks[i])):
            dct_coeff = dct(blocks[i][j], dct_matrix) # dct 
            qua_res_matrix = quantization(dct_coeff, quantization_matrix) # 量化
            #反量化
            dct_coefficient = de_quantization(qua_res_matrix,quantization_matrix)
            # IDCT
            idct_res_matrix = idct(dct_coefficient,dct_matrix)
            res_blocks.append(idct_res_matrix); #将操作结果存在块中
    #拼接块
    res = combine_blocks(res_blocks,img.shape)
    return res


def encoder(path,sava_path):
    im = Image.open(path)
    im = np.asarray(im)
    #首先转化为 ycbcr
    ycbcr=rgb2ycbcr(im)
    #首先进行 padding
    ycbcr  = fit_block_size(ycbcr)
    (height,width) = ycbcr.shape[0], ycbcr.shape[1]
    #二次采样
    img_y, img_cb,img_cr = sampling(ycbcr)
    #分别对每个进行操作
    res_y = process(img_y,brightness_quantization_matrix)
    res_cb = process(img_cb,chroma_quantization_matrix)
    res_cr = process(img_cr,chroma_quantization_matrix)
    #合并图像
    (res_cb,res_cr)= inverse_sampling(res_cb,res_cr,height,width)
    img = np.zeros([height,width,3], dtype=np.uint8)
    ycbcr2rgb(img,res_cb,res_cr,res_y)
    #比较图像失真率
    cv2.imshow("result", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite(sava_path,img)
    return img

def compare(img1,img2):
    # print(img1.shape,img2.shape)
    height  = min(img1.shape[0],img2.shape[0])
    width  = min(img1.shape[1],img2.shape[1])
    diff = 0
    diff_r =  np.uint64(0)
    diff_g =  np.uint64(0)
    diff_b =  np.uint64(0)
    for i in range(height):
        for j in range(width):
            diff_r += (img1[i,j,0]-img2[i,j,0])**2;
            print(img1[i,j,0]-img2[i,j,0])
            diff_g += (img1[i,j,1]-img2[i,j,1])**2;
            diff_b += (img1[i,j,2]-img2[i,j,2])**2;
    diff+= diff_r
    diff+= diff_g
    diff+= diff_b
    return diff / (height * width*3)

if __name__ == '__main__':
    cartoon_encode = encoder("../data/cartoon.jpg","../result/cartoon.jpeg")
    animal_encode = encoder("../data/animal.jpg","../result/animal.jpeg")
   
    cartoon_origin = np.asarray(Image.open("../data/cartoon.jpg"))
    animal_origin = np.asarray(Image.open("../data/animal.jpg"))
     #比较失真率 jpeg:
    jpeg_cartoon_loss = compare(cartoon_origin,cartoon_encode)
    jpeg_animal_loss = compare(animal_origin,animal_encode)
    print(jpeg_cartoon_loss,jpeg_animal_loss)
    #GIF:
