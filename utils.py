import os
import cv2
import numpy as np
import random
import math
from math import log10, sqrt 
from tqdm import tqdm
random.seed(0)
os.getcwd()


def video_to_images(input_path, img_save=False, ex_frame=0, freq=1, out_path='output'):
    '''Create directory and extract frames for each video.
    
       args:
            input_path (str): input video path
            img_save (bool): save image or not
            ex_frame (int): initial frames to exclude
            freq (int): every n frames to save
            out_path (str): output path to save the frames
       returns:
            frames (list): list of frames
            fps (float): FPS of input video
       '''
    
    if img_save:
        if not os.path.exists(out_path):
            os.makedirs(out_path)
    vid_name = input_path.split('/')[-1].split('.')[0]
    vidcap = cv2.VideoCapture(input_path)
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    count = 1
    frames = []
    num_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    print("number of frames: ",num_frames)
    with tqdm(total=num_frames, desc="Reading frames") as pbar:
        while vidcap.isOpened():
            ret, frame = vidcap.read()
            if ret:
                h, w = frame.shape[:2]
                if img_save:
                    if count >= ex_frame:
                        if count%freq==0:  # save 1 every n consecutive frames
                            cv2.imwrite(os.path.join(out_path, '%s_%d.png')%(vid_name, count),frame)
                    
                else:
                    if count%freq==0:  # take 1 every n consecutive frames
                        frames.append(frame)
                count += 1
                pbar.update(1)
            else:
                break
    vidcap.release()

    return frames, fps

# crop an image from center
def center_crop_image(image, target_size):
    # Get the dimensions of the image
    height, width = image.shape[:2]
    # Calculate the center coordinates
    center_x, center_y = width // 2, height // 2
    # Calculate the crop half size based on the target size
    crop_half = target_size // 2
    if height > target_size and width > target_size:
        # Calculate the crop region
        x1 = center_x - crop_half
        x2 = center_x + crop_half
        y1 = center_y - crop_half
        y2 = center_y + crop_half
        # Perform the crop
        cropped_image = image[y1:y2, x1:x2]
        return cropped_image
    elif height > target_size and width == target_size:
        # Calculate the crop region
        y1 = center_y - crop_half
        y2 = center_y + crop_half
        # Perform the crop
        cropped_image = image[y1:y2, 0:width]
        return cropped_image
    elif height == target_size and width > target_size:
        # Calculate the crop region
        x1 = center_x - crop_half
        x2 = center_x + crop_half
        # Perform the crop
        cropped_image = image[0:height, x1:x2]
        return cropped_image
    else:
        print("Check image shape and target size!")





def reflection_removal(img, alg="NS"):
    hh, ww = img.shape[:2]
    # threshold
    lower = (150,150,150)
    upper = (255,255,255) 
    thresh = cv2.inRange(img, lower, upper)
    # apply morphology close and open to make mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10,10))  #(10,10)
    morph = cv2.morphologyEx(morph, cv2.MORPH_DILATE, kernel, iterations=1)
    # morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel, iterations=1)

    # floodfill the outside with black
    black = np.zeros([hh + 2, ww + 2], np.uint8)
    mask = morph.copy()
    mask = cv2.floodFill(mask, black, (0,0), 0, 0, 0, flags=8)[1]
    # use mask with input to do inpainting
    if alg == "NS":
        result = cv2.inpaint(img, mask, 3.5, cv2.INPAINT_NS)
    else:
        result = cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA)

    return result



def sliding_window(image, stepSize, windowSize):
    ''' Slice image according windowsize
        Args: 
            image(numpy array): input image
            stepSize(int): step size to shift the sliding window
            windowSize(int, int): height and width of the sliding window
        Return: image patch 
    '''
    # slide a window across the image
    for y in range(0, image.shape[0], stepSize):
        for x in range(0, image.shape[1], stepSize):
            if image[y:y + windowSize[1], x:x + windowSize[0]].shape[0] == windowSize[0] and image[y:y + windowSize[1], x:x + windowSize[0]].shape[1] == windowSize[1]:
#                 print("success!")
                # yield the current window
                yield (image[y:y + windowSize[1], x:x + windowSize[0]])  #x, y, 
            else:
                break




def contrast_enhance(image):

    ''' Enhance contrast of image
    '''
    # Adaptive histogram equalization for further enhancement
    lab= cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a, b = cv2.split(lab)

    # Applying CLAHE to L-channel
    # feel free to try different values for the limit and grid size:
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(l_channel)
    # merge the CLAHE enhanced L-channel with the a and b channel
    limg = cv2.merge((cl,a,b))
    # Converting image from LAB Color model to BGR color spcae
    enhanced_img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
 
    return enhanced_img


def zero_out_triangle(arr, length: int, side='bottom_right'):
    '''Input: 
        arr (ndarray): Input array
        length (int): length of triangle to zero out
        side (str): which side to zero out

       Return: zero out array'''
    
    if side == 'bottom_right':
        mask = lambda N, k: np.fliplr(np.tri(N, k=k-N)) ==1
    elif side == 'top_left':
        mask = lambda N, k: np.flip(np.fliplr(np.tri(N, k=k-N))) == 1
    
    arr[mask(arr.shape[0], length)] = 0

    return arr


def closest_power_of_2(max_dim: int) -> int:
    '''
    Calculates the closest power of 2 to the given maximum dimension of image.

    Args:
        max_dim (int): maximum dimension height/width of image

    Returns:
        closest power of 2 of the maximum dimension (int)
    '''

    # Calculate the exponent for the closest power of 2
    exponent = int(math.floor(math.log2(max_dim)))

    # Calculate the two closest powers of 2
    lower_power = 2 ** exponent
    upper_power = 2 ** (exponent + 1)

    # Return the closer of the two powers
    return lower_power if abs(max_dim - lower_power) <= abs(max_dim - upper_power) else upper_power


def make_square_zero_padded_image(image, max_height: int, max_width: int):
    """Creates a square image with zero padding using OpenCV.

    Args:
        image: A NumPy array representing the image.

    Returns:
        A NumPy array representing the padded square image.
    """

    height, width = image.shape[:2]
    if height < max_height and width < max_width :
        top = int((max_height - height) / 2)
        bottom = int(max_height - height - top)
        left = int((max_width - width) / 2)
        right = int(max_width - width - left)
        
    elif height < max_height and width >= max_width :
        top = int((max_height - height) / 2)
        bottom = int(max_height - height - top)
        left = 0
        right = 0

    elif height >= max_height and width < max_width :
        top = 0
        bottom = 0
        left = int((max_width - width) / 2)
        right = int(max_width - width - left)
    
    padded_image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])

    return padded_image

def PSNR(original, compressed, mean=False):
    if mean:
        psnr_vals = []
        psnr_l = []
        for org, com in zip(original, compressed):
            mse = np.mean((org - com) ** 2) 
            if(mse == 0):  # MSE is zero means no noise is present in the signal . 
                        # Therefore PSNR have no importance. 
                return 100
            max_pixel = 255.0
            psnr = 20 * log10(max_pixel / sqrt(mse)) 
            psnr_l.append(psnr)
            psnr_vals.append("PSNR: {0:.3f}".format(psnr))

        mean_psnr = sum(psnr_l)/len(psnr_l)
        return psnr_vals, mean_psnr
    else:        
        mse = np.mean((original - compressed) ** 2) 
        if(mse == 0):  # MSE is zero means no noise is present in the signal . 
                    # Therefore PSNR have no importance. 
            return 100
        max_pixel = 255.0
        psnr = 20 * log10(max_pixel / sqrt(mse)) 
        return psnr 
    



def image_resize(image):
    '''
    Resize the image to square shape of dimension 2^n for walsh-hadamard transform

    Args:
        image: Input image

    Returns:
        image: Resized image
        closest_power (int): Integer value of closest power of 2 of image max dimension
    '''

    # Get the closest power of 2 using the maximum image shape
    closest_power = closest_power_of_2(max(image.shape))
    if closest_power > image.shape[0] or closest_power > image.shape[1]:
        image = make_square_zero_padded_image(image, closest_power, closest_power)
        if closest_power < image.shape[0] or closest_power < image.shape[1]: 
            image = center_crop_image(image, closest_power)
    elif closest_power < image.shape[0] or closest_power < image.shape[1]:
        # Perform the center crop
        image = center_crop_image(image, closest_power)

    return image, closest_power