import os
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import random
import math
from math import log10, sqrt 
from tqdm import tqdm
from stream_loader import StreamLoader
import pdb
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
                            # cv2.imwrite(os.path.join(out_path, '%d.png')%count,frame[0:h, 470:w-80])
                    
                else:
                    if count%freq==0:  # take 1 every n consecutive frames
                        frames.append(frame)
                count += 1
                pbar.update(1)
            else:
                break
    vidcap.release()

    return frames, fps


def cover_image_portion(image, x1=0, y1=690, x2=140, y2=768):
  """
  Covers a rectangular portion of an image with black pixels.

  Args:
    image_path: Path to the image file.
    x1, y1: Top-left corner coordinates of the rectangular portion.
    x2, y2: Bottom-right corner coordinates of the rectangular portion.

  Returns:
    A new image with the specified portion covered in black.
  """

  # Create a black rectangle with the specified dimensions
  mask = np.zeros_like(image)  # Create mask with same channels as image
  #Flip the mask color to white
  mask[:, :] = [255, 255, 255]
#   print(mask.shape)
  mask = cv2.rectangle(mask, (x1, y1), (x2, y2), (0, 0, 0), -1)  # Fill rectangle with black
  # Apply the mask to the image using bitwise AND operation
  covered_image = cv2.bitwise_and(image, mask)

  return covered_image


def process_video(input_path, txt_file=None, stframe=0, endframe=38711, x1=345, y1= 0, yh=0, xw=85, freq=1, out_path=None):
    '''Create directory and save processed video.'''    
    
    if txt_file:
        with open(txt_file, 'r') as f:
            files = f.readlines()
        frames_tokeep = [int(i) for i in files]   # [int(i.split('.')[0]) for i in files]
        frames_tokeep.sort()
    else:
        frames_tokeep = []
        for i in range(stframe,endframe+1,1):
            frames_tokeep.append(i)

    cap = cv2.VideoCapture(input_path)
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if not os.path.exists(out_path):
            os.makedirs(out_path)

    h, w = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height, width = h-y1-yh, w-x1-xw
    # Define the codec and create VideoWriter object
    #fourcc = cv2.cv.CV_FOURCC(*'DIVX')
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    #out = cv2.VideoWriter('output.avi',fourcc, 20.0, (640,480))
    out_file = Path(input_path)
    out = cv2.VideoWriter(os.path.join(out_path, (out_file.stem + '.mp4')), fourcc, fps, (width,height))

    count = 1
    # Progress bar setup
    with tqdm(total=num_frames, desc="Saving video") as pbar:
        while(cap.isOpened()):
            ret, frame = cap.read()
            
            if ret==True:
                # frame = cv2.flip(frame,0)
                h, w = frame.shape[:2]
                if count in frames_tokeep:
                    if count%freq==0:  # save 1 every n consecutive frames
                        # cover unwanted region
                        cov_img = cover_image_portion(frame[y1:h-yh, x1:w-xw])  # frame[0:h, 470:w-80] , frame[0:h, 340:w-65]
                        out.write(cov_img)  # save cropped frame
                count += 1
                pbar.update(1)
            
            else:
                break
    # Release everything if job is finished
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print("Finished saving video!!")

    return 0

def make_video(input_dir: str, chunk_size: int, skip_freq: int, add_score: bool=False, sliding_window: bool=False, scores: pd.DataFrame=pd.DataFrame(), fname: str='test.mp4', out_path: str='output/') -> int:
    '''Create directory and save video.'''    

    if not os.path.exists(out_path):
            os.makedirs(out_path)
    

    video_stream = StreamLoader(input_dir,
                            0, chunk_size, True, False, skip_freq)
    frames, start_frame , fps, total_frames = video_stream.run()
    height, width = frames[0].shape[:2]
    # Define the codec and create VideoWriter object
    #fourcc = cv2.cv.CV_FOURCC(*'DIVX')
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    out = cv2.VideoWriter(os.path.join(out_path, fname), fourcc, fps, (width,height))

    streaming = True
    total_chunks = int(total_frames/chunk_size)
    while streaming:
        # Progress bar setup
        with tqdm(total=total_chunks, desc="Saving video") as pbar:
            for i,frame in tqdm(enumerate(frames)):
                if add_score:
                    if sliding_window:
                        score = scores[scores['frame_number']== start_frame+i]
                        cols = ['mean_ratio','mse','ssim','psnr', 'ks_statistics']
                        mse_thresh = score[score['mse']<65]
                        blurriness = len(mse_thresh)/len(score)
                        if blurriness > 0.5:
                            color = (0,0,255)
                        else:
                            color = (0,255,0)
                        for j,col in enumerate(cols):
                            txt = col+': '+ ', '.join([f'{num:.3f}' for num in score[col].tolist()])
                            cv2.putText(frame, txt, (20, 50*(j+1)), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.6, 
                                    color, 
                                    1, 
                                    lineType = cv2.LINE_AA)
                    else:
                        score = scores.iloc[start_frame+i]
                        for j in range(score.iloc[1:].shape[0]):
                            if score['mse'] < 65:
                                color = (0,0,255)
                            else:
                                color = (0,255,0)
                            txt = score.keys()[j+1] + ": {:.3f}".format(score[j+1])
                            cv2.putText(frame, txt, (20, 50*(j+1)), cv2.FONT_HERSHEY_SIMPLEX,
                                    1, 
                                    color, 
                                    1, 
                                    lineType = cv2.LINE_AA)
                    
                out.write(frame)  
                pbar.update(1)

        frames, start_frame, fps, _ = video_stream.run()
        if not(len(frames)):
            streaming = False

    out.release()
    print("Finished saving video!!")
    return 0


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


def image_norm(image, type="L2"):
    
    # https://en.wikipedia.org/wiki/Matrix_norm#Frobenius_norm
    # https://en.wikipedia.org/wiki/Matrix_norm
    # https://www.kdnuggets.com/2023/05/vector-matrix-norms-numpy-linalg-norm.html
    
    # Ensure the image is a NumPy array
    image = np.array(image, dtype=np.float32)
    
    if type == "L2":
        ## L2 Norm
        # Step 1: Square each element of the matrix
        squared_elements = np.square(image)
        # Step 2: Sum up all the squared elements
        sum_of_squared_elements = np.sum(squared_elements)
        # Step 3: Take the square root of the result to get the L2 norm
        L2_norm = np.sqrt(sum_of_squared_elements)
        return L2_norm
    
    elif type == "inf":
        ## Infinite Norm
        # Step 1: Calculate the absolute sum for each row of the matrix
        absolute_row_sums = np.sum(np.abs(image), axis=1)
        # Step 2: Identify the row with the largest absolute sum
        infinite_norm = np.max(absolute_row_sums)

        return infinite_norm

    else:
        print("Norm type is not mentioned!")


def lp_norm(image, ord=None):

    # Flatten the image into a 1D array
    flat_image = image.flatten()

    # Calculate Lp norm
    if ord == 'inf':
        norm_value = np.linalg.norm(flat_image, ord=np.inf)
    else:
        norm_value = np.linalg.norm(flat_image)

    return norm_value


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