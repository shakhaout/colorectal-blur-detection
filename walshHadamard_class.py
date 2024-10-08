import numpy as np
from scipy.linalg import hadamard
import random
from utils import  sliding_window, zero_out_triangle
random.seed(0)




class WalshHadamardMatrix():
    '''
    Generate Walsh-Hadamard sequency matrix of input matrix size (2^n)

    Resources:
         https://en.wikipedia.org/wiki/Walsh_matrix

         https://en.wikipedia.org/wiki/Bit-reversal_permutation

         https://en.wikipedia.org/wiki/Gray_code

         https://github.com/phy710/Hadamard-Network/blob/main/hadamard.py

         https://www.exptech.co.in/2021/03/video-44-walsh-hadamard-transform.html
    '''

    def __init__(self, matrix_size) -> None:
        '''
        Args:
            matrix_size (int): Integer value (2^n) of walsh matrix size
        '''
        self.matrix_size = matrix_size
        

    def reverseBits(self,number) -> int:
        '''
        Reverse the bits of input number 
        Args:
            number (int): Integer number

        Returns:
            Reversed bit integer number
        '''
        # Convert number into binary representation
        bitsize = np.log2(self.matrix_size).astype(int)
        reverse = str(int(bin(number)[2:].zfill(int(np.log2(self.matrix_size)))))
        reverse = (bitsize - len(reverse))*'0' + reverse
        reverse = reverse[::-1]
        # converts reversed binary string into integer
        return (int(reverse,2))

    def grayCode(self, number) -> int:
        '''
        Convert a value to a Gray code with the given base and digits.
        Args:
            number (int): Integer number

        Returns:
            GrayCode of Input number
        ''' 
        # Right Shift the number 
        # by 1 taking xor with  
        # original number 
        return number ^ (number >> 1) 

    def walsh_matrix(self):
        '''
        Create a walsh-hadamard sequency matrix
        Args:
            matrix_size (int): Integer matrix size of 2^n
        Return:
            walsh (matrix): walsh-hadamard matrix
        
        '''
        # Make a hadamard matrix of m_size
        Hadamard = hadamard(self.matrix_size)
        # bit-reversal permutation on the hadamard matrix
        bitreversed = np.array([Hadamard[self.reverseBits(i)] for i in range(self.matrix_size)])
        # gray-Code permutation on the bit-reversal matrix yeilds walsh matrix
        walsh = np.array([bitreversed[self.grayCode(i)] for i in range(self.matrix_size)])
        
        return walsh



class WalshHadamardTransform():
    '''
    Apply walsh-hadamard transform in the image
    '''

    def __init__(self, WHmatrix) -> None:
        '''
        Args:
            WHmatrix (matrix): walsh hadamard matrix
            closest_power (int): Nearest closest power of 2 of image shape
        '''
        self.WHmatrix = WHmatrix


    def walsh_transform(self, image, sliding_win=False, sl_win_size=8, stepsize=8, inverse=False, 
                        zero_out=False, zo_ptn=1.6, zo_Side="bottom_right", rcon_patch=False):
        '''
        Input: 
            Image, Image patch size

        Operation: 
            Apply Gaussian blur and Walsh transform to input images
        Return: 
            Return Walsh transformed images and Variance of walsh transformed image
        '''
        
        if sliding_win:
            WH_patch = []
            for window in sliding_window(image, stepSize=stepsize, windowSize=(sl_win_size, sl_win_size)):
                if len(image.shape) == 1:
                    if inverse:
                        walsh_img = (window@self.WHmatrix).astype('float64')
                    else:
                        walsh_img = ((window@self.WHmatrix)/(window.shape[0])).astype('float64')
                elif len(image.shape) == 2:
                    if inverse:
                        walsh_img = (self.WHmatrix@window@self.WHmatrix).astype('float64')
                    else:
                        walsh_img = ((self.WHmatrix@window@self.WHmatrix)/(window.shape[0]*window.shape[0])).astype('float64')
                # WH_img = (np.dot(walsh_img.T,walsh_mat)/np.sqrt(walsh_img.T.shape[0])).astype('float64')
                if zero_out:
                    walsh_img = zero_out_triangle(walsh_img, int(walsh_img.shape[0]*zo_ptn), side=zo_Side)
                WH_patch.append(walsh_img)

            return WH_patch
        
        elif rcon_patch:
            WH_patch = []
            for img_patch in image:
                if len(img_patch.shape) == 1:
                    if inverse:
                        walsh_img = (img_patch@self.WHmatrix).astype('float64')
                elif len(img_patch.shape) == 2:
                    if inverse:
                        walsh_img = (self.WHmatrix@img_patch@self.WHmatrix).astype('float64')
                WH_patch.append(walsh_img)
            return WH_patch
        
        else:
            # Apply Walsh Transformation
            if len(image.shape) == 1:
                if inverse:
                    walsh_img = (image@self.WHmatrix).astype('float64')
                else:
                    walsh_img = ((image@self.WHmatrix)/(image.shape[0])).astype('float64')
            elif len(image.shape) == 2:
                if inverse:
                    walsh_img = (self.WHmatrix@image@self.WHmatrix).astype('float64')
                else:
                    walsh_img = ((self.WHmatrix@image@self.WHmatrix)/(image.shape[0]*image.shape[0])).astype('float64')
            if zero_out:
                walsh_img = zero_out_triangle(walsh_img, int(walsh_img.shape[0]*zo_ptn), side=zo_Side)

            return walsh_img
    