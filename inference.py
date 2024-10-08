import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import cv2
import yaml
from utils import reflection_removal
from walshHadamard_class import WalshHadamardMatrix, WalshHadamardTransform
from scipy.stats import  ks_2samp



class BlurDetect():
    '''
    Detect blurry images
    '''

    def __init__(self) -> None:
        # read the config file
        with open('config.yml', "r") as file:
            try:
                self.cfg = yaml.safe_load(file)
                print("Config file read successfully!")
            except yaml.YAMLError as exc:
                print(exc)


    def blur_analysis(self, resized_img, gray_img, gray_blur, live=False, dst_row=None)->dict:
        '''
        Analyze blur in images using walsh-hadamard transform.
        '''

        val_dict = {}
        if self.cfg['image_selection']['sliding_window']:
            walsh_matrix = WalshHadamardMatrix(self.cfg['image_selection']['sl_win_size'])
            walsh_mat = walsh_matrix.walsh_matrix()
            # Apply the Walsh-Hadamard transform
            wh_transform = WalshHadamardTransform(walsh_mat)
            # walsh-hadamard transform
            transformed_img = wh_transform.walsh_transform(gray_img, sliding_win=self.cfg['image_selection']['sliding_window'], 
                                                                sl_win_size=self.cfg['image_selection']['sl_win_size'], 
                                                                stepsize=self.cfg['image_selection']['stepsize'],
                                                                zero_out=self.cfg['image_selection']['zero_out'])
            transformed_blur = wh_transform.walsh_transform(gray_blur, sliding_win=self.cfg['image_selection']['sliding_window'], 
                                                        sl_win_size=self.cfg['image_selection']['sl_win_size'], 
                                                        stepsize=self.cfg['image_selection']['stepsize'],
                                                        zero_out=self.cfg['image_selection']['zero_out'])
        
            ks_statistics = []
            for t_p, b_p in zip(transformed_img, transformed_blur):
                ks_statistic, _ = ks_2samp(np.absolute(t_p[t_p!=0].ravel()), np.absolute(b_p[b_p!=0].ravel()))
                ks_statistics.append(ks_statistic)
            val_dict['ks_statistic'] = np.mean(ks_statistics)

        else:
            walsh_matrix = WalshHadamardMatrix(resized_img.shape[0])
            walsh_mat = walsh_matrix.walsh_matrix()
            # Apply the Walsh-Hadamard transform
            wh_transform = WalshHadamardTransform(walsh_mat)
            # walsh-hadamard transform
            transformed_img = wh_transform.walsh_transform(gray_img, 
                                                            zero_out=self.cfg['image_selection']['zero_out'])
            transformed_blur = wh_transform.walsh_transform(gray_blur, 
                                                        zero_out=self.cfg['image_selection']['zero_out'])
            
            ks_statistic, ks_pval = ks_2samp(np.absolute(transformed_img[transformed_img!=0].ravel()), np.absolute(transformed_blur[transformed_blur!=0].ravel()))
        
        val_dict['ks_statistic'] = ks_statistic
        val_dict['ks_pval'] = ks_pval
    
        if live:
            val_dict['DMOS'] = dst_row['dmos']
            val_dict['src_frame'] = dst_row['refnames_all']
            val_dict['distortions'] = dst_row['distortions']  

        return val_dict


    def blur_detection(self)-> int:
        '''
        Detect blur in images and save blurriness data.
        Returns:
            0 (int): return 0 for success
        '''

        filepath = []
        blur_val = []
        for dir,subdir,files in tqdm(os.walk(self.cfg['image_selection']['input_path'])):
            if self.cfg['image_selection']['sliding_window']:
                blur_dir = os.path.join(self.cfg['image_selection']['out_path'], 'sl_wn/blur')
                sharp_dir = os.path.join(self.cfg['image_selection']['out_path'], 'sl_wn/sharp')
            else:
                blur_dir = os.path.join(self.cfg['image_selection']['out_path'], 'blur')
                sharp_dir = os.path.join(self.cfg['image_selection']['out_path'], 'sharp')
            os.makedirs(blur_dir, exist_ok=True)
            os.makedirs(sharp_dir, exist_ok=True)
            for file in tqdm(files):
                print(file)
                if file.endswith('.png'):
                    img = cv2.imread(os.path.join(dir,file))
                    if self.cfg['image_selection']['sliding_window']:
                        resized_img = cv2.resize(img.copy(),(1024,1024))
                    else:
                        resized_img = cv2.resize(img.copy(),(512,512))
                    resized_img = reflection_removal(resized_img)
                    reblur = cv2.GaussianBlur(resized_img, (0,0),1.3)
                    gray_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)
                    gray_blur = cv2.cvtColor(reblur, cv2.COLOR_BGR2GRAY)
                    val_dict = self.blur_analysis(resized_img, gray_img, gray_blur)
                    filepath.append(os.path.join(dir,file))
                    blur_val.append(val_dict['ks_statistic'])
                    txt = "Blurriness: {}".format(str(val_dict['ks_statistic']))
                    cv2.putText(img, txt, (20, 50), cv2.FONT_HERSHEY_SIMPLEX,
                                                    0.6, 
                                                    (0,255,0), 
                                                    1, 
                                                    lineType = cv2.LINE_AA)
                    if val_dict['ks_statistic'] > self.cfg['image_selection']['blur_thresh']:
                        cv2.imwrite(os.path.join(sharp_dir,file), img)
                    else:
                        cv2.imwrite(os.path.join(blur_dir,file), img)
        print("Finished Processing!")
        df = pd.DataFrame(np.column_stack([filepath, blur_val]), columns=['filepath', 'blur'])
        os.makedirs(os.path.join(self.cfg['image_selection']['out_path']),exist_ok=True)
        df.to_csv(os.path.join(self.cfg['image_selection']['out_path'],self.cfg['image_selection']['out_file']))

        return 0

if __name__ == '__main__': 
  
    blur_detect = BlurDetect()
    blur_detect.blur_detection()


