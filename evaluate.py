import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import cv2
import yaml
from utils import make_square_zero_padded_image
from inference import BlurDetect
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


class Evaluate():
    ''' 
    Evaluate on public datasets
    '''

    def __init__(self) -> None:
            # read the config file
            with open('config.yml', "r") as file:
                try:
                    self.cfg = yaml.safe_load(file)
                    print("Config file read successfully!")
                except yaml.YAMLError as exc:
                    print(exc)


    def NormalizeData(self, data):
        return (data - np.min(data)) / (np.max(data) - np.min(data))

    def csiq_analysis_no_reference(self)->int:
        '''
        Evaluate blurriness on CSIQ dataset
        '''

        DMOS = pd.read_excel(self.cfg['csiq']['dmos_file'], 'all_by_image')
        DMOS['image'] = DMOS['image'].astype(str)
        b_d = BlurDetect()
        dst_imgs = {}
        if os.path.isdir(self.cfg['csiq']['dst_path']):
            for dir, subdir, files in os.walk(self.cfg['csiq']['dst_path']):
                if self.cfg['csiq']['blur_only']:
                        if 'blur' != dir.split('/')[-1]:
                            continue
                for file in files:
                    if file.endswith('png'):
                        img = cv2.imread(os.path.join(dir,file))
                        dst_imgs[file] = img

        analysis_dict = {}
        for key,dst_frame in tqdm(dst_imgs.items()):
            if self.cfg['image_selection']['sliding_window']:
                resized_img = dst_frame.copy()
            else:
                resized_img = cv2.resize(dst_frame.copy(),(512,512))

            img_p=resized_img.copy()
            reblur = cv2.GaussianBlur(img_p, (0,0),1.3)
            # Convert to grayscale
            gray_img = cv2.cvtColor(img_p, cv2.COLOR_BGR2GRAY)
            gray_blur = cv2.cvtColor(reblur, cv2.COLOR_BGR2GRAY)
            val_dict = b_d.blur_analysis(resized_img, gray_img, gray_blur)
            analysis_dict[key] = val_dict
    

        an_df = pd.DataFrame.from_dict(analysis_dict, orient='index')
        an_df['filename'] = an_df.index
        an_df['image'] = an_df['filename'].str.split('.').str[0]
        an_df['dst_type'] = an_df['filename'].str.split('.').str[1]
        an_df['dst_lev'] = an_df['filename'].str.split('.').str[2].astype('int64')
        an_df.drop(['filename'], axis=1, inplace=True)
        an_df.reset_index(drop=True, inplace=True)
        an_df['dst_type']= an_df['dst_type'].replace({'jpeg2000': 'jpeg 2000',
                                                    'BLUR': 'blur',
                                                    'JPEG': 'jpeg',
                                                    'AWGN': 'noise' })
        
        csiq_df = pd.merge(an_df, DMOS, on=['image', 'dst_lev', 'dst_type'], how='inner')
        csiq_df.sort_values(by=['dst_idx', 'image'], inplace=True)
        csiq_df.reset_index(drop=True, inplace=True)
        csiq_df.dropna(inplace=True)

        SROCC = stats.spearmanr(csiq_df['ks_statistic'], csiq_df['dmos'])[0]
        KROCC = stats.kendalltau(csiq_df['ks_statistic'], csiq_df['dmos'])[0]
        PLCC = stats.pearsonr(csiq_df['ks_statistic'], csiq_df['dmos'])[0]
        RMSE = np.sqrt(((csiq_df['ks_statistic']-csiq_df['dmos']) ** 2).mean())
        print('SROCC: {} KROCC: {} PLCC: {} RMSE: {}'.format(SROCC, KROCC, PLCC, RMSE))
        # normalize ks_statistics values
        ks_scaled = self.NormalizeData(csiq_df['ks_statistic'].values)
        csiq_df['ks_statistic'] = ks_scaled
        os.makedirs(self.cfg['csiq']['outpath'],exist_ok=True)
        print('Analysis data saved at: ',self.cfg['csiq']['analysis_csv'])
        csiq_df.to_csv(self.cfg['csiq']['analysis_csv'],index=False)

        csiq_df.rename({'dmos':'DMOS'},axis=1,inplace=True)

        sns.regplot(data=csiq_df, x='ks_statistic', y='DMOS', ci=None, scatter=True, marker='*',label='Images in CSIQ dataset',
                    scatter_kws={'color':'b'},logistic=True, 
                    line_kws={'color':"r", 'label':'curve fitted with logistic function'}) 
        plt.legend()
        plt.savefig(self.cfg['csiq']['regplot'])
        plt.clf()

        return 0



    def liveii_analysis_no_reference(self) -> int: 
        '''
        Evaluate blurriness on Live_II dataset
        '''

        live_df = pd.read_csv(self.cfg['live']["live_data"])
        dst_df = live_df[live_df['orgs']==0]
        if self.cfg['live']['blur_only']:
            dst_df = dst_df[dst_df['distortions']=='gaussian_blur']
        b_d = BlurDetect()
        analysis_dict = {}
        for _,dst_row in tqdm(dst_df.iterrows()):
            dst_frame = cv2.imread(dst_row['filepath'])
            if self.cfg['image_selection']['sliding_window']:
                if dst_frame.shape[0] % self.cfg['image_selection']['sl_win_size'] != 0:
                    height_pad = (np.ceil(dst_frame.shape[0] / self.cfg['image_selection']['sl_win_size']) * self.cfg['image_selection']['sl_win_size'])
                else:
                    height_pad = dst_frame.shape[0]
                if dst_frame.shape[1] % self.cfg['image_selection']['sl_win_size'] != 0:   
                    width_pad = (np.ceil(dst_frame.shape[1] / self.cfg['image_selection']['sl_win_size']) * self.cfg['image_selection']['sl_win_size'])
                else:
                    width_pad = dst_frame.shape[1]
                if height_pad > dst_frame.shape[0] or width_pad > dst_frame.shape[1]:
                    resized_img = make_square_zero_padded_image(dst_frame.copy(), int(height_pad), int(width_pad))
                else:
                    resized_img = dst_frame.copy()
            else:
                resized_img = cv2.resize(dst_frame.copy(),(512,512))
            
            img_p = resized_img.copy()
            reblur = cv2.GaussianBlur(img_p, (0,0),1.3)
            # Convert to grayscale
            gray_img = cv2.cvtColor(img_p, cv2.COLOR_BGR2GRAY)
            gray_blur = cv2.cvtColor(reblur, cv2.COLOR_BGR2GRAY)

            val_dict = b_d.blur_analysis(resized_img, gray_img, gray_blur, live=True, dst_row=dst_row)
            dst_key = dst_row['distortions'] + '.'+ dst_row['filepath'].split('/')[-1]
            analysis_dict[dst_key] = val_dict

        live_df = pd.DataFrame.from_dict(analysis_dict, orient='index')
        live_df['filename'] = live_df.index
        live_df.reset_index(drop=True, inplace=True)

        #data normalization
        x_scaled = self.NormalizeData(live_df['DMOS'].values)
        live_df['DMOS'] = x_scaled
        ks_scaled = self.NormalizeData(live_df['ks_statistic'].values)
        live_df['ks_statistic'] = ks_scaled

        SROCC = stats.spearmanr(live_df['ks_statistic'], live_df['DMOS'])[0]
        KROCC = stats.kendalltau(live_df['ks_statistic'], live_df['DMOS'])[0]
        PLCC = stats.pearsonr(live_df['ks_statistic'], live_df['DMOS'])[0]
        RMSE = np.sqrt(((live_df['ks_statistic']-live_df['DMOS']) ** 2).mean())
        print('SROCC: {} KROCC: {} PLCC: {} RMSE: {}'.format(SROCC, KROCC, PLCC, RMSE))

        os.makedirs(self.cfg['live']['outpath'],exist_ok=True)
        print('Analysis data saved at: ',self.cfg['live']['analysis_csv'])
        live_df.to_csv(self.cfg['live']['analysis_csv'],index=False)

        sns.regplot(data=live_df, x='ks_statistic', y='DMOS', ci=None, scatter=True, marker='*',label='Images in LIVE_II dataset',
                scatter_kws={'color':'b'},logistic=True, 
                line_kws={'color':"r", 'label':'curve fitted with logistic function'}) 

        plt.legend()
        plt.savefig(self.cfg['live']['regplot'])
        plt.clf()
            
        return 0
            


if __name__ == '__main__': 
  
    evaluate = Evaluate()
    evaluate.csiq_analysis_no_reference()
    evaluate.liveii_analysis_no_reference()

