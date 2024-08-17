import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import glob
import cv2
import yaml
import json 
from utils import make_square_zero_padded_image
from inference import blur_analysis
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats



# read the config file
with open('config.yml', "r") as file:
    try:
        cfg = yaml.safe_load(file)
        print("Config file read successfully!")
    except yaml.YAMLError as exc:
        print(exc)




def csiq_analysis_no_reference()->int:
    '''
    Evaluate blurriness on CSIQ dataset
    '''

    DMOS = pd.read_excel(cfg['csiq']['dmos_file'], 'all_by_image')
    DMOS['image'] = DMOS['image'].astype(str)

    dst_imgs = {}
    if os.path.isdir(cfg['csiq']['dst_path']):
        for dir, subdir, files in os.walk(cfg['csiq']['dst_path']):
            for file in files:
                if cfg['csiq']['blur_only']:
                    if not 'blur' in dir:
                        continue
                if file.endswith('png'):
                    img = cv2.imread(os.path.join(dir,file))
                    dst_imgs[file] = img

    analysis_dict = {}
    for key,dst_frame in tqdm(dst_imgs.items()):
        if cfg['image_selection']['sliding_window']:
            resized_img = dst_frame.copy()
        else:
            resized_img = cv2.resize(dst_frame.copy(),(512,512))

        img_p=resized_img.copy()
        reblur = cv2.GaussianBlur(img_p, (0,0),1.3)
        # Convert to grayscale
        gray_img = cv2.cvtColor(img_p, cv2.COLOR_BGR2GRAY)
        gray_blur = cv2.cvtColor(reblur, cv2.COLOR_BGR2GRAY)
        val_dict = blur_analysis(dst_frame, resized_img, gray_img, gray_blur)
        analysis_dict[key] = val_dict
  
    print('Analysis json file saved at: ',cfg['csiq']['analysis_json'])
    with open(cfg['csiq']['analysis_json'], 'w') as file:
        json.dump(analysis_dict, file)

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
    print('Analysis data saved at: ',cfg['csiq']['analysis_csv'])
    csiq_df.to_csv(cfg['csiq']['analysis_csv'],index=False)

    csiq_df.dropna(inplace=True)

    SROCC = stats.spearmanr(csiq_df['ks_statistic'], csiq_df['dmos'])[0]
    KROCC = stats.kendalltau(csiq_df['ks_statistic'], csiq_df['dmos'])[0]
    PLCC = stats.pearsonr(csiq_df['ks_statistic'], csiq_df['dmos'])[0]
    RMSE = np.sqrt(((csiq_df['ks_statistic']-csiq_df['dmos']) ** 2).mean())
    print('SROCC: {} KROCC: {} PLCC: {} RMSE: {}'.format(SROCC, KROCC, PLCC, RMSE))

    csiq_df.rename({'dmos':'DMOS'},axis=1,inplace=True)

    sns.regplot(data=csiq_df, x='ks_statistic', y='DMOS', ci=None, scatter=True, marker='*',label='Images in CSIQ dataset',
                scatter_kws={'color':'b'},logistic=True, 
                line_kws={'color':"r", 'label':'curve fitted with logistic function'}) #order =2
    plt.legend()
    plt.savefig(os.path.join(cfg['csiq']['regplot'],'csiq_regfit.png'))

    return 0

csiq_analysis_no_reference()


live_data = 'LIVE_II_Dataset/LIVE_II.csv'
analysis_json = 'output/live_ii_analysis/liveii_analysis_no_reference_image_v2.json'
analysis_csv = 'output/live_ii_analysis/liveii_analysis_df_no_reference_image_v2.csv'

def liveii_analysis_with_reference_image() -> int: 
    '''
    Evaluate blurriness on Live_II dataset
    '''

    live_df = pd.read_csv(cfg['live']["live_data"])
    dst_df = live_df[live_df['orgs']==0]
    if cfg['live']['blur_only']:
        dst_df = dst_df[dst_df['distortions']=='gaussian_blur']
    
    analysis_dict = {}
    for _,dst_row in tqdm(dst_df.iterrows()):
        dst_frame = cv2.imread(dst_row['filepath'])
        if cfg['image_selection']['sliding_window']:
            if dst_frame.shape[0] % cfg['image_selection']['sl_win_size'] != 0:
                height_pad = (np.ceil(dst_frame.shape[0] / cfg['image_selection']['sl_win_size']) * cfg['image_selection']['sl_win_size'])
            else:
                height_pad = dst_frame.shape[0]
            if dst_frame.shape[1] % cfg['image_selection']['sl_win_size'] != 0:   
                width_pad = (np.ceil(dst_frame.shape[1] / cfg['image_selection']['sl_win_size']) * cfg['image_selection']['sl_win_size'])
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

        val_dict = blur_analysis(resized_img, gray_img, gray_blur, live=True, dst_row=dst_row)
        dst_key = dst_row['distortions'] + '.'+ dst_row['filepath'].split('/')[-1]
        analysis_dict[dst_key] = val_dict

    live_df = pd.DataFrame.from_dict(analysis_dict, orient='index')
    live_df['filename'] = live_df.index
    live_df.reset_index(drop=True, inplace=True)
    print('Analysis data saved at: ',cfg['live']['analysis_csv'])
    live_df.to_csv(cfg['live']['analysis_csv'],index=False)

    SROCC = stats.spearmanr(live_df['ks_statistic'], live_df['DMOS'])[0]
    KROCC = stats.kendalltau(live_df['ks_statistic'], live_df['DMOS'])[0]
    PLCC = stats.pearsonr(live_df['ks_statistic'], live_df['DMOS'])[0]
    RMSE = np.sqrt(((live_df['ks_statistic']-live_df['DMOS']) ** 2).mean())
    print('SROCC: {} KROCC: {} PLCC: {} RMSE: {}'.format(SROCC, KROCC, PLCC, RMSE))

    sns.regplot(data=live_df, x='ks_statistic', y='DMOS', ci=None, scatter=True, marker='*',label='Images in LIVE_II dataset',
            scatter_kws={'color':'b'},logistic=True, 
            line_kws={'color':"r", 'label':'curve fitted with logistic function'}) #order =2

    plt.legend()
    plt.savefig(cfg['live']['regplot'])
        
    return 0
        
liveii_analysis_with_reference_image()




