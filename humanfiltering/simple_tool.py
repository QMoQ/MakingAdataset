import os
import matplotlib.pyplot as plt
import glob
import pickle
import sys
import numpy as np
import cv2
import time
import pyautogui
from tqdm.auto import tqdm

class ImageViewer:
'''
    folder: image folder
    pklpath: previous filter information 
    restart: start from no.1 image or not (default start from the image in the lastest turn)
    rewrite_pth: savefile for filter information in this turn
    imshow_type: only imshow specific type (for example, '-1' for invalid, '0' for deleted, '1' for saved, imshow_type=1 means only show saved ones)
'''
    def __init__(self, folder, pklpath=None, restart=False, rewrite_pth=None, imshow_type=None):
        self.folder = folder
        self.note_dict = ['remove', 'save', 'invalid']
        self.ind = 0
        self.rewrite_pth = rewrite_pth
        self.pklpath = pklpath 
        print(f'interaction: \n \tx-save&quit\n\ts-save\n\t<-zoom move left\n\t>-zoom-move right\n\ty-yes\n\tn-no')
        self.filterinfo =  {}
        if os.path.exists(self.rewrite_pth):
            with open(self.rewrite_pth, 'rb') as pp:
                self.filterinfo = pickle.load(pp)
            if 'images' not in self.filterinfo:
                self.image_paths = glob.glob(os.path.join(self.folder, "*")) #glob.glob(os.path.join(self.folder, "*.jpg")) + glob.glob(os.path.join(self.folder, "*.png"))
                self.filterinfo['images'] = self.image_paths
            else:
                self.image_paths = self.filterinfo['images']
            self.names = [os.path.basename(img) for img in self.image_paths]
            if 'dict' not in self.filterinfo:
                self.filterinfo['dict'] = {}
                self.make_dict()
            else:
                self.pps_dict = self.filterinfo['dict']['pps']
                self.kvq_dict = self.filterinfo['dict']['kvq']
        else:
            self.image_paths = glob.glob(os.path.join(self.folder, "*")) #glob.glob(os.path.join(self.folder, "*.jpg")) + glob.glob(os.path.join(self.folder, "*.png"))
            self.filterinfo['images'] = self.image_paths
            self.filterinfo['dict'] = {}
            self.names = [os.path.basename(img) for img in self.image_paths]
            self.make_dict()
        

        self.picsnum = len(self.image_paths)

        if 'cost_time(s)' not in self.filterinfo:
            self.time = 0.
        else:
            self.time = self.filterinfo['cost_time(s)']

        if 'human_filtered' not in self.filterinfo or restart:
            self.human_filtered = np.array([-1 for i in range(self.picsnum)])
        else:
            self.human_filtered = np.array(self.filterinfo['human_filtered'])
            # self.ind = np.argwhere(self.human_filtered == 2)[0][0]

        if imshow_type is None:
            self.indexes = np.array(range(self.picsnum))
            # self.ind = np.argwhere(np.array(self.names) == 'poco_21629252_313394508.jpg')[0][0]
            self.ind = np.argwhere(self.human_filtered == -1)[0][0]
        else:
            self.indexes = np.argwhere(self.human_filtered == imshow_type)
            self.indexes = [ind[0] for ind in self.indexes]
            self.ind = 0
        print('ready to filter~')

            
    def make_dict(self):
        self.pps_dict = {}
        self.kvq_dict = {}
        if self.pklpath is not None:
            with open(self.pklpath, 'rb') as pp:
                self.pkl = pickle.load(pp)
            print('geting kvq/pps diction...')
            for info in tqdm(self.pkl):
                p = info[0]
                name = os.path.basename(p)
                if name in self.names:
                    pps = info[4]
                    kvq = info[5]
                    self.pps_dict[name] = pps
                    self.kvq_dict[name] = kvq
        else:
            for name in self.names:
                self.pps_dict[name] = ''
                self.kvq_dict[name] = ''
        self.filterinfo['dict']['pps'] = self.pps_dict
        self.filterinfo['dict']['kvq'] = self.kvq_dict

    def init_show_pics(self):
        # 获取屏幕宽度和高度
        self.win_w, self.win_h = pyautogui.size()
        self.crop_size = 200 #crop窗口的大小
        self.zoom_scale = 3  # 放大的倍数
        self.zoom_window_size = (self.crop_size * self.zoom_scale, self.crop_size * self.zoom_scale)  # 放大窗口的大小
        self.zoom_window_position = (self.win_w - self.zoom_window_size[0], 0)
        self.zoom_window_size = (self.crop_size * self.zoom_scale, self.crop_size * self.zoom_scale)  # 放大窗口的大小
        print(f'crop size: {self.crop_size} x {self.crop_size}')
  

    def deal_with_pics(self):
        self.t1 = time.time()
        cycle = True
        # 加载显示新图像
        img = cv2.imread(self.image_paths[self.indexes[self.ind]])
        self.cur_name = os.path.basename(self.image_paths[self.indexes[self.ind]]).split('.')[0]
        self.name = os.path.basename(self.image_paths[self.indexes[self.ind]])
        h, w = img.shape[:2]
        try:
            title = f"No.{self.indexes[self.ind]+1}/{self.picsnum} [{self.cur_name}]-[{h}x{w}] [{self.kvq_dict[self.name]:.02f}, {self.pps_dict[self.name]:.02f}] --> {self.note_dict[self.human_filtered[self.indexes[self.ind]]]} "
        except:
            title = f"No.{self.indexes[self.ind]+1}/{self.picsnum} [{self.cur_name}]-[{h}x{w}]  --> {self.note_dict[self.human_filtered[self.indexes[self.ind]]]} "
        cv2.namedWindow(title, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(title, self.win_w * 3 // 4, self.win_h * 3 // 4)
        cv2.imshow(title, img)
        cv2.setMouseCallback(title, self.mouse_callback, param=(img, title))
        key = cv2.waitKey(1)
        while cycle:
            if key == ord('y'):
                self.human_filtered[self.indexes[self.ind]] = 1
                try:
                    title = f"No.{self.indexes[self.ind]+1}/{self.picsnum} [{self.cur_name}]-[{h}x{w}] [{self.kvq_dict[self.name]:.02f}, {self.pps_dict[self.name]:.02f}] --> {self.note_dict[self.human_filtered[self.indexes[self.ind]]]} "
                except:
                    title = f"No.{self.indexes[self.ind]+1}/{self.picsnum} [{self.cur_name}]-[{h}x{w}]  --> {self.note_dict[self.human_filtered[self.indexes[self.ind]]]} "
                cv2.namedWindow(title, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(title, self.win_w * 3 // 4, self.win_h * 3 // 4)
                cv2.imshow(title, img)
                self.ind = (self.ind + 1) % len(self.indexes)
                cv2.waitKey(100)
            elif key == ord('n'):
                self.human_filtered[self.indexes[self.ind]] = 0
                try:
                    title = f"No.{self.indexes[self.ind]+1}/{self.picsnum} [{self.cur_name}]-[{h}x{w}] [{self.kvq_dict[self.name]:.02f}, {self.pps_dict[self.name]:.02f}] --> {self.note_dict[self.human_filtered[self.indexes[self.ind]]]} "
                except:
                    title = f"No.{self.indexes[self.ind]+1}/{self.picsnum} [{self.cur_name}]-[{h}x{w}]  --> {self.note_dict[self.human_filtered[self.indexes[self.ind]]]} "
                cv2.namedWindow(title, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(title, self.win_w * 3 // 4, self.win_h * 3 // 4)
                cv2.imshow(title, img)
                self.ind = (self.ind + 1) % len(self.indexes)
                cv2.waitKey(100)
            elif key == ord('a'):
                self.ind = (self.ind - 1) % len(self.indexes)
            elif key == ord('d'):
                self.ind = (self.ind + 1) % len(self.indexes)
            elif key == ord('s'):
                self.write_attrs()
            elif key == ord(','):
                # 记录缩放窗口的位置
                # self.zoom_window_position = cv2.getWindowImageRect('Zoomed View')[:2]
                self.zoom_window_position = (self.zoom_window_position[0] - self.zoom_window_size[0]//2, 0)
            elif key == ord('.'):
                self.zoom_window_position = (self.zoom_window_position[0] + self.zoom_window_size[0]//2, 0)
            elif key == ord('x'):
                self.write_attrs()
                plt.close()
                self.t2 = time.time()
                print(f"退出，本次消耗时间:{self.t2-self.t1} s，总消耗时间为:{self.filterinfo['cost_time(s)'] // 600 } min")
                sys.exit(0)
            else:
                key = cv2.waitKey(1)
                continue
            
            if self.ind == len(self.indexes):
                plt.title("This is the last image. Thank U for your effort~")
                self.write_attrs()
                cv2.waitKey(1000)
                plt.close()
                break

            if self.ind % 10 == 0:
                self.write_attrs()

            # 加载并显示新图像
            cv2.destroyAllWindows()
            img = cv2.imread(self.image_paths[self.indexes[self.ind]])
            self.cur_name = os.path.basename(self.image_paths[self.indexes[self.ind]]).split('.')[0]
            h, w = img.shape[:2]
            self.name = os.path.basename(self.image_paths[self.indexes[self.ind]])
            try:
                title = f"No.{self.indexes[self.ind]+1}/{self.picsnum} [{self.cur_name}]-[{h}x{w}] [{self.kvq_dict[self.name]:.02f}, {self.pps_dict[self.name]:.02f}] --> {self.note_dict[self.human_filtered[self.indexes[self.ind]]]} "
            except:
                title = f"No.{self.indexes[self.ind]+1}/{self.picsnum} [{self.cur_name}]-[{h}x{w}]  --> {self.note_dict[self.human_filtered[self.indexes[self.ind]]]} "
            cv2.namedWindow(title, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(title, self.win_w * 3 // 4, self.win_h * 3 // 4)
            cv2.imshow(title, img)
            cv2.setMouseCallback(title, self.mouse_callback, param=(img, title))
            key = cv2.waitKey(1)

    def mouse_callback(self, event, x, y, flags, param):
        img, title = param
        if event == cv2.EVENT_MOUSEMOVE:
            h, w = img.shape[:2]
            zoom_img = cv2.resize(img[max(y - self.crop_size, 0):min(y + self.crop_size, h), max(x - self.crop_size, 0):min(x + self.crop_size, w)], self.zoom_window_size, interpolation=cv2.INTER_LANCZOS4)
            cv2.imshow(f'Zoomed View | {self.names[self.indexes[self.ind]]}', zoom_img)
            cv2.moveWindow(f'Zoomed View | {self.names[self.indexes[self.ind]]}', *self.zoom_window_position)  # 使用上次的位置
        


    def write_attrs(self):
        self.t2 = time.time()
        self.human_filtered = [int(xx) for xx in self.human_filtered]
        self.filterinfo['human_filtered'] = self.human_filtered
        if 'cost_time(s)' in self.filterinfo:
            self.filterinfo['cost_time(s)'] += self.t2 - self.t1
        else:
            self.filterinfo['cost_time(s)'] = self.t2 - self.t1
        with open(self.rewrite_pth, 'wb') as at:
            pickle.dump(self.filterinfo, at)
        print(f'筛选记录写入： {self.rewrite_pth}')
        self.human_filtered = np.array(self.human_filtered)
        already_done = sum(self.human_filtered != -1)
        already_save = sum(self.human_filtered == 1)
        print(f"目前累计留存率： {already_save / 1. / already_done}")

if __name__ == '__main__':
    viewer = ImageViewer('./images', None, rewrite_pth='./humanfiltered_info.pkl', imshow_type=None)
    viewer.init_show_pics()
    viewer.deal_with_pics()
    viewer.write_attrs()