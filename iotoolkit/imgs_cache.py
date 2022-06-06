import numpy as np
import cv2

class ImgsCache(object):
    def __init__(self,files,img_size=None,cache_limit=-1,mem_limit=-1):
        '''
        Args:
            files: list of img files path.
            img_size: (H,W)
            cache_limit: cache number limit
            mem_limit: meme limit (G)
        '''
        self.img_files = files
        self.img_size = img_size
        if cache_limit<=1 and mem_limit>0 and img_size is not None:
            per_img_size = img_size[0]*img_size[1]*3
            cache_limit = mem_limit*1e6/per_img_size
        self.cache_limit = cache_limit
        self.cache_data = {}
        print(f"Cache limit is {self.cache_limit}.")

    def __getitem__(self, item):
        assert isinstance(item,int), f"Error item type: {item}"
        if item in self.cache_data:
            return self.cache_data[item].copy()
        else:
            img = self.load_image_data(item)
            if self.cache_limit<=1 \
                    or (self.cache_limit>1 and len(self.cache_data)<self.cache_limit):
                self.cache_data[item] = img.copy()
            return img

    def load_image_data(self,idx):
        if self.img_size is not None:
            return self.load_resized_img(idx)
        else:
            return self.load_image(idx)

    def load_resized_img(self, index):
        img = self.load_image(index)
        r = min(self.img_size[0] / img.shape[0], self.img_size[1] / img.shape[1])
        resized_img = cv2.resize(
            img,
            (int(img.shape[1] * r), int(img.shape[0] * r)),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.uint8)

        return resized_img

    def load_image(self, index):
        img_file = self.img_files[index]
        img = cv2.imread(img_file, cv2.IMREAD_COLOR)
        if img is None or img.shape[0]<2 or img.shape[1]<2:
            print(f"ERROR: error img {img}, shape {img.shape if img is not None else 0}, file_path {self.img_files[index]}")
            raise Exception("Empty img.")

        return img

