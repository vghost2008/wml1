import os
import os.path as osp
import pickle
import xml.etree.ElementTree as ET
from loguru import logger
import glob
import cv2
import numpy as np
import wml_utils as wmlu
import random
from .datasets_wrapper import Dataset
import copy
import sys
from object_detection2.data.transforms.transform_toolkit import motion_blur

class _AnnotationTransform(object):

    """Transforms a VOC annotation into a Tensor of bbox coords and label index
    Initilized with a dictionary lookup of classnames to indexes

    Arguments:
        class_to_ind (dict, optional): dictionary lookup of classnames -> indexes
            (default: alphabetic indexing of VOC's 20 classes)
        keep_difficult (bool, optional): keep difficult instances or not
            (default: False)
        height (int): height
        width (int): width
    """

    def __init__(self, class_to_ind, keep_difficult=True):
        self.class_to_ind = class_to_ind 
        self.keep_difficult = keep_difficult

    def __call__(self, target):
        """
        Arguments:
            target (annotation) : the target annotation to be made usable
                will be an ET.Element
        Returns:
            a list containing lists of bounding boxes  [bbox coords, class name][xmin, ymin, xmax, ymax, label_ind]
        """
        res = np.empty((0, 5))
        for obj in target.iter("object"):
            difficult = obj.find("difficult")
            if difficult is not None:
                difficult = int(difficult.text) == 1
            else:
                difficult = False
            if not self.keep_difficult and difficult:
                continue
            name = obj.find("name").text.strip()
            bbox = obj.find("bndbox")

            pts = ["xmin", "ymin", "xmax", "ymax"]
            bndbox = []
            for i, pt in enumerate(pts):
                cur_pt = int(bbox.find(pt).text) - 1
                # scale height or width
                # cur_pt = cur_pt / width if i % 2 == 0 else cur_pt / height
                bndbox.append(cur_pt)
            label_idx = self.class_to_ind[name]
            bndbox.append(label_idx)
            res = np.vstack((res, bndbox))  # [xmin, ymin, xmax, ymax, label_ind]
            # img_id = target.find('filename').text[:-4]

        width = int(target.find("size").find("width").text)
        height = int(target.find("size").find("height").text)
        img_info = (height, width)
        res = np.maximum(res,0)

        return res, img_info

class XmlBaseDataset(Dataset):

    def __init__(
        self,
        class_to_ind,
        classes,
        img_size=(416, 416),
        preproc=None,
        target_transform=None,
        dataset_name="BaseXmlDataset",
        cache_dir=None,
    ):
        super().__init__(img_size)
        self.img_size = img_size
        self.preproc = preproc

        if target_transform is not None:
            self.target_transform = target_transform
        else:
            self.target_transform = _AnnotationTransform(class_to_ind)

        self.name = dataset_name
        if cache_dir is not None:
            self.cache_dir = cache_dir
        else:
            self.cache_dir = osp.join("/dl_cache",dataset_name)

        self._classes = classes
        self.class_ids = classes
        self.always_make_generate_cache_files = False
        self.cache_imgs = True

    def __len__(self):
        #return 100 #debug
        return len(self.img_files)

    def _load_coco_annotations(self):
        print(f"Loading annotations.")
        sys.stdout.flush()
        res = [self.load_anno_from_ids(_ids) for _ids in range(len(self.img_files))]
        print(f"Loading annotations finish.")
        sys.stdout.flush()
        return res

    def do_cache(self,cache_file,cache_xml_file):
        self.annotations = self._load_coco_annotations()

        if self.cache_imgs:
            max_h = self.img_size[0]
            max_w = self.img_size[1]
            logger.info(
                "Caching images for the frist time. This might take about 3 minutes for VOC"
            )
            self.imgs = np.memmap(
                cache_file,
                shape=(len(self.img_files), max_h, max_w, 3),
                dtype=np.uint8,
                mode="w+",
            )
            from tqdm import tqdm
            from multiprocessing.pool import ThreadPool
    
            NUM_THREADs = min(8, os.cpu_count())
            loaded_images = ThreadPool(NUM_THREADs).imap(
                lambda x: self.load_resized_img(x),
                range(len(self.annotations)),
            )
            pbar = tqdm(enumerate(loaded_images), total=len(self.annotations))
            for k, out in pbar:
                self.imgs[k][: out.shape[0], : out.shape[1], :] = out.copy()
    
            '''pbar = tqdm(range(len(self.annotations)), total=len(self.annotations))
            for i in pbar:
                out =  self.load_resized_img(i)
                self.imgs[i][: out.shape[0], : out.shape[1], :] = out.copy()
                del out'''
    
            self.imgs.flush()
            pbar.close()

        with open(cache_xml_file,"wb") as f:
            pickle.dump(self.xml_files,f)


        cache_xml_data_file = wmlu.change_suffix(cache_xml_file,"pth")
        with open(cache_xml_data_file,"wb") as f:
            pickle.dump(self.annotations,f)


    def _cache_images(self):
        logger.warning(
            "\n********************************************************************************\n"
            "You are using cached images in RAM to accelerate training.\n"
            "This requires large system RAM.\n"
            "Make sure you have 60G+ RAM and 19G available disk space for training VOC.\n"
            "********************************************************************************\n"
        )
        max_h = self.img_size[0]
        max_w = self.img_size[1]
        cache_dir = self.cache_dir
        wmlu.create_empty_dir(cache_dir,remove_if_exists=False)
        cache_file = osp.join(cache_dir, "img_resized_cache_" + self.name + ".array")
        cache_xml_file= osp.join(cache_dir,"img_resized_cache_" + self.name + "_xml.array")
        cache_xml_data_file = wmlu.change_suffix(cache_xml_file,"pth")

        if self.always_make_generate_cache_files:
            print(f"Force update cached images.")
            if os.path.exists(cache_file):
                os.remove(cache_file)
            if os.path.exists(cache_xml_file):
                 os.remove(cache_xml_file)

        if ((not os.path.exists(cache_file)) and self.cache_imgs) \
             or not os.path.exists(cache_xml_file):
            self.do_cache(cache_file,cache_xml_file)
        else:
            with open(cache_xml_file,"rb") as f:
                xml_files = pickle.load(f)
            if not self.same_files(xml_files,self.xml_files):
                if os.path.exists(cache_file):
                    os.remove(cache_file)
                if os.path.exists(cache_xml_file):
                    os.remove(cache_xml_file)
                print(f"Remove cached data.")
                sys.stdout.flush()
                self.do_cache(cache_file,cache_xml_file)
            else:    
                self.xml_files = xml_files
                if osp.exists(cache_xml_data_file):
                    with open(cache_xml_data_file,"rb") as f:
                        self.annotations = pickle.load(f)
                else:
                    self.annotations = self._load_coco_annotations()
                    with open(cache_xml_data_file,"wb") as f:
                        pickle.dump(self.annotations,f)
                    if len(self.annotations) != len(self.xml_files):
                        print(f"Remove cached data.")
                        self.do_cache(cache_file,cache_xml_file)
                print("WARNING: You are using cached imgs! Make sure your dataset is not changed!!")

        if self.cache_imgs:
            logger.info("Loading cached imgs...")
            self.imgs = np.memmap(
                cache_file,
                shape=(len(self.img_files), max_h, max_w, 3),
                dtype=np.uint8,
                mode="r+",
            )
        else:
            self.imgs = None
            self.img_files = [wmlu.change_suffix(x,"jpg") for x in self.xml_files]
        sys.stdout.flush()
    
    @staticmethod
    def same_files(filesa,filesb):
        if len(filesa) != len(filesb):
            return False
        new_files_a = copy.deepcopy(filesa)
        random.shuffle(new_files_a)
        new_files_a = new_files_a[:1000]
        for x in new_files_a:
            if x not in filesb:
                return False
        return True

    def load_anno_from_ids(self, index):
        annopath = self.xml_files[index]
        try:
            with open(annopath,"r") as f:
                target = ET.parse(f).getroot()
    
            assert self.target_transform is not None
            res, img_info = self.target_transform(target)
            height, width = img_info
    
            if height < 5 or width<5:
                print(f"Force update width height")
                imgpath = self.img_files[index]
                height,width,_ = cv2.imread(imgpath).shape
                img_info = (height,width)
    
            r = min(self.img_size[0] / height, self.img_size[1] / width)
            res[:, :4] *= r
            resized_info = (int(height * r), int(width * r))
    
            return (res, img_info, resized_info)
        except Exception as e:
            print(f"Read {annopath} faild: {e}.")
            raise e

    def load_anno(self, index):
        return self.annotations[index][0]

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
        if img is None or img.shape[0]<10 or img.shape[1]<10:
            print(f"ERROR: error img {img}, shape {img.shape if img is not None else 0}")

        return img

    def pull_item(self, index):
        """Returns the original image and target at an index for mixup

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to show
        Return:
            img, target
        """
        if self.imgs is not None:
            target, img_info, resized_info = self.annotations[index]
            pad_img = self.imgs[index]
            img = pad_img[: resized_info[0], : resized_info[1], :].copy()
        else:
            img = self.load_resized_img(index)
            target, img_info, _ = self.annotations[index]
            target = target.copy()
        '''if np.random.rand()<0.2 or True:
            degree = random.randint(2,5)
            angle = random.randint(-180,180)
            img = motion_blur(img,degree=degree,angle=angle)'''

        return img, target, img_info, index

    @Dataset.mosaic_getitem
    def __getitem__(self, index):
        img, target, img_info, img_id = self.pull_item(index)

        if self.preproc is not None:
            img, target = self.preproc(img, target, self.input_dim)
        #target: [N,5] (xmin,ymin,xmax,ymax,label) in resized size
        return img, target, img_info, img_id
