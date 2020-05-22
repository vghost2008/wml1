# coding: UTF-8
import ctypes
from ctypes import *
import numpy as np
import os

class TMAP(object):

    tmap_libs = None
    uImageThumbnail = 0
    uImageNavigate = 1
    uImageMacro = 2
    uImageLabel = 3
    uImageMacroLabel = 4
    uImageTile = 5
    uImageWhole = 6
    uImageAll = 7

    def __init__(self,path=None):
        self.path = None
        self.handle = None
        self._width = None
        self._height = None
        self._depth = None
        self.scan_scale = None
        if TMAP.tmap_libs is None:
            TMAP.load_lib()
        if path is None or not os.path.exists(path):
            print(f"Error path {path}.")
            raise ValueError(f"Error path {path}")
        self.__open(path)

    @staticmethod
    def load_lib():
        ll = ctypes.cdll.LoadLibrary
        #lib = ll("libiViewerSDK67.so")
        lib = ll("/usr/lib/libiViewerSDK67.so")
        if lib is not None:
            print(f'load {lib} success')
            TMAP.tmap_libs = lib
        else:
            print(f'load {lib} faild.')

    def __open(self,pstr):
        lib = self.tmap_libs
        lib.OpenTmapFile.restype = c_void_p
        p = c_char_p(bytes(pstr, "UTF-8"))
        h = lib.OpenTmapFile(p, len(pstr))

        if h == 0:
            print("open faild.")
            return False

        self.handle = c_void_p(h)
        print("open success")

        lib.GetScanScale.restype = c_int32
        img_size = self.__GetImageInfoEx(self.uImageWhole)
        self.scan_scale = lib.GetScanScale(self.handle);
        self._width = img_size.width
        self._height = img_size.height
        self._depth = img_size.depth

        return True

    def width(self,scale=-1):
        if scale<=0 or scale>=self.scan_scale:
            return self._width
        return int(self._width*scale/self.scan_scale)

    def height(self,scale=-1):
        if scale<=0 or scale>=self.scan_scale:
            return self._height
        return int(self._height*scale/self.scan_scale)

    def depth(self):
        return self._depth

    def get_focus_number(self):
        '''
        获取对焦层数
        :return:
        '''
        lib = self.tmap_libs

        lib.GetFocusNumber.restype = c_int32
        res = lib.GetFocusNumber(self.handle)

        return res

    def set_focus_layer(self,layer):
        lib = self.tmap_libs
        res = lib.SetFocusLayer(self.handle,layer)
        if not res:
            print(f"Set focus layer {layer} faild.")
        else:
            print(f"Set focus layer {layer} success.")

        return res

    def __GetImageInfoEx(self,etype):
        lib = self.tmap_libs
        class ImgSize(Structure):
            _fields_=[('imgsize',c_longlong),
                      ('width',c_int),
                      ('height',c_int),
                      ('depth',c_int)
                     ]
        GetImageInfoEx_fun = lib.GetImageInfoEx
        GetImageInfoEx_fun.restype = ImgSize
        fun_ImgSize = GetImageInfoEx_fun(self.handle,c_int32(etype))
        return fun_ImgSize

    def get_label_img(self):
        img = self.get_image_data(TMAP.uImageLabel)
        if img is None:
            img = self.get_macro_label_img()
            if img is None:
                return img
            size = min(img.shape[0],img.shape[1])
            return img[:size,:size,:]
        return img

    def get_macro_label_img(self):
        return self.get_image_data(TMAP.uImageMacroLabel)

    def get_image(self,etype):
        fun_ImgSize = self.__GetImageInfoEx(etype)
        lib = self.tmap_libs

        nBufferLength = fun_ImgSize.width * fun_ImgSize.height * fun_ImgSize.depth / 8
        nBufferLength = int(nBufferLength)

        lib.GetImageDataEx.restype = POINTER(c_ubyte)
        buffer = lib.GetImageDataEx(self.handle,c_int32(etype),c_int32(nBufferLength))

        W=fun_ImgSize.width
        H=fun_ImgSize.height
        pic = TMAP.buffer_to_ndarray(buffer,H*W*3)
        if pic is None:
            return None

        pic = pic.reshape((H,W,3))
        pic = np.asarray(pic[:,:,::-1], dtype=np.uint8)

        return pic

    def crop_img_in_all_focus(self,left,top,width,height,scale=-1):
        nr = self.get_focus_number()
        if 1 == nr:
            yield self.crop_img(left,top,width,height,scale)
        else:
            for i in range(-nr,nr+1):
                self.set_focus_layer(i)
                yield self.crop_img(left, top, width, height, scale)

    def crop_img(self,left=0,top=0,width=-1,height=-1,scale=-1,layer=None):
        '''
        :param left: scale下的坐标
        :param top:
        :param width:
        :param height:
        :param scale:
        :return:
        '''

        lib = self.tmap_libs

        if scale>self.scan_scale:
            print(f"Error scale {scale}, scan scale {self.scan_scale}")
            return None
        elif scale<=0:
            scale = self.scan_scale
        if width<=0:
            width = self.width()
        if height<=0:
            height = self.height()

        if layer is not None:
            self.set_focus_layer(layer)

        GetCropImageDataEx_fun = lib.GetCropImageDataEx
        GetCropImageDataEx_fun.restype = POINTER(c_ubyte)

        nLeft = left*scale/self.scan_scale
        nRight = (left+width)*scale/self.scan_scale
        nTop = top*scale/self.scan_scale
        nBottom = (top+height)*scale/self.scan_scale
        nLeft = c_int32(int(max(0,nLeft)))
        nTop = c_int32(int(max(0,nTop)))
        nBottom = c_int32(int(min(self.height()-1,nBottom)))
        nRight = c_int32(int(min(self.width()-1,nRight)))
        buffer_size = c_int32(int(width*height*3))
        scale = c_float(scale)

        f = GetCropImageDataEx_fun(self.handle,1, nLeft, nTop, nRight, nBottom, scale, buffer_size)

        pic = TMAP.buffer_to_ndarray(f,width*height*3)
        pic = pic.reshape((height,width,3))
        pic = np.asarray(pic[:,:,::-1], dtype=np.uint8)

        return pic

    def get_all_img_crops(self,width,height,scale=-1):

        for y in range(0,self.height(scale),height):
            for x in range(0, self.width(scale), width):
                yield self.crop_img(x,y,width,height,scale)

    def get_all_img_crops_in_all_focus(self, width, height, scale=-1):

        for y in range(0, self.height(scale), height):
            for x in range(0, self.width(scale), width):
                for img in self.crop_img_in_all_focus(x, y, width, height, scale):
                    yield img

    @staticmethod
    def buffer_to_ndarray(buffer,length,buffer_length=None,type=np.uint8):
        addr = addressof(buffer.contents)
        if addr == 0:
            return None
        if buffer_length is None:
            buffer_length = length
        ArrayTypeLen = c_uint8 * buffer_length
        array = np.frombuffer(ArrayTypeLen.from_address(addr), type, length)
        return array

    def get_focus_prefix(self):
        nr = self.get_focus_number()
        res = []
        for i in range(-nr,nr+1):
            if i<0:
                res.append(f"N{-i}")
            else:
                res.append(f"P{i}")
        return res

    def __del__(self):
        lib = self.tmap_libs
        if self.handle is None:
            return
        lib.CloseTmapFile(self.handle)
        print("close tmap file.")
