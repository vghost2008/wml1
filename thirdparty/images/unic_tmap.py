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
        self.path = path
        self.handle = None
        self._width = None
        self._height = None
        self._depth = None
        self.scan_scale = None
        if TMAP.tmap_libs is None:
            TMAP.load_lib()
        if path is None or not os.path.exists(path):
            print(f"Error path {path}.")
            self.is_open = False
        else:
            self.is_open = self.__open()

    @staticmethod
    def load_lib():
        ll = ctypes.cdll.LoadLibrary
        #lib = ll("libiViewerSDK67.so")
        lib_path = "/usr/lib/libiViewerSDK67.so"
        if not os.path.exists(lib_path):
            lib_path = "libiViewerSDK67.so"
        lib = ll(lib_path)
        if lib is not None:
            print(f'load {lib} success')
            TMAP.tmap_libs = lib
        else:
            print(f'load {lib} faild.')

    def __open(self):
        try:
            lib = self.tmap_libs
            lib.OpenTmapFile.restype = c_void_p
            p = c_char_p(bytes(self.path, "UTF-8"))
            h = lib.OpenTmapFile(p, len(self.path))

            if h == 0:
                print(f"open {self.path} faild.")
                return False

            self.handle = c_void_p(h)

            lib.GetScanScale.restype = c_int32
            img_size = self.__GetImageInfoEx(self.uImageWhole)
            self.scan_scale = lib.GetScanScale(self.handle);
            self._width = img_size.width
            self._height = img_size.height
            self._depth = img_size.depth
            if self._width>0 and self._height>0 and self._depth>0:
                return True
            else:
                print(f"Error size {self._width,self._height,self._depth}")
                return False
        except:
            print(f"open {self.path} faild.")
            return False

    def __len__(self):
        if self.is_open:
            return self._width*self._height*self._depth
        else:
            return 0

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

    @staticmethod
    def get_label_img_by_path(path):
        return TMAP(path).get_label_img()

    def get_label_img(self):
        img = self.get_image(TMAP.uImageLabel)
        if img is None:
            img = self.get_macro_label_img()
            if img is None:
                return img
            size = min(img.shape[0],img.shape[1])
            return img[:size,:size,:]
        return img

    def get_macro_label_img(self):
        return self.get_image(TMAP.uImageMacroLabel)

    def get_macro_img(self):
        return self.get_image(TMAP.uImageMacro)

    def get_whole_img(self,scale=1,layer=None):
        return self.crop_img(scale=scale,layer=layer)

    def get_image(self,etype):
        fun_ImgSize = self.__GetImageInfoEx(etype)
        lib = self.tmap_libs

        nBufferLength = fun_ImgSize.width * fun_ImgSize.height * fun_ImgSize.depth / 8
        nBufferLength = int(nBufferLength)

        lib.GetImageDataEx.restype = POINTER(c_ubyte)
        buffer = lib.GetImageDataEx(self.handle,c_int32(etype),c_int32(nBufferLength))

        W=fun_ImgSize.width
        H=fun_ImgSize.height
        pic = TMAP.buffer2img(buffer,W,H,3)

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
            width = self.width(scale)
        if height<=0:
            height = self.height(scale)

        if layer is not None:
            self.set_focus_layer(layer)

        GetCropImageDataEx_fun = lib.GetCropImageDataEx
        GetCropImageDataEx_fun.restype = POINTER(c_ubyte)

        nLeft = left*self.scan_scale/scale
        nRight = (left+width)*self.scan_scale/scale
        nTop = top*self.scan_scale/scale
        nBottom = (top+height)*self.scan_scale/scale
        r_left = int(max(0,nLeft))
        r_top = int(max(0,nTop))
        r_bottom = int(min(self.height(),nBottom))
        r_right = int(min(self.width(),nRight))
        nLeft = c_int32(r_left)
        nTop = c_int32(r_top)
        nBottom = c_int32(r_bottom)
        nRight = c_int32(r_right)
        buffer_size = c_int32(int(width*height*3))
        nScale = c_float(scale)

        f = GetCropImageDataEx_fun(self.handle,1, nLeft, nTop, nRight, nBottom, nScale, buffer_size)

        pic = TMAP.buffer2img(f,(r_right-r_left)*scale//self.scan_scale,(r_bottom-r_top)*scale//self.scan_scale,3)

        return pic

    def get_all_img_crops(self,width,height,scale=-1,with_pos_info=False):

        for y in range(0,self.height(scale),height):
            for x in range(0, self.width(scale), width):
                if with_pos_info:
                    yield self.crop_img(x,y,width,height,scale),x,y
                else:
                    yield self.crop_img(x,y,width,height,scale)

    def get_all_img_crops_in_all_focus(self, width, height, scale=-1,with_pos_info=False):
        nr = self.get_focus_number()
        for y in range(0, self.height(scale), height):
            for x in range(0, self.width(scale), width):
                for z,img in enumerate(self.crop_img_in_all_focus(x, y, width, height, scale)):
                    if with_pos_info:
                        yield img,x,y,z-nr
                    else:
                        yield img

    @staticmethod
    def buffer2img(buffer,width,height,channel=3,buffer_length=None,type=np.uint8):
        try:
            addr = addressof(buffer.contents)
            if addr == 0:
                return None
            length = width*height*channel
            if buffer_length is None:
                buffer_length = length
            ArrayTypeLen = c_uint8 * buffer_length
            array = np.frombuffer(ArrayTypeLen.from_address(addr), type, length)
            pic = array.reshape((height,width,3))
            pic = np.asarray(pic[:,:,::-1], dtype=type)
            return pic
        except:
            return None

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
