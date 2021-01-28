from thirdparty.images.unic_tmap import TMAP
import matplotlib.pyplot as plt

img = TMAP("/home/wj/ai/ddbk/cell/pos_cell_for_train/stage_02/m1/2020-09-04/SLICEID_20200904110216_A_8_M1.TMAP")

#取标签图
label_img = img.get_label_img()
plt.imshow(label_img)
plt.show()

#取宏观图
macro_img = img.get_macro_img()
plt.imshow(macro_img)
plt.show()

#取1倍图
whole_img = img.get_whole_img(scale=1)
plt.imshow(whole_img)
print(whole_img.shape)
plt.show()

#遍历2倍图
for c_img in img.get_all_img_crops(height=800,width=900,scale=2):
    plt.imshow(c_img)
    print(c_img.shape)
    plt.show()

#指定位置取图
c_img= img.crop_img(left=100,top=128,width=512,height=256,scale=20)
plt.imshow(c_img)
print(c_img.shape)
plt.show()
