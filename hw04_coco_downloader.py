#%matplotlib inline
import argparse
from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import pylab
import os
import PIL 
from PIL import Image, ImageDraw, ImageFont

parser = argparse.ArgumentParser ( description = 'HW04 COCO downloader')
parser.add_argument ( '--root_path' , required = True , type =str )
parser.add_argument ( '--coco_json_path', required = True ,type = str )
parser.add_argument ( '--class_list' , required = True , nargs ='*' , type = str )
parser.add_argument ( '--images_per_class' , required = True ,type = int )
args , args_other = parser.parse_known_args ()
pylab.rcParams['figure.figsize'] = (8.0, 10.0)
dataDir=''
dataType=''
annFile=args.coco_json_path+"instances_train2017.json"
coco=COCO(annFile)
# display COCO categories and supercategories
cats = coco.loadCats(coco.getCatIds())
catId = coco.getCatIds(catNms=['cat','dog','airplane']);
print(catId)
nms=[cat['name'] for cat in cats]
print('COCO categories: \n{}\n'.format(' '.join(nms)))

nms = set([cat['supercategory'] for cat in cats])
print('COCO supercategories: \n{}'.format(' '.join(nms)))
# get all images containing given categories, select one at random
catIds = coco.getCatIds(catNms=['dog']);
imgIds = coco.getImgIds(catIds=catIds );
#imgIds = coco.getImgIds(imgIds = [324158])
# load and display image
# I = io.imread('%s/images/%s/%s'%(dataDir,dataType,img['file_name']))
# use url to load image
# load and display image
# I = io.imread('%s/images/%s/%s'%(dataDir,dataType,img['file_name']))
# use url to load image
from PIL import Image, ImageDraw, ImageFont
import PIL
for id in args.class_list:
 class_list_index=0
 
 catIds = coco.getCatIds(catNms=[id]);
 imgIds = coco.getImgIds(catIds=catIds );
 print(args.root_path)
 if not os.path.exists(args.root_path):
    os.makedirs(args.root_path)
 for i in range(len(imgIds)):
    img_path=os.path.join(args.root_path, args.class_list[class_list_index])
    print(img_path)
    if not os.path.exists(img_path):
     os.mkdir(os.path.join(args.root_path, args.class_list[class_list_index]), mode=0o666)
    img = coco.loadImgs(imgIds[i])[0]
    image= io.imread(img['coco_url'])
    img = Image.fromarray(image)
    draw = ImageDraw.Draw(img)
    img= img.resize((64, 64), Image.BOX)
    img.save(img_path+"/"+args.class_list[class_list_index]+str(i)+".jpg")
 
 class_list_index+=1
# load and display image
# I = io.imread('%s/images/%s/%s'%(dataDir,dataType,img['file_name']))
# use url to load image
# load and display image
# I = io.imread('%s/images/%s/%s'%(dataDir,dataType,img['file_name']))
# use url to load image


plt.axis('off')
plt.imshow(img)
plt.show()
