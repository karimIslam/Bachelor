
import os
import glob
import ntpath

from arango import ArangoClient, ArangoError
from pycocotools.coco import COCO

dataType='val2014'
ins_annFile='annotations/instances_%s.json'%(dataType)


coco = COCO(ins_annFile)

cats = coco.loadCats(coco.getCatIds())
# get all the images details form the instance annotations as well
imgs = coco.loadImgs(coco.getImgIds())

client = ArangoClient(
   protocol ='tcp',
   host = 'localhost',
   port = 8529,
   username = 'root',
   password = '',
   enable_logging = True

)

# Create a new graph
client = ArangoClient()
graph = client.db('my_database2').create_graph('cocograph')
supercatsdb = graph.create_vertex_collection('superctgs')
categoriesdb = graph.create_vertex_collection('cts')
imagesdb = graph.create_vertex_collection('images')
iscatof = graph.create_edge_definition(
    name='isctgof',
    from_collections=['cts'],
    to_collections=['images']
)
#dict1 = {}
#for cat in cats:
 #   x = cat['supercategory']
  #  if x not in dict1.values():
   # 	dict1.append(x)

#for d in dict1:
 #   supercatsdb.insert({'_key': d,'supercat': d})

for cat in cats:
    x = cat['supercategory']
    try:
       supercatsdb.insert({'_key': x,'superctg': x})
    except ArangoError as exc:
        print(exc.message)

for ct in cats:
    try:
        categoriesdb.insert({'_key':ct['name'],'ctname':ct['name']})
        for im in coco.loadImgs(coco.getImgIds(catIds=coco.getCatIds(catNms = [ct['name']]))):
           x = im['id']
           iscatof.insert({'_from':'cts/' + ct['name'], '_to':'images/' + str(x)})
    except ArangoError as exc:
        ct['name'] = ct['name'].replace(" ", "")
        try:
            categoriesdb.insert({'_key':ct['name'],'ctname':ct['name']})
            for im in coco.loadImgs(coco.getImgIds(catIds=coco.getCatIds(catNms = [ct['name']]))):
               x = im['id']
               iscatof.insert({'_from':'cts/' + ct['name'], '_to':'images/' + str(x)})
        except ArangoError as exc:
            print(exc.message)
for im in imgs:
    x = im['id']
    try:
        imagesdb.insert({'_key':str(x),'image_id':str(x)})
    except ArangoError as exc:
        print(exc.message)

# print (""+ str(int(im['image_id'])))


    #
