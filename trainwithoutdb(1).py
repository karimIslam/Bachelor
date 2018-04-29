from pycocotools.coco import COCO
from arango import ArangoClient, ArangoError
import tensorflow as tf
import math
batch_size = 50
client = ArangoClient(
   protocol ='http',
   host = 'localhost',
   port = 8529,
   username = 'root',
   password = '',
   enable_logging = True

)
db = client.db('my_database2')
cocog = db.graph('cocograph')
imagesdb = cocog.vertex_collection('images')
catsdb = cocog.vertex_collection('cts')
iscats = cocog.edge_collection('ctgs')

list1 = []
list2= []
allcats = []

for cat in catsdb:
    x = cat['ctname'] #remove duplicates..
    if " " in x:
        x = x.replace(" ", "")
    if x not in allcats:
        allcats.append(x)
allcats = sorted(allcats, key=str.lower)

for image in imagesdb:
  templist = [len(allcats)]
  templist = [None]*len(allcats)
  traversal_results = cocog.traverse(
    start_vertex='images/'+image['image_id'],
    direction='inbound',
    strategy='bfs',
    edge_uniqueness='global',
    vertex_uniqueness='global',
  )
  for i, outt in enumerate(allcats):
      if allcats[i] in str(traversal_results):
        templist[i] = 1
      else:
        templist[i] = 0
  strout = "val2014/COCO_val2014_"


  l = len(str(image['image_id']))
  if l <12:
      for k in range(12-l,0,-1):
          strout = strout + "0"
  list1.append( strout + str(image['image_id'])+".jpg")
  list2.append(templist)




result1 = '/. '.join(list1)

lines2 = []
for row2 in list2:
    lines2.append(',. '.join(map(str, row2)))
result2 = '///. '.join(lines2)

#cocog.delete_vertex_collection('output', purge=True)
labels = cocog.create_vertex_collection('output')

labels.insert({'_key': 'vallbls' ,'lblarr': result2})
labels.insert({'_key': 'valimgs', 'imgarr': result1})
labels.insert({'_key': 'valcats' , 'catsarr' : ',. '.join(map(str, allcats))})
