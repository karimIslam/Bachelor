import math
from arango import ArangoClient, ArangoError
import tensorflow as tf
import numpy 


client = ArangoClient(
   protocol ='http',
   host = 'localhost',
   port = 8529,
   username = 'root',
   password = '',
   enable_logging = True

)

db = client.db('my_database')
cocog = db.graph('cocograph')

out = cocog.vertex_collection('output')
x = out.get('valcats')
x2 = out.get('valimgs')
x3 = out.get('vallbls')
list1 = x2['imgarr'].split('/. ')
tmp = x3['lblarr'].split('///. ')
list2 = []
for t in tmp:
    l = [int(x) for x in t.split(',. ')]
    list2.append(l)

allcats = x['catsarr'].split(',. ')


db2 = client.db('my_database2')
cocog2 = db.graph('cocograph')

out2 = cocog2.vertex_collection('output')
x22 = out2.get('valimgs')
x32 = out.get('vallbls')
list12 = x22['imgarr'].split('/. ')
tmp2 = x32['lblarr'].split('///. ')
list22 = []
for t2 in tmp2:
    l = [int(x) for x in t2.split(',. ')]
    list22.append(l)


#tf code



def _parse_function(filename, label):
  image_string = tf.read_file(filename)
  image = tf.image.decode_jpeg(image_string, channels=3)
  # This will convert to float values in [0, 1]
  image = tf.image.convert_image_dtype(image, tf.float32)
  image = tf.image.resize_images(image, [64, 64])
  return image, label

# A vector of filenames.
filenames = tf.constant(list1)

# `labels[i]` is the label for the image in `filenames[i].
labels = tf.constant(list2)
dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
dataset = dataset.repeat()
dataset = dataset.map(_parse_function,num_parallel_calls=8)
dataset = dataset.batch(50)


# step 4: create iterator and final input tensor
iterator = dataset.make_one_shot_iterator()


filenames2 = tf.constant(list12)

# `labels[i]` is the label for the image in `filenames[i].
labels2 = tf.constant(list22)
dataset2 = tf.data.Dataset.from_tensor_slices((filenames2, labels2))
dataset2 = dataset2.repeat()
dataset2 = dataset2.map(_parse_function,num_parallel_calls=8)
dataset2 = dataset2.batch(1)


# step 4: create iterator and final input tensor
iterator2 = dataset2.make_one_shot_iterator()


def reshapeX(_batch_size):
    return tf.reshape(X, tf.stack([_batch_size, -1]))
def reshapeY(_batch_size):
    return tf.reshape(Y_, tf.stack([_batch_size, -1]))



#tf code start                                                        -----------------------------

tf.set_random_seed(0)
X = tf.placeholder(tf.float32, [None, 64, 64, 3])
C = tf.placeholder(tf.int32)
V = tf.placeholder(tf.int32)
sumseen = tf.placeholder(tf.float32)
sumtrue = tf.placeholder(tf.float32)
avgg = tf.placeholder(tf.float32)
# correct answers will go here
l = tf.placeholder(tf.float32, [80])
Y_ = tf.placeholder(tf.float32, [None, 80])
# variable learning rate
lr = tf.placeholder(tf.float32)
batch_size = tf.placeholder(tf.int32)
pkeep = tf.placeholder(tf.float32)

reshaped = reshapeX(batch_size)
reshaped = reshapeY(batch_size)

W = tf.Variable(tf.truncated_normal([5, 5, 3, 4], stddev=0.1))
B = tf.Variable(tf.ones([4])/10) # 2 is the number of output channels

W1 = tf.Variable(tf.truncated_normal([5, 5, 4, 8], stddev=0.1))
B1 = tf.Variable(tf.ones([8])/10)

W2 = tf.Variable(tf.truncated_normal([4, 4, 8, 12], stddev=0.1))
B2 = tf.Variable(tf.ones([12])/10)

W3 = tf.Variable(tf.truncated_normal([16*16*12, 200], stddev=0.1))
B3 = tf.Variable(tf.zeros([200]))

W4 = tf.Variable(tf.truncated_normal([200, 80] ,stddev=0.1))
B4 = tf.Variable(tf.zeros([80]))



stride = 1  # output is still 28x28
Ycnvl1 = tf.nn.conv2d(X, W, strides=[1, stride, stride, 1], padding='SAME')
Yl1 = tf.nn.relu(Ycnvl1 + B)
Ycnvl2 = tf.nn.conv2d(Yl1, W1, strides=[1, 2, 2, 1], padding='SAME')
Yl2 = tf.nn.relu(Ycnvl2 + B1)
Ycnvl3 = tf.nn.conv2d(Yl2, W2, strides=[1, 2, 2, 1], padding='SAME')
Yl3 = tf.nn.relu(Ycnvl3 + B2)

Y4 = tf.nn.relu(tf.matmul(tf.reshape(Yl3, shape=[-1, 16*16*12]), W3) + B3)
Ylogits = tf.matmul(tf.nn.dropout(Y4, pkeep), W4) + B4
Y = tf.nn.sigmoid(Ylogits)


cross_entropy = tf.losses.sigmoid_cross_entropy(logits=Ylogits, multi_class_labels=Y_)
#cross_entropy = tf.multiply(tf.reduce_mean(cross_entropy,axis = 0),tf.constant(100))

rounded = tf.round(Y) #print this    di keda el batch rounded which needs to be compared for each element fa ha map
C = 0
V = 0
sumseen = 0.
sumtrue = 0.
avgg = 0.
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
def avgfunc(elm):
  global C
  global V
  global sumseen
  global sumtrue
  global avgg
  global sess
  tru = Y_[C]
  elm = tf.Print(elm, [elm], message="This is elm: ")
  print(elm.eval(session = sess))
  if(elm ==1.):
      print("here")
      sumseen+=1
      if(elm == tru[V]):
          sumtrue+=1
  V = V+1
  if(V == 80):
      V = 0
      avgg = (sumtrue/sumseen)*100
  return sumseen
  
  
  


def _comparefunc(elem_probs):
  global C
  tru = Y_[C]
  mapout = tf.map_fn(avgfunc, elem_probs)
  


  #elem_probs = tf.Print(elem_probs, [elem_probs], message="This is pred: ", summarize=1000)
#   mapout = tf.map_fn(comp2, elem_probs)
#   p = tf.equal(elem_probs, tru)
#   avg = ((tf.count_nonzero(tf.cast(p,tf.float32)))/80)
#   avg = tf.cast(avg,tf.float32)
  #avg = tf.Print(avg, [avg], message="This is avg: ")
  C = C+1
  if(C == 50):
      C = 0
  return avgg

outputafter = tf.map_fn(_comparefunc, rounded)
accuracyofbatch = tf.reduce_mean(outputafter)


# correct_prediction = tf.equal(tf.round(Y), tf.round(Y_))
# #mean accuracy over all labels --%of labels predicted correctly
# accuracy1 = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
# #accuracy where all labels need to be correct -- % of images with all labels predicted correctly
# #tf.reduce min takes min of correct prediction for each element of the batch
# #this min is 1 if all elements are 1( all predictions are correct) and 0 if at least one element is false
# all_labels_true = tf.reduce_min(tf.cast(correct_prediction,tf.float32),1)
# accuracy2 = tf.reduce_mean(all_labels_true)
# #top_k = tf.reduce_mean(tf.cast(tf.nn.in_top_k(predictions = Ylogits,targets = tf.cast(Y_,tf.int32), k = 5),tf.float32))   top_k2 = tf.reduce_mean(tf.cast(tf.nn.in_top_k(predictions = Ylogits,targets = tf.cast(Y_,tf.int32), k = 5),tf.float32))

numpy.set_printoptions(threshold=numpy.nan)
train_step = tf.train.AdamOptimizer(lr).minimize(cross_entropy)

# init

saver = tf.train.Saver()
r = iterator.get_next()
            
testimgs, testlbls = sess.run(iterator2.get_next())
print("Started Training")
if(0):
    saver.restore(sess, "cnn/5_layer.ckpt")
    
def training_step(i):
    if(1):
        for j in range(i):
            images, labels = sess.run(r)
            max_learning_rate = 0.003
            min_learning_rate = 0.0001
            decay_speed = 2000.0
            learning_rate = min_learning_rate + \
                (max_learning_rate - min_learning_rate) * math.exp(-i/decay_speed)

            _, loss,result1 = sess.run([train_step, cross_entropy,outputafter], {
                               X: images, Y_: labels, lr: learning_rate, pkeep: 0.75, batch_size: 50})
            #print("acc",result1, "loss", loss)

            k=  tf.unstack(outputafter)
            print(k)

        #print("loss", loss)
        #print("iteration", j)
        #print("Accuracy1",result1)
        #print("Accuracy2",result2)
        #print("ds",images)
        #print(images.shape)
        #pd = dataset[0]
        #pd = pd.reshape(28,28)
        #im = sess.run(iterator.get_next())[:,:,0]
        #Image.fromarray(np.asarray(images)).show()
        #pyplot.imshow(images)
        #pyplot.show()
        #break
        saver.save(sess, "cnn/5_layer.ckpt")
    #for k in range(4150):
    #    pred = sess.run([Y], {X: testimgs, lr: learning_rate, pkeep: 0.75, batch_size : 1})
    #    rounded = sess.run(tf.round(pred))
    #    print("test img",k , "output",rounded)
training_step(8279)





