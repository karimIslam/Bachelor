import math
from arango import ArangoClient, ArangoError
import tensorflow as tf
import numpy as np
batch_size = 50
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



def _parse_function(filename, label):
  image_string = tf.read_file(filename)
  image = tf.image.decode_jpeg(image_string, channels=3)
  # This will convert to float values in [0, 1]
  image = tf.image.convert_image_dtype(image, tf.float32)
  image = tf.image.resize_images(image, [299, 299])
  return image, label

# A vector of filenames.
filenames = tf.constant(list1)

# `labels[i]` is the label for the image in `filenames[i].
labels = tf.constant(list2)
dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
dataset = dataset.repeat()
dataset = dataset.map(_parse_function,num_parallel_calls=8)
dataset = dataset.batch(batch_size)


# step 4: create iterator and final input tensor
iterator = dataset.make_one_shot_iterator()




#inception --------------------------------------------------------------------



map1 = 32
map2 = 64
num_fc1 = 700 #1028
num_fc2 = 80
reduce1x1 = 16
dropout=0.5

graph = tf.Graph()
with graph.as_default():
    lr = tf.placeholder(tf.float32)
    #train data and labels
    X = tf.placeholder(tf.float32,shape=(batch_size,299,299,3))
    y_ = tf.placeholder(tf.float32,shape=(batch_size,80))
    
    def createWeight(size,Name):
        return tf.Variable(tf.truncated_normal(size, stddev=0.1),
                          name=Name)
    
    def createBias(size,Name):
        return tf.Variable(tf.constant(0.1,shape=size),
                          name=Name)
    
    def conv2d_s1(x,W):
        return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')
    
    def max_pool_3x3_s1(x):
        return tf.nn.max_pool(x,ksize=[1,3,3,1],
                             strides=[1,1,1,1],padding='SAME')
    
    
    #Inception Module1
    #
    #follows input
    W_conv1_1x1_1 = createWeight([1,1,3,map1],'W_conv1_1x1_1')
    b_conv1_1x1_1 = createWeight([map1],'b_conv1_1x1_1')
    
    #follows input
    W_conv1_1x1_2 = createWeight([1,1,3,reduce1x1],'W_conv1_1x1_2')
    b_conv1_1x1_2 = createWeight([reduce1x1],'b_conv1_1x1_2')
    
    #follows input
    W_conv1_1x1_3 = createWeight([1,1,3,reduce1x1],'W_conv1_1x1_3')
    b_conv1_1x1_3 = createWeight([reduce1x1],'b_conv1_1x1_3')
    
    #follows 1x1_2
    W_conv1_3x3 = createWeight([3,3,reduce1x1,map1],'W_conv1_3x3')
    b_conv1_3x3 = createWeight([map1],'b_conv1_3x3')
    
    #follows 1x1_3
    W_conv1_5x5 = createWeight([5,5,reduce1x1,map1],'W_conv1_5x5')
    b_conv1_5x5 = createBias([map1],'b_conv1_5x5')
    
    #follows max pooling
    W_conv1_1x1_4= createWeight([1,1,3,map1],'W_conv1_1x1_4')
    b_conv1_1x1_4= createWeight([map1],'b_conv1_1x1_4')
    
    
    
    #Inception Module2
    #
    #follows inception1
    W_conv2_1x1_1 = createWeight([1,1,4*map1,map2],'W_conv2_1x1_1')
    b_conv2_1x1_1 = createWeight([map2],'b_conv2_1x1_1')
    
    #follows inception1
    W_conv2_1x1_2 = createWeight([1,1,4*map1,reduce1x1],'W_conv2_1x1_2')
    b_conv2_1x1_2 = createWeight([reduce1x1],'b_conv2_1x1_2')
    
    #follows inception1
    W_conv2_1x1_3 = createWeight([1,1,4*map1,reduce1x1],'W_conv2_1x1_3')
    b_conv2_1x1_3 = createWeight([reduce1x1],'b_conv2_1x1_3')
    
    #follows 1x1_2
    W_conv2_3x3 = createWeight([3,3,reduce1x1,map2],'W_conv2_3x3')
    b_conv2_3x3 = createWeight([map2],'b_conv2_3x3')
    
    #follows 1x1_3
    W_conv2_5x5 = createWeight([5,5,reduce1x1,map2],'W_conv2_5x5')
    b_conv2_5x5 = createBias([map2],'b_conv2_5x5')
    
    #follows max pooling
    W_conv2_1x1_4= createWeight([1,1,4*map1,map2],'W_conv2_1x1_4')
    b_conv2_1x1_4= createWeight([map2],'b_conv2_1x1_4')
    
    

    #Fully connected layers
    #since padding is same, the feature map with there will be 4 28*28*map2
    W_fc1 = createWeight([299*299*(4*map2),num_fc1],'W_fc1')
    b_fc1 = createBias([num_fc1],'b_fc1')
    
    W_fc2 = createWeight([num_fc1,num_fc2],'W_fc2')
    b_fc2 = createBias([num_fc2],'b_fc2')

    def model(x,train=True):
        #Inception Module 1
        conv1_1x1_1 = conv2d_s1(x,W_conv1_1x1_1)+b_conv1_1x1_1
        conv1_1x1_2 = tf.nn.relu(conv2d_s1(x,W_conv1_1x1_2)+b_conv1_1x1_2)
        conv1_1x1_3 = tf.nn.relu(conv2d_s1(x,W_conv1_1x1_3)+b_conv1_1x1_3)
        conv1_3x3 = conv2d_s1(conv1_1x1_2,W_conv1_3x3)+b_conv1_3x3
        conv1_5x5 = conv2d_s1(conv1_1x1_3,W_conv1_5x5)+b_conv1_5x5
        maxpool1 = max_pool_3x3_s1(x)
        conv1_1x1_4 = conv2d_s1(maxpool1,W_conv1_1x1_4)+b_conv1_1x1_4
        
        #concatenate all the feature maps and hit them with a relu
        inception1 = tf.nn.relu(tf.concat(3, [conv1_1x1_1,conv1_3x3,conv1_5x5,conv1_1x1_4]))

        
        #Inception Module 2
        conv2_1x1_1 = conv2d_s1(inception1,W_conv2_1x1_1)+b_conv2_1x1_1
        conv2_1x1_2 = tf.nn.relu(conv2d_s1(inception1,W_conv2_1x1_2)+b_conv2_1x1_2)
        conv2_1x1_3 = tf.nn.relu(conv2d_s1(inception1,W_conv2_1x1_3)+b_conv2_1x1_3)
        conv2_3x3 = conv2d_s1(conv2_1x1_2,W_conv2_3x3)+b_conv2_3x3
        conv2_5x5 = conv2d_s1(conv2_1x1_3,W_conv2_5x5)+b_conv2_5x5
        maxpool2 = max_pool_3x3_s1(inception1)
        conv2_1x1_4 = conv2d_s1(maxpool2,W_conv2_1x1_4)+b_conv2_1x1_4
        
        #concatenate all the feature maps and hit them with a relu
        inception2 = tf.nn.relu(tf.concat(3,[conv2_1x1_1,conv2_3x3,conv2_5x5,conv2_1x1_4]))

        #flatten features for fully connected layer
        inception2_flat = tf.reshape(inception2,[-1,299*299*4*map2])
        
        #Fully connected layers
        if train:
            h_fc1 =tf.nn.dropout(tf.nn.relu(tf.matmul(inception2_flat,W_fc1)+b_fc1),dropout)
        else:
            h_fc1 = tf.nn.relu(tf.matmul(inception2_flat,W_fc1)+b_fc1)

        return tf.matmul(h_fc1,W_fc2)+b_fc2
    
    unscaledlogits = model(X)
    Y = tf.nn.sigmoid(unscaledlogits)
    cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=unscaledlogits, labels=Y_)
    cross_entropy = tf.reduce_mean(tf.reduce_sum(cross_entropy,axis = 1))*100


    correct_prediction = tf.equal(tf.round(Y), tf.round(Y_))
#mean accuracy over all labels --%of labels predicted correctly
    accuracy1 = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#accuracy where all labels need to be correct -- % of images with all labels predicted correctly
#tf.reduce min takes min of correct prediction for each element of the batch
#this min is 1 if all elements are 1( all predictions are correct) and 0 if at least one element is false
    all_labels_true = tf.reduce_min(tf.cast(correct_prediction,tf.float32),1)
    accuracy2 = tf.reduce_mean(all_labels_true)
#top_k = tf.reduce_mean(tf.cast(tf.nn.in_top_k(predictions = Ylogits,targets = tf.cast(Y_,tf.int32), k = 5),tf.float32))   top_k2 = tf.reduce_mean(tf.cast(tf.nn.in_top_k(predictions = Ylogits,targets = tf.cast(Y_,tf.int32), k = 5),tf.float32))


    train_step = tf.train.AdamOptimizer(lr).minimize(cross_entropy)
    
    
    
    #initialize variable
    init = tf.initialize_all_variables()





    num_steps = 20000
    sess.run(init)
    print("Model initialized.")
    images, labels = sess.run(iterator.get_next()) 
    sess = tf.Session(graph=graph)
 
#initialize variables
    
    
    
    for j in range(16500):
        max_learning_rate = 0.003
        min_learning_rate = 0.0001
        decay_speed = 2000.0
        learning_rate = min_learning_rate + (max_learning_rate - min_learning_rate) * math.exp(-i/decay_speed)

        _,loss = sess.run([train_step,cross_entropy], {X: images, Y_: labels, lr: learning_rate})
        result1,result2 = sess.run([accuracy1,accuracy2], {X: images, Y_: labels, lr: learning_rate})
        print("loss", loss)
        print("iteration", j)
        print("Accuracy1",result1)
        print("Accuracy2",result2)






