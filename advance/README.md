# Inception_v4_slim

 **模型** ：slim框架下的Inception_v4模型 

Inception_v4的Checkpoint：http://download.tensorflow.org/models/inception_v4_2016_09_09.tar.gz 

 **数据集** ：google的flower数据集http://download.tensorflow.org/example_images/flower_photos.tgz 5种类别的花

本文内容是我学习智亮老师图像识别课程的一些笔记与想法，加深学习，并方便自己回顾。智亮老师的课程讲的还是挺不错的，受益匪浅。
 
 **代码** ：https://codeload.github.com/isiosia/models/zip/lession 

 **GitHub** ：https://github.com/isiosia/models/tree/lession

 **数据准备** 

数据集下下来之后按/home/lwp/data/flower/my_flower_5路径放好，可以看到它是这个样子的，每个类的花一个文件夹

![输入图片说明](http://img.blog.csdn.net/20170727103647557?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvbHdwbHdm/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast "在这里输入图片标题")

打开一个我们可以看到里面是各种图片

![输入图片说明](http://img.blog.csdn.net/20170727103756273?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvbHdwbHdm/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast "在这里输入图片标题")

在模型目录source/models/slim下有一个脚本文件convert_tfrecord.sh 
convert_tfrecord.sh文件内容如下：

```
source env_set.sh
python download_and_convert_data.py \
  --dataset_name=$DATASET_NAME \
  --dataset_dir=$DATASET_DIR
```

可以看到通过env_set.sh传递变量 
env_set.sh文件内容如下：

```
export DATASET_NAME=my_flower_5
export DATASET_DIR=/home/lwp/data/flower
export CHECKPOINT_PATH=/home/lwp/pre_trained/inception_v4.ckpt
export TRAIN_DIR=/tmp/my_train_20170725
```

文件定义了：

- DATASET_NAME：数据集名称
- DATASET_DIR：数据集路径
- CHECKPOINT_PATH：预训练的inception_v4模型路径
- TRAIN_DIR：训练生成checkpoint存储路径

环境变量配置完后进入到模型目录下

```
$ cd source/models/slim
```

执行脚本：

```
$ ./convert_tfrecord.sh
```

完成后数据就准备好了 

![输入图片说明](http://img.blog.csdn.net/20170727105532449?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvbHdwbHdm/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast "在这里输入图片标题")


 **预训练模型准备** 

Inception_v4的Checkpoint：http://download.tensorflow.org/models/inception_v4_2016_09_09.tar.gz 
下载好之后存放在下面路径（路径在env_set.sh中定义）：

```
/home/lwp/pre_trained
```

 **运行训练脚本** 

（在修改好模型相关参数的前提下，如训练程序执行脚本run_train.sh,测试程序执行脚本run_eval.sh,环境变量env_set.sh等）

```
$ ./run_train.sh
```

run_train.sh内容如下：

```
source env_set.sh

nohup python -u train_image_classifier.py \
  --dataset_name=$DATASET_NAME \
  --dataset_dir=$DATASET_DIR \
  --checkpoint_path=$CHECKPOINT_PATH \
  --model_name=inception_v4 \
  --checkpoint_exclude_scopes=InceptionV4/Logits,InceptionV4/AuxLogits/Aux_logits \
  --trainable_scopes=InceptionV4/Logits,InceptionV4/AuxLogits/Aux_logits \
  --train_dir=$TRAIN_DIR \
  --learning_rate=0.001 \
  --learning_rate_decay_factor=0.76\
  --num_epochs_per_decay=50 \
  --moving_average_decay=0.9999 \
  --optimizer=adam \
  --ignore_missing_vars=True \
  --batch_size=32 > output.log 2>&1 &
```

http://blog.csdn.net/lwplwf/article/details/76099010中讲了在后台执行程序，run_train.sh脚本文件中设置了后台执行，因此通过下面命令监控程序运行情况：

```
$ tail -f output.log # 当前日志动态显示
# 或者
$ cat output.log # 一次显示整个log文件
```

如下所示

```
INFO:tensorflow:Summary name /clone_loss is illegal; using clone_loss instead.
INFO:tensorflow:Fine-tuning from /home/lwp/pre_trained/inception_v4.ckpt
2017-07-27 08:32:08.547822: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use SSE4.1 instructions, but these are available on your machine and could speed up CPU computations.
2017-07-27 08:32:08.547847: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use SSE4.2 instructions, but these are available on your machine and could speed up CPU computations.
2017-07-27 08:32:08.547868: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use AVX instructions, but these are available on your machine and could speed up CPU computations.
2017-07-27 08:32:08.547887: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use AVX2 instructions, but these are available on your machine and could speed up CPU computations.
2017-07-27 08:32:08.547892: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use FMA instructions, but these are available on your machine and could speed up CPU computations.
2017-07-27 08:32:08.861766: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:893] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2017-07-27 08:32:08.862322: I tensorflow/core/common_runtime/gpu/gpu_device.cc:955] Found device 0 with properties: 
name: GeForce GTX 1080 Ti
major: 6 minor: 1 memoryClockRate (GHz) 1.582
pciBusID 0000:01:00.0
Total memory: 10.91GiB
Free memory: 10.58GiB
2017-07-27 08:32:08.862342: I tensorflow/core/common_runtime/gpu/gpu_device.cc:976] DMA: 0 
2017-07-27 08:32:08.862350: I tensorflow/core/common_runtime/gpu/gpu_device.cc:986] 0:   Y 
2017-07-27 08:32:08.862359: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1045] Creating TensorFlow device (/gpu:0) -> (device: 0, name: GeForce GTX 1080 Ti, pci bus id: 0000:01:00.0)
INFO:tensorflow:Restoring parameters from /home/lwp/pre_trained/inception_v4.ckpt
INFO:tensorflow:Starting Session.
INFO:tensorflow:Saving checkpoint to path /tmp/my_train_20170725/model.ckpt
INFO:tensorflow:Starting Queues.
INFO:tensorflow:global_step/sec: 0
INFO:tensorflow:Recording summary at step 1.
INFO:tensorflow:global step 10: loss = 2.9544 (0.277 sec/step)
INFO:tensorflow:global step 20: loss = 2.7159 (0.267 sec/step)
INFO:tensorflow:global step 30: loss = 3.0572 (0.261 sec/step)
```

在/tmp/my_train_20170725路径下可以看到训练生成的checkpoint：meta、data、index

![这里写图片描述](http://img.blog.csdn.net/20170727084611516?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvbHdwbHdm/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast "在这里输入图片标题")

该路径在环境变量设置脚本env_set.sh中定义

 **运行测试脚本** 

```
$ ./run_eval.sh
```

run_eval.sh的内容如下：

```

source env_set.sh
python -u eval_image_classifier.py \
  --dataset_name=$DATASET_NAME \
  --dataset_dir=$DATASET_DIR \
  --dataset_split_name=validation \
  --model_name=inception_v4 \
  --checkpoint_path=$TRAIN_DIR \
  --eval_dir=/tmp/eval/validation \
  --eval_interval_secs=60 \
  --batch_size=32
``` 

其中eval_interval_secs=60是指定两次验证的最小间隔时间为60s，具体定义在eval_image_classifier.py文件中。

这里训练和验证程序是分开的，模型在刚开始训练的时候效果必然很差，并不需要去验证，而且训练过程持续时间很长，如果将训练和验证放在一起的话，无用的验证就占用的很多时间。 
将训练和验证分开这样就可以在其他机器上访问checkpoint（路径为/tmp/my_train_20170725）去做验证，这样就可以把资源分散开。

执行后如下：

```
.
.
.
name: GeForce GTX 1080 Ti
major: 6 minor: 1 memoryClockRate (GHz) 1.582
pciBusID 0000:01:00.0
Total memory: 10.91GiB
Free memory: 2.24GiB
2017-07-27 09:27:33.151287: I tensorflow/core/common_runtime/gpu/gpu_device.cc:976] DMA: 0 
2017-07-27 09:27:33.151292: I tensorflow/core/common_runtime/gpu/gpu_device.cc:986] 0:   Y 
2017-07-27 09:27:33.151299: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1045] Creating TensorFlow device (/gpu:0) -> (device: 0, name: GeForce GTX 1080 Ti, pci bus id: 0000:01:00.0)
INFO:tensorflow:Restoring parameters from /tmp/my_train_20170725/model.ckpt-11028
INFO:tensorflow:Starting evaluation at 2017-07-27-01:27:47
2017-07-27 09:27:49.207742: W tensorflow/core/common_runtime/bfc_allocator.cc:217] Allocator (GPU_0_bfc) ran out of memory trying to allocate 1.51GiB. The caller indicates that this is not a failure, but may mean that there could be performance gains if more memory is available.
INFO:tensorflow:Evaluation [1/12]
INFO:tensorflow:Evaluation [2/12]
INFO:tensorflow:Evaluation [3/12]
INFO:tensorflow:Evaluation [4/12]
INFO:tensorflow:Evaluation [5/12]
INFO:tensorflow:Evaluation [6/12]
INFO:tensorflow:Evaluation [7/12]
INFO:tensorflow:Evaluation [8/12]
INFO:tensorflow:Evaluation [9/12]
INFO:tensorflow:Evaluation [10/12]
INFO:tensorflow:Evaluation [11/12]
INFO:tensorflow:Evaluation [12/12]
INFO:tensorflow:Finished evaluation at 2017-07-27-01:27:56
2017-07-27 09:27:57.363998: I tensorflow/core/kernels/logging_ops.cc:79] eval/Recall_5[1]
2017-07-27 09:27:57.364187: I tensorflow/core/kernels/logging_ops.cc:79] eval/Accuracy[0.87760419]
INFO:tensorflow:Waiting for new checkpoint at /tmp/my_train_20170725
```

 **循环验证**  

可以看到给出了验证结果，注意最后一行Waiting for new checkpoint at /tmp/my_train_20170725，这是在eval_image_classifier.py中自定义了一个loop，去监听/tmp/my_train_20170725，一旦有新的checkpoint生成，就去执行一次验证。
 
**可视化训练：TensorBoard** 

执行：

```
$ tensorboard --logdir /tmp/my_train_20170725
```

得到：

```
Starting TensorBoard 55 at http://lw:6006
(Press CTRL+C to quit)
```

查看本机IP：

```
$ ifconfig -a
```

在浏览器中输入地址：

```
http://192.168.0.102：6006
```

![这里写图片描述](http://img.blog.csdn.net/20170727094702590?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvbHdwbHdm/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast "在这里输入图片标题")

如果出现TensorBoard但不显示内容的情况，可以尝试换一个浏览器，我用Fire fox就是不显示，换chrome就好了。

 **结束训练** 

查看Python进程 
执行：

```
$ ps -ef |grep python
```

得到：

```
lwp       2780  2025 99 08:31 pts/0    03:38:22 python -u train_image_classifier.py --dataset_name=my_flower_5 --dataset_dir=/home/lwp/data/flower --checkpoint_path=/home/lwp/pre_trained/inception_v4.ckpt --model_name=inception_v4 --checkpoint_exclude_scopes=InceptionV4/Logits,InceptionV4/AuxLogits/Aux_logits --trainable_scopes=InceptionV4/Logits,InceptionV4/AuxLogits/Aux_logits --train_dir=/tmp/my_train_20170725 --learning_rate=0.001 --learning_rate_decay_factor=0.76 --num_epochs_per_decay=50 --moving_average_decay=0.9999 --optimizer=adam --ignore_missing_vars=True --batch_size=32
lwp      18830  3674  1 09:40 pts/2    00:00:15 /usr/bin/python /usr/local/bin/tensorboard --logdir /tmp/my_train_20170725
lwp      24837  2763  0 09:53 pts/0    00:00:00 grep --color=auto python
```

可以看到模型训练的进程号为2780

杀掉进程，结束训练

```
$ kill 2780
```

 **模型导出和使用** 

 **模型导出** 
 
执行脚本：

```
$ ./export_freeze.sh
```

得到3个文件： 
![这里写图片描述](http://img.blog.csdn.net/20170727100046250?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvbHdwbHdm/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast "在这里输入图片标题") 

分别存储的是模型的label、权重、结构

export_freeze.sh文件内容如下：

```
source env_set.sh
python -u export_inference_graph.py \
  --model_name=inception_v4 \
  --output_file=./my_inception_v4.pb \
  --dataset_name=$DATASET_NAME \
  --dataset_dir=$DATASET_DIR


NEWEST_CHECKPOINT=$(ls -t1 $TRAIN_DIR/model.ckpt*| head -n1)
NEWEST_CHECKPOINT=${NEWEST_CHECKPOINT%.*}
python -u ~/tensorflow/tensorflow/python/tools/freeze_graph.py \
  --input_graph=my_inception_v4.pb \
  --input_checkpoint=$NEWEST_CHECKPOINT \
  --output_graph=./my_inception_v4_freeze.pb \
  --input_binary=True \
  --output_node_name=InceptionV4/Logits/Predictions

cp $DATASET_DIR/labels.txt ./my_inception_v4_freeze.label
```
 
**模型使用**  
基于python的webserver 
执行脚本：

```
$ ./server.sh
```

得到：

```
listening on port 5001
2017-07-27 10:04:54.279779: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use SSE4.1 instructions, but these are available on your machine and could speed up CPU computations.
2017-07-27 10:04:54.279800: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use SSE4.2 instructions, but these are available on your machine and could speed up CPU computations.
2017-07-27 10:04:54.279806: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use AVX instructions, but these are available on your machine and could speed up CPU computations.
2017-07-27 10:04:54.279810: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use AVX2 instructions, but these are available on your machine and could speed up CPU computations.
2017-07-27 10:04:54.279814: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use FMA instructions, but these are available on your machine and could speed up CPU computations.
2017-07-27 10:04:54.411389: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:893] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2017-07-27 10:04:54.411804: I tensorflow/core/common_runtime/gpu/gpu_device.cc:955] Found device 0 with properties: 
name: GeForce GTX 1080 Ti
major: 6 minor: 1 memoryClockRate (GHz) 1.582
pciBusID 0000:01:00.0
Total memory: 10.91GiB
Free memory: 10.50GiB
2017-07-27 10:04:54.411818: I tensorflow/core/common_runtime/gpu/gpu_device.cc:976] DMA: 0 
2017-07-27 10:04:54.411822: I tensorflow/core/common_runtime/gpu/gpu_device.cc:986] 0:   Y 
2017-07-27 10:04:54.411828: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1045] Creating TensorFlow device (/gpu:0) -> (device: 0, name: GeForce GTX 1080 Ti, pci bus id: 0000:01:00.0)
 * Running on http://0.0.0.0:5001/ (Press CTRL+C to quit)
```

在浏览器输入地址：

```
http://本机IP:5001
```

![这里写图片描述](http://img.blog.csdn.net/20170727101054212?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvbHdwbHdm/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast "在这里输入图片标题")

选择一张图片并上传，然后就会显示识别结果 
（注意，图片所在路径为/tmp/upload，在server.sh文件中定义）

server.sh文件内容如下：

```
python -u server.py \
  --model_name=my_inception_v4_freeze.pb \
  --label_file=my_inception_v4_freeze.label \
  --upload_folder=/tmp/uploadpython -u server.py \
  --model_name=my_inception_v4_freeze.pb \
  --label_file=my_inception_v4_freeze.label \
  --upload_folder=/tmp/upload
```

具体定义在server.py文件中

![这里写图片描述](http://img.blog.csdn.net/20170727101145271?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvbHdwbHdm/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast "在这里输入图片标题")

如图得到5个分类的得分值，识别为sunflowers的score为0.79741

一些思考：我们刚才做的是5分类，分别是几种花，如果我们现在有一张猫的图片，这张图片对模型数据来说是未标识的，也就是对未标识的物体进行预测会是什么结果？ 
我们来试一下： 
![这里写图片描述](http://img.blog.csdn.net/20170727110209377?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvbHdwbHdm/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast "在这里输入图片标题")

可以看到，同样也给出了分类预测的得分值，可是这只猫当然不是蒲公英，这也是目前图像识别模型普遍存在的问题，也就是它不知道自己不知道。对人类而言，对于这5类花的预测分类，如果碰见这只猫，我们会说这不是花，或者遇见一种不认识的不属于这5类的我们会说我们不认识，或者不属于这5类，但是对于模型而言，它目前做不到，它最终只会把这只猫分到其中某一类花里面去。

