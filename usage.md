

<h1>
    <p align="center"> YOLOv8的TensorRT推理部署</p>
</h1>
# TensorRT模型转换



# C++推理部署

## 参数说明

- v8中`output0`输出8400个结果，每个结果的维度是116，而v5是25200个结果，每个结果的维度是117。

简述一下v8输出的是啥，116为4+80+32，4为box的`cx`，`cy`，` w`，` h`；80是每个类的置信度，32是分割需要用到的，v8和v5的区别在于少了目标的置信度，v5是4+1+80+32，这个1就是是否为目标的置信度。



可以看到，对于目标检测模型yolov8n来说，其输出为1 x 84 x 8400 ；而分割模型yolov8n-seg的输出为1 x 116 x 8400 

![image-20230507193908941](usage.assets/image-20230507193908941.png)



# Reference

- [Yolov8实例分割Tensorrt部署实战](https://blog.csdn.net/qq_41043389/article/details/128682057)
-  [c++下面部署yolov8检测和实例分割模型](https://blog.csdn.net/qq_34124780/article/details/128800661?)
- [渣渣的yolo踩坑记录](https://blog.csdn.net/qq_34124780/category_10872730.html)
