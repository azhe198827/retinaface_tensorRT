# -*- coding: UTF-8 -*-
import src.c2oObject as Node
##-----------------------------------------------------Crop层-------------------------------------------------------##

#计算输出维度
def getCropOutShape(input_shape):
    output_shape = [input_shape[1]]
    return output_shape
#构建节点
def createCrop(layer, nodename, inname, outname, input_shape):
    output_shape = getCropOutShape(input_shape)
    #构建node
    node = Node.c2oNode(layer, nodename, "Crop", inname, outname, input_shape, output_shape)
    print(nodename, "节点构建完成")
    return node