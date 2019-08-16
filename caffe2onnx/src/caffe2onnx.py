# -*- coding: UTF-8 -*-
import src.OPs as op
from src.c2oObject import *
from onnx import helper
import copy
import numpy as np
from src.op_layer_info import *
class Caffe2Onnx():
    def __init__(self,net,model,onnxname):
        self.onnxmodel = c2oGraph(onnxname)
        self.__NetLayer = self.__getNetLayer(net)
        self._ModelLayer = self.__getModelLayer(model)

        self.model_input_name = []
        self.model_input_shape = []
        self.__n = 0
        self.NodeList = []
        LayerList = self.__addInputsTVIandGetLayerList(net)
        self.__getNodeList(LayerList)
        self.___addOutputsTVIandValueInfo()

    def __getNetLayer(self,net):
        if len(net.layer)==0 and len(net.layers)!=0:
            return net.layers
        elif len(net.layer)!=0 and len(net.layers)==0:
            return net.layer
        else:
            print("prototxt layer error")
            return -1

    def __getModelLayer(self,model):
        if len(model.layer) == 0 and len(model.layers) != 0:
            return model.layers
        elif len(model.layer) != 0 and len(model.layers) == 0:
            return model.layer
        else:
            print("caffemodel layer error")
            return -1

    def __addInputsTVIandGetLayerList(self,net):
        if net.input == [] and self.__NetLayer[0].type == "Input":
            layer_list = []
            for lay in self.__NetLayer:
                if lay.type == "Input":
                    in_tvi = helper.make_tensor_value_info(lay.name+"_input", TensorProto.FLOAT, lay.input_param.shape[0].dim)
                    self.model_input_name.append(lay.name+"_input")
                    self.model_input_shape.append(lay.input_param.shape[0].dim)
                    self.onnxmodel.addInputsTVI(in_tvi)
                else:
                    layer_list.append(lay)
            return layer_list

        elif net.input !=[]:
            in_tvi = helper.make_tensor_value_info("input", TensorProto.FLOAT, net.input_dim)
            self.model_input_name.append("input")
            self.model_input_shape.append(net.input_dim)
            self.onnxmodel.addInputsTVI(in_tvi)
            return self.__NetLayer

        else:
            print("error:the caffe model has no input")
            return -1



    def __addInputsTVIfromParams(self,layer,ParamName,ParamType):
        #print(layer.type)
        ParamShape = []
        ParamData = []
        for model_layer in self._ModelLayer:
            if layer.name == model_layer.name:
                Params = copy.deepcopy(model_layer.blobs)
                ParamShape = [p.shape.dim for p in Params]
                ParamData = [p.data for p in Params]
                if layer.type == "BatchNorm":
                    ParamShape = ParamShape[0:len(ParamShape) - 1]
                    ParamData = ParamData[0:len(ParamData) - 1]
                break


        if ParamShape != []:
            ParamName = ParamName[0:len(ParamShape)]
            ParamType = ParamType[0:len(ParamShape)]
            for i in range(len(ParamShape)):
                #print(ParamName[i])
                ParamName[i] = layer.name+ParamName[i]
                p_tvi = helper.make_tensor_value_info(ParamName[i], ParamType[i], ParamShape[i])
                p_t = helper.make_tensor(ParamName[i],ParamType[i],ParamShape[i],ParamData[i])
                self.onnxmodel.addInputsTVI(p_tvi)
                self.onnxmodel.addInitTensor(p_t)
        return ParamName

    def __addInputsTVIfromMannul(self,layer,ParamName,ParamType,ParamShape,ParamData):
        Param_Name = copy.deepcopy(ParamName)
        for i in range(len(ParamShape)):
            Param_Name[i] = layer.name + ParamName[i]
            p_tvi = helper.make_tensor_value_info(Param_Name[i], ParamType[i], ParamShape[i])
            if i ==0 and ParamData[i]==-1:
                p_t = helper.make_tensor(Param_Name[i], ParamType[i], ParamShape[i], ParamData[i],True)
            else:
                p_t = helper.make_tensor(Param_Name[i], ParamType[i], ParamShape[i], ParamData[i])
            self.onnxmodel.addInputsTVI(p_tvi)
            self.onnxmodel.addInitTensor(p_t)
        return Param_Name


    def __getLastLayerOutNameAndShape(self,layer):
        outname = []
        outshape = []
        if self.NodeList == []:
            outname = self.model_input_name
            outshape = self.model_input_shape
        else:
            for i in range(len(layer.bottom)):
                for node in self.NodeList:
                    for j in range(len(node.top)):
                        print(layer.bottom[i] , node.top[j])
                        if layer.bottom[i] == node.top[j]:
                            name = node.outputs_name[j]
                            shape = node.outputs_shape[j]
                outname.append(name)
                outshape.append(shape)
        return outname,outshape

    def __getCurrentLayerOutName(self,layer):
        return [layer.name+"_Y"]



    def __getNodeList(self,Layers):
        for i in range(len(Layers)):
            # Convolution
            print (Layers[i].type)
            if Layers[i].type == "Convolution" or Layers[i].type == Layer_CONVOLUTION:
                inname, input_shape = self.__getLastLayerOutNameAndShape(Layers[i])
                outname = self.__getCurrentLayerOutName(Layers[i])
                nodename = Layers[i].name
                print input_shape

                conv_pname = self.__addInputsTVIfromParams(Layers[i],op_pname["Conv"],op_ptype["Conv"])
                inname.extend(conv_pname)


                conv_node = op.createConv(Layers[i],nodename,inname,outname,input_shape)


                self.NodeList.append(conv_node)
                self.__n += 1

            #BatchNorm+Scale
            elif Layers[i].type == "BatchNorm":

                inname,input_shape = self.__getLastLayerOutNameAndShape(Layers[i])
                outname = self.__getCurrentLayerOutName(Layers[i])
                nodename = Layers[i].name


                scale_pname = self.__addInputsTVIfromParams(Layers[i + 1],op_pname["Scale"],op_ptype["Scale"])
                inname.extend(scale_pname)
                bn_pname = self.__addInputsTVIfromParams(Layers[i],op_pname["BatchNorm"],op_ptype["BatchNorm"])
                inname.extend(bn_pname)



                bn_node = op.createBN(Layers[i], nodename, inname, outname, input_shape)


                self.NodeList.append(bn_node)
                self.__n += 1

            #Pooling
            elif Layers[i].type == "Pooling" or Layers[i].type == Layer_POOLING:

                inname,input_shape = self.__getLastLayerOutNameAndShape(Layers[i])
                outname = self.__getCurrentLayerOutName(Layers[i])
                nodename = Layers[i].name


                pool_node = op.createPooling(Layers[i], nodename, inname, outname, input_shape)


                self.NodeList.append(pool_node)
                self.__n += 1


            #Eltwise
            elif Layers[i].type == "Eltwise" or Layers[i].type == Layer_ELTWISE:

                inname,input_shape = self.__getLastLayerOutNameAndShape(Layers[i])
                outname = self.__getCurrentLayerOutName(Layers[i])
                nodename = Layers[i].name


                eltwise_node = op.createEltwise(Layers[i], nodename, inname, outname, input_shape)


                self.NodeList.append(eltwise_node)
                self.__n += 1

            #Softmax
            elif Layers[i].type == "Softmax" or Layers[i].type == Layer_SOFTMAX:

                inname,input_shape = self.__getLastLayerOutNameAndShape(Layers[i])
                outname = self.__getCurrentLayerOutName(Layers[i])
                nodename = Layers[i].name


                softmax_node = op.createSoftmax(Layers[i],nodename, inname, outname, input_shape)


                self.NodeList.append(softmax_node)
                self.__n += 1

            #Relu
            elif Layers[i].type == "ReLU" or Layers[i].type == Layer_RELU:

                inname,input_shape = self.__getLastLayerOutNameAndShape(Layers[i])
                outname = self.__getCurrentLayerOutName(Layers[i])
                nodename = Layers[i].name


                relu_node = op.createRelu(Layers[i], nodename, inname, outname, input_shape)


                self.NodeList.append(relu_node)
                self.__n += 1

            #LRN
            elif Layers[i].type == "LRN" or Layers[i].type == Layer_LRN:

                inname,input_shape = self.__getLastLayerOutNameAndShape(Layers[i])
                outname = self.__getCurrentLayerOutName(Layers[i])
                nodename = Layers[i].name


                LRN_node = op.createLRN(Layers[i],nodename, inname, outname, input_shape)


                self.NodeList.append(LRN_node)
                self.__n += 1

            #Dropout
            elif Layers[i].type == "Dropout" or Layers[i].type == Layer_DROPOUT:

                inname,input_shape = self.__getLastLayerOutNameAndShape(Layers[i])
                outname = self.__getCurrentLayerOutName(Layers[i])
                nodename = Layers[i].name


                Dropout_node = op.createDropout(Layers[i], nodename, inname, outname, input_shape)


                self.NodeList.append(Dropout_node)
                self.__n += 1


            #Upsample
            elif Layers[i].type == "Upsample" or Layers[i].type == Layer_UPSAMPLE:

                inname, input_shape = self.__getLastLayerOutNameAndShape(Layers[i])
                outname = self.__getCurrentLayerOutName(Layers[i])
                nodename = Layers[i].name


                paramshape = [[4, 1]]
                paramdata = [[1.0, 1.0, Layers[i].upsample_param.scale, Layers[i].upsample_param.scale]]
                pname = self.__addInputsTVIfromMannul(Layers[i],op_pname["Upsample"],op_ptype["Upsample"],paramshape,paramdata)
                inname.extend(pname)


                Upsample_node = op.createUpsample(Layers[i], nodename, inname, outname, input_shape)


                self.NodeList.append(Upsample_node)
                self.__n += 1

            #Concat
            elif Layers[i].type == "Concat" or Layers[i].type == Layer_CONCAT:

                inname,input_shape = self.__getLastLayerOutNameAndShape(Layers[i])
                outname = self.__getCurrentLayerOutName(Layers[i])
                nodename = Layers[i].name


                Concat_node = op.createConcat(Layers[i], nodename, inname, outname, input_shape)


                self.NodeList.append(Concat_node)
                self.__n += 1

            #PRelu
            elif Layers[i].type == "PReLU":

                inname,input_shape = self.__getLastLayerOutNameAndShape(Layers[i])
                outname = self.__getCurrentLayerOutName(Layers[i])
                nodename = Layers[i].name


                paramshape = [input_shape[0]]
                paramdata = [0.25 * np.ones(input_shape[0]).reshape(1, -1)[0]]
                pname = self.__addInputsTVIfromMannul(Layers[i],op_pname["PRelu"],op_ptype["PRelu"],paramshape,paramdata)
                inname.extend(pname)



                PRelu_node = op.createPRelu(Layers[i], nodename, inname, outname, input_shape)


                self.NodeList.append(PRelu_node)
                self.__n += 1

            elif Layers[i].type == "Reshape" or Layers[i].type == Layer_INNER_PRODUCT:

                reshape_layer = copy.deepcopy(Layers[i])

                reshape_inname, reshape_input_shape = self.__getLastLayerOutNameAndShape(reshape_layer)
                reshape_outname = [reshape_layer.name + "_Reshape_Y"]
                reshape_nodename = reshape_layer.name+"_Reshape"


                paramshape = [[4]]
                paramdata = op.getReshapeOutShape(Layers[i],reshape_input_shape)
                reshape_pname = self.__addInputsTVIfromMannul(reshape_layer,op_pname["Reshape"],op_ptype["Reshape"],paramshape,paramdata)
                reshape_inname.extend(reshape_pname)


                reshape_node = op.createReshape(reshape_layer,reshape_nodename, reshape_inname, reshape_outname, reshape_input_shape)


                self.NodeList.append(reshape_node)
                self.__n += 1

            elif Layers[i].type == "Crop":

                crop_layer = copy.deepcopy(Layers[i])

                crop_inname, crop_input_shape = self.__getLastLayerOutNameAndShape(crop_layer)
                outname = self.__getCurrentLayerOutName(Layers[i])
                nodename = Layers[i].name

                Concat_node = op.createCrop(Layers[i], nodename, crop_inname, outname, crop_input_shape)

                #self.NodeList.append(Concat_node)
                #self.__n += 1


            # InnerProduct

            elif Layers[i].type == "InnerProduct" or Layers[i].type == Layer_INNER_PRODUCT:

                reshape_layer = copy.deepcopy(Layers[i])

                reshape_inname, reshape_input_shape = self.__getLastLayerOutNameAndShape(reshape_layer)  #
                reshape_outname = [reshape_layer.name + "_Reshape_Y"]
                reshape_nodename = reshape_layer.name+"_Reshape"


                paramshape = [[2]]
                paramdata = op.getReshapeOutShape(Layers[i],reshape_input_shape)
                reshape_pname = self.__addInputsTVIfromMannul(reshape_layer,op_pname["Reshape"],op_ptype["Reshape"],paramshape,paramdata)
                reshape_inname.extend(reshape_pname)


                reshape_node = op.createReshape(reshape_layer,reshape_nodename, reshape_inname, reshape_outname, reshape_input_shape)


                self.NodeList.append(reshape_node)
                self.__n += 1



                gemm_layer = copy.deepcopy(Layers[i])

                gemm_inname = reshape_outname
                gemm_input_shape = self.NodeList[self.__n-1].outputs_shape
                gemm_outname = [gemm_layer.name+"_Gemm_Y"]
                gemm_nodename = gemm_layer.name+"_Gemm"


                gemm_pname = self.__addInputsTVIfromParams(gemm_layer,op_pname["InnerProduct"],op_ptype["InnerProduct"])  #
                gemm_inname.extend(gemm_pname)



                matmul_node = op.createGemm(gemm_layer, gemm_nodename, gemm_inname, gemm_outname, gemm_input_shape, gemm_layer.inner_product_param.num_output)


                self.NodeList.append(matmul_node)
                self.__n += 1

            # Deconvolution
            elif Layers[i].type == "Deconvolution":
                #1.获取节点输入名、输入维度、输出名、节点名
                inname, input_shape = self.__getLastLayerOutNameAndShape(Layers[i])
                outname = self.__getCurrentLayerOutName(Layers[i])
                nodename = Layers[i].name

                #2.生成节点参数tensor value info,并获取节点参数名,将参数名加入节点输入名列表
                conv_pname = self.__addInputsTVIfromParams(Layers[i], op_pname["ConvTranspose"], op_ptype["ConvTranspose"])
                inname.extend(conv_pname)

                #3.构建conv_node
                conv_node = op.createConvTranspose(Layers[i], nodename, inname, outname, input_shape)
                #if True:
                #    self.__print_debug_info(nodename, inname, outname, input_shape, conv_node.outputs_shape)

                #4.添加节点到节点列表
                self.NodeList.append(conv_node)
                self.__n += 1



    def judgeoutput(self,current_node,nodelist):
        for outname in current_node.outputs_name:
            for node in nodelist:
                if outname in node.inputs_name:
                    return False
        return True


    def ___addOutputsTVIandValueInfo(self):
        for i in range(len(self.NodeList)):
            if self.judgeoutput(self.NodeList[i],self.NodeList):
                lastnode = self.NodeList[i]
                for j in range(len(lastnode.outputs_shape)):
                    output_tvi = helper.make_tensor_value_info(lastnode.outputs_name[j], TensorProto.FLOAT,lastnode.outputs_shape[j])
                    self.onnxmodel.addOutputsTVI(output_tvi)
            else:
                innernode = self.NodeList[i]
                for k in range(len(innernode.outputs_shape)):
                    hid_out_tvi = helper.make_tensor_value_info(innernode.outputs_name[k], TensorProto.FLOAT,innernode.outputs_shape[k])
                    self.onnxmodel.addValueInfoTVI(hid_out_tvi)



    def createOnnxModel(self):
        node_def = [Node.node for Node in self.NodeList]
        graph_def = helper.make_graph(
            node_def,
            self.onnxmodel.name,
            self.onnxmodel.in_tvi,
            self.onnxmodel.out_tvi,
            self.onnxmodel.init_t,
            value_info=self.onnxmodel.hidden_out_tvi
        )
        model_def = helper.make_model(graph_def, producer_name='htshinichi')

        return model_def

