from google.protobuf import text_format
from proto import caffe_upsample_pb2
import onnx

def loadcaffemodel(net_path,model_path):
    # read prototxt
    net = caffe_upsample_pb2.NetParameter()
    text_format.Merge(open(net_path).read(), net)
    # read caffemodel
    model = caffe_upsample_pb2.NetParameter()
    f = open(model_path, 'rb')
    model.ParseFromString(f.read())
    f.close()
    return net,model

def loadonnxmodel(onnx_path):
    onnxmodel = onnx.load(onnx_path)
    return onnxmodel

def saveonnxmodel(onnxmodel,onnx_save_path):
    try:
        onnx.checker.check_model(onnxmodel)
        onnx.save_model(onnxmodel, onnx_save_path+".onnx")
    except Exception as e:
        pass
