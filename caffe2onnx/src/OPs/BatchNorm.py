import src.c2oObject as Node


def getBNAttri(layer):

    dict = {"epsilon": 0.00001,
            "momentum": 0.9
            }
    return dict

def getBNOutShape(input_shape):
    output_shape = input_shape
    return output_shape

def createBN(layer, nodename, inname, outname, input_shape):
    dict = getBNAttri(layer)

    output_shape = getBNOutShape(input_shape)


    node = Node.c2oNode(layer, nodename, "BatchNormalization", inname, outname, input_shape, output_shape,dict)

    return node
