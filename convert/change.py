import onnx
from cnocr import CnOcr

# from onnx.onnx_ml_pb2 import Dimention

# model = onnx.load(r'./jp_det.onnx')
# model = onnx.load(r'./ch_pp_jp_rec.onnx')

ocr = CnOcr(det_model_fp=r'./ch_pp_jp_det_1.onnx',
      rec_model_fp=r'./ch_pp_jp_rec.onnx',
      rec_vocab_fp=r'./label_jp.txt',)

sd = ocr.ocr('./122.jpg')
# import cv2
# trans = [x['position'] for x in sd]
# # img = cv2.imread('./122.jpg')
# import numpy as np
# import cv2
# trans = np.array(trans).astype(np.int32)
# img = cv2.imread('./122.jpg')
# # show box of text region on img
# for pr in trans:
#     cv2.rectangle(img, pr[0], pr[2], (0, 255, 0), 2)

# cv2.imshow('img', img)
# cv2.waitKey(0)
print(sd)

# model.graph.input[0].type.tensor_type.shape.dim[2].dim_param = model.graph.input[0].type.tensor_type.shape.dim[3].dim_param
# model.graph.input[0].type.tensor_type.shape.dim[3].dim_param = b'768'
# model.graph.output[0].type.tensor_type.shape.dim[2].dim_param = b'448'
# model.graph.output[0].type.tensor_type.shape.dim[3].dim_param = b'768'
# # print(model.graph.input[0].type.tensor_type.shape)
# print(model.graph.input)
# onnx.save(model, 'ch_pp_jp_det.onnx')