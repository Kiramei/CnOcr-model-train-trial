import onnx
path='./jp_det.onnx'
model = onnx.load(path)
model.ir_version = 7
onnx.save(model, path)

path='./jp_det.onnx'
model = onnx.load(path)
model.ir_version = 7
onnx.save(model, path)
