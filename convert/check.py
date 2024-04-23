import onnx,time
import numpy as np
import onnxruntime
 
onnx_file = './ch_pp_jp_rec.onnx' 
onnx_model = onnx.load(onnx_file)
onnx.checker.check_model(onnx_model)
print('The model is checked!')
 
x = np.random.random((1,3,48,320)).astype('float32') # 此处也可用cv读取某张图片作为输入
print("x:",x)
 
ort_sess = onnxruntime.InferenceSession(onnx_file)
ort_inputs = {ort_sess.get_inputs()[0].name: x}
start = time.time()
ort_outs = ort_sess.run(None, ort_inputs)
end = time.time()
print("Runtime:{}".format(end-start))
