from cnocr import CnOcr


def test():
    ocr = CnOcr(rec_model_fp='my_model.onnx')
    res = ocr.ocr('data/0a0d30c348f46086de395e0bd6425493.png')
    print("Predicted Chars:", res)


if __name__ == '__main__':
    test()