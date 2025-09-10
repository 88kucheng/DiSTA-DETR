import warnings, os
# os.environ["CUDA_VISIBLE_DEVICES"]="-1"
# os.environ["CUDA_VISIBLE_DEVICES"]="0"
warnings.filterwarnings('ignore')
from ultralytics import RTDETR



if __name__ == '__main__':
    model = RTDETR(r'D:\Development\code\DiSTA-DETR\RTDETR-main\model\DiSTA-DETR.yaml')
    # model.load('') # loading pretrain weights
    model.train(data='D:\Development\code\DiSTA-DETR\RTDETR-main\datasets\hit-uav\dataset.yaml',
                cache=False,
                imgsz=640,
                epochs=200,
                batch=4, #
                workers=0, #
                device='0', #
                # resume='', # last.pt path
                project='runs/train',
                name='exp',
                )