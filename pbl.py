import os
import cv2
import time
import torch
import argparse
from pathlib import Path
from numpy import random
from random import randint
import torch.backends.cudnn as cudnn
import pygame
#import neopixel
#import board

import sys
from PyQt5.QtWidgets import *
from PyQt5 import uic
from PyQt5.QtGui import *
form_class = uic.loadUiType("pbl.ui")[0]


from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, \
                check_imshow, non_max_suppression, apply_classifier, \
                scale_coords, xyxy2xywh, strip_optimizer, set_logging, \
                increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, \
                time_synchronized, TracedModel
from utils.download_weights import download
from datetime import datetime
#For SORT tracking
import skimage
from sort import *
#pixels = neopixel.NeoPixel(board.D18, 25)
cnt = [0] * 100
idx=0
stop_detect = np.zeros((640,480))

detected_garbage_time = 0
detected_smoking_time = 0
garbage_warning_time = -30000
print_time = 0

def check(x,y,i):
    cnt[i%100]+=1
    if cnt[i%100]>10:
        stop_detect[x-5 if x-5>0 else 0 : x+5 if x+5<640 else 639, y-5 if y-5>0 else 0 : y+5 if y+5<640 else 479 ] = 1
        cnt[i%100]=0
        return 1
    return 0

def Check_Garbage_Warning_Time():
    global garbage_warning_time
    now_Time = pygame.time.get_ticks()
    if now_Time - garbage_warning_time > 10000:
        return 1
    return 0
    #............................... Bounding Boxes Drawing ............................

class WindowClass(QMainWindow, form_class) :
    def __init__(self) :
        super().__init__()
        self.setupUi(self)
        self.pushButton.clicked.connect(self.button1Function)
        #detect()


    def button1Function(self) : 
        with torch.no_grad():
            if opt.update:  # update all models (to fix SourceChangeWarning)
                for opt.weights in ['yolov7.pt']:
                    detect(self)
                    strip_optimizer(opt.weights)
            else:
                detect(self)

\
"""Function to Draw Bounding boxes"""
def draw_boxes(img, bbox, identities=None, categories=None, names=None, save_with_object_id=False, path=None,offset=(0, 0)):
    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(i) for i in box]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]
        cat = int(categories[i]) if categories is not None else 0
        id = int(identities[i]) if identities is not None else 0
        data = (int((box[0]+box[2])/2),(int((box[1]+box[3])/2)))
        label = str(id) + ":"+ names[cat]
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(img, (x1, y1), (x2, y2), (255,0,20), 2)
        cv2.rectangle(img, (x1, y1 - 20), (x1 + w, y1), (255,144,30), -1)
        cv2.putText(img, label, (x1, y1 - 5),cv2.FONT_HERSHEY_SIMPLEX, 
                    0.6, [255, 255, 255], 1)
        # cv2.circle(img, data, 6, color,-1)   #centroid of box
        txt_str = ""
        if save_with_object_id:
            txt_str += "%i %i %f %f %f %f %f %f" % (
                id, cat, int(box[0])/img.shape[1], int(box[1])/img.shape[0] , int(box[2])/img.shape[1], int(box[3])/img.shape[0] ,int(box[0] + (box[2] * 0.5))/img.shape[1] ,
                int(box[1] + (
                    box[3]* 0.5))/img.shape[0])
            txt_str += "\n"
            with open(path + '.txt', 'a') as f:
                f.write(txt_str)
    return img
#..............................................................................


def detect(self,save_img=False):

    source, weights, view_img, save_txt, imgsz, trace, colored_trk, save_bbox_dim, save_with_object_id= opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, not opt.no_trace, opt.colored_trk, opt.save_bbox_dim, opt.save_with_object_id
    save_img = not opt.nosave and not source.endswith('.txt')  # save inference images
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))

    global print_time
#------------------------------------

    pygame.init()
    smk= pygame.mixer.Sound('./No_Smoking.mp3')
    GarbageDetect = pygame.mixer.Sound('./GarbageDetect.mp3')
    GarbageWarning = pygame.mixer.Sound('./GarbageWarning.mp3')
    global tracked_dets
    global detected_garbage_time
    global detected_smoking_time
    global garbage_warning_time
    global now
    throw_flag=0
    smoking_flag=0
    smoking_cnt=0
#--------------------------------------

    #.... Initialize SORT .... 
    #......................... 
    sort_max_age = 20 
    sort_min_hits = 2
    sort_iou_thresh = 0.1
    sort_tracker = Sort(max_age=sort_max_age,
                       min_hits=sort_min_hits,
                       iou_threshold=sort_iou_thresh)
    #......................... 
    
    
    #........Rand Color for every trk.......
    rand_color_list = []
    for i in range(0,5005):
        r = randint(0, 255)
        g = randint(0, 255)
        b = randint(0, 255)
        rand_color = (r, g, b)
        rand_color_list.append(rand_color)
    #......................................
   

    # Directories
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    (save_dir / 'labels' if save_txt or save_with_object_id else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size

    if trace:
        model = TracedModel(model, device, opt.img_size)

    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    old_img_w = old_img_h = imgsz
    old_img_b = 1

    t0 = time.time()

    
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Warmup
        if device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
            old_img_b = img.shape[0]
            old_img_h = img.shape[2]
            old_img_w = img.shape[3]
            for i in range(3):
                model(img, augment=opt.augment)[0]

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=opt.augment)[0]
        t2 = time_synchronized()

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t3 = time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            #---------------------
            tt=[]
            #---------------------

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                #..................USE TRACK FUNCTION....................
                #pass an empty array to sort
                dets_to_sort = np.empty((0,6))
                
                # NOTE: We send in detected object class too
                for x1,y1,x2,y2,conf,detclass in det.cpu().detach().numpy():
                    dets_to_sort = np.vstack((dets_to_sort, 
                                np.array([x1, y1, x2, y2, conf, detclass])))
                
                # Run SORT
                tracked_dets = sort_tracker.update(dets_to_sort)
                tracks =sort_tracker.getTrackers()
                

                txt_str = ""
                #time for save img
                now = datetime.now().strftime("%d_%H-%M-%S")
                #loop over tracks
                for track in tracks:
                    # color = compute_color_for_labels(id)
                    #draw colored tracks
                    if colored_trk:
                        [cv2.line(im0, (int(track.centroidarr[i][0]),
                                    int(track.centroidarr[i][1])), 
                                    (int(track.centroidarr[i+1][0]),
                                    int(track.centroidarr[i+1][1])),
                                    rand_color_list[track.id], thickness=2) 
                                    for i,_ in  enumerate(track.centroidarr) 
                                      if i < len(track.centroidarr)-1 ] 
                    #draw same color tracks
                    else:
                        for i,_ in  enumerate(track.centroidarr):
                            if i < len(track.centroidarr)-1:
                                if len(tracked_dets)>0 :
                                    if names[int(tracked_dets[0,4])]=='garbage_bag' and int(tracked_dets[0,-1])-1 == track.id:  # 쓰레기만 객체 추적하도록
                                        cv2.line(im0, (int(track.centroidarr[i][0]),
                                            int(track.centroidarr[i][1])), 
                                            (int(track.centroidarr[i+1][0]),
                                            int(track.centroidarr[i+1][1])),
                                            (255,0,0), thickness=2)
                            elif i == len(track.centroidarr)-1 :
                                if len(tracked_dets)>0 :
                                    # 감지된 개체를 리스트에 저장
                                    tt.append(int(tracked_dets[0,4]))
                                    # 감지된 개체가 정지되어 있음을 확인
                                    if names[int(tracked_dets[0,4])]=='garbage_bag' and int(tracked_dets[0,-1])-1 == track.id:
                                        detected_garbage_time = pygame.time.get_ticks()
                                        # 쓰레기가 감지되고, 그것이 버려진 쓰레기가 아니라면
                                        if pygame.mixer.Channel(0).get_busy() == False and stop_detect[int(track.centroidarr[i][0])][int(track.centroidarr[i][1])] != 1:
                                            if Check_Garbage_Warning_Time() == True:
                                                garbage_warning_time = pygame.time.get_ticks()
                                                GarbageWarning.play()
                                        # 감지된 개체가 정지되어 있고, 그것이 쓰레기 봉투 이면
                                        if  (abs(int(track.centroidarr[i][0])-int(track.centroidarr[i-1][0])))<5 and (abs(int(track.centroidarr[i][1])-int(track.centroidarr[i-1][1])))<5 and stop_detect[int(track.centroidarr[i][0])][int(track.centroidarr[i][1])] != 1 :
                                            # check 함수로 진행
                                            if check(int(track.centroidarr[i][0]),int(track.centroidarr[i][1]),track.id) == 1:
                                                print(f'{i} is stopped')
                                                print(f'{i} is stopped')
                                                print(f'{i} is stopped')
                                                #---------------------------------
                                                self.qPixmapVar = QPixmap('siren2.png')
                                                self.qPixmapVar = self.qPixmapVar.scaled(200,200)
                                                self.label.setPixmap(self.qPixmapVar)
                                                self.qPixmapVar2 = QPixmap('trash.png')
                                                self.qPixmapVar2 = self.qPixmapVar2.scaled(200,200)
                                                self.label_2.setPixmap(self.qPixmapVar2)
                                                print_time = pygame.time.get_ticks()
                                                #------------------------------------------
                                                # 쓰레기 버려졌을경우 소리 출력
                                                pygame.mixer.stop()
                                                if pygame.mixer.Channel(0).get_busy() == False:
                                                    GarbageDetect.play()
                                            else:
                                                cv2.imwrite(f'./garbage_img/{str(now)}_garbage.jpg',im0)
                                                
                                    
                                    # 담배 개체를 확인 
                                    if names[int(tracked_dets[0,4])]=='smoking_hand' and int(tracked_dets[0,-1])-1 ==track.id:
                                        # 담배 개체가 탐지될때마다 cnt 증가
                                        smoking_cnt+=1
                                        if smoking_cnt >= 5 :
                                            smoking_flag = 1
                                            detected_smoking_time = pygame.time.get_ticks()
                                            print(f'{smoking_cnt}      smoking detected')
                                            print(f'{smoking_cnt}      smoking detected')
                                            print(f'{smoking_cnt}      smoking detected')
                                            #-----------------------------------------------
                                            self.qPixmapVar = QPixmap('siren2.png')
                                            self.qPixmapVar = self.qPixmapVar.scaled(200,200)
                                            self.label.setPixmap(self.qPixmapVar)
                                            self.qPixmapVar2 = QPixmap('ciggar.jpg')
                                            self.qPixmapVar2 = self.qPixmapVar2.scaled(200,200)
                                            self.label_2.setPixmap(self.qPixmapVar2)
                                            print_time = pygame.time.get_ticks()
                                            #--------------------------------------------------
                                            # 담배개체 탐지된경우 소리출력
                                            cv2.imwrite(f'./smoking_img/{str(now)}_smoking.jpg',im0)
                                            if pygame.mixer.Channel(0).get_busy() == False:
                                                smk.play()




                    if save_txt and not save_with_object_id:
                        # Normalize coordinates
                        txt_str += "%i %i %f %f" % (track.id, track.detclass, track.centroidarr[-1][0] / im0.shape[1], track.centroidarr[-1][1] / im0.shape[0])
                        if save_bbox_dim:
                            txt_str += " %f %f" % (np.abs(track.bbox_history[-1][0] - track.bbox_history[-1][2]) / im0.shape[0], np.abs(track.bbox_history[-1][1] - track.bbox_history[-1][3]) / im0.shape[1])
                        txt_str += "\n"
                
                if save_txt and not save_with_object_id:
                    with open(txt_path + '.txt', 'a') as f:
                        f.write(txt_str)

                # draw boxes for visualization
                if len(tracked_dets)>0:
                    bbox_xyxy = tracked_dets[:,:4]
                    identities = tracked_dets[:, 8]
                    categories = tracked_dets[:, 4]
                    draw_boxes(im0, bbox_xyxy, identities, categories, names, save_with_object_id, txt_path)
                #........................................................

            # 감지된 개체에 담배가 없으면 cnt 초기화
            
            if 0 not in tt:
                smoking_cnt=0    

            if print_time !=0 and pygame.time.get_ticks() - print_time>3000:
                self.label.clear()
                self.label_2.clear()
                print_time=0
            # Print time (inference + NMS)
            print(f'{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')
            
        
            # Stream results
            if view_img:
                cv2.imshow(str(p), im0)
                if cv2.waitKey(1) == ord('q'):  # q to quit
                  cv2.destroyAllWindows()
                  raise StopIteration

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                    print(f" The image with the result is saved in: {save_path}")
                else:  # 'video' or 'stream'
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer.write(im0)

    if save_txt or save_img or save_with_object_id:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        #print(f"Results saved to {save_dir}{s}")

    print(f'Done. ({time.time() - t0:.3f}s)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov7.pt', help='model.pt path(s)')
    parser.add_argument('--download', action='store_true', help='download model weights automatically')
    parser.add_argument('--no-download', dest='download', action='store_false',help='not download model weights if already exist')
    parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='object_tracking', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--no-trace', action='store_true', help='don`t trace model')
    parser.add_argument('--colored-trk', action='store_true', help='assign different color to every track')
    parser.add_argument('--save-bbox-dim', action='store_true', help='save bounding box dimensions with --save-txt tracks')
    parser.add_argument('--save-with-object-id', action='store_true', help='save results with object id to *.txt')

    parser.set_defaults(download=True)
    opt = parser.parse_args()
    print(opt)
    #check_requirements(exclude=('pycocotools', 'thop'))
    if opt.download and not os.path.exists(str(opt.weights)):
        print('Model weights not found. Attempting to download now...')
        download('./')

    # with torch.no_grad():
    #     if opt.update:  # update all models (to fix SourceChangeWarning)
    #         for opt.weights in ['yolov7.pt']:
    #             detect()
    #             strip_optimizer(opt.weights)
    #     else:
    #         detect()

    #QApplication : 프로그램을 실행시켜주는 클래스
    app = QApplication(sys.argv) 

    #WindowClass의 인스턴스 생성
    myWindow = WindowClass() 

    #프로그램 화면을 보여주는 코드
    myWindow.show()

    #프로그램을 이벤트루프로 진입시키는(프로그램을 작동시키는) 코드
    app.exec_()
