# !pip install falcon
# !pip install ultralytics

import falcon
from ultralytics import YOLO
import json
import math
from wsgiref.simple_server import make_server

# サーバを作成
serverIPAddress='127.0.0.1' #サーバIPアドレス
serverPort=60003 #サーバポート番号
serverAPIname='/server_ImageRecognize' #サービスAPIの名前

#falconAPIの初期化
app = falcon.App()

# モデルの選択
print('モデル呼び出し中')
model = YOLO('yolo11n.pt')
print('モデル呼び出し完了')

# # モデルのトレーニング
# print('モデルのトレーニング中')
# res = model.train(data='coco8.yaml', epochs=100, imgsz=640)
# print('モデルのトレーニング完了')

###############################################
# 検出結果　　　　　　　　　　　　　　　　　　    #
# -検出した物体の位置・確率・名前を文にして出力　 #
###############################################
def generate_description(list):
    if not list:
        result_sentence = 'この写真には何も写っていません。'
    
    descriptions = []
    for obj in list:
        place = obj['境界']
        confidence = obj['確率']
        label = obj['物体']

        descriptions.append(f'{place}の位置に')
        descriptions.append(f'{confidence}の確率で')
        descriptions.append(f'{label}が存在。')
    
    # 認識したものの列挙
    result_sentence = ''.join(descriptions)
    return result_sentence


###############################################
# 厳選結果　　　　　　　　　　　　　　　　　　　  #
# -確率の低い物体は除外　　　　 　　　　　　　　　#
###############################################
def select_objects(list):
    select_objects = []
    for obj in list:
        if obj['確率'] >= 0.7:
            select_objects.append(obj)

    return select_objects


###############################################
# 厳選結果表示　　　　　　　　　　　　　　　　　  #
# -検出した物体を文にして出力　 　　　　　　　　　#
###############################################
def select_objects_sentence(list):
    if not list:
        result_sentence = 'この写真には何も有力なものは写っていません。'
    
    descriptions = []
    for obj in list:
        place = obj['境界']
        confidence = obj['確率']
        label = obj['物体']

        descriptions.append(f'{place}の位置に')
        descriptions.append(f'{confidence}の確率で')
        descriptions.append(f'{label}が存在。')

    # 厳選したものの列挙
    result_sentence = '有力なものの候補として、' + ''.join(descriptions)
    return result_sentence


###############################################
# 分析結果　　　　　　　　　　　　　　　　　　    #
# -バウンディングボックスの中心座標を計算　　　　 #
# -2つのボックス間の距離を計算　　　　　　　　　　#
# -2つのボックスが重なっているかどうかを判定　　　#
# -位置関係から予測　　　　　　　　　　　　　　   #
###############################################
def get_center(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2, (y1 + y2) / 2)

def get_distance(box1, box2):
    center1 = get_center(box1)
    center2 = get_center(box2)
    return math.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)

def get_buttom_distance(box1, box2):
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    return math.sqrt((y2_1 - y2_2)**2)

def is_overlapping(box1, box2, percent):
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2

    # 重なり領域の座標
    overlap_x1 = max(x1_1, x1_2)
    overlap_y1 = max(y1_1, y1_2)
    overlap_x2 = min(x2_1, x2_2)
    overlap_y2 = min(y2_1, y2_2)

    # 重なり領域の幅と高さ
    overlap_width = max(0, overlap_x2 - overlap_x1)
    overlap_height = max(0, overlap_y2 - overlap_y1)

    # 重なり領域の面積
    overlap_area = overlap_width * overlap_height

    # 各ボックスの面積
    box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
    box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)

    # 重なりの割合
    smallest_area = min(box1_area, box2_area)
    overlap_ratio = overlap_area / smallest_area if smallest_area > 0 else 0

    return overlap_ratio >= percent

def consider_description(list):
    for obj1 in list:
        for obj2 in list:
            if obj1 == obj2:
                continue

            # サッカーをしている
            if obj1['物体'] == 'person' and obj2['物体'] == 'sports ball':
                buttom_distance = get_buttom_distance(obj1['境界'].values(), obj2['境界'].values())
                if buttom_distance <= 50:
                    return '人がボールを蹴っています。'
                else:
                    if is_overlapping(obj1['境界'].values(), obj2['境界'].values(), 0.8):
                        return '人がボールを持っています。'
                    else:
                        return '人がボールを投げています。'
                
            # 椅子または机に座っている
            if obj1['物体'] == 'chair' and obj2['物体'] == 'person':
                if is_overlapping(obj1['境界'].values(), obj2['境界'].values(), 0.8):
                    for obj3 in list:
                        if obj3['物体'] == 'dining table':
                            if is_overlapping(obj3['境界'].values(), obj2['境界'].values(), 0.15):
                                return '人が席についています。'
                    return '人が椅子に座っています。'

            # 車に乗っている
            if obj1['物体'] == 'person' and obj2['物体'] == 'car':
                if is_overlapping(obj1['境界'].values(), obj2['境界'].values()):
                    return '人が車に乗っています。'

    return '特に目立ったアクティビティは検出されませんでした。'


############################################
# 画像認識システム（YOLOv11）              　#
############################################
class ImageRecognize(object):
    def on_post(self, req, res):
        # 送られてきた画像パス名を取得
        params = req.media
        print('受信したJSONデータ', params)

        image_url = params.get('image')
        print('受信した画像パス', image_url)

        if(image_url):
            # 画像内のオブジェクトを検出
            results = model(image_url, save=True, show=True)
            print('YOLOの返答', results)

            #resultsオブジェクトの情報をリスト形式で取得
            detected_objects = []
            for box in results[0].boxes.data.tolist():
                label_index = int(box[5]) if len(box) > 5 else -1
                label_name = results[0].names[label_index] if label_index in results[0].names else '不明'
                label_confidence = box[4]
                x1, y1, x2, y2 = box[:4]

                # 検出物をリストに追加
                detected_objects.append({
                    '物体': label_name,
                    '確率': label_confidence,
                    '境界': {'左端': x1, '上端': y1, '右端': x2, '下端': y2}
                })
            print(detected_objects)

            # 検出結果
            res_description = generate_description(detected_objects)

            # 厳選結果
            selected_objects = select_objects(detected_objects)
            res_select = select_objects_sentence(selected_objects)

            # 分析結果
            res_consider = consider_description(selected_objects)

            # レスポンスデータの作成
            res.media = {
                '検出結果': res_description,
                '厳選結果': res_select,
                '分析結果': res_consider
            }
        else:
            res.status = falcon.HTTP_400
            res.media = {'error': '画像が指定されていません。'}


# リソース【/server_YOLOv11】 と　AppResource()を結びつける
app.add_route(serverAPIname, ImageRecognize())

if __name__ == '__main__':  #main処理
    with make_server('', serverPort, app) as httpd:
        print('- アクセスURL ->  http://{}:{}{}'.format(serverIPAddress, serverPort,serverAPIname))
    
        httpd.serve_forever() #ずっとサーバを起動しておく

