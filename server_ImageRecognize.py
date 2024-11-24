# !pip install falcon
# !pip install ultralytics

import falcon
import base64
from io import BytesIO
from PIL import Image
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
        if obj['確率'] >= 0.65:
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
# 分析条件　　　　　　　　　　　　　　　　　　    #
# -バウンディングボックスの中心座標を計算　　　　 #
# -ボックス間の距離を計算　　　　　　　　　　　　 #
# -ボックス間の頂点の差を計算　　　　　　　　　　 #
# -ボックス間の底の差を計算　　　　　　　　　　　 #
# -ボックスが重なっているかどうかを判定　　　　　 #
###############################################
def get_center(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2, (y1 + y2) / 2)

def get_distance(box1, box2):
    center1 = get_center(box1)
    center2 = get_center(box2)
    return math.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)

def get_top_distance(box1, box2):
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    return y1_1 - y1_2

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


###############################################
# 分析結果　　　　　　　　　　　　　　　　　　    #
# -位置関係から予測　　　　　　　　　　　　　　   #
# -判定ルールの定義(辞書を格納したリスト)　　　　 #
# -ルールを適用して判定　　　　　　　　　　　 　  #
###############################################
def consider_description(list):
    rules = [
        {
            "対象物": ("person", "sports ball"),
            "判定": lambda obj1, obj2: get_buttom_distance(obj1['境界'].values(), obj2['境界'].values()) <= 50,
            "結果": "人がボールを蹴っています。"
        },
        {
            "対象物": ("person", "sports ball"),
            "判定": lambda obj1, obj2: is_overlapping(obj1['境界'].values(), obj2['境界'].values(), 0.8),
            "結果": "人がボールを持っています。"
        },
        {
            "対象物": ("person", "sports ball"),
            "判定": lambda obj1, obj2: True,
            "結果": "人がボールを投げています。"
        },
        {
            "対象物": ("chair", "person"),
            "判定": lambda obj1, obj2: is_overlapping(obj1['境界'].values(), obj2['境界'].values(), 0.8),
            "追加判定": lambda obj2, list: any(
                obj3['物体'] == 'dining table' and is_overlapping(obj3['境界'].values(), obj2['境界'].values(), 0.15)
                for obj3 in list
            ),
            "結果": ["人が席についています。", "人が椅子に座っています。"]
        },
        {
            "対象物": ("car", "person"),
            "判定": lambda obj1, obj2: get_top_distance(obj1['境界'].values(), obj2['境界'].values()) >= 50,
            "結果": "人が車のそばに立っています。"
        },
        {
            "対象物": ("car", "person"),
            "判定": lambda obj1, obj2: is_overlapping(obj1['境界'].values(), obj2['境界'].values(), 0.9),
            "結果": "人が車に乗っています。"
        },
        {
            "対象物": ("car", "person"),
            "判定": lambda obj1, obj2: True,
            "結果": "人が車のそばに座っています。"
        }
    ]

    for obj1 in list:
        for obj2 in list:
            if obj1 == obj2:
                continue

            for rule in rules:
                if (obj1['物体'], obj2['物体']) == rule["対象物"]:
                    if rule["判定"](obj1, obj2):
                        if "追加判定" in rule:
                            if rule["追加判定"](obj2, list):
                                return rule["結果"][0]
                            else:
                                return rule["結果"][1]
                        return rule["結果"]

    return "分析に必要な検出結果が不十分です。他の画像を指定してください。"


############################################
# 画像認識システム（YOLOv11）              　#
############################################
class ImageRecognize(object):
    def on_post(self, req, res):
        params = req.media
        image_base64 = params.get('image')  # クライアントから送られてきたbase64エンコードされた画像

        # base64エンコードされた画像をデコードしてPIL画像に変換
        image_data = base64.b64decode(image_base64)
        image = Image.open(BytesIO(image_data))

        if image:            
            # 一時ファイルに保存
            image.save('received_image.jpg')

            # YOLOで画像を処理
            results = model('received_image.jpg', save=True, show=True)
            print('いいいい',results)
            
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
                print('ああああ',detected_objects)

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
            print('結果',res)
        else:
            res.status = falcon.HTTP_400
            res.media = {'error': '画像が指定されていません。'}


# リソース【/server_YOLOv11】 と　AppResource()を結びつける
app.add_route(serverAPIname, ImageRecognize())

if __name__ == '__main__':  #main処理
    with make_server('', serverPort, app) as httpd:
        print('- アクセスURL ->  http://{}:{}{}'.format(serverIPAddress, serverPort,serverAPIname))
    
        httpd.serve_forever() #ずっとサーバを起動しておく

