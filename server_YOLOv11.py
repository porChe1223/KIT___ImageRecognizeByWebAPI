# !pip install falcon
# !pip install ultralytics

import falcon
from ultralytics import YOLO
import json
from wsgiref.simple_server import make_server

# サーバを作成
serverIPAddress='127.0.0.1' #サーバIPアドレス
serverPort=60003 #サーバポート番号
serverAPIname='/server_YOLOv11' #サービスAPIの名前

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
# -検出した物体を確率で絞る　　 　　　　　　　　　#
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
# -確率の低い物体は除外　　　　　　　　　　　　   #
# -位置関係から予測　　　　　　　　　　　　　　   #
###############################################
def consider_description(list):
    if any(obj['物体'] == 'person' for obj in list):
        if any(obj['物体'] == 'sports ball' for obj in list):
            result_sentence = 'ボールで遊んでいます。'
        elif any(obj['物体'] == 'teddy bear' for obj in list):
            result_sentence = 'テディベアが人といます。'
        elif any(obj['物体'] == 'car' for obj in list):
            result_sentence = '人が車に乗っています。'
        else:
            result_sentence = '人がいます。'
    else:
        if any(obj['物体'] == 'teddy bear' for obj in list):
            if any(obj['物体'] == 'chair' for obj in list):
                result_sentence = 'テディベアが椅子に座っています。'
            else:
                result_sentence = 'テディベアが置かれています。'
        else:
            result_sentence = '人がいません。'
    
    return result_sentence


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

# リソース【/Sample3App】 と　AppResource()を結びつける
app.add_route(serverAPIname, ImageRecognize())

if __name__ == '__main__':  #main処理
    with make_server('', serverPort, app) as httpd:
        print('- アクセスURL ->  http://{}:{}{}'.format(serverIPAddress, serverPort,serverAPIname))
    
        httpd.serve_forever() #ずっとサーバを起動しておく

