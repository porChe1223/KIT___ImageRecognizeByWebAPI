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
model = YOLO("yolo11n.pt")
print('モデル呼び出し完了')

# # モデルのトレーニング
# print('モデルのトレーニング中')
# res = model.train(data="coco8.yaml", epochs=100, imgsz=640)
# print('モデルのトレーニング完了')

# 認識結果の文章化
def generate_description(detected_objects):
    if not detected_objects:
        return 'この写真には何も写っていません。'
    
    descriptions = []
    for obj in detected_objects:
        count = obj["count"]
        label = obj["label"]
        
        # 単数形・複数形の処理
        if count == 1:
            descriptions.append(f"1 {label}")
        else:
            descriptions.append(f"{count} {label}s")
    
    # 文を生成（例: "4 persons, 1 car, and 1 sports ball are in the image."）
    result_sentence = ", ".join(descriptions[:-1]) + f", and {descriptions[-1]} are in the image."
    return result_sentence


# 画像認識システム（YOLOv11）
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

            #resultsオブジェクトをJSON型に変換
            detected_objects = {}
            for box in results[0].boxes.data.tolist():
                label_index = int(box[5]) if len(box) > 5 else -1
                label_name = results[0].names[label_index] if label_index in results[0].names else "Unknown"
                
                # 検出数をカウント
                if label_name in detected_objects:
                    detected_objects[label_name] += 1
                else:
                    detected_objects[label_name] = 1

            # 検出内容を文章化
            detected_list = [{"label": label, "count": count} for label, count in detected_objects.items()]
            response_text = generate_description(detected_list)

            # レスポンスデータの作成
            res.media = {
                'YOLOの返答': detected_list,
                'description': response_text
            }
        else:
            res.status = falcon.HTTP_400
            res.media = {'error': '画像が指定されていません。'}

# リソース【/Sample3App】 と　AppResource()を結びつける
app.add_route(serverAPIname, ImageRecognize())

if __name__ == "__main__":  #main処理
    with make_server('', serverPort, app) as httpd:
        print('- アクセスURL ->  http://{}:{}{}'.format(serverIPAddress, serverPort,serverAPIname))
    
        httpd.serve_forever() #ずっとサーバを起動しておく

