import os
import base64
import requests

# サーバIPアドレス
server_url = 'http://127.0.0.1:60003/server_ImageRecognize'

while(1):
    # 画像パスを指定
    base_dir = '/home/porche1223/KIT___WebAPI画像認識システム/KIT___ImageRecognize_WebAPI/WebAPI_Dataset/'
    image = input('画像パスを入力してください(exitで退出)： ')
    if image == 'exit':
        break
    image_path = os.path.join(base_dir, image)

    with open(image_path, 'rb') as img_file:
        image_base64 = base64.b64encode(img_file.read()).decode('utf-8')

    # 送信データ
    post_data = {
        'image': image_base64
    }

    # 画像パス例
    # kicking_ball.jpeg
    # holding_ball.jpg
    # throwing_ball.jpg
    # person_chair.jpg
    # person_table_chair.jpg
    # standing_near_car.jpg
    # sitting_near_car.jpg
    # riding_car.jpg

    # POSTリクエストの送信
    response = requests.post(server_url, json=post_data).json()

    print('検出結果： ', response['検出結果'])
    print('予測結果： ', response['分析結果'])
    print('\n')