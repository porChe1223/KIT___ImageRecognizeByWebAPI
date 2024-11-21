import os
import requests

# サーバIPアドレス
server_url = 'http://127.0.0.1:60003/server_ImageRecognize'

while(1):
    # 画像パスを指定
    base_dir = '/home/porche1223/3-3___WebAPI画像処理システム/ImageRecognizeByWebAPI/WebAPI_Dataset/'
    image = input('画像パスを入力してください(exitで退出)： ')
    if image == 'exit':
        break
    image_path = os.path.join(base_dir, image)

    # 送信データ
    post_data = {
        'image': image_path
    }

    # 画像パス例
    # kicking_ball.jpeg
    # holding_ball.jpg
    # throwing_ball.jpg
    # person_table_chair.jpg
    # person_chair.jpg
    # standing_near_car.jpg
    # sitting_near_car.jpg
    # riding_car.jpg

    # POSTリクエストの送信
    response = requests.post(server_url, json=post_data).json()

    print('検出結果： ', response['検出結果'])
    print('予測結果： ', response['分析結果'])
    print('\n')