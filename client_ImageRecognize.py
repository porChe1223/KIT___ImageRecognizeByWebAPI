import os
import base64
import requests

# サーバIPアドレス
server_url = 'http://127.0.0.1:60003/server_ImageRecognize'

# ベースディレクトリ
base_dir = '/home/porche1223/KIT___WebAPI画像認識システム/KIT___ImageRecognize_WebAPI/WebAPI_Dataset/'
model_dir = '/home/porche1223/KIT___WebAPI画像認識システム/KIT___ImageRecognize_WebAPI/'


while True:
    #################
    # リクエスト送信 #
    #################

    # モード選択
    mode = input('モードを選択してください (R: 画像認識, M: MyModelで画像認識, F: ファインチューニング, X: 終了)： ')
    if mode == 'X':
        break


    # 画像認識
    if mode == 'R':
        # 画像パスを指定
        image = input('画像パスを入力してください(X: 終了)： ')
        if image == 'X':
            break

        image_path = os.path.join(base_dir, image)
        with open(image_path, 'rb') as img_file:
            image_base64 = base64.b64encode(img_file.read()).decode('utf-8')

        # 送信データ
        post_data = {
            'mode': 'R',
            'image': image_base64
        }


    # 自分のモデルで画像認識
    elif mode == 'M':
        # モデルパスを指定
        model = input('モデルを入力してください(X: 終了)： ')
        if model == 'X':
            break

        model_path = os.path.join(model_dir, model)

        # 画像パスを指定
        image = input('画像パスを入力してください(X: 終了)： ')
        if image == 'X':
            break

        image_path = os.path.join(base_dir, image)
        with open(image_path, 'rb') as img_file:
            image_base64 = base64.b64encode(img_file.read()).decode('utf-8')

        # 送信データ
        post_data = {
            'mode': 'M',
            'model': model_path,
            'image': image_base64
        }


    # ファインチューニング
    elif mode == 'F':
        # ディレクトリパスを指定
        dir_path = input('ディレクトリパスを入力してください(X: 終了)： ')
        if dir_path == 'X':
            break
        dir_path = os.path.join(base_dir, dir_path)

        images_base64 = []
        for root, dirs, files in os.walk(dir_path):
            for file in files:
                file_path = os.path.join(root, file)
                with open(file_path, 'rb') as img_file:
                    image_base64 = base64.b64encode(img_file.read()).decode('utf-8')
                    images_base64.append(image_base64)

        # 送信データ
        post_data = {
            'mode': 'F',
            'images': images_base64
        }

    else:
        print('無効な選択です。もう一度選択してください。')
        continue

    # POSTリクエストの送信
    response = requests.post(server_url, json=post_data)
    print(response.json())

    #################
    # レスポンス受信 #
    #################

    if mode == 'R' or mode == 'M':
        print('\n')
        print('検出結果： ', response.json()['検出結果'])
        print('予測結果： ', response.json()['分析結果'])
    
    elif mode == 'F':
        print('\n')
        print('ファインチューニングされたモデルがダウンロードされました。')
        print('サーバ上でのあなたのモデルの保存場所： ', response.json()['new_model'])
        print('このパスはあなたのモデルを動かすために必要です。')
        print('必ず保存しておいてください。')

    print('\n')
    print('\n')