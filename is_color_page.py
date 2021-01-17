import sklearn
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import pickle
import cv2
import numpy as np

class IsColorPageSVM:
    def __init__(self, kernel='linear', random_state=None):
        self.svm = SVC(kernel=kernel, random_state=random_state)
        self.scaler =  StandardScaler() 

    # 参考 : https://qiita.com/kazuki_hayakawa/items/18b7017da9a6f73eba77
    def learn(self, vec_array, label_array, test_size=0.3, random_state=None):
        # 訓練データとテストデータに分離する
        vec_train, vec_test, label_train, label_test = sklearn.model_selection.train_test_split(vec_array, label_array, test_size=test_size, random_state=random_state )

        # データの標準化処理
        self.scaler.fit(vec_train)
        # モデルの学習
        self.svm.fit(self.scaler.transform(vec_train), label_train)

        # 訓練データとテストデータの正解率を返す
        return self.test(vec_train, label_train), self.test(vec_test, label_test)

    # テストデータと答えを渡すと正解率を返す
    def test(self, vec_array, label_array):
        vec_array_std   = self.scaler.transform(vec_array)
        pred_train        = self.svm.predict(vec_array_std)
        return sklearn.metrics.accuracy_score(label_array, pred_train)


# 画像データを学習用データに変換する
# 今回はHSVデータの色相/彩度/明度それぞれの, 標準偏差/平均値/最大値/最小値を要素としたデータに変換する　
def img_as_vector(cv2_img):
    hsv = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2HSV )
    h, s, v = cv2.split(hsv)
    s_pix = np.array(s).flatten()
    h_pix = np.array(h).flatten()
    v_pix = np.array(v).flatten()
    funcs = [np.std, np.average, np.max, np.min]
    ret = []
    for f in funcs:
        for x in [ h, s, v]:
            ret.append(f(x))
    return ret

# 画像リストからデータ用のcsvを作成する
def save_as_csv(img_array, out_path):
    vec_array = np.ndarray( [ img_as_vector(img) for img in img_array  ])
    np.savetxt(vec_array, 'out_path', dellimiter=',')    

def load_from_csv(in_path):
    return np.loadtxt(in_path, delattr=',')

def main():
    color_vec_array = np.loadtxt('color_page.csv', delimiter=',')
    gray_vec_array = np.loadtxt('gray_page.csv', delimiter=',')
    # 白黒ページの学習データ多すぎるので少なくする
    gray_vec_array = gray_vec_array[: len(color_vec_array),:]

    # 1つの2次元配列にマージ    
    vec_array = np.append(color_vec_array, gray_vec_array, axis=0)

    # カラー画像のラベルは0, 白黒画像のラベルを1にする
    label_array = [0] * len(color_vec_array) +[1] * len(gray_vec_array)
    
    for i in range(0,vec_array.shape[1]):
        for j in range(i+1,vec_array.shape[1]):
            train_score, test_score = 0.0, 0.0
            # 100回訓練した結果の平均をとる
            test_num = 100
            print('{0:2}:{1:2}要素のデータ'.format(i, j), end='  ')
            for n in range(test_num):
                data_list = vec_array[:,i:j]
                model = IsColorPageSVM()                    
                train, test = model.learn(data_list, label_array)
                train_score += train
                test_score += test
            print(u'正解率(訓練/テスト):{0:0.3} - {1:0.3}'.format(train_score/test_num, test_score/test_num))

if __name__ == '__main__':
    main()
    
