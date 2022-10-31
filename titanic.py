#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
タイタニック問題

author by Iwadon
"""

import torch
import torch.nn as nn

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import pandas as pd

from matplotlib import pyplot

class TitanicDataSet(Dataset):
    """
    タイタニック問題データセットクラス
    """
    __data = []
    __is_test = False

    def __init__(self, data_filepath, is_test=False):
        self.__data = pd.read_csv(data_filepath)
        # 欠損データを補完
        self.__data["Age"] = self.__data["Age"].fillna(self.__data["Age"].median())
        self.__data["Embarked"] = self.__data["Embarked"].fillna("S")

        self.__is_test = is_test

    def __len__(self):
        return len(self.__data)

    def __getitem__(self, idx):
        passId = self.__data.at[idx, "PassengerId"]
        sex = self.__trans_sex(self.__data.at[idx, "Sex"])
        age = self.__data.at[idx, "Age"]
        related_person = self.__data.at[idx, "SibSp"] + self.__data.at[idx, "Parch"]  # 乗車した家族の数
        pclass = self.__data.at[idx, "Pclass"]  # チケットクラス　(上級: 1 , 中級: 2, 下級: 3)
        if self.__is_test:
            label = passId
        else:
            label = self.__data.at[idx, "Survived"]  # 0:死亡, 1:生存
        data = [sex, age, related_person,pclass]
        tdata = torch.tensor(data,dtype=torch.float32)

        if self.__is_test:
            return tdata, label

        return tdata, torch.tensor(label, dtype=torch.float32)

    def __trans_embarked(self,value):
        if value == "S":
            return 0
        elif value == "C":
            return 1
        elif value == "Q":
            return 2
        else:
            print("Unknown Value")
            return -1

    def __trans_sex(self,value):
        if value == "male":
            return 0
        elif value == "female":
            return 1
        else:
            print("Unknown Value")
            return -1


class TitanicModel(nn.Module):
    def __init__(self):
        super(TitanicModel, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(4, 25),
            nn.ReLU(),
            nn.Linear(25, 20),
            nn.ReLU(),
            nn.Linear(20, 2)
        )

    def forward(self, x):
        logits = self.network(x)
        return logits

def exec_train(dataloader, model, loss_fn, optimizer):
    """
    モデルに対して学習を実行します
    :param dataloader: データローダー(学習用)
    :param model: モデルインスタンス
    :param loss_fn: 損失関数
    :param optimizer: オプティマイザー
    :return: 損失値
    """
    for batch, (data, label) in enumerate(dataloader):
        label = label.long()
        X, label = data.to(device), label.to(device)

        # 損失誤差を計算
        pred = model(X)

        loss = loss_fn(pred, label)

        # バックプロパゲーション
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return loss.item()

def exec_validation(dataloader, model, loss_fn):
    """
    パラメータやモデルのパフォーマンス測定のための検証を行う
    Lossと分類の正解率を算出する

    :param dataloader: データローダー(検証用)
    :param model: 学習中のモデル
    :return:　損失値, 正解率
    """
    size = len(dataloader.dataset)
    model.eval()
    total_loss, correct_count = 0, 0
    soft_max_f =nn.Softmax(dim=1)
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            output = model(X)
            pred = soft_max_f(output)
            correct_idx = y
            max, argmax = torch.max(pred, dim=1)

            total_loss += loss_fn(output, y.long()).item()
            for i in range(len(argmax)):
                if correct_idx[i] == argmax[i]:
                    correct_count += 1

    avg_loss = total_loss / size
    correct_rate = correct_count / size
    print("size : " + str(size) + "   loss avg : " + str(avg_loss) + "   correct : " + str(correct_count) + " / " + str(size) + "  correct rate : " + str(correct_rate * 100) + "%")
    return avg_loss, correct_rate

def exec_test(dataloader, model):
    """
    テストデータをモデルに入力して、推論結果を出力します。

    :param dataloader: データローダー(テストデータ)
    :param model: 学習済みタイタニックモデル
    :return: None
    """

    row_data = []
    model.eval()
    soft_max_f = nn.Softmax(dim=1)
    row_data.append("PassengerId,Survived")

    with torch.no_grad():
        for X, id in dataloader:
            X, id = X.to(device), id.to(device)
            output = model(X)
            pred = soft_max_f(output)
            max, argmax = torch.max(pred, dim=1)

            for i in range(len(argmax)):
                row_data.append(str(id[i].item()) + "," + str(argmax[i].item()))

    try:
        with open("result.csv", 'w', encoding="utf-8") as f:
            for item in row_data:
                f.write(item+"\n")
    except Exception as e:
        print(e)

# ハイパーパラメータ
TRAIN_BATCH_SIZE = 50
VALID_BATCH_SIZE = 50
LEARNING_RATE = 0.005
EPOCH_NUM = 1000


# データセットに関して
TRAIN_DATA_PATH = r"Please Input Your CSV PATH"
TEST_DATA_PATH = r"Please Input Your CSV PATH"
TOTAL_TRAIN_DATA_SIZE = 891 # 学習データ全件数
SPLIT_SIZE = 750 # 学習データの内、学習に利用するデータの数(残りは検証用に)
TEST_DATA_SIZE = 418 # テストデータ全件数

if __name__ == "__main__":
    print("Start Titanic Model")

    # シードを固定
    torch.manual_seed(0)

    # データセット読み込み
    train_data = TitanicDataSet(TRAIN_DATA_PATH, False)
    # テストデータ読み込み
    test = TitanicDataSet(TEST_DATA_PATH, True)

    train, valid = torch.utils.data.random_split(
        train_data,
        [SPLIT_SIZE, TOTAL_TRAIN_DATA_SIZE-SPLIT_SIZE]
    )

    train_dataloader = DataLoader(train, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
    valid_dataloader = DataLoader(valid, batch_size=VALID_BATCH_SIZE, shuffle=True)
    test_dataloader = DataLoader(test, batch_size=TEST_DATA_SIZE, shuffle=False)

    # GPU環境があればGPUで実行
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Use device : ' + device)

    # モデルインスタンス生成
    model = TitanicModel().to(device)
    print(model)

    # 損失関数設定
    loss_fn = nn.CrossEntropyLoss()
    # オプティマイザー設定
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)

    loss_history = []
    correct_rate_history = []
    for t in range(EPOCH_NUM):
        print("------------------------------------------------")
        print("epoch : " + str(t))
        loss = exec_train(train_dataloader, model, loss_fn, optimizer)
        loss, correct_rate = exec_validation(valid_dataloader, model, loss_fn)
        loss_history.append(loss)
        correct_rate_history.append(correct_rate)

    fig = pyplot.figure(figsize=(10, 10), facecolor='lightblue')
    ax1 = fig.add_subplot(2, 1, 1)
    ax2 = fig.add_subplot(2, 1, 2)

    ax1.plot(range(EPOCH_NUM), loss_history, color="blue", label='loss')
    ax2.plot(range(EPOCH_NUM), correct_rate_history, color="green",  label='Correct Rate')

    # グラフタイトル
    ax1.set_title('Titanic Model Loss')
    ax2.set_title('Titanic Model Correct Rate')

    # グラフの軸
    pyplot.xlabel('epoch')
    ax1.set_ylabel('loss')
    ax2.set_ylabel('Correct Rate')

    # グラフの凡例
    ax1.legend()
    ax2.legend()

    pyplot.show()
    torch.save(model, 'model.pth')

    # テストデータで実行
    exec_test(test_dataloader, model)
    print("Finish!")
