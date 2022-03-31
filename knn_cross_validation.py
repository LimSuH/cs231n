import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt


class NearestNeighbor(object):
    def __init__(self):
        pass

    def train(self, X, y):
        """ X is N x D where each row is an example. Y is 1-dimension of size N """
        # the nearest neighbor classifier simply remembers all the training data
        self.Xtr = X
        self.ytr = y

    def predict(self, dists, k):
        num_test = dists.shape[0]
        predict = np.zeros(num_test)
        for i in range(num_test):
            nearest = np.argsort(dists[i])[:k]  # i번째 테스트셋의 가장가까운 n개 이웃을 찾기(정렬)
            predict[i] = self.ytr[
                np.argmax(np.bincount(nearest))]  # n개 이웃들중 가장 많이 나오는 라벨값 찾기(빈도수를 반환하는 함수 배열 중 최고갑의 인덱스)

        return predict

    def distance(self, X):
        num_test = X.shape[0]
        num_train = self.Xtr.shape[0]
        dist = np.zeros((num_test, num_train))

        for i in range(num_test):
            for j in range(num_train):
                dist[i][j] = np.sqrt(np.sum(np.square(X[i] - self.X_train[j])))

        return dist


def load_CIFAR10():
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=60000, shuffle=True, num_workers=2)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=60000, shuffle=False, num_workers=2)

    dataiter = iter(trainloader)
    train_images, train_labels = dataiter.next()

    dataiter = iter(testloader)
    test_images, test_labels = dataiter.next()

    return train_images, train_labels, test_images, test_labels


def run():
    Xtr, Ytr, Xte, Yte = load_CIFAR10()
    Xtr_rows = Xtr.reshape(Xtr.shape[0], 32 * 32 * 3)  # Xtr_rows becomes 50000 x 3072

    # cross validation 이용
    foldNum = 5
    # 트레이닝 셋을 다섯개로 나누기--> [[0~9999], [10000~19999]....]
    validation_x = np.array_split(Xtr_rows, fold_num)
    validation_Y = np.array_split(Ytr, fold_num)

    kAcc = {}
    nn = NearestNeighbor()
    print("0%... |", end='')
    for k in [1, 3, 5, 8, 10, 12, 15, 20, 50, 100]:
        acc = []
        # use a particular value of k and evaluation on validation data
        correctNum = 0
        for t in range(foldNum):  # 폴드 하나는 validation(xtr_folds), 나머지는 트레이닝 셋(xtr_fold)
            training_x = np.concatenate([x for num, x in enumerate(validation_x) if num != t])
            training_Y = np.concatenate([y for num, y in enumerate(validation_Y) if num != t])

            nn.train(training_x, training_Y)
            prediction = nn.predict(validation_x[t], k)
            correctNum = np.sum(prediction == np.array(validation_Y[t]))  # 실제 라벨과 예측 라벨이 맞으면 맞춘 횟수 카운트
            acc.append(float(correctNum) / len(validation_Y[t]))  # 트레이닝 셋의 예측과 validation 셋의 숫자로 평균..
        kAcc[k] = acc # 각 k별로 정확도 저장
        print("## ", end='')
    print("| 100%")

    # 각 k 별로 정확도 출력

    for k in [1, 3, 5, 8, 10, 12, 15, 20, 50, 100]:
        print('k = {}, accuracy = {}'.format(k, kAcc[k]))

    for k in [1, 3, 5, 8, 10, 12, 15, 20, 50, 100]:
        accuracies = kAcc[k]
        plt.scatter([k] * len(accuracies), accuracies)

    # plot the trend line with error bars that correspond to standard deviation
    accuracies_mean = np.array([np.mean(v) for k, v in sorted(kAcc.items())])
    accuracies_std = np.array([np.std(v) for k, v in sorted(kAcc.items())])
    plt.errorbar([1, 3, 5, 8, 10, 12, 15, 20, 50, 100], accuracies_mean, yerr=accuracies_std)
    plt.title('Cross-validation on k')
    plt.xlabel('k')
    plt.ylabel('Cross-validation accuracy')
    plt.show()


if __name__ == '__main__':
    run()

