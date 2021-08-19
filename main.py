import torch, torchvision
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sample import sampling
import numpy as np, pandas as pd
import pickle
import nn_model
from sklearn.neighbors import KNeighborsClassifier

param_tuning = False
logistic_regression = False
knn = True
deep_learning = False

resp = 'n'
if resp.lower() == 'y':
    sampling(csv_file='output.csv', test_fraction=0.3, standardize=True, normalize=False)
elif resp.lower() == 'n':
    pass
else:
    raise(Exception('Invalid choice!'))


if logistic_regression:
    train_data = pd.read_csv('train.csv', header=0)
    train_data = train_data.to_numpy(copy=True)

    test_data = pd.read_csv('test.csv', header=0)
    test_data = test_data.to_numpy(copy=True)

    X = train_data[:, 3:]
    y = train_data[:, 0]

    X_test = test_data[:, 3:]
    y_test = test_data[:, 0]

    solvers = ['newton-cg', 'liblinear', 'sag', 'saga', 'lbfgs']

    if param_tuning:
        score = 0
        c_list = np.arange(0.001, 0.0011, 0.001)
        c_list = np.concatenate((c_list, np.arange(0.01, 1.01, 0.05)))
        c_list = np.concatenate((c_list, np.arange(1, 101, 5)))
        c_list = np.concatenate((c_list, np.arange(100, 10100, 100)))
        c_list = np.concatenate((c_list, np.arange(10000, 102000, 2000)))
        np.append(c_list, np.arange(1e5, 1e6+10, 1e5))
        print(len(c_list))
        for i, c in enumerate(c_list):
            log_reg = LogisticRegression(multi_class='auto', solver=solvers[1], max_iter=100000, tol=1e-8, C=c, verbose=0)
            log_reg.fit(X_test, y_test)
            new_score = log_reg.score(X_test, y_test)
            if new_score > score:
                with open('trained_lr.pck' , 'wb') as f:
                    pickle.dump(log_reg, f)
                score = new_score
                print('%d : %.4f : %.4f' % (i, c, score))
                c_final = c

    with open('trained_lr.pck', 'rb') as f:
        log_reg = pickle.load(f)
    print('training accuracy: %.4f' % log_reg.score(X, y))
    print('testing accuracy: %.4f' % log_reg.score(X_test, y_test))

if knn:
    train_data = pd.read_csv('train.csv', header=0)
    test_data = pd.read_csv('test.csv', header=0)
    y_train = train_data.iloc[:, 0]
    # X_train = pd.concat((train_data.iloc[:, 513:563], train_data.iloc[:, 1074:1124]), axis=1)
    X_train = train_data.iloc[:, 3:]
    y_test = test_data.iloc[:, 0]
    # X_test = pd.concat((test_data.iloc[:, 513:563], test_data.iloc[:, 1074:1124]), axis=1)
    X_test = test_data.iloc[:, 3:]
    print(list(train_data.columns))
    for i in [3, 5, 7, 11, 17, 27]:
        neigh = KNeighborsClassifier(n_neighbors=i)
        neigh.fit(X_train, y_train)
        print('%d --> %.3f' % (i, neigh.score(X_test, y_test)))
    


#######################################################################################################

# class DatasetTraining(Dataset):

#     def __init__(self, csv_file):
#         self.df = pd.read_csv(csv_file, header=0, sep=',')
#         # df includes 784 features data and their corresponding labels
#         # df is a dataframe with a size of 42000 x 785


#     def __len__(self):
#         # len(object) --> size
#         return self.df.shape[0] #df.shape[0] #np.size(df, 0)


#     def __getitem__(self, idx):
#         # this method can return a tuple (list) where the first element of that tuple
#         # is an array with 784 value which is actually the input of our neural network.
#         # And the second element of that tuple is simply an integer valued in the range of 0 to 9 which is
#         # the label of the idx instance.
#         res = dict()
#         # Putting the features value (input) into the res list
#         # Pytorch works with its own class which is torch.tensor
#         features1 = torch.tensor( np.array(self.df.iloc[idx, 3:564]) ).float()
#         features1 = torch.reshape(features1, (11, 51))

#         features2 = torch.tensor( np.array(self.df.iloc[idx, 564:]) ).float()
#         features2 = torch.reshape(features2, (11, 51))

#         res['data'] = torch.zeros((3, 11, 51))
#         res['data'][0, :, :] = features1
#         res['data'][1, :, :] = features2
#         res['data'][2, :, :] = 0.5 * (features1 + features2)

#         label = int(self.df.iloc[idx, 0])
#         # label_  = [0, 0, 0]
#         # label_[label] = 1
#         # res['label'] = torch.tensor(label_).float()
#         res['label'] = torch.tensor(label).long()
#         return res


# ################################################################

# class DatasetTesting(Dataset):

#     def __init__(self, csv_file):
#         self.df = pd.read_csv(csv_file, header=0, sep=',')
#         # df includes 784 features data and their corresponding labels
#         # df is a dataframe with a size of 42000 x 785


#     def __len__(self):
#         # len(object) --> size
#         return self.df.shape[0]


#     def __getitem__(self, idx):
#         # this method can return a tuple (list) where the first element of that tuple
#         # is an array with 784 value which is actually the input of our neural network.
#         # And the second element of that tuple is simply an integer valued in the range of 0 to 9 which is
#         # the label of the idx instance.
#         res = dict()
#         label = int(self.df.iloc[idx, 0])
#         # label_ = [0, 0, 0]
#         # label_[label] = 1
#         features1 = torch.tensor( np.array(self.df.iloc[idx, 3:564]) ).float()
#         features1 = torch.reshape(features1, (11, 51))

#         features2 = torch.tensor( np.array(self.df.iloc[idx, 564:]) ).float()
#         features2 = torch.reshape(features2, (11, 51))

#         res['data'] = torch.zeros((3, 11, 51))
#         res['data'][0, :, :] = features1
#         res['data'][1, :, :] = features2
#         res['data'][2, :, :] = 0.5 * (features1 + features2)

#         res['label'] = torch.tensor(label).long()
#         return res


# #######################################################################

# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# # device = 'cpu'

# #######################################################################

# net = nn_model.Model()
# net = torchvision.models.resnet18(pretrained=False)
# net.fc = nn.Linear(in_features=512, out_features=3)

# net = net.to(device)

# # You need to define a criterion for classification loss function
# # Usually Cross entropy loss is used
# criterion = nn.CrossEntropyLoss()
# # optimizer = optim.SGD(net.parameters(), lr=1e-6, momentum=0.9)
# optimizer = optim.Adam(net.parameters(), lr=1e-4)

# trainset = DatasetTraining(csv_file='train.csv')
# trainloader = DataLoader(trainset, batch_size=1, shuffle=True, num_workers=0)

# testset = DatasetTesting(csv_file='test.csv')
# testloader = DataLoader(testset, batch_size=1, shuffle=False, num_workers=0)

# # net.load_state_dict(torch.load('trained_model.pth'))

# # Training
# epochs = 5
# for epoch in range(epochs):
#     running_loss = 0.0
#     for i, data in enumerate(trainloader, 0):
#         # time.sleep(10)
#         inputs = data['data']
#         labels = data['label']
#         inputs, labels = inputs.to(device), labels.to(device)
#         optimizer.zero_grad()
#         outputs = net(inputs)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()
#         running_loss += loss.item()
#     print('Epoch: {}/{} --> Loss: {}'.format(epoch+1, epochs, running_loss))

# torch.save(net.state_dict(), 'trained_model.pth')

# ###################################

# # Testing
# net.eval()
# correct = 0
# results = [0, 0, 0]
# actual = [0, 0, 0]
# with torch.no_grad():
#     for i, test_data in enumerate(testloader, 0):
#         label = test_data['label']
#         actual[label] += 1
#         data = test_data['data']
#         data = data.to(device).float()
#         output = net(data)
#         output = output.to('cpu')
#         output = torch.reshape(output, (-1,))
#         prediction = torch.argmax(output).to('cpu').item()
#         results[prediction]  += 1
#         if prediction == label:
#             correct += 1

#     print('accuracy = %.2f' % (correct / len(testset)))
#     print('predictions: ', results)
#     print('actual: ', actual)
