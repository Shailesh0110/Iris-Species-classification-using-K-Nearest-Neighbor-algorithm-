
from sklearn import datasets
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report,f1_score,precision_score,recall_score,roc_auc_score
from sklearn.model_selection import train_test_split
from scipy.spatial import distance
from sklearn.metrics.pairwise import euclidean_distances

def euc(a,b):

    return distance.euclidean(a,b)
    #print(distance.euclidean(a, b))


class KNeighborsClassifier1:

    def fit(self,X_train,y_train):
        self.X_train=X_train
        self.y_train=y_train
        print('inside fit')

    def predict(self, X_test):
        prediction = []
        for value in X_test:
            print('inside predict1')
            result = self.closet(value)
            print(result)
            prediction.append(result)
            print('inside predict')
        return prediction
            #print(prediction)


    def closet(self,row):
        minmumdistance=euc(row,self.X_train[0])
        print(minmumdistance)
        print("============================================")
        minmumindex=0
        print('inside closet1')
        print(len(self.X_train))
        for i in range(1,len(self.X_train)):

            Distance=euc(row,self.X_train[i])
            #print(Distance)
            if Distance < minmumdistance:
                print("min dist 0.42:",Distance)
                minmumdistance=Distance
                print(minmumdistance)
                minmumindex=i
                print(minmumindex)
            else:
                pass
        print("==================ENd=======================")
        print(self.y_train)
        print(minmumindex)
        print(self.y_train[minmumindex])
        return (self.y_train[minmumindex])
        #print(self.y_train[minmumindex])
        #print('inside closet')

def marvellousML():
    Dataset=datasets.load_iris() # Load the data

    Data=Dataset.data
    Target=Dataset.target

    Data_Train,Data_Test,Target_Train,Target_Test =train_test_split(Data,Target,test_size=0.8)
    #model=KNeighborsClassifier()
    print('inside mavrllousML')
    model = KNeighborsClassifier1()
    model.fit(Data_Train,Target_Train)
    Target_pred=model.predict(Data_Test)
    Accuracy=accuracy_score(Target_Test,Target_pred)
    return Accuracy




def main():
    accuracy_return=marvellousML()
    #print("Accuracy of Iris data set in knn:",ans*100)
    print("Accuracy of Iris data set in knn:", accuracy_return * 100)

if __name__ == "__main__":
    main()