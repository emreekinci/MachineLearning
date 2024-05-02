import pandas as pd
import  numpy as np
import seaborn as sns
import  matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap 

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier, NeighborhoodComponentsAnalysis, LocalOutlierFactor
from sklearn.decomposition import PCA

#missing value: 2 cesit , 0 değeri olması bazı veri setlerinde mv  , bazen de gerçekten boştur alan.  -->concavity mean
# warning library
import warnings
warnings.filterwarnings("ignore")

data = pd.read_csv("cancer_data.csv")
data.drop(['Unnamed: 32','id'], inplace = True, axis = 1) # csv de sonda , silinebilir ya da bu şekilde axis=1(column) drop edilir

data = data.rename(columns = {"diagnosis":"target"}) # yeniden kolon isimlendiriyoruz

sns.countplot(data["target"]) # veriyi görselleştiriyoruz sade şekilde
#print()
print(data.target.value_counts())

# M(kötü huylu hücre:kanser):1 B:0(sağlıklı) --> M,B harfleri csv dosyamdaki görselleştirme dışında kalacak onları değiştirmem gerek
data["target"] = [1 if i.strip() == "M" else 0 for i in data.target] # verilerde boşluk varsa kaldırmak için i.strip() /(M,B) 1:iyi hücre 0: kötü

print(len(data)) # sample sayısını yazdırmak için
print(data.head()) # ilk 5 satıra bak
print("Data Shape: ", data.shape)
data.info() # missing value lara bakmak için kullanılır burada miss value yok. 31 numerik feature a sahibim : 30 float , 1 int
describe = data.describe() 
print(describe) #  count: sample sayısı, mean:ortalama, std: standart sapma  

"""
STANDARDIZATION:
    
 --> verilere bakınca aralarında büyük scale farkları vardır area mean,radius mean arasında meseala 
 --> missing value :none

Gerekli kütühaneler import edildi
veri seti yüklendi basit veri analizi yapıldı..
"""


# %% EDA: Exploratory data analysis (Açınsayıcı Veri Çözümlemesi)

"""
 Genellikle istatistiksel grafikler ve diğer veri görselleştirme yöntemlerini kullanarak temel özelliklerini özetlemek için 
 veri kümelerini analiz etme yaklaşımıdır. istatistiksel bir model kullanılabilir veya kullanılamaz --> kullanıyoruz..
"""

#numerık verilere sahibiz correlation "korelasyon" matrisine bakmamız gerek

#correlation
crl_mtrx =  data.corr() # numerik degerlerdeki korelasyona bakılır. --> string degerimiz yok bizim
# seaborn kutuphanesini kullanıyorum korelasyon matrisimi görselleştirip  anlaşılır hale getirelim
#feature lar arası ilişkiye bakıyorum eğer iki feature arasındaki ilişide ilişki 1 se %100 doğru orantılı -1 ise %100 ters orantılı
sns.clustermap(crl_mtrx, annot= True, fmt = ".2f") #annot: true degerler görünsün, sadece 2 floating point göreyim
plt.title("Correlation Between Features (-1 to 1)") # korelasyon aralıkları 
plt.show()

"""
sonucta birbirine yakın degerleri(radius_mean ,area_mean, perimeter_worst...) algoritmamı egitmek icin kullanmam mantıklı olmayacaktır yakın degerler birbiriyle alakalı degerler demektir
 ML MODEL imde çeşitliliğe gitmek zorundayım birbiri ile ilişkili olmayan(symmetry_worst,dimension_se..) feature lar seçmem gerek.
"""

# daha özel bir plot çizimi
threshold = 0.75
filt = np.abs(crl_mtrx["target"]) > threshold 
corr_features = crl_mtrx.columns[filt].tolist() # sınırlandırma
sns.clustermap(data[corr_features].corr(), annot = True, fmt= ".2f")# bu defa datama sınırlandırdıgım(filtrelenen) satırlar gelecek   
plt.title("Correlation Between Features with corr threshold 0.75") # korelasyon aralıkları 
plt.show() # 4 feature ile targeet variable yüksek ilişkilidir, daha ozel plot

"""
there some correlated features: ilerleyen zamanda farklı veri setlerinde eğer birrbirleriyle doğru oranılı veya ters orantılı feature lar varsa bunları ortadan kaldırmak gerek
ya da regularization yöntemleri kullanılmalı:regex
"""

#box plot
data_melted = pd.melt(data, id_vars = "target", var_name = "features", value_name = "value") #2 class seklinde gorsellestirmek istiyorum

plt.figure()
sns.boxplot(x = "features", y = "value", hue = "target", data = data_melted)
plt.xticks(rotation = 90) # feature isimleri 90 derece dondu dik oldular
plt.show() # cok yuksek scale de iki deger ortaya cıktı : box plot tan anlam çıkarmak için :data standardization veya normalization

"""
normalization / standardization : box plot sonra tekrar çizdirilecek
"""

#pair plot : veriler gene düzgün olmayacak veriler standardize değil
sns.pairplot(data[corr_features], diag_kind = "kde", markers="+", hue = "target") # sadece correlated feture lara bak, kde: histogram şeklinde göster, target: 2 class
plt.show() # 0 : iyi huylu kanser 1: kotu huylu // positive skewness-right tail, negative skewness-left tail, gaussian distrubition(normal dağılım: insan boyları)

"""skewness"""

# pzitif veya negatif çarpıklık olduğu zaman bunu normalize etmeye çalışıyoruz
# skewness lığı handle edebilecek(normal dağılıma çevirecek)  outlier detection yöntemi seçilmeli 

# positive skewness: gelir dağılımı örneğin mean degeri sagdadır, medyan az solu, mod degerin en yuksek oldugu  yer
# negative skewness: genelde yüksek not alınan sınav degerleri  ortalama, medyan, mod şekinde soldan sağadır (0,....,100) 
# bu projede skew() metotları kullanılmadı bu değer 1 den büyük + skew -1 den küçükse negatif skew

# %% outlier

# datamı x ler ve y ler olmak üzere ikiye ayırıyorum
# x: features, y:target variable olacak şekilde

#    data içerisinde target diye bir feature ım var bunlar class labellarımız. 
#    bunları çıkarttığımız zaman geriye normal feature lar kalmış oluyor.
y = data.target
x = data.drop(["target"], axis=1)
columns = x.columns.tolist() # datasetimdeki featur ları colums ta depoladım

# LocalOutlierFactor() bize negative_outlier_factor parametresini return edecek. Bize LOF un ters işaretli negatif çarpımlı sonucunu verir.
clf = LocalOutlierFactor() # KNN için k,kaç n_neighbour(komşu):default=20 seçeceğimiz önemli parametreler
y_pred = clf.fit_predict(x) # outliers için --> return -1 , inliers ve diğerleri için --> return 1
x_score = clf.negative_outlier_factor_

outlier_score = pd.DataFrame()
outlier_score["score"] = x_score

#treshold atıp 2.5 ten veya -2 den büyük birkaç değer var onları çıkarabiliriz hassasiyet için --> raporda png referans ver**
threshold = -2 # tek değer var 
filtre = outlier_score["score"] < threshold
outlier_index = outlier_score[filtre].index.tolist()

#plt scatter
# 0. ve 1. sütunlara baktık onlar radius_mean ve texture_mean başka outlier dağılımlara bakıp daha güzel sonuçlar elde edebilir miyiz bakılır 
plt.figure()
plt.scatter(x.iloc[outlier_index,0], x.iloc[outlier_index,1], color="blue", s=50, label = "Outlier") # 0 ve 1 column larını kullanıyorum görselleştirmek için
plt.scatter(x.iloc[:,0], x.iloc[:,1], color="k", s=3, label = "Data Points") # 0 ve 1 column larını kullanıyorum görselleştirmek için

radius = (x_score.max() - x_score) / (x_score.max() - x_score.min())
outlier_score["radius"] = radius # outlier içerisine normalize ettiğim 0 ile 1 arası değerleri attım neden durumu görselleştirebilmek için
# datapointlerimi çizdiriyorum:x.iloc[:,0],x.iloc[:1,]  , s(size): 1000 * radius  , edgecolor: nokta_renk,facecolor:iç renk(none:çember),label:etiket 
plt.scatter(x.iloc[:,0], x.iloc[:,1], s = 1000 * radius, edgecolors="r", facecolors= "none", label ="Outliers Score")
plt.legend() # label lar görünür olsun diye
plt.show()

#drop outliers //    type:pandas.core.frame.DataFrame  type:array e döner
 
x = x.drop(outlier_index)
y = y.drop(outlier_index).values 

# %% Train Test Split

test_size = 0.3 # veri boyutu büyüdükçe küçültülür 0.00005 gibi
# X_train, X_test, Y_train, Y_test train test in return ettiği değerler , default shuffle=True (veri karıştırma default)
# random_state = 42 ile her seferinde aynı shuffle ı kullanabiliyoruz bu yüzden önemli buna göre train ve test setlerim oluşuyor.
# KNN den önce bu veri setlerimiz aynı olsun ki karışmasın farklı adımlar ilerde atmamıza gerek kalmasın.
X_train, X_test, Y_train, Y_test = train_test_split(x,y, test_size = test_size, random_state = 42) # x ve y split edilecek x ve y arasında train oranı olması gerek ne kadarı train edilecek ne kadarı split edilecek gibi.

# %%
# standardize edildikten sonra mean 0 a yaklaşır ve standart sapma da 1 e yaklaşır.

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train) # StandartScaler kullanarak X_train datasına göre bir scaler tanımla fit et ve bunu X_train datasını kullanarak transforme et demek
# X_train e göre eğitilmiş scaler ımı X_test üzerinde uyguluyorum aşağıda
X_test = scaler.transform(X_test) 

# artık boxplot um görselleştirilebilir : mean std değerlerim düzgün 
X_train_df = pd.DataFrame(X_train, columns = columns)
X_train_df_describe = X_train_df.describe() # describe train edilmiş ve edillmemiş: 7-3_train_describe.png 7-2non_train_describe.png
X_train_df["target"] = Y_train
#box plot
data_melted = pd.melt(X_train_df, id_vars = "target", var_name = "features", value_name = "value") #2 class seklinde gorsellestirmek istiyorum

plt.figure()
sns.boxplot(x = "features", y = "value", hue = "target", data = data_melted)
plt.xticks(rotation = 90) # feature isimleri 90 derece dondu dik oldular
plt.show() 

# sonuçta: her bir feature a ait dağılımları ,farklı class lar için dağılımları ve outlier değerleri görebiliyoruz -->0,1 iyi huylu kötü huylu ve outlierlar
# ilerde eğer modelimiz istenen sonuç elde edilemezse outlier lar çıkarılır 

#pair plot
sns.pairplot(X_train_df[corr_features], diag_kind = "kde", markers="+", hue = "target") # sadece correlated feture lara bak, kde: histogram şeklinde göster, target: 2 class
plt.show()

# %% Basic KNN method

knn = KNeighborsClassifier(n_neighbors=2) # default:5
knn.fit(X_train, Y_train)
y_pred = knn.predict(X_test)
cm = confusion_matrix(Y_test, y_pred)
acc = accuracy_score(Y_test,y_pred)
score = knn.score(X_test, Y_test)
print("Score: ",score)
print("CM: ",cm)
print("Basic KNN Accuracy: ",acc)

"""
SONUC:
Score:  0.9529411764705882
CM:  [[107   0]   --> 107 İYİ HUYLU DOGRU TAHMIN ETMISIM
     [  8  55]]   --> 55 KOTU HUYLU DOGRU TAHMIN ETMISIM, 8 YANLIS TAHMIN --> ACCURACY -%5 kayıp  
Basic KNN Accuracy:  0.9529411764705882
"""

# %% choose best parameters

def KNN_Best_Params(x_train, x_test, y_train, y_test):
    # 30 deger için en uygun k degerini bulmaya calısıyoruz
    k_range = list(range(1,31)) 
    weight_options = ["uniform","distance"]
    print()
    param_grid = dict(n_neighbors = k_range, weights = weight_options)    
    
    knn = KNeighborsClassifier()
    grid = GridSearchCV(knn, param_grid, cv =10, scoring= "accuracy")
    grid.fit(x_train, y_train)
    
    print("Best training score: {} with parameters {}".format(grid.best_score_, grid.best_params_))
    print()
    
    knn = KNeighborsClassifier(**grid.best_params_)
    knn.fit(x_train, y_train)
    
    y_pred_test = knn.predict(x_test)
    y_pred_train = knn.predict(x_train)
    
    cm_test = confusion_matrix(y_test, y_pred_test)
    cm_train = confusion_matrix(y_train, y_pred_train)

    acc_test = accuracy_score(y_test, y_pred_test)
    acc_train = accuracy_score(y_train, y_pred_train)
    print("Test Score: {}, Train Score: {}".format(acc_test, acc_train))
    print() 
    print("CM Test: {}",cm_test)
    print("CM Train: {}",cm_train)
    
    return grid

grid = KNN_Best_Params(X_train, X_test, Y_train, Y_test)

"""SONUC
Best training score: 0.9692948717948718 with parameters {'n_neighbors': 4, 'weights': 'distance'}

Test Score: 0.9470588235294117, Train Score: 1.0 # train score um test score dan buyuk overfitted lık soz konusu yani ezberleme soz konusu

CM Test: {} [[104   3]
 [  6  57]]
CM Train: {} [[249   0]
 [  0 145]]

"""


# %% PCA
"""
PCA : Mümkün olduğunca bilgi tutarak verinin boyutunun azaltılmasını sağlayan yöntem
 Neden kullanıyoruz zaman,güç kısıtımız varsaveri boyutu çoksa ki projemizdeki dimension dan feature lardan bahsediyoruz.
 Bu dueumda belli başlı feature ları azaltabiliriz yani verinin boyutunu azaltabiliriz, ikinci nedense eğer korelasyon matrisimiz
 varsa ki bizim var bu feature lardan bazıları birbiriyle ilişkili ise ve bu feature ları nasıl çıkaracağımızı bilmiyorsak 
 bu feature ları kaldırmak için de kullanılabilir. Diğer bir amacı da görselleştirmedir. 
"""
# 30 boyutlu veriyi 2 boyutlu veriye dönüştürüp çizdiriyoruz
# x içinde 30 tane feature var
scaler = StandardScaler()    
x_scaled = scaler.fit_transform(x) 
    
pca = PCA(n_components=2)
pca.fit(x_scaled)
X_reduced_pca = pca.transform(x_scaled)
pca_data =pd.DataFrame(X_reduced_pca, columns = ["p1","p2"])
pca_data["target"] = y
sns.scatterplot(x ="p1",y = "p2", hue= "target",  data = pca_data)
plt.title("PCA: p1 vs p2")


X_train_pca, X_test_pca, Y_train_pca, Y_test_pca = train_test_split(X_reduced_pca, y, test_size = test_size, random_state = 42)
grid_pca = KNN_Best_Params(X_train_pca, X_test_pca, Y_train_pca, Y_test_pca)

#visualize
cmap_light = ListedColormap(['orange', 'cornflowerblue'])
cmap_bold = ListedColormap(['darkorange', 'darkblue'])

h= .05 #  görüntüyü güzelleştirmek için adım sayısını küçük tutuyoruz
X = X_reduced_pca
x_min, x_max = X[:,0].min() -1 , X[:, 0].max() + 1 
y_min, y_max = X[:,1].min() -1 , X[:, 1].max() + 1
xx, yy =np.meshgrid(np.arange(x_min, x_max, h),
                    np.arange(y_min, y_max, h)) 

Z = grid_pca.predict(np.c_[xx.ravel(), yy.ravel()]) # aldığım her noktayı sınıflandırıyorum

#sonucun renklendirilmesi
Z = Z.reshape(xx.shape)
plt.figure()
plt.pcolormesh(xx, yy, Z, cmap=cmap_light)


plt.scatter(X[:,0], X[:,1], c=y, cmap=cmap_bold,
            edgecolor='k',s =20)
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.title("%i-Class Classification (k = %i, weights='%s')"
          % (len(np.unique(y)) ,grid_pca.best_estimator_.n_neighbors, grid_pca.best_estimator_.weights))

"""SONUC

Best training score: 0.9593589743589742 with parameters {'n_neighbors': 9, 'weights': 'uniform'}

Test Score: 0.9235294117647059, Train Score: 0.9593908629441624

CM Test: {} [[102   5]
 [  8  55]]
CM Train: {} [[243   6]
 [ 10 135]]
Out[49]: Text(0.5, 1.0, "2-Class Classification (k = 9, weights='uniform')")
"""

# %% NCA

nca = NeighborhoodComponentsAnalysis(n_components=2, random_state = 42)
nca.fit(x_scaled, y)
X_reduced_nca = nca.transform(x_scaled) 
nca_data = pd.DataFrame(X_reduced_nca, columns= ["p1","p2"])
nca_data["target"] = y
sns.scatterplot(x = "p1", y= "p2", hue = "target", data =  nca_data)
plt.title("NCA: p1 vs p2")

        
X_train_nca, X_test_nca, Y_train_nca, Y_test_nca = train_test_split(X_reduced_nca, y, test_size = test_size, random_state = 42)
grid_nca = KNN_Best_Params(X_train_nca, X_test_nca, Y_train_nca, Y_test_nca)

#visualize
cmap_light = ListedColormap(['orange', 'cornflowerblue'])
cmap_bold = ListedColormap(['darkorange', 'darkblue'])

h= .2 #  görüntüyü güzelleştirmek için adım sayısını küçük tutuyoruz
X = X_reduced_nca
x_min, x_max = X[:,0].min() -1 , X[:, 0].max() + 1 
y_min, y_max = X[:,1].min() -1 , X[:, 1].max() + 1
xx, yy =np.meshgrid(np.arange(x_min, x_max, h),
                    np.arange(y_min, y_max, h)) 

Z = grid_nca.predict(np.c_[xx.ravel(), yy.ravel()]) # aldığım her noktayı sınıflandırıyorum

#sonucun renklendirilmesi
Z = Z.reshape(xx.shape)
plt.figure()
plt.pcolormesh(xx, yy, Z, cmap=cmap_light)


plt.scatter(X[:,0], X[:,1], c=y, cmap=cmap_bold,
            edgecolor='k',s =20)
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.title("%i-Class Classification (k = %i, weights='%s')"
          % (len(np.unique(y)) ,grid_nca.best_estimator_.n_neighbors, grid_nca.best_estimator_.weights))

























