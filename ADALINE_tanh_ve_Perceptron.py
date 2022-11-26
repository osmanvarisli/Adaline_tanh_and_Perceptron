# -*- coding: utf-8 -*-
"""
Created on Wed Nov  2 22:04:54 2022

@author: Osman VARIŞLI
"""

import numpy as np
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split


#bias değeri 
bias=1
#öğrenme hizi
c=0.7
E=0.001

def adaline(data,result,test_data,test_result,W,bias,c,E):
    hata_payi_list=[]
    num_rows, num_cols = data.shape; 
    
    iterasyon_yap=True
    iterasyon_saysi=0;
    while iterasyon_yap : 
        iterasyon_saysi+=1
        hata_payi_list.clear()#hata paylarini sifirla
        for row in range(num_rows):
            matrix_carpimi=np.dot(data[row],W)

            tanh_Values = np.tanh(matrix_carpimi)
            
            hata_payi=result[row]-tanh_Values
            hata_payi_list.append(hata_payi)
            
            q=1-tanh_Values*tanh_Values

            W=W+c*hata_payi*q*data[row]
           
        #İterasyon sonunda durdurma ölçütünün sağlayıp sağlamadığı kontrol edilir
        
        ortalama_ete=np.dot(hata_payi_list,hata_payi_list)/len(hata_payi_list)
        if ortalama_ete<=E : iterasyon_yap=False
    print('Adaline Eğitim Tamamlandı.')
    print('Güncel Ağırlık : '+str(W))
    
    num_rows, num_cols = test_data.shape;
    test_y=np.zeros(num_rows)
    for row in range(num_rows):
        test_y[row]=round(np.tanh(np.dot(test_data[row],W)))
        if (test_y[row]==test_result[row]):
            print(str(row+1)+'. Test Başarılı')
        else:
            print(str(row+1)+'. Test Başarısız')

    print('--------------------------------')   
    
    
def perceptron(data,result,test_data,test_result,W,bias,c):
    num_rows, num_cols = data.shape;

    y=np.zeros(num_rows) 
    iterasyon_yap=True
    iterasyon_saysi=0;
    while iterasyon_yap : 
        iterasyon_saysi+=1
        for row in range(num_rows):

            matrix_carpimi=np.dot(data[row],W)
            if matrix_carpimi>0 :y[row]=1
            else: y[row]=-1
            
            #yeni ağırlık oluştur
            W=W+0.5*c*(result[row]-y[row])*data[row]
    
        #kaç sınıflandırmanın doğru olduğunu bulalım
        dogru_say=0;
        yanlis_say=0;
        for row in range(num_rows):
            if y[row]== result[row]:
                dogru_say+=1;
            else:
                yanlis_say+=1;
        if (yanlis_say==0 or iterasyon_saysi>50) : iterasyon_yap=False;
   
    print('Perceptron Eğitim Tamamlandı.')
    print('Güncel Ağırlık : '+str(W))   
    
    
    num_rows, num_cols = test_data.shape;
    test_y=np.zeros(num_rows)
    for row in range(num_rows):
        test_y[row]=np.dot(test_data[row],W)
        
        if test_y[row]>0 :test_y[row]=1
        else: test_y[row]=-1
        if (test_y[row]==test_result[row]):
            print(str(row+1)+'. Test Başarılı')
        else:
            print(str(row+1)+'. Test Başarısız')
    print('--------------------------------')  
    

    
boyut=5
"""
data, result = make_blobs(n_samples=30, centers=2, n_features=boyut)
#data burdan oluşturuldu,
"""
data=np.array([[ -3.237562,     2.47665956,  -7.41351187,   2.10948091,   4.17033364],
 [ -4.89794309,   3.92666623,  -8.02136723,   4.0521347 ,   6.09943641],
 [  3.32420518,  -2.18925043,  -9.76155863,  -8.98992045,   2.32205583],
 [ -4.07777608,   3.75855678,  -5.81920138,   3.42927032,   4.19170126],
 [ -4.36356261,   1.74171017,  -7.13772377,   4.27163918,   3.21876991],
 [  3.20160556,  -3.55780931,  -9.54154525,  -8.73953734,   0.47399297],
 [  3.62624349,  -2.9517149 ,  -8.98664458,  -8.42928658,   0.56701341],
 [  2.96772838,  -4.94043839,  -8.70034634,  -6.9898668 ,   0.29655068],
 [ -6.93106155,   4.11264858,  -7.49441906,   3.43286723,   5.63264221],
 [ -6.89994921,   4.61008975,  -8.28919678,   5.6211398 ,   4.70976822],
 [ -4.37495511,   3.63477257,  -8.35183107,   1.78788964,   3.90100198],
 [ -4.38264238,   3.87531973,  -6.45758494,   3.03034567,   4.24687009],
 [  4.729163  ,  -2.47368584,  -9.65226501,  -8.33632826,   1.66286901],
 [ -5.04617253,   5.05252333,  -7.67960224,   1.30354646,   5.6631479 ],
 [ -4.51738825,   3.68842526,  -8.68143397,   3.19596824,   3.86934504],
 [  3.89616554,  -4.66615717,  -9.8041634 ,  -8.17656343,   0.04358009],
 [ -5.49464331,   4.59941496,  -6.47002073,   4.58016815,   4.24184915],
 [  5.67081753,  -2.9700927 ,  -8.21830736,  -5.9238531 ,  -0.80022083],
 [  2.87228177,  -3.69554117,  -8.26025602,  -7.03637001,   1.94621885],
 [  2.32890749,  -3.26351803,  -9.74023656,  -6.18781169,   2.38419145],
 [ -2.83198726,   4.96488483,  -7.32947979,   4.361344  ,   5.70773689],
 [  4.66234165,  -2.05964916, -10.5796913 ,  -8.02125422,   0.28834926],
 [  2.93134127,  -3.12675315,  -8.07895189,  -8.42564441,   1.1999915 ],
 [ -5.99796845,   2.66199947,  -5.63806291,   2.47942862,   6.47180544],
 [  3.25497925,  -4.56597476,  -7.94456327,  -9.62409288,   1.52269873],
 [  3.74092418,  -0.85978359,  -7.70336385,  -7.13351419,   1.86372154],
 [  2.8655447 ,  -3.09881461,  -9.07033265,  -6.70363609,   0.47850712],
 [  3.03964478,  -2.70077946,  -7.9911041 ,  -8.19519002,   1.43099582],
 [ -5.20321467,   4.53910897,  -9.40928775,   2.83022726,   4.68383153],
 [ -3.83297455,   4.71047673,  -7.27999072,   3.5584829 ,   5.75060944]])
result=np.array([1,1,0,1,1,0,0,0,1,1,1,1,0,1,1,0,1,0,0,0,1,0,0,1,0,0,0,0,1,1])
np.place(result, result==0, [-1])#0 => -1 dönüşümü

#bias değeri eklendi
data =np.insert(data, boyut, bias, axis=1)

#eğitim ve test olarak 10-20 bölündü
X_train, X_test, y_train, y_test = train_test_split(data, result, test_size=0.33)

W=np.array([1,-1,1,1,1,1])#ilk ağırlık

adaline(X_train,y_train,X_test,y_test,W,bias,c,E)
perceptron(X_train,y_train,X_test,y_test,W,bias,c)


"""

#derste yapılan örnek çalışmalar

#adaline 
boyut=2
data=[[-3,2],[3,1],[-1,1],[3/2,3]]
data =np.insert(data,boyut, bias, axis=1)
result=[1,-1,1,-1]

test_data=np.array([[5,2],[-3/2,3]])
test_data =np.insert(test_data, boyut, bias, axis=1)
test_result=[-1,1]

W=np.array([0,-1,0])#ilk ağırlık

adaline(data,result,test_data,test_result,W,bias,c,E)



#perceptron
boyut=2
data = np.array([[-1,1],[1, 1],[2 ,-1],[-1, -1],[3, 0],[-2 ,0]])
data =np.insert(data,boyut, bias, axis=1)
result=[1,-1,-1,1,-1,1]

test_data=np.array([[12,12],[-12,-11]])
test_data =np.insert(test_data, boyut, bias, axis=1)
test_result=[-1,1]

W = np.array([1,1,1])
perceptron(data,result,test_data,test_result,W,bias,c)
  
"""    
    
    
    