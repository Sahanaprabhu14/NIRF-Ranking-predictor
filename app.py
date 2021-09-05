import numpy as np
from flask import Flask, request, render_template
import pickle
from sklearn.preprocessing import PolynomialFeatures
import math
import pandas as pd
from sklearn.preprocessing import MinMaxScaler 



scaling = MinMaxScaler()
pr = pickle.load(open('poly_reg.pkl', 'rb'))
poly_reg=PolynomialFeatures(degree=4)
#read data and normalise
pred_df = pd.read_csv('2020RankingEngg.csv')
pred_df = pred_df.head(100)
pred_df[['TLR (100)', 'RPC (100)', 'GO (100)', 'OI (100)', 'PERCEPTION (100)']] = scaling.fit_transform(pred_df[['TLR (100)', 'RPC (100)', 'GO (100)', 'OI (100)', 'PERCEPTION (100)']])

#calculate score and rank
def label_race (row):
   return row['TLR (100)'] * 0.30 + row['RPC (100)'] * 0.30 + row['GO (100)'] * 0.20 + row['OI (100)'] * 0.10 + row['PERCEPTION (100)'] * 0.10

pred_df['SCORE'] = pred_df.apply(label_race, axis=1)
pred_df = pred_df.sort_values(by=['SCORE'], ascending=False)
pred_df['Rank'] = range(1, 1+len(pred_df))

#prepare X for model
pred_X = np.array(pred_df['SCORE'])

#predicting using model(from pickle)
final_pred = pr.predict(poly_reg.fit_transform(pred_X.reshape(-1, 1)))

#find difference between prediction and actual rank
diff = []
for i in range(0, 100):
    diff.append(abs(final_pred[i] - pred_df['Rank'][i]))

#finding avg diff for every ten ranks
avg_diff = []

for i in range(0, 100, 10):
    avg_diff.append(math.ceil(sum(diff[i:i+10])/10))







app = Flask(__name__)
# model_2019 = pickle.load(open('C:/Users/user/Desktop/flask/rank_model_2019.pkl', 'rb'))
min_max =[ [[91.85,40.86],[94.68,0.68],[88.31,42.07],[82.94,48.05],[84.24,1.46]],
           [[93.83,39.58],[96.04,1.68],[88.9,35.53],[65.63,29.09],[100,0]],
           [[93.55,41.83],[96.18,2.16],[89.84,5.46],[68.5,32.1],[100,1.63]]
                
            ]

# avg_min_max =[[93.0766,40.7566],[95.633,1.5066],[89.0166,27.6866],[72.3566,36.4133],[94.766,1.03]]
avg_min_max = min_max[0]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])

def predict():
    fp=[]
    f = [float(x) for x in request.form.values()]
    print(f)
    index=0
    print(avg_min_max)
    for i in avg_min_max:
        fp.append((f[index]-i[1])/(i[0]-i[1]))
        index+=1
    print(fp)


    # fp = [f[0]/94.4,f[1]/96.93,f[2]/90.605,f[3]/61.28,f[4]/107.88]
    # fp = [f[0]/93.55,f[1]/96.18,f[2]/89.84,f[3]/68.5,f[4]/100]

    score = (0.3*(fp[0]+fp[1]))+(0.2*fp[2])+((fp[3]+fp[4])*0.1)
    poly_reg = PolynomialFeatures(degree=4)
    y = pr.predict(poly_reg.fit_transform([[score]]))[0]
    if y < 0:
        y = 1
    error = int(9 if y > 100 else y//10)
    if (int(y)-avg_diff[error]<1):
        output = (str(int(y)) + ' + ' + str(avg_diff[error]))
    else:
        output = (str(int(y)) + ' ± ' + str(avg_diff[error]))

    return render_template('index.html', prediction_text='College Rank May Be {}'.format(output))


if __name__ == "__main__":
    app.run(debug=True)