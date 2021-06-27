from flask import Flask,render_template,url_for,request
import os
import joblib
import numpy as np

# Vectorize
vec = open(os.path.join("ststic/models/final_news_cv_vectorizer.pkl"),"rb")
news_cv = joblib.load(vec)

app = Flask(__name__)

def get_keys(val,my_dict):
	for key,value in my_dict.items():
		if val == value:
			return key 

def max_prob(scores):
    current_max = 0
    category = ''
    
    for cat,prob in scores.items():
        if prob > current_max:
            current_max = prob
            category = cat
        else:
            pass
    return category,round(current_max,2)



@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/predict',methods=['GET','POST'])
def predict():
    if request.method == 'POST':
        rawtext = request.form['rawtext']
        vectorized_text = news_cv.transform([rawtext]).toarray()

        nb_model = open(os.path.join("ststic/models/newsclassifier_NB_model.pkl"),"rb")
        clf = joblib.load(nb_model)

        #prediction
        prediction_labels = {"Business":0,"Tech":1,"Sport":2,"Health":3,"Politics":4,"Entertainment":5}
        prediction = clf.predict(vectorized_text)  
        final_result = get_keys(prediction,prediction_labels)
        
        pred_prob = clf.predict_proba(np.array(vectorized_text).reshape(1,-1))
        pred_probalility_score = {"Business":pred_prob[0][0]*100,"Tech":pred_prob[0][1]*100, "Sport":pred_prob[0][2]*100,
        "Health":pred_prob[0][3]*100, "Politics":pred_prob[0][4]*100, "Entertainment":pred_prob[0][5]*100}

        final_score = max_prob(pred_probalility_score)
    

    return render_template('predict.html', rawtext=rawtext.upper(),final_result=final_result,final_score=final_score)

if __name__ == '__main__':
    app.run(debug=True)