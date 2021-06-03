#%%
import pandas as pd
import re
import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from bayes_opt import BayesianOptimization
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.decomposition import TruncatedSVD
from sklearn.svm import SVC

df_test = pd.read_json('test.json')
df_train = pd.read_json('train.json')
most_use = pd.read_csv('most_use.csv').rename(columns={'Unnamed: 0':'ingredients'})
dump_img_record=[]
# preprocessing
def preprocess_df(df):
    nltk.download('wordnet')
    def process_string(x):
        x = [" ".join([WordNetLemmatizer().lemmatize(q) for q in p.split()]) for p in x] #Lemmatization ,list of ing.
        x = list(map(lambda x: re.sub(r'\(.*oz.\)|crushed |crumbles |ground |minced |powder |chopped |sliced ','', x), x)) # delete 處理方式,如:粉,丁,條狀..等等
        x = list(map(lambda x: re.sub("[^a-zA-Z]", " ", x), x))   # To remove everything except a-z and A-Z
        x = " ".join(x)                                 # To make list element a string element 
        x = x.lower()
        return x
    
    df = df.drop('id',axis=1)
    df['ingredients'] = df['ingredients'].apply(process_string)
    
    return df

def get_cuisine_cumulated_ingredients(df):
    cuisine_df = pd.DataFrame(columns=['ingredients'])

    for cus in df["cuisine"].unique():
        st = ""
        for x in df[df.cuisine == cus]['ingredients']:
            st += x
            st += " "
        cuisine_df.loc[cus,'ingredients'] = st

    cuisine_df = cuisine_df.reset_index()
    cuisine_df = cuisine_df.rename(columns ={'index':'cuisine'})
    return cuisine_df


def tfidf_vectorizer(train, test=None):
    """
    TFiDF Vectorizer
    """
    tfidf = TfidfVectorizer(stop_words='english',
                             ngram_range = ( 1 , 1 ),analyzer="word", 
                             max_df = .57 , binary=False , token_pattern=r'\w+' , sublinear_tf=False)
    train = tfidf.fit_transform(train)
    if test is not None:
        test = tfidf.transform(test)
        return train, test, tfidf
    else:
        return train, tfidf

def evalfn(C, gamma):
    s = SVC(C=float(C), gamma=float(gamma), kernel='rbf', class_weight='balanced')
    f = cross_val_score(s, train_tfidf, target, cv=5, scoring='f1_micro')
    return f.max()

# delete dump_ing
def delete_ingredients(df,dump_ing):
    df_clean = df.copy()
    for per_cuisine in df_clean['ingredients']:
            for ing in dump_ing:
                if ing in per_cuisine: per_cuisine.remove(ing)

    return df_clean

#%%
def test_dump_ing(i):
    dump_ing = most_use['ingredients'][:i]
    df_train_clean = delete_ingredients(df_train,dump_ing)
    df_test_clean = delete_ingredients(df_test,dump_ing)


    df = preprocess_df(df_train_clean) # shape=(39774,2)
    test_df = preprocess_df(df_test_clean) #(9944,2)
    cuisine_df = get_cuisine_cumulated_ingredients(df) # (20,total_ing)

    train = df['ingredients']
    target = df['cuisine']
    test = test_df['ingredients']

    train_tfidf, test_tfidf, tfidf = tfidf_vectorizer(train,test)
    cuisine_data_tfidf, cuisine_tfidf = tfidf_vectorizer(cuisine_df['ingredients'])


    C = 604.5300203551828
    gamma = 0.9656489284085462
    para={'decomp__n_components':[2000]}
    pipe = Pipeline([('decomp',TruncatedSVD()), ('classify', SVC(C=float(C), gamma=float(gamma), kernel='rbf'))])
    grid = GridSearchCV(pipe,para, n_jobs=-1, scoring='f1_micro')
    # fit the gridsearch object
    grid.fit(train_tfidf, target)
    # get our results
    print(grid.best_score_, grid.best_params_)

    y_pred = grid.predict(test_tfidf)
    from datetime import datetime
    now = datetime.now()
    test_ids = df_test['id']
    my_submission = pd.DataFrame({'id':test_ids})
    my_submission['cuisine'] = y_pred
    date_time = now.strftime("%Y_%m_%d__%H_%M_%S")
    my_submission.to_csv('submission_{}_{}.csv'.format(str(i),date_time), index=False)
    print('Saved file to disk as submission_{}.csv.'.format(date_time))

    dump_img_record.append(grid.best_score_)

# %%
for i in np.arange(3,10):
    test_dump_ing(i)

# %%

# %%
# %%
