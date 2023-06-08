import pandas as pd
pd.set_option('display.max_colwidth', -1)
import matplotlib.pyplot as plt
import spacy
import warnings
import pickle
import re
warnings.filterwarnings(action='ignore')
from tqdm import tqdm

def prompt(pair):
    '''
    input: [("Translate 'Hello' to French.", "Translate 'Hello' to French.", sentiment), ("Translate 'Goodbye' to French.", "Au revoir", sentiment)]
    return examples = [("Translate 'Hello' to French.", "Bonjour"), ("Translate 'Goodbye' to French.", "Au revoir")]
    '''
    dataset = []
    for p in pair:
        p = list(p)
        new_source_string = re.sub(r'@\w+', '', p[0])
        new_source_string = re.sub(r'http\S+', '', new_source_string)
        
        new_target_string = re.sub(r'@\w+', '', p[1])
        new_target_string = re.sub(r'http\S+', '', new_target_string)
        if (new_source_string == '') or (new_target_string == ''):
            continue
        dataset.append(("Rewrite '" + new_source_string + "' to convey a '" + p[2] + "' emotion", new_target_string))

    print(dataset)
    with open('dataset.pickle', 'wb') as f:
        pickle.dump(dataset, f)
    
def df_analysis(emo_dict):
    df = pd.read_csv('tweet_emotions.csv')

    for idx in range(len(df)):
        if df['sentiment'][idx] == 'relief':
            df['sentiment'][idx] = 'neutral'
        elif df['sentiment'][idx] == 'boredom':
            df['sentiment'][idx] = 'neutral'
        elif df['sentiment'][idx] == 'happiness':
            df['sentiment'][idx] = 'happy'
        elif df['sentiment'][idx] == 'love':
            df['sentiment'][idx] = 'happy'
        elif df['sentiment'][idx] == 'fun':
            df['sentiment'][idx] = 'happy'
        elif df['sentiment'][idx] == 'sadness':
            df['sentiment'][idx] = 'sad'
        elif df['sentiment'][idx] == 'enthusiasm':
            df['sentiment'][idx] = 'surprise'
        elif df['sentiment'][idx] == 'worry':
            df['sentiment'][idx] = 'fear'
        elif df['sentiment'][idx] == 'hate':
            df['sentiment'][idx] = 'contempt'
    df = df[df['sentiment']!='empty']
    print(df)
    emotion_list = df['sentiment'].tolist()
    
    print(df['sentiment'].value_counts())
    plt.hist(emotion_list)
    plt.xticks(rotation=45)
    plt.savefig('hist.png')
    return df
    
def text_similarity(df):
    '''
    input: df
    return: pair = [("Translate 'Hello' to French.", "Translate 'Hello' to French.", sentiment), ("Translate 'Goodbye' to French.", "Au revoir", sentiment)]
    '''
    spacyModel = spacy.load('en_core_web_lg')
    
    neutral = df[df['sentiment']=='neutral']['content'].values
    target = df[df['sentiment']!='neutral']['content'].values
    sentiment = df[df['sentiment']!='neutral']['sentiment'].values
    print(neutral)
    print(target)
    print(sentiment)
    
    # get similarity
    pbar3 = tqdm(neutral, position=0, leave=True, desc='neutral')
    pbar4 = tqdm(target, position=0, leave=True, desc='target')
    list1SpacyDocs = [spacyModel(x) for x in pbar3]
    list2SpacyDocs = [spacyModel(x) for x in pbar4]
    
    # list1 = ["Hi, first example 1", "Hi, first example 1"]
    # list2 = ["Now, second example","hello, a new example 1 in the third row","And now something completely different"]

    # list1SpacyDocs = [spacyModel(x) for x in list1]
    # list2SpacyDocs = [spacyModel(x) for x in list2]
    
    pbar1 = tqdm(list1SpacyDocs, position=1, leave=False, desc='source')
    similarityMatrix = [[x.similarity(y) for x in list2SpacyDocs] for y in pbar1]
    # print(similarityMatrix)
    
    # get pair
    pair = []
    for i, row in enumerate(similarityMatrix):
        source_idx = i
        target_idx = row.index(max(row))
        pair.append((neutral[source_idx], target[target_idx], sentiment[target_idx]))
    
    with open('pair.pickle', 'wb') as f:
        pickle.dump(pair, f)

def clear():
    with open('pair.pickle', 'rb') as f:
        pair = pickle.load(f)
        
    clear_pair = []
    for p in pair:
        p = list(p)
        new_source_string = re.sub(r'@\w+', '', p[0])
        new_source_string = re.sub(r'http\S+', '', new_source_string)
        
        new_target_string = re.sub(r'@\w+', '', p[1])
        new_target_string = re.sub(r'http\S+', '', new_target_string)
        if (new_source_string == '') or (new_target_string == ''):
            continue
        clear_pair.append((new_source_string, new_target_string, p[2]))
    
    with open('clear_pair.pickle', 'wb') as f:
        pickle.dump(clear_pair, f)
        
        
if __name__=="__main__":
    # emo_dict={'relief':'neutral',
    #         'boredom':'neutral',
    #         'happiness':'happy',
    #         'love':'happy',
    #         'fun':'happy',
    #         'sadness':'sad',
    #         'enthusiasm':'sad',
    #         'worry':'fear',
    #         'hate':'contempt',
    #         }
    # df = df_analysis(emo_dict)
    # df.to_csv('tweet_emotions_7.csv')
    df = pd.read_csv('tweet_emotions_7.csv')
    text_similarity(df)
    
    with open('pair.pickle', 'rb') as f:
        pair = pickle.load(f)
    print(pair)
    prompt(pair)
    
    #clear()

