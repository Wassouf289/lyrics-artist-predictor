import argparse
from lyrics_classify_model import predict,pipeline

parser=argparse.ArgumentParser(description='This program predicts the artist of a given text \n current list of artists [Frank Sinatra,Ed Sheeran,Taylor Swift]')
parser.add_argument('text',help='give the text that you want to know who sings it')

args=parser.parse_args()

print("\n ------------------------------------------------------------")
artist,prob=predict(pipeline,[args.text])
print('\n This looks like a song from:',artist,'\n with a probability of',round(prob,2))
