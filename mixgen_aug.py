import cv2
import numpy
import pandas as pd

# given a dataframe do mixgen augementation
def mixGen(df, path_dir, lb=0.5):
    batch = df.values.tolist()
    
    aug = []
    for i in range(len(batch) - 1):
        
        im1 = cv2.imread(path_dir + str(batch[i][0]) + '.jpg')
        im2 = cv2.imread(path_dir + str(batch[i+1][0]) + '.jpg')
       
        im2 = cv2.resize(im2, (im1.shape[1], im1.shape[0]))

        # mix of two images
        im3 = im1 * lb + im2 * (1 - lb)
        im3 = im3.astype(numpy.uint8)
        
        # string concatenation
        title = batch[i][1] + ' ' + batch[i+1][1] 
        txt = batch[i][2] + ' ' + batch[i+1][2] 

        id = str(batch[i][0]) + str(batch[i+1][0]) + '_'

        cv2.imwrite(path_dir+id+'.jpg', im3) 
        
        # warning, 0 label embedded, change it in case
        aug.append([id, title, txt, 0])

    return aug

if __name__ == '__main__':
    # path to csv file
    file_path = 'Recovery\\train.csv'

    # path to image dir
    path_dir = 'Recovery\\images\\'

    df = pd.read_csv(file_path, sep=',').drop(['Unnamed: 0'], axis=1)

    df_rel = df[df['label'] == 0]

    df_unrel = df[df['label'] == 1]

    # augmentation only for 0 label
    a = mixGen(df_rel, path_dir)
    # creation of dataframe from list
    b = pd.DataFrame(a, columns=['id', 'title', 'text', 'label'])

    final = pd.concat([pd.concat([df_rel, b]), df_unrel])
    
    final.to_csv('Recovery/train_aug.csv')






