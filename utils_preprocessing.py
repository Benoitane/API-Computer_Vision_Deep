import os
from sklearn.model_selection import train_test_split
import random
import glob

random.seed(2042)


def checking_data_csv(df, path_img):
    """
    Check if there is a mismatch between files present in csv and those in folder

    Inputs
    --------
        df (DataFrame): dataframe to check
        path_img (str): path to the image folder

    Outputs
    --------
        df_match (DataFrame) : dataframe of observations present in both csv and folder
    """
    print("Checking if the csv is well built for the corresponding image dataset...")
    df['target'] = df['target'].apply(lambda x: str(x))
    df['image_name_only'] = df['image_name'].apply(lambda x: str(x).split('/')[1])
    df['target_only'] = df['image_name'].apply(lambda x: str(x).split('/')[0])
    list_jpg_incsv = list(set(df['image_name_only']))
    if len(list_jpg_incsv) == df.shape[0]:
        print('There is no duplicates in csv file')
    if df[df['target'] != df['target_only']].shape[0] == 0:
        print('There is no problem of missclassification in the csv file')
    list_jpg_in_folder = [x.replace(path_img + 'data/images/', '') for x in
                          list(glob.glob(path_img + 'data/images/' + "*"))]
    if len(list_jpg_incsv) != len(list_jpg_in_folder):
        print('Warning : images in csv do not match images in folder')
        list_mismatch = [x for x in list_jpg_incsv if x not in list_jpg_in_folder]
        print('There are {} images which are in csv and not in folder'.format(len(list_mismatch)))
        print('These rows are not considered in our case because we do not have observations')
    print("Re structuring csv...")
    df_match = df[~df['image_name_only'].isin(list_mismatch)]
    return df_match


def description_cleanded_data(cleaned_data):
    """
    Function to describe and separate specific cas in cleaned dataet before splitting into train, val and test set

    Inputs
    --------
        cleaned_data (DataFrame): dataframe of observations present in both csv and folder

    Outputs
    --------
        data_new (DataFrame): dataframe without classes with 1 or 2 observations
        list_target_3 (list): list of classes with only 3 observations
        data_to_split (DataFrame): dataframe with all classes and observations with more than 3 observations
        data_to_split_separately (DataFrame): dataframe with all classes and observations with 3 observations
    """
    print("Quick description of cleaned dataset...")
    grouped_data = cleaned_data.groupby('target').size().reset_index().sort_values([0], ascending=False)
    print('Number of observations observations :', str(cleaned_data.shape[0]))
    list_of_classes = list(set(cleaned_data['target']))
    print('Number of classes :', str(len(cleaned_data['target'].unique())))
    most_class = grouped_data['target'].iloc[0]
    most_class_0 = grouped_data[0].iloc[0]
    less_class = grouped_data['target'].iloc[-1]
    less_class_0 = grouped_data[0].iloc[-1]

    print('class with the largest sample size : {} (with size {})'.format(most_class, most_class_0))
    print('class with the smallest sample size : {} (with size {})'.format(less_class, less_class_0))

    print('Classes with 1 or 2 observations will be set aside because we do not have enough observations to include')
    print('them in a train-val-test model. This choice is questionable and could be discuss later in the next interview')

    list_target_1 = list(set(grouped_data[grouped_data[0] == 1].target))
    data_new = cleaned_data[~cleaned_data['target'].isin(list_target_1)]
    list_target_3 = list(set(grouped_data[grouped_data[0] == 3].target))
    data_to_split = data_new[
        ~data_new['target'].isin(list(grouped_data['target'][grouped_data[0].isin([1, 2, 3])].unique()))]
    data_to_split_separately = data_new[
        data_new['target'].isin(list(grouped_data['target'][grouped_data[0].isin([3])].unique()))]

    return list_target_3, data_new, data_to_split, data_to_split_separately



def create_train_val_test_set(data_new, data_to_split, data_to_split_separately, list_target_3):
    """
    Function to split dataset into train, val and test set

    Inputs
    --------
        data_new (DataFrame): dataframe without classes with 1 or 2 observations
        list_target_3 (list): list of classes with only 3 observations
        data_to_split (DataFrame): dataframe with all classes and observations with more than 3 observations
        data_to_split_separately (DataFrame): dataframe with all classes and observations with 3 observations

    Outputs
    --------
        train (DataFrame): train set
        val (DataFrame): validation set
        test (DataFrame): test set
        list_train_target (list): list with targets included in train set
        list_val_target (list): list with targets included in validation set
        list_test_target (list): list with targets included in test set
    """

    print('Creating assignment to train-val-test set...')
    test_s = (0.4 * data_new.shape[0]) / (data_new.shape[0] - data_to_split_separately.shape[0])
    train, val_to_split = train_test_split(data_to_split, test_size=test_s, stratify=data_to_split['target'])
    val, test = train_test_split(val_to_split, test_size=0.5, stratify=val_to_split['target'])
    temp3 = data_to_split_separately[data_to_split_separately['target'].isin(list_target_3)]
    temp3_train, temp3_val_t = train_test_split(temp3, test_size=0.66, stratify=temp3['target'])
    temp3_val, temp3_test = train_test_split(temp3_val_t, test_size=0.5, stratify=temp3_val_t['target'])

    train = train.append(temp3_train)
    val = val.append(temp3_val)
    test = test.append(temp3_test)

    print('Size of train : {}'.format(train.shape[0]))
    print('Size of validation : {}'.format(val.shape[0]))
    print('Size of test : {}'.format(test.shape[0]))

    print('Number of train targets : {}'.format(len(list(set(train.target)))))
    print('Number of validation targets : {}'.format(len(list(set(val.target)))))
    print('Number of test targets : {}'.format(len(list(set(test.target)))))

    if train.shape[0] + val.shape[0] + test.shape[0] == data_new.shape[0]:
        print('We have all observations : sum of different sets match the total number of rows (after subseting)')
    list_val_target = list(set(val.target))
    list_train_target = list(set(train.target))
    list_test_target = list(set(test.target))
    if list_val_target == list_train_target:
        print('We have the same classes in val and train set')
    return train, val, test, list_train_target, list_val_target, list_test_target


def create_and_distribute_images(parent_dir,train,val,test,list_train,list_val,list_test):
    """
    Function to split dataset into train, val and test set

    Inputs
    --------
        parent_dir (str): path to the right folder where you want to put train/val/test folder
        train (DataFrame): train set
        val (DataFrame): validation set
        test (DataFrame): test set
        list_train_target (list): list with targets included in train set
        list_val_target (list): list with targets included in validation set
        list_test_target (list): list with targets included in test set

    Outputs
    --------
        No object return. Folder train val and test created with observations into right folder class
    """
    print("Creation of folder for train-val-test...")
    list_folder = ["train","val",'test']
    for folder in list_folder:
        path = os.path.join(parent_dir, folder)
        os.mkdir(path)
        print("Directory {} created".format(folder))
    print("Creation of subfolder for each label...")
    for folder in list_folder:
        if folder == 'train':
            for class_ in list_train:
                path = os.path.join(parent_dir+'/'+folder+'/',str(class_))
                os.mkdir(path)
            print("sub folders created for {}".format(folder))
        if folder == 'val':
            for class_ in list_val:
                path = os.path.join(parent_dir+'/'+folder+'/',str(class_))
                os.mkdir(path)
            print("sub folders created for {}".format(folder))
        if folder == 'test':
            for class_ in list_test:
                path = os.path.join(parent_dir+'/'+folder+'/',str(class_))
                os.mkdir(path)
            print("sub folders created for {}".format(folder))
    path_img_train = 'data/images/train/'
    path_img_val = 'data/images/val/'
    path_img_test = 'data/images/test/'
    list_images_train = list(set(train['image_name']))
    list_images_val = list(set(val['image_name']))
    list_images_test = list(set(test['image_name']))
    for img in list_images_train:
        os.replace(parent_dir + img.split('/')[1], path_img_train + img.split('/')[0] + '/' + img.split('/')[1])
    print('files moved to the right folder for train')
    for img in list_images_val:
        os.replace(parent_dir + img.split('/')[1], path_img_val + img.split('/')[0] + '/' + img.split('/')[1])
    print('files moved to the right folder for val')
    for img in list_images_test:
        os.replace(parent_dir + img.split('/')[1], path_img_test + img.split('/')[0] + '/' + img.split('/')[1])
    print('files moved to the right folder for test')


