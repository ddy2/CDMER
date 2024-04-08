import cv2
import os
import pandas as pd
import random

# CASME2 dataset
def CASME2():
    raf_path = 'datasets/CASME2'

    SUBJECT_COLUMN = 0
    NAME_COLUMN = 1
    ONSET_COLUMN = 2
    APEX_COLUMN = 3
    OFF_COLUMN = 4
    LABEL_AU_COLUMN = 5
    LABEL_ALL_COLUMN = 6

    # df = pd.read_excel(os.path.join(self.raf_path, 'CASME2-coding-20190701.xlsx'),usecols=[0,1,3,4,5,7,8])
    df = pd.read_excel(os.path.join(raf_path, 'CASME2-coding-20140508.xlsx'), usecols=[0, 1, 3, 4, 5, 7, 8])
    df['Subject'] = df['Subject'].apply(str)
    dataset = df

    Subject = dataset.iloc[:, SUBJECT_COLUMN].values
    File_names = dataset.iloc[:, NAME_COLUMN].values
    Label_all = dataset.iloc[:,
                LABEL_ALL_COLUMN].values  # 0:Surprise, 1:Fear, 2:Disgust, 3:Happiness, 4:Sadness, 5:Anger, 6:Neutral
    Onset_num = dataset.iloc[:, ONSET_COLUMN].values
    Apex_num = dataset.iloc[:, APEX_COLUMN].values
    Offset_num = dataset.iloc[:, OFF_COLUMN].values
    Label_au = dataset.iloc[:, LABEL_AU_COLUMN].values
    file_paths_on = []
    file_paths_apex = []
    # use aligned images for training/testing
    for (f, sub, onset, apex, offset, label_all, label_au) in zip(File_names, Subject, Onset_num, Apex_num,
                                                                  Offset_num, Label_all, Label_au):

        if label_all == 'happiness' or label_all == 'repression' or label_all == 'disgust' or label_all == 'surprise' or label_all == 'fear' or label_all == 'sadness':

            file_paths_on.append(onset)
            file_paths_apex.append(apex)
            path_on = os.path.join(raf_path, 'Cropped-updated', 'Cropped', 'sub%02d' % int(sub), f, 'reg_img' + str(onset) + '.jpg')
            path_apex = os.path.join(raf_path, 'Cropped-updated', 'Cropped', 'sub%02d' % int(sub), f, 'reg_img' + str(apex) + '.jpg')
            img_on = cv2.imread(path_on)
            img_on = cv2.resize(img_on, (224, 224), interpolation=cv2.INTER_LANCZOS4)
            cv2.imwrite('dataset/CASME2/' + sub + '_' + f + '_' + 'reg_img' + str(onset) + '.jpg', img_on)
            img_apex = cv2.imread(path_apex)
            img_apex = cv2.resize(img_apex, (224, 224), interpolation=cv2.INTER_LANCZOS4)
            cv2.imwrite('dataset/CASME2/' + sub + '_' + f + '_' + 'reg_img' + str(apex) + '.jpg', img_apex)
def SAMM():
    raf_path = 'SAMM'
    df = pd.read_excel(os.path.join(raf_path, 'SAMM_Micro_FACS_Codes_v2.xlsx'),
                       converters={'Subject': str, 'Filename': str, 'Onset Frame': str, 'Apex Frame': str},
                       usecols=[0, 1, 3, 4, 5, 8, 9])
    # df['Subject'] = df['Subject'].apply(str)
    dataset = df
    SUBJECT_COLUMN = 0
    NAME_COLUMN = 1
    ONSET_COLUMN = 2
    APEX_COLUMN = 3
    OFF_COLUMN = 4
    LABEL_AU_COLUMN = 5
    LABEL_ALL_COLUMN = 6
    Subject = dataset.iloc[:, SUBJECT_COLUMN].values
    File_names = dataset.iloc[:, NAME_COLUMN].values
    Label_all = dataset.iloc[:,
                LABEL_ALL_COLUMN].values  # 0:Surprise, 1:Fear, 2:Disgust, 3:Happiness, 4:Sadness, 5:Anger, 6:Neutral
    Onset_num = dataset.iloc[:, ONSET_COLUMN].values
    Apex_num = dataset.iloc[:, APEX_COLUMN].values
    Offset_num = dataset.iloc[:, OFF_COLUMN].values
    Label_au = dataset.iloc[:, LABEL_AU_COLUMN].values

    file_paths_on = []
    file_paths_apex = []
    file_paths_on2 = []
    file_paths_apex2 = []
    label = []
    for (f, sub, onset, apex, offset, label_all, label_au) in zip(File_names, Subject, Onset_num, Apex_num,
                                                                  Offset_num, Label_all, Label_au):
        if label_all == 'Happiness' or label_all == 'Contempt' or label_all == 'Disgust' or label_all == 'Surprise' or label_all == 'Fear' or label_all == 'Sadness' or label_all == 'Anger':

            file_paths_on.append(os.path.join(raf_path, sub, f, sub + '_' + str(onset).zfill(5) + '.jpg'))
            file_paths_apex.append(os.path.join(raf_path, sub, f, sub + '_' + str(apex).zfill(5) + '.jpg'))
            file_paths_on2.append(os.path.join(raf_path, sub, f, sub + '_' + str(onset).zfill(5) + '.jpg'))
            file_paths_apex2.append(os.path.join(raf_path, sub, f, sub + '_' + str(apex).zfill(5) + '.jpg'))

            if label_all == 'Happiness':
                label.append(0)
            elif label_all == 'Surprise':
                label.append(1)
            else:
                label.append(2)


    with open('data/SAMM.txt', 'w', encoding='utf-8') as f:  # txtname 根据具体所需的命名即可
        for i in range(0, len(file_paths_on)):
            f.write(file_paths_on[i] + '\t' + file_paths_apex[i] + '\t' +
                    file_paths_on2[i] + '\t' + file_paths_apex2[i] + '\t' + str(label[i]) + '\n')


def CASME222():
    raf_path = 'Cas(me)^2'
    data = pd.read_excel(os.path.join(raf_path, 'CAS(ME)^2.xlsx'), usecols=[0, 1, 2, 3, 5, 7, 8])
    print(data)
    dataset = data
    SUBJECT_COLUMN = 0
    NAME_COLUMN = 1
    ONSET_COLUMN = 2
    APEX_COLUMN = 3
    OFF_COLUMN = 4
    LABEL_AU_COLUMN = 5
    LABEL_ALL_COLUMN = 6
    Subject = dataset.iloc[:, SUBJECT_COLUMN].values
    File_names = dataset.iloc[:, NAME_COLUMN].values
    Label_all = dataset.iloc[:,
                LABEL_ALL_COLUMN].values  # 0:Surprise, 1:Fear, 2:Disgust, 3:Happiness, 4:Sadness, 5:Anger, 6:Neutral
    Onset_num = dataset.iloc[:, ONSET_COLUMN].values
    Apex_num = dataset.iloc[:, APEX_COLUMN].values
    Offset_num = dataset.iloc[:, OFF_COLUMN].values

    file_paths_on = []
    file_paths_apex = []
    file_paths_on2 = []
    file_paths_apex2 = []
    label = []
    for (f, sub, onset, apex, offset, label_all) in zip(File_names, Subject, Onset_num, Apex_num,
                                                                  Offset_num, Label_all):


        path_on = os.path.join(raf_path, 'cropped', str(sub), f, 'img' + str(onset) + '.jpg')
        path_apex = os.path.join(raf_path, 'cropped', str(sub), f, 'img' + str(apex) + '.jpg')
        img_on = cv2.imread(path_on)
        img_on = cv2.resize(img_on, (224, 224), interpolation=cv2.INTER_LANCZOS4)
        print('dataset/CASME222/' + str(sub) + '__' + f + '__' + 'img' + str(onset) + '.jpg')
        cv2.imwrite('dataset/CASME222/' + str(sub) + '__' + f + '__' + 'img' + str(onset) + '.jpg', img_on)
        img_apex = cv2.imread(path_apex)
        img_apex = cv2.resize(img_apex, (224, 224), interpolation=cv2.INTER_LANCZOS4)
        cv2.imwrite('dataset/CASME222/' + str(sub) + '__' + f + '__' + 'img' + str(apex) + '.jpg', img_apex)
        file_paths_on.append(path_on)
        file_paths_apex.append(path_apex)
        label.append(label_all)

    with open('data/CASME222.txt', 'w', encoding='utf-8') as f:  # txtname 根据具体所需的命名即可
        for i in range(0, len(file_paths_on)):
            f.write(file_paths_on[i] + '\t' + file_paths_apex[i] + '\t' + str(label[i]) + '\n')

def SMIC():
    raf_path = 'SMIC_crop'
    data = pd.read_excel(os.path.join(raf_path, 'SMIC_HS_E_spotting.xlsx'))
    file_paths_on = []
    file_paths_apex = []
    label = []

    for index, row in data.iterrows():
        if (row['NumME'] > 0):
            onset = 8
            offset = 9
            me_type = 7
            for i in range(0, row['NumME']):
                path_on = os.path.join(raf_path, 'cropped', 's'+str(row['subject']).zfill(2), row['video_name'],
                                       'image' + str(row[onset]).split('.')[0].zfill(6) + '.jpg')
                path_apex = os.path.join(raf_path, 'cropped', 's'+str(row['subject']).zfill(2), row['video_name'],
                                       'image' + str((int(row[onset])+int(row[offset]))/2).split('.')[0].zfill(6) + '.jpg')
                file_paths_on.append(path_on)
                file_paths_apex.append(path_apex)
                if row[me_type].split('_')[-2] == 'po':
                    label.append(0)
                elif row[me_type].split('_')[-2] == 'sur':
                    label.append(1)
                else:
                    label.append(2)
                onset = onset + 3
                offset = offset + 3
                me_type = me_type + 3

    with open('data/SMIC.txt', 'w', encoding='utf-8') as f:  # txtname 根据具体所需的命名即可
        for i in range(0, len(file_paths_on)):
            f.write(file_paths_on[i] + '\t' + file_paths_apex[i] + '\t' + file_paths_on[i] + '\t' + file_paths_apex[i] + '\t' + str(label[i]) + '\n')
def SMIC_all():
    raf_path = 'SMIC_all_cropped/NIR'
    file_paths_on = []
    file_paths_apex = []
    label = []
    subjects = os.listdir(raf_path)
    for sub in subjects:
        sub_path = os.path.join(raf_path, sub, 'micro')
        expressions = os.listdir(sub_path)
        for exp in expressions:
            if exp == 'positive':
                l = 0
            elif exp == 'surprise':
                l = 1
            else:
                l = 2
            exp_path = os.path.join(sub_path, exp)
            files = os.listdir(exp_path)
            for file in files:
                filename = os.listdir(os.path.join(exp_path, file))
                filename.sort()
                path_on = os.path.join(exp_path, file, filename[0])
                path_apex = os.path.join(exp_path, file, 'reg_'+str(int((int(filename[0].split('.')[0].strip('reg_image')) + int(filename[-1].split('.')[0].strip('reg_image')))/2)).zfill(5)+'.bmp')
                file_paths_on.append(path_on)
                file_paths_apex.append(path_apex)
                label.append(l)


    with open('data/SMIC_NIR.txt', 'w', encoding='utf-8') as f:  # txtname 根据具体所需的命名即可
        for i in range(0, len(file_paths_on)):
            f.write(file_paths_on[i] + '\t' + file_paths_apex[i] + '\t' + file_paths_on[i] + '\t' + file_paths_apex[i] + '\t' + str(label[i]) + '\n')
def casme_new():
    raf_path = 'datasets/CASME2'
    df = pd.read_excel(os.path.join(raf_path, 'CASME2-coding-20140508.xlsx'), usecols=[0, 1, 3, 4, 5, 7, 8])
    df['Subject'] = df['Subject'].apply(str)
    dataset = df
    SUBJECT_COLUMN = 0
    NAME_COLUMN = 1
    ONSET_COLUMN = 2
    APEX_COLUMN = 3
    OFF_COLUMN = 4
    LABEL_AU_COLUMN = 5
    LABEL_ALL_COLUMN = 6
    Subject = dataset.iloc[:, SUBJECT_COLUMN].values
    File_names = dataset.iloc[:, NAME_COLUMN].values
    Label_all = dataset.iloc[:,
                LABEL_ALL_COLUMN].values  # 0:Surprise, 1:Fear, 2:Disgust, 3:Happiness, 4:Sadness, 5:Anger, 6:Neutral
    Onset_num = dataset.iloc[:, ONSET_COLUMN].values
    Apex_num = dataset.iloc[:, APEX_COLUMN].values
    Offset_num = dataset.iloc[:, OFF_COLUMN].values
    Label_au = dataset.iloc[:, LABEL_AU_COLUMN].values

    file_paths_on = []
    file_paths_apex = []
    file_paths_on2 = []
    file_paths_apex2 = []
    label = []
    for (f, sub, onset, apex, offset, label_all, label_au) in zip(File_names, Subject, Onset_num, Apex_num,
                                                                  Offset_num, Label_all, Label_au):
        if label_all == 'happiness' or label_all == 'disgust' or label_all == 'surprise' or label_all == 'fear' or label_all == 'sadness':
            file_paths_on.append(os.path.join(raf_path, 'Cropped-updated', 'Cropped', 'sub%02d' % int(sub), f, 'reg_img' + str(onset) + '.jpg'))
            file_paths_apex.append(os.path.join(raf_path, 'Cropped-updated', 'Cropped', 'sub%02d' % int(sub), f, 'reg_img' + str(apex) + '.jpg'))
            file_paths_on2.append(os.path.join(raf_path, 'CASME2_RAW', 'sub%02d' % int(sub), f,
                                              'img' + str(onset) + '.jpg'))
            file_paths_apex2.append(os.path.join(raf_path, 'CASME2_RAW', 'sub%02d' % int(sub), f,
                                                'img' + str(apex) + '.jpg'))
            if label_all == 'happiness':
                label.append(0)
            elif label_all == 'surprise':
                label.append(1)
            else:
                label.append(2)

    with open('data/CASME2.txt', 'w', encoding='utf-8') as f:  # txtname 根据具体所需的命名即可
        for i in range(0, len(file_paths_on)):
            f.write(file_paths_on[i] + '\t' + file_paths_apex[i] + '\t' +
                    file_paths_on2[i] + '\t' + file_paths_apex2[i] + '\t'+ str(label[i]) + '\n')

def FDME():
    raf_path = '4DME'
    path = "micro_short_gray_video_cropped"
    df = pd.read_excel(os.path.join(raf_path, 'Micro_and_Macro-expression_labels_new_2D_oneLabel.xlsx'),
                       converters={'SubID': str, 'fold_name': str, 'Onset': str, 'Apex': str},
                       usecols=[1, 2, 3, 4, 5, 8, 9, 13])
    # df['Subject'] = df['Subject'].apply(str)
    dataset = df
    SUBID = dataset.iloc[:, 0].values
    VideoNo = dataset.iloc[:, 1].values
    longclipNo = dataset.iloc[:, 2].values
    MENo = dataset.iloc[:, 3].values
    foldname = dataset.iloc[:, 4].values
    Onset = dataset.iloc[:, 5].values
    Apex = dataset.iloc[:, 6].values
    Emotion = dataset.iloc[:, 7].values
    
    file_paths_on = []
    file_paths_apex = []
    file_paths_on2 = []
    file_paths_apex2 = []
    label = []
    for (sub, video, clip, me, fold, onset, apex, label_all) in zip(SUBID, VideoNo, longclipNo, MENo,
                                                                  foldname, Onset, Apex, Emotion):
        if label_all == 'Positive' or label_all == 'Surprise' or label_all == 'Negative':
            file_paths_on.append(os.path.join(raf_path, path, sub, fold, 'Frame_' + str(onset).zfill(9) + '.jpg'))
            file_paths_apex.append(os.path.join(raf_path, path, sub, fold, 'Frame_' + str(apex).zfill(9) + '.jpg'))
            file_paths_on2.append(os.path.join(raf_path, path, sub, fold, 'Frame_' + str(onset).zfill(9) + '.jpg'))
            file_paths_apex2.append(os.path.join(raf_path, path, sub, fold, 'Frame_' + str(apex).zfill(9) + '.jpg'))
            
            if label_all == 'Positive':
                label.append(0)
            elif label_all == 'Surprise':
                label.append(1)
            else:
                label.append(2)

    with open('data/4DME.txt', 'w', encoding='utf-8') as f:  # txtname 根据具体所需的命名即可
        for i in range(0, len(file_paths_on)):
            f.write(file_paths_on[i] + '\t' + file_paths_apex[i] + '\t' +
                    file_paths_on2[i] + '\t' + file_paths_apex2[i] + '\t' + str(label[i]) + '\n')
            print((file_paths_on[i] + '\t' + file_paths_apex[i] + '\t' +
                    file_paths_on2[i] + '\t' + file_paths_apex2[i] + '\t' + str(label[i]) + '\n'))


def MMEW():
    raf_path = 'MMEW'
    path = "Micro_Expression"
    df = pd.read_excel(os.path.join(raf_path, 'MMEW_Micro_Exp_new.xlsx'),
                       converters={'Subject': str, 'Filename': str, 'OnsetFrame': str, 'ApexFrame': str},
                       usecols=[0, 1, 2, 3, 6])
    # df['Subject'] = df['Subject'].apply(str)
    dataset = df
    Subject = dataset.iloc[:, 0].values
    Filename = dataset.iloc[:, 1].values
    Onset = dataset.iloc[:, 2].values
    Apex = dataset.iloc[:, 3].values
    Emotion = dataset.iloc[:, 4].values

    file_paths_on = []
    file_paths_apex = []
    file_paths_on2 = []
    file_paths_apex2 = []
    label = []
    for (sub, fold, onset, apex, label_all) in zip(Subject, Filename, Onset, Apex, Emotion):
        if label_all == 'happiness' or label_all == 'surprise' or label_all == 'anger' or label_all == 'disgust' or label_all == 'fear' or label_all == 'sadness':
            file_paths_on.append(os.path.join(raf_path, path, fold, str(onset) + '.jpg'))
            file_paths_apex.append(os.path.join(raf_path, path, fold, str(apex) + '.jpg'))
            file_paths_on2.append(os.path.join(raf_path, path, fold, str(onset) + '.jpg'))
            file_paths_apex2.append(os.path.join(raf_path, path, fold, str(apex) + '.jpg'))

            if label_all == 'happiness':
                label.append(0)
            elif label_all == 'surprise':
                label.append(1)
            else:
                label.append(2)

    with open('data/MMEW.txt', 'w', encoding='utf-8') as f:  # txtname 根据具体所需的命名即可
        for i in range(0, len(file_paths_on)):
            f.write(file_paths_on[i] + '\t' + file_paths_apex[i] + '\t' +
                    file_paths_on2[i] + '\t' + file_paths_apex2[i] + '\t' + str(label[i]) + '\n')
            print((file_paths_on[i] + '\t' + file_paths_apex[i] + '\t' +
                   file_paths_on2[i] + '\t' + file_paths_apex2[i] + '\t' + str(label[i]) + '\n'))

def CASME3():
    raf_path = 'CASME^3/part_A'
    path = "data/part_A_split/part_A_short_cropped"
    df = pd.read_excel(os.path.join(raf_path, 'annotation/cas(me)3_part_A_ME_label_JpgIndex_v2.xlsx'),
                       converters={'Subject': str, 'Filename': str, 'Onset': str, 'Apex': str},
                       usecols=[0, 1, 2, 3, 7])
    # df['Subject'] = df['Subject'].apply(str)
    dataset = df
    Subject = dataset.iloc[:, 0].values
    Filename = dataset.iloc[:, 1].values
    Onset = dataset.iloc[:, 2].values
    Apex = dataset.iloc[:, 3].values
    Emotion = dataset.iloc[:, 4].values

    file_paths_on = []
    file_paths_apex = []
    file_paths_on2 = []
    file_paths_apex2 = []
    label = []
    for (sub, fold, onset, apex, label_all) in zip(Subject, Filename, Onset, Apex, Emotion):
        if label_all == 'happy' or label_all == 'surprise' or label_all == 'anger' or label_all == 'disgust' or label_all == 'fear' or label_all == 'sad':
            file_paths_on.append(os.path.join(raf_path, path, sub, fold, 'color', str(onset) + '.jpg'))
            file_paths_apex.append(os.path.join(raf_path, path, sub, fold, 'color', str(apex) + '.jpg'))
            file_paths_on2.append(os.path.join(raf_path, path, sub, fold, 'color', str(onset) + '.jpg'))
            file_paths_apex2.append(os.path.join(raf_path, path, sub, fold, 'color', str(apex) + '.jpg'))

            if label_all == 'happy':
                label.append(0)
            elif label_all == 'surprise':
                label.append(1)
            else:
                label.append(2)

    with open('data/CASME3.txt', 'w', encoding='utf-8') as f:  # txtname 根据具体所需的命名即可
        for i in range(0, len(file_paths_on)):
            f.write(file_paths_on[i] + '\t' + file_paths_apex[i] + '\t' +
                    file_paths_on2[i] + '\t' + file_paths_apex2[i] + '\t' + str(label[i]) + '\n')
            print((file_paths_on[i] + '\t' + file_paths_apex[i] + '\t' +
                   file_paths_on2[i] + '\t' + file_paths_apex2[i] + '\t' + str(label[i]) + '\n'))

def DFEW():
    raf_path = 'DFEW/DFEW/DFEW/Clip/clip_224x224'
    file_paths_on = []
    file_paths_apex = []
    file_paths_on2 = []
    file_paths_apex2 = []
    label = []
    for part in os.listdir(raf_path):
        if not part.split('.')[-1] == 'zip':
            for file in os.listdir(os.path.join(raf_path, part)):
                list = os.listdir(os.path.join(raf_path, part, file))
                file_list = sorted(list)
                for i in range(50):
                    start = random.randint(0, len(file_list)-2)
                    end = random.randint(start, len(file_list)-1)
                    start_file = file_list[start]
                    end_file = file_list[end]
                    # print(start_file)
                    file_paths_on.append(os.path.join(raf_path, part, file, start_file))
                    file_paths_apex.append(os.path.join(raf_path, part, file, end_file))
                    file_paths_on2.append(os.path.join(raf_path, part, file, start_file))
                    file_paths_apex2.append(os.path.join(raf_path, part, file, end_file))
                    label.append(0)

                # print(os.path.join(raf_path, part, file, file_list[0]))

    with open('data/DFEW.txt', 'w', encoding='utf-8') as f:  # txtname 根据具体所需的命名即可
        for i in range(0, len(file_paths_on)):
            f.write(file_paths_on[i] + '\t' + file_paths_apex[i] + '\t' +
                    file_paths_on2[i] + '\t' + file_paths_apex2[i] + '\t' + str(label[i]) + '\n')
            print((file_paths_on[i] + '\t' + file_paths_apex[i] + '\t' +
                   file_paths_on2[i] + '\t' + file_paths_apex2[i] + '\t' + str(label[i]) + '\n'))

if __name__ == "__main__":
    # CASME2()
    # SAMM()
    # CASME222()
    # SMIC()
    # SMIC_all()
    # casme_new()
    # FDME()
    # MMEW()
    # CASME3()
    DFEW()
