
def path_concat(path1, path2, path_out):
    list1 = open(path1).readlines()
    list2 = open(path2).readlines()
    list_final = list1 + list2
    with open(path_out, 'w', encoding='utf-8') as f:  # txtname 根据具体所需的命名即可
        for i in range(0, len(list_final)):
            print(list_final[i])
            f.write(list_final[i])

def paths_concat(path1, path2, path3, path4, path5, path6, path7, path_out):
    list1 = open(path1).readlines()
    list2 = open(path2).readlines()
    list3 = open(path3).readlines()
    list4 = open(path4).readlines()
    list5 = open(path5).readlines()
    list6 = open(path6).readlines()
    list7 = open(path7).readlines()
    list_final = list1 + list2 + list3 + list4 + list5 + list6 + list7
    with open(path_out, 'w', encoding='utf-8') as f:  # txtname 根据具体所需的命名即可
        for i in range(0, len(list_final)):
            print(list_final[i])
            f.write(list_final[i])

def label_split(path):
    list = open(path).readlines()
    path_pos = []
    path_sur = []
    path_neg = []
    for i in list:
        if i.split()[-1] == '0':
            path_pos.append(i)
        elif i.split()[-1] == '1':
            path_sur.append(i)
        else:
            path_neg.append(i)

    with open("data/data_Positive.txt", 'w', encoding='utf-8') as f:  # txtname 根据具体所需的命名即可
        for i in range(0, len(path_pos)):
            f.write(path_pos[i])

    with open("data/data_Surprise.txt", 'w', encoding='utf-8') as f:  # txtname 根据具体所需的命名即可
        for i in range(0, len(path_sur)):
            f.write(path_sur[i])

    with open("data/data_Negtive.txt", 'w', encoding='utf-8') as f:  # txtname 根据具体所需的命名即可
        for i in range(0, len(path_neg)):
            f.write(path_neg[i])



if __name__ == "__main__":
    # path1 = "data_new/CASME2_new.txt"
    # path2 = "data_new/CASME2_SMIC_VIS.txt"

    # path1 = "data_new_4/SMIC_VIS_CASME2_final.txt"
    # path2 = "data_new/SMIC_VIS_CASME2_5.txt"
    # path_out = "data_new_5/SMIC_VIS_CASME2_final.txt"
    # path1 = "data/4DME.txt"
    # path2 = "data/CASME2.txt"
    # path3 = "data/MMEW.txt"
    # path4 = "data/SAMM.txt"
    # path5 = "data/SMIC_HS.txt"
    # path6 = "data/SMIC_NIR.txt"
    # path7 = "data/SMIC_VIS.txt"
    # path_out = "data/data_ALL.txt"
    # paths_concat(path1, path2, path3, path4, path5, path6, path7, path_out)
    # label_split(path_out)
    path1 = "data/data_ALL.txt"
    path2 = "data/CASME3.txt"
    path_out = "data/data_ALL_6.txt"

    path_concat(path1, path2, path_out)

