import os
def count_labels(raf_path):
    print('||||||||||||||||||||||||||', raf_path.split('.')[0], '||||||||||||||||||||||||||')
    imagelists = open(raf_path).readlines()
    len_ = len(imagelists)
    labels = {'0': 0, '1': 0, '2': 0}
    for i in range(0, len_):
        l = str(imagelists[i].split()[-1])
        labels[l] = labels[l] + 1
    print(labels)

if __name__ == '__main__':
    casme2_path = 'data/CASME2.txt'
    smic_hs_path = 'data/SMIC_HS.txt'
    count_labels(casme2_path)
    count_labels(smic_hs_path)