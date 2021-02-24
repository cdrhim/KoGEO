import pandas as pd
import os
import csv
import json

def make_generationfile():
    mawps = {}
    mawpsorig = {}


    root_path = r'/dataset/mawps'
    title_paths = [r'mawps_fold0_test{}.jsonl',r'mawps_fold0_train{}.jsonl']
    for title_path in title_paths:
        datapath = os.path.join(root_path,title_path.format('.orig'))
        with open(datapath, "r", encoding="utf-8-sig") as f:
            reader = csv.reader(f, delimiter="\t")
            for n, line in enumerate(reader):
                line = json.loads(line[0])
                mawpsorig[line['original']['sQuestion'].replace(' ','')] = line

        datapath = os.path.join(root_path,title_path.format(''))
        with open(datapath, "r", encoding="utf-8-sig") as f:
            reader = csv.reader(f, delimiter="\t")
            for n, line in enumerate(reader):
                line = json.loads(line[0])
                mawps[line['id']] = line

    CCroot = r'/home/dg/PycharmProjects/Coling_JournalofSuperComputing/dataset/CC'
    ILroot = r'/home/dg/PycharmProjects/Coling_JournalofSuperComputing/dataset/IL'

    CCfolder = os.path.join(CCroot, 'tokens')
    ILfolder = os.path.join(ILroot, 'tokens')

    os.makedirs(CCfolder, exist_ok=True)
    os.makedirs(ILfolder, exist_ok=True)

    with open(os.path.join(CCfolder,"CC_fold0_train.jsonl"), "w") as cctoken:
        with open(r'/dataset/CC/CC.json', "r") as datapath:
            datas = json.load(datapath)
            for data in datas:
                data = mawpsorig[data['sQuestion'].replace(' ','')]
                try:
                    del data['original']
                except:
                    pass
                cctoken.write(json.dumps(data)+'\n')

    with open(os.path.join(ILfolder,"IL_fold0_train.jsonl"), "w") as cctoken:
        with open(r'/dataset/IL/IL.json', "r") as datapath:
            datas = json.load(datapath)
            for data in datas:
                try:
                    data = mawpsorig[data['sQuestion'].replace(' ','')]
                except:
                    data = mawpsorig[data['sQuestion'].replace(' ','').replace('-','')]
                try:
                    del data['original']
                except:
                    pass
                cctoken.write(json.dumps(data)+'\n')

    return True
if __name__ == "__main__":
    make_generationfile()
    print("")