import os 

def Find_Files2(start_path, name):
    flist=[]
    for relpath, dirs, files in os.walk(start_path):
        for fname in files:
            #print (fname)
            if name in fname:
                full_path = os.path.join(start_path, relpath, fname)
                flist.append(os.path.normpath(os.path.abspath(full_path)))
    return flist



def Find_Files(start_path, name):
    if os.path.exists(start_path):
        pass
    else:
        print('path not exit')
        print('error path:', start_path)
        exit('path error')
        
    flist=[] #这里加一条判断文件路径的语句
    for relpath, dirs, files in os.walk(start_path): #遍历文件夹里面所有文件
        print('relpath:', relpath) #/data/fzg/3DConformersGen/Ch_EcConf/scripts/QM9_chiraltag/datasets/train
        print('dirs:', dirs) #[]
        print('files:', files) # ['part_10.pickle', 'part_3.pickle', 'flist.csv', 'part_6.pickle', 'part_4.pickle', 'part_0.pickle', 'part_2.pickle', 'part_11.pickle', 'part_8.pickle', 'part_5.pickle', 'part_1.pickle', 'part_9.pickle', 'part_7.pickle']
        for fname in files:
            print('start_path:', start_path) #/data/fzg/3DConformersGen/Ch_EcConf/scripts/QM9_chiraltag/datasets/train
            print ('fname2:', fname) #part_10.pickle
            print('name:', name)
            
            if name in fname:
                #full_path = os.path.join(start_path, relpath, fname)
                full_path = os.path.join(relpath, fname)
                print('full_path:', full_path)
                #raise Exception('stop')
                flist.append(os.path.normpath(os.path.abspath(full_path)))
    return flist