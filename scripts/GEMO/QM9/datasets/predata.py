import pickle
import math



def filt_data():
    #对于Drugs来说，某些分子的原子数量很大，能达到50多个，但是此类分子不多，却非常占用GPU资源，导致模型跑不起来，所以去掉

    '''
    conf num: 199273
    drop conf num: 0
    conf.natoms != len(conf.edge_type:199273)
    '''

    print('trian data')
    new_data = []
    with open('GeoDiff_QM9/train/ecconf_all.pickle', 'rb') as f:
        dt = pickle.load(f)
    count = 0
    for conf in dt:
        #if conf.natoms != len(conf.atom_type):
            #print(f'conf.natoms != len(conf.atom_type)') #这两个不相等,是正常的，由于EcConf的数据集是除了H原子的，所以不一样，因此原子的数量应该按conf.natoms，而len(conf.atom_type)是未去氢原子的情况
            #print(f'{conf.natoms} != {len(conf.atom_type)}')
            #count += 1
        if conf.natoms <= 50:
            new_data.append(conf)

    print('conf num:', len(new_data))
    print(f'drop conf num:', len(dt) - len(new_data)) #0?

    with open('GeoDiff_QM9/train/filt_ecconf_all.pickle', 'wb') as f:
        pickle.dump(new_data, f)
    

    drop_smiles_set = set()
    with open('GeoDiff_QM9/train/drop_conf50.txt', 'w') as f:
        drop_confs  = set(dt) - set(new_data)
        for i in drop_confs:
            drop_smiles_set.add(i.smiles)

        for sm in drop_smiles_set:
            f.write(sm + '\n')




    print('valid data')
    new_data = []
    with open('GeoDiff_QM9/valid/ecconf_all.pickle', 'rb') as f:
        dt = pickle.load(f)
    count = 0
    for conf in dt:
        #if conf.natoms != len(conf.atom_type):
            #print(f'conf.natoms != len(conf.atom_type)') #这两个不相等,是正常的，由于EcConf的数据集是除了H原子的，所以不一样，因此原子的数量应该按conf.natoms，而len(conf.atom_type)是未去氢原子的情况
            #print(f'{conf.natoms} != {len(conf.atom_type)}')
            #count += 1
        if conf.natoms <= 50:
            new_data.append(conf)

    print('conf num:', len(new_data))
    print(f'drop conf num:', len(dt) - len(new_data)) #0?

    with open('GeoDiff_QM9/valid/filt_ecconf_all.pickle', 'wb') as f:
        pickle.dump(new_data, f)
    

    drop_smiles_set = set()
    with open('GeoDiff_QM9/valid/drop_conf50.txt', 'w') as f:
        drop_confs  = set(dt) - set(new_data)
        for i in drop_confs:
            drop_smiles_set.add(i.smiles)

        for sm in drop_smiles_set:
            f.write(sm + '\n')





    print('test data')
    new_data = []
    with open('GeoDiff_QM9/test/ecconf_all.pickle', 'rb') as f:
        dt = pickle.load(f)
    count = 0
    for conf in dt:
        #if conf.natoms != len(conf.atom_type):
            #print(f'conf.natoms != len(conf.atom_type)') #这两个不相等,是正常的，由于EcConf的数据集是除了H原子的，所以不一样，因此原子的数量应该按conf.natoms，而len(conf.atom_type)是未去氢原子的情况
            #print(f'{conf.natoms} != {len(conf.atom_type)}')
            #count += 1
        if conf.natoms <= 50:
            new_data.append(conf)

    print('conf num:', len(new_data))
    print(f'drop conf num:', len(dt) - len(new_data)) #0?

    with open('GeoDiff_QM9/test/ecconf_all.pickle', 'wb') as f:
        pickle.dump(new_data, f)
    

    drop_smiles_set = set()
    with open('GeoDiff_QM9/test/drop_conf50.txt', 'w') as f:
        drop_confs  = set(dt) - set(new_data)
        for i in drop_confs:
            drop_smiles_set.add(i.smiles)

        for sm in drop_smiles_set:
            f.write(sm + '\n')




def split_data():
    #将数据分割成多段
    with open('GeoDiff_QM9/train/filt_ecconf_all.pickle', 'rb') as f:
        dt = pickle.load(f)

    lens = math.ceil(len(dt) / 30000)

    for idx in range(lens):
        dt_sub = dt[idx*30000:(idx + 1)*30000]
        with open(f'GeoDiff_QM9/train/ecconf_part{idx}.pickle', 'wb') as f:
            pickle.dump( dt_sub, f)





    with open('GeoDiff_QM9/valid/filt_ecconf_all.pickle', 'rb') as f:
        dt = pickle.load(f)

    lens = math.ceil(len(dt) / 30000)

    for idx in range(lens):
        dt_sub = dt[idx*30000:(idx + 1)*30000]
        with open(f'GeoDiff_QM9/valid/ecconf_part{idx}.pickle', 'wb') as f:
            pickle.dump( dt_sub, f)




if __name__ == '__main__':
    filt_data()
    split_data()