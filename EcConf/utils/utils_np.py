import  numpy   as np
from numpy.linalg import norm
import math 
def calc_bond(x1,x2):
    vec1=x1-x2
    #print ('bond',vec1,norm(vec1,axis=1),norm(vec1))
    return norm(vec1,axis=1)

def calc_angle(x1, x2, x3):
    """Calculate angle between 3 atoms"""
    vec1 = x2 - x1
    uvec1 = vec1 / np.reshape(norm(vec1,axis=1),(-1,1))
    #print (vec1,norm(vec1,axis=1),uvec1)
    vec2 = x2 - x3
    uvec2 = vec2 / np.reshape(norm(vec2,axis=1),(-1,1))
    #print ('angle,',np.arccos(np.sum(np.multiply(uvec1, uvec2),axis=1)))
    return np.arccos(np.sum(np.multiply(uvec1, uvec2),axis=1))#*(180.0/math.pi)

def calc_dihedral(x1,x2,x3,x4):
    """
       Calculate dihedral angle between 4 atoms
       For more information, see:
           http://math.stackexchange.com/a/47084
    """
    # Vectors between 4 atoms
    b1 = x2 - x1
    b2 = x2 - x3
    b3 = x4 - x3
    #print ('-'*20)
    # Normal vector of plane containing b1,b2
    n1 = np.cross(b1, b2)
    un1 = n1 / np.reshape(norm(n1,axis=1),(-1,1))
    # Normal vector of plane containing b1,b2
    n2 = np.cross(b2, b3)
    un2 = n2 / np.reshape(norm(n2,axis=1),(-1,1))

    # un1, ub2, and m1 form orthonormal frame
    ub2 = b2 / np.reshape(norm(b2,axis=1),(-1,1))
    um1 = np.cross(un1, ub2)

    # dot(ub2, n2) is always zero
    x = np.sum(np.multiply(un1, un2),axis=1)
    y = np.sum(np.multiply(um1, un2),axis=1)

    dihedral = np.arctan2(y, x)#*(180.0/math.pi)
    dihedral=np.where(dihedral<0,2*math.pi+dihedral,dihedral)
    #print ('dihedral',dihedral)
    return dihedral

def np_adjs_to_ctable(adjs):
    ctable=np.sum(adjs,axis=-1).astype(float)
    #print (ctable)
    ctable[0,0]=1
    if len(ctable)>2:
        #print ('ctable',ctable[0,2])
        if ctable[0,2]==0:
            #print ('ctable',ctable[0,2])
            ctable[0,2]=0.5
            ctable[2,0]=0.5
        #print ('ctable',ctable[0,2])
        
        if ctable[1,2]==0:
            ctable[1,2]=0.5
            ctable[2,1]=0.5
    return ctable
    
def np_adjs_to_zmat(adjs):
    ctable=np_adjs_to_ctable(adjs)
    #print (ctable)
    it,jt=np.where(ctable>0)
    rbij=ctable[it,jt]
    #print (ctable)
    ids=np.where((it>jt)|(it<3))
    #print (ids,it[ids],jt[ids],rbij[ids])
    it=it[ids];jt=jt[ids];rbij=rbij[ids]
    zb=np.stack((it,jt))
    #print (zb)
    atable=ctable[jt]
    indext,kt=np.where(atable>0)
    rbjk=atable[indext,kt]
    it=it[indext];jt=jt[indext];rbij=rbij[indext]
    ids=np.where((it>kt)|(it<3))
    it=it[ids];jt=jt[ids];kt=kt[ids];rbij=rbij[ids];rbjk=rbjk[ids]
    za=np.stack((it,jt,kt))
    #print (za)
    dtable=ctable[kt]
    indext,lt=np.where(dtable>0)
    it=it[indext];jt=jt[indext];kt=kt[indext];rbij=rbij[indext];rbjk=rbjk[indext]
    rbkl=dtable[indext,lt]
    ids=np.where((it>lt)|(it<3))
    it=it[ids];jt=jt[ids];kt=kt[ids];lt=lt[ids];rbij=rbij[ids];rbjk=rbjk[ids];rbkl=rbkl[ids]
    zd=np.stack((it,jt,kt,lt))
    #print (zd)
    ids_all=np.where((it!=jt)&(it!=kt)&(it!=lt)&(jt!=kt)&(kt!=lt)&(jt!=lt)&(it>4)&(rbij+rbjk+rbkl==3.0))[0]
    ids_3=np.where((it==3)&(rbij+rbjk+rbkl==3.0)&((it!=jt)&(it!=kt)&(it!=lt)&(jt!=lt)&(jt!=kt)&(kt!=lt)))[0]
    #print (ids_3)
    if ids_3.shape[0]==0:
        #print ('*')
        ids_3=np.where((it==3)&(rbij+rbjk+rbkl==2.5)&((it!=jt)&(it!=kt)&(it!=lt)&(jt!=lt)&(jt!=kt)&(kt!=lt)))[0]
        #print (ids_3)
    ids_4=np.where((it==4)&(rbij+rbjk+rbkl==3.0)&((it!=jt)&(it!=kt)&(it!=lt)&(jt!=lt)&(jt!=kt)&(kt!=lt)))[0]
    if ids_4.shape[0]==0:
        ids_4=np.where((it==4)&(rbij+rbjk+rbkl==2.5)&((it!=jt)&(it!=kt)&(it!=lt)&(jt!=lt)&(jt!=kt)&(kt!=lt)))[0] 
    ids_2=np.where((it==2)&((it!=jt)&(it!=kt)&(it!=lt)&(jt!=kt))&(rbij+rbjk+rbkl==3))[0]
    if ids_2.shape[0]==0:
        ids_2=np.where((it==2)&((it!=jt)&(it!=kt)&(it!=lt)&(jt!=kt))&(rbij+rbjk+rbkl==2.5))[0]
    ids_1=np.where((it<=1))[0]
    #print (ids_all,ids_4,ids_3,ids_2,ids_1)
    ids=np.sort(np.concatenate((ids_all,ids_4,ids_3,ids_2,ids_1),axis=0))
    it=it[ids];jt=jt[ids];kt=kt[ids];lt=lt[ids];rbij=rbij[ids];rbjk=rbjk[ids];rbkl=rbkl[ids] 
    #print (ids)
    uni,ids=np.unique(it,return_index=True)
    #print (uni,ids)
    it=it[ids];jt=jt[ids];kt=kt[ids];lt=lt[ids];rbij=rbij[ids];rbjk=rbjk[ids];rbkl=rbkl[ids]  
    Zmat=np.stack((it,jt,kt,lt,rbij+rbjk+rbkl),axis=1)
    #print (Zmat)
    return Zmat

