#!/usr/bin/env python3

import numpy as np 
import time

#Implementation for computing the
#TOC= Total Operating Charcateristic Curve
#Author:S. Ivvan Valdez 
#Centro de Investigación en Ciencias de Información Geoespacial AC
#Querétaro, México.



#This function computes the coordinates in:
#   The x axis= Hits + False Alarms, this can be accessed as toc['TP+FP']
#   The y axis= Hits, this can be accessed as toc['TP']
#   The threshold for each coordinate
#   The area ratio, that is to say: Area under the curve/ Parellelepiped area


# from memory_profiler import profile
# @profile(precision=12)
def compute(rank, groundtruth , reverse = False):
    
    import numpy as np 
    T = dict()
    
    #Sorting the classification rank and getting the indices
    indices = sorted(range(len(rank)),key = lambda index: rank[index], reverse = reverse)    
    
    #Data size, this is the total number of samples
    T['ndata'] = n = len(rank)
    T['type'] = 'TOC'
    
    #This is the number of class 1 in the input data
    T['npos'] = P = sum(groundtruth==1)
    T['TP+FP'] = np.append(np.array(range(n)),n)
    T['TP'] = np.append(0,np.cumsum(groundtruth[indices]))
    T['thresholds'] = np.append(rank[indices[0]]+1e-6,rank[indices])
    T['areaRatio'] = (sum(T['TP'])-0.5*T['TP'][-1]-(P*P/2))/((n-P)*P)
    return T    


#This method generates a curve with a similar shape than T1 but with number of sample of T2.
#The normalized area of the resulting curve is approximates that of T1
def resample(T1, T2):
    n1=T1['ndata']
    n2=T2['ndata']
    pn1=T1['npos']/n1
    pn2=T2['npos']/n2

    if (n2<n1):
        print("The second curve T2 most have more elements than the first.")

    T3=T2.copy()
    T3['npos']=pn1*n2
    n3=n2
    T3['TP']=np.zeros(np.size(T2['TP']))
    dfp=(T3['npos'])/(T1['npos'])
    df=(T3['npos']*n1)/(T1['npos']*n3)
    j=1
    for i in range(n1):
        while(j<=n2  and  (T3['TP+FP'][j]/n2)<=(T1['TP+FP'][i+1]/n1)):
            h=T1['TP'][i]*dfp
            T3['TP'][j]=T1['TP'][i]*dfp+(float(j)-(i*n3/n1))*df*float(T1['TP'][i+1]-T1['TP'][i])
            j+=1

    T3['npos']=np.round(T3['npos'])
    T3['TP']=np.round(T3['TP'])
    n3=T3['ndata']
    T3['areaRatio']=(sum(T3['TP'])-0.5*T3['TP'][-1]-(T3['npos']*T3['npos']/2))/((n3-T3['npos'])*T3['npos'])
    return(T3)

#This method computes the difference between two TOC curves, the 
def TOCdiff(T1,T2):
    n1=T1['ndata']
    n2=T2['ndata']
    if (n1<n2):
        Tw1=resample(T1,T2)
        Tw2=T2
    if (n2<n1):
        Tw2=resample(T2,T1)
        Tw1=T1
    n1=Tw1['ndata']
    n2=Tw2['ndata']
    
    pn1=Tw1['npos']/n1    
    pn2=Tw2['npos']/n2
    Tdiff=Tw1.copy()
    Tarea=((1-pn1)+(1.0-pn2))/2.0
    Tdiff['TP']=Tw1['TP']/Tw1['npos']-Tw2['TP']/Tw2['npos']
    Tdiff['npos']=max(abs(Tdiff['TP']))
    Tdiff['areaRatio']=sum(abs(Tdiff['TP']))/(Tarea*n1)
    Tdiff['type']='diff'
    return(Tdiff)


def plotComp(T1, T2, match_quantity = True, TOCname1='2000-2018',TOCname2='2018-2021',title="default",filename='',height=1800,width=1800,dpi=300):
    print("new toc comp")
    import numpy as np 
    import matplotlib.pyplot as plt
    
    
    if (T1['type'] != 'TOC' or T2['type'] != 'TOC'):
        return;
    n1 = T1['ndata']
    n2 = T2['ndata']
    
    pn1 = T1['npos']/n1
    pn2 = T2['npos']/n2
    

    ngainT1 = T1["npos"]
    ngainT2 = T2["npos"]
    plt.scatter(T1['TP+FP'][ngainT1]/n1, T1['TP'][ngainT1]/T1['npos'])
    plt.scatter(T2['TP+FP'][ngainT2]/n2, T2['TP'][ngainT2]/T2['npos'])
    
    
    rx1 = np.array([0,pn1,1,1-pn1])
    rx2 = np.array([0,pn2,1,1-pn2])
    ry = np.array([0,1,1,0])
    plt.clf()
    fig = plt.figure(figsize=(3, 3), dpi=dpi)
    plt.ylim(0, 1.01)
    plt.xlim(0, 1.01)
    plt.xlabel("Hits+False Alarms")
    plt.ylabel("Hits")
    if title == 'default':
        plt.title("Comparison" + " " + TOCname1 + ' vs ' + TOCname2)
    else:
        plt.title(title)
    #plt.text(0.5,0.025,'AUC-'+TOCname2+'=')
    #plt.text(0+0.275,0.02,"AUC="+str(round(T2['areaRatio'],2)))
    #plt.text(0.5,0.075,'AUC-'+TOCname1+'=')
    #plt.text(0+0.275,0.080,"AUC="+str(round(T1['areaRatio'],2)))
    #str(T1['areaRatio'])
    plt.plot(rx1, ry, 'r--')
    plt.plot(rx2, ry, 'b--')
    plt.plot(np.array([0,1]),np.array([0,1]),'k-.')
    plt.plot(T1['TP+FP']/n1,T1['TP']/T1['npos'],'r-',label=TOCname1+" AUC={:0.2f}".format(((T1['areaRatio']))),linewidth=0.8)
    
        
    plt.plot(T2['TP+FP']/n2, T2['TP']/T2['npos'],'b-', label=TOCname2+" AUC={:0.2f}".format(((T2['areaRatio']))))
    plt.legend()
    if (filename!=''):
        plt.savefig(filename+".png", format = "png", dpi = 150)
        plt.close(fig)
    else:
        plt.show()
    
    
    
    
    

#This function plots the TOC to the terminal or a file
def plot(T, filename = '', match_quantity = True, title = 'default', TOCname = 'TOC', normalized = False, height = 1800, width = 1800, dpi = 300):
    
    import matplotlib.pyplot as plt
    import numpy as np
    
    plt.clf()
    ngain = T['npos']
    fig = plt.figure(figsize=(3, 3), dpi = dpi)
    
    if (filename!=''):
        #fig=plt.figure(figsize=(height/dpi, width/dpi), dpi=dpi)
        fig=plt.figure(figsize=(15, 15), dpi=dpi)
        
    plt.xlabel("Hits+False Alarms")
    plt.ylabel("Hits")    
    if (T['type']=='TOC'):
        if (title=='default'):
                title="Total operating characteristic"
        plt.title(title)
        if(normalized):        
            rx = np.array([0, T['npos']/T['ndata'], 1, 1-T['npos']/T['ndata']])
            
            ry = np.array([0, 1, 1, 0])   
            
            plt.ylim(0, 1.01)
            plt.xlim(0, 1.01)            
            plt.text(0.575,0.025,'AUC=')
            plt.text(0.675,0.025,str(round(T['areaRatio'],4)))
            plt.plot(rx, ry,'b--')
            plt.plot(T['TP+FP']/T['ndata'],T['TP']/T['npos'],'r-',label=TOCname,linewidth=3)
        else:
            rx = np.array([0, T['npos'], T['ndata'], T['ndata']-T['npos']])
            ry = np.array([0, T['npos'], T['npos'], 0])
            
            plt.ylim(0, 1.01*T['npos'])
            plt.xlim(0, 1.01*T['ndata'])
            plt.text(0.575*T['ndata'],0.025*T['npos'],'AUC=')
            plt.text(0.675*T['ndata'],0.025*T['npos'],str(round(T['areaRatio'],4)))
            plt.plot(rx, ry,'b--')
            #plt.plot(rx[0,1], ry[0,1],'k--')
            plt.plot(T['TP+FP'],T['TP'],'r-',label="TOC")
            if match_quantity:
                plt.scatter(T['TP+FP'][ngain], T['TP'][ngain])
           
        plt.legend()
        
    if(T['type']=='diff'):
        if (title=='default'):
            title="Difference between 2 TOCs"
        plt.title(title)
        plt.ylim(-1.01, 1.01)
        plt.xlim(-0.01, 1.01)            
        plt.text(0.575,0.025,'AUC=')
        plt.text(0.675,0.025,str(round(T['areaRatio'],4)))
        plt.plot(T['TP+FP']/T['ndata'],T['TP']/T['npos'],'r-',label=TOCname,linewidth=3)
        plt.legend(loc='lower right')       
        
    if (filename!=''):
        plt.savefig(filename)
        plt.close(fig)
    else:
        plt.show()
        
        


def spline(T):
    n=T['ndata']+1
    ns=(n-1)

    a=np.ones((ns))
    b=np.ones((ns))
    c=np.ones((ns))

    c[0]=0.0
    error=100.0
    berror=1000.0
    derror=1000.0
    i=np.array(range(ns))
    x=(T['TP+FP']/ns).astype(float)
    itmax=n+1000
    it=0
    while(error>1e-5 and it<itmax):
        ts=T['TP'][i].astype(float)-b[i]*x[i]-c[i]*(x[i])**2
        error=sum((a[i]-ts)**2)
        a[i]=ts
        ts=(T['TP'][i+1].astype(float)-a[i]-c[i]*x[i+1]**2)/x[i+1]
        error=sum((b[i]-ts)**2)+error
        b[i]=ts
        ts=((b[i[1:]]-b[i[-1]])+2.0*c[i[1:]]*x[i[1:]])/(2.0*x[i[1:]])
        error=sum((c[i[1:]]-ts)**2)+error
        c[i[1:]]=ts
        it=it+1
    
    S['ncoeff']=ns
    S['a0']=a
    S['a1']=b
    S['a2']=c
    return S







#S are the splines given by their coefficients a0,a1,a2, the segments are a0+a1*x+a2*x**2

def interpolateSpline(S,nin=10):

    import matplotlib.pyplot as plt
    xs=np.zeros((10*ns))
    ys=np.zeros((10*ns))
    for i in range(ns):
        for j in range(10):
            xs[i*10+j]=0.1*j*(x[i+1]-x[i])+x[i]
            xx=xs[i*10+j]
            ys[i*10+j]=a[i]+b[i]*xx+c[i]*xx**2

    plt.plot(ns*xs,ys,'r-',label="TOC", linewidth=4)
    plt.plot(T['TP+FP'],T['TP'],marker='o')
    plt.show()



