import cv2
import numpy as np
import sys
import math


def getLineVariance(img,angle):
    rot = cv2.getRotationMatrix2D((img.shape[1]/2,img.shape[0]/2), angle, 1)
    workImg = cv2.warpAffine(img,rot,(img.shape[1],img.shape[0]),None,cv2.INTER_NEAREST+cv2.WARP_FILL_OUTLIERS)
    profile = workImg.sum(axis=1)
    mean = profile.sum()/profile.shape[0]
    return np.power(profile-mean,2).sum()/profile.shape[0]

def deskew(img):
    smallImg = cv2.resize(img, None, None, 0.25, 0.25)
    resAngle = 1
    bestVar = getLineVariance(smallImg,0)
    bestAngle = 0
    bestVarRight=bestVar
    for curAngle in range(1,30,resAngle):
        var = getLineVariance(smallImg,curAngle)
        if var>bestVar:
            bestVar=var
            bestAngle=curAngle
        else:
            break

    bestAngleRight=0
    for curAngle in range(-1,-30,-resAngle):
        var = getLineVariance(smallImg,curAngle)
        if var>bestVarRight:
            bestVarRight=var
            bestAngleRight=curAngle
        else:
            break
    if bestVarRight>bestVar:
        bestAngle=bestAngleRight
    rot = cv2.getRotationMatrix2D((img.shape[1]/2,img.shape[0]/2), bestAngle, 1)
    img = cv2.warpAffine(img,rot,(img.shape[1],img.shape[0]),None,cv2.INTER_NEAREST+cv2.WARP_FILL_OUTLIERS)

    return bestAngle, img

def threshSmallCC(pred,ccRes):
    numCCs = ccRes[0]
    ccs = ccRes[1]
    ccStats = ccRes[2]
    for cc in range(1,numCCs):
        if max(ccStats[cc][cv2.CC_STAT_WIDTH],ccStats[cc][cv2.CC_STAT_HEIGHT]) < 70:
            for y in range(ccStats[cc][cv2.CC_STAT_TOP],ccStats[cc][cv2.CC_STAT_TOP]+ccStats[cc][cv2.CC_STAT_HEIGHT]):
                for x in range(ccStats[cc][cv2.CC_STAT_LEFT],ccStats[cc][cv2.CC_STAT_LEFT]+ccStats[cc][cv2.CC_STAT_WIDTH]):
                    if ccs[y,x]==cc:
                        pred[y,x]=0
                        ccs[y,x]=0
            ccStats[cc][cv2.CC_STAT_WIDTH]=0
            ccStats[cc][cv2.CC_STAT_HEIGHT]=0
            
def mergeCheck(x,y,curCC,ccs,ccStats,pred,ccMerges,superCCs,superMap,todo,centersOfMass, task3):
    mergeCC = max(ccs[max(0,y-3),x],ccs[y,x],ccs[min(pred.shape[0]-1,y+3),x])
    if mergeCC>0:
        if mergeCC in ccMerges and curCC in ccMerges[mergeCC]:
            return False
        #draw connector
        #if x < ccStats[curCC][cv2.CC_STAT_LEFT]:
        #    curStart = (ccStats[curCC][cv2.CC_STAT_LEFT],ccStats[curCC][cv2.CC_STAT_TOP]+ccStats[curCC][cv2.CC_STAT_HEIGHT]/2)
        #    mergeEnd = (x,ccStats[mergeCC][cv2.CC_STAT_TOP]+ccStats[mergeCC][cv2.CC_STAT_HEIGHT]/2)
        #else:
        #    curStart = (ccStats[curCC][cv2.CC_STAT_LEFT]+ccStats[curCC][cv2.CC_STAT_WIDTH],ccStats[curCC][cv2.CC_STAT_TOP]+ccStats[curCC][cv2.CC_STAT_HEIGHT]/2)
        #    mergeEnd = (x,ccStats[mergeCC][cv2.CC_STAT_TOP]+ccStats[mergeCC][cv2.CC_STAT_HEIGHT]/2)

        if x < ccStats[curCC][cv2.CC_STAT_LEFT]:
            curStartX = ccStats[curCC][cv2.CC_STAT_LEFT]+1
            dir=1
        else:
            curStartX = ccStats[curCC][cv2.CC_STAT_LEFT]+ccStats[curCC][cv2.CC_STAT_WIDTH]-2
            dir=-1
        curStartY=centersOfMass[curCC] #ccStats[curCC][cv2.CC_STAT_TOP]+ccStats[curCC][cv2.CC_STAT_HEIGHT]/2
        for out in range(0,ccStats[curCC][cv2.CC_STAT_WIDTH]/2):
            if dir>0 and curStartX+out<pred.shape[1] and ccs[curStartY,curStartX+out]==curCC:
                curStartX=curStartX+out
                break
            elif dir<0 and curStartX-out>=0 and ccs[curStartY,curStartX-out]==curCC:
                curStartX=curStartX-out
                break;
        if task3:
            endX = x
            endY=centersOfMass[mergeCC] #ccStats[curCC][cv2.CC_STAT_TOP]+ccStats[curCC][cv2.CC_STAT_HEIGHT]/2
            for out in range(0,ccStats[curCC][cv2.CC_STAT_WIDTH]/2):
                if dir<0 and endX+out<pred.shape[1] and ccs[endY,endX+out]==mergeCC:
                    endX=endX+out
                    break
                elif dir>0 and endX-out>=0 and ccs[endY,endX-out]==mergeCC:
                    endX=endX-out
                    break;
        else:
            endX=x
            endY=y
        cv2.line(pred, (curStartX, curStartY), (endX,endY), 255, 7)

        #merge
        newRX = max(ccStats[curCC][cv2.CC_STAT_LEFT]+ccStats[curCC][cv2.CC_STAT_WIDTH],ccStats[mergeCC][cv2.CC_STAT_LEFT]+ccStats[mergeCC][cv2.CC_STAT_WIDTH])
        ccStats[curCC][cv2.CC_STAT_LEFT] = min(ccStats[curCC][cv2.CC_STAT_LEFT],ccStats[mergeCC][cv2.CC_STAT_LEFT])
        ccStats[curCC][cv2.CC_STAT_WIDTH] = newRX-ccStats[curCC][cv2.CC_STAT_LEFT]
        newBY = max(ccStats[curCC][cv2.CC_STAT_TOP]+ccStats[curCC][cv2.CC_STAT_HEIGHT],ccStats[mergeCC][cv2.CC_STAT_TOP]+ccStats[mergeCC][cv2.CC_STAT_HEIGHT])
        ccStats[curCC][cv2.CC_STAT_TOP] = min(ccStats[curCC][cv2.CC_STAT_TOP],ccStats[mergeCC][cv2.CC_STAT_TOP])
        ccStats[curCC][cv2.CC_STAT_HEIGHT] = newBY-ccStats[curCC][cv2.CC_STAT_TOP]

        ccStats[mergeCC][cv2.CC_STAT_LEFT]=ccStats[curCC][cv2.CC_STAT_LEFT]
        ccStats[mergeCC][cv2.CC_STAT_WIDTH]=ccStats[curCC][cv2.CC_STAT_WIDTH]
        ccStats[mergeCC][cv2.CC_STAT_TOP]=ccStats[curCC][cv2.CC_STAT_TOP]
        ccStats[mergeCC][cv2.CC_STAT_HEIGHT]=ccStats[curCC][cv2.CC_STAT_HEIGHT]
        if mergeCC in ccMerges:
            for otherCC in ccMerges[mergeCC]:
                ccStats[otherCC][cv2.CC_STAT_LEFT]=ccStats[curCC][cv2.CC_STAT_LEFT]
                ccStats[otherCC][cv2.CC_STAT_WIDTH]=ccStats[curCC][cv2.CC_STAT_WIDTH]
                ccStats[otherCC][cv2.CC_STAT_TOP]=ccStats[curCC][cv2.CC_STAT_TOP]
                ccStats[otherCC][cv2.CC_STAT_HEIGHT]=ccStats[curCC][cv2.CC_STAT_HEIGHT]
                ccMerges[otherCC].append(curCC)
            ccMerges[curCC]=ccMerges[mergeCC][:]
            ccMerges[curCC].append(mergeCC)
            ccMerges[mergeCC].append(curCC)
            superCCs[superMap[mergeCC]].append(curCC)
            superMap[curCC]=superMap[mergeCC]

        else:
            ccMerges[curCC]=[mergeCC]
            ccMerges[mergeCC]=[curCC]
            superMap[mergeCC]=len(superCCs)
            superMap[curCC]=len(superCCs)
            superCCs.append([mergeCC,curCC])
            todo.append(mergeCC)

        return True
    return False

def connectHorz(pred,ccRes,task3):
    ccMerges={}
    superCCs=[]
    superMap={}
    numCCs = ccRes[0]
    ccs = ccRes[1]
    ccStats = ccRes[2]
    centersOfMass=[0]
    for cc in range(1,numCCs):
        if ccStats[cc][cv2.CC_STAT_WIDTH]!=0:
            ySum=0.0
            count=0
            for x in range(ccStats[cc][cv2.CC_STAT_LEFT],ccStats[cc][cv2.CC_STAT_LEFT]+ccStats[cc][cv2.CC_STAT_WIDTH]):
                for y in range(ccStats[cc][cv2.CC_STAT_TOP],ccStats[cc][cv2.CC_STAT_TOP]+ccStats[cc][cv2.CC_STAT_HEIGHT]):
                    if ccs[y,x]==cc:
                        ySum+=y
                        count+=1
            centersOfMass.append(int(ySum/count))
        else:
            centersOfMass.append(1)
    for cc in range(1,numCCs):
        if ccStats[cc][cv2.CC_STAT_WIDTH]!=0 and cc not in ccMerges:
            todo=[cc]
            merged=False
            while len(todo)>0:
                curCC = todo.pop()
                y = centersOfMass[curCC] #ccStats[curCC][cv2.CC_STAT_TOP]+ccStats[curCC][cv2.CC_STAT_HEIGHT]/2
                #left
                for x in range(ccStats[curCC][cv2.CC_STAT_LEFT]-1,max(0,ccStats[curCC][cv2.CC_STAT_LEFT]-1000),-1):
                    if mergeCheck(x,y,curCC,ccs,ccStats,pred,ccMerges,superCCs,superMap,todo,centersOfMass, task3):
                        merged = True
                        break
                #right
                for x in range(ccStats[curCC][cv2.CC_STAT_LEFT]+ccStats[curCC][cv2.CC_STAT_WIDTH],min(pred.shape[1]-1,ccStats[curCC][cv2.CC_STAT_LEFT]+ccStats[curCC][cv2.CC_STAT_WIDTH]+1000)):
                    if mergeCheck(x,y,curCC,ccs,ccStats,pred,ccMerges,superCCs,superMap,todo,centersOfMass, task3):
                        merged = True
                        break
            if not merged:
                superMap[cc]=len(superCCs)
                superCCs.append([cc])
    return superCCs

def findOutlierCCs(pred,ccRes):
    meanH = 0.0
    numCCs = ccRes[0]
    ccStats = ccRes[2]
    ccCount=0
    for cc in range(1,numCCs):
        if ccStats[cc][cv2.CC_STAT_HEIGHT]>0:
            meanH += ccStats[cc][cv2.CC_STAT_HEIGHT]
            ccCount+=1
    meanH/=ccCount
    std=0.0
    for cc in range(1,numCCs):
        if ccStats[cc][cv2.CC_STAT_HEIGHT]>0:
            std += (meanH-ccStats[cc][cv2.CC_STAT_HEIGHT])**2
    std = math.sqrt(std/ccCount)
    
    outliers=[]
    for cc in range(1,numCCs):
        if ccStats[cc][cv2.CC_STAT_HEIGHT]>0 and abs(meanH-ccStats[cc][cv2.CC_STAT_HEIGHT]) > std:
                outliers.append(cc)
    return outliers

#this searchs strictly up or down to tryin to reach the destY by traveling along the cc
def searchFor(destY,ccs,cc,dir,startY,startX,pred):
    visited = np.zeros(ccs.shape)
    toSearch=[(startY+dir,startX-1),(startY+dir,startX+1),(startY+dir,startX)]
    while len(toSearch)>0:
        p = toSearch.pop()
        #pred[p[0],p[1]]=200
        #print (cc,p)
        #print (p[0]>=0 , p[0]<ccs.shape[0] , p[1]>=0 , p[1]<ccs.shape[1] , ccs[p[0],p[1]]==cc , visited[p[0],p[1]]==0)
        if p[0]>=0 and p[0]<ccs.shape[0] and p[1]>=0 and p[1]<ccs.shape[1] and ccs[p[0],p[1]]==cc and visited[p[0],p[1]]==0:
            if p[0]==destY:
                return True
            #pred[p[0],p[1]]=150
            visited[p[0],p[1]]=1
            toSearch.append((p[0]+dir,p[1]-1))
            toSearch.append((p[0]+dir,p[1]+1))
            toSearch.append((p[0]+dir,p[1]))
    return False

def cutCC(cc,pred,ccRes):
    ccs = ccRes[1]
    ccStats = ccRes[2]
    if ccStats[cc][cv2.CC_STAT_TOP]==0 or ccStats[cc][cv2.CC_STAT_WIDTH]==0:
        return
    subPred = pred[ccStats[cc][cv2.CC_STAT_TOP]:ccStats[cc][cv2.CC_STAT_TOP]+ccStats[cc][cv2.CC_STAT_HEIGHT], ccStats[cc][cv2.CC_STAT_LEFT]:ccStats[cc][cv2.CC_STAT_LEFT]+ccStats[cc][cv2.CC_STAT_WIDTH]]
    subCCs = ccs[ccStats[cc][cv2.CC_STAT_TOP]:ccStats[cc][cv2.CC_STAT_TOP]+ccStats[cc][cv2.CC_STAT_HEIGHT], ccStats[cc][cv2.CC_STAT_LEFT]:ccStats[cc][cv2.CC_STAT_LEFT]+ccStats[cc][cv2.CC_STAT_WIDTH]]
    profile = subPred.sum(axis=1)
    mean = profile.sum()/subPred.shape[0]
    regionPeaks=[]
    curMax=0
    curPeak=None
    on=False
    for y in range(subPred.shape[0]):
        if profile[y]>mean:
            on=True
            if profile[y]>curMax:
                curMax=profile[y]
                curPeak=y
        elif on:
            on=False
            regionPeaks.append(curPeak)
            curMax=0
            curPeak=None
    if on:
        regionPeaks.append(curPeak)

    for i in range(1,len(regionPeaks)):
        splitY = (regionPeaks[i-1]+regionPeaks[i])/2
        x=0
        while x<subPred.shape[1]:
            if subCCs[splitY,x]==cc:
                if searchFor(regionPeaks[i-1],subCCs,cc,-1,splitY,x,subPred) and searchFor(regionPeaks[i],subCCs,cc,1,splitY,x,subPred):
                    #split.
                    while subCCs[splitY,x]==cc:
                        subPred[splitY,x]=0
                        subCCs[splitY,x]=0
                        x+=1
                else:
                    while subCCs[splitY,x]==cc:
                        x+=1
            else:
                #subPred[splitY,x]=155
                x+=1
            




#task2
def merge(pred):

    angle, pred = deskew(pred)
    ccRes = cv2.connectedComponentsWithStats(pred, 4, cv2.CV_32S)
    threshSmallCC(pred,ccRes)
    superCCs = connectHorz(pred,ccRes,False)
    rot = cv2.getRotationMatrix2D((pred.shape[1]/2,pred.shape[0]/2), -angle, 1)
    pred = cv2.warpAffine(pred,rot,(pred.shape[1],pred.shape[0]),None,cv2.INTER_NEAREST+cv2.WARP_FILL_OUTLIERS)
    return pred

#task3
def cutAndMerge(pred):

    angle, pred = deskew(pred)
    ccRes = cv2.connectedComponentsWithStats(pred, 4, cv2.CV_32S)
    threshSmallCC(pred,ccRes)
    outliers = findOutlierCCs(pred,ccRes)
    for cc in outliers:
        cutCC(cc,pred,ccRes)
    ccRes = cv2.connectedComponentsWithStats(pred, 4, cv2.CV_32S)
    superCCs = connectHorz(pred,ccRes,True)
    rot = cv2.getRotationMatrix2D((pred.shape[1]/2,pred.shape[0]/2), -angle, 1)
    pred = cv2.warpAffine(pred,rot,(pred.shape[1],pred.shape[0]),None,cv2.INTER_NEAREST+cv2.WARP_FILL_OUTLIERS)
    return pred


if __name__ == "__main__":

    #file = sys.argv[1]

    #predFile = '../../results/cbad_simple_base_weights_round_weighted_1_3/train/verbose/'+file+'/pred.png'
    #origFile = '../../results/cbad_simple_base_weights_round_weighted_1_3/train/verbose/'+file+'/pred_on_original.png'

    if len(sys.argv) < 3:
        print 'Usage: '+sys.argv[0]+' predImage outImage [task2]'
        exit(0)

    predFile = sys.argv[1]
    outName = sys.argv[2]
    task3 = len(sys.argv) == 3


    pred = cv2.imread(predFile,0)
    assert pred is not None
    cv2.threshold(pred, 130, 255, cv2.THRESH_BINARY, pred)

    angle, pred = deskew(pred)
    ccRes = cv2.connectedComponentsWithStats(pred, 4, cv2.CV_32S)
    threshSmallCC(pred,ccRes)
    if task3:
        outliers = findOutlierCCs(pred,ccRes)
        for cc in outliers:
            cutCC(cc,pred,ccRes)
        ccRes = cv2.connectedComponentsWithStats(pred, 4, cv2.CV_32S)
    superCCs = connectHorz(pred,ccRes,task3)
    rot = cv2.getRotationMatrix2D((pred.shape[1]/2,pred.shape[0]/2), -angle, 1)
    pred = cv2.warpAffine(pred,rot,(pred.shape[1],pred.shape[0]),None,cv2.INTER_NEAREST+cv2.WARP_FILL_OUTLIERS)
    cv2.imwrite(outName,pred)

