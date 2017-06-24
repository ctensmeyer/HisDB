import cv2
import numpy as np
import sys
import math
from collections import defaultdict, deque
from skimage.graph import route_through_array


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
    #cv2.imwrite('testDeskew.png',img)

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
            
def mergeCheck(x,y,curCC,ccs,ccStats,pred,ccMerges,superCCs,superMap,superLines,centersOfMass, task3):
    mergeCC = max(ccs[max(0,y-3),x],ccs[y,x],ccs[min(pred.shape[0]-1,y+3),x])
    if mergeCC>0:
        #if (mergeCC in ccMerges and curCC in ccMerges[mergeCC]) or (curCC in ccMerges and mergeCC in ccMerges[curCC]):
        if mergeCC in superMap and curCC in superMap and superMap[mergeCC]==superMap[curCC]:
            return True
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
        for out in range(0,ccStats[curCC][cv2.CC_STAT_WIDTH]):
            if dir>0 and curStartX+out<pred.shape[1] and ccs[curStartY,curStartX+out]==curCC:
                curStartX=curStartX+out
                break
            elif dir<0 and curStartX-out>=0 and ccs[curStartY,curStartX-out]==curCC:
                curStartX=curStartX-out
                break;
        if task3:
            endX = x
            endY=centersOfMass[mergeCC] #ccStats[curCC][cv2.CC_STAT_TOP]+ccStats[curCC][cv2.CC_STAT_HEIGHT]/2
            for out in range(0,ccStats[curCC][cv2.CC_STAT_WIDTH]):
                if dir<0 and endX+out<pred.shape[1] and ccs[endY,endX+out]==mergeCC:
                    endX=endX+out
                    break
                elif dir>0 and endX-out>=0 and ccs[endY,endX-out]==mergeCC:
                    endX=endX-out
                    break;
        else:
            endX=x
            endY=y
        cv2.line(pred, (curStartX, curStartY), (endX,endY), 255, 1)
        #pred[endY,endX]=200
        #pred[curStartY,curStartX]=100

        #merge
        if mergeCC in superMap and curCC in superMap:
           superId =  superMap[curCC]
           oldId = superMap[mergeCC]
           toChange = superCCs[oldId]
           for cc in toChange:
               superMap[cc]=superId
           superCCs[superId] += superCCs[oldId]
           superLines[superId] += superLines[oldId]
           del superCCs[oldId]
           del superLines[oldId]
        elif mergeCC in superMap:
            superId =  superMap[mergeCC]
            superMap[curCC] = superId
            superCCs[superId].append(curCC)
        elif curCC in superMap:
            superId =  superMap[curCC]
            superMap[mergeCC] = superId
            superCCs[superId].append(mergeCC)
        else:
            superId=curCC
            superMap[curCC]=superId
            superMap[mergeCC]=superId
            superCCs[superId]=[curCC,mergeCC]
        superLines[superMap[curCC]].append(((curStartX, curStartY),(endX,endY)))
        #if mergeCC not in ccMerges:
        #    todo.append(mergeCC)
        #if curCC in ccMerges:
        #    if mergeCC in ccMerges:
        #        for otherCC in ccMerges[mergeCC]:
        #            ccMerges[curCC].append(otherCC)
        #    for otherCC in ccMerges[curCC]:
        #        ccMerges[otherCC].append(mergeCC)
        #        if mergeCC in ccMerges:
        #            for otherCC2 in ccMerges[mergeCC]:
        #                ccMerges[otherCC].append(otherCC2)

        #if mergeCC in ccMerges:
        #    if curCC in ccMerges:
        #        for otherCC in ccMerges[curCC]:
        #            ccMerges[mergeCC].append(otherCC)
        #    for otherCC in ccMerges[mergeCC]:
        #        ccMerges[otherCC].append(curCC)
        #        if curCC in ccMerges:
        #            for otherCC2 in ccMerges[curCC]:
        #                ccMerges[otherCC].append(otherCC2)
        #   
        #ccMerges[curCC].append(mergeCC)
        #ccMerges[mergeCC].append(curCC)


        return True
    return False

def connectHorz(pred,ccRes,task3):
    distThresh=500
    ccMerges=defaultdict(list)
    superCCs=defaultdict(list)
    superMap={}
    superLines = defaultdict(list)
    ccLines = defaultdict(list)
    numCCs = ccRes[0]
    ccs = ccRes[1]
    ccStats = ccRes[2]
    centersOfMass=[0]
    ccDone=[]
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
        if ccStats[cc][cv2.CC_STAT_WIDTH]!=0:
            merged=False
            #while len(todo)>0:
            #    curCC = todo.pop()
            curCC=cc
            y = centersOfMass[curCC] #ccStats[curCC][cv2.CC_STAT_TOP]+ccStats[curCC][cv2.CC_STAT_HEIGHT]/2
            #left
            for x in range(ccStats[curCC][cv2.CC_STAT_LEFT]-1,max(0,ccStats[curCC][cv2.CC_STAT_LEFT]-distThresh),-1):
                m = mergeCheck(x,y,curCC,ccs,ccStats,pred,ccMerges,superCCs,superMap,superLines,centersOfMass, task3)
                if m:
                    merged = True
                    break
            #right
            for x in range(ccStats[curCC][cv2.CC_STAT_LEFT]+ccStats[curCC][cv2.CC_STAT_WIDTH],min(pred.shape[1]-1,ccStats[curCC][cv2.CC_STAT_LEFT]+ccStats[curCC][cv2.CC_STAT_WIDTH]+distThresh)):
                m= mergeCheck(x,y,curCC,ccs,ccStats,pred,ccMerges,superCCs,superMap,superLines,centersOfMass, task3)
                if m:
                    merged = True
                    break
            #if not merged:
            #    superMap[cc]=len(superCCs)
            #    superCCs.append([cc])
    #cv2.imwrite('testMerge.png',pred)
    for cc in range(1,numCCs):
        if ccStats[cc][cv2.CC_STAT_WIDTH]!=0:
            if cc not in superMap:
                superCCs[cc]=[cc]

    return superCCs, superLines

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

#this searchs  tryin to reach the destY by traveling along the cc
def searchFor(destY,killY,ccs,cc,dir,startY,startX,pred):
    visited = np.zeros(ccs.shape)
    toSearch=[(startY,startX-1),(startY,startX+1),(startY+dir,startX-1),(startY+dir,startX+1),(startY+dir,startX)]
    reverseSearch=[]#[(startY-dir,startX)]
    while len(toSearch)>0:
        p = toSearch.pop()
        if p[0]==killY:
            return False
        #pred[p[0],p[1]]=200
        #print (cc,p)
        #print (p[0]>=0 , p[0]<ccs.shape[0] , p[1]>=0 , p[1]<ccs.shape[1] , ccs[p[0],p[1]]==cc , visited[p[0],p[1]]==0)
        if p[0]>=0 and p[0]<ccs.shape[0] and p[1]>=0 and p[1]<ccs.shape[1] and ccs[p[0],p[1]]==cc and visited[p[0],p[1]]==0:
            if p[0]==destY:
                return True
            #pred[p[0],p[1]]=150
            visited[p[0],p[1]]=1
            toSearch.append((p[0],p[1]-1))
            toSearch.append((p[0],p[1]+1))
            toSearch.append((p[0]+dir,p[1]-1))
            toSearch.append((p[0]+dir,p[1]+1))
            toSearch.append((p[0]+dir,p[1]))
            if p[0]-dir!=startY and p[0]!=startY:
                reverseSearch.append((p[0]-dir,p[1]))
        if len(toSearch)==0 and len(reverseSearch)>0:
            toSearch.append(reverseSearch.pop())
    return False

#https://gist.github.com/mdsrosa/c71339cb23bc51e711d8
class Graph(object):
    def __init__(self):
        self.nodes = set()
        self.edges = defaultdict(list)
        self.distances = {}

    def add_node(self, value):
        self.nodes.add(value)

    def add_edge(self, from_node, to_node, distance):
        self.edges[from_node].append(to_node)
        self.edges[to_node].append(from_node)
        self.distances[(from_node, to_node)] = distance


def dijkstra(graph, initial, destination):
    visited = {initial: 0}
    path = {}

    nodes = set(graph.nodes)

    done=False
    while len(nodes)>0 and not done:
        min_node = None
        for node in nodes:
            if node in visited:
                if min_node is None:
                    min_node = node
                elif visited[node] < visited[min_node]:
                    min_node = node
        if min_node is None:
            break

        nodes.remove(min_node)
        current_weight = visited[min_node]

        for edge in graph.edges[min_node]:
            try:
                weight = current_weight + graph.distances[(min_node, edge)]
            except:
                continue
            if edge not in visited or weight < visited[edge]:
                visited[edge] = weight
                path[edge] = min_node
                if edge==destination:
                    done=True
                    break

    return visited, path


def shortest_path(graph, origin, destination):
    visited, paths = dijkstra(graph, origin, destination)
    full_path = deque()
    _destination = paths[destination]

    while _destination != origin:
        full_path.appendleft(_destination)
        _destination = paths[_destination]

    full_path.appendleft(origin)
    full_path.append(destination)

    return visited[destination], list(full_path)

def splitTwo(pred,subCCs,cc):
    if pred.shape[0]>1 and pred.shape[1]>1:
        midY=subCCs.shape[0]/2

        g = np.zeros(pred.shape)
        for x in range(0,subCCs.shape[1]):
            for y in range(0,subCCs.shape[0]):
                if subCCs[y,x]==cc:
                    g[y,x]=100
                else:
                    g[y,x]=0.01+abs(y-midY)*0.001
        indices, weight = route_through_array(g, (midY,0), (midY,subCCs.shape[1]-1),fully_connected=False)
        #print '--------------------'
        #print indices
        #print '--------------------'
        for p in indices:
            if subCCs[p]==cc:
                subCCs[p]=0
                pred[p]=0

    #g = Graph()
    #for x in range(0,subCCs.shape[1]):
    #    for y in range(0,subCCs.shape[0]):
    #        g.add_node((y,x))
    #        for nbr in [(y-1,x),(y,x+1),(y+1,x),(y,x-1)]:
    #            if nbr[0]>=0 and nbr[0]<subCCs.shape[0] and nbr[1]>=0 and nbr[1]<subCCs.shape[1]:
    #                if subCCs[nbr]==cc:
    #                    g.add_edge((y,x),nbr,100.0)
    #                else:
    #                    cost=0.01
    #                    if abs(midY-y)>abs(midY-nbr[0]):
    #                        cost-=0.0014
    #                    elif abs(midY-y)<abs(midY-nbr[0]):
    #                        cost+=0.0014
    #                    g.add_edge((y,x),nbr,cost)
    #v, path = shortest_path(g,(midY,0),(midY,subCCs.shape[1]-1))
    #for p in path:
    #    if subCCs[p]==cc:
    #        subCCs[p]=0
    #        pred[p]=0




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
                if searchFor(regionPeaks[i-1],regionPeaks[i],subCCs,cc,-1,splitY,x,subPred) and searchFor(regionPeaks[i],regionPeaks[i-1],subCCs,cc,1,splitY,x,subPred):
                    #split.
                    startX=x
                    endX=x
                    while x<subPred.shape[1] and subCCs[splitY,x]==cc:
                        #subPred[splitY,x]=0
                        #subCCs[splitY,x]=0
                        x+=1
                        endX+=1
                    midX=(startX+endX)/2
                    dist = endX-startX
                    startX= max(0,midX-dist)
                    #startX= max(0,midX-(regionPeaks[i]-regionPeaks[i-1]))
                    while startX>=0 and subCCs[splitY,startX]==cc:
                        startX-=1
                    endX= min(subPred.shape[1]-1,midX+dist)
                    #endX= max(subPred.shape[1]-1,midX+(regionPeaks[i]-regionPeaks[i-1]))
                    while endX<subPred.shape[1] and subCCs[splitY,endX]==cc:
                        endX+=1
                    splitTwo(subPred[regionPeaks[i-1]+1:regionPeaks[i],startX:endX],subCCs[regionPeaks[i-1]+1:regionPeaks[i],startX:endX],cc)
                else:
                    while x<subPred.shape[1] and subCCs[splitY,x]==cc:
                        x+=1
            else:
                #subPred[splitY,x]=155
                x+=1
            

    #showHist = np.zeros((subPred.shape[0],subPred.shape[0]))
    #maxV= profile.max()
    #for y in range(0,subPred.shape[0]):
    #    v = int(subPred.shape[0] * profile[y]/maxV)
    #    for x in range(0,v):
    #        showHist[y,x]=255
    #showHist[:,int(subPred.shape[0] * mean/maxV)] = 155
    #for peak in regionPeaks:
    #    showHist[peak,:]=100
    #    #[peak,:]=100

    #cv2.imwrite('test'+str(cc)+'.png',subPred)
    #cv2.imwrite('testHist'+str(cc)+'.png',showHist)


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

def getSuperCC(pred,superCC,lines,ccRes,i=None):
    ccs = ccRes[1]
    ccStats = ccRes[2]
    minX=999999
    minY=999999
    maxX=0 #+1
    maxY=0 #+1
    for cc in superCC:
        if ccStats[cc][cv2.CC_STAT_LEFT] < minX:
            minX = ccStats[cc][cv2.CC_STAT_LEFT]
        if ccStats[cc][cv2.CC_STAT_TOP] < minY:
            minY = ccStats[cc][cv2.CC_STAT_TOP]
        if ccStats[cc][cv2.CC_STAT_LEFT]+ccStats[cc][cv2.CC_STAT_WIDTH] > maxX:
            maxX = ccStats[cc][cv2.CC_STAT_LEFT]+ccStats[cc][cv2.CC_STAT_WIDTH]
        if ccStats[cc][cv2.CC_STAT_TOP]+ccStats[cc][cv2.CC_STAT_HEIGHT] > maxY:
            maxY = ccStats[cc][cv2.CC_STAT_TOP]+ccStats[cc][cv2.CC_STAT_HEIGHT]
    minX-=3
    minY-=3
    maxX+=3
    maxY+=3
    ccIm = np.zeros((maxY-minY,maxX-minX),dtype=np.uint8)
    for x in range(max(0,minX),min(pred.shape[1]-1,maxX)):
        for y in range(max(0,minY),min(pred.shape[0]-1,maxY)):
           if ccs[y,x] in superCC:
               ccIm[y-minY,x-minX]=155
    for line in lines:
        cv2.line(ccIm, (line[0][0]-minX,line[0][1]-minY), (line[1][0]-minX,line[1][1]-minY),255, 14)

    if i is not None:
        show = ccIm.copy()

    im,contours, hie = cv2.findContours(ccIm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE, None, None, (minX,minY))
    points = np.squeeze(contours[0],axis=1)
    if i is not None:
        for p in points:
            show[p[1]-minY,p[0]-minX]=255

        cv2.imwrite('superCC_'+str(i)+'.png',show)
    return points

def getContours(predRaw, orig=None):
    global_threshold = 127
    ret, pred = cv2.threshold(predRaw,global_threshold,255,cv2.THRESH_BINARY)
    angle, pred = deskew(pred)
    ccRes = cv2.connectedComponentsWithStats(pred, 4, cv2.CV_32S)
    threshSmallCC(pred,ccRes)
    outliers = findOutlierCCs(pred,ccRes)
    for cc in outliers:
        cutCC(cc,pred,ccRes)
    ccRes = cv2.connectedComponentsWithStats(pred, 4, cv2.CV_32S)

    superCCs, superLines = connectHorz(pred,ccRes,True)

    rot = cv2.getRotationMatrix2D((pred.shape[1]/2,pred.shape[0]/2), -angle, 1)

    contours=[]
    for i in superCCs:
        points = getSuperCC(pred,superCCs[i],superLines[i],ccRes)
        for ii in range(0,len(points)):
            newP = np.matmul(rot,np.array([[points[ii][0]],[points[ii][1]],[1]]))
            points[ii]=(newP[0][0],newP[1][0])

        contours.append(points)
    return contours

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
    print 'start thresh small'
    threshSmallCC(pred,ccRes)
    if task3:
        print 'start find outliers'
        outliers = findOutlierCCs(pred,ccRes)
        print 'start cutting'
        for cc in outliers:
            cutCC(cc,pred,ccRes)
        print 'finish cutting'
        ccRes = cv2.connectedComponentsWithStats(pred, 4, cv2.CV_32S)
    print 'start connecting'
    superCCs, superLines = connectHorz(pred,ccRes,task3)
    print 'finish connecting'
    rot = cv2.getRotationMatrix2D((pred.shape[1]/2,pred.shape[0]/2), -angle, 1)
    contours=[]
    print 'start contouts'
    for i in superCCs:
        points = getSuperCC(pred,superCCs[i],superLines[i],ccRes)
        for ii in range(0,len(points)):
            newP = np.matmul(rot,np.array([[points[ii][0]],[points[ii][1]],[1]]))
            points[ii]=(newP[0][0],newP[1][0])

        contours.append(points)

    print 'finish contours'
    pred = cv2.warpAffine(pred,rot,(pred.shape[1],pred.shape[0]),None,cv2.INTER_NEAREST+cv2.WARP_FILL_OUTLIERS)
    for points in contours:
        for p in points:
            pred[p[1],p[0]]=155
    cv2.imwrite(outName,pred)

