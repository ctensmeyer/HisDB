import numpy as np
import cv2
import math
import sys
import scipy.ndimage as nd


def get_split_col(im,pred):
        #im = cv2.resize(im, (0,0), fx=0.5, fy=0.5)
        #im = nd.filters.gaussian_filter(im, 2)
        ##cv2.imwrite('blurred.png', im)
        #im = nd.filters.minimum_filter(im, size=(11,1))
        ##cv2.imwrite('eroded.png', im)
        #im = cv2.resize(im, (0,0), fx=2, fy=2)

        proj = np.mean(im, axis=0, dtype=np.float32)
        proj = np.squeeze(cv2.bilateralFilter(proj[np.newaxis,:], 9, 12, 12))

        l = proj.shape[0]

        trunc_proj = proj[l / 4:-l / 4]
        l2 = trunc_proj.shape[0]

        max_idx = np.argmax(trunc_proj)
        min_idx = np.argmin(trunc_proj)
        _min = np.min(trunc_proj)
        _max = np.max(trunc_proj)

        max_dist = abs(max_idx - (l2 / 2)) / float(l2)
        min_dist = abs(min_idx - (l2 / 2)) / float(l2)
        #print max_dist, min_dist

        #plt.plot(trunc_proj)

        copy = np.copy(trunc_proj)
        copy[max(0, min_idx-50): min(trunc_proj.shape[0] - 1, min_idx + 50)] = _max
        next_min = np.min(copy)

        inner_max = np.max(trunc_proj[int(.4 * l2): int(.6 * l2)])

        ret = None
        if max_dist < 0.1:
                #plt.axvline(x=max_idx, linewidth=3, color='red')
                ret = max_idx + l / 4
        elif min_dist < 0.1 and (next_min - _min) > 10:
                #plt.axvline(x=min_idx, linewidth=3, color='green')
                ret = min_idx + l / 4
        elif max_dist < 0.2 and (_max - inner_max) < 10:
                #plt.axvline(x=max_idx, linewidth=3, color='orange')
                ret = max_idx + l / 4

        #double check we aren't splicing a lot of predictions
        if ret is not None:
            #colSum = np.sum(pred[:,ret])/float(7*255)
            colSum=0
            for y in range(1,pred.shape[0]):
                if pred[y-1,ret]==0 and pred[y,ret]>0:
                    colSum+=1
            if colSum>6:
                ret=None

        return ret

def get_vert_lines(im):

        edges = cv2.Sobel(im, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=5)
        edges[edges < 0] = 0
        edges = (255 * (edges / np.max(edges))).astype(np.uint8)
        edges[edges < 50] = 0
        edges[edges != 0] = 255
        #cv2.imwrite(sys.argv[4], edges)

        structure = np.ones((11,1))
        edges = nd.binary_closing(edges, structure=structure)
        #cv2.imwrite(sys.argv[5], (255 * edges).astype(np.uint8))
        structure = np.ones((41,1))
        edges = nd.binary_opening(edges, structure=structure)

        edges = (255 * edges).astype(np.uint8)
        #cv2.imwrite(sys.argv[2], edges)

        proj = np.mean(edges, axis=0, dtype=np.float32)
        proj = np.squeeze(cv2.bilateralFilter(proj[np.newaxis,:], 9, 20, 20))
        #copy = np.copy(proj)

        vert_lines = list()

        while True:
                idx = np.argmax(proj)
                if proj[idx] < 15:
                        break
                #if idx > proj.shape[0] / 20 and idx < 0.95 * proj.shape[0]:
                vert_lines.append((idx, proj[idx]))
                proj[max(0, idx - 50):min(idx + 50, proj.shape[0])] = 0

        out = list()
        for idx, val in vert_lines:
                if len(vert_lines) > 3 or val > 25:
                        out.append(idx)
                        #plt.axvline(idx, linewidth=3, color='red')
        #plt.plot(copy)

        #plt.savefig(sys.argv[3])
        return out

def cropBlack(img,gt):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    median = np.median(gray)
    thresh = median*0.8

    cutTop=0
    while np.median(gray[cutTop,:]) < thresh:
        cutTop+=1

    cutBot=-1
    while np.median(gray[cutBot,:]) < thresh:
        cutBot-=1

    cutLeft=0
    while np.median(gray[:,cutLeft]) < thresh:
        cutLeft+=1

    cutRight=-1
    while np.median(gray[:,cutRight]) < thresh:
        cutRight-=1

    #return img[cutTop:cutBot,cutLeft:cutRight], gt[cutTop:cutBot,cutLeft:cutRight], cutTop, -1*(cutBot+1), cutLeft, -1*(cutRight+1)
    return cutTop, -1*(cutBot+1), cutLeft, -1*(cutRight+1)

def getPageFold(img):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    kernel = np.array([2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,1.5,1,0.5,0,-25,-48,-25,0,0.5,1,1.5,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2]) / 800.0
    #kernel = np.array([4.0,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,3,2,1,0,-50,-96,-50,0,1,2,3,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4]) / 1600.0
    #kernel = np.array([2,2,2,2,2,2,2,1.5,1,0.5,0,-9,-16,-9,0,0.5,1,1.5,2,2,2,2,2,2,2])
    #kernel = np.array([5,5,5,5,5,5,4,3,2,1,0,-18,-34,-18,0,1,2,3,4,5,5,5,5,5,5])
    kernel = np.repeat(kernel,25,axis=0)
    edges = cv2.filter2D(gray,-1,kernel)
    #edges = np.absolute(edges)
    #off = (gray.shape[0]-edges.shape[0])/2
    #print off

    maxV = np.amax(edges)
    minV = np.amin(edges)
    print maxV, minV
    edges[:,:] = (edges[:,:]-minV)*(255.0/(maxV-minV))

    cv2.imwrite('testEdges.png',edges)
    """
    thresh=4000
    while True:
        rho_thetas = cv2.HoughLines(edges, 1,np.pi/180,thresh)
        if (len(rho_thetas[0])==1):
            break
        else if len(rho_thetas[0])==0:
            thresh*=1.5
        else:
            thresh/=2

    a = np.cos(rho_thetas[0][1])
    b = np.sin(rho_thetas[0][1])
    x0 = a*rho_thetas[0][0]
    y0 = b*rho_thetas[0][0]
    #x1 = int(x0 + 4000*(-b))
    #y1 = int(y0 + 4000*(a))
    #x2 = int(x0 - 4000*(-b))
    #y2 = int(y0 - 4000*(a))

    return x0
    """
    return edges

def removeCC(ccId, ccs, stats, removeFrom):
    #check=0
    for y in range(stats[ccId,cv2.CC_STAT_TOP],stats[ccId,cv2.CC_STAT_HEIGHT]+stats[ccId,cv2.CC_STAT_TOP]):
        for x in range(stats[ccId,cv2.CC_STAT_LEFT],stats[ccId,cv2.CC_STAT_WIDTH]+stats[ccId,cv2.CC_STAT_LEFT]):
            if ccs[y,x]==ccId:
                removeFrom[y,x]=0
    #            check+=1
    #assert check == stats[ccId,cv2.CC_STAT_AREA]
    #print 'removed ['+str(ccId)+'] of area '+str(check)


def getLen(line):
    return math.sqrt( (line['x1']-line['x2'])**2 + (line['y1']-line['y2'])**2 )

def convertToLineSegments(pred, ccRes):
    ret=[]
    numLabels, labels, stats, cent = ccRes #cv2.connectedComponentsWithStats(pred, 4, cv2.CV_32S)
    for l in range(1,numLabels):
        if stats[l,cv2.CC_STAT_WIDTH]>30:
            #xs=[]
            #ys=[]
            #for x in range(stats[l,cv2.CC_STAT_LEFT],stats[l,cv2.CC_STAT_WIDTH]+stats[l,cv2.CC_STAT_LEFT]):
            #    for y in range(stats[l,cv2.CC_STAT_TOP],stats[l,cv2.CC_STAT_HEIGHT]+stats[l,cv2.CC_STAT_TOP]):
            #        if labels[y,x]==l:
            #            xs.append(x)
            #            ys.append(y)
            #slope, intercept, r_value, p_value, std_err = stats.linregress(xs,ys)
            topLeft=-1
            topRight=-1
            for y in range(stats[l,cv2.CC_STAT_TOP],stats[l,cv2.CC_STAT_HEIGHT]+stats[l,cv2.CC_STAT_TOP]):
                if topLeft == -1 and labels[y,stats[l,cv2.CC_STAT_LEFT]]==l:
                    topLeft=y
                    if topRight != -1:
                        break
                if topRight == -1 and labels[y,stats[l,cv2.CC_STAT_LEFT]+stats[l,cv2.CC_STAT_WIDTH]-1]==l:
                    topRight=y
                    if topLeft != -1:
                        break

            botLeft=-1
            botRight=-1
            for y in range(stats[l,cv2.CC_STAT_HEIGHT]+stats[l,cv2.CC_STAT_TOP]-1,stats[l,cv2.CC_STAT_TOP],-1):
                if botLeft == -1 and labels[y,stats[l,cv2.CC_STAT_LEFT]]==l:
                    botLeft=y
                    if botRight != -1:
                        break
                if botRight == -1 and labels[y,stats[l,cv2.CC_STAT_LEFT]+stats[l,cv2.CC_STAT_WIDTH]-1]==l:
                    botRight=y
                    if botLeft != -1:
                        break
            ret.append({'x1':  stats[l,cv2.CC_STAT_LEFT],
                        'y1':  (topLeft+botLeft)/2,
                        'x2':  stats[l,cv2.CC_STAT_WIDTH]+stats[l,cv2.CC_STAT_LEFT]-1,
                        'y2':  (topRight+botRight)/2,
                        'cc':  l
                        })
        else:
            removeCC(l,labels,stats,pred)

    return ret, labels, stats



#assumes binary prediction
def removeSidePredictions(pred,orig,ccRes):
    cropTop, cropBot, cropLeft, cropRight = cropBlack(orig,pred)

    #clear pred on black areas
    if cropTop>0:
        pred[:cropTop,:]=0
    if cropBot>0:
        pred[-cropBot:,:]=0
    if cropLeft>0:
        pred[:,:cropLeft]=0
    if cropRight>0:
        pred[:,-cropRight:]=0


    lines, ccs, ccStats = convertToLineSegments(pred, ccRes)

    if len(lines) == 0:
        return pred, None

    meanLen=0
    for line in lines:
        meanLen += getLen(line)
    meanLen/=len(lines)
    #print 'mean line: '+str(meanLen)

    lineIm = np.zeros(pred.shape)
    for line in lines:
        if line is not None:
            cv2.line(lineIm, (line['x1'],line['y1']), (line['x2'],line['y2']), 1, 1)
    hist = np.sum(lineIm, axis=0)
    if cropLeft<4 or cropRight<4: #we can skip if we found black on both ends
        #pageLine = getPageFold(origCropped) too hard

        #vert hist of lines

        #construct linear filter based on mean line length
        kValues = [0.0]*int(meanLen*0.75)
        lenh=int(meanLen*0.75)/2
        for i in range(lenh):
            kValues[i] = -1.0*(lenh-i)
            kValues[-i] = (lenh-i)
        kernelLeftEdge = np.array(kValues)/lenh
        #kernelLeftEdge = np.array([-3,-3,-3,-3,-2,-2,-2.0,-2,-2,-1,0,1,2,2,2,2,2,3,3,3,3])/15.0
        #kernelLeftEdge = np.array([-6,-5,-5,-4,-4,-4,-3,-3,-3,-3,-2,-2,-2.0,-2,-2,-1,0,1,2,2,2,2,2,3,3,3,3,4,4,4,5,5,6])/21.0
        #kernelRightEdge = np.array([2.0,2,2,1,0,-1,-2,-2,-2])/5.0
        leftEdges = cv2.filter2D(hist,-1,kernelLeftEdge,None, (-1,-1), 0, cv2.BORDER_REPLICATE)
        #rightEdges = cv2.filter2D(hist,-1,kernelRightEdge)
        #val = filters.threshold_otsu(hist)

        maxV = np.amax(leftEdges)
        minV = np.amin(leftEdges)

        threshLeft = minV+(maxV-minV)*0.5
        threshRight = minV+(maxV-minV)*0.5

        leftPeaks = []
        hitLeft=False
        leftV=0
        rightPeaks = []
        hitRight=True
        rightV=-9999999
        for x in range(1,leftEdges.shape[0]-1):
            if leftEdges[x]>threshLeft and leftEdges[x]>leftEdges[x-1] and leftEdges[x]>leftEdges[x+1]:
                if hitRight:
                    hitRight=False
                    rightV=0
                if hitLeft:
                    if leftEdges[x]>leftV:
                        leftV=leftEdges[x]
                        leftPeaks[-1]=x
                else:
                    leftPeaks.append(x)
                    hitLeft=True
                    leftV=leftEdges[x]
            if leftEdges[x]<threshRight and leftEdges[x]<leftEdges[x-1] and leftEdges[x]<leftEdges[x+1]:
                if hitLeft:
                    hitLeft=False
                    leftV=0
                if hitRight:
                    if leftEdges[x]<rightV:
                        rightV=leftEdges[x]
                        rightPeaks[-1]=x
                else:
                    rightPeaks.append(x)
                    hitRight=True
                    rightV=leftEdges[x]

        #oldLeftPeaks=leftPeaks[:]
        #oldRightPeaks=rightPeaks[:]

        #prune peaks, assuming max left mataches min right and so on
        newLeftPeaks=[]
        newRightPeaks=[]
        while len(leftPeaks)>0 and len(rightPeaks)>0:
            maxLeft=leftPeaks[0]
            maxLeftV=leftEdges[maxLeft]
            for l in leftPeaks[1:]:
                if leftEdges[l] > maxLeftV:
                    maxLeft=l
                    maxLeftV=leftEdges[maxLeft]

            i=0
            while i < len(rightPeaks) and rightPeaks[i]<maxLeft:
                i+=1
            if i == len(rightPeaks):
                #then maxLeft has no matching peak
                newLeftPeaks.append(maxLeft)
                leftPeaks.remove(maxLeft)
                continue
            minRight=rightPeaks[i]
            minRightV=leftEdges[minRight]
            for r in rightPeaks[i:]:
                if leftEdges[r] < minRightV:
                    minRight=r
                    minRightV=leftEdges[minRight]

            if maxLeft>=minRight:
                print 'Error in peak pruning: '+predFile
                break

            newLeftPeaks.append(maxLeft)
            newRightPeaks.append(minRight)
            i=0
            while i < len(leftPeaks):
                if leftPeaks[i]>=maxLeft and leftPeaks[i]<=minRight:
                    del leftPeaks[i]
                else:
                    i+=1
            i=0
            while i < len(rightPeaks):
                if rightPeaks[i]>=maxLeft and rightPeaks[i]<=minRight:
                    del rightPeaks[i]
                else:
                    i+=1

        #pickup spare right peak
        if len(rightPeaks)>0:
            minRight=rightPeaks[0]
            minRightV=leftEdges[minRight]
            for r in rightPeaks[0:]:
                if leftEdges[r] < minRightV:
                    minRight=r
                    minRightV=leftEdges[minRight]
            newRightPeaks.append(minRight)
            keepRight = rightPeaks[-1]
        else:
            keepRight = pred.shape[1]-1

        if len(leftPeaks)>0:
            minLeft=leftPeaks[0]
            minLeftV=leftEdges[minLeft]
            for r in leftPeaks[0:]:
                if leftEdges[r] < minLeftV:
                    minLeft=r
                    minLeftV=leftEdges[minLeft]
            newLeftPeaks.append(minLeft)
            keepLeft=leftPeaks[0]
        else:
            keepLeft=0

        leftPeaks=sorted(newLeftPeaks)
        rightPeaks=sorted(newRightPeaks)


        #drawing
        """
        leftEdges = np.reshape(leftEdges,(1,leftEdges.shape[0]))
        leftEdges[:] = (leftEdges[:]-minV)*(255.0/(maxV-minV))
        origCropped[0:30,:,1]=leftEdges
        origCropped[0:30,:,0]=0
        origCropped[0:30,:,2]=0

        #for x in oldLeftPeaks:
        #    origCropped[0:30,x,2]=255
        #for x in oldRightPeaks:
        #    origCropped[0:30,x,0]=255
        for x in leftPeaks:
            origCropped[0:30,x,:]=0
            origCropped[0:30,x,2]=255
        for x in rightPeaks:
            origCropped[0:30,x,:]=0
            origCropped[0:30,x,2]=255
        """

        #if len(leftPeaks)>2:
        #    print 'Warning: '+predFile+' post-proc may be in error. Too many sections starts, '+str(len(leftPeaks))+' detected.'
        #if len(rightPeaks)>2:
        #    print 'Warning: '+predFile+' post-proc may be in error. Too many sections ends, '+str(len(leftPeaks))+' detected.'

        #check if up agains edge
        if cropLeft<4:  #Left side
            prune=-1
            if len(rightPeaks)>1:
                if rightPeaks[0] < leftPeaks[0]:
                    if rightPeaks[0] < rightPeaks[1]-leftPeaks[0]:
                        prune= rightPeaks[0]
                        keepLeft = leftPeaks[0]
                else:
                    if leftPeaks[0]<meanLen*0.4 and rightPeaks[0]-leftPeaks[0] < rightPeaks[1]-leftPeaks[1]:
                        prune= rightPeaks[0]
                        keepLeft=leftPeaks[1]

            for i in range(len(lines)):
                line=lines[i]
                if (line['x1']<=meanLen/5 and getLen(line)<meanLen*0.75 and line['x2']<keepLeft) or (prune!=-1 and prune-line['x1']>line['x2']-prune):
                    removeCC(line['cc'],ccs,ccStats,pred)
                    lines[i]=None

        if cropRight<4: #Right side
            width = orig.shape[1]
            prune=-1
            if len(leftPeaks)>1:
                print leftPeaks
                print rightPeaks
                if rightPeaks[-1] < leftPeaks[-1]:
                    if width-leftPeaks[-1] < rightPeaks[-1]-leftPeaks[-2]:
                        prune= leftPeaks[-1]
                        keepRight = rightPeaks[-1]
                else:
                    if rightPeaks[-1]-leftPeaks[-1] < rightPeaks[-2]-leftPeaks[-2]:
                        prune= leftPeaks[-1]
                        keepRight = rightPeaks[-2]

            for i in range(len(lines)):
                line=lines[i]

                if line is not None and ((line['x2']>=pred.shape[1]-(1+meanLen/5) and getLen(line)<meanLen*0.75 and line['x1']>keepRight) or (prune!=-1 and prune-line['x1']<line['x2']-prune)):
                    removeCC(line['cc'],ccs,ccStats,pred)
                    lines[i]=None




    #draw
    """
    origCropped[:,0]=(255,0,0)
    origCropped[:,-1]=(255,0,0)
    origCropped[0,:]=(255,0,0)
    origCropped[-1,:]=(255,0,0)


    #origCropped[:,:,1]=pageLine

    for line in lines:
        if line is not None:
            cv2.line(origCropped, (line['x1'],line['y1']), (line['x2'],line['y2']), (0,0,255), 7)
            #cv2.putText(origCropped, str(int(getLen(line))), (line['x1'],line['y1']), cv2.FONT_HERSHEY_PLAIN, 4, (255,0,0),3)

    #cv2.imshow('res',orig)

    #cv2.waitKey()
    cv2.imwrite('test.png',orig)
    """
    trans01=[]
    trans10=[]
    for x in range(1,hist.shape[0]):
        if hist[x-1]<=3 and hist[x]>3:
            trans01.append(x)
        if hist[x-1]>3 and hist[x]<=3:
            trans10.append(x-1)
    #print trans01
    #print trans10

    boundariesRet=[]
    #leftBs=[x for x in trans01 if x>=keepLeft and x<=keepRight]
    #rightBs=[x for x in trans10 if x>=keepLeft and x<=keepRight]
    #print leftBs
    #print rightBs
    if len(trans01)!=0 and len(trans10)!=0:
        leftBs = trans01
        rightBs = trans10
        lastLeft=leftBs[0]
        lastRight=rightBs[0]
        leftI=0
        rightI=0
        while leftI<len(leftBs) and rightI<len(rightBs):
            while rightI<len(rightBs) and rightBs[rightI]<leftBs[leftI]:
                rightI+=1
            lastLeft=leftBs[leftI]
            rightB_ = pred.shape[1]
            if rightI<len(rightBs):
                rightB_=rightBs[rightI]
            while leftI<len(leftBs) and leftBs[leftI]<rightB_:
                leftI+=1
            while rightI<len(rightBs) and (leftI>=len(leftBs) or rightBs[rightI]<leftBs[leftI]):
                rightI+=1

            boundariesRet.append((lastLeft,rightBs[rightI-1]))

    return pred, boundariesRet

def getClusterLine(cluster,bb):
    leftY=0
    leftCount=0
    rightY=0
    rightCount=0
    for line in cluster:
        if abs(line[0]-bb[0])<5:
            leftY+=line[1]
            leftCount+=1
        if abs(line[2]-bb[2])<5:
            rightY+=line[3]
            rightCount+=1

    return (bb[0], leftY/leftCount, bb[2], rightY/rightCount)

def clusterPrune(lines,pred, ccLabels):

    ccMap={}
    cluster={}
    for line in lines:
        y=line[1]
        step=1
        if line[2]<line[0]:
            step=-1
        if line[2] == line[0]:
            continue
        slope = float(line[3]-line[1])/float(line[2]-line[0])
        i=0
        ccFirst=None

        for x in range(line[0],line[2],step):
            y = int(line[1] + i*slope)
            if y>max(line[1],line[3]) or y<min(line[1],line[3]):
                print (x,y,line,slope)
                assert False
            cc = ccLabels[y,x]
            if cc==0 or pred[y,x]==0:
                continue
            while cc in ccMap and ccMap[cc] is not None:
                cc=ccMap[cc]
            if ccFirst is None:
                ccFirst=cc
                if cc not in cluster:
                    cluster[cc]=[]
            elif ccFirst != cc:
                ccMap[cc]=ccFirst
                if cc in cluster:
                    cluster[ccFirst] += cluster[cc]
                    cluster[cc]=None
            cluster[ccFirst].append(line)


            i+=1

    ret = []
    for cc,cLines in cluster.items():
        if cLines is not None:
            maxDist=0
            maxLine=None
            for line in cLines:
                dist = math.sqrt( ((line[0]-line[2])**2) + ((line[1]-line[3])**2) )
                if dist>maxDist:
                    maxDist=dist
                    maxLine=line
            ret.append(maxLine)
    return ret


def lineEq(line):
    m = float(line[3]-line[1])/float(line[2]-line[0])
    b = line[1]-m*line[0]
    return m,b

def goodIntersection(line1, line2):
    m1, b1 = lineEq(line1)
    m2, b2 = lineEq(line2)
    if m1==m2:
        return False
    xIntersection = (b1-b2)/(m2-m1)
    return xIntersection>min(line1[0],line2[0]) and xIntersection<max(line1[0],line2[0]) and \
            xIntersection>min(line1[2],line2[2]) and xIntersection<max(line1[2],line2[2])


def connectLines(pred,boundaries,splits, ccRes, rhoRes, thetaRes, threshold, minLineLength, maxLenGap):


    img=pred.copy()
    element = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
    predEroded = cv2.erode(pred,element)
    lines = cv2.HoughLinesP(predEroded, rhoRes, thetaRes, threshold, None, minLineLength, maxLenGap)


    angles=[]
    angleMean=0
    if lines is None:
       return
    for line in lines:
        x1=line[0,0]
        y1=line[0,1]
        x2=line[0,2]
        y2=line[0,3]
        angle = math.atan2(y2-y1,x2-x1)
        angleMean += angle
        angles.append(angle)

    angleMean /= len(angles)
    angleStd=0
    for angle in angles:
        angleStd += (angleMean-angle)**2
    angleStd = math.sqrt(angleStd/len(angles))


    prunedLines=[]

    #potting.idprune by angle
    if angleStd!=0:
        for line in lines:
            x1=line[0,0]
            y1=line[0,1]
            x2=line[0,2]
            y2=line[0,3]
            angle = math.atan2(y2-y1,x2-x1)
            if abs((angle-angleMean)/angleStd)<2.5:
                prunedLines.append((x1,y1,x2,y2))

    prunedLines2 = clusterPrune(prunedLines, pred, ccRes)

    prunedLines3 = prunedLines2[:]
    for i in range(len(prunedLines2)):
        for j in range(i,len(prunedLines2)):
            if goodIntersection(prunedLines2[i],prunedLines2[j]):
                if prunedLines2[i][0]>prunedLines2[j][2]:
                    prunedLines3.append( (prunedLines2[i][0],prunedLines2[i][1],prunedLines2[j][2],prunedLines2[j][3]) )
                    #cv2.line(pred, (prunedLines2[i][0],prunedLines2[i][1]), (prunedLines2[j][2],prunedLines2[j][3]), 255, 7)
                elif prunedLines2[j][0]>prunedLines2[i][2]:
                    prunedLines3.append( (prunedLines2[j][0],prunedLines2[j][1],prunedLines2[i][2],prunedLines2[i][3]) )
                    #cv2.line(pred, (prunedLines2[j][0],prunedLines2[j][1]), (prunedLines2[i][2],prunedLines2[i][3]), 255, 7)
    #prune by boundaries
    #print boundaries
    for i in range(1,len(boundaries)):
        dontCross = (boundaries[i-1][1]+boundaries[i][0])/2
        #orig[:,dontCross]=(255,0,0)
        #orig[:,boundaries[i-1][1]]=(155,0,0)
        #orig[:,boundaries[i][0]]=(155,0,0)
        for l in range(len(prunedLines3)):
            line = prunedLines3[l]
            if line is not None and min(line[0],line[2])<dontCross and max(line[0],line[2])>dontCross:
                prunedLines3[l]=None
    if splits is not None:
        for dontCross in splits:
            #orig[:,dontCross]=(255,0,0)
            #orig[:,boundaries[i-1][1]]=(155,0,0)
            #orig[:,boundaries[i][0]]=(155,0,0)
            for l in range(len(prunedLines3)):
                line = prunedLines3[l]
                if line is not None and min(line[0],line[2])<dontCross and max(line[0],line[2])>dontCross:
                    prunedLines3[l]=None




    #for (x1,y1,x2,y2) in prunedLines:
        #cv2.line(orig, (x1,y1), (x2,y2), (0,0,255), 1)
    for line in prunedLines3:
        if line is not None:
            x1,y1,x2,y2 = line
            #cv2.line(orig, (x1,y1), (x2,y2), (0,255,0), 1)
            cv2.line(pred, (x1,y1), (x2,y2), 255, 7)

            #debuf
            #for dontCross in splits:
            #    if pred[:,dontCross].sum()>0:
            #        print line
            #        print dontCross

    #cv2.imwrite('test.png',orig)
    #cv2.imwrite('testPred.png',pred)
    #exit(0)

def splitBaselines(pred,ccRes,split):
    CUT_THRESH=60
    if split<CUT_THRESH-2 or pred.shape[1]-split<CUT_THRESH-2:
        return
    okCCs=[]
    dontCCs=[]
    ccs=ccRes[1]
    ccStats=ccRes[2]
    for y in range(pred.shape[0]):
        if pred[y,split]>0:
            cc = ccs[y,split]
            if cc in okCCs:
                pred[y,split-1:split+2]=0
            elif cc not in dontCCs:
                left = split-ccStats[cc][cv2.CC_STAT_LEFT]
                right = (ccStats[cc][cv2.CC_STAT_LEFT]+ccStats[cc][cv2.CC_STAT_WIDTH]-1)-split
                #ok=True
                #for direction in [-1,1]:
                #    #is the CC on this side long enough to split?
                #    for len in range(1,CUT_THRESH):
                #        cont=False
                #        for yCheck in range (ccStats[cc][cv2.CC_STAT_TOP],ccStats[cc][cv2.CC_STAT_TOP]+ccStats[cc][cv2.CC_STAT_HEIGHT]):
                #            if pred[yCheck,split+direction*len]>0 and ccs[yCheck,split+direction*len]==cc:
                #                #pred[yCheck,split+direction*len]=155
                #                cont=True
                #                break
                #        if not cont:
                #            ok=False
                #            break
                #    if not ok:
                #        break
                if left>CUT_THRESH and right>CUT_THRESH:
                    okCCs.append(cc)
                    pred[y,split-1:split+2]=0
                else:
                    dontCCs.append(cc)

def linePreprocess(pred,orig, ccRes, complex=False):
    rhoRes = 1.0
    thetaRes = math.pi/180
    threshold = 200
    minLineLength = 30
    maxLenGap = 200

    if pred.sum() == 0:
        return pred
    boundaries = []
    # newPred, boundaries = removeSidePredictions(pred,orig, ccRes)
    # if boundaries is None: #meaning there were no baselines predicted
    #     return pred
    #splits = [get_split_col(orig,pred)]
    splits = None
    if complex:
        #threshold=2000
        #maxLenGap=30
        if orig.shape[2]>1:
            gray = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
        else:
            gray=orig
        splits = get_vert_lines(gray)
        for split in splits:
            #pred[:,split-1:split+2]=0
            splitBaselines(pred,ccRes,split)
    else:
        connectLines(pred,boundaries,splits,ccRes[1], rhoRes, thetaRes, threshold, minLineLength, maxLenGap)
    return pred

if __name__ == "__main__":

    #file = sys.argv[1]

    #predFile = '../../results/cbad_simple_base_weights_round_weighted_1_3/train/verbose/'+file+'/pred.png'
    #origFile = '../../results/cbad_simple_base_weights_round_weighted_1_3/train/verbose/'+file+'/pred_on_original.png'

    if len(sys.argv) < 4:
        print 'Usage: '+sys.argv[0]+' predImage origImage outPredImage [complex]'
        exit(0)

    predFile = sys.argv[1]
    origFile = sys.argv[2]
    outName = sys.argv[3]

    complex = len(sys.argv) > 4

    pred = cv2.imread(predFile,0)
    assert pred is not None
    orig = cv2.imread(origFile)
    assert orig is not None

    ccRes = cv2.connectedComponentsWithStats(pred, 4, cv2.CV_32S)
    finalPred = linePreprocess(pred,orig,ccRes, complex)

    cv2.imwrite(outName,finalPred)
