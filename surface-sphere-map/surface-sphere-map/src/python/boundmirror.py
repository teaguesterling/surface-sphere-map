from numpy import *
def BoundMirrorExpand(A):
    '''% Expand the matrix using mirror boundary condition
    % 
    % for example 
    %
    % A = [
    %     1  2  3  11
    %     4  5  6  12
    %     7  8  9  13
    %     ]
    %
    % B = BoundMirrorExpand(A) will yield
    %
    %     5  4  5  6  12  6
    %     2  1  2  3  11  3
    %     5  4  5  6  12  6 
    %     8  7  8  9  13  9 
    %     5  4  5  6  12  6
    %'''
    [m,n] = A.shape
    yi = arange(1,m+1,dtype = int)
    xi = arange(1,n+1,dtype = int)
    B = zeros((m+2,n+2))
    #B[ix_([yi],[xi])] = A.copy()
    for y in range(0,m):
        for x in range(0,n):
            B[y+1,x+1] = A[y,x]
            
    
    B[ix_([0,m+1],[0,n+1])] = B[ix_([2,m-1],[2,n-1])]# mirror corners
    B[ix_([0, m+1],xi)] = B[ix_([2, m-1],(xi))] # mirror left and right boundary
    B[ix_((yi),[0,n+1])] = B[ix_((yi),[2,n-1])] # mirror left and right boundary

    
    return B

def BoundMirrorShrink(A):
# Shrink the matrix to remove the padded mirror boundaries
#
# for example 
#
# A = [
#     5  4  5  6  12  6
#     2  1  2  3  11  3
#     5  4  5  6  12  6 
#     8  7  8  9  13  9 
#     5  4  5  6  12  6
#     ]
# 
# B = BoundMirrorShrink(A) will yield
#
#     1  2  3  11
#     4  5  6  12
#     7  8  9  13
    [m,n] = A.shape
    yi = arange(1,m-1)
    xi = arange(1,n-1)
    B = A[ix_(yi,xi)]
    return B

def BoundMirrorEnsure(A):
    [m,n] = A.shape
    if (m<3 or n<3): 
        print "error, size not suitable"
        return
    B = A.copy()
                
    yi = arange(1,m-1,dtype = int)
    xi = arange(1,n-1,dtype = int)
    B = A.copy();
    B[ix_([0,m-1],[0,n-1])] = B[ix_([2,m-1],[2,n-1])] # mirror corners
    B[ix_([0,m-1],xi)] = B[ix_([2,m-3],xi)] # mirror left and right boundary
    B[ix_(yi,[0,n-1])] = B[ix_(yi,[2,n-3])] # mirror top and bottom boundary
    return B

#A = mat("1 2 3 11; 4 5 6 12; 7 8 9 13")
#print A
#print BoundMirrorExpand(A)
