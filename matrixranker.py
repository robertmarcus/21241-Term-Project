import os
import numpy as np
from PIL import Image #aka python imaging library
from sklearn.metrics.pairwise import cosine_distances ### use this to measure similarity!!
import random
import math
import csv 
import copy
#from scipy import spatial

print("This program will take up to several minutes to load. Please be patient.")

def uniform_find(art,grid_level): #for an image and a subdivisioning size, find the average [r,g,b] of each subgrid
    #grid_level = 0, full canvas, grid_level = 1, 4 quadrants, in general, 4**grid_level subdivisions
    #method works for canvases of any dimension
    print("Currently computing the average color of ", 4**grid_level, "subdivisions of", art.title)
    img_file = Image.open(art.filename) #open image (find size info etc)
    img = img_file.load() #load image (for reading pixels)
    
    [x_std,y_std] = img_file.size #[x,y] dimensions of the canvas
    aspect_ratio = x_std/y_std
    
    if(x_std%2==1): #verify even dimensions for subdivisioning. Can't easily subdivide for odd sized dimension!
        x_std-=1
    if(y_std%2==1):
        y_std-=1

    avg_subgrid_list = [] #will return avg [r,g,b] of each subgrid
    subgrid_colors = [0,0,0] #measures total r,g,b values of each subgrid
    
    number_subdivision = 2**grid_level #how many divisions are made on x axis, y axis
    
    block_size_increments_x = x_std//number_subdivision #x,y dimensions for each subdivision (rectangular subdivisions)
    block_size_increments_y = y_std//number_subdivision
    pixels_per_subgrid = block_size_increments_x*block_size_increments_x #used for averaging r,g,b values
    
    start_x = 0
    start_y = 0
    
    for i in range(number_subdivision):
        for j in range(number_subdivision):
            
            start_x=i*block_size_increments_x #initial position for each new grid x,y
            start_y=j*block_size_increments_y
            for x in range(start_x,start_x+block_size_increments_x): #hit every pixel in a grid with this method
                for y in range(start_y,start_y+block_size_increments_y):
                    [r,g,b]=img[x,y]
                    subgrid_colors[0]+=r 
                    subgrid_colors[1]+=g
                    subgrid_colors[2]+=b 
                    
            subgrid_colors[0]/=pixels_per_subgrid #sum all r values and find average throughout grid
            subgrid_colors[1]/=pixels_per_subgrid #same for g
            subgrid_colors[2]/=pixels_per_subgrid #same for b
            
            avg_subgrid_list.append(subgrid_colors)  #populate avg_subgrid_list with avg's
            subgrid_colors = [0,0,0] #clear      
    return(avg_subgrid_list,aspect_ratio) #returns a 2d vector with 4**grid_level many [r,g,b] values

def compareArt(a,b):
    return(np.mean(1-cosine_distances(a[0],b[0])))# return similarity measure of two images AvgColorList... take mean b/c otherwise is 1d array of size=number of quadrants

class Art(object): #artist_domain_name_year.jpg
    def __init__(self, name, artist, title, domain, date, value0, value1, value2, value3, value4, avg_similarity, compute_time,aspect_ratio):
        self.filename = "{}".format(name)
        self.artist = artist #name.split("_")[0]
        self.title = title #name.split("_")[2]
        self.domain = domain #will either be public or private
        self.date = date #when painting was painted
        self.zero = uniform_find(self, value0) 
        self.one = uniform_find(self, value1)
        self.two = uniform_find(self, value2)
        self.three = uniform_find(self, value3)
        self.four = uniform_find(self, value4)
        self.avg_similarity = avg_similarity #defined later 
        self.compute_time = compute_time #defined later
        self.aspect_ratio = aspect_ratio #defined in uniform_find
        
artList = [] #all art in the file
publicList = [] #all public domain art
privateList = [] #all private domain art
overall_matrix = [] #2d matrix with each entry = [delta_date, compare_zero,one,two,three,four,avg, random, aspect_ratio]


for root, dirs, files in os.walk("."): #find files in our folder
    for filename in files:  #look through the file!
        if (filename.split('.')[1]=='jpg'): #make sure we are only inspecting jpg's!!!
            art = Art(filename,(filename.split("_")[0]).capitalize(), filename.split("_")[2].replace("-"," "), filename.split("_")[1],filename.split(".")[0][-4:],0,1,2,3,4,0,0,0)
            art.compute_time=(art.zero)[1]+(art.one)[1]+(art.two)[1]+(art.three)[1]+(art.four)[1]#was having bug doing this with the rest above...
            art.aspect_ratio = art.zero[1]
            artList.append(art) #populate list for all art
            if(art.domain=="public"): publicList.append(art) #check if public or private
            else: privateList.append(art)
            
def pixel_list_creator(searchList, img): #for a list of pixel coordinates, create a list of [r,g,b] values at those coordinates
    
    pixel_list = []
    
    for coordinates in searchList:
        x = coordinates[0] #searchList is a 2d list in form [[x,y],...]
        y = coordinates[1]
        [r,g,b] = img[x,y] #img already comes loaded!
        pixel_list.append([r,g,b])
    return pixel_list #returns a 2d list in form [[r,g,b],...]

def search_boundary(a,b): #e.g., artList[i], artList[i+1]
    img_file_a = Image.open(a.filename) #open two different images
    img_file_b = Image.open(b.filename)
    
    img_a = img_file_a.load() #load
    img_b = img_file_b.load()
    
    [xa,ya] = img_file_a.size #determine size of each
    [xb,yb] = img_file_b.size
    
    xBound = min(xa,xb) #this method is set up to compare the same pixels in overlapping regions of two potentially different images
    yBound = min(ya,yb) #overlapping region is found as min(x_a,x_b),min(y_a,y_b). otherwise will have out of bound problems...
    #Note: Uniform find can also work with non-equal sized images. 
    searchSetSize = math.floor((xBound*yBound)/1000) #WARNING: any 'lower' than 1000 and runtime skyrockets because of recurse feature below...
    if(searchSetSize>1000): searchSetSize = 1000
    #^quantify a limit to the number of random pixels we will choose to compare - don't want to use all of them! fewer the better
    searchList = [] #to be populated by above method

    while(len(searchList)<=searchSetSize): #fill with searchSetSize many random (x,y) pixels that are in both img_a and img_b
        possible_x = random.randint(0,xBound)
        possible_y = random.randint(0,yBound)
        if [possible_x,possible_y] not in searchList: #check if [x,y] is in searchList e.g., [[x_i,y_i],...] yet
            searchList.append([possible_x,possible_y]) #if not yet in, add it! NOTE: using sets would be preferable but was buggy with 2d lists...
    try:
        return(np.mean(1-cosine_distances(pixel_list_creator(searchList,img_a),pixel_list_creator(searchList,img_b)))) #See if this works! See below note for why we recurse!
    except:
        return(search_boundary(a,b)) #for some reason doesn't always work - gets image index out of range, so just recurse until it works. Ugly but solves bug!


def compare_random(art, target_index): #pick an image in the private catalog w target_index and compare a random set of pixels for a similarity score!

    comparison_results = dict()
    
    for art in publicList:
        comparison_results[art]=search_boundary(privateList[target_index],art)
    return(comparison_results)
    
def compare_hybrid(random,uniform): #make hybrid comparison from avg of random and uniform!
    compare_hybrid_results = 0
    
    
    weight_random=2
    weight_uniform=1
    
    compare_hybrid_results = (weight_random*random+weight_uniform*uniform)/(weight_random+weight_uniform)
    
    return(compare_hybrid_results)
    
def maxItemLength(a):
    maxLen = 0
    rows = len(a)
    cols = len(a[0])
    for row in range(rows):
        for col in range(cols):
            maxLen = max(maxLen, len(str(a[row][col])))
    return maxLen

def print2dList(a):
    if (a == []):
        # So we don't crash accessing a[0]
        print([])
        return
    rows = len(a)
    cols = len(a[0])
    fieldWidth = maxItemLength(a)
    print("[ ", end="")
    for row in range(rows):
        if (row > 0): print("\n  ", end="")
        print("[ ", end="")
        for col in range(cols):
            if (col > 0): print(", ", end="")
            # The next 2 lines print a[row][col] with the given fieldWidth
            formatSpec = "%" + str(fieldWidth) + "s"
            print(formatSpec % str(a[row][col]), end="")
        print(" ]", end="")
    print("]")
            
#want to return MAX after weights are applied    
def apply_weights(matrix):
    #Want to normalize everything to 1, so find max in a given column, make 1, then everything else a percetage of that
    rows = len(matrix)
    cols = len(matrix[0])
    print2dList(matrix)
    #run through column entries, add to new list, return max, rewrite everything in the new list as a ratio of that. 
    for i in range(1,rows):
        column_entries_list = []
        maximum_element = 0
        for j in range(1,cols): 
            print(matrix[i][j])
            column_entries_list.append(matrix[i][j])
        maximum_element = max(column_entries_list)
        for j in range(cols):
            matrix[i][j] /= maximum_element
    return(matrix)
            
def make_unitary(matrix):
    rows = len(matrix)
    cols = len(matrix[0])
    
    for j in range(1,cols):
        column_entries = []
        for i in range(rows):
            column_entries.append(matrix[i][j])
        #print(column_entries)
        maximum_element = max(column_entries)
        for i in range(rows):
            matrix[i][j]/=maximum_element
    return((matrix))
    
def sum_weighted_matrix(matrix):
    summed_matrix = []
    for i in range(len(matrix)):
        title = (matrix[i][0]).title
        artist = (matrix[i][0]).artist
        row_sum=0
        for j in range(1,len(matrix[0])):
            row_sum+=matrix[i][j]
        summed_matrix.append([title,artist,row_sum])
    return summed_matrix

def rank_style_one(matrix):
    #rank_one_weights = [None, math.log10(1/input), input,input,input,input,input*20,input*20,input*10,input*20]
    matrix_ranked_style_one = copy.deepcopy(make_unitary(matrix))
    rows = len(matrix)
    for i in range(rows):
        #matrix_ranked_style_one[i][1] = math.log10(matrix[i][1]**(-1))
        if(matrix[i][1]!=0.0): 
            matrix_ranked_style_one[i][1] = math.log10(1/matrix[i][1])
            print(matrix_ranked_style_one[i][1],matrix[i][1])
        else:
            matrix_ranked_style_one[i][1] = 1
        matrix_ranked_style_one[i][7] = matrix[i][7]*20
        matrix_ranked_style_one[i][8] = matrix[i][8]*20
        matrix_ranked_style_one[i][9] = matrix[i][9]*10
        matrix_ranked_style_one[i][10] = matrix[i][10]*20
    return(sum_weighted_matrix(matrix_ranked_style_one))


def rank_style_two(matrix):
    #ART / LOG(1/DD)/ *1           / * 1 /*1   /    *1 /    *1/ *5 /  *20   / *10         /  *25 
    matrix_ranked_style_two = copy.deepcopy(make_unitary(matrix))
    rows = len(matrix)
    for i in range(rows):
        if(matrix[i][1]!=0.0): 
            matrix_ranked_style_two[i][1] = math.log10(1/matrix[i][1])
            print(matrix_ranked_style_two[i][1],matrix[i][1])
        else:
            matrix_ranked_style_two[i][1] = 1
        matrix_ranked_style_two[i][7] = matrix[i][7]*5
        matrix_ranked_style_two[i][8] = matrix[i][8]*20
        matrix_ranked_style_two[i][9] = matrix[i][9]*10
        matrix_ranked_style_two[i][10] = matrix[i][10]*2
    return(sum_weighted_matrix(matrix_ranked_style_two))

def weigh_and_sort_matrix(matrix):
    sorted_matrix_by_decreasing_sums_one = sorted(rank_style_one(matrix),key=lambda x: x[2],reverse=True)
    sorted_matrix_by_decreasing_sums_two = sorted(rank_style_two(matrix),key=lambda x: x[2],reverse=True)
 #sorted(rank_style_one(matrix).items(), key=lambda kv: kv[1],reverse=True) #inspired by: https://stackoverflow.com/questions/613183/how-do-i-sort-a-dictionary-by-value 
    return(sorted_matrix_by_decreasing_sums_one,sorted_matrix_by_decreasing_sums_two) #dict
            
    
    

def main(): #create each entry in matrix
    private = privateList[0]
    delta_date = 0
    compare_at_zero_subdivision = 0
    compare_at_one = 0
    compare_at_two = 0
    compare_at_three = 0
    compare_at_four = 0
    avg_compare = 0 #(compare_at_zero_subdivision+compare_at_one+compare_at_two+compare_at_three+compare_at_four)/5
    random_compare = 0
    hybrid = 0
    aspect_ratio = 0
    for art in publicList: #format:
        delta_date = abs(int(private.date)-int(art.date))
        compare_at_zero_subdivision = compareArt(art.zero,private.zero)
        compare_at_one = compareArt(art.one,private.one)
        compare_at_two = compareArt(art.two,private.two)
        compare_at_three = compareArt(art.three,private.three)
        compare_at_four = compareArt(art.four,private.four)
        avg_compare = (compare_at_zero_subdivision+compare_at_one+compare_at_two+compare_at_three+compare_at_four)/5
        
        solid_state_random = search_boundary(art,private)
        
        random_compare = solid_state_random
        hybrid = (solid_state_random*2+avg_compare)/3   #numpy.dot(art.aspect_ratio,private.aspect_ratio)
        #print("Before",art.title, art.aspect_ratio,private.aspect_ratio)
        aspect_ratio = 1/np.dot(art.aspect_ratio,private.aspect_ratio) #abs(art.aspect_ratio-private.aspect_ratio) #need to improve so don;'t end up with a potential 0...
        #print("Post",aspect_ratio,)
        row = [art, delta_date,compare_at_zero_subdivision,compare_at_one,compare_at_two,compare_at_three,compare_at_four,avg_compare,random_compare,hybrid,aspect_ratio]
        overall_matrix.append(row)
#    with open("unweighted.csv","w+") as my_csv:
#        csvWriter = csv.writer(my_csv,delimiter=',')
#        csvWriter.writerows(overall_matrix)
#    with open("weighted.csv","w+") as my_csv:
#        csvWriter = csv.writer(my_csv,delimiter=',')
#        csvWriter.writerows(make_unitary(overall_matrix))
    try:
        os.remove('weight_one.csv')
    except:
        pass
    try: os.remove('weight_two.csv')
    except: pass
    with open("weight_one.csv","w+") as my_csv:
        csvWriter = csv.writer(my_csv,delimiter=',')
        csvWriter.writerows(weigh_and_sort_matrix(overall_matrix)[0])
    with open("weight_two.csv","w+") as my_csv:
        csvWriter = csv.writer(my_csv,delimiter=',')
        csvWriter.writerows(weigh_and_sort_matrix(overall_matrix)[1])
    return(print2dList(weigh_and_sort_matrix(overall_matrix)))
main()
        
#FORMAT: 
#ART / DELTA_DATE / UNIFORM.ZERO / ONE / TWO / THREE / FOUR / AVG / RANDOM / MIX UNI RAN / ASPECT RATIO
#ART / 1 / *1           / * 1 /*1   /    *1 /    *1/ *20 /  *20   / *10         /  *20
#ART / 1/ *1           / * 1 /*1   /    *1 /    *1/ *5 /  *20   / *10         /  *2
        
        
        
        
        
        
        