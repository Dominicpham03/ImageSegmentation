Object Recogniton 
For Object Recognition I would first extract data from the txt which I would then use string manipulation to strip the file 
location and label data for future reference. then I would open each image and extract descriptor and would append them all 
into a array with all descriptors from all photos. After word I would perform k means clustering where I would create ten cluster  
which would allow me to produce my list of centriods. Then I would extract features from the img which I would then compare the descriptors 
with my centriods by using the Euclidean Distance.  If the length was under 10000 then I would add 1 to the histogram feature vector which 
represent each centriod. I would then return the histogram as a array. For train_classifer I would use K nearest neighbor where I set the 
n_nbeighbors to 3. I would thern iterate through each image and extract features to create my histogram features. I would then append all 
feature vector together and fit it to my training model which would then complete my training process. For the last procedure classify image 
I would extract features form the test_img and I would then obtain my feat_vector. I would then reshape the vector to make sure that it fits 
the dimension of the classifer.Last of all i performed classifer.predict which would then give me my output where 1 is a  cat and 0 is the dog 
anything else would produce no results. 


Segmentation
For thresholding I decided to iterate through each pixel and examine where a pixel is a strong  or weak  by providing a specfic threshold number. 
If the image is in between the threshold then I labeld thje pixel as 128. I would then iterate through the image once again and pair pixel whose value 
is 128 and would see whether the local pixel is similar. If it is then I would label the pixel black.

For grow region I selected a specfic pixel which is a part of the cat. I would then create a stack and include the current pixel which we would then go 
into a while loop where the loop doesnt end until the stack is empty. I would then computer the euclidean distance and check wheter the distance is under a 
specfic threshold if it is we would look at its neighbor and add the neighboring pixel into the stack. We would continue until there is no pixel left.

For split region I took a similar apporach and create a region where it would take in the current region and its dimension size. I would then 
divide the input depending on the variance of pixel intensity values within each region. I would then split region in the image if the variance within the 
region is greater than a aspecfic threshold it would continue to splut until the region variance is within the acceptable range or the region is just a single pixel.

For merge region I use a ;obaru function slic which segment the image into section which makes it easier to define the merging map. Then I used regionsprops 
from skilearn which return properties of each region where I use it to find the mean intensity.Then I created a label map which keep track of which region 
should be merge and then iterate over the paris of region checking wherether the difference in mean intesnity value is less than .01. finally a bimnary 
image is created which represent the regions. The function would return the binary image which represent merging based on mean intensity similarity.


Segment_image 
I decided to leave threshold and grow region by itself because it seems that it produced the best results when using it alone. 
I paried split_region and merge because it was discuss upon lecture that it was ideal to use both algorithm to create an efficent segmentation 
which it did. Besides that the results were not amazing for split_regions and merge probably because my planning was lackluster with the implementation
