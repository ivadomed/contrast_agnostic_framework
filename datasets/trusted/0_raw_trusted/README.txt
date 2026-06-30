*** TRUSTED DATASET ***


*SUMMARY*

Here is the TRUSTED dataset.

IMPORTANT!  Any user must read and sign the Licence.txt file.


The folder "TRUSTED_dataset_for_submission" contains:
 
1- The CT data (in "CT_DATA"):
 	* 48 CT images, each having two kidneys (TRUSTED_dataset_for_submission/CT_DATA/CT_images/)
 	* 48 manual segmentation masks of the kidneys from Annotator1, Annotator2, and the estimated Ground-Truth (TRUSTED_dataset_for_submission/CT_DATA/CT_masks/)
 	* 96 sets of landmarks (each per kidney) from Annotator1, Annotator2, and the estimated Ground-Truth (TRUSTED_dataset_for_submission/CT_DATA/CT_landmarks/)
  	* 96 sets of landmarks (each per kidney) from Annotator1, Annotator2, and the estimated Ground-Truth, in the CT device coordinate systems (TRUSTED_dataset_for_submission/CT_DATA/CT_landmarks_inimg/)
 	* 96 meshes of the kidney surfaces (each per kidney) from Annotator1, Annotator2, and the estimated Ground-Truth (TRUSTED_dataset_for_submission/CT_DATA/CT_meshes/)
  	* 96 meshes of the kidney surfaces (each per kidney) from Annotator1, Annotator2, and the estimated Ground-Truth, in the CT device coordinate systems (TRUSTED_dataset_for_submission/CT_DATA/CT_meshes_inimg/)
 	* 48 transform matrices to get the landmarks in the CT device coordinate systems (TRUSTED_dataset_for_submission/CT_DATA/CT_tbackldk_transforms/)
 	* 48 transform matrices to get the meshes in the CT device coordinate systems (TRUSTED_dataset_for_submission/CT_DATA/CT_tbackmesh_transforms/)
 	
   The 5-fold cross validation splits (patient ID):     
    CT_cv1 = ['263', '794', '592', '206', '579', '915', '250', '905', '249']
    CT_cv2 = ['561', '418', '636', '258', '283', '737', '610', '517', '801']
    CT_cv3 = ['443', '506', '641', '398', '711', '532', '371', '721', '735', '680']
    CT_cv4 = ['755', '510', '701', '948', '329', '239', '284', '656', '704', '399']
    CT_cv5 = ['861', '200', '220', '716', '348', '466', '738', '314', '832', '406']
 		
 	
2- The US data (in "US_DATA"):
 	* 59 US images, each having a single kidney (TRUSTED_dataset_for_submission/US_DATA/US_images/)
 	* 59 manual segmentation masks of the kidneys from Annotator1, Annotator2, and the estimated Ground-Truth (TRUSTED_dataset_for_submission/US_DATA/US_masks/)
 	* 59 sets of landmarks (each per kidney) from Annotator1, Annotator2, and the estimated Ground-Truth
 	(TRUSTED_dataset_for_submission/US_DATA/US_landmarks/)
 	* 59 meshes of the kidney surfaces (each per kidney) from Annotator1, Annotator2, and the estimated Ground-Truth (TRUSTED_dataset_for_submission/US_DATA/US_meshes/)
 	* 59 meshes of the kidney surfaces in the US device coordinate systems (TRUSTED_dataset_for_submission/US_DATA/US_meshes_inimg/)
 	
   The 5-fold cross validation splits (patient kidney ID): 
    US_cv1 = ['263R', '263L', '794R', '794L', '592R', '206R', '206L', '579R', '579L', '915L', '250R', '250L']
    US_cv2 = ['561R', '418R', '418L', '636R', '258R', '258L', '283L', '610L', '517R', '517L']
    US_cv3 = ['443R', '506R', '506L', '641R', '641L', '398R', '711L', '532R', '532L', '371R', '721L', '735R', '680L']
    US_cv4 = ['755R', '755L', '510R', '510L', '701R', '701L', '948R', '948L', '329R', '329L', '704L', '399R']
    US_cv5 = ['861R', '861L', '200R', '220R', '716R', '348R', '348L', '466R', '466L', '738R', '314R', '314L']  
 	

3-The initial transforms from noisy landmarks (in "Initial_Transforms_from_noisy_landmarks"), used for the initialization sensitivity analysis:
	* Five sub-folders named "ldks_transforms_stdX" (X in [2.0, 4.0, 6.0, 8.0, 10.0] represents the standard deviation of the white Gaussian noise), storing the 5 transforms of each  and ldks_transforms_std0.0 .

The generation of these transforms is detailed in the Quantitative Registration Results section


4- The README.txt file


5- The Licence.txt file (to read and sign by each user)


*QUICK VISUALIZATION (for example with 3D Slicer)*

The meshes and landmarks have been saved in different coordinate systems depending on the algorithm used (Marching Cubes implementation for meshes), and the landmarks annotation protocol.

For a modality and an inividual, to visualize mesh and landmarks in the device coordinates system (where are image and mask), please use the data in the folders: 
	- "TRUSTED_dataset_for_submission/CT_DATA/CT_landmarks_inimg/"
	- "TRUSTED_dataset_for_submission/CT_DATA/CT_meshes_inimg/"
	- "TRUSTED_dataset_for_submission/US_DATA/US_meshes_inimg/"

Note: The US landmarks are already in the US device coordinate system. So just load them

Five patients had one or more renal lesions in one kidney (250R, 283L, 371R, 915L), and one patient had renal lesions in both kidneys (314L, 314R).


