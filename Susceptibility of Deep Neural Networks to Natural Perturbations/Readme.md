The Project Analyzes Sucseptibility of a deep Neural Network Against Natural Perturbations <a name="TOP"></a>
===================

Project Results <a name="TOP"></a>
===================

![1](https://user-images.githubusercontent.com/58717184/108569930-cb0d1000-72da-11eb-9647-5acb028a59e6.JPG)


![1_a](https://user-images.githubusercontent.com/58717184/108569934-ccd6d380-72da-11eb-8bf2-d17aaa6d84ee.JPG)


![2](https://user-images.githubusercontent.com/58717184/108569945-cfd1c400-72da-11eb-8f54-07b6458b6f46.JPG)


![3](https://user-images.githubusercontent.com/58717184/108569950-d2341e00-72da-11eb-9ffd-022ab329d293.JPG)
![4](https://user-images.githubusercontent.com/58717184/108569955-d4967800-72da-11eb-9472-fe242e08c49e.JPG)
![4_a](https://user-images.githubusercontent.com/58717184/108569960-d5c7a500-72da-11eb-91ea-86ca5f2525f8.JPG)
![4_b](https://user-images.githubusercontent.com/58717184/108569962-d7916880-72da-11eb-8adb-97f1d0da0b93.JPG)
![5](https://user-images.githubusercontent.com/58717184/108569967-d95b2c00-72da-11eb-885b-5099b52bae3b.JPG)
![5_a](https://user-images.githubusercontent.com/58717184/108569972-db24ef80-72da-11eb-8abd-2a802a353780.JPG)




---
To run this project you need to have following required packages installed for running this code 

keras
tensorflow (preferably tensorflow-gpu)
matplotlib
scikit-image
scikit-learn
scipy
numpy
torch
torchvision
http://download.tensorflow.org/models/deeplab_cityscapes_xception71_trainfine_2018_09_08.tar.gz


1. Place CityScapes test image into the folder project_code/Input/Images (I am placing one image in it for reference) 

2. run add_snow.py , it will generate and store template at project_code/Input/harmonization

3. (optional depends upon which type of noise you want to add) run add_hail.py , it will generate and store template at project_code/Input/harmonization

4. run add_rain.py, it will generate rain image in the main directory project_code its name will be ending by the text rain

5. run the bash file sin_gan by command bash sin_gan, this will train your sinGAN model against images placed in project_code/Input/Images folder

6. FINALLY GENERATE PERTURBATIONS, 

To harmonize a pasted object into an image (See example in Fig. 13 in our paper), please first train SinGAN model for the desire background image (as described above), then save the naively pasted reference image and it's binary mask under "Input/Harmonization" (see saved images for an example). Run the command

python harmonization.py --input_name <training_image_file_name> --ref_name <naively_pasted_reference_image_file_name> --harmonization_start_scale <scale to inject>g format>

sample commands 

python harmonization.py --input_name berlin_000000_000019_leftImg8bit.png --ref_name berlin_000000_000019_leftImg8bit_snow_naive.png --harmonization_start_scale 6

python harmonization.py --input_name berlin_000000_000019_leftImg8bit.png --ref_name berlin_000000_000019_leftImg8bit_hail_naive.png --harmonization_start_scale 6

Some key points have to be noted for this step, else it might generate errors

--input_name is the name of the sample image in the project_code/Input/Images folder
--ref_image is the name of the template image for perturbation affect, it has same name as the input file but ends in '_snow_naive.png' or '_hail_naive.png'

7. Step 6 will generate adversarial outputs, these outputs will be stored in the folder Output/Harmonization/(name_of_test_image)/(name_of_test_image_naive_out)/start_scale=6.png


8. Copy image outputed from 7th step, this will be your adversarial image, copy corresponding input image from project_code/Input/Images folder to main project_code folder

9. download pretrained model from http://download.tensorflow.org/models/deeplab_cityscapes_xception71_trainfine_2018_09_08.tar.gz , extract it and place .pb file in main project_code folder

10. run the command for testing 

python test_cityscapes_image --IMAGE_FILE <name of input image> --perturb_image <name of perturb image>

example

python test_cityscapes_image --IMAGE_FILE berlin_000000_000019_leftImg8bit.png --perturn_image start_scale=6.png

this can be done for all type of perturbation, make sure to generate the perturbed image before 

Project Insights <a name="TOP"></a>
===================

![Picture1](https://user-images.githubusercontent.com/58717184/108569801-8a14fb80-72da-11eb-96d8-f77d2857bc2d.png)

![Picture2](https://user-images.githubusercontent.com/58717184/108569807-8da88280-72da-11eb-9975-5a5c471eead2.png)
![Picture3](https://user-images.githubusercontent.com/58717184/108569811-8f724600-72da-11eb-8b20-2c2eccb903f1.png)
![Picture4](https://user-images.githubusercontent.com/58717184/108569814-91d4a000-72da-11eb-9f42-01bc8455f1e4.png)

![Picture6](https://user-images.githubusercontent.com/58717184/108569826-96995400-72da-11eb-9ccd-b8bb64bd1cb9.png)
![Picture5](https://user-images.githubusercontent.com/58717184/108569828-97ca8100-72da-11eb-9e4e-953d03f08c19.png)


