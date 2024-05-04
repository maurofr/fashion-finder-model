# inditex-similarity

[![Static Badge](https://img.shields.io/badge/-MongoDB-brightgreen)](https://www.mongodb.com/)

[![MongoDB](https://img.shields.io/badge/MongoDB-4EA94B?style=for-the-badge&logo=mongodb&logoColor=white)](https://www.mongodb.com/)

Inditex-similarity is a project developed at [HackUPC 2024](https://hackupc.com/). It tries to solve the challenge proposed by [InditexTech](https://www.zaratalent.com/es/tech/). <br>
The aim of the challenge is, given a dataset of garment images from various angles, developing an algorithm that identifies duplicated or very similar images not belonging to the same set. <br> Each set consists of three consecutive photos. An example is:
| | | |
| --- | --- | --- |
| ![Image 1](images/img1.jpg) | ![Image 2](images/img2.jpg) | ![Image 3](images/img3.jpg) |
| | | |

## Proposed Solution
Our solution is a combination of deep learning models, that allow the retrieval of the closest images from our dataset, given a trio of images (such as the one above). Moreover, a LLM is included, which allows the users huge creativity, enabling them to find the most similar garment given in their own words. <br>
A tool such as this one can be very powerful, both for the retailer (for fostering sells) and for the customer (for finding the desired piece of clothing with maximum flexibility).

## Organization
This repository is organized in the following way:


