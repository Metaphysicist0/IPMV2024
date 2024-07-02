#include <iostream>
#include <vector>
#include "opencv2/opencv.hpp"
#include "opencv2/stitching.hpp"
#include "opencv2/xfeatures2d.hpp"

using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;

bool try_use_gpu = false;
vector<Mat> imgs;
string result_name = "panoResult.jpg"; // default image name for the panorama output

void printUsage();
int parseCmdArgs(int argc, char** argv);
void loadImagesFromDirectory(const string& directory);
Mat stitchImages(const vector<Mat>& images);
int main(int argc, char* argv[])
{
    int retval = parseCmdArgs(argc, argv); // parse the command line arguments to get the input directory
    if (retval) return -1;
    
    Mat pano; // Mat to store the output panorama image
    Ptr<Stitcher> stitcher = Stitcher::create(Stitcher::PANORAMA); // create a Stitcher object using smart pointer
    Stitcher::Status status = stitcher->stitch(imgs, pano); // stitch the input images together using the pointer
    
    if (status != Stitcher::OK)
    {
        cout << "Can't stitch images, error code = " << int(status) << endl;
        return -1;
    }
    
    imwrite(result_name, pano); // write the result to the output image
    return 0;
}
/*
int main(int argc, char* argv[])
{
    int retval = parseCmdArgs(argc, argv); // parse the command line arguments to get the input directory
    if (retval) return -1;

    if (imgs.size() < 2)
    {
        cout << "Need more images to stitch" << endl;
        return -1;
    }

    Mat pano = stitchImages(imgs);

    if (!pano.empty())
    {
        imwrite(result_name, pano); // write the result to the output image
    }
    else
    {
        cout << "Error stitching images." << endl;
        return -1;
    }

    return 0;
}

Mat stitchImages(const vector<Mat>& images)
{
    // Step 1: Detect features
    vector<detail::ImageFeatures> features(images.size());
    Ptr<SIFT> finder = SIFT::create();
    for (size_t i = 0; i < images.size(); ++i)
    {
        finder->detectAndCompute(images[i], noArray(), features[i].keypoints, features[i].descriptors);
        features[i].img_idx = static_cast<int>(i);
    }

    // Step 2: Match features
    vector<detail::MatchesInfo> pairwise_matches;
    Ptr<detail::BestOf2NearestMatcher> matcher = makePtr<detail::BestOf2NearestMatcher>(false, try_use_gpu);
    (*matcher)(features, pairwise_matches);
    matcher->collectGarbage();

    // Step 3: Estimate camera parameters
    vector<int> indices = detail::leaveBiggestComponent(features, pairwise_matches, 0.3f);
    vector<Mat> images_subset;
    for (size_t i = 0; i < indices.size(); ++i)
    {
        images_subset.push_back(images[indices[i]]);
    }
    vector<detail::CameraParams> cameras;
    Ptr<detail::HomographyBasedEstimator> estimator = makePtr<detail::HomographyBasedEstimator>();
    if (!(*estimator)(features, pairwise_matches, cameras))
    {
        cout << "Homography estimation failed." << endl;
        return Mat();
    }

    for (size_t i = 0; i < cameras.size(); ++i)
    {
        Mat R;
        cameras[i].R.convertTo(R, CV_32F);
        cameras[i].R = R;
    }

    Ptr<detail::BundleAdjusterRay> adjuster = makePtr<detail::BundleAdjusterRay>();
    adjuster->setConfThresh(1);
    Mat_<uchar> refine_mask = Mat::zeros(3, 3, CV_8U);
    refine_mask(0, 0) = 1;
    refine_mask(0, 1) = 1;
    refine_mask(0, 2) = 1;
    refine_mask(1, 1) = 1;
    refine_mask(1, 2) = 1;
    adjuster->setRefinementMask(refine_mask);
    if (!(*adjuster)(features, pairwise_matches, cameras))
    {
        cout << "Camera parameters adjusting failed." << endl;
        return Mat();
    }

    // Find median focal length
    vector<double> focals;
    for (size_t i = 0; i < cameras.size(); ++i)
    {
        focals.push_back(cameras[i].focal);
    }
    sort(focals.begin(), focals.end());
    float warped_image_scale;
    if (focals.size() % 2 == 1)
        warped_image_scale = static_cast<float>(focals[focals.size() / 2]);
    else
        warped_image_scale = static_cast<float>(focals[focals.size() / 2 - 1] + focals[focals.size() / 2]) * 0.5f;

    // Step 4: Warp images
    Ptr<WarperCreator> warper_creator = makePtr<cv::SphericalWarper>();
    Ptr<detail::RotationWarper> warper = warper_creator->create(warped_image_scale);

    vector<Point> corners(images.size());
    vector<UMat> masks_warped(images.size());
    vector<UMat> images_warped(images.size());
    vector<Size> sizes(images.size());
    vector<UMat> masks(images.size());

    for (size_t i = 0; i < images.size(); ++i)
    {
        masks[i].create(images[i].size(), CV_8U);
        masks[i].setTo(Scalar::all(255));
    }

    for (size_t i = 0; i < images.size(); ++i)
    {
        Mat K;
        cameras[i].K().convertTo(K, CV_32F);
        corners[i] = warper->warp(images[i], K, cameras[i].R, INTER_LINEAR, BORDER_REFLECT, images_warped[i]);
        sizes[i] = images_warped[i].size();

        warper->warp(masks[i], K, cameras[i].R, INTER_NEAREST, BORDER_CONSTANT, masks_warped[i]);
    }

    // Step 5: Compensate exposure
    Ptr<detail::ExposureCompensator> compensator = 
        detail::ExposureCompensator::createDefault(detail::ExposureCompensator::GAIN_BLOCKS);
    compensator->feed(corners, images_warped, masks_warped);

    for (size_t i = 0; i < images.size(); ++i)
        compensator->apply(i, corners[i], images_warped[i], masks_warped[i]);

    // Step 6: Find seam masks
    Ptr<detail::SeamFinder> seam_finder = makePtr<detail::GraphCutSeamFinder>(detail::GraphCutSeamFinderBase::COST_COLOR);
    seam_finder->find(images_warped, corners, masks_warped);

    // Step 7: Blend images
    Ptr<detail::Blender> blender = detail::Blender::createDefault(detail::Blender::MULTI_BAND, false);
    Size dst_sz = detail::resultRoi(corners, sizes).size();
    float blend_strength = 5;
    blender->prepare(Rect(0, 0, dst_sz.width, dst_sz.height));
    Mat img_warped_s;
    Mat dilated_mask, seam_mask, mask_warped;
    for (size_t i = 0; i < images.size(); ++i)
    {
        images_warped[i].convertTo(img_warped_s, CV_16S);
        masks_warped[i].convertTo(mask_warped, CV_8U);
        dilate(masks_warped[i], dilated_mask, Mat());
        resize(dilated_mask, seam_mask, mask_warped.size());
        mask_warped = seam_mask & mask_warped;
        blender->feed(img_warped_s, mask_warped, corners[i]);
    }

    Mat result, result_mask;
    blender->blend(result, result_mask);
    result.convertTo(result, CV_8U);

    return result;
}
*/
void loadImagesFromDirectory(const string& directory)
{
    vector<String> filenames;
    glob(directory + "/*.png", filenames, false); // Change "*.jpg" to another pattern if you need different file types

    for (const auto& filename : filenames)
    {
        Mat img = imread(filename);
        if (!img.empty())
        {
            imgs.push_back(img);
        }
        else
        {
            cout << "Failed to load image: " << filename << endl;
        }
    }
}

void printUsage()
{
    cout <<
    "\nPanoramic Image Stitcher\n\n"
    "Usage:\n$ ./main /path/to/images\n\n"
    "Flags:\n"
    "  --try_use_gpu (yes|no)\n"
    "      Try to use GPU. The default value is 'no'. All default values\n"
    "      are for CPU mode.\n"
    "  --output <result_img>\n"
    "      The default is 'panoResult.jpg'.\n\n";
}

int parseCmdArgs(int argc, char** argv)
{
    if (argc == 1)
    {
        printUsage();
        return -1;
    }
    for (int i = 1; i < argc; ++i)
    {
        if (string(argv[i]) == "--help" || string(argv[i]) == "/?")
        {
            printUsage();
            return -1;
        }
        else if (string(argv[i]) == "--try_use_gpu")
        {
            if (i + 1 >= argc)
            {
                cout << "Missing value for --try_use_gpu\n";
                return -1;
            }
            try_use_gpu = string(argv[i + 1]) == "yes";
            i++; // Skip the next argument
        }
        else if (string(argv[i]) == "--output")
        {
            if (i + 1 >= argc)
            {
                cout << "Missing value for --output\n";
                return -1;
            }
            result_name = argv[i + 1];
            i++; // Skip the next argument
        }
        else
        {
            // Assuming the remaining argument is the directory path
            loadImagesFromDirectory(argv[i]);
        }
    }
    return imgs.empty() ? -1 : 0;
}
