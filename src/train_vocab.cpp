#include <DBoW2/DBoW2.h>
#include <iostream>
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>

#include "lidar_vo/utils_slam.hpp"

// execute file with ./install/lidar_vo/lib/lidar_vo/train_vocab or run it the same way as any other ros2 node (ros2 run lidar_vo train_vocab)
// reference: https://github.com/manthan99/ORB_SLAM_vocab-build/blob/master/DBoW2/demo/demo.cpp

const std::string dataset_path = "/media/dino/T7/data_odometry_gray/dataset/sequences/00/image_0/";
const std::string db_path = "/home/dino/3dvid/lidar_visual_odometry_ws/src/lidar_vo/vocab_db";

int main(int argc, char** argv)
{
    std:: cout << "Reading images from folder: " << dataset_path << std::endl;
    auto image_names = get_filenames(dataset_path);
    auto imgs = load_imgs(image_names);
    std::cout << image_names[0] << std::endl;   
    
    std:: cout << "Detect ORB features for " << imgs.size() << " images." << std::endl;
    auto descriptors = get_image_descriptors(imgs);
    
    // branching factor and depth levels
    int k = 10, L = 6;
    OrbVocabulary vocab(k, L, DBoW2::TF_IDF, DBoW2::L1_NORM); 
    vocab.create(descriptors);

    std::cout << "Vocabulary info: " << vocab << std::endl;
    std::cout << "Saving vocabulary in " << db_path + "/vocab.yml.gz" << std::endl;
    vocab.save(db_path + "/vocab.yml.gz");
    std::cout << "Vocabulary trained!" << std::endl;

    // create and save database
    std::cout << "Creating the databse." << std::endl;
    OrbDatabase db(vocab, false, 0);
    for (int i = 0; i < imgs.size(); i++)
        db.add(descriptors[i]);
    std::cout << "Database info " << db << std::endl;
    
    // example query
    std::cout << "Example query: " << std::endl;
    DBoW2::QueryResults ret;
    db.query(descriptors[0], ret, 4);
    std::cout << "  Query image: " + image_names[0] << std::endl;
    std::cout << "  Best match: " + image_names[ret[1].Id] << std::endl;

    cv::Mat query_img = cv::imread(image_names[0], cv::IMREAD_COLOR);  
    cv::Mat best_match_img = cv::imread(image_names[ret[1].Id], cv::IMREAD_COLOR);

    // resize the images to fit the screen
    double scale = 0.5;
    cv::resize(query_img, query_img, cv::Size(), scale, scale);
    cv::resize(best_match_img, best_match_img, cv::Size(), scale, scale);

    cv::imshow("Query Image", query_img);
    cv::imshow("Best Match", best_match_img);
    cv::waitKey(0);

    db.save(db_path + "/db.yml.gz");
}