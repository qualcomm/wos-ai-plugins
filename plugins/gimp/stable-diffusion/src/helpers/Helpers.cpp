/*
**************************************************************************************************
* Copyright (c) 2026 Qualcomm Innovation Center, Inc. All rights reserved.
* SPDX-License-Identifier: BSD-3-Clause-Clear
**************************************************************************************************
*/


#include "Helpers.hpp"

std::vector<uchar> Helpers::CvMatToArray( cv::Mat imageMat) {
    std::vector<uchar> array;

    cv::Mat outputColMat = imageMat.reshape(1, imageMat.rows * imageMat.cols * imageMat.channels());
    array.reserve(imageMat.rows * imageMat.cols * imageMat.channels());
    array = outputColMat.clone();

    return array;
}

void Helpers::split
( const std::string &s, char delim, std::vector<std::string> &elems )
{
    std::stringstream ss;
    ss.str(s);
    std::string item;
    while (std::getline(ss, item, delim)) {
        elems.push_back(item);
    }
}

std::vector<std::string> Helpers::split
( const std::string &s, char delim )
{
    //TODO: see if we can change to return split(s, delim, std::vector<std::string>());
    std::vector<std::string> elems;
    Helpers::split(s, delim, elems);
    return elems;
}

std::string Helpers::stripPath
( const std::string& path
)
{
   size_t pos = path.find_last_of("/\\");
   return (pos == std::string::npos) ? "." : path.substr(0, pos);
}

std::string Helpers::joinPath
( const std::string& path1,
  const std::string& path2
)
{
   return path1 + "\\" + path2;
}

bool isPathAbsolute
( const std::string& path
)
{
   return path[0] == '/';
}

std::string Helpers::makePathAbsolute
( const std::string& parentPath,
  const std::string& path
)
{
   return (isPathAbsolute(path)) ? path : joinPath(parentPath, path);
}

bool Helpers::checkKeysPresent
( std::map<std::string,
  std::string>& kvpMap, std::vector<std::string>& requiredKeys
)
{
   // Look for each key listed in requiredKeys in the given map(kvpMap)
   for (size_t i = 0; i < requiredKeys.size(); i++)
   {
      if (kvpMap.find(requiredKeys[i]) == kvpMap.end())
      {
         DEMO_ERROR("Missing required key: %s", requiredKeys[i].c_str());
         return false;
      }
   }
   return true;
}

void Helpers::readKeyValueTxtFile
( std::string &path,
  std::map<std::string, std::string>& kvpMap
)
{
    // expected text file is key=value
    std::ifstream file(path);
    std::string str;
    while (std::getline(file, str))
    {
        // ignore comment lines
        if (str[0] == '#') {
            continue;
        }
        size_t assignPos = str.find_first_of('=');
        // to ignore newline as well as lines without key=value format
        if (assignPos != std::string::npos)
        {
            const auto& key = str.substr(0, assignPos);
            const auto& value = str.substr(assignPos+1);
            if (kvpMap.count(key) == 0)
                kvpMap[key] = value;
            else
                kvpMap[key] = kvpMap[key] + ";" + value;
        }
    }
}

std::map<std::string,float> Helpers::read_threshold_file
( std::string threshold_path
)
{
    std::ifstream infile(threshold_path);
    std::string line;
    std::map<std::string,float> m_ThresholdMap;

    if ( infile.peek() == std::ifstream::traits_type::eof() )
    {
        //DEMO_DEBUG("Threshold File is empty, something went wrong when copying!" );

    }
    /* *** The Threshold file should have this format for this to work:
       // <- a bunch of comment lines with Qualcomm's Copyright informations
       ___ <- ONE(only one) empty line
       Airplane, 2.0 <- line now starts with the threshold file
    */
    // we will need to first throw away the comments
    while (std::getline(infile, line))
        // this is the exit condition: an empty line
        if (line == "") { break; }


    // now we can start to populate the map
    while (std::getline(infile, line)) {
        //just incase we have trailing new lines :)
        if (line != ""){
            // this will return string and int
            std::vector<std::string> vec = Helpers::split(line, ',');
            // populate map with string as key and the int32_t as value
            m_ThresholdMap[vec[0]] = std::atof(vec[1].c_str());
            //TODO: this should have worked std::stof(vec[1]);
        }
    }
    return m_ThresholdMap;
}

std::vector<std::string> Helpers::readTextFile(const std::string& filename)
{
    std::ifstream infile(filename);
    if (infile.peek() == std::ifstream::traits_type::eof())
    {
        DEMO_ERROR("file %s is not found!", filename.c_str());
        return {};
    }

    std::string line;
    std::vector<std::string> v_Text;
    // now we can start to populate the map
    while (std::getline(infile, line)) {
        // If we have any empty lines
        if (line != "") {
            v_Text.push_back(line);
        }
    }
    infile.close();

    return v_Text;
}

bool Helpers::writeTextFile(const std::string& filename, const std::vector<std::string>& v_Text)
{
    if (true != CreateDirsIfNotExist(stripPath(filename))) return false;

    std::ofstream outfile(filename);
    for (const auto& line : v_Text) {
        outfile << line << std::endl;
    }
    outfile.close();

    return true;
}

bool Helpers::fillDims(std::vector<size_t>& dims, uint32_t* inDimensions, uint32_t rank) {
    if (nullptr == inDimensions) {
        DEMO_ERROR("input dimensions is nullptr");
        return false;
    }
    for (size_t r = 0; r < rank; r++) {
        dims.push_back(inDimensions[r]);
    }
    return true;
}

std::string Helpers::getTimeBasedFileName(std::string basepath, std::string extension)
{
    if (true != CreateDirsIfNotExist(basepath)) return "";

    // Getting the time
    time_t rawtime;
    time (&rawtime);

    // Converting it to localtime
    struct tm* timeinfo = localtime(&rawtime);

    // Formatting the time
    char buffer[80];
    strftime(buffer, sizeof(buffer), "%d_%m_%Y_%H_%M_%S", timeinfo);

    // joining the filename to basepath and adding extension
    std::string filename(buffer);
    filename = joinPath(basepath, filename) + "." + extension;

    return filename;
}

bool Helpers::CreateDirsIfNotExist(const std::string& path)
{
    if (path.empty() || path == "." || path == "..")
    {
        return true;
    }
    
    auto i = path.find_last_of('/');
    std::string prefix = path.substr(0, i); 

    // recurse until first dir in path
    if (i != std::string::npos && !CreateDirsIfNotExist(prefix))
    {
        return false;
    }

    int rc = mkdir(path.c_str());
    if (rc == -1 && errno != EEXIST)
    {
        return false;
    }
    else
    {
        struct stat st;
        // make path is created
        if (stat(path.c_str(), &st) == -1)
        {
            return false;
        }
        // makes sure the created path is a directory
        return S_ISDIR(st.st_mode);
    }
}

std::string Helpers::GetFileNameFromPath(std::string filePath)
{
    std::string filename;
    size_t sep = filePath.find_last_of("/");
    // make sure we have a filename and not just given a directory path
    if (sep != std::string::npos && sep + 1 <= filePath.size())
        filePath = filePath.substr(sep + 1, std::string::npos); // grab until the end

    // remove extension but stripping off the string after the last '.' character
    size_t dot = filePath.find_last_of(".");
    if (dot != std::string::npos)
        filename = filePath.substr(0, dot);
    else
        filename = filePath;

    return filename;

}

std::string Helpers::GetFileExtensionFromPath(std::string filePath)
{
    std::string extension;
    // get the value of the extension for filepath by getting all characters after
    // the last '.' character
    size_t dot = filePath.find_last_of(".");
    if (dot != std::string::npos && dot + 1 <= filePath.size())
        extension = filePath.substr(dot + 1, std::string::npos);
    else
        extension = filePath;

    return extension;

}

std::vector<std::string> Helpers::FindAllFilesInDir(std::string dirPath, std::string extension)
{
    DIR *dir = opendir(dirPath.c_str());
    if (dir == NULL) {
        return {};
    }

    std::vector<std::string> filesList;
    struct dirent *entry;
    while ((entry = readdir(dir)) != NULL) {
        if (strcmp(entry->d_name, ".")==0 || strcmp(entry->d_name, "..")==0)
            continue;
        else if (entry->d_type == DT_REG)
            filesList.push_back(std::string(entry->d_name));
    }
    closedir(dir);

    if (!extension.empty())
    {
        std::vector<std::string> filteredFilesList;
        std::copy_if(filesList.begin(), filesList.end(), std::back_inserter(filteredFilesList), 
                     [&extension](const std::string& str) {
                            return str.size() >= extension.size() && 
                               0 == str.compare(str.size()-extension.size(), extension.size(), extension);
                     }
                    );

        return filteredFilesList;
    }

    return filesList;
}

std::vector<std::string> Helpers::FindAllDirsInDir(std::string dirPath)
{
    DIR *dir = opendir(dirPath.c_str());
    if (dir == NULL) {
        return {};
    }

    std::vector<std::string> dirsList;
    struct dirent *entry;
    while ((entry = readdir(dir)) != NULL) {
        if (strcmp(entry->d_name, ".")==0 || strcmp(entry->d_name, "..")==0)
            continue;
        else if (entry->d_type == DT_DIR)
            dirsList.push_back(std::string(entry->d_name));
    }
    closedir(dir);

    return dirsList;
}

unsigned long Helpers::GetFileSize(const std::string& filename) {
    unsigned long ret = 0;
    FILE* fp = fopen(filename.c_str(), "rb");
    if (fp) {
        fseek(fp, 0, SEEK_END);
        ret = ftell(fp);
        fclose(fp);
    }
    return ret;
}

bool Helpers::writeRawData(void* tensorData, uint32_t tensorSize, const std::string& filename) {
    if (true != CreateDirsIfNotExist(stripPath(filename))) return false;

    std::ofstream decodedFile(filename, std::ofstream::binary);
    decodedFile.write((char*)tensorData, tensorSize);
    decodedFile.close();

    return true;
}

bool Helpers::readRawData(void* tensorData, size_t tensorSize, const std::string& filename) {
    if (GetFileSize(filename) != tensorSize) {
        DEMO_DEBUG("Filename = %s, FileSize = %lu, Buffer size = %zu doesn't match!", filename.c_str(), GetFileSize(filename), tensorSize);
        return false;
    }
        
    std::ifstream decodedFile(filename, std::ifstream::binary);
    decodedFile.read((char*)tensorData, tensorSize);
    decodedFile.close();

    return true;
}
