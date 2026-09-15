#pragma once

#include <cuda_runtime.h>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>


// Writes 8-bit RGB pixel data to a binary Netpbm (PPM P6) image file
[[nodiscard]] inline bool writePpm(
    const std::string& filePath,
    int width,
    int height,
    const std::vector<uchar3>& pixelBuffer) 
{
    std::ofstream fileStream{filePath, std::ios::binary};
    if (!fileStream.is_open()) {
        std::fprintf(stderr, "Error: Unable to open file '%s' for writing\n", filePath.c_str());
        return false;
    }

    // P6 header: magic identifier, image dimensions, and maximum channel intensity
    fileStream << "P6\n" << width << " " << height << "\n255\n";

    const auto totalBytes{static_cast<std::streamsize>(pixelBuffer.size() * sizeof(uchar3))};
    const char* rawPixelData{reinterpret_cast<const char*>(pixelBuffer.data())};

    fileStream.write(rawPixelData, totalBytes);
    return fileStream.good();
}


// Formats a numbered frame file path: "<outputDirectory>/frame_00000.ppm"
[[nodiscard]] inline std::string framePath(const std::string& outputDirectory, int frameIndex) {
    char filenameBuffer[64]{};
    std::snprintf(filenameBuffer, sizeof(filenameBuffer), "/frame_%05d.ppm", frameIndex);
    return outputDirectory + filenameBuffer;
}
