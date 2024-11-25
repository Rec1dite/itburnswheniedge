#include <arrayfire.h>
#include <stdio.h>
#include <af/util.h>
#include <cstdlib>
 
using namespace af;
 
void prewitt(array &mag, array &dir, const array &in) {
    static float h1[] = {1, 1, 1};
    static float h2[] = {-1, 0, 1};
    static array colf(3, 1, h1);
    static array rowf(3, 1, h2);
 
    // Find the gradients
    array Gy = convolve(rowf, colf, in);
    array Gx = convolve(colf, rowf, in);
 
    // Find magnitude and direction
    mag = hypot(Gx, Gy);
    dir = atan2(Gy, Gx);
}
 
void sobelFilter(array &mag, array &dir, const array &in) {
    array Gx, Gy;
    sobel(Gx, Gy, in, 3);
    // Find magnitude and direction
    mag = hypot(Gx, Gy);
    dir = atan2(Gy, Gx);
}
 
array normalize(const array &in) {
    float mx = max<float>(in);
    float mn = min<float>(in);
    return (in - mn) / (mx - mn);
}
 
array edge(const array &in, int method = 0) {
    int w = 5;
    if (in.dims(0) < 512) w = 3;
    if (in.dims(0) > 2048) w = 7;
 
    int h = 5;
    if (in.dims(0) < 512) h = 3;
    if (in.dims(0) > 2048) h = 7;
 
    array ker    = gaussianKernel(w, h);
    array smooth = convolve(in, ker);
    array mag, dir;
 
    switch (method) {
        case 1: prewitt(mag, dir, smooth); break;
        case 2: sobelFilter(mag, dir, smooth); break;
        case 3:
            mag = canny(in, AF_CANNY_THRESHOLD_AUTO_OTSU, 0.18, 0.54).as(f32);
            break;
        default: throw af::exception("Unsupported type");
    }
 
    return normalize(mag);
}
 
void edge() {
    af::Window myWindow("Edge Dectectors");
    af::Window myWindow2(512, 512, "Histogram");
 
    array in = loadImage("../Dataset/Img0.png", false);
 
    array prewitt     = edge(in, 1);
    array sobelFilter = edge(in, 2);
    array hst         = histogram(in, 256, 0, 255);
    array cny         = edge(in, 3);
 
    myWindow2.setAxesTitles("Bins", "Frequency");
 
    while (!myWindow.close() && !myWindow2.close()) {
        /* show input, prewitt and sobel edge detectors in a grid */
        myWindow.grid(2, 2);
 
        myWindow(0, 0).image(in / 255, "Input Image");
        myWindow(0, 1).image(prewitt, "Prewitt");
        myWindow(1, 0).image(sobelFilter, "Sobel");
        myWindow(1, 1).image(cny, "Canny");
 
        myWindow.show();
 
        /* show histogram on input in separate window */
        myWindow2.hist(hst, 0, 255);
    }
}
 
int main1(int argc, char *argv[]) {
    int device = argc > 1 ? atoi(argv[1]) : 0;
 
    try {
        af::setDevice(device);
        af::info();
 
        printf("** ArrayFire Edge Detection Demo **\n");
        edge();
 
    } catch (af::exception &e) {
        fprintf(stderr, "%s\n", e.what());
        throw;
    }
 
    return 0;
}